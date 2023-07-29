"""
Routines for Titanic Model generation and evaluation
"""
from titanic import load_prep

from time import time

import pandas as pd
import numpy as np
import scipy

from sklearn.base import BaseEstimator, TransformerMixin, is_classifier, clone
from sklearn.preprocessing import (
    FunctionTransformer,
    StandardScaler,
    OneHotEncoder,
    OrdinalEncoder,
)
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.model_selection import check_cv
from sklearn.utils.metaestimators import _safe_split

# estimators
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import xgboost as xgb
import lightgbm as lgbm
from sklearn.model_selection import StratifiedKFold
import sklearn.metrics as skm


def get_encoder(strategy="onehot"):
    """either onehot or ordinal encoding"""
    if strategy == "onehot":
        cat_encoder = OneHotEncoder(
            sparse_output=False,
            drop="if_binary",
            handle_unknown="infrequent_if_exist",
            min_frequency=5,
        )
    elif strategy == "ordinal":
        cat_encoder = OrdinalEncoder()
    else:
        raise ValueError("strategy must be one of onehot, ordinal")

    col_encoder = make_column_transformer(
        (cat_encoder, make_column_selector(dtype_include="category")),
        remainder="passthrough",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")
    return col_encoder


def get_imputer(strategy="iterative_BR"):
    """supports three imputer strategies of increasing complexity"""
    if strategy == "simple_median":
        imputer = SimpleImputer(strategy="median", add_indicator=True)
    elif strategy == "iterative_BR":
        imputer = IterativeImputer(
            BayesianRidge(),
            max_iter=10,
            initial_strategy="median",
            skip_complete=True,
            add_indicator=True,
        )
    elif strategy == "iterative_RF":
        imputer = IterativeImputer(
            RandomForestRegressor(),
            max_iter=10,
            initial_strategy="median",
            skip_complete=True,
            add_indicator=True,
        )
    else:
        raise ValueError(
            "strategy must be one of simple_median, iterative_BR, iterative_RF"
        )

    imputer.set_output(transform="pandas")
    return imputer


class LGBMProxy(lgbm.LGBMClassifier):
    """LightGBM wrapper that conforms to sklearn interface

    specifically move callbacks and validation data to initialization
    rather than needing to be passed during the call to fit itself

    Parameters
    ----------
    callbacks : list, optional
        lightgbm fit callback functions
    validation : tuple (X, y), optional
        validation data, required to use early stopping
    **params : optional
        parameters to be passed to LGBMClassifier initialization
    """

    def __init__(self, callbacks=None, validation=None, **params):
        super().__init__(**params)
        self.callbacks = callbacks
        self.validation = validation

    @classmethod
    def _get_param_names(cls):
        return sorted(
            set(["callbacks", "validation"] + lgbm.LGBMClassifier._get_param_names())
        )

    def fit(self, X, y):
        if self.validation is not None:
            eval_set = [self.validation, (X, y)]
            eval_names = ["validation", "training"]
        else:
            eval_set = None
            eval_names = None
        super().fit(
            X,
            y,
            eval_set=eval_set,
            eval_names=eval_names,
            eval_metric=None,  # defaulting to training objective
            callbacks=self.callbacks,
        )
        return self


def get_classifier(strategy="xgboost", params=None):
    """return classifier to be used in classification pipeline"""
    params = params if params is not None else {}

    if strategy == "xgboost":
        clf = xgb.XGBClassifier(**params)
    elif strategy == "lightgbm":
        clf = LGBMProxy(**params)
    elif strategy == "passthrough":
        clf = "passthrough"
    return clf


def clf_pipeline(
    encoder_strategy="onehot",
    imputer_strategy="iterative_BR",
    clf_strategy="xgboost",
    clf_params=None,
):
    """generate complete classification pipeline

    Include named steps:
     - preprocesser
     - encoder
     - imputer
     - classifier

    take configuration parameters as input to produce pipeline variations
    for cross-validation and investigation
    """
    clf_params = clf_params if clf_params is not None else {}

    pipe = Pipeline(
        [
            ("preprocessor", load_prep.preprocess_pipeline()),
            ("encoder", get_encoder(strategy=encoder_strategy)),
            ("imputer", get_imputer(strategy=imputer_strategy)),
            ("clf", get_classifier(strategy=clf_strategy, params=clf_params)),
        ]
    )

    return pipe


def get_imputed_df(X, y):
    """utility function to retrieve imputed data with categoricals intact (as after preprocessing)"""
    pipe = make_pipeline(load_prep.preprocess_pipeline(), get_encoder(), get_imputer())
    X_imputed = pipe.fit_transform(X, y)
    X_pp = pipe[0].transform(X)
    X_pp[["Fare", "Age", "Fare_Missing", "Age_Missing"]] = X_imputed[
        ["Fare", "Age", "missingindicator_Fare", "missingindicator_Age"]
    ]
    X_pp["Fare_Missing"] = pd.Categorical(X_pp.Fare_Missing)
    X_pp["Age_Missing"] = pd.Categorical(X_pp.Age_Missing)
    return X_pp


def fit_with_validation(
    clf_pipe: Pipeline,
    X_train,
    y_train,
    X_valid=None,
    y_valid=None,
):
    """fit routine for clf pipeline

    fits first stages first in order to pass transformed
    validation data to final classifier if necessary
    """
    if X_valid is None:
        # simple call to fit
        return clf_pipe.fit(X_train, y_train)

    X_train_pre = clf_pipe[:-1].fit_transform(X_train, y_train)
    X_valid_pre = clf_pipe[:-1].transform(X_valid)

    # last step has to accept validation as a param
    clf_pipe[-1].set_params(validation=(X_valid_pre, y_valid))
    clf_pipe[-1].fit(X_train_pre, y_train)

    return clf_pipe


def cv_with_validation(estimator, X, y, cv, callbacks=None):
    """Perform cross-validation while passing validation data to the estimator in each fold

    estimator is fit using `fit_with_validation`, and is assumed to be a pipeline that can
    accept a validation data parameter

    return results dictionary with one key per callback
    """
    cv = check_cv(cv, y, classifier=is_classifier(estimator))
    n_splits = cv.get_n_splits(X, y)
    callbacks = callbacks if callbacks is not None else {}

    result = {k: {} for k in callbacks.keys()}
    result["train_time"] = {}

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        fold_estimator = clone(estimator)

        X_train, y_train = _safe_split(fold_estimator, X, y, train_idx)
        X_valid, y_valid = _safe_split(fold_estimator, X, y, test_idx, train_indices=train_idx)

        start_time = time()

        fit_with_validation(fold_estimator, X_train, y_train, X_valid, y_valid)

        fit_time = time() - start_time
        result["train_time"][fold] = fit_time

        for k, func in callbacks.items():
            result[k][fold] = func(
                fold=fold, 
                estimator=fold_estimator, 
                indices=(train_idx, test_idx), 
                train_data=(X_train, y_train), 
                test_data=(X_valid, y_valid)
            )
    return result


def _score_classifier(clf, X, y, eval_name=None):
    """Score fitted classifier for common metrics
    """
    other_evals = {
        "accuracy": skm.accuracy_score,
        "f1": skm.f1_score,
        "precision": skm.precision_score,
        "recall": skm.recall_score,
        "roc_auc": skm.roc_auc_score,
        "average_precision": skm.average_precision_score,
        "cohen_k": skm.cohen_kappa_score,
        "mcc": skm.matthews_corrcoef,
    }

    y_pred = clf.predict(X)
    eval_vals = {s: f(y_pred, y) for s, f in other_evals.items()}
    if eval_name is not None:
        eval_vals['eval_set'] = eval_name
    
    return pd.DataFrame(pd.Series(eval_vals)).T


def _score_train(*, estimator, train_data, **kwargs):
    X, y = train_data
    return _score_classifier(estimator, X, y, eval_name='train')


def _score_test(*, estimator, test_data, **kwargs):
    X, y = test_data
    return _score_classifier(estimator, X, y, eval_name='test')


def _predict_proba_test(*, estimator, test_data, **kwargs):
    X, y = test_data
    return estimator.predict_proba(X)


def _lgbm_fit_metrics(*, estimator, **kwargs):
    """Return fit metrics for fitted LightGBM model
    """
    clf = estimator['clf']
    best_ntree = clf.best_iteration_ if clf.best_iteration_ else clf.n_estimators
    best_idx = best_ntree - 1

    lgbm_evals = {
        **{'train_' + k: v[best_idx] for k, v in clf.evals_result_['training'].items()},
        **{'test_' + k: v[best_idx] for k, v in clf.evals_result_['validation'].items()},
        **{'best_ntree': best_ntree}
    }
    return pd.DataFrame(pd.Series(lgbm_evals)).T

def common_cv_callbacks():
    """Generates dictionary of common CV callback functions for cv_with_validation.

    includes:
        - fitted estimator
        - fold indices
        - evaluation metrics on training data
        - evaluation metrics on test data
        - prediction probabilities
        - test target data
    """
    callbacks = {
        'estimator': lambda *, estimator, **kw: estimator, 
        'indices': lambda *, indices, **kw: indices,
        'eval_train': _score_train,
        'eval_test': _score_test,
        'predict_proba_test': _predict_proba_test, 
        'y_test': lambda *, test_data, **kw: test_data[1]
    }
    return callbacks


def eval_xgb_cv(X, y, params=None, cv=None, callbacks=None):
    """Perform cross-validation on xgboost classifier with pipeline preprocessing

    this CV implementation is necessary for early stopping in xgboost

    Parameters
    ----------
        cv: defaults to Stratified 5 fold if not provided
    """
    params = params if params is not None else {}
    cv = cv if cv is not None else StratifiedKFold(n_splits=5)

    other_evals = {
        "accuracy": skm.accuracy_score,
        "f1": skm.f1_score,
        "precision": skm.precision_score,
        "recall": skm.recall_score,
        "roc_auc": skm.roc_auc_score,
        "average_precision": skm.average_precision_score,
        "cohen_k": skm.cohen_kappa_score,
        "mcc": skm.matthews_corrcoef,
    }

    eval_df = pd.DataFrame()

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_fold, y_fold = X.iloc[train_idx].copy(), y.iloc[train_idx]
        Xt_fold, yt_fold = X.iloc[test_idx].copy(), y.iloc[test_idx]

        pre_pipe = clf_pipeline(clf_strategy="passthrough")

        X_pre = pre_pipe.fit_transform(X_fold, y_fold)
        Xt_pre = pre_pipe.transform(Xt_fold)

        clf = xgb.XGBClassifier(**params, callbacks=callbacks)
        clf.fit(
            X_pre, y_fold, eval_set=[(X_pre, y_fold), (Xt_pre, yt_fold)], verbose=False
        )

        best_ntree = clf.best_ntree_limit  # going to be deprecated
        best_idx = clf.best_iteration

        # training series
        train_eval = pd.DataFrame(clf.evals_result()["validation_0"]).loc[[best_idx], :]
        y_pred = clf.predict(X_pre)
        for s, f in other_evals.items():
            train_eval[s] = f(y_fold, y_pred)
        train_eval["eval_set"] = "train"
        # test series
        test_eval = pd.DataFrame(clf.evals_result()["validation_1"]).loc[[best_idx], :]
        yt_pred = clf.predict(Xt_pre)
        for s, f in other_evals.items():
            test_eval[s] = f(yt_fold, yt_pred)
        test_eval["eval_set"] = "test"

        fold_eval = pd.concat([train_eval, test_eval], axis=0, ignore_index=True)
        fold_eval["fold"] = fold
        fold_eval["best_ntree"] = best_ntree

        eval_df = pd.concat([eval_df, fold_eval], ignore_index=True)

    return eval_df


def eval_lgbm_cv(X, y, params=None, cv=None, callbacks=None):
    """Perform cross-validation on lightgbm classifier with pipeline preprocessing

    this CV implementation is necessary for early stopping in lightgbm

    Parameters
    ----------
        cv: defaults to Stratified 5 fold if not provided
    """
    params = params if params is not None else {}
    cv = cv if cv is not None else StratifiedKFold(n_splits=5)

    other_evals = {
        "accuracy": skm.accuracy_score,
        "f1": skm.f1_score,
        "precision": skm.precision_score,
        "recall": skm.recall_score,
        "roc_auc": skm.roc_auc_score,
        "average_precision": skm.average_precision_score,
        "cohen_k": skm.cohen_kappa_score,
        "mcc": skm.matthews_corrcoef,
    }

    eval_df = pd.DataFrame()

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_fold, y_fold = X.iloc[train_idx].copy(), y.iloc[train_idx]
        Xt_fold, yt_fold = X.iloc[test_idx].copy(), y.iloc[test_idx]

        pre_pipe = clf_pipeline(clf_strategy="passthrough")

        X_pre = pre_pipe.fit_transform(X_fold, y_fold)
        Xt_pre = pre_pipe.transform(Xt_fold)

        clf = lgbm.LGBMClassifier(**params)
        clf.fit(
            X_pre,
            y_fold,
            eval_set=[(Xt_pre, yt_fold), (X_pre, y_fold)],
            eval_names=["validation", "training"],
            eval_metric=None,  # defaulting to training objective
            callbacks=callbacks,
        )

        best_ntree = clf.best_iteration_ if clf.best_iteration_ else clf.n_estimators
        best_idx = best_ntree - 1

        # training series
        train_eval = pd.DataFrame(clf.evals_result_["training"]).loc[[best_idx], :]
        y_pred = clf.predict(X_pre)
        for s, f in other_evals.items():
            train_eval[s] = f(y_fold, y_pred)
        train_eval["eval_set"] = "train"
        # test series
        test_eval = pd.DataFrame(clf.evals_result_["validation"]).loc[[best_idx], :]
        yt_pred = clf.predict(Xt_pre)
        for s, f in other_evals.items():
            test_eval[s] = f(yt_fold, yt_pred)
        test_eval["eval_set"] = "test"

        fold_eval = pd.concat([train_eval, test_eval], axis=0, ignore_index=True)
        fold_eval["fold"] = fold
        fold_eval["best_ntree"] = best_ntree

        eval_df = pd.concat([eval_df, fold_eval], ignore_index=True)

    return eval_df
