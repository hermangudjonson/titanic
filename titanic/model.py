"""
Routines for Titanic Model generation and evaluation
"""
from titanic import load_prep

import pandas as pd
import numpy as np
import scipy

from sklearn.base import BaseEstimator, TransformerMixin
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

# estimators
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import xgboost as xgb
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


def get_imputer(strategy="iterative_RF"):
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


def get_classifier(strategy="xgboost", params={}):
    if strategy == "xgboost":
        clf = xgb.XGBClassifier(**params)
    elif strategy == "passthrough":
        clf = "passthrough"
    return clf


def clf_pipeline(
    encoder_strategy="onehot", 
    imputer_strategy="iterative_BR", 
    clf_strategy="xgboost",
    clf_params={}
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


def eval_xgb_cv(
    X, 
    y, 
    params={}, 
    cv=StratifiedKFold(n_splits=5),
    callbacks=None
):
    """Perform cross-validation on xgboost classifier with pipeline preprocessing

    this CV implementation is necessary for early stopping in xgboost
    """

    other_evals = {
        'accuracy': skm.accuracy_score,
        'f1': skm.f1_score,
        'precision': skm.precision_score,
        'recall': skm.recall_score,
        'roc_auc': skm.roc_auc_score,
        'average_precision': skm.average_precision_score,
        'cohen_k': skm.cohen_kappa_score,
        'mcc': skm.matthews_corrcoef
    }

    eval_df = pd.DataFrame()

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_fold, y_fold = X.iloc[train_idx].copy(), y.iloc[train_idx]
        Xt_fold, yt_fold = X.iloc[test_idx].copy(), y.iloc[test_idx]

        pre_pipe = clf_pipeline(clf_strategy='passthrough')

        X_pre = pre_pipe.fit_transform(X_fold, y_fold)
        Xt_pre = pre_pipe.transform(Xt_fold)

        clf = xgb.XGBClassifier(**params, callbacks=callbacks)
        clf.fit(X_pre, y_fold, eval_set=[(X_pre, y_fold), (Xt_pre, yt_fold)], verbose=False)

        best_ntree = clf.best_ntree_limit
        best_idx = clf.best_iteration
        
        # training series
        train_eval = pd.DataFrame(clf.evals_result()['validation_0']).loc[[best_idx],:]
        y_pred = clf.predict(X_pre)
        for s, f in other_evals.items():
            train_eval[s] = f(y_fold, y_pred)
        train_eval['eval_set'] = 'train'
        # test series
        test_eval = pd.DataFrame(clf.evals_result()['validation_1']).loc[[best_idx],:]
        yt_pred = clf.predict(Xt_pre)
        for s, f in other_evals.items():
            test_eval[s] = f(yt_fold, yt_pred)
        test_eval['eval_set'] = 'test'

        fold_eval = pd.concat([train_eval, test_eval], axis=0, ignore_index=True)
        fold_eval['fold'] = fold
        fold_eval['best_ntree'] = best_ntree

        eval_df = pd.concat([eval_df, fold_eval], ignore_index=True)
    
    return eval_df