"""
Ensemble estimators -- voting and stacking
"""
from pathlib import Path

import cloudpickle
import fire
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold, check_cv
from sklearn.utils.validation import check_is_fitted
from titanic import load_prep, model, utils


class StackingClassifier(BaseEstimator, ClassifierMixin):
    """Stacked classifier based on provided estimators and metalearner.

    Parameters
    ----------
    estimators : dict[str, str | Classifier]
        either configured classifier to be fit or file with cv_results to be loaded
    """

    def __init__(self, estimators, final_estimator=None, *, cv=None, passthrough=False):
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.cv = cv
        self.passthrough = passthrough

    def _load_estimators(self):
        """Load pickled pre-fit estimators (otherwise None)"""
        est_dict = {}
        for name, est in self.estimators.items():
            if isinstance(est, str):
                with open(est, "rb") as f:
                    est_dict[name] = cloudpickle.load(f)
            else:
                est_dict[name] = None
        return est_dict

    def _validate_cv(self, est_dict, cv, X, y):
        """Ensure pre-loaded cv splits are consistent with cv parameter."""
        preloaded = []
        for _, cvr in est_dict.items():
            if cvr:
                preloaded.append(list(cvr["indices"].values()))

        if not preloaded:
            cv = check_cv(cv, y, classifier=True)
            return cv

        if not cv:
            # adopt first observed splits and turn into BaseCrossValidator
            # otherwise assume we have already have one provided
            cv = check_cv(preloaded[0], y, classifier=True)
        splits = cv.split(X, y)

        for est_indices in preloaded:
            if not all([all(t1 == t2) and all(v1 == v2) for (t1, v1), (t2, v2) in zip(est_indices, splits)]):
                raise ValueError("CV indices from preloaded estimators do not match.")

        # we may also consider validating that test splits form a full partition
        return cv

    def _out_of_fold(self, cv_results):
        """Recombine out of fold predictions from `cv_results` for a single estimator

        `cv_results` is assumed to be generated from `cv_with_validation` with common callbacks.
        Return predict_proba predictions in original data order with first column removed
        Returns a 2D numpy array
        """
        test_indices = np.concatenate(
            [test for _, test in cv_results["indices"].values()]
        )
        inv_test_indices = np.empty(len(test_indices), dtype=int)
        inv_test_indices[test_indices] = np.arange(len(test_indices))

        predictions = np.concatenate(list(cv_results["predict_proba_test"].values()))
        return predictions[inv_test_indices, 1:]  # shape (N_X, N_classes - 1)

    def _concatenate_predictions(self, X, predictions):
        """Collate training data X_meta from first-stage predictions for metalearner.

        pass raw data if passthrough option is True
        """
        if self.passthrough:
            predictions.append(X)

        X_meta = np.hstack(predictions)
        return X_meta

    def fit(self, X, y):
        """Fit first-stage estimators and metalearner.

        Generates the following attributes:
        self.estimators_ : dictionary of cv_results
        self.final_estimator_ : fitted metalearner
        """
        # pre-load estimators if available
        self.estimators_ = self._load_estimators()
        if self.final_estimator is None:
            self.final_estimator_ = LogisticRegressionCV()
        else:
            self.final_estimator_ = clone(self.final_estimator)

        # validate cv
        cv = self._validate_cv(self.estimators_, self.cv, X, y)

        # fit first stage
        for name, cvr in self.estimators_.items():
            if cvr is None:
                self.estimators_[name] = model.cv_with_validation(
                    clone(self.estimators[name]),
                    X,
                    y,
                    cv,
                    callbacks=model.common_cv_callbacks(),
                )

        # aggregate out of fold predictions for metalearner
        X_meta = self._concatenate_predictions(
            X, [self._out_of_fold(cvr) for cvr in self.estimators_.values()]
        )

        # fit second stage
        self.final_estimator_.fit(X_meta, y)
        return self

    def _transform(self, X):
        """Transform new data with first stage estimators.

        For each estimator, average predict_proba from each fold estimator (stacking variant A).
        """
        check_is_fitted(self)

        # average first stage
        predictions = []
        for cvr in self.estimators_.values():
            fold_predictions = np.stack(
                [fold_est.predict_proba(X) for fold_est in cvr["estimator"].values()],
                axis=0,
            )
            # remove first column of predict_proba results
            predictions.append(fold_predictions.mean(axis=0)[:, 1:])
        return self._concatenate_predictions(X, predictions)

    def predict(self, X):
        """predict"""
        check_is_fitted(self)

        X_meta = self._transform(X)
        return self.final_estimator_.predict(X_meta)

    def predict_proba(self, X):
        """predict_proba"""
        check_is_fitted(self)

        X_meta = self._transform(X)
        return self.final_estimator_.predict_proba(X_meta)

    def decision_function(self, X):
        """decision_function"""
        check_is_fitted(self)

        X_meta = self._transform(X)
        return self.final_estimator_.decision_function(X_meta)


def get_reduced_stacker():
    """Configure reduced stacker model with pretrained estimators

    trying just gbms and random forest
    """
    PRE_DIR = Path('/kaggle/input/pretrained') if utils.ON_KAGGLE else utils.WORKING_DIR / "data/hpo"

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)
    # define models
    lightgbm_pipe = str(PRE_DIR / 'lgbm_best_cv.pkl')
    xgboost_pipe = str(PRE_DIR / 'xgb_best_cv.pkl')
    catboost_pipe = str(PRE_DIR / 'ctb_best_cv.pkl')
    randomforest_pipe = str(PRE_DIR / 'rf_best_cv.pkl')
    extrarandomforest_pipe = str(PRE_DIR / 'erf_best_cv.pkl')

    stacker = StackingClassifier(
        {
            'lightgbm': lightgbm_pipe,
            'xgboost': xgboost_pipe,
            'catboost': catboost_pipe,
            'randomforest': randomforest_pipe,
            'extrarandomforest': extrarandomforest_pipe,
        }, 
        final_estimator=LogisticRegressionCV(),
        cv=cv
    )
    return stacker


def get_full_stacker():
    """Configure full stacker model with pretrained estimators

    should only need to fit svm, logistic, neighbors and naivebayes
    """
    PRE_DIR = Path('/kaggle/input/pretrained') if utils.ON_KAGGLE else utils.WORKING_DIR / "data/hpo"

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)
    # define models
    lightgbm_pipe = str(PRE_DIR / 'lgbm_best_cv.pkl')
    xgboost_pipe = str(PRE_DIR / 'xgb_best_cv.pkl')
    catboost_pipe = str(PRE_DIR / 'ctb_best_cv.pkl')
    randomforest_pipe = str(PRE_DIR / 'rf_best_cv.pkl')
    extrarandomforest_pipe = str(PRE_DIR / 'erf_best_cv.pkl')
    mlp_pipe = str(PRE_DIR / 'mlp_best_cv.pkl')
    svm_pipe = model.clf_pipeline(
        clf_strategy='svm', 
        clf_params={
            "kernel": "rbf",
            "probability": True,
            "class_weight": "balanced"
        }
    )
    logistic_pipe = model.clf_pipeline(clf_strategy='logistic', clf_params={'max_iter':1000})
    neighbors_pipe = model.clf_pipeline(clf_strategy='neighbors', clf_params={'n_neighbors':10})
    naivebayes_pipe = model.clf_pipeline(clf_strategy='naivebayes')

    stacker = StackingClassifier(
        {
            'lightgbm': lightgbm_pipe,
            'xgboost': xgboost_pipe,
            'catboost': catboost_pipe,
            'randomforest': randomforest_pipe,
            'extrarandomforest': extrarandomforest_pipe,
            'mlp': mlp_pipe,
            'svm': svm_pipe,
            'logistic': logistic_pipe, 
            'neighbors': neighbors_pipe,
            'naivebayes': naivebayes_pipe,
        }, 
        final_estimator=LogisticRegressionCV(),
        cv=cv
    )
    return stacker


def get_stacker(strategy="reduced"):
    if strategy == "full":
        stacker = get_full_stacker()
    elif strategy == "reduced":
        stacker = get_reduced_stacker()
    return stacker


def _cvr_eval_df(cvr):
    return pd.concat(cvr['eval_test'].values(), ignore_index=True)


def submit(strategy="reduced"):
    """Train full stacker model and generate submission

    also record eval results for all component models on validation folds
    """
    # load test data
    X, y = load_prep.raw_train()
    X_test = load_prep.raw_test()

    stacker_clf = get_stacker(strategy=strategy)
    stacker_clf.fit(X, y)
    
    # record validation evals
    evals = (
        {
            k: _cvr_eval_df(stacker_clf.estimators_[k]).mean()
            for k in stacker_clf.estimators_.keys()
        }
    )
    pd.concat(evals, axis=1).to_csv("stacker_estimators_evals.csv")

    stacker_predict = pd.Series(stacker_clf.predict(X_test), name='Survived', index=X_test.index)
    stacker_predict.to_csv("stacker_predict.csv")


if __name__ == "__main__":
    fire.Fire()