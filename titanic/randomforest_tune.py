"""
Random Forest hyperparameter optimization with optuna

optuna experiments defined by objectives (search space), study creation and runs

tuning progression:
 - broad parameter sweep
"""

from titanic import load_prep, model, utils

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
import catboost as ctb
import optuna
import cloudpickle

import functools
import warnings
import fire


def _cv_results_df(cv_results: dict):
    """collate eval test results from cv_with_validation return
    """
    folds = cv_results['eval_test'].keys()
    cv_results_df = pd.concat(
        [
            pd.concat(cv_results['lgbm_metrics'].values(), keys=folds, ignore_index=True),
            pd.concat(cv_results['eval_test'].values(), keys=folds, ignore_index=True)
        ], 
        axis=1
    ).infer_objects()
    return cv_results_df


def stage0_objective(trial, X, y, extra_trees=False):
    """objective for stage0 broad parameter sweep
    """
    lgbm_params = {
        "objective": 'binary', 
        "verbosity": -1, 
        "n_estimators": 2000, 
        "num_leaves": trial.suggest_int("num_leaves", 2, 4095),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 2, 100),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10, log=True),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.05, 1.0), 
        "feature_fraction_bynode": trial.suggest_float("feature_fraction", 0.05, 1.0),
    }

    sfold = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=1234)

    strategy = 'extrarandomforest' if extra_trees else 'randomforest'
    clf_pipe = model.clf_pipeline(clf_strategy=strategy, clf_params=lgbm_params)
    cv_results = model.cv_with_validation(
        clf_pipe, 
        X, 
        y, 
        sfold, 
        callbacks = model.common_cv_callbacks() | {'lgbm_metrics': model.lgbm_fit_metrics}
    )
    cv_results_df = _cv_results_df(cv_results)

    eval_test = cv_results_df.mean(numeric_only=True)
    for k, v in eval_test.items():
        trial.set_user_attr(k, v)
    return eval_test['test_binary_logloss']


def stage0(n_trials=100, timeout=3600, outdir=".", extra_trees=False):
    """run optuna random forest stage0 hyperparamter optimization

    option to optimize random forest or extra random forest
    """
    study_str = "erf_stage0" if extra_trees else "rf_stage0"
    sql_file = f'sqlite:///{str(utils.WORKING_DIR / outdir / f"{study_str}.db")}'

    study = optuna.create_study(
        storage=sql_file,
        load_if_exists=True,
        study_name=study_str,
        pruner=optuna.pruners.NopPruner(), 
        direction="minimize", 
        sampler=optuna.samplers.TPESampler()
    )
    
    warnings.simplefilter('ignore') # to suppress multiple callback warning
    # pre-load data for trials
    raw_train_df, target_ds = load_prep.raw_train()
    study.optimize(
        functools.partial(stage0_objective, X=raw_train_df, y=target_ds, extra_trees=extra_trees),
        n_trials=n_trials,
        timeout=timeout
    )
    warnings.resetwarnings()
    return study


if __name__ == "__main__":
    fire.Fire()