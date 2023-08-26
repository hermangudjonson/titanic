"""
Catboost hyperparameter optimization with optuna

optuna experiments defined by objectives (search space), study creation and runs

tuning progression:
 - no tuning (just CV eval)
 - n_estimators sweep (no early stopping), for timing info
 - learning_rate sweep to evaluate early stopping points
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
            pd.concat(cv_results['ctb_metrics'].values(), keys=folds, ignore_index=True),
            pd.concat(cv_results['eval_test'].values(), keys=folds, ignore_index=True)
        ], 
        axis=1
    ).infer_objects()
    return cv_results_df


def n_estimators_objective(trial, X, y):
    """objective for n_estimators grid
    """
    ctb_params = {
        "objective": "Logloss", 
        "n_estimators": trial.suggest_int("n_estimators", 1, 1000),
        "learning_rate": 5e-3, 
        "allow_writing_files": False,
    }

    sfold = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=1234)

    clf_pipe = model.clf_pipeline(clf_strategy='catboost', clf_params=ctb_params)
    cv_results = model.cv_with_validation(
        clf_pipe, 
        X, 
        y, 
        sfold, 
        callbacks = model.common_cv_callbacks() | {'ctb_metrics': model.ctb_fit_metrics}
    )
    cv_results_df = _cv_results_df(cv_results)

    eval_test = cv_results_df.mean(numeric_only=True)
    for k, v in eval_test.items():
        trial.set_user_attr(k, v)
    return eval_test['test_Logloss']


def n_estimators_grid(n_trials=20, outdir="."):
    """run optuna lightgbm n_estimators grid
    """
    sql_file = f'sqlite:///{str(utils.WORKING_DIR / outdir / "ctb_n_estimators_grid.db")}'

    study = optuna.create_study(
        storage=sql_file,
        load_if_exists=False,
        study_name='ctb_n_estimators',
        pruner=optuna.pruners.NopPruner(), 
        direction="minimize", 
        sampler=optuna.samplers.GridSampler(
            search_space={"n_estimators": np.geomspace(1, 10, num=n_trials, dtype=int).tolist()}
        )
    )
    
    warnings.simplefilter('ignore') # to suppress multiple callback warning
    # pre-load data for trials
    raw_train_df, target_ds = load_prep.raw_train()
    study.optimize(
        functools.partial(n_estimators_objective, X=raw_train_df, y=target_ds)
    )
    warnings.resetwarnings()
    return study


def early_stopping_objective(trial, X, y):
    """objective for n_estimators grid
    """
    ctb_params = {
        "objective": "Logloss", 
        "n_estimators": 10000,
        "allow_writing_files": False, 
        "learning_rate": trial.suggest_float('learning_rate', 1e-5, 1.0, log=True), 
        "early_stopping_rounds": 20
    }

    sfold = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=1234)

    clf_pipe = model.clf_pipeline(clf_strategy='catboost', clf_params=ctb_params)
    cv_results = model.cv_with_validation(
        clf_pipe, 
        X, 
        y, 
        sfold, 
        callbacks = model.common_cv_callbacks() | {'ctb_metrics': model.ctb_fit_metrics}
    )
    cv_results_df = _cv_results_df(cv_results)

    eval_test = cv_results_df.mean(numeric_only=True)
    for k, v in eval_test.items():
        trial.set_user_attr(k, v)
    return eval_test['test_Logloss']


def early_stopping_grid(n_trials=21, outdir="."):
    """run optuna lightgbm early stopping grid
    """
    sql_file = f'sqlite:///{str(utils.WORKING_DIR / outdir / "ctb_early_stopping_grid.db")}'

    study = optuna.create_study(
        storage=sql_file,
        load_if_exists=False,
        study_name='ctb_early_stopping', 
        pruner=optuna.pruners.NopPruner(), 
        direction="minimize", 
        sampler=optuna.samplers.GridSampler(
            search_space={"learning_rate": np.geomspace(1e-5, 1.0, num=n_trials).tolist()}
        )
    )
    
    warnings.simplefilter('ignore') # to suppress multiple callback warning
    # pre-load data for trials
    raw_train_df, target_ds = load_prep.raw_train()
    study.optimize(
        functools.partial(early_stopping_objective, X=raw_train_df, y=target_ds)
    )
    warnings.resetwarnings()
    return study


if __name__ == "__main__":
    fire.Fire()