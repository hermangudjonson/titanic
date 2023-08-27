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
    """objective for early_stopping grid
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
    """run optuna catboost early stopping grid
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


def stage0_objective(trial, X, y):
    """objective for stage0 broad parameter sweep
    """
    ctb_params = {
        "objective": "Logloss", 
        "n_estimators": 2000, 
        "allow_writing_files": False, 
        "learning_rate": 5e-2, 
        "early_stopping_rounds": 20, 
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-8, 10, log=True),
        "depth": trial.suggest_int("depth", 2, 12), 
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 10), 
        "random_strength": trial.suggest_float("random_strength", 1e-1, 10, log=True), 
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.4, 1.0), 
        "callbacks": [
            optuna.integration.CatBoostPruningCallback(trial, 'Logloss')
        ]
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


def stage0(prune=False, n_trials=100, timeout=3600, outdir="."):
    """run optuna catboost stage0 hyperparamter optimization
    """
    if prune:
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=5, n_warmup_steps=200, interval_steps=50, n_min_trials=5)
    else:
        pruner = optuna.pruners.NopPruner()
    
    sql_file = f'sqlite:///{str(utils.WORKING_DIR / outdir / "ctb_stage0.db")}'

    study = optuna.create_study(
        storage=sql_file,
        load_if_exists=True,
        study_name='ctb_stage0',
        pruner=pruner, 
        direction="minimize", 
        sampler=optuna.samplers.TPESampler()
    )
    
    warnings.simplefilter('ignore') # to suppress multiple callback warning
    # pre-load data for trials
    raw_train_df, target_ds = load_prep.raw_train()
    study.optimize(
        functools.partial(stage0_objective, X=raw_train_df, y=target_ds),
        n_trials=n_trials,
        timeout=timeout
    )
    warnings.resetwarnings()
    return study


def cv_best_trial(learning_rate=5e-1, outdir=None):
    """Fit across CV folds with best Catboost hyperparameters.
    """
    ctb_params = {

    }
    # best trial params
    trial_params = {

    }
    ctb_params = ctb_params | trial_params

    raw_train_df, target_ds = load_prep.raw_train()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)

    clf_pipe = model.clf_pipeline(clf_strategy='catboost', clf_params=ctb_params)
    cv_results = model.cv_with_validation(
        clf_pipe, 
        raw_train_df, 
        target_ds, 
        cv, 
        callbacks = model.common_cv_callbacks() | {'ctb_metrics': model.ctb_fit_metrics}
    )
    cv_results_df = _cv_results_df(cv_results)

    if outdir is not None:
        # pickle cv results
        with open(utils.WORKING_DIR / outdir / "ctb_best_cv.pkl", 'wb') as f:
            cloudpickle.dump(cv_results, f)
        cv_results_df.to_csv(utils.WORKING_DIR / outdir / "ctb_best_eval_test.csv")
        
    return cv_results


if __name__ == "__main__":
    fire.Fire()