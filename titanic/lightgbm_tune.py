"""
LightGBM hyperparameter optimization with optuna

optuna experiments defined by objectives (search space), study creation and runs

tuning progression:
 - no tuning (just CV eval)
 - n_estimators sweep (no early stopping), for timing info
"""

from titanic import load_prep, model, utils

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
import lightgbm as lgbm
import optuna

import functools
import warnings
import fire


def n_estimators_objective(trial, X, y):
    """objective for n_estimators grid
    """
    lgbm_params = {
        "objective": 'binary',
        "n_estimators": trial.suggest_int("n_estimators", 1, 1000),
        "learning_rate": 5e-3
    }

    sfold = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=1234)
    # this is approximately 1e-3 decay in 10k steps
    learning_rates = list(lgbm_params['learning_rate'] * np.float_power(1 - 7e-4, np.arange(lgbm_params['n_estimators'])))
    
    eval_df = model.eval_lgbm_cv(
        X,
        y,
        params=lgbm_params, 
        cv=sfold,
        callbacks=[
            optuna.integration.LightGBMPruningCallback(trial, 'binary_logloss', valid_name='validation'),
            lgbm.reset_parameter(learning_rate=learning_rates)
        ]
    )
    
    eval_test = eval_df.loc[eval_df.eval_set == 'test',:].mean(numeric_only=True)
    for k, v in eval_test.items():
        trial.set_user_attr(k, v)
    return eval_test['binary_logloss']

def n_estimators_grid(n_trials=20, outdir="."):
    """run optuna lightgbm n_estimators grid
    """
    sql_file = f'sqlite:///{str(utils.WORKING_DIR / outdir / "lgbm_n_estimators_grid.db")}'

    study = optuna.create_study(
        storage=sql_file,
        load_if_exists=False,
        study_name='lgbm_n_estimators',
        pruner=optuna.pruners.NopPruner(), 
        direction="minimize", 
        sampler=optuna.samplers.GridSampler(
            search_space={"n_estimators":np.geomspace(1, 1000, num=n_trials, dtype=int).tolist()}
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
    """objective for early stopping grid
    """
    lgbm_params = {
        "objective": 'binary',
        "n_estimators": 10000,
        "learning_rate": trial.suggest_float('learning_rate', 1e-5, 1.0, log=True)
    }

    sfold = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=1234)
    # this is approximately 1e-3 decay in 10k steps
    # learning_rates = list(lgbm_params['learning_rate'] * np.float_power(1 - 7e-4, np.arange(lgbm_params['n_estimators'])))
    
    eval_df = model.eval_lgbm_cv(
        X,
        y,
        params=lgbm_params, 
        cv=sfold,
        callbacks=[
            optuna.integration.LightGBMPruningCallback(trial, 'binary_logloss', valid_name='validation'),
            lgbm.early_stopping(20, first_metric_only=True)
            # lgbm.reset_parameter(learning_rate=learning_rates)
        ]
    )
    
    eval_test = eval_df.loc[eval_df.eval_set == 'test',:].mean(numeric_only=True)
    for k, v in eval_test.items():
        trial.set_user_attr(k, v)
    return eval_test['binary_logloss']

def early_stopping_grid(n_trials=21, outdir="."):
    """run optuna lightgbm early stopping grid
    """
    sql_file = f'sqlite:///{str(utils.WORKING_DIR / outdir / "lgbm_early_stopping_grid.db")}'

    study = optuna.create_study(
        storage=sql_file,
        load_if_exists=False,
        study_name='lgbm_early_stopping',
        pruner=optuna.pruners.NopPruner(), 
        direction="minimize", 
        sampler=optuna.samplers.GridSampler(
            search_space={"learning_rate":np.geomspace(1e-5, 1.0, num=n_trials).tolist()}
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
    lgbm_params = {
        "objective": 'binary',
        "n_estimators": 10000,
        "learning_rate": 1e-3, # expect around 2k trials 2-3min
        "num_leaves": trial.suggest_int("num_leaves", 7, 4095),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 100),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7)
    }

    sfold = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=1234)
    # this is approximately 1e-3 decay in 10k steps
    # learning_rates = list(lgbm_params['learning_rate'] * np.float_power(1 - 7e-4, np.arange(lgbm_params['n_estimators'])))
    
    eval_df = model.eval_lgbm_cv(
        X,
        y,
        params=lgbm_params, 
        cv=sfold,
        callbacks=[
            optuna.integration.LightGBMPruningCallback(trial, 'binary_logloss', valid_name='validation'),
            lgbm.early_stopping(20, first_metric_only=True)
            # lgbm.reset_parameter(learning_rate=learning_rates)
        ]
    )
    
    eval_test = eval_df.loc[eval_df.eval_set == 'test',:].mean(numeric_only=True)
    for k, v in eval_test.items():
        trial.set_user_attr(k, v)
    return eval_test['binary_logloss']

def stage0(prune=False, n_trials=100, timeout=3600, outdir="."):
    """run optuna lightgbm stage0 hyperparamter optimization
    """
    if prune:
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=5, n_warmup_steps=200, interval_steps=50, n_min_trials=5)
    else:
        pruner = optuna.pruners.NopPruner()
    
    sql_file = f'sqlite:///{str(utils.WORKING_DIR / outdir / "lgbm_stage0.db")}'

    study = optuna.create_study(
        storage=sql_file,
        load_if_exists=True,
        study_name='lgbm_stage0',
        pruner=pruner, 
        direction="minimize", 
        sampler=optuna.samplers.TPESampler()
    )
    
    warnings.simplefilter('ignore') # to suppress multiple callback warning
    # pre-load data for trials
    raw_train_df, target_ds = load_prep.raw_train()
    study.optimize(
        functools.partial(early_stopping_objective, X=raw_train_df, y=target_ds),
        n_trials=n_trials,
        timeout=timeout
    )
    warnings.resetwarnings()
    return study

if __name__ == "__main__":
    fire.Fire()