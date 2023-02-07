"""
XGBoost hyperparameter optimization with optuna

optuna experiments defined by objectives (search space), study creation and runs

no tuning (just CV eval)
n_estimators sweep (no early stopping), for timing info
max_depth sweep
stage0 broad sweep 
"""

from titanic import load_prep, model, utils

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
import xgboost as xgb
import optuna

import functools
import warnings
import fire

# n_estimators grid
def n_estimators_objective(trial, X, y):
    """objective for n_estimators grid
    """
    xgb_params = dict(
        objective='binary:logistic',
        n_estimators=trial.suggest_int("n_estimators", 1, 1000),
        learning_rate=5e-3,
        early_stopping_rounds=None, # no early stopping
        eval_metric=['error','auc','aucpr','logloss'],
    )

    sfold = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=1234)
    # this is approximately 1e-3 decay in 10k steps
    learning_rates = list(xgb_params['learning_rate'] * np.float_power(1 - 7e-4, np.arange(xgb_params['n_estimators'])))
    
    eval_df = model.eval_xgb_cv(
        X,
        y,
        params=xgb_params, 
        cv=sfold,
        callbacks=[
            optuna.integration.XGBoostPruningCallback(trial, 'validation_1-logloss'),
            xgb.callback.LearningRateScheduler(learning_rates)
        ]
    )
    
    eval_test = eval_df.loc[eval_df.eval_set == 'test',:].mean(numeric_only=True)
    for k, v in eval_test.items():
        trial.set_user_attr(k, v)
    return eval_test['logloss']

def n_estimators_grid(n_trials=20, outdir="."):
    """run optuna xgboost n_estimators grid 
    """
    sql_file = f'sqlite:///{str(utils.WORKING_DIR / outdir / "xgb_n_estimators_grid.db")}'

    study = optuna.create_study(
        storage=sql_file,
        load_if_exists=False,
        study_name='xgb_n_estimators',
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

# max_depth grid
def max_depth_objective(trial, X, y):
    """objective for max_depth grid
    """
    xgb_params = dict(
        objective='binary:logistic',
        n_estimators=10000,
        learning_rate=5e-3,
        max_depth=trial.suggest_int("max_depth", 1, 30),
        early_stopping_rounds=20,
        eval_metric=['error','auc','aucpr','logloss'],
    )

    sfold = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=1234)
    # this is approximately 1e-3 decay in 10k steps
    learning_rates = list(xgb_params['learning_rate'] * np.float_power(1 - 7e-4, np.arange(xgb_params['n_estimators'])))
    
    eval_df = model.eval_xgb_cv(
        X,
        y,
        params=xgb_params, 
        cv=sfold,
        callbacks=[
            optuna.integration.XGBoostPruningCallback(trial, 'validation_1-logloss'),
            xgb.callback.LearningRateScheduler(learning_rates)
        ]
    )
    
    eval_test = eval_df.loc[eval_df.eval_set == 'test',:].mean(numeric_only=True)
    for k, v in eval_test.items():
        trial.set_user_attr(k, v)
    return eval_test['logloss']

def max_depth_grid(outdir="."):
    """run optuna xgboost max_depth grid 
    """
    sql_file = f'sqlite:///{str(utils.WORKING_DIR / outdir / "xgb_max_depth_grid.db")}'

    study = optuna.create_study(
        storage=sql_file,
        load_if_exists=False,
        study_name='xgb_max_depth',
        pruner=optuna.pruners.NopPruner(), 
        direction="minimize", 
        sampler=optuna.samplers.GridSampler(
            search_space={"max_depth":list(range(2, 31))}
        )
    )
    
    warnings.simplefilter('ignore') # to suppress multiple callback warning
    # pre-load data for trials
    raw_train_df, target_ds = load_prep.raw_train()
    study.optimize(
        functools.partial(max_depth_objective, X=raw_train_df, y=target_ds)
    )
    warnings.resetwarnings()
    return study

# stage0 (broad sweep)
def stage0_objective(trial, X, y):
    """objective for stage0 broad parameter sweep
    """
    xgb_params = dict(
        objective='binary:logistic',
        n_estimators=10000,
        learning_rate=trial.suggest_float('learning_rate', 1e-8, 1.0, log=True),
        alpha=trial.suggest_float('alpha', 1e-8, 1.0, log=True),
        max_depth=trial.suggest_int("max_depth", 1, 15),
        subsample=trial.suggest_float('subsample', 0.5, 1.0),
        colsample_bytree=trial.suggest_float('colsample_bytree', 0.5, 1.0),
        gamma=trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        grow_policy=trial.suggest_categorical('grow_policy', ['depthwise','lossguide']),
        
        early_stopping_rounds=20,
        eval_metric=['error','auc','aucpr','logloss'],
    )
    xgb_params['lambda'] = trial.suggest_float('lambda', 1e-8, 1.0, log=True)

    sfold = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=1234)
    # this is approximately 1e-3 decay in 10k steps
    learning_rates = list(xgb_params['learning_rate'] * np.float_power(1 - 7e-4, np.arange(xgb_params['n_estimators'])))
    
    eval_df = model.eval_xgb_cv(
        X,
        y,
        params=xgb_params, 
        cv=sfold,
        callbacks=[
            optuna.integration.XGBoostPruningCallback(trial, 'validation_1-logloss'),
            xgb.callback.LearningRateScheduler(learning_rates)
        ]
    )
    
    eval_test = eval_df.loc[eval_df.eval_set == 'test',:].mean(numeric_only=True)
    for k, v in eval_test.items():
        trial.set_user_attr(k, v)
    return eval_test['logloss']

def stage0(prune=False, n_trials=100, timeout=3600, outdir="."):
    """run optuna xgboost stage0 hyperparameter optimization
    """
    if prune:
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=5, n_warmup_steps=30, interval_steps=10, n_min_trials=5)
    else:
        pruner = optuna.pruners.NopPruner()
    
    sql_file = f'sqlite:///{str(utils.WORKING_DIR / outdir / "xgb_stage0.db")}'

    study = optuna.create_study(
        storage=sql_file,
        load_if_exists=True,
        study_name='xgb_stage0',
        pruner=pruner, 
        direction="minimize", 
        sampler=optuna.samplers.TPESampler(),
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

if __name__ == "__main__":
    fire.Fire()