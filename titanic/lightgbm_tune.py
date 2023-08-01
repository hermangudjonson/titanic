"""
LightGBM hyperparameter optimization with optuna

optuna experiments defined by objectives (search space), study creation and runs

tuning progression:
 - no tuning (just CV eval)
 - n_estimators sweep (no early stopping), for timing info
"""

from titanic import load_prep, model, utils

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
import lightgbm as lgbm
import optuna
import cloudpickle

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
        "n_estimators": 2000, # reduce this for timing
        "learning_rate": 5e-2, # expect around 2k trials 2-3min, 5k seemed to take 5 hrs?
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
        functools.partial(stage0_objective, X=raw_train_df, y=target_ds),
        n_trials=n_trials,
        timeout=timeout
    )
    warnings.resetwarnings()
    return study


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
    )
    return cv_results_df


def cv_best_trial(outdir=None):
    """Fit across CV folds with best LightGBM hyperparameters.
    """
    lgbm_params = {
        "objective": 'binary',
        "n_estimators": 10000, # extended for final fit
        "learning_rate": 5e-1, # expect around 2k trials 2-3min, 5k seemed to take 5 hrs?
        "callbacks": [lgbm.early_stopping(20, first_metric_only=True)]
    }
    # best trial params
    trial_params = {
        'bagging_fraction': 0.6846162556761146,
        'bagging_freq': 4,
        'feature_fraction': 0.6486652591422358,
        'lambda_l1': 0.17370643342030245,
        'lambda_l2': 1.5596407749195192e-06,
        'min_data_in_leaf': 17,
        'num_leaves': 747
    }
    lgbm_params = lgbm_params | trial_params

    raw_train_df, target_ds = load_prep.raw_train()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)

    clf_pipe = model.clf_pipeline(clf_strategy='lightgbm', clf_params=lgbm_params)
    cv_results = model.cv_with_validation(
        clf_pipe, 
        raw_train_df, 
        target_ds, 
        cv, 
        callbacks = model.common_cv_callbacks() | {'lgbm_metrics': model._lgbm_fit_metrics}
    )
    cv_results_df = _cv_results_df(cv_results)

    if outdir is not None:
        # pickle cv results
        with open(utils.WORKING_DIR / outdir / "lgbm_best_cv.pkl", 'wb') as f:
            cloudpickle.dump(cv_results, f)
        cv_results_df.to_csv(utils.WORKING_DIR / outdir / "lgbm_best_eval_test.csv")
        
    return cv_results


if __name__ == "__main__":
    fire.Fire()