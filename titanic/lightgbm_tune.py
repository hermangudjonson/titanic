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

def stage0_objective(trial, X, y):
    """
    """

def stage0():
    """
    """


if __name__ == "__main__":
    fire.Fire()