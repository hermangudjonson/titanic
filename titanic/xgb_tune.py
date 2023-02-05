"""
XGBoost hyperparameter optimization with optuna

optuna experiments defined by objectives (search space), study creation and runs

no tuning (just CV eval)
n_estimators sweep
max_depth sweep

"""

from titanic import load_prep, model

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
import xgboost as xgb
import optuna

import functools

# stage0 (broad sweep)
def stage0_objective(trial, X, y):
    xgb_params = dict(
        objective='binary:logistic',
        n_estimators=1000,
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
    
    eval_test = eval_df.loc[eval_df.eval_set == 'test',:].mean()
    for k, v in eval_test.iteritems():
        trial.set_user_attr(k, v)
    return eval_test['logloss']

def stage0(output_dir, prune=False, n_trials=100, timeout=3600):
    """run optuna xgboost stage0 hyperparameter optimization
    """
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5, n_warmup_steps=30, interval_steps=10, n_min_trials=5)
    study = optuna.create_study(
        storage='sqlite:///xgb_tune.db',
        load_if_exists=True,
        study_name='xgb_tune_noLR',
        pruner=pruner, 
        direction="minimize", 
        sampler=optuna.samplers.TPESampler(),
    )
    
    raw_train_df, target_ds = load_prep.raw_train()
    study.optimize(
        functools.partial(stage0_objective, raw_train_df, target_ds), 
        n_trials=100, 
        timeout=3600
    )