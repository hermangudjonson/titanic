"""
LightGBM hyperparameter optimization with optuna

optuna experiments defined by objectives (search space), study creation and runs

tuning progression:
 - no tuning (just CV eval)
"""

from titanic import load_prep, model, utils

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
import lightgbm as lgb
import optuna

import functools
import warnings
import fire

