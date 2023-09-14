"""
Tuning routines for other models, saving cross-validation results
if training is expensive.

 - MLP with optuna search CV
"""


import cloudpickle
import fire
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from titanic import load_prep, model, utils


def _cv_results_df(cv_results: dict):
    """collate eval test results from cv_with_validation return"""
    folds = cv_results["eval_test"].keys()
    cv_results_df = pd.concat(
        [
            pd.concat(cv_results["eval_test"].values(), keys=folds, ignore_index=True),
        ],
        axis=1,
    ).infer_objects()
    return cv_results_df


def mlp_cv_best(outdir=None):
    """Fit and save MLPClassifier cross-validation"""
    nn_params = {
        # static params
        "hidden_layer_sizes": (100, 20, 20),
        "max_iter": 1000,
        "early_stopping": True,
        "n_iter_no_change": 50,
    }

    raw_train_df, target_ds = load_prep.raw_train()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)

    clf_pipe = model.clf_pipeline(clf_strategy="neuralnet", clf_params=nn_params)
    cv_results = model.cv_with_validation(
        clf_pipe, raw_train_df, target_ds, cv, callbacks=model.common_cv_callbacks()
    )
    cv_results_df = _cv_results_df(cv_results)

    if outdir is not None:
        # pickle cv results
        with open(utils.WORKING_DIR / outdir / "mlp_best_cv.pkl", "wb") as f:
            cloudpickle.dump(cv_results, f)
        cv_results_df.to_csv(utils.WORKING_DIR / outdir / "mlp_best_eval_test.csv")


if __name__ == "__main__":
    fire.Fire()
