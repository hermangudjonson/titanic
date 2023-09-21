"""writing some sample pytests
"""

import pandas as pd

from titanic import load_prep


def test_raw_train() -> None:
    train_df, target_ds = load_prep.raw_train()

    assert train_df.shape[1] == 10
    assert "Survived" not in train_df.columns
    assert isinstance(target_ds, pd.Series)


def test_raw_test() -> None:
    test_df = load_prep.raw_test()
    assert test_df.shape[1] == 10
    assert "Survived" not in test_df.columns
