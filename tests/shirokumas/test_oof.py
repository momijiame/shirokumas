import polars as pl
import pytest
from polars.testing import assert_frame_equal
from sklearn.model_selection import KFold

from shirokumas import OutOfFoldEncodeWrapper
from shirokumas._target import _GreedyTargetEncoder


class TestOutOfFoldEncodeWrapper:
    def test_cv(self):
        train_df = pl.DataFrame(
            {
                "fruits": ["apple", "banana", "banana", "apple"],
            }
        )
        train_y = pl.Series(
            name="target",
            values=[1, 0, 1, 1],
        )
        inner_encoder = _GreedyTargetEncoder()
        folds = KFold(n_splits=4, shuffle=False)
        outer_encoder = OutOfFoldEncodeWrapper(inner=inner_encoder, folds=folds)
        outer_encoder.fit(train_df, train_y)
        encoded_df = outer_encoder.transform(train_df)

        expected_df = pl.DataFrame(
            {
                "fruits": [1.0, 1.0, 0.0, 1.0],
            }
        )
        assert_frame_equal(encoded_df, expected_df)

        test_df = pl.DataFrame(
            {
                "fruits": ["apple", "cherry", "banana", "apple"],
            }
        )
        encoded_df = outer_encoder.transform(test_df)
        expected_df = pl.DataFrame(
            {
                "fruits": [1.0, 0.75, 0.5, 1.0],
            }
        )
        assert_frame_equal(encoded_df, expected_df)

    def test_indices(self):
        train_df = pl.DataFrame(
            {
                "fruits": ["apple", "banana", "banana", "apple"],
            }
        )
        train_y = pl.Series(
            name="target",
            values=[1, 0, 1, 1],
        )
        inner_encoder = _GreedyTargetEncoder()
        folds = KFold(n_splits=4, shuffle=False)
        split_indices = folds.split(train_df, train_y)
        outer_encoder = OutOfFoldEncodeWrapper(inner=inner_encoder, folds=split_indices)
        outer_encoder.fit(train_df, train_y)
        encoded_df = outer_encoder.transform(train_df)

        expected_df = pl.DataFrame(
            {
                "fruits": [1.0, 1.0, 0.0, 1.0],
            }
        )
        assert_frame_equal(encoded_df, expected_df)

        test_df = pl.DataFrame(
            {
                "fruits": ["apple", "cherry", "banana", "apple"],
            }
        )
        encoded_df = outer_encoder.transform(test_df)
        expected_df = pl.DataFrame(
            {
                "fruits": [1.0, 0.75, 0.5, 1.0],
            }
        )
        assert_frame_equal(encoded_df, expected_df)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-svv"]))
