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

    def test_non_contiguous_folds_keep_row_order(self):
        train_df = pl.DataFrame(
            {
                "fruits": ["apple", "banana", "apple", "banana"],
            }
        )
        train_y = pl.Series(
            name="target",
            values=[1, 0, 0, 1],
        )
        # rows 2 and 3 are encoded by the encoder trained on rows 0 and 1,
        # and vice versa
        split_indices = [
            ([0, 1], [2, 3]),
            ([2, 3], [0, 1]),
        ]
        inner_encoder = _GreedyTargetEncoder()
        outer_encoder = OutOfFoldEncodeWrapper(inner=inner_encoder, folds=split_indices)
        outer_encoder.fit(train_df, train_y)
        encoded_df = outer_encoder.transform(train_df)

        expected_df = pl.DataFrame(
            {
                # rows 0, 1 from the rows 2, 3 encoder: apple -> 0, banana -> 1
                # rows 2, 3 from the rows 0, 1 encoder: apple -> 1, banana -> 0
                "fruits": [0.0, 1.0, 1.0, 0.0],
            }
        )
        assert_frame_equal(encoded_df, expected_df)

    def test_shuffled_folds_keep_row_order(self):
        train_df = pl.DataFrame(
            {
                "fruits": ["apple", "banana", "apple", "banana", "apple", "banana"],
                "row_id": [0, 1, 2, 3, 4, 5],
            }
        )
        train_y = pl.Series(
            name="target",
            values=[1, 0, 1, 0, 1, 0],
        )
        inner_encoder = _GreedyTargetEncoder(cols=["fruits"], remainder="passthrough")
        folds = KFold(n_splits=3, shuffle=True, random_state=42)
        outer_encoder = OutOfFoldEncodeWrapper(inner=inner_encoder, folds=folds)
        outer_encoder.fit(train_df, train_y)
        encoded_df = outer_encoder.transform(train_df)

        # the passed-through row_id has to still line up with the input
        assert encoded_df.get_column("row_id").to_list() == [0, 1, 2, 3, 4, 5]


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-svv"]))
