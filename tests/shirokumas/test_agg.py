import pickle
import tempfile

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from shirokumas import AggregateEncoder
from shirokumas._exceptions import NotFittedException


class TestAggregateEncoder:
    def test(self):
        train_df = pl.DataFrame(
            {
                "fruits": ["apple", "banana", "banana"],
                "vegetables": ["avocados", "beetroot", "cabbage"],
                "price": [100, 200, 300],
            }
        )
        encoder = AggregateEncoder(
            cols=[
                "fruits",
                "vegetables",
            ],
            agg_exprs={
                "mean": pl.col("price").mean(),
                "max": pl.col("price").max(),
            },
        )
        encoder.fit(train_df)
        encoded_df = encoder.transform(train_df)

        expected_df = pl.DataFrame(
            {
                "fruits_mean": [100.0, 250.0, 250.0],
                "fruits_max": [100, 300, 300],
                "vegetables_mean": [100.0, 200.0, 300.0],
                "vegetables_max": [100, 200, 300],
            },
        )
        assert_frame_equal(encoded_df, expected_df)

        test_df = pl.DataFrame(
            {
                "fruits": ["unseen", None, "banana"],
                "vegetables": ["avocados", "avocados", "cabbage"],
            }
        )
        encoded_df = encoder.transform(test_df)

        unknown = -1
        missing = -2

        expected_df = pl.DataFrame(
            {
                "fruits_mean": [unknown, missing, 250.0],
                "fruits_max": [unknown, missing, 300],
                "vegetables_mean": [100.0, 100.0, 300.0],
                "vegetables_max": [100, 100, 300],
            },
        )
        assert_frame_equal(encoded_df, expected_df)

    def test_not_fitted(self):
        train_df = pl.DataFrame(
            {
                "fruits": ["apple", "banana", "banana"],
                "price": [100, 200, 300],
            }
        )
        encoder = AggregateEncoder(
            cols=[
                "fruits",
            ],
            agg_exprs={
                "mean": pl.col("price").mean(),
            },
        )
        with pytest.raises(NotFittedException):
            encoder.transform(train_df)

    def test_handle_missing_error_fit(self):
        train_df = pl.DataFrame(
            {
                "fruits": ["apple", None, "banana"],
                "price": [100, 200, 300],
            }
        )
        encoder = AggregateEncoder(
            cols=["fruits"],
            agg_exprs={
                "max": pl.col("price").max(),
            },
            handle_missing="error",
        )
        with pytest.raises(ValueError):
            encoder.fit(train_df)

    def test_handle_missing_error_transform(self):
        train_df = pl.DataFrame(
            {
                "fruits": ["apple", "banana", "banana"],
                "price": [100, 200, 300],
            }
        )
        encoder = AggregateEncoder(
            cols=["fruits"],
            agg_exprs={
                "max": pl.col("price").max(),
            },
            handle_missing="error",
        )

        encoder.fit(train_df)

        test_df = pl.DataFrame(
            {
                "fruits": ["apple", None, "banana"],
            }
        )
        with pytest.raises(ValueError):
            encoder.transform(test_df)

    def test_handle_uknown_error(self):
        train_df = pl.DataFrame(
            {
                "fruits": ["apple", "banana", "banana"],
                "price": [100, 200, 300],
            }
        )
        encoder = AggregateEncoder(
            cols=["fruits"],
            agg_exprs={
                "max": pl.col("price").max(),
            },
            handle_unknown="error",
        )
        encoder.fit(train_df)

        test_df = pl.DataFrame(
            {
                "fruits": ["banana", "apple"],
            }
        )
        encoder.transform(test_df)

        test_df = pl.DataFrame(
            {
                "fruits": ["apple", "cherry"],
            }
        )
        with pytest.raises(ValueError):
            encoder.transform(test_df)

    def test_pickle(self):
        train_df = pl.DataFrame(
            {
                "fruits": ["apple", "banana", "banana"],
                "vegetables": ["avocados", "beetroot", "cabbage"],
                "price": [100, 200, 300],
            }
        )
        pickle_encoder = AggregateEncoder(
            cols=[
                "fruits",
                "vegetables",
            ],
            agg_exprs={
                "mean": pl.col("price").mean(),
                "max": pl.col("price").max(),
            },
        )
        pickle_encoder.fit(train_df)

        with tempfile.NamedTemporaryFile() as fp:
            pickle.dump(pickle_encoder, fp)
            fp.flush()
            fp.seek(0)
            loaded_encoder = pickle.load(fp)

        test_df = pl.DataFrame(
            {
                "fruits": ["apple", "banana"],
                "vegetables": ["avocados", "beetroot"],
            }
        )
        encoded_df = loaded_encoder.transform(test_df)

        expected_df = pl.DataFrame(
            {
                "fruits_mean": [100.0, 250.0],
                "fruits_max": [100, 300],
                "vegetables_mean": [100.0, 200.0],
                "vegetables_max": [100, 200],
            },
        )
        assert_frame_equal(encoded_df, expected_df)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-svv"]))
