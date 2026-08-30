import pickle
import tempfile

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from shirokumas import NullEncoder
from shirokumas._exceptions import NotFittedException


class TestNullEncoder:
    def test(self):
        train_df = pl.DataFrame(
            {
                "fruits": [None, "banana", "banana"],
                "prices": [None, 100, 200],
            }
        )
        encoder = NullEncoder()
        encoder.fit(train_df)
        encoded_df = encoder.transform(train_df)

        expected_df = pl.DataFrame(
            {
                "fruits": [True, False, False],
                "prices": [True, False, False],
            },
        )
        assert_frame_equal(encoded_df, expected_df)

        test_df = pl.DataFrame(
            {
                "fruits": ["unseen", None, "banana"],
                "prices": [300.0, 400.0, np.nan],
            },
        )
        encoded_df = encoder.transform(test_df)

        expected_df = pl.DataFrame(
            {
                "fruits": [False, True, False],
                "prices": [False, False, False],
            },
        )
        assert_frame_equal(encoded_df, expected_df)

    def test_cols(self):
        train_df = pl.DataFrame(
            {
                "fruits": ["apple", None, "banana"],
                "prices": [100, 200, 300],
            }
        )
        encoder = NullEncoder(cols=["fruits"])
        encoder.fit(train_df)
        encoded_df = encoder.transform(train_df)

        expected_df = pl.DataFrame(
            {
                "fruits": [False, True, False],
            },
        )
        assert_frame_equal(encoded_df, expected_df)

    def test_not_fitted(self):
        train_df = pl.DataFrame(
            {
                "fruits": ["apple", "banana", "banana"],
            }
        )
        encoder = NullEncoder()
        with pytest.raises(NotFittedException):
            encoder.transform(train_df)

    def test_pickle(self):
        train_df = pl.DataFrame(
            {
                "fruits": [None, "banana", "banana"],
                "prices": [None, 100, 200],
            }
        )
        pickle_encoder = NullEncoder()
        pickle_encoder.fit(train_df)

        with tempfile.NamedTemporaryFile() as fp:
            pickle.dump(pickle_encoder, fp)
            fp.flush()
            fp.seek(0)
            loaded_encoder = pickle.load(fp)

        test_df = pl.DataFrame(
            {
                "fruits": ["cherry", None, "apple"],
                "prices": [200, 100, None],
            }
        )
        encoded_df = loaded_encoder.transform(test_df)

        expected_df = pl.DataFrame(
            {
                "fruits": [False, True, False],
                "prices": [False, False, True],
            },
        )
        assert_frame_equal(encoded_df, expected_df)

    def test_remainder_passthrough(self):
        train_df = pl.DataFrame(
            {
                "fruits": ["apple", None, "banana"],
                "category": ["A", "B", "A"],
                "score": [1.0, 2.0, 3.0],
                "ids": [10, 20, 30],
            }
        )
        encoder = NullEncoder(cols=["fruits"], remainder="passthrough")
        encoder.fit(train_df)
        encoded_df = encoder.transform(train_df)

        expected_df = pl.DataFrame(
            {
                "fruits": [False, True, False],
                "category": ["A", "B", "A"],
                "score": [1.0, 2.0, 3.0],
                "ids": [10, 20, 30],
            }
        )
        assert_frame_equal(encoded_df, expected_df)

    def test_remainder_drop_default(self):
        train_df = pl.DataFrame(
            {
                "fruits": ["apple", None, "banana"],
                "category": ["A", "B", "A"],
                "score": [1.0, 2.0, 3.0],
            }
        )
        encoder = NullEncoder(cols=["fruits"])
        encoder.fit(train_df)
        encoded_df = encoder.transform(train_df)

        expected_df = pl.DataFrame(
            {
                "fruits": [False, True, False],
            }
        )
        assert_frame_equal(encoded_df, expected_df)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-svv"]))
