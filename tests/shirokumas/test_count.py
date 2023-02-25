import pickle
import tempfile

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from shirokumas import CountEncoder
from shirokumas._exceptions import NotFittedException


class TestCountEncoder:
    def test(self):
        train_df = pl.DataFrame(
            {
                "fruits": ["apple", "banana", "banana"],
                "users": ["alice", "bob", "carol"],
            }
        )
        encoder = CountEncoder()
        encoder.fit(train_df)
        encoded_df = encoder.transform(train_df)

        expected_df = pl.DataFrame(
            {
                "fruits": [1, 2, 2],
                "users": [1, 1, 1],
            },
        )
        assert_frame_equal(encoded_df, expected_df)

        test_df = pl.DataFrame(
            {
                "fruits": ["unseen", None, "banana"],
                "users": ["alice", "unseen", np.nan],
            },
        )
        encoded_df = encoder.transform(test_df)

        unknown = -1
        missing = -2

        expected_df = pl.DataFrame(
            {
                "fruits": [unknown, missing, 2],
                "users": [1, unknown, missing],
            },
        )
        assert_frame_equal(encoded_df, expected_df)

    def test_cols(self):
        train_df = pl.DataFrame(
            {
                "fruits": ["apple", "banana", "banana"],
                "users": ["alice", "bob", "carol"],
            }
        )
        encoder = CountEncoder(cols=["fruits"])
        encoder.fit(train_df)
        encoded_df = encoder.transform(train_df)

        expected_df = pl.DataFrame(
            {
                "fruits": [1, 2, 2],
            },
        )
        assert_frame_equal(encoded_df, expected_df)

    def test_not_fitted(self):
        train_df = pl.DataFrame(
            {
                "fruits": ["apple", "banana", "banana"],
            }
        )
        encoder = CountEncoder()
        with pytest.raises(NotFittedException):
            encoder.transform(train_df)

    def test_handle_missing_error_fit(self):
        train_df = pl.DataFrame(
            {
                "fruits": ["apple", None],
                "users": ["alice", "bob"],
            }
        )
        encoder = CountEncoder(handle_missing="error")
        with pytest.raises(ValueError):
            encoder.fit(train_df)

    def test_handle_missing_error_transform(self):
        train_df = pl.DataFrame(
            {
                "fruits": ["apple", "apple"],
                "users": ["alice", "bob"],
            }
        )
        encoder = CountEncoder(handle_missing="error")
        encoder.fit(train_df)

        test_df = pl.DataFrame(
            {
                "fruits": ["apple", None],
                "users": ["alice", "bob"],
            }
        )
        with pytest.raises(ValueError):
            encoder.transform(test_df)

    def test_handle_uknown_error(self):
        train_df = pl.DataFrame(
            {
                "fruits": ["apple", "banana"],
                "users": ["alice", "bob"],
            }
        )
        encoder = CountEncoder(handle_unknown="error")
        encoder.fit(train_df)

        test_df = pl.DataFrame(
            {
                "fruits": ["banana", "apple"],
                "users": ["bob", "alice"],
            }
        )
        encoder.transform(test_df)

        test_df = pl.DataFrame(
            {
                "fruits": ["apple", "cherry"],
                "users": ["alice", "bob"],
            }
        )
        with pytest.raises(ValueError):
            encoder.transform(test_df)

    def test_pickle(self):
        train_df = pl.DataFrame(
            {
                "fruits": ["apple", "banana", "banana"],
                "users": ["alice", "bob", "carol"],
            }
        )
        pickle_encoder = CountEncoder()
        pickle_encoder.fit(train_df)

        with tempfile.NamedTemporaryFile() as fp:
            pickle.dump(pickle_encoder, fp)
            fp.flush()
            fp.seek(0)
            loaded_encoder = pickle.load(fp)

        test_df = pl.DataFrame(
            {
                "fruits": ["cherry", "banana", "apple"],
                "users": ["carol", "bob", "alice"],
            }
        )
        encoded_df = loaded_encoder.transform(test_df)

        expected_df = pl.DataFrame(
            {
                "fruits": [-1, 2, 1],
                "users": [1, 1, 1],
            },
        )
        assert_frame_equal(encoded_df, expected_df)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-svv"]))
