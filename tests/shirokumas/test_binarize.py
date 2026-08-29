import pickle
import tempfile

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from shirokumas import MultiLabelBinarizer
from shirokumas import OneHotEncoder
from shirokumas._exceptions import NotFittedException


class TestOneHotEncoder:
    def test(self):
        train_df = pl.DataFrame(
            {
                "fruits": ["apple", "banana", "banana"],
                "users": ["alice", "bob", "charlie"],
            }
        )
        encoder = OneHotEncoder()
        encoder.fit(train_df)
        encoded_df = encoder.transform(train_df)

        expected_df = pl.DataFrame(
            {
                "fruits_apple": [True, False, False],
                "fruits_banana": [False, True, True],
                "users_alice": [True, False, False],
                "users_bob": [False, True, False],
                "users_charlie": [False, False, True],
            }
        )
        assert_frame_equal(encoded_df, expected_df)

        test_df = pl.DataFrame(
            {
                "fruits": ["unseen", None, "banana"],
                "users": ["alice", "unseen", None],
            },
        )
        encoded_df = encoder.transform(test_df)

        expected_df = pl.DataFrame(
            {
                "fruits_apple": [False, False, False],
                "fruits_banana": [False, False, True],
                "users_alice": [True, False, False],
                "users_bob": [False, False, False],
                "users_charlie": [False, False, False],
            }
        )
        assert_frame_equal(encoded_df, expected_df)

    def test_cols(self):
        train_df = pl.DataFrame(
            {
                "fruits": ["apple", "banana", "banana"],
                "users": ["alice", "bob", "charlie"],
            }
        )
        encoder = OneHotEncoder(cols=["fruits"])
        encoder.fit(train_df)
        encoded_df = encoder.transform(train_df)

        expected_df = pl.DataFrame(
            {
                "fruits_apple": [True, False, False],
                "fruits_banana": [False, True, True],
            }
        )
        assert_frame_equal(encoded_df, expected_df)

    def test_not_fitted(self):
        train_df = pl.DataFrame(
            {
                "fruits": ["apple", "banana", "banana"],
            }
        )
        encoder = OneHotEncoder()
        with pytest.raises(NotFittedException):
            encoder.transform(train_df)

    def test_handle_missing_error_fit(self):
        train_df = pl.DataFrame(
            {
                "fruits": ["apple", None],
                "users": ["alice", "bob"],
            }
        )
        encoder = OneHotEncoder(handle_missing="error")
        with pytest.raises(ValueError):
            encoder.fit(train_df)

    def test_handle_missing_error_transform(self):
        train_df = pl.DataFrame(
            {
                "fruits": ["apple", "apple"],
                "users": ["alice", "bob"],
            }
        )
        encoder = OneHotEncoder(handle_missing="error")
        encoder.fit(train_df)

        test_df = pl.DataFrame(
            {
                "fruits": ["apple", "apple"],
                "users": ["alice", None],
            }
        )
        with pytest.raises(ValueError):
            encoder.transform(test_df)

    def test_handle_unknown_error(self):
        train_df = pl.DataFrame(
            {
                "fruits": ["apple", "banana"],
                "users": ["alice", "bob"],
            }
        )
        encoder = OneHotEncoder(handle_unknown="error")
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
                "users": ["alice", "bob", "charlie"],
            }
        )
        pickle_encoder = OneHotEncoder()
        pickle_encoder.fit(train_df)

        with tempfile.NamedTemporaryFile() as fp:
            pickle.dump(pickle_encoder, fp)
            fp.flush()
            fp.seek(0)
            loaded_encoder = pickle.load(fp)

        test_df = pl.DataFrame(
            {
                "fruits": ["cherry", "banana", "apple"],
                "users": ["charlie", "bob", "alice"],
            }
        )
        encoded_df = loaded_encoder.transform(test_df)

        expected_df = pl.DataFrame(
            {
                "fruits_apple": [False, False, True],
                "fruits_banana": [False, True, False],
                "users_alice": [False, False, True],
                "users_bob": [False, True, False],
                "users_charlie": [True, False, False],
            }
        )
        assert_frame_equal(encoded_df, expected_df)

    def test_remainder_passthrough(self):
        train_df = pl.DataFrame(
            {
                "fruits": ["apple", "banana", "banana"],
                "users": ["alice", "bob", "charlie"],
                "scores": [1.0, 2.0, 3.0],
                "ids": [10, 20, 30],
            }
        )
        encoder = OneHotEncoder(cols=["fruits"], remainder="passthrough")
        encoder.fit(train_df)
        encoded_df = encoder.transform(train_df)

        expected_df = pl.DataFrame(
            {
                "fruits_apple": [True, False, False],
                "fruits_banana": [False, True, True],
                "users": ["alice", "bob", "charlie"],
                "scores": [1.0, 2.0, 3.0],
                "ids": [10, 20, 30],
            }
        )
        assert_frame_equal(encoded_df, expected_df)

    def test_remainder_drop_default(self):
        train_df = pl.DataFrame(
            {
                "fruits": ["apple", "banana", "banana"],
                "users": ["alice", "bob", "charlie"],
                "scores": [1.0, 2.0, 3.0],
            }
        )
        encoder = OneHotEncoder(cols=["fruits"])
        encoder.fit(train_df)
        encoded_df = encoder.transform(train_df)

        expected_df = pl.DataFrame(
            {
                "fruits_apple": [True, False, False],
                "fruits_banana": [False, True, True],
            }
        )
        assert_frame_equal(encoded_df, expected_df)

    def test_remainder_passthrough_all_cols(self):
        train_df = pl.DataFrame(
            {
                "fruits": ["apple", "banana", "banana"],
                "users": ["alice", "bob", "charlie"],
            }
        )
        encoder = OneHotEncoder(remainder="passthrough")
        encoder.fit(train_df)
        encoded_df = encoder.transform(train_df)

        expected_df = pl.DataFrame(
            {
                "fruits_apple": [True, False, False],
                "fruits_banana": [False, True, True],
                "users_alice": [True, False, False],
                "users_bob": [False, True, False],
                "users_charlie": [False, False, True],
            }
        )
        assert_frame_equal(encoded_df, expected_df)

    def test_remainder_passthrough_name_collision(self):
        # the generated 'fruits_apple' would clash with the existing column
        train_df = pl.DataFrame(
            {
                "fruits": ["apple", "banana", "banana"],
                "fruits_apple": [0, 1, 1],
            }
        )
        encoder = OneHotEncoder(cols=["fruits"], remainder="passthrough")
        encoder.fit(train_df)
        with pytest.raises(ValueError):
            encoder.transform(train_df)


class TestMultiLabelBinarizer:
    def test(self):
        train_df = pl.DataFrame(
            {
                "fruits": [
                    ["apple"],
                    ["banana"],
                    ["apple", "banana"],
                ],
                "users": [
                    ["alice"],
                    ["bob"],
                    ["charlie"],
                ],
            }
        )
        encoder = MultiLabelBinarizer()
        encoder.fit(train_df)
        encoded_df = encoder.transform(train_df)

        expected_df = pl.DataFrame(
            {
                "fruits_apple": [True, False, True],
                "fruits_banana": [False, True, True],
                "users_alice": [True, False, False],
                "users_bob": [False, True, False],
                "users_charlie": [False, False, True],
            }
        )
        assert_frame_equal(encoded_df, expected_df)

        test_df = pl.DataFrame(
            {
                "fruits": [
                    ["unseen"],
                    [None],
                    ["banana"],
                ],
                "users": [
                    ["alice"],
                    ["unseen"],
                    [np.nan],
                ],
            }
        )
        encoded_df = encoder.transform(test_df)

        expected_df = pl.DataFrame(
            {
                "fruits_apple": [False, False, False],
                "fruits_banana": [False, False, True],
                "users_alice": [True, False, False],
                "users_bob": [False, False, False],
                "users_charlie": [False, False, False],
            }
        )
        assert_frame_equal(encoded_df, expected_df)

    def test_cols(self):
        train_df = pl.DataFrame(
            {
                "fruits": [
                    ["apple"],
                    ["banana"],
                    ["apple", "banana"],
                ],
                "users": [
                    ["alice"],
                    ["bob"],
                    ["charlie"],
                ],
            }
        )
        encoder = MultiLabelBinarizer(cols=["fruits"])
        encoder.fit(train_df)
        encoded_df = encoder.transform(train_df)

        expected_df = pl.DataFrame(
            {
                "fruits_apple": [True, False, True],
                "fruits_banana": [False, True, True],
            }
        )
        assert_frame_equal(encoded_df, expected_df)

    def test_fit_contains_not_list(self):
        train_df = pl.DataFrame(
            {
                "fruits": [
                    ["apple"],
                    ["banana"],
                    ["apple", "banana"],
                ],
                "users": [
                    "alice",
                    "bob",
                    "charlie",
                ],
            }
        )
        encoder = MultiLabelBinarizer()
        with pytest.raises(ValueError):
            encoder.fit(train_df)

    def test_transform_contains_not_list(self):
        train_df = pl.DataFrame(
            {
                "fruits": [
                    ["apple", "banana"],
                    ["apple"],
                ],
                "users": [
                    ["alice"],
                    ["bob"],
                ],
            }
        )
        encoder = MultiLabelBinarizer()
        encoder.fit(train_df)

        test_df = pl.DataFrame(
            {
                "fruits": [
                    ["apple"],
                    ["apple"],
                ],
                "users": [
                    "alice",
                    "bob",
                ],
            }
        )
        with pytest.raises(ValueError):
            encoder.transform(test_df)

    def test_not_fitted(self):
        train_df = pl.DataFrame(
            {
                "fruits": [
                    ["apple"],
                    ["banana"],
                    ["banana"],
                ],
            }
        )
        encoder = MultiLabelBinarizer()
        with pytest.raises(NotFittedException):
            encoder.transform(train_df)

    def test_handle_missing_error_fit(self):
        train_df = pl.DataFrame(
            {
                "fruits": [
                    ["apple", "banana"],
                    None,
                ],
                "users": [
                    ["alice"],
                    ["bob"],
                ],
            }
        )
        encoder = MultiLabelBinarizer(handle_missing="error")
        with pytest.raises(ValueError):
            encoder.fit(train_df)

    def test_handle_missing_error_fit_item(self):
        train_df = pl.DataFrame(
            {
                "fruits": [
                    ["apple", "banana"],
                    [None],
                ],
                "users": [
                    ["alice"],
                    ["bob"],
                ],
            }
        )
        encoder = MultiLabelBinarizer(handle_missing="error")
        with pytest.raises(ValueError):
            encoder.fit(train_df)

    def test_handle_missing_error_transform(self):
        train_df = pl.DataFrame(
            {
                "fruits": [
                    ["apple", "banana"],
                    ["apple"],
                ],
                "users": [
                    ["alice"],
                    ["bob"],
                ],
            }
        )
        encoder = MultiLabelBinarizer(handle_missing="error")
        encoder.fit(train_df)

        test_df = pl.DataFrame(
            {
                "fruits": [
                    ["apple"],
                    ["apple"],
                ],
                "users": [
                    ["alice"],
                    None,
                ],
            }
        )
        with pytest.raises(ValueError):
            encoder.transform(test_df)

    def test_handle_missing_error_transform_value(self):
        train_df = pl.DataFrame(
            {
                "fruits": [
                    ["apple", "banana"],
                    ["apple"],
                ],
                "users": [
                    ["alice"],
                    ["bob"],
                ],
            }
        )
        encoder = MultiLabelBinarizer(handle_missing="error")
        encoder.fit(train_df)

        test_df = pl.DataFrame(
            {
                "fruits": [
                    ["apple"],
                    ["apple"],
                ],
                "users": [
                    ["alice"],
                    [None],
                ],
            }
        )
        with pytest.raises(ValueError):
            encoder.transform(test_df)

    def test_empty_list_is_not_missing_fit(self):
        train_df = pl.DataFrame(
            {
                "fruits": [
                    ["apple", "banana"],
                    [],
                ],
                "users": [
                    ["alice"],
                    ["bob"],
                ],
            }
        )
        encoder = MultiLabelBinarizer(handle_missing="error")
        encoded_df = encoder.fit_transform(train_df)

        expected_df = pl.DataFrame(
            {
                "fruits_apple": [True, False],
                "fruits_banana": [True, False],
                "users_alice": [True, False],
                "users_bob": [False, True],
            }
        )
        assert_frame_equal(encoded_df, expected_df)

    def test_empty_list_is_not_missing_transform(self):
        train_df = pl.DataFrame(
            {
                "fruits": [
                    ["apple", "banana"],
                    ["apple"],
                ],
            }
        )
        encoder = MultiLabelBinarizer(handle_missing="error")
        encoder.fit(train_df)

        test_df = pl.DataFrame(
            {
                "fruits": [
                    ["apple"],
                    [],
                ],
            }
        )
        encoded_df = encoder.transform(test_df)

        expected_df = pl.DataFrame(
            {
                "fruits_apple": [True, False],
                "fruits_banana": [False, False],
            }
        )
        assert_frame_equal(encoded_df, expected_df)

    def test_empty_list_is_not_unknown(self):
        train_df = pl.DataFrame(
            {
                "fruits": [
                    ["apple"],
                    ["banana"],
                ],
            }
        )
        encoder = MultiLabelBinarizer(handle_unknown="error")
        encoder.fit(train_df)

        test_df = pl.DataFrame(
            {
                "fruits": [
                    ["apple"],
                    [],
                ],
            }
        )
        encoded_df = encoder.transform(test_df)

        expected_df = pl.DataFrame(
            {
                "fruits_apple": [True, False],
                "fruits_banana": [False, False],
            }
        )
        assert_frame_equal(encoded_df, expected_df)

    def test_handle_unknown_error(self):
        train_df = pl.DataFrame(
            {
                "fruits": [
                    ["apple"],
                    ["banana"],
                ],
                "users": [
                    ["alice"],
                    ["bob"],
                ],
            }
        )
        encoder = MultiLabelBinarizer(handle_unknown="error")
        encoder.fit(train_df)

        test_df = pl.DataFrame(
            {
                "fruits": [
                    ["banana"],
                    ["apple"],
                ],
                "users": [
                    ["bob"],
                    ["alice"],
                ],
            }
        )
        encoder.transform(test_df)

        test_df = pl.DataFrame(
            {
                "fruits": [
                    ["apple"],
                    ["cherry"],
                ],
                "users": [
                    ["alice"],
                    ["bob"],
                ],
            }
        )
        with pytest.raises(ValueError):
            encoder.transform(test_df)

    def test_pickle(self):
        train_df = pl.DataFrame(
            {
                "fruits": [
                    ["apple"],
                    ["banana"],
                    ["banana"],
                ],
                "users": [
                    ["alice"],
                    ["bob"],
                    ["charlie"],
                ],
            }
        )
        pickle_encoder = MultiLabelBinarizer()
        pickle_encoder.fit(train_df)

        with tempfile.NamedTemporaryFile() as fp:
            pickle.dump(pickle_encoder, fp)
            fp.flush()
            fp.seek(0)
            loaded_encoder = pickle.load(fp)

        test_df = pl.DataFrame(
            {
                "fruits": [
                    ["cherry"],
                    ["banana"],
                    ["apple"],
                ],
                "users": [
                    ["charlie"],
                    ["bob"],
                    ["alice"],
                ],
            }
        )
        encoded_df = loaded_encoder.transform(test_df)

        expected_df = pl.DataFrame(
            {
                "fruits_apple": [False, False, True],
                "fruits_banana": [False, True, False],
                "users_alice": [False, False, True],
                "users_bob": [False, True, False],
                "users_charlie": [True, False, False],
            }
        )
        assert_frame_equal(encoded_df, expected_df)

    def test_remainder_passthrough(self):
        train_df = pl.DataFrame(
            {
                "fruits": [
                    ["apple"],
                    ["banana"],
                    ["apple", "banana"],
                ],
                "users": [
                    ["alice"],
                    ["bob"],
                    ["charlie"],
                ],
                "scores": [1.0, 2.0, 3.0],
                "ids": [10, 20, 30],
            }
        )
        encoder = MultiLabelBinarizer(cols=["fruits"], remainder="passthrough")
        encoder.fit(train_df)
        encoded_df = encoder.transform(train_df)

        expected_df = pl.DataFrame(
            {
                "fruits_apple": [True, False, True],
                "fruits_banana": [False, True, True],
                "users": [
                    ["alice"],
                    ["bob"],
                    ["charlie"],
                ],
                "scores": [1.0, 2.0, 3.0],
                "ids": [10, 20, 30],
            }
        )
        assert_frame_equal(encoded_df, expected_df)

    def test_remainder_drop_default(self):
        train_df = pl.DataFrame(
            {
                "fruits": [
                    ["apple"],
                    ["banana"],
                    ["apple", "banana"],
                ],
                "users": [
                    ["alice"],
                    ["bob"],
                    ["charlie"],
                ],
                "scores": [1.0, 2.0, 3.0],
            }
        )
        encoder = MultiLabelBinarizer(cols=["fruits"])
        encoder.fit(train_df)
        encoded_df = encoder.transform(train_df)

        expected_df = pl.DataFrame(
            {
                "fruits_apple": [True, False, True],
                "fruits_banana": [False, True, True],
            }
        )
        assert_frame_equal(encoded_df, expected_df)

    def test_remainder_passthrough_all_cols(self):
        train_df = pl.DataFrame(
            {
                "fruits": [
                    ["apple"],
                    ["banana"],
                    ["apple", "banana"],
                ],
                "users": [
                    ["alice"],
                    ["bob"],
                    ["charlie"],
                ],
            }
        )
        encoder = MultiLabelBinarizer(remainder="passthrough")
        encoder.fit(train_df)
        encoded_df = encoder.transform(train_df)

        expected_df = pl.DataFrame(
            {
                "fruits_apple": [True, False, True],
                "fruits_banana": [False, True, True],
                "users_alice": [True, False, False],
                "users_bob": [False, True, False],
                "users_charlie": [False, False, True],
            }
        )
        assert_frame_equal(encoded_df, expected_df)

    def test_remainder_passthrough_name_collision(self):
        # the generated 'tags_x' would clash with the existing column
        train_df = pl.DataFrame(
            {
                "tags": [["x", "y"], ["y"], ["x"]],
                "tags_x": [1, 0, 1],
            }
        )
        encoder = MultiLabelBinarizer(cols=["tags"], remainder="passthrough")
        encoder.fit(train_df)
        with pytest.raises(ValueError):
            encoder.transform(train_df)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-svv"]))
