import pickle
import tempfile

import polars as pl
import pytest
from polars.testing import assert_frame_equal
from sklearn.model_selection import KFold

from shirokumas import TargetEncoder
from shirokumas._exceptions import NotFittedException
from shirokumas._target import _GreedyTargetEncoder


class TestGreedyTargetEncoder:
    def test_smoothing_none(self):
        train_df = pl.DataFrame(
            {
                "fruits": ["apple", "apple", "banana", "banana", "cherry", "cherry"],
                "users": ["alice", "alice", "alice", "alice", "bob", "bob"],
            }
        )
        train_y = pl.Series(
            name="target",
            values=[0, 0, 0, 1, 1, 1],
        )
        encoder = _GreedyTargetEncoder()
        encoder.fit(train_df, train_y)
        encoded_df = encoder.transform(train_df)

        expected_df = pl.DataFrame(
            {
                "fruits": [0.0, 0.0, 0.5, 0.5, 1.0, 1.0],
                "users": [0.25, 0.25, 0.25, 0.25, 1.0, 1.0],
            }
        )
        assert_frame_equal(encoded_df, expected_df)

        test_df = pl.DataFrame(
            {
                "fruits": ["unseen", None, "apple"],
                "users": ["alice", "bob", "bob"],
            }
        )
        encoded_df = encoder.transform(test_df)

        unknown = missing = train_y.mean()

        expected_df = pl.DataFrame(
            {
                "fruits": [unknown, missing, 0.0],
                "users": [0.25, 1.0, 1.0],
            }
        )
        assert_frame_equal(encoded_df, expected_df)

    def test_smoothing_m_estimate(self):
        train_df = pl.DataFrame(
            {
                "fruits": ["apple", "banana", "banana", "apple", "cherry"],
            }
        )
        train_y = pl.Series(
            name="target",
            values=[0, 1, 1, 1, 0],
        )
        encoder = _GreedyTargetEncoder(
            smoothing_method="m-estimate",
            smoothing_params={"m": 0},
        )
        encoder.fit(train_df, train_y)
        encoded_df = encoder.transform(train_df)

        expected_df = pl.DataFrame(
            {
                "fruits": [0.5, 1.0, 1.0, 0.5, 0.0],
            }
        )
        assert_frame_equal(encoded_df, expected_df)

        test_df = pl.DataFrame(
            {
                "fruits": ["unseen", None, "apple"],
            }
        )
        encoded_df = encoder.transform(test_df)

        unknown = missing = train_y.mean()

        expected_df = pl.DataFrame(
            {
                "fruits": [unknown, missing, 0.5],
            }
        )
        assert_frame_equal(encoded_df, expected_df)

    def test_smoothing_eb(self):
        train_df = pl.DataFrame(
            {
                "fruits": ["apple", "banana", "banana", "apple", "cherry", "cherry"],
            }
        )
        train_y = pl.Series(
            name="target",
            values=[0, 1, 1, 1, 0, 0],
        )
        encoder = _GreedyTargetEncoder(
            smoothing_method="eb",
            smoothing_params={"k": 2, "f": 2},
        )
        encoder.fit(train_df, train_y)
        encoded_df = encoder.transform(train_df)

        global_mean = train_y.mean()
        smoothing_factor = 0.5
        expected_df = pl.DataFrame(
            {
                "fruits": [
                    0.5 * smoothing_factor + global_mean * (1 - smoothing_factor),
                    1.0 * smoothing_factor + global_mean * (1 - smoothing_factor),
                    1.0 * smoothing_factor + global_mean * (1 - smoothing_factor),
                    0.5 * smoothing_factor + global_mean * (1 - smoothing_factor),
                    0.0 * smoothing_factor + global_mean * (1 - smoothing_factor),
                    0.0 * smoothing_factor + global_mean * (1 - smoothing_factor),
                ],
            }
        )
        assert_frame_equal(encoded_df, expected_df)

        test_df = pl.DataFrame(
            {
                "fruits": ["unseen", None, "apple"],
            }
        )
        encoded_df = encoder.transform(test_df)

        unknown = missing = train_y.mean()

        expected_df = pl.DataFrame(
            {
                "fruits": [
                    unknown,
                    missing,
                    0.5 * smoothing_factor + global_mean * (1 - smoothing_factor),
                ],
            }
        )
        assert_frame_equal(encoded_df, expected_df)

    def test_smoothing_eb_negative_exponent(self):
        train_df = pl.DataFrame(
            {
                "fruits": ["apple", "banana", "banana", "apple", "cherry", "cherry"],
            }
        )
        train_y = pl.Series(
            name="target",
            values=[0, 1, 1, 1, 0, 0],
        )
        encoder = _GreedyTargetEncoder(
            smoothing_method="eb",
            smoothing_params={"k": 10, "f": 10},
        )
        encoder.fit(train_df, train_y)
        encoded_df = encoder.transform(train_df)

        expected_df = pl.DataFrame(
            {
                "fruits": [
                    0.5,
                    0.655013,
                    0.655013,
                    0.5,
                    0.344987,
                    0.344987,
                ],
            }
        )
        assert_frame_equal(encoded_df, expected_df)

    def test_smoothing_eb_default_params(self):
        train_df = pl.DataFrame(
            {
                "fruits": ["apple", "banana", "banana", "apple", "cherry", "cherry"],
            }
        )
        train_y = pl.Series(
            name="target",
            values=[0, 1, 1, 1, 0, 0],
        )
        encoder = _GreedyTargetEncoder(
            smoothing_method="eb",
        )
        encoder.fit(train_df, train_y)
        encoded_df = encoder.transform(train_df)

        expected_df = pl.DataFrame(
            {
                "fruits": [
                    0.5,
                    0.570926,
                    0.570926,
                    0.5,
                    0.429074,
                    0.429074,
                ],
            }
        )
        assert_frame_equal(encoded_df, expected_df)

    def test_cols(self):
        train_df = pl.DataFrame(
            {
                "fruits": ["apple", "apple", "banana", "banana", "cherry", "cherry"],
                "users": ["alice", "alice", "alice", "alice", "bob", "bob"],
            }
        )
        train_y = pl.Series(
            name="target",
            values=[0, 0, 0, 1, 1, 1],
        )
        encoder = _GreedyTargetEncoder(cols=["fruits"])
        encoder.fit(train_df, train_y)
        encoded_df = encoder.transform(train_df)

        expected_df = pl.DataFrame(
            {
                "fruits": [0.0, 0.0, 0.5, 0.5, 1.0, 1.0],
            }
        )
        assert_frame_equal(encoded_df, expected_df)

    def test_not_fitted(self):
        train_df = pl.DataFrame(
            {
                "fruits": ["apple", "banana", "cherry"],
            }
        )
        encoder = _GreedyTargetEncoder()
        with pytest.raises(NotFittedException):
            encoder.transform(train_df)

    def test_handle_missing_error_fit(self):
        train_df = pl.DataFrame(
            {
                "fruits": ["apple", None],
            }
        )
        train_y = pl.Series(
            name="target",
            values=[0, 0],
        )
        encoder = _GreedyTargetEncoder(handle_missing="error")
        with pytest.raises(ValueError):
            encoder.fit(train_df, train_y)

    def test_handle_missing_error_transform(self):
        train_df = pl.DataFrame(
            {
                "fruits": ["apple", "apple"],
            }
        )
        train_y = pl.Series(
            name="target",
            values=[0, 0],
        )
        encoder = _GreedyTargetEncoder(handle_missing="error")
        encoder.fit(train_df, train_y)

        test_df = pl.DataFrame(
            {
                "fruits": ["apple", None],
            }
        )
        with pytest.raises(ValueError):
            encoder.transform(test_df)

    def test_handle_uknown_error(self):
        train_df = pl.DataFrame(
            {
                "fruits": ["apple", "banana"],
            }
        )
        train_y = pl.Series(
            name="target",
            values=[0, 1],
        )
        encoder = _GreedyTargetEncoder(handle_unknown="error")
        encoder.fit(train_df, train_y)

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


class TestTargetEncoder:
    def test(self):
        train_df = pl.DataFrame(
            {
                "fruits": ["apple", "banana", "banana", "apple"],
            }
        )
        train_y = pl.Series(
            name="target",
            values=[1, 0, 1, 1],
        )
        folds = KFold(n_splits=4, shuffle=False)
        outer_encoder = TargetEncoder(folds=folds)
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

    def test_fit_transform(self):
        train_df = pl.DataFrame(
            {
                "fruits": ["apple", "banana", "banana", "apple"],
            }
        )
        train_y = pl.Series(
            name="target",
            values=[1, 0, 1, 1],
        )
        folds = KFold(n_splits=4, shuffle=False)
        outer_encoder = TargetEncoder(folds=folds)
        encoded_df = outer_encoder.fit_transform(train_df, train_y)

        expected_df = pl.DataFrame(
            {
                "fruits": [1.0, 1.0, 0.0, 1.0],
            }
        )
        assert_frame_equal(encoded_df, expected_df)

    def test_not_fitted(self):
        train_df = pl.DataFrame(
            {
                "fruits": ["apple", "banana", "banana", "apple"],
            }
        )
        folds = KFold(n_splits=4, shuffle=False)
        encoder = TargetEncoder(folds=folds)
        with pytest.raises(NotFittedException):
            encoder.transform(train_df)

    def test_pickle(self):
        train_df = pl.DataFrame(
            {
                "fruits": ["apple", "banana", "banana", "apple"],
            }
        )
        train_y = pl.Series(
            name="target",
            values=[1, 0, 1, 1],
        )
        folds = KFold(n_splits=4, shuffle=False)
        pickle_encoder = TargetEncoder(folds=folds)
        pickle_encoder.fit(train_df, train_y)

        with tempfile.NamedTemporaryFile() as fp:
            pickle.dump(pickle_encoder, fp)
            fp.flush()
            fp.seek(0)
            loaded_encoder = pickle.load(fp)

        test_df = pl.DataFrame(
            {
                "fruits": ["apple", "banana", "banana", "apple"],
            }
        )
        encoded_df = loaded_encoder.transform(test_df)

        expected_df = pl.DataFrame(
            {
                "fruits": [1.0, 1.0, 0.0, 1.0],
            }
        )
        assert_frame_equal(encoded_df, expected_df)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-svv"]))
