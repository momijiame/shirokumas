import polars as pl
import pytest
from sklearn.base import clone

from shirokumas import AggregateEncoder
from shirokumas import CountEncoder
from shirokumas import MultiLabelBinarizer
from shirokumas import NullEncoder
from shirokumas import OneHotEncoder
from shirokumas import OrdinalEncoder
from shirokumas._target import _GreedyTargetEncoder


def _flat_df():
    return pl.DataFrame(
        {
            "fruits": ["apple", "banana"],
            "users": ["alice", "bob"],
        }
    )


def _list_df():
    return pl.DataFrame(
        {
            "fruits": [["apple"], ["banana"]],
            "users": [["alice"], ["bob"]],
        }
    )


def _aggregate_encoder(**kwargs):
    return AggregateEncoder(agg_exprs={"count": pl.len()}, **kwargs)


def _fit(encoder, df):
    """Fit an encoder, supplying a target only for the ones that need one."""
    if isinstance(encoder, _GreedyTargetEncoder):
        return encoder.fit(df, pl.Series("target", [0, 1]))
    return encoder.fit(df)


ENCODERS = [
    pytest.param(CountEncoder, _flat_df, id="CountEncoder"),
    pytest.param(NullEncoder, _flat_df, id="NullEncoder"),
    pytest.param(OneHotEncoder, _flat_df, id="OneHotEncoder"),
    pytest.param(OrdinalEncoder, _flat_df, id="OrdinalEncoder"),
    pytest.param(_aggregate_encoder, _flat_df, id="AggregateEncoder"),
    pytest.param(MultiLabelBinarizer, _list_df, id="MultiLabelBinarizer"),
    pytest.param(_GreedyTargetEncoder, _flat_df, id="GreedyTargetEncoder"),
]


@pytest.mark.parametrize("make_encoder, make_df", ENCODERS)
def test_fit_does_not_overwrite_cols(make_encoder, make_df):
    encoder = make_encoder(cols=None)
    _fit(encoder, make_df())

    assert encoder.get_params()["cols"] is None


@pytest.mark.parametrize("make_encoder, make_df", ENCODERS)
def test_clone_of_fitted_encoder_keeps_cols_none(make_encoder, make_df):
    encoder = make_encoder(cols=None)
    _fit(encoder, make_df())

    assert clone(encoder).get_params()["cols"] is None


@pytest.mark.parametrize("make_encoder, make_df", ENCODERS)
def test_cols_is_unset_before_fit(make_encoder, make_df):
    encoder = make_encoder(cols=None)

    assert not hasattr(encoder, "cols_")


@pytest.mark.parametrize("make_encoder, make_df", ENCODERS)
def test_fit_resolves_every_column_into_cols(make_encoder, make_df):
    train_df = make_df()
    encoder = make_encoder(cols=None)
    _fit(encoder, train_df)

    assert encoder.cols_ == train_df.columns


@pytest.mark.parametrize("make_encoder, make_df", ENCODERS)
def test_explicit_cols_are_kept_as_given(make_encoder, make_df):
    train_df = make_df()
    cols = train_df.columns[:1]
    encoder = make_encoder(cols=cols)
    _fit(encoder, train_df)

    assert encoder.get_params()["cols"] == cols
    assert encoder.cols_ == cols
    # cols_ is a copy, so nothing done to it can reach back into cols
    assert encoder.cols_ is not encoder.cols


@pytest.mark.parametrize("make_encoder, make_df", ENCODERS)
def test_clone_of_fitted_encoder_still_means_all_columns(make_encoder, make_df):
    train_df = make_df()
    encoder = make_encoder(cols=None)
    _fit(encoder, train_df)

    other_df = train_df.rename({name: f"other_{name}" for name in train_df.columns})
    fresh = clone(encoder)
    _fit(fresh, other_df)

    assert fresh.cols_ == other_df.columns
