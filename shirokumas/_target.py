from __future__ import annotations

from typing import Any
from typing import Iterable
from typing import Literal
from typing import cast

import polars as pl
from scipy.special import expit  # pylint: disable=no-name-in-module
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.model_selection import BaseCrossValidator

from . import OutOfFoldEncodeWrapper
from ._base import BaseEncoder
from ._exceptions import NotFittedException

_UNKNOWN_VALUE = -1.0
_MISSING_VALUE = -2.0


class _GreedyTargetEncoder(BaseEncoder):
    def __init__(
        self,
        smoothing_method: Literal["none", "m-estimate", "eb"] = "none",
        smoothing_params: dict[str, Any] | None = None,
        cols: list[str] | None = None,
        handle_unknown: Literal["value", "error"] = "value",
        handle_missing: Literal["value", "error"] = "value",
        remainder: Literal["drop", "passthrough"] = "drop",
    ):
        super().__init__(cols, handle_unknown, handle_missing, remainder)
        self.smoothing_method = smoothing_method
        self.smoothing_params = smoothing_params

        self.encoder: BaseEstimator | None = None
        self.global_mean: float | None = None

    def _build_encoder(self) -> BaseEstimator:
        encoder_classes = {
            "none": _NoneSmoothingStrategy,
            "m-estimate": _MEstimateStrategy,
            "eb": _EmpiricalBayesianStrategy,
        }
        encoder_cls = encoder_classes[self.smoothing_method]
        return encoder_cls(
            **(self.smoothing_params or {}),
        )

    def _fit(self, X: pl.DataFrame, y: pl.Series | None = None, **fit_params):
        if y is None:
            raise ValueError("Need 'y' parameter")

        # Series.mean() is declared wider than a target column can be
        self.global_mean = cast(float, y.mean())

        # built here rather than in __init__ so that set_params() takes effect
        self.encoder = self._build_encoder()

        X = X.select(self.cols_)
        return self.encoder.fit(X, y, **fit_params)

    def _transform(self, X: pl.DataFrame, **transform_params) -> pl.DataFrame:
        X = X.select(self.cols_)

        transformed = self.encoder.transform(X, **transform_params)

        if self.handle_unknown == "error":
            contains_unknown = transformed.select(pl.col("*") == _UNKNOWN_VALUE).sum()
            for col in contains_unknown.columns:
                if contains_unknown.get_column(col)[0] > 0:
                    raise ValueError("Columns to be encoded can not contain unknown value")

        transformed_lazy: pl.LazyFrame = transformed.lazy()

        for col in self.cols_:
            col_value_is_missing: pl.Expr = pl.col(col) == _MISSING_VALUE
            col_value_is_unknown: pl.Expr = pl.col(col) == _UNKNOWN_VALUE
            replace_expr = (
                pl.when(col_value_is_missing | col_value_is_unknown).then(self.global_mean).otherwise(pl.col(col))
            )
            transformed_lazy = transformed_lazy.with_columns(replace_expr.alias(col))

        return transformed_lazy.collect()


class _NoneSmoothingStrategy(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.mappings: dict[str, pl.DataFrame] = {}

    def fit(self, X: pl.DataFrame, y: pl.Series):
        X_lazy: pl.LazyFrame = X.lazy().with_columns(y)

        for col in X.columns:
            local_df = (
                X_lazy.group_by(col)
                .agg(
                    [
                        pl.col(y.name).mean().alias(f"{col}_mean"),
                    ]
                )
                .collect()
            )
            self.mappings[col] = local_df

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        X_lazy: pl.LazyFrame = X.lazy()

        for col in self.mappings.keys():
            remapping = {category: local_mean for category, local_mean in self.mappings[col].rows()}
            remapping[None] = _MISSING_VALUE
            expr = pl.col(col).replace_strict(
                remapping,
                default=_UNKNOWN_VALUE,
            )
            X_lazy = X_lazy.with_columns(expr.alias(col))

        return X_lazy.collect()


class _MEstimateStrategy(BaseEstimator, TransformerMixin):
    def __init__(self, m: float = 1.0):
        self.m = m

        self.mappings: dict[str, pl.DataFrame] = {}
        self.global_mean: float | None = None

    def fit(self, X: pl.DataFrame, y: pl.Series):
        # Series.mean() is declared wider than a target column can be
        self.global_mean = cast(float, y.mean())

        X_lazy: pl.LazyFrame = X.lazy().with_columns(y)

        for col in X.columns:
            local_df = (
                X_lazy.group_by(col)
                .agg(
                    [
                        pl.col(y.name).count().alias(f"{col}_count"),
                        pl.col(y.name).sum().alias(f"{col}_sum"),
                    ]
                )
                .collect()
            )
            self.mappings[col] = local_df

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        X_lazy: pl.LazyFrame = X.lazy()

        for col in self.mappings.keys():
            remapping = {
                category: ((local_sum + self.m * self.global_mean) / (local_count + self.m))
                for category, local_count, local_sum in self.mappings[col].rows()
            }
            remapping[None] = _MISSING_VALUE
            expr = pl.col(col).replace_strict(
                remapping,
                default=_UNKNOWN_VALUE,
            )
            X_lazy = X_lazy.with_columns(expr.alias(col))

        return X_lazy.collect()


class _EmpiricalBayesianStrategy(BaseEstimator, TransformerMixin):
    def __init__(self, k: int = 20, f: int = 10):
        self.k = k
        self.f = f

        self.mappings: dict[str, pl.DataFrame] = {}
        self.global_mean: float | None = None

    def fit(self, X: pl.DataFrame, y: pl.Series):
        # Series.mean() is declared wider than a target column can be
        self.global_mean = cast(float, y.mean())

        X_lazy: pl.LazyFrame = X.lazy().with_columns(y)

        for col in X.columns:
            local_df = (
                X_lazy.group_by(col)
                .agg(
                    [
                        pl.col(y.name).mean().alias(f"{col}_mean"),
                        ((pl.col(col).count().cast(pl.Float64) - self.k) / self.f).alias(f"{col}_exp"),
                    ]
                )
                .with_columns(
                    expit(pl.col(f"{col}_exp")).alias(f"{col}_lambda"),
                )
                .drop(f"{col}_exp")
                .collect()
            )
            self.mappings[col] = local_df

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        X_lazy: pl.LazyFrame = X.lazy()

        for col in self.mappings.keys():
            remapping = {
                category: (smoothing_factor * local_mean + (1 - smoothing_factor) * self.global_mean)
                for category, local_mean, smoothing_factor in self.mappings[col].rows()
            }
            remapping[None] = _MISSING_VALUE
            expr = pl.col(col).replace_strict(
                remapping,
                default=_UNKNOWN_VALUE,
            )
            X_lazy = X_lazy.with_columns(expr.alias(col))

        return X_lazy.collect()


class TargetEncoder(BaseEstimator, TransformerMixin):
    """Encode target statistics aggregated by categorical features."""

    def __init__(
        self,
        folds: Iterable | BaseCrossValidator,
        folds_params: dict[str, Any] | None = None,
        smoothing_method: Literal["none", "m-estimate", "eb"] = "none",
        smoothing_params: dict[str, Any] | None = None,
        cols: list[str] | None = None,
        handle_unknown: Literal["value", "error"] = "value",
        handle_missing: Literal["value", "error"] = "value",
        remainder: Literal["drop", "passthrough"] = "drop",
    ):
        """

        :param folds:
            to prevent data leakage, use hold-out target statistics.
            (1) scikit-learn's BaseCrossValidator implemented instance.
            (2) iterable object that provides tuples of row numbers for training and evaluation.
        :param folds_params:
            parameters when calling split() method, if you use BaseCrossValidator instance for folds.
        :param smoothing_method:
            smoothing method.
            defaults to 'none', no smoothing.
            if 'm-estimate' is selected, m-probability estimate smoothing.
            if 'eb' is selected, empirical bayesian smoothing.
        :param smoothing_params:
            smoothing parameters.
            if 'm-estimate' is selected for smoothing_method, need to specify 'm'.
            if 'eb' is selected for smoothing_method, need to specify 'k' and 'f'.
        :param cols:
            a list of columns to encode.
            if None is specified, all columns will be encoded.
        :param handle_unknown:
            choice of handling unknown values.
            defaults to 'value', unknown values are replaced by global mean.
            if 'error' is selected, ValueError is thrown when an unknown value is encountered.
        :param handle_missing:
            choice of handling missing values.
            defaults to 'value', missing values are replaced by global mean.
            if 'error' is selected, ValueError is thrown when a missing value is encountered.
        :param remainder:
            specify how to handle columns that are not listed in `cols`.
            defaults to 'drop', which removes unspecified columns from the output.
            if set to 'passthrough', unspecified columns are included in the output.
        """
        self.folds = folds
        self.folds_params = folds_params
        self.smoothing_method = smoothing_method
        self.smoothing_params = smoothing_params
        self.cols = cols
        self.handle_unknown = handle_unknown
        self.handle_missing = handle_missing
        self.remainder = remainder

        self.encoder: OutOfFoldEncodeWrapper | None = None

    def _build_encoder(self) -> OutOfFoldEncodeWrapper:
        inner_encoder = _GreedyTargetEncoder(
            smoothing_method=self.smoothing_method,
            smoothing_params=self.smoothing_params,
            cols=self.cols,
            handle_unknown=self.handle_unknown,
            handle_missing=self.handle_missing,
            remainder=self.remainder,
        )
        return OutOfFoldEncodeWrapper(
            inner=inner_encoder,
            folds=self.folds,
            folds_params=self.folds_params,
        )

    def fit(self, X: pl.DataFrame, y: pl.Series, **fit_params):
        """Train the features.

        :param X:
            explanatory feature.
        :param y:
            objective feature.
        """
        # built here rather than in __init__ so that set_params() takes effect
        self.encoder = self._build_encoder()
        self.encoder.fit(X, y, **fit_params)
        return self

    def transform(self, X: pl.DataFrame, **transform_params) -> pl.DataFrame:
        """Transform the features.

        :param X:
            explanatory feature.
        """
        if self.encoder is None:
            raise NotFittedException("This encoder instance is not fitted yet")
        return self.encoder.transform(X, **transform_params)
