from __future__ import annotations

from abc import abstractmethod
from typing import Literal

import polars as pl
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

from ._exceptions import NotFittedException


class BaseEncoder(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        cols: list[str] | None,
        handle_unknown: Literal["value", "error"] | None = "value",
        handle_missing: Literal["value", "error"] | None = "value",
    ):
        self.cols = cols
        self.handle_unknown = handle_unknown
        self.handle_missing = handle_missing

        self._fitted: bool = False

    def fit(self, X: pl.DataFrame, y: pl.Series | None = None, **fit_params):
        """Train the features.

        :param X:
            explanatory feature.
        :param y:
            objective feature.
        """
        self.cols = self.cols or X.columns

        if self.handle_missing == "error":
            contains_missing = X.select(self.cols).null_count() > 0
            for col in contains_missing.columns:
                if contains_missing.get_column(col)[0]:
                    raise ValueError("Columns to be encoded can not contain null")

        self._fit(X, y, **fit_params)

        self._fitted = True

        return self

    @abstractmethod
    def _fit(self, X: pl.DataFrame, y: pl.Series | None = None, **fit_params):
        raise NotImplementedError()

    def transform(self, X: pl.DataFrame, **transform_params) -> pl.DataFrame:
        """Transform the features.

        :param X:
            explanatory feature.
        """
        if not self._fitted:
            raise NotFittedException("This encoder instance is not fitted yet")

        if self.handle_missing == "error":
            contains_missing = X.select(self.cols).null_count() > 0
            for col in contains_missing.columns:
                if contains_missing.get_column(col)[0]:
                    raise ValueError("Columns to be encoded can not contain null")

        X_transformed = self._transform(X, **transform_params)
        return X_transformed

    @abstractmethod
    def _transform(self, X: pl.DataFrame, **transform_params) -> pl.DataFrame:
        raise NotImplementedError()
