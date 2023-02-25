from __future__ import annotations

from typing import Any
from typing import Iterable

import polars as pl
from polars.testing import assert_series_equal
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.base import clone
from sklearn.model_selection import BaseCrossValidator

from ._base import BaseEncoder
from ._exceptions import NotFittedException


class OutOfFoldEncodeWrapper(BaseEstimator, TransformerMixin):
    _test_encoder: BaseEncoder
    _train_row_hashes: pl.Series
    _train_col_names: list[str]
    _split_indices: list

    def __init__(
        self,
        inner: BaseEncoder,
        folds: Iterable | BaseCrossValidator,
        folds_params: dict[str, Any] | None = None,
    ):
        self.inner = inner
        self.folds = folds
        self.folds_params = folds_params

        self.train_encoders: list[BaseEncoder] = []
        self._fitted: bool = False

    def fit(self, X: pl.DataFrame, y: pl.Series, **fit_params):
        if isinstance(self.folds, BaseCrossValidator):
            indices_iter = self.folds.split(X, y, **(self.folds_params or {}))
        else:
            indices_iter = self.folds
        self._split_indices = list(indices_iter)

        n_splits = len(self._split_indices)
        self.train_encoders = [clone(self.inner) for _ in range(n_splits)]

        for encoder, (train_indices, _) in zip(
            self.train_encoders, self._split_indices
        ):
            X_train, y_train = X[train_indices], y[train_indices]
            encoder.fit(X_train, y_train, **fit_params)

        self._test_encoder = clone(self.inner)
        self._test_encoder.fit(X, y, **fit_params)

        self._train_col_names = X.columns
        self._train_row_hashes = X.hash_rows(seed=42)
        self._fitted = True

    def _is_train_df(self, X: pl.DataFrame) -> bool:
        if self._train_col_names != X.columns:
            return False
        try:
            other_row_hashes = X.hash_rows(seed=42)
            assert_series_equal(self._train_row_hashes, other_row_hashes)
            return True
        except AssertionError:
            return False

    def transform(self, X: pl.DataFrame, **transform_params) -> pl.DataFrame:
        if not self._fitted:
            raise NotFittedException("This encoder instance is not fitted yet")

        if self._is_train_df(X):
            return self._transform_train(X, **transform_params)
        return self._transform_test(X, **transform_params)

    def _transform_train(self, X: pl.DataFrame, **transform_params) -> pl.DataFrame:
        transformed_dfs = []
        for encoder, (_, eval_indices) in zip(self.train_encoders, self._split_indices):
            X_eval = X[eval_indices]
            transformed_df = encoder.transform(X_eval, **transform_params)
            transformed_dfs.append(transformed_df)

        return pl.concat(transformed_dfs)

    def _transform_test(self, X: pl.DataFrame, **transform_params) -> pl.DataFrame:
        return self._test_encoder.transform(X, **transform_params)
