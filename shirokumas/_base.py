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
        remainder: Literal["drop", "passthrough"] = "drop",
    ):
        self.cols = cols
        self.handle_unknown = handle_unknown
        self.handle_missing = handle_missing
        self.remainder = remainder

        self._fit_col_names: list[str] = []
        self._fitted: bool = False

    def fit(self, X: pl.DataFrame, y: pl.Series | None = None, **fit_params):
        """Train the features.

        :param X:
            explanatory feature.
        :param y:
            objective feature.
        """
        if self.remainder not in ("drop", "passthrough"):
            raise ValueError(
                f"remainder must be either 'drop' or 'passthrough', got {self.remainder!r}"
            )

        self.cols = self.cols or X.columns

        if self.handle_missing == "error":
            contains_missing = X.select(self.cols).null_count() > 0
            for col in contains_missing.columns:
                if contains_missing.get_column(col)[0]:
                    raise ValueError("Columns to be encoded can not contain null")

        self._fit(X, y, **fit_params)

        self._fit_col_names = X.columns
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

        self._check_col_names(X)

        if self.handle_missing == "error":
            contains_missing = X.select(self.cols).null_count() > 0
            for col in contains_missing.columns:
                if contains_missing.get_column(col)[0]:
                    raise ValueError("Columns to be encoded can not contain null")

        X_transformed = self._transform(X, **transform_params)
        return self._handle_remainder(X, X_transformed)

    @abstractmethod
    def _transform(self, X: pl.DataFrame, **transform_params) -> pl.DataFrame:
        raise NotImplementedError()

    def _check_col_names(self, X: pl.DataFrame) -> None:
        """Make sure the columns match the ones seen at fit time.

        only checked for 'passthrough': there the output schema is the input
        schema, so an extra column at transform time would silently show up in
        the output and the train and test frames would no longer agree.
        'drop' outputs the encoded columns whatever else is handed to it, and
        an encoder is allowed to need a column only at fit time, so nothing
        diverges there.

        :param X:
            explanatory feature.
        """
        if self.remainder != "passthrough":
            return

        if X.columns == self._fit_col_names:
            return

        fitted = set(self._fit_col_names)
        given = set(X.columns)
        missing = [col for col in self._fit_col_names if col not in given]
        unexpected = [col for col in X.columns if col not in fitted]
        if missing or unexpected:
            raise ValueError(
                "Columns do not match the ones seen during fit "
                f"(missing: {missing}, unexpected: {unexpected})"
            )
        raise ValueError(
            "Columns must be in the same order as the ones seen during fit "
            f"(expected: {self._fit_col_names}, got: {X.columns})"
        )

    def _encoded_cols(self) -> list[str]:
        """Return the columns that `_transform` actually consumes.

        subclasses that encode a set of columns derived from something other
        than `cols` (a supplied `mappings`, for example) must override this.
        the declared `cols` is not a substitute: `fit` expands it to every
        column, and a subclass may encode more or fewer columns than it lists.
        """
        return list(self.cols or [])

    def _handle_remainder(
        self,
        original_df: pl.DataFrame,
        transformed_df: pl.DataFrame,
    ) -> pl.DataFrame:
        """Handle the columns that were not encoded.

        :param original_df:
            the original input dataframe.
        :param transformed_df:
            the dataframe after the encoding transformation.
        """
        if self.remainder == "passthrough":
            encoded_cols = self._encoded_cols()
            # Get columns that were not encoded
            passthrough_cols = [
                col for col in original_df.columns if col not in encoded_cols
            ]
            collisions = [
                col for col in passthrough_cols if col in transformed_df.columns
            ]
            if collisions:
                raise ValueError(
                    "Encoded columns collide with the columns to pass through: "
                    f"{collisions}"
                )
            if passthrough_cols:
                passthrough_data = original_df.select(passthrough_cols)
                # Concatenate encoded columns (left) with passthrough columns (right)
                return pl.concat([transformed_df, passthrough_data], how="horizontal")
        return transformed_df
