from __future__ import annotations

from typing import Literal

import polars as pl

from ._base import BaseEncoder


class NullEncoder(BaseEncoder):
    """Encode whether the feature is null or not with boolean values."""

    def __init__(
        self,
        cols: list[str] | None = None,
        remainder: Literal["drop", "passthrough"] = "drop",
    ):
        """

        :param cols:
            a list of columns to encode.
            if None is specified, all columns will be encoded.
        :param remainder:
            specify how to handle columns that are not listed in `cols`.
            defaults to 'drop', which removes unspecified columns from the output.
            if set to 'passthrough', unspecified columns are included in the output.
        """
        super().__init__(cols, None, None, remainder)

    def _fit(self, X: pl.DataFrame, y: pl.Series | None = None, **fit_params):
        # nothing to learn: whether a value is null is decided per row
        pass

    def _transform(self, X: pl.DataFrame, **transform_params) -> pl.DataFrame:
        X_lazy: pl.LazyFrame = X.select(self.cols_).lazy()

        for col in self.cols_:
            expr = pl.when(pl.col(col).is_null()).then(1).otherwise(0).cast(pl.Boolean)
            X_lazy = X_lazy.with_columns(expr.alias(col))

        transformed = X_lazy.collect()

        return transformed
