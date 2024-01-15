from __future__ import annotations

import warnings
from typing import Literal

import polars as pl

from ._base import BaseEncoder


class OrdinalEncoder(BaseEncoder):
    """Encode categorical features as ordinal values."""

    def __init__(
        self,
        cols: list[str] | None = None,
        mappings: dict[str, dict[str, int]] | None = None,
        handle_unknown: Literal["value", "error"] = "value",
        handle_missing: Literal["value", "error"] = "value",
    ):
        """

        :param cols:
            a list of columns to encode.
            if None is specified, all columns will be encoded.
        :param mappings:
            mappings between the original values and the values to be encoded.
        :param handle_unknown:
            choice of handling unknown values.
            defaults to 'value', unknown values are replaced by -1.
            If 'error' is selected, ValueError is thrown when an unknown value is encountered.
        :param handle_missing:
            choice of handling missing values.
            defaults to 'value', missing values are replaced by -2.
            If 'error' is selected, ValueError is thrown when a missing value is encountered.
        """
        super().__init__(cols, handle_unknown, handle_missing)
        self.mappings = mappings
        self.mappings_supplied = mappings is not None

    def _fit(self, X: pl.DataFrame, y: pl.Series | None = None, **fit_params):
        if self.mappings_supplied:
            return

        self.mappings = {}
        cols = self.cols or X.columns
        for col in cols:
            unique_values = X.get_column(col).unique(maintain_order=True).to_list()
            self.mappings[col] = {
                value: i for i, value in enumerate(unique_values, start=1)
            }

    def _transform(self, X: pl.DataFrame, **transform_params) -> pl.DataFrame:
        if self.mappings is None:
            warnings.warn("no mappings exists, nothing to do")
            self.mappings = {}

        unknown_value = -1
        missing_value = -2

        X_lazy: pl.LazyFrame = X.select(self.mappings.keys()).lazy()

        for col in self.mappings.keys():
            remapping = self.mappings[col]
            remapping[None] = missing_value
            expr = pl.col(col).replace(
                remapping,
                default=unknown_value,
            )
            X_lazy = X_lazy.with_columns(expr.alias(col))

        transformed = X_lazy.collect()

        if self.handle_unknown == "error":
            contains_unknown = transformed.select(pl.col("*") == unknown_value).sum()
            for col in contains_unknown.columns:
                if contains_unknown.get_column(col)[0] > 0:
                    raise ValueError(
                        "Columns to be encoded can not contain unknown value"
                    )

        return transformed
