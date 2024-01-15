from __future__ import annotations

from collections import defaultdict
from typing import Literal

import polars as pl

from ._base import BaseEncoder


class AggregateEncoder(BaseEncoder):
    """Encode summary statistics aggregated by categorical features."""

    def __init__(
        self,
        agg_exprs: dict[str, pl.Expr],
        cols: list[str],
        handle_unknown: Literal["value", "error"] = "value",
        handle_missing: Literal["value", "error"] = "value",
    ):
        """

        :param agg_exprs:
            a list of aggregation expressions.
            for example, to calculate the average value of the 'a' column for each group: [pl.col('x').mean()]
        :param cols:
            a list of columns to aggregate.
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
        self.agg_exprs = agg_exprs

        self.mappings: dict[str, pl.DataFrame] = {}

    def _fit(self, X: pl.DataFrame, y: pl.Series | None = None, **fit_params):
        for col in self.cols:
            self.mappings[col] = X.group_by(
                by=col,
            ).agg(
                [expr.alias(f"{col}_{name}") for name, expr in self.agg_exprs.items()]
            )

    def _transform(self, X: pl.DataFrame, **transform_params) -> pl.DataFrame:
        unknown_value = -1
        missing_value = -2

        X_lazy: pl.LazyFrame = X.select(self.mappings.keys()).lazy()

        for col, mapping in self.mappings.items():
            col_remappings: dict[str, dict[str | None, float | int]] = defaultdict(
                lambda: {None: missing_value}
            )
            for category, *agg_values in mapping.rows():
                for agg_name, agg_value in zip(mapping.columns[1:], agg_values):
                    col_remappings[agg_name][category] = agg_value
            X_lazy = X_lazy.with_columns(
                [
                    pl.col(col)
                    .replace(remapping, default=unknown_value)
                    .alias(agg_name)
                    for agg_name, remapping in col_remappings.items()
                ]
            )

        transformed = X_lazy.drop(list(self.mappings.keys())).collect()

        if self.handle_unknown == "error":
            contains_unknown = transformed.select(pl.col("*") == unknown_value).sum()
            for col in contains_unknown.columns:
                if contains_unknown.get_column(col)[0] > 0:
                    raise ValueError(
                        "Columns to be encoded can not contain unknown value"
                    )

        return transformed
