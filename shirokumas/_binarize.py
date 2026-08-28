from __future__ import annotations

from typing import Literal

import polars as pl

from ._base import BaseEncoder


class OneHotEncoder(BaseEncoder):
    """Encode categorical features as a one-hot matrix."""

    def __init__(
        self,
        cols: list[str] | None = None,
        handle_unknown: Literal["value", "error"] = "value",
        handle_missing: Literal["value", "error"] = "value",
        remainder: Literal["drop", "passthrough"] = "drop",
    ):
        """

        :param cols:
            a list of columns to encode.
            if None is specified, all columns will be encoded.
        :param handle_unknown:
            choice of handling unknown values.
            defaults to 'value', unknown values are replaced by all zero columns.
            if 'error' is selected, ValueError is thrown when an unknown value is encountered.
        :param handle_missing:
            choice of handling missing values.
            defaults to 'value', missing values are replaced by all zero columns.
            if 'error' is selected, ValueError is thrown when a missing value is encountered.
        :param remainder:
            specify how to handle columns that are not listed in `cols`.
            defaults to 'drop', which removes unspecified columns from the output.
            if set to 'passthrough', unspecified columns are included in the output.
        """
        super().__init__(cols, handle_unknown, handle_missing, remainder)
        self.mappings: dict[str, pl.Series] = {}

    def _fit(self, X: pl.DataFrame, y: pl.Series | None = None, **fit_params):
        cols = self.cols or X.columns

        for col in cols:
            unique_values = X.get_column(col).unique(maintain_order=True)
            self.mappings[col] = unique_values

    def _encoded_cols(self) -> list[str]:
        # the encoded set comes from the mappings, which may differ from cols
        return list(self.mappings.keys())

    def _transform(self, X: pl.DataFrame, **transform_params) -> pl.DataFrame:
        X_lazy: pl.LazyFrame = X.select(self.mappings.keys()).lazy()

        for col, unique_values in self.mappings.items():
            if self.handle_unknown == "error":
                transform_unique_values = set(X.get_column(col).unique().to_list())
                mapping_unique_values = set(unique_values.to_list())
                unknown_value = transform_unique_values - mapping_unique_values
                if len(unknown_value) > 0:
                    raise ValueError(
                        "Columns to be encoded can not contain unknown value"
                    )

            exprs = []
            for unique_value in unique_values:
                expr = (
                    pl.when(pl.col(col) == unique_value)
                    .then(1)
                    .otherwise(0)
                    .cast(pl.Boolean)
                    .alias(f"{col}_{unique_value}")
                )
                exprs.append(expr)
            X_lazy = X_lazy.with_columns(exprs)

        transformed = X_lazy.drop(list(self.mappings.keys())).collect()

        return transformed


class MultiLabelBinarizer(BaseEncoder):
    """Encode list of categorical features as a multi-hot matrix."""

    def __init__(
        self,
        cols: list[str] | None = None,
        handle_unknown: Literal["value", "error"] = "value",
        handle_missing: Literal["value", "error"] = "value",
        remainder: Literal["drop", "passthrough"] = "drop",
    ):
        """

        :param cols:
            a list of columns to binarize.
            if None is specified, all columns will be encoded.
        :param handle_unknown:
            choice of handling unknown values.
            defaults to 'value', unknown values are replaced by all zero columns.
            if 'error' is selected, ValueError is thrown when an unknown value is encountered.
        :param handle_missing:
            choice of handling missing values.
            defaults to 'value', missing values are replaced by all zero columns.
            if 'error' is selected, ValueError is thrown when a missing value is encountered.
        :param remainder:
            specify how to handle columns that are not listed in `cols`.
            defaults to 'drop', which removes unspecified columns from the output.
            if set to 'passthrough', unspecified columns are included in the output.
        """
        super().__init__(cols, handle_unknown, handle_missing, remainder)
        self.mappings: dict[str, pl.Series] = {}

    def _fit(self, X: pl.DataFrame, y: pl.Series | None = None, **fit_params):
        cols = self.cols or X.columns

        for col in cols:
            if X.get_column(col).dtype != pl.List:
                raise ValueError("Columns are expected to contain only List")

            exploded_items = X.get_column(col).explode()

            if self.handle_missing == "error":
                contains_missing = exploded_items.is_null().sum() > 0
                if contains_missing:
                    raise ValueError("Columns to be encoded can not contain null")

            unique_values = exploded_items.unique(maintain_order=True)
            self.mappings[col] = unique_values

    def _encoded_cols(self) -> list[str]:
        # the encoded set comes from the mappings, which may differ from cols
        return list(self.mappings.keys())

    def _transform(self, X: pl.DataFrame, **transform_params) -> pl.DataFrame:
        X_lazy: pl.LazyFrame = X.select(self.mappings.keys()).lazy()

        for col, unique_values in self.mappings.items():
            if X.get_column(col).dtype != pl.List:
                raise ValueError("Columns are expected to contain only List")

            if self.handle_missing == "error":
                exploded_items = X.get_column(col).explode()
                contains_missing = exploded_items.is_null().sum() > 0
                if contains_missing:
                    raise ValueError("Columns to be encoded can not contain null")

            if self.handle_unknown == "error":
                exploded_items = X.get_column(col).explode()
                transform_unique_values = set(exploded_items.unique().to_list())
                mapping_unique_values = set(unique_values.to_list())
                unknown_value = transform_unique_values - mapping_unique_values
                if len(unknown_value) > 0:
                    raise ValueError(
                        "Columns to be encoded can not contain unknown value"
                    )

            exprs = []
            for unique_value in unique_values:
                expr = (
                    pl.lit(unique_value)
                    .is_in(pl.col(col))
                    .alias(f"{col}_{unique_value}")
                )
                exprs.append(expr)
            X_lazy = X_lazy.with_columns(exprs)

        transformed = X_lazy.drop(list(self.mappings.keys())).collect()

        return transformed
