import numpy as np
import pandas as pd


def standardize_missing_values(df: pd.DataFrame, missing_values_types=None, cols=None) -> pd.DataFrame:
    """Replace placeholder tokens and missing markers with NaN in the selected columns."""
    if missing_values_types is None:
        missing_values_types = []
    if cols is None:
        cols = df.select_dtypes(include=["object", "category", "string"]).columns.tolist()

    for col in cols:
        if col in df.columns:
            df[col] = df[col].replace(missing_values_types, np.nan)
    return df


def transform_cat_to_num(df: pd.DataFrame, cols=None) -> pd.DataFrame:
    """Convert numeric text columns to numeric values while coercing invalid strings to NaN."""
    if cols is None:
        cols = []

    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def change_categorical_col(df: pd.DataFrame, category_maps=None, cat_cols=None) -> pd.DataFrame:
    """Canonicalize values in categorical columns based on a mapping dictionary."""
    if category_maps is None:
        category_maps = {}
    if cat_cols is None:
        cat_cols = list(category_maps.keys())

    for col in cat_cols:
        if col not in df.columns:
            continue

        mapping = category_maps.get(col, {})
        if not mapping:
            continue

        cleaned = df[col].apply(
            lambda value: str(value).strip().lower() if pd.notna(value) and str(value).strip() != "" else np.nan
        )
        mapped = cleaned.map(mapping)
        df[col] = mapped.where(mapped.notna(), df[col])

    return df


def clean_numerical_cols(df: pd.DataFrame, rules=None, cols=None) -> pd.DataFrame:
    """Apply min/max validity rules to a set of numeric columns, setting invalid values to NaN."""
    if rules is None:
        rules = {}
    if cols is None:
        cols = list(rules.keys())

    for col in cols:
        if col not in df.columns:
            continue
        bounds = rules.get(col, {})
        if not bounds:
            continue

        numeric = pd.to_numeric(df[col], errors="coerce")
        if "min" in bounds:
            df.loc[numeric < bounds["min"], col] = np.nan
        if "max" in bounds:
            df.loc[numeric > bounds["max"], col] = np.nan

    return df


def remove_duplicates(df: pd.DataFrame, subset=None) -> pd.DataFrame:
    """Drop duplicate rows, optionally within a unique id column."""
    if subset is not None and subset in df.columns:
        return df.drop_duplicates(subset=subset, keep="first")
    return df.drop_duplicates()


def clean_dataset(df: pd.DataFrame, diagnostics_config: dict) -> pd.DataFrame:
    """Run the entire cleaning pipeline using the configuration file as the source of truth."""
    if df is None:
        raise ValueError("A dataframe is required for cleaning.")

    out = df.copy()
    config = diagnostics_config or {}

    placeholder_tokens = list(config.get("placeholder_tokens", ["-", "?", "n/a", "N/A", "na", "NA", ""]))
    numeric_text_columns = config.get("numeric_text_columns", [])
    canonical_categories = config.get("canonical_categories", {})
    validity_rules = config.get("validity_rules", {})
    redundant_columns = config.get("redundant_columns", [])
    id_column = config.get("id_column")

    out = standardize_missing_values(
        out,
        missing_values_types=placeholder_tokens,
        cols=out.select_dtypes(include=["object", "category", "string"]).columns.tolist(),
    )

    out = transform_cat_to_num(out, cols=numeric_text_columns)

    out = change_categorical_col(
        out,
        category_maps=canonical_categories,
        cat_cols=list(canonical_categories.keys()),
    )

    out = clean_numerical_cols(
        out,
        rules=validity_rules,
        cols=list(validity_rules.keys()),
    )

    out = remove_duplicates(out, subset=id_column if id_column and id_column in out.columns else None)

    for col in redundant_columns:
        if col in out.columns:
            out = out.drop(columns=[col])

    return out
