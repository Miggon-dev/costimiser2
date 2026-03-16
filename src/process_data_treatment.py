import pandas as pd
import streamlit as st


def remove_empty_columns(df):
    """Remove columns that are completely empty (100% NaN)."""
    df_cleaned = df.dropna(axis=1, how='all')
    print(f"Removed empty columns. Remaining columns: {df_cleaned.shape[1]}")
    return df_cleaned


def drop_rows_with_missing_values(df, columns):
    """Drop rows where any of the specified columns have NaN values."""
    df_cleaned = df.dropna(subset=columns)
    print(f"Dropped rows with missing values in {columns}. Remaining rows: {df_cleaned.shape[0]}")
    return df_cleaned


def forward_fill_by_group(df, group_col="AB_Grade_ID"):
    """Apply forward-fill (ffill) within each group, ensuring missing values are filled sequentially."""
    df_sorted = df.sort_values(by=[group_col, 'Wedge_Time'])  # Ensure correct order before filling
    df_filled = df_sorted.groupby(group_col).apply(lambda group: group.fillna(method='ffill'))
    df_filled = df_filled.reset_index(drop=True)
    print(f"Forward-filled missing values within each {group_col} group.")
    return df_filled


def replace_placeholder_values(df, placeholder1=999999, placeholder2=-999999):
    """Replace placeholder values with NaN column by column (avoiding recursion)."""
    for col in df.columns:
        df[col] = df[col].mask(df[col] == placeholder1, pd.NA)
        df[col] = df[col].mask(df[col] == placeholder2, pd.NA)
    print(f"Replaced {placeholder1} and {placeholder2} with NaN column by column.")
    return df



def fill_na_with_median(df, columns, group_col="AB_Grade_ID"):
    """Fill NaN values in specified columns using the median of their respective group."""
    missing_before = df[columns].isna().sum().sum()
    if missing_before == 0:
        print(f"No missing values in {columns}, skipping median imputation.")
        return df

    for column in columns:
        df[column] = df.groupby(group_col)[column].transform(lambda x: x.fillna(x.median()))

    missing_after = df[columns].isna().sum().sum()
    print(f"Median imputation applied. Missing values reduced from {missing_before} to {missing_after}.")
    return df
