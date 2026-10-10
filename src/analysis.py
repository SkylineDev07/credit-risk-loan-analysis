
"""Generate exploratory data analysis tables for credit risk."""

import pandas as pd

import config


def load_cleaned_data() -> pd.DataFrame:
    """Load and validate the processed dataset."""
    if not config.CLEANED_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {config.CLEANED_DATA_PATH}. "
            "Run python -m src.preprocessing first."
        )

    df = pd.read_csv(config.CLEANED_DATA_PATH)

    if df.columns.tolist() != config.REQUIRED_COLUMNS:
        raise ValueError("Cleaned dataset does not match the required schema.")

    if df.empty:
        raise ValueError("Cleaned dataset is empty.")

    if df.isna().any().any():
        raise ValueError("Cleaned dataset contains missing values.")

    if not df[config.TARGET_COLUMN].isin(
        [config.DEFAULT_LABEL, config.NON_DEFAULT_LABEL]
    ).all():
        raise ValueError("Default_Status must contain only 0 and 1.")

    return df


def summarize_default_status(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize application counts and the overall default rate."""
    target = config.TARGET_COLUMN
    total = len(df)
    defaults = int((df[target] == config.DEFAULT_LABEL).sum())
    non_defaults = int((df[target] == config.NON_DEFAULT_LABEL).sum())

    return pd.DataFrame([{
        "Total_Applications": total,
        "Non_Defaults": non_defaults,
        "Defaults": defaults,
        "Default_Rate_Percent": round(defaults / total * 100, 2),
    }])


def summarize_numerical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate descriptive statistics for numerical features."""
    numerical_columns = df.select_dtypes(include="number").columns.tolist()

    if config.IDENTIFIER_COLUMN in numerical_columns:
        numerical_columns.remove(config.IDENTIFIER_COLUMN)

    return df[numerical_columns].describe().T.reset_index(
        names="Feature"
    )


def compare_by_default_status(df: pd.DataFrame) -> pd.DataFrame:
    """Compare numerical feature averages by default status."""
    numerical_columns = df.select_dtypes(include="number").columns.tolist()

    excluded = {
        config.IDENTIFIER_COLUMN,
        config.TARGET_COLUMN,
    }
    numerical_columns = [
        column for column in numerical_columns if column not in excluded
    ]

    result = df.groupby(config.TARGET_COLUMN)[numerical_columns].mean().T
    result = result.reset_index(names="Feature")
    result.columns.name = None
    result = result.rename(columns={
        config.NON_DEFAULT_LABEL: "Non_Default_Average",
        config.DEFAULT_LABEL: "Default_Average",
    })

    return result


def default_rate_by_category(
    df: pd.DataFrame,
    column: str,
) -> pd.DataFrame:
    """Calculate application counts and default rates by category."""
    target = config.TARGET_COLUMN

    result = (
        df.groupby(column, dropna=False)
        .agg(
            Applications=(target, "size"),
            Defaults=(target, lambda values: (
                values == config.DEFAULT_LABEL
            ).sum()),
        )
        .reset_index()
    )

    result["Default_Rate_Percent"] = (
        result["Defaults"] / result["Applications"] * 100
    ).round(2)

    return result.sort_values(
        "Default_Rate_Percent",
        ascending=False,
    ).reset_index(drop=True)


def default_rate_by_dti(df: pd.DataFrame) -> pd.DataFrame:
    """Compare default rates across provisional DTI bands."""
    result = df.copy()

    result["DTI_Band"] = pd.cut(
        result["Debt_to_Income"],
        bins=[0, 0.20, 0.35, float("inf")],
        labels=["Low (<20%)", "Medium (20%-<35%)", "High (>=35%)"],
        right=False,
        include_lowest=True,
    )

    return default_rate_by_category(result, "DTI_Band")


def build_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate correlations among numerical variables."""
    numerical_df = df.select_dtypes(include="number").drop(
        columns=[config.IDENTIFIER_COLUMN],
        errors="ignore",
    )

    return numerical_df.corr().reset_index(names="Feature")


def generate_analysis_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build all required EDA tables."""
    tables = {
        "default_summary.csv": summarize_default_status(df),
        "numerical_summary.csv": summarize_numerical_features(df),
        "default_group_comparison.csv": compare_by_default_status(df),
        "default_by_employment.csv": default_rate_by_category(
            df, "Employment_Type"
        ),
        "default_by_loan_purpose.csv": default_rate_by_category(
            df, "Loan_Purpose"
        ),
        "default_by_property.csv": default_rate_by_category(
            df, "Property_Ownership"
        ),
        "default_by_late_payments.csv": default_rate_by_category(
            df, "Late_Payments"
        ),
        "default_by_existing_loans.csv": default_rate_by_category(
            df, "Existing_Loans"
        ),
        "default_by_dti.csv": default_rate_by_dti(df),
        "correlation_matrix.csv": build_correlation_matrix(df),
    }

    return tables


def main() -> None:
    """Run EDA and save its tables."""
    df = load_cleaned_data()

    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)

    tables = generate_analysis_tables(df)

    print("\nEXPLORATORY DATA ANALYSIS")
    print("=" * 45)

    for filename, table in tables.items():
        output_path = config.TABLES_DIR / filename
        table.to_csv(output_path, index=False)

        print(f"Saved: {filename} ({len(table)} rows)")

    print(f"\nApplications analysed: {len(df)}")
    print(
        "Default rate: "
        f"{(df[config.TARGET_COLUMN] == config.DEFAULT_LABEL).mean() * 100:.2f}%"
    )
    print(f"Tables directory: {config.TABLES_DIR}")
    print("EDA table generation: COMPLETED")


if __name__ == "__main__":
    main()
