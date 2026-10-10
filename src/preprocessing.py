
"""Clean and validate the raw credit-risk dataset."""

import pandas as pd

import config


EMPLOYMENT_CATEGORY_MAP = {
    "salaried": "Salaried",
    "Self Employed": "Self-Employed",
}


def load_raw_data() -> pd.DataFrame:
    """Load the raw CSV without modifying the source file."""
    if not config.RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Raw dataset not found: {config.RAW_DATA_PATH}"
        )

    df = pd.read_csv(config.RAW_DATA_PATH)

    if df.columns.tolist() != config.REQUIRED_COLUMNS:
        raise ValueError(
            "The raw dataset does not match the required schema."
        )

    return df


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Clean a copy of the dataset and record the changes."""
    cleaned = df.copy()

    if cleaned.columns.tolist() != config.REQUIRED_COLUMNS:
        raise ValueError(
            "The dataset does not match the required schema."
        )

    if cleaned[config.TARGET_COLUMN].isna().any():
        raise ValueError("Default_Status contains missing values.")

    if not cleaned[config.TARGET_COLUMN].isin(
        [config.DEFAULT_LABEL, config.NON_DEFAULT_LABEL]
    ).all():
        raise ValueError(
            "Default_Status contains values other than 0 and 1."
        )

    summary = {
        "input_rows": len(cleaned),
        "input_columns": len(cleaned.columns),
        "exact_duplicates_removed": int(cleaned.duplicated().sum()),
    }

    # Remove exact duplicate records, retaining the first occurrence.
    cleaned = cleaned.drop_duplicates().copy()

    # Normalize inconsistent employment category labels.
    employment = cleaned["Employment_Type"].astype("string").str.strip()

    summary["salaried_labels_normalized"] = int(
        employment.eq("salaried").sum()
    )
    summary["self_employed_labels_normalized"] = int(
        employment.eq("Self Employed").sum()
    )

    cleaned["Employment_Type"] = employment.replace(
        EMPLOYMENT_CATEGORY_MAP
    )

    # Convert numeric fields safely.
    numeric_columns = [
        "Age",
        "Employment_Years",
        "Annual_Income",
        "Credit_Score",
        "Loan_Amount",
        "Loan_Term_Months",
        "Interest_Rate",
        "Existing_Loans",
        "Debt_to_Income",
        "Late_Payments",
    ]

    for column in numeric_columns:
        cleaned[column] = pd.to_numeric(
            cleaned[column], errors="coerce"
        )

    # Treat negative financial values as missing.
    invalid_income = cleaned["Annual_Income"] < 0
    invalid_loan = cleaned["Loan_Amount"] < 0

    summary["invalid_income_values"] = int(invalid_income.sum())
    summary["invalid_loan_amount_values"] = int(invalid_loan.sum())

    cleaned.loc[invalid_income, "Annual_Income"] = float("nan")
    cleaned.loc[invalid_loan, "Loan_Amount"] = float("nan")

    # Treat out-of-range credit scores as missing.
    credit_scores = cleaned["Credit_Score"]

    invalid_credit_score = ~credit_scores.between(
        config.CREDIT_SCORE_MIN,
        config.CREDIT_SCORE_MAX,
    )

    summary["invalid_credit_scores_including_missing"] = int(
        invalid_credit_score.sum()
    )

    summary["out_of_range_credit_scores"] = int(
        (
            credit_scores.notna()
            & invalid_credit_score
        ).sum()
    )

    cleaned.loc[invalid_credit_score, "Credit_Score"] = float("nan")

    summary["missing_values_before_imputation"] = int(
        cleaned.isna().sum().sum()
    )

    # Impute missing values using medians.
    imputation_columns = [
        "Employment_Years",
        "Annual_Income",
        "Credit_Score",
        "Loan_Amount",
        "Debt_to_Income",
    ]

    summary["imputation_medians"] = {}

    for column in imputation_columns:
        median_value = cleaned[column].median()

        if pd.isna(median_value):
            raise ValueError(
                f"Cannot calculate a valid median for {column}."
            )

        summary["imputation_medians"][column] = float(median_value)

        cleaned[column] = cleaned[column].fillna(median_value)

    summary["missing_values_after_imputation"] = int(
        cleaned.isna().sum().sum()
    )

    # Validate before returning the cleaned data.
    validate_cleaned_data(cleaned)

    summary["output_rows"] = len(cleaned)
    summary["output_columns"] = len(cleaned.columns)

    return cleaned, summary


def validate_cleaned_data(df: pd.DataFrame) -> None:
    """Validate the cleaned dataset and raise errors if checks fail."""
    if df.empty:
        raise ValueError("Cleaned dataset is empty.")

    if df.columns.tolist() != config.REQUIRED_COLUMNS:
        raise ValueError("Cleaned dataset schema has changed.")

    if df.duplicated().any():
        raise ValueError("Exact duplicate rows remain.")

    if df[config.IDENTIFIER_COLUMN].duplicated().any():
        raise ValueError("Duplicate Loan_ID values remain.")

    if df.isna().any().any():
        raise ValueError("Missing values remain.")

    if (df["Annual_Income"] < 0).any():
        raise ValueError("Negative annual income remains.")

    if (df["Loan_Amount"] < 0).any():
        raise ValueError("Negative loan amount remains.")

    if not df["Credit_Score"].between(
        config.CREDIT_SCORE_MIN,
        config.CREDIT_SCORE_MAX,
    ).all():
        raise ValueError("Out-of-range credit scores remain.")

    if not df[config.TARGET_COLUMN].isin(
        [config.DEFAULT_LABEL, config.NON_DEFAULT_LABEL]
    ).all():
        raise ValueError("Invalid target values remain.")

    expected_employment_types = {
        "Salaried",
        "Self-Employed",
        "Business",
        "Unemployed",
    }

    actual_employment_types = set(
        df["Employment_Type"].dropna().unique()
    )

    if actual_employment_types != expected_employment_types:
        raise ValueError(
            "Unexpected employment categories: "
            f"{actual_employment_types}"
        )


def main() -> None:
    """Run preprocessing and save the cleaned dataset."""
    raw_df = load_raw_data()
    cleaned_df, summary = clean_data(raw_df)

    # Create the destination directory if it does not exist.
    config.PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save a separate processed file; the raw file remains unchanged.
    cleaned_df.to_csv(config.CLEANED_DATA_PATH, index=False)

    print("\nDATA PREPROCESSING SUMMARY")
    print("=" * 40)

    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\nValidation: PASSED")
    print(f"Output file: {config.CLEANED_DATA_PATH}")
    print("Raw dataset: unchanged")


if __name__ == "__main__":
    main()
