"""Read-only data quality audit for the raw loan dataset."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "loan_default.csv"

REQUIRED_COLUMNS = [
    "Loan_ID", "Age", "Gender", "Employment_Type",
    "Employment_Years", "Annual_Income", "Credit_Score",
    "Loan_Amount", "Loan_Term_Months", "Interest_Rate",
    "Existing_Loans", "Debt_to_Income", "Late_Payments",
    "Property_Ownership", "Education", "Loan_Purpose",
    "Default_Status",
]

NUMERIC_COLUMNS = [
    "Age", "Employment_Years", "Annual_Income", "Credit_Score",
    "Loan_Amount", "Loan_Term_Months", "Interest_Rate",
    "Existing_Loans", "Debt_to_Income", "Late_Payments",
    "Default_Status",
]

CATEGORICAL_COLUMNS = [
    "Gender", "Employment_Type", "Property_Ownership",
    "Education", "Loan_Purpose",
]


def load_raw_data() -> pd.DataFrame:
    """Load the raw CSV without modifying it."""
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Raw dataset not found: {RAW_DATA_PATH}")
    return pd.read_csv(RAW_DATA_PATH)


def audit_structure(df: pd.DataFrame) -> None:
    """Report dimensions and schema."""
    print("\n=== STRUCTURE AUDIT ===")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Schema matches expected columns: {df.columns.tolist() == REQUIRED_COLUMNS}")
    print("Missing columns:", [c for c in REQUIRED_COLUMNS if c not in df.columns])
    print("Unexpected columns:", [c for c in df.columns if c not in REQUIRED_COLUMNS])
    print("\nData types:")
    print(df.dtypes.to_string())


def audit_missing_values(df: pd.DataFrame) -> None:
    """Report missing values."""
    print("\n=== MISSING VALUES ===")
    missing = df.isna().sum()
    print(missing[missing > 0].to_string() if missing.any() else "None found.")


def audit_duplicates(df: pd.DataFrame) -> None:
    """Report duplicate rows and identifiers."""
    print("\n=== DUPLICATE AUDIT ===")
    print("Exact duplicate rows:", int(df.duplicated().sum()))
    print("Duplicate Loan_ID values:", int(df["Loan_ID"].duplicated().sum()))


def audit_numeric_ranges(df: pd.DataFrame) -> None:
    """Report numerical ranges and suspicious values."""
    print("\n=== NUMERICAL RANGE AUDIT ===")

    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            values = df[column]
            print(
                f"{column}: min={values.min()}, max={values.max()}, "
                f"negative values={int((values < 0).sum())}"
            )

    scores = df["Credit_Score"]
    print(
        "Credit scores outside 300-850:",
        int((~scores.between(300, 850)).sum()),
    )
    print(
        "Invalid target values:",
        int((~df["Default_Status"].isin([0, 1])).sum()),
    )


def audit_categories(df: pd.DataFrame) -> None:
    """Display category values for review."""
    print("\n=== CATEGORICAL AUDIT ===")

    for column in CATEGORICAL_COLUMNS:
        print(f"\n{column}:")
        print(df[column].value_counts(dropna=False).to_string())


def main() -> None:
    """Run read-only audit checks."""
    df = load_raw_data()

    print("RAW DATA QUALITY AUDIT")
    print(f"Source: {RAW_DATA_PATH}")

    audit_structure(df)
    audit_missing_values(df)
    audit_duplicates(df)
    audit_numeric_ranges(df)
    audit_categories(df)

    print("\nAudit complete. The raw dataset was not modified.")


if __name__ == "__main__":
    main()