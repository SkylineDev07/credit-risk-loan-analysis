"""Main entry point for Credit Risk and Loan Default Analysis."""

import config
from src import data_quality


EXPECTED_TABLES = [
    "correlation_matrix.csv",
    "default_by_dti.csv",
    "default_by_employment.csv",
    "default_by_existing_loans.csv",
    "default_by_late_payments.csv",
    "default_by_loan_purpose.csv",
    "default_by_property.csv",
    "default_group_comparison.csv",
    "default_summary.csv",
    "numerical_summary.csv",
]

EXPECTED_CHARTS = [
    "annual_income_distribution.png",
    "correlation_heatmap.png",
    "credit_score_distribution.png",
    "default_distribution.png",
    "default_rate_by_dti_band.png",
    "default_rate_by_employment.png",
    "default_rate_by_late_payments.png",
    "default_rate_by_loan_purpose.png",
]


def report_existing_outputs(directory, expected_files):
    """Report existing outputs without modifying them."""
    existing = []
    missing = []

    for filename in expected_files:
        if (directory / filename).is_file():
            existing.append(filename)
        else:
            missing.append(filename)

    print(f"\nDirectory: {directory}")
    print(f"Existing files: {len(existing)}/{len(expected_files)}")

    for filename in existing:
        print(f"  [OK] {filename}")

    for filename in missing:
        print(f"  [MISSING] {filename}")

    return not missing


def main():
    """Audit the raw data and verify existing analysis outputs."""
    print("=" * 60)
    print("CREDIT RISK AND LOAN DEFAULT ANALYSIS")
    print("=" * 60)

    if not config.CLEANED_DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {config.CLEANED_DATA_PATH}"
        )

    print("\n[1/3] Running read-only raw-data quality audit...")
    data_quality.main()

    print("\n[2/3] Checking existing EDA tables...")
    tables_ok = report_existing_outputs(
        config.TABLES_DIR, EXPECTED_TABLES
    )

    print("\n[3/3] Checking existing charts...")
    charts_ok = report_existing_outputs(
        config.CHARTS_DIR, EXPECTED_CHARTS
    )

    print("\n" + "=" * 60)

    if tables_ok and charts_ok:
        print("ALL EXPECTED OUTPUT FILES ARE PRESENT.")
    else:
        print("SOME EXPECTED OUTPUT FILES ARE MISSING.")

    print("Existing tables and charts were not regenerated.")
    print(f"Cleaned dataset: {config.CLEANED_DATA_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()





