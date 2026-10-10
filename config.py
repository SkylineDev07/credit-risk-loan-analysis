
"""Central configuration for the credit risk analysis project."""

from pathlib import Path


# Project directories
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Dataset paths
RAW_DATA_PATH = RAW_DATA_DIR / "loan_default.csv"
CLEANED_DATA_PATH = PROCESSED_DATA_DIR / "loan_default_cleaned.csv"

# Output directories
TABLES_DIR = OUTPUTS_DIR / "tables"
CHARTS_DIR = OUTPUTS_DIR / "charts"
REPORTS_DIR = OUTPUTS_DIR / "reports"

# Dataset configuration
TARGET_COLUMN = "Default_Status"
IDENTIFIER_COLUMN = "Loan_ID"

DEFAULT_LABEL = 1
NON_DEFAULT_LABEL = 0

CREDIT_SCORE_MIN = 300
CREDIT_SCORE_MAX = 850

# Required dataset schema
REQUIRED_COLUMNS = [
    "Loan_ID",
    "Age",
    "Gender",
    "Employment_Type",
    "Employment_Years",
    "Annual_Income",
    "Credit_Score",
    "Loan_Amount",
    "Loan_Term_Months",
    "Interest_Rate",
    "Existing_Loans",
    "Debt_to_Income",
    "Late_Payments",
    "Property_Ownership",
    "Education",
    "Loan_Purpose",
    "Default_Status",
]
