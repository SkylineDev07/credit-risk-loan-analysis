"""
Script: generate_dataset.py

Title:
    Credit Risk & Loan Default Analysis - Raw Dataset Generator

Description:
    Generates a reproducible synthetic raw loan dataset containing
    6,000 base borrower/loan records.

    The dataset includes:
    - Logical relationships between borrower and loan variables
    - Probabilistic Default_Status generation
    - Controlled data-quality issues for preprocessing practice
    - CSV and Excel output

    NOTE:
    The dataset is synthetic and intended for the project's
    industry-oriented internship analysis workflow.
"""

import os
import numpy as np
import pandas as pd


# =============================================================================
# 1. CONFIGURATION
# =============================================================================

RANDOM_SEED = 42
BASE_RECORDS = 6000

OUTPUT_DIR = "data/raw"

CSV_PATH = os.path.join(
    OUTPUT_DIR,
    "loan_default.csv"
)

EXCEL_PATH = os.path.join(
    OUTPUT_DIR,
    "loan_default.xlsx"
)

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
    "Default_Status"
]


# =============================================================================
# 2. FEATURE GENERATION
# =============================================================================

def generate_base_features(n_samples: int) -> dict:
    """
    Generate the base borrower and loan features.

    The generated variables contain logical relationships rather than
    being completely independent random variables.
    """

    # -------------------------------------------------------------------------
    # Age
    # -------------------------------------------------------------------------
    age = np.random.randint(
        21,
        66,
        size=n_samples
    )

    # -------------------------------------------------------------------------
    # Employment Years
    # -------------------------------------------------------------------------
    # Assumption:
    # Employment experience cannot exceed the number of years since age 21.
    max_experience = np.maximum(
        0,
        age - 21
    )

    employment_years = np.array([
        np.random.randint(0, max_exp + 1)
        for max_exp in max_experience
    ])

    # -------------------------------------------------------------------------
    # Gender
    # -------------------------------------------------------------------------
    gender = np.random.choice(
        [
            "Male",
            "Female",
            "Other"
        ],
        size=n_samples,
        p=[
            0.51,
            0.47,
            0.02
        ]
    )

    # -------------------------------------------------------------------------
    # Education
    # -------------------------------------------------------------------------
    education = np.random.choice(
        [
            "High School",
            "Graduate",
            "Post Graduate",
            "Doctorate"
        ],
        size=n_samples,
        p=[
            0.30,
            0.50,
            0.16,
            0.04
        ]
    )

    # -------------------------------------------------------------------------
    # Employment Type
    # -------------------------------------------------------------------------
    employment_type = np.random.choice(
        [
            "Salaried",
            "Self-Employed",
            "Business",
            "Unemployed"
        ],
        size=n_samples,
        p=[
            0.55,
            0.25,
            0.15,
            0.05
        ]
    )

    # -------------------------------------------------------------------------
    # Annual Income
    # -------------------------------------------------------------------------
    # Log-normal distribution produces a realistic right-skewed income
    # distribution with a meaningful upper tail.
    base_income = np.random.lognormal(
        mean=11.0,
        sigma=0.65,
        size=n_samples
    )

    education_multiplier = {
        "High School": 1.00,
        "Graduate": 1.25,
        "Post Graduate": 1.55,
        "Doctorate": 1.90
    }

    employment_multiplier = {
        "Salaried": 1.00,
        "Self-Employed": 1.10,
        "Business": 1.25,
        "Unemployed": 0.35
    }

    edu_multiplier = np.array([
        education_multiplier[value]
        for value in education
    ])

    emp_multiplier = np.array([
        employment_multiplier[value]
        for value in employment_type
    ])

    experience_boost = (
        1 + (employment_years * 0.015)
    )

    annual_income = (
        base_income
        * edu_multiplier
        * emp_multiplier
        * experience_boost
    )

    annual_income = np.round(
        np.clip(
            annual_income,
            15000,
            1500000
        ),
        -2
    )

    # -------------------------------------------------------------------------
    # Credit Score
    # -------------------------------------------------------------------------
    credit_score = np.random.normal(
        loc=680,
        scale=75,
        size=n_samples
    )

    credit_score = np.clip(
        credit_score,
        300,
        850
    ).astype(int)

    # -------------------------------------------------------------------------
    # Interest Rate
    # -------------------------------------------------------------------------
    # Higher credit score -> generally lower interest rate.
    base_interest = (
        24.0
        - ((credit_score - 300) / 550.0) * 18.0
    )

    interest_rate = (
        base_interest
        + np.random.normal(
            0,
            1.2,
            size=n_samples
        )
    )

    interest_rate = np.round(
        np.clip(
            interest_rate,
            5.0,
            29.99
        ),
        2
    )

    # -------------------------------------------------------------------------
    # Loan Amount
    # -------------------------------------------------------------------------
    # Loan amount is related to annual income but retains random variation.
    loan_multiplier = np.random.uniform(
        0.30,
        3.50,
        size=n_samples
    )

    loan_amount = (
        annual_income
        * loan_multiplier
    )

    loan_amount = np.round(
        np.clip(
            loan_amount,
            25000,
            5000000
        ),
        -2
    )

    # -------------------------------------------------------------------------
    # Loan Term
    # -------------------------------------------------------------------------
    loan_term_months = np.random.choice(
        [
            12,
            24,
            36,
            48,
            60,
            120,
            180,
            240,
            360
        ],
        size=n_samples,
        p=[
            0.08,
            0.10,
            0.25,
            0.10,
            0.25,
            0.08,
            0.05,
            0.04,
            0.05
        ]
    )

    # -------------------------------------------------------------------------
    # Existing Loans
    # -------------------------------------------------------------------------
    existing_loans = np.random.poisson(
        lam=1.5,
        size=n_samples
    )

    existing_loans = np.clip(
        existing_loans,
        0,
        8
    )

    # -------------------------------------------------------------------------
    # Late Payments
    # -------------------------------------------------------------------------
    # Lower credit score -> generally more late payments.
    risk_factor = (
        850 - credit_score
    ) / 550.0

    late_payments = np.random.poisson(
        lam=risk_factor * 2.5,
        size=n_samples
    )

    late_payments = np.clip(
        late_payments,
        0,
        12
    )

    # -------------------------------------------------------------------------
    # Debt-to-Income Ratio
    # -------------------------------------------------------------------------
    dti_base = (
        np.random.beta(
            a=2,
            b=5,
            size=n_samples
        )
        * 0.65
    )

    debt_to_income = np.round(
        np.clip(
            dti_base
            + (late_payments * 0.02),
            0.05,
            0.75
        ),
        3
    )

    # -------------------------------------------------------------------------
    # Property Ownership
    # -------------------------------------------------------------------------
    property_ownership = np.random.choice(
        [
            "Rent",
            "Mortgage",
            "Own"
        ],
        size=n_samples,
        p=[
            0.45,
            0.40,
            0.15
        ]
    )

    # -------------------------------------------------------------------------
    # Loan Purpose
    # -------------------------------------------------------------------------
    loan_purpose = np.random.choice(
        [
            "Debt Consolidation",
            "Home Improvement",
            "Auto",
            "Business",
            "Personal",
            "Education"
        ],
        size=n_samples,
        p=[
            0.30,
            0.18,
            0.17,
            0.12,
            0.13,
            0.10
        ]
    )

    return {
        "Age": age,
        "Gender": gender,
        "Employment_Type": employment_type,
        "Employment_Years": employment_years,
        "Annual_Income": annual_income,
        "Credit_Score": credit_score,
        "Loan_Amount": loan_amount,
        "Loan_Term_Months": loan_term_months,
        "Interest_Rate": interest_rate,
        "Existing_Loans": existing_loans,
        "Debt_to_Income": debt_to_income,
        "Late_Payments": late_payments,
        "Property_Ownership": property_ownership,
        "Education": education,
        "Loan_Purpose": loan_purpose
    }


# =============================================================================
# 3. DEFAULT STATUS GENERATION
# =============================================================================

def generate_default_status(data: dict) -> np.ndarray:
    """
    Generate Default_Status probabilistically using a logistic risk function.

    0 = Non-default
    1 = Default
    """

    std_credit = (
        data["Credit_Score"] - 680
    ) / 75.0

    std_dti = (
        data["Debt_to_Income"] - 0.30
    ) / 0.15

    std_interest = (
        data["Interest_Rate"] - 13.0
    ) / 5.0

    std_late = data["Late_Payments"]

    std_experience = (
        data["Employment_Years"] - 6.0
    ) / 5.0

    # Baseline risk model.
    # Coefficients create relationships between risk indicators
    # and the probability of default.
    logit = (
        -1.80
        - 0.85 * std_credit
        + 0.55 * std_dti
        + 0.45 * std_interest
        + 0.60 * std_late
        + 0.25 * (data["Existing_Loans"] > 3)
        - 0.35 * std_experience
    )

    probabilities = (
        1.0
        / (1.0 + np.exp(-logit))
    )

    default_status = np.random.binomial(
        n=1,
        p=probabilities
    )

    return default_status


# =============================================================================
# 4. CONTROLLED DATA-QUALITY ISSUES
# =============================================================================

def inject_data_quality_issues(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Introduce a small, controlled number of data-quality issues.

    The resulting dataset represents RAW data.
    Cleaning must be performed later by the preprocessing pipeline.
    """

    df_flawed = df.copy()
    n = len(df_flawed)

    # -------------------------------------------------------------------------
    # A. Missing Values
    # -------------------------------------------------------------------------
    # Approximately 1% missing values in selected columns.
    for column in [
        "Employment_Years",
        "Credit_Score",
        "Debt_to_Income"
    ]:
        mask = np.random.rand(n) < 0.01
        df_flawed.loc[mask, column] = np.nan

    # -------------------------------------------------------------------------
    # B. Invalid Numerical Values
    # -------------------------------------------------------------------------
    # Negative Annual Income
    negative_income_indices = np.random.choice(
        n,
        size=5,
        replace=False
    )

    df_flawed.loc[
        negative_income_indices,
        "Annual_Income"
    ] *= -1

    # Negative Loan Amount
    negative_loan_indices = np.random.choice(
        n,
        size=5,
        replace=False
    )

    df_flawed.loc[
        negative_loan_indices,
        "Loan_Amount"
    ] *= -1

    # -------------------------------------------------------------------------
    # C. Out-of-Range Credit Scores
    # -------------------------------------------------------------------------
    invalid_credit_indices = np.random.choice(
        n,
        size=5,
        replace=False
    )

    df_flawed.loc[
        invalid_credit_indices,
        "Credit_Score"
    ] = np.random.randint(
        880,
        951,
        size=5
    )

    # -------------------------------------------------------------------------
    # D. Categorical Inconsistencies
    # -------------------------------------------------------------------------
    salaried_indices = (
        df_flawed[
            df_flawed["Employment_Type"] == "Salaried"
        ]
        .sample(
            n=15,
            random_state=RANDOM_SEED
        )
        .index
    )

    df_flawed.loc[
        salaried_indices,
        "Employment_Type"
    ] = "salaried"

    self_employed_indices = (
        df_flawed[
            df_flawed["Employment_Type"] == "Self-Employed"
        ]
        .sample(
            n=15,
            random_state=RANDOM_SEED
        )
        .index
    )

    df_flawed.loc[
        self_employed_indices,
        "Employment_Type"
    ] = "Self Employed"

    # -------------------------------------------------------------------------
    # E. Duplicate Records
    # -------------------------------------------------------------------------
    # More than 20 duplicate rows are intentionally added.
    duplicate_indices = np.random.choice(
        n,
        size=30,
        replace=False
    )

    duplicates = df_flawed.iloc[
        duplicate_indices
    ].copy()

    df_flawed = pd.concat(
        [
            df_flawed,
            duplicates
        ],
        ignore_index=True
    )

    return df_flawed


# =============================================================================
# 5. VALIDATION & SUMMARY
# =============================================================================

def print_summary_and_validation(
    df_raw: pd.DataFrame,
    base_count: int,
    seed: int
) -> None:
    """
    Print dataset-generation summary and validation information.
    """

    print("\n" + "=" * 70)
    print("       CREDIT RISK DATASET GENERATION SUMMARY")
    print("=" * 70)

    print(f"Random Seed              : {seed}")
    print(f"Base Record Count        : {base_count}")
    print(
        f"Final Raw Record Count   : {len(df_raw)}"
    )
    print(
        f"Total Columns            : {len(df_raw.columns)}"
    )

    print(
        f"Schema Compliance        : "
        f"{list(df_raw.columns) == REQUIRED_COLUMNS}"
    )

    # -------------------------------------------------------------------------
    # Target Validation
    # -------------------------------------------------------------------------
    print("\n--- Target Variable ---")

    unique_target_values = set(
        df_raw["Default_Status"]
        .dropna()
        .unique()
    )

    target_valid = unique_target_values.issubset(
        {0, 1}
    )

    print(
        f"Valid Target Values      : {target_valid}"
    )

    default_counts = (
        df_raw["Default_Status"]
        .value_counts()
        .to_dict()
    )

    default_rate = (
        df_raw["Default_Status"].mean()
        * 100
    )

    print(
        f"Non-Default (0)         : "
        f"{default_counts.get(0, 0)}"
    )

    print(
        f"Default (1)             : "
        f"{default_counts.get(1, 0)}"
    )

    print(
        f"Overall Default Rate    : "
        f"{default_rate:.2f}%"
    )

    # -------------------------------------------------------------------------
    # Missing Values
    # -------------------------------------------------------------------------
    print("\n--- Missing Values ---")

    missing_values = df_raw.isnull().sum()

    if missing_values.sum() > 0:
        print(
            missing_values[
                missing_values > 0
            ].to_string()
        )
    else:
        print("No missing values.")

    # -------------------------------------------------------------------------
    # Duplicate Rows
    # -------------------------------------------------------------------------
    print("\n--- Duplicate Rows ---")

    duplicate_count = df_raw.duplicated().sum()

    print(
        f"Exact Duplicate Rows    : "
        f"{duplicate_count}"
    )

    # -------------------------------------------------------------------------
    # Numerical Ranges
    # -------------------------------------------------------------------------
    print("\n--- Numerical Range Summary ---")

    numerical_columns = [
        "Age",
        "Employment_Years",
        "Annual_Income",
        "Credit_Score",
        "Loan_Amount",
        "Loan_Term_Months",
        "Interest_Rate",
        "Existing_Loans",
        "Debt_to_Income",
        "Late_Payments"
    ]

    summary = (
        df_raw[numerical_columns]
        .agg(
            [
                "min",
                "max",
                "mean"
            ]
        )
        .T
    )

    print(summary.to_string())

    # -------------------------------------------------------------------------
    # Relationship Validation
    # -------------------------------------------------------------------------
    print("\n--- Intended Relationships ---")

    for column in [
        "Credit_Score",
        "Late_Payments",
        "Debt_to_Income",
        "Interest_Rate",
        "Employment_Years"
    ]:

        correlation = (
            df_raw[column]
            .corr(
                df_raw["Default_Status"]
            )
        )

        print(
            f"{column:20s} vs Default_Status : "
            f"{correlation:+.3f}"
        )

    income_loan_correlation = (
        df_raw["Annual_Income"]
        .corr(
            df_raw["Loan_Amount"]
        )
    )

    print(
        f"{'Annual_Income':20s} vs Loan_Amount    : "
        f"{income_loan_correlation:+.3f}"
    )

    print("=" * 70)
    print()


# =============================================================================
# 6. MAIN EXECUTION PIPELINE
# =============================================================================

def main() -> None:
    """
    Execute the complete raw dataset generation pipeline.
    """

    # Set reproducible random seed.
    np.random.seed(RANDOM_SEED)

    # -------------------------------------------------------------------------
    # Step 1: Generate base features
    # -------------------------------------------------------------------------
    feature_data = generate_base_features(
        BASE_RECORDS
    )

    # -------------------------------------------------------------------------
    # Step 2: Generate probabilistic target
    # -------------------------------------------------------------------------
    feature_data["Default_Status"] = (
        generate_default_status(
            feature_data
        )
    )

    # -------------------------------------------------------------------------
    # Step 3: Create DataFrame
    # -------------------------------------------------------------------------
    df = pd.DataFrame(
        feature_data
    )

    # -------------------------------------------------------------------------
    # Step 4: Generate Loan IDs
    # -------------------------------------------------------------------------
    loan_ids = [
        f"LNK-{100001 + i}"
        for i in range(BASE_RECORDS)
    ]

    df.insert(
        0,
        "Loan_ID",
        loan_ids
    )

    # -------------------------------------------------------------------------
    # Step 5: Verify base schema
    # -------------------------------------------------------------------------
    if list(df.columns) != REQUIRED_COLUMNS:
        raise ValueError(
            "Base dataset schema does not match "
            "the required 17-column schema."
        )

    # -------------------------------------------------------------------------
    # Step 6: Inject controlled data-quality issues
    # -------------------------------------------------------------------------
    df_raw = inject_data_quality_issues(
        df
    )

    # -------------------------------------------------------------------------
    # Step 7: Create output directory
    # -------------------------------------------------------------------------
    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # -------------------------------------------------------------------------
    # Step 8: Save CSV
    # -------------------------------------------------------------------------
    df_raw.to_csv(
        CSV_PATH,
        index=False
    )

    print(
        f"[SUCCESS] CSV saved to: "
        f"{CSV_PATH}"
    )

    # -------------------------------------------------------------------------
    # Step 9: Save Excel
    # -------------------------------------------------------------------------
    try:

        df_raw.to_excel(
            EXCEL_PATH,
            index=False,
            engine="openpyxl"
        )

        print(
            f"[SUCCESS] Excel saved to: "
            f"{EXCEL_PATH}"
        )

    except ImportError:

        print(
            "[WARNING] openpyxl is not installed. "
            "Install it using: pip install openpyxl"
        )

    # -------------------------------------------------------------------------
    # Step 10: Final validation summary
    # -------------------------------------------------------------------------
    print_summary_and_validation(
        df_raw,
        BASE_RECORDS,
        RANDOM_SEED
    )


# =============================================================================
# 7. SCRIPT ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()