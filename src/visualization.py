
"""Generate visualizations for the credit risk analysis project."""

import matplotlib

# Support running on systems without a graphical display.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import config


def load_cleaned_data() -> pd.DataFrame:
    """Load the cleaned dataset and validate essential fields."""
    if not config.CLEANED_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {config.CLEANED_DATA_PATH}. "
            "Run python -m src.preprocessing first."
        )

    df = pd.read_csv(config.CLEANED_DATA_PATH)

    if df.empty:
        raise ValueError("Cleaned dataset is empty.")

    if config.TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column {config.TARGET_COLUMN} is missing."
        )

    if not df[config.TARGET_COLUMN].isin(
        [config.DEFAULT_LABEL, config.NON_DEFAULT_LABEL]
    ).all():
        raise ValueError("Default_Status must contain only 0 and 1.")

    return df


def save_chart(filename: str) -> None:
    """Save the current figure and release its memory."""
    output_path = config.CHARTS_DIR / filename
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filename}")


def plot_default_distribution(df: pd.DataFrame) -> None:
    """Plot the number of default and non-default applications."""
    counts = (
        df[config.TARGET_COLUMN]
        .map({
            config.NON_DEFAULT_LABEL: "Non-default",
            config.DEFAULT_LABEL: "Default",
        })
        .value_counts()
        .reindex(["Non-default", "Default"], fill_value=0)
    )

    plt.figure(figsize=(8, 5))
    ax = sns.barplot(
        x=counts.index,
        y=counts.values,
    )
    ax.set_title("Loan Default Status Distribution")
    ax.set_xlabel("Loan Status")
    ax.set_ylabel("Number of Applications")

    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f")

    save_chart("default_distribution.png")


def plot_default_rate_by_category(
    df: pd.DataFrame,
    column: str,
    filename: str,
    title: str,
) -> None:
    """Plot default rates across categories."""
    result = (
        df.groupby(column, dropna=False)[config.TARGET_COLUMN]
        .mean()
        .mul(100)
        .sort_values(ascending=False)
        .reset_index(name="Default_Rate_Percent")
    )

    result[column] = result[column].astype(str)

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=result,
        x=column,
        y="Default_Rate_Percent",
    )

    ax.set_title(title)
    ax.set_xlabel(column.replace("_", " "))
    ax.set_ylabel("Default Rate (%)")
    ax.tick_params(axis="x", rotation=25)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%", padding=3)

    save_chart(filename)


def plot_default_rate_by_dti_band(df: pd.DataFrame) -> None:
    """Plot default rates across provisional DTI bands."""
    plot_df = df.copy()

    plot_df["DTI_Band"] = pd.cut(
        plot_df["Debt_to_Income"],
        bins=[0, 0.20, 0.35, float("inf")],
        labels=[
            "Low (<20%)",
            "Medium (20%-<35%)",
            "High (>=35%)",
        ],
        right=False,
        include_lowest=True,
    )

    result = (
        plot_df.groupby("DTI_Band", observed=False)[config.TARGET_COLUMN]
        .mean()
        .mul(100)
        .reset_index(name="Default_Rate_Percent")
    )

    result["DTI_Band"] = result["DTI_Band"].astype(str)

    plt.figure(figsize=(9, 5))
    ax = sns.barplot(
        data=result,
        x="DTI_Band",
        y="Default_Rate_Percent",
    )

    ax.set_title("Default Rate by Debt-to-Income Band")
    ax.set_xlabel("Debt-to-Income Band")
    ax.set_ylabel("Default Rate (%)")

    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%", padding=3)

    save_chart("default_rate_by_dti_band.png")


def plot_numeric_distribution(
    df: pd.DataFrame,
    column: str,
    filename: str,
    title: str,
) -> None:
    """Plot the distribution of a numerical feature."""
    plt.figure(figsize=(9, 5))
    sns.histplot(
        data=df,
        x=column,
        bins=30,
        kde=True,
    )

    plt.title(title)
    plt.xlabel(column.replace("_", " "))
    plt.ylabel("Number of Applications")

    save_chart(filename)


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """Plot correlations among numerical features."""
    numeric_df = df.select_dtypes(include="number").drop(
        columns=[config.IDENTIFIER_COLUMN],
        errors="ignore",
    )

    correlation = numeric_df.corr()

    plt.figure(figsize=(12, 9))
    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
    )

    plt.title("Correlation Heatmap of Numerical Features")

    save_chart("correlation_heatmap.png")


def main() -> None:
    """Generate all project charts."""
    df = load_cleaned_data()

    config.CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")

    plot_default_distribution(df)

    plot_default_rate_by_category(
        df,
        "Employment_Type",
        "default_rate_by_employment.png",
        "Default Rate by Employment Type",
    )

    plot_default_rate_by_category(
        df,
        "Loan_Purpose",
        "default_rate_by_loan_purpose.png",
        "Default Rate by Loan Purpose",
    )

    plot_default_rate_by_category(
        df,
        "Late_Payments",
        "default_rate_by_late_payments.png",
        "Default Rate by Number of Late Payments",
    )

    plot_default_rate_by_dti_band(df)

    plot_numeric_distribution(
        df,
        "Credit_Score",
        "credit_score_distribution.png",
        "Credit Score Distribution",
    )

    plot_numeric_distribution(
        df,
        "Annual_Income",
        "annual_income_distribution.png",
        "Annual Income Distribution",
    )

    plot_correlation_heatmap(df)

    print(f"\nAll 8 charts generated in: {config.CHARTS_DIR}")


if __name__ == "__main__":
    main()
