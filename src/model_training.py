"""Train and evaluate a baseline credit default prediction model."""

import json

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config


RANDOM_STATE = 42
TEST_SIZE = 0.20

MODEL_DIR = config.OUTPUTS_DIR / "models"
METRICS_DIR = config.OUTPUTS_DIR / "metrics"


def load_training_data() -> pd.DataFrame:
    """Load the existing cleaned dataset without modifying it."""
    if not config.CLEANED_DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {config.CLEANED_DATA_PATH}"
        )

    df = pd.read_csv(config.CLEANED_DATA_PATH)

    if df.empty:
        raise ValueError("The cleaned dataset is empty.")

    if config.TARGET_COLUMN not in df.columns:
        raise ValueError("The target column is missing.")

    if df[config.TARGET_COLUMN].isna().any():
        raise ValueError("The target column contains missing values.")

    if set(df[config.TARGET_COLUMN].unique()) != {
        config.NON_DEFAULT_LABEL,
        config.DEFAULT_LABEL,
    }:
        raise ValueError("The target must contain both 0 and 1.")

    if config.IDENTIFIER_COLUMN not in df.columns:
        raise ValueError("The identifier column is missing.")

    if df[config.IDENTIFIER_COLUMN].duplicated().any():
        raise ValueError("Duplicate loan identifiers were detected.")

    return df


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    """Build a preprocessing and Logistic Regression pipeline."""
    numeric_columns = X.select_dtypes(include="number").columns.tolist()
    categorical_columns = X.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_columns),
        ("categorical", categorical_pipeline, categorical_columns),
    ])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )),
    ])


def train_and_evaluate(df: pd.DataFrame) -> tuple[Pipeline, dict]:
    """Split data, train the model, and evaluate held-out predictions."""
    X = df.drop(
        columns=[config.TARGET_COLUMN, config.IDENTIFIER_COLUMN]
    )
    y = df[config.TARGET_COLUMN].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = build_pipeline(X_train)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": "Logistic Regression",
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "training_rows": len(X_train),
        "test_rows": len(X_test),
        "features_used": X.shape[1],
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision": round(
            float(precision_score(y_test, predictions, zero_division=0)), 4
        ),
        "recall": round(
            float(recall_score(y_test, predictions, zero_division=0)), 4
        ),
        "f1_score": round(float(f1_score(y_test, predictions)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
        "confusion_matrix": confusion_matrix(
            y_test, predictions, labels=[0, 1]
        ).tolist(),
        "classification_report": classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Non-default", "Default"],
            output_dict=True,
            zero_division=0,
        ),
    }

    return model, metrics


def save_artifacts(model: Pipeline, metrics: dict) -> None:
    """Save outputs only if their target paths do not already exist."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / "logistic_regression.joblib"
    metrics_path = METRICS_DIR / "baseline_metrics.json"

    if model_path.exists() or metrics_path.exists():
        raise FileExistsError(
            "Model or metrics output already exists. "
            "Nothing was saved. Choose new output filenames or review "
            "the existing artifacts before proceeding."
        )

    joblib.dump(model, model_path)

    try:
        with metrics_path.open("x", encoding="utf-8") as file:
            json.dump(metrics, file, indent=4)
    except Exception:
        # Avoid leaving a partially completed pair of artifacts.
        model_path.unlink(missing_ok=True)
        raise

    print(f"Saved model: {model_path}")
    print(f"Saved metrics: {metrics_path}")


def main() -> None:
    """Train the baseline model and save new artifacts safely."""
    df = load_training_data()
    print(f"Loaded cleaned dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    model, metrics = train_and_evaluate(df)

    print("\nBASELINE MODEL EVALUATION")
    print("=" * 40)
    for name in (
        "accuracy", "precision", "recall", "f1_score", "roc_auc"
    ):
        print(f"{name.replace('_', ' ').title()}: {metrics[name]:.4f}")

    print("\nConfusion matrix (actual rows, predicted columns):")
    print(pd.DataFrame(
        metrics["confusion_matrix"],
        index=["Actual non-default", "Actual default"],
        columns=["Predicted non-default", "Predicted default"],
    ))

    save_artifacts(model, metrics)
    print("\nBaseline training completed.")


if __name__ == "__main__":
    main()

