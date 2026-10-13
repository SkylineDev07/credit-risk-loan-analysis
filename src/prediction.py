"""Predict loan default risk using a saved trained model."""

import argparse
from pathlib import Path

import joblib
import pandas as pd

import config


MODEL_DIR = config.OUTPUTS_DIR / "models"

MODEL_FILES = {
    "logistic_regression": "logistic_regression.joblib",
    "decision_tree": "decision_tree.joblib",
    "random_forest": "random_forest.joblib",
    "gradient_boosting": "gradient_boosting.joblib",
}

TARGET_COLUMN = config.TARGET_COLUMN
IDENTIFIER_COLUMN = config.IDENTIFIER_COLUMN


def predict_default(input_data: dict, model_name: str = "random_forest") -> dict:
    """Return a default prediction and estimated probability."""
    if model_name not in MODEL_FILES:
        raise ValueError(
            f"Unknown model: {model_name}. Choose from {list(MODEL_FILES)}"
        )

    model_path = MODEL_DIR / MODEL_FILES[model_name]

    if not model_path.is_file():
        raise FileNotFoundError(f"Trained model not found: {model_path}")

    # Load the saved model; do not retrain or modify it.
    model = joblib.load(model_path)

    # Loan_ID is an identifier, not a predictive feature.
    input_df = pd.DataFrame([input_data])
    input_df = input_df.drop(
        columns=[TARGET_COLUMN, IDENTIFIER_COLUMN],
        errors="ignore",
    )

    expected_features = list(model.feature_names_in_)
    missing_features = set(expected_features) - set(input_df.columns)
    extra_features = set(input_df.columns) - set(expected_features)

    if missing_features or extra_features:
        raise ValueError(
            f"Missing features: {sorted(missing_features)}; "
            f"unexpected features: {sorted(extra_features)}"
        )

    input_df = input_df[expected_features]

    probability = float(model.predict_proba(input_df)[0, 1])
    prediction = int(model.predict(input_df)[0])

    return {
        "model": model_name,
        "prediction": prediction,
        "result": "Default risk flagged" if prediction == 1
        else "No default predicted",
        "default_probability": round(probability, 4),
        "default_probability_percent": round(probability * 100, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict credit default risk for a loan applicant."
    )
    parser.add_argument(
        "--model",
        choices=list(MODEL_FILES),
        default="random_forest",
        help="Name of the saved model to use.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to a CSV containing exactly one applicant row.",
    )
    args = parser.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(f"Input CSV not found: {args.input}")

    applicant = pd.read_csv(args.input)

    if len(applicant) != 1:
        raise ValueError("Input CSV must contain exactly one applicant row.")

    result = predict_default(
        applicant.iloc[0].to_dict(),
        model_name=args.model,
    )

    print("\nLOAN DEFAULT PREDICTION")
    print("=" * 35)
    print(f"Model: {result['model']}")
    print(f"Result: {result['result']}")
    print(
        "Estimated default probability: "
        f"{result['default_probability_percent']:.2f}%"
    )
    print(
        "Predicted class: "
        f"{result['prediction']} "
        "(1 = default, 0 = non-default)"
    )


if __name__ == "__main__":
    main()





