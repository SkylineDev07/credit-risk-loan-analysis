"""Train and compare additional credit default prediction models."""

import json

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

import config
from src.model_training import (
    RANDOM_STATE,
    TEST_SIZE,
    build_pipeline,
    load_training_data,
)


MODEL_DIR = config.OUTPUTS_DIR / "models"
METRICS_DIR = config.OUTPUTS_DIR / "metrics"

# Separate filenames preserve the existing Logistic Regression artifacts.
CANDIDATE_MODELS = {
    "decision_tree": DecisionTreeClassifier(
        max_depth=8,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
    "gradient_boosting": GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=2,
        random_state=RANDOM_STATE,
    ),
}


def evaluate_model(model, X_test, y_test) -> dict:
    """Evaluate a fitted model on the same held-out test set."""
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision": round(
            float(precision_score(y_test, predictions, zero_division=0)),
            4,
        ),
        "recall": round(
            float(recall_score(y_test, predictions, zero_division=0)),
            4,
        ),
        "f1_score": round(
            float(f1_score(y_test, predictions, zero_division=0)),
            4,
        ),
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
        "confusion_matrix": confusion_matrix(
            y_test, predictions, labels=[0, 1]
        ).tolist(),
    }


def main() -> None:
    """Train candidate models and save new artifacts without overwriting."""
    df = load_training_data()

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

    # Fit preprocessing using training data only.
    preprocessing_template = build_pipeline(X_train).named_steps[
        "preprocessor"
    ]

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    comparison = []

    print("CREDIT DEFAULT MODEL COMPARISON")
    print("=" * 45)
    print(f"Training rows: {len(X_train)}")
    print(f"Test rows: {len(X_test)}")

    for name, classifier in CANDIDATE_MODELS.items():
        model_path = MODEL_DIR / f"{name}.joblib"
        metrics_path = METRICS_DIR / f"{name}_metrics.json"

        if model_path.exists() or metrics_path.exists():
            print(
                f"\nSKIPPED {name}: an output already exists. "
                "Existing files were preserved."
            )
            continue

        # Each model receives its own fitted preprocessing pipeline.
        from sklearn.pipeline import Pipeline

        model = Pipeline([
            ("preprocessor", clone(preprocessing_template)),
            ("classifier", clone(classifier)),
        ])

        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)

        record = {
            "model": name,
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
            "training_rows": len(X_train),
            "test_rows": len(X_test),
            "features_used": X.shape[1],
            **metrics,
        }

        # Open metrics in exclusive mode to prevent accidental overwrites.
        with metrics_path.open("x", encoding="utf-8") as file:
            try:
                joblib.dump(model, model_path)
                json.dump(record, file, indent=4)
            except Exception:
                model_path.unlink(missing_ok=True)
                metrics_path.unlink(missing_ok=True)
                raise

        comparison.append(record)

        print(f"\n{name.replace('_', ' ').title()}")
        for metric in ("accuracy", "precision", "recall", "f1_score", "roc_auc"):
            print(f"{metric.replace('_', ' ').title()}: {metrics[metric]:.4f}")
        print(f"Saved model: {model_path}")
        print(f"Saved metrics: {metrics_path}")

    if comparison:
        comparison_path = METRICS_DIR / "model_comparison.csv"

        if comparison_path.exists():
            print(
                "\nComparison CSV already exists; it was not overwritten."
            )
        else:
            pd.DataFrame(comparison).to_csv(
                comparison_path, index=False
            )
            print(f"\nSaved comparison table: {comparison_path}")

    print("\nModel comparison run completed.")


if __name__ == "__main__":
    main()
