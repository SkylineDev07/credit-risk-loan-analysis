"""Generate visual reports for saved credit default model metrics."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import config


METRICS_DIR = config.OUTPUTS_DIR / "metrics"
CHARTS_DIR = config.OUTPUTS_DIR / "charts"

MODEL_FILES = {
    "Logistic Regression": "baseline_metrics.json",
    "Decision Tree": "decision_tree_metrics.json",
    "Random Forest": "random_forest_metrics.json",
    "Gradient Boosting": "gradient_boosting_metrics.json",
}

METRICS = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]


def load_metrics() -> dict:
    """Load the four existing model evaluation files."""
    results = {}

    for model_name, filename in MODEL_FILES.items():
        path = METRICS_DIR / filename

        if not path.is_file():
            raise FileNotFoundError(f"Missing metrics file: {path}")

        with path.open("r", encoding="utf-8") as file:
            results[model_name] = json.load(file)

    return results


def plot_metric_comparison(results: dict) -> None:
    """Save a grouped bar chart comparing evaluation metrics."""
    names = list(results)
    x = np.arange(len(METRICS))
    width = 0.19

    fig, ax = plt.subplots(figsize=(13, 7))

    for index, name in enumerate(names):
        values = [results[name][metric] for metric in METRICS]
        positions = x + (index - 1.5) * width
        ax.bar(positions, values, width, label=name)

    ax.set_title("Credit Default Model Performance Comparison")
    ax.set_ylabel("Score")
    ax.set_xlabel("Evaluation Metric")
    ax.set_xticks(x)
    ax.set_xticklabels(
        ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    )
    ax.set_ylim(0, 1)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    output = CHARTS_DIR / "model_metrics_comparison.png"
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output}")


def plot_confusion_matrices(results: dict) -> None:
    """Save four confusion-matrix heatmaps."""
    names = list(results)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for ax, name in zip(axes.flat, names):
        matrix = np.asarray(results[name]["confusion_matrix"])

        if matrix.shape != (2, 2):
            raise ValueError(f"Invalid confusion matrix for {name}")

        image = ax.imshow(matrix, cmap="Blues")
        ax.set_title(name)
        ax.set_xlabel("Predicted Class")
        ax.set_ylabel("Actual Class")
        ax.set_xticks([0, 1], labels=["Non-default", "Default"])
        ax.set_yticks([0, 1], labels=["Non-default", "Default"])

        for row in range(2):
            for col in range(2):
                ax.text(
                    col, row, f"{matrix[row, col]:,}",
                    ha="center", va="center",
                    color="white" if matrix[row, col] > matrix.max() / 2
                    else "black",
                    fontsize=12,
                )

        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Confusion Matrices — Credit Default Models", fontsize=15)
    fig.tight_layout()

    output = CHARTS_DIR / "confusion_matrices.png"
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output}")


def main() -> None:
    """Generate charts from existing evaluation metrics."""
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    results = load_metrics()

    plot_metric_comparison(results)
    plot_confusion_matrices(results)

    print("\nVisual model report completed.")


if __name__ == "__main__":
    main()




