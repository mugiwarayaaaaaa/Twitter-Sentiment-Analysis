"""
Evaluate the trained sentiment model on the held-out test split and save:
    - reports/confusion_matrix.png
    - reports/classification_report.txt

Usage:
    python src/evaluate.py
"""

import os
import sys

import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

MODELS_DIR = os.path.join(ROOT, "models")
REPORTS_DIR = os.path.join(ROOT, "reports")
TEST_SPLIT_PATH = os.path.join(ROOT, "data", "processed", "test_split.csv")


def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)

    print("Loading model + vectorizer...")
    vectorizer = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib"))
    model = joblib.load(os.path.join(MODELS_DIR, "sentiment_model.joblib"))

    print("Loading held-out test split...")
    test_df = pd.read_csv(TEST_SPLIT_PATH).dropna(subset=["clean_text"])
    X_test_vec = vectorizer.transform(test_df["clean_text"])
    y_test = test_df["sentiment"]

    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"\nAccuracy: {acc * 100:.2f}%\n")
    print(report)

    report_path = os.path.join(REPORTS_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Twitter Sentiment Analysis - Model Evaluation\n")
        f.write(f"{'=' * 50}\n\n")
        f.write(f"Test set size: {len(y_test):,}\n")
        f.write(f"Accuracy: {acc * 100:.2f}%\n\n")
        f.write(report)
    print(f"Saved report -> {report_path}")

    labels = sorted(y_test.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix (Accuracy: {acc * 100:.1f}%)")
    plt.tight_layout()

    cm_path = os.path.join(REPORTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    print(f"Saved confusion matrix -> {cm_path}")


if __name__ == "__main__":
    main()
