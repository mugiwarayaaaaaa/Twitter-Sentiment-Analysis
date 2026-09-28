"""
Train a TF-IDF + Logistic Regression sentiment classifier on the tweet
dataset and save the fitted vectorizer + model to models/.

Usage:
    python src/train.py
    python src/train.py --data data/raw/tweets.csv --test-size 0.2
"""

import argparse
import os
import sys
import time

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import clean_series

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


def load_data(path):
    df = pd.read_csv(path)
    df = df.dropna(subset=["text", "sentiment"])
    return df


def train(data_path, test_size=0.2, random_state=42, max_features=8000):
    print(f"Loading dataset from {data_path} ...")
    df = load_data(data_path)
    print(f"  {len(df):,} tweets loaded")
    print(df["sentiment"].value_counts())

    print("\nCleaning text (lowercasing, stopword removal, stemming)...")
    t0 = time.time()
    df["clean_text"] = clean_series(df["text"])
    print(f"  done in {time.time() - t0:.1f}s")

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["sentiment"],
        test_size=test_size, random_state=random_state, stratify=df["sentiment"],
    )
    print(f"\nTrain size: {len(X_train):,}  |  Test size: {len(X_test):,}")

    print("\nFitting TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    print(f"  vocabulary size: {len(vectorizer.vocabulary_):,}")

    print("\nTraining Logistic Regression classifier...")
    model = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced", random_state=random_state)
    model.fit(X_train_vec, y_train)

    print("\nEvaluating on held-out test set...")
    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {acc * 100:.2f}%\n")
    print(classification_report(y_test, y_pred))

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(vectorizer, os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib"))
    joblib.dump(model, os.path.join(MODELS_DIR, "sentiment_model.joblib"))
    print(f"Saved vectorizer + model to {MODELS_DIR}/")

    # Save test split for the evaluate.py script (confusion matrix, etc.)
    test_df = pd.DataFrame({"clean_text": X_test, "sentiment": y_test})
    test_df.to_csv(os.path.join(os.path.dirname(MODELS_DIR), "data", "processed", "test_split.csv"), index=False)

    return acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the tweet sentiment classifier.")
    parser.add_argument("--data", default=os.path.join(os.path.dirname(MODELS_DIR), "data", "raw", "tweets.csv"))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--max-features", type=int, default=8000)
    args = parser.parse_args()

    train(args.data, test_size=args.test_size, max_features=args.max_features)
