"""
Load the trained model and classify new tweet text.

Usage:
    python src/predict.py "I absolutely love this new phone, best purchase ever!"
    python src/predict.py --file path/to/tweets.txt   # one tweet per line
"""

import argparse
import os
import sys

import joblib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.preprocess import clean_text

MODELS_DIR = os.path.join(ROOT, "models")


def load_pipeline():
    vectorizer = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib"))
    model = joblib.load(os.path.join(MODELS_DIR, "sentiment_model.joblib"))
    return vectorizer, model


def predict(texts, vectorizer=None, model=None):
    """texts: list[str] of raw tweet text. Returns list of (label, confidence) tuples."""
    if vectorizer is None or model is None:
        vectorizer, model = load_pipeline()

    cleaned = [clean_text(t) for t in texts]
    X = vectorizer.transform(cleaned)
    labels = model.predict(X)
    probs = model.predict_proba(X)

    results = []
    for label, prob_row in zip(labels, probs):
        class_idx = list(model.classes_).index(label)
        confidence = prob_row[class_idx]
        results.append((label, round(float(confidence), 4)))
    return results


def main():
    parser = argparse.ArgumentParser(description="Classify tweet sentiment.")
    parser.add_argument("text", nargs="?", help="A single tweet to classify.")
    parser.add_argument("--file", help="Path to a text file with one tweet per line.")
    args = parser.parse_args()

    vectorizer, model = load_pipeline()

    if args.file:
        with open(args.file) as f:
            texts = [line.strip() for line in f if line.strip()]
    elif args.text:
        texts = [args.text]
    else:
        print("Provide a tweet as an argument or use --file. Example:")
        print('  python src/predict.py "I love this product!"')
        return

    results = predict(texts, vectorizer, model)
    for text, (label, confidence) in zip(texts, results):
        print(f"[{label.upper():8s}] ({confidence * 100:.1f}% confidence)  {text}")


if __name__ == "__main__":
    main()
