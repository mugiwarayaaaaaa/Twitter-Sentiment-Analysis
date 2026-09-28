"""
Marketing trend analysis: applies the trained sentiment model across the
full tweet dataset and aggregates results by brand and by week, producing
the kind of output a marketing team would use to track brand perception
over time.

Usage:
    python src/trend_analysis.py

Outputs:
    reports/sentiment_by_brand.csv
    reports/sentiment_trend.png
"""

import os
import sys

import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.preprocess import clean_series

MODELS_DIR = os.path.join(ROOT, "models")
REPORTS_DIR = os.path.join(ROOT, "reports")
DATA_PATH = os.path.join(ROOT, "data", "raw", "tweets.csv")


def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)

    print("Loading trained model...")
    vectorizer = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib"))
    model = joblib.load(os.path.join(MODELS_DIR, "sentiment_model.joblib"))

    print("Loading full tweet dataset...")
    df = pd.read_csv(DATA_PATH, parse_dates=["created_at"])
    print(f"  {len(df):,} tweets")

    print("Cleaning + predicting sentiment for every tweet...")
    df["clean_text"] = clean_series(df["text"])
    X = vectorizer.transform(df["clean_text"])
    df["predicted_sentiment"] = model.predict(X)

    # --- Sentiment breakdown by brand ---
    by_brand = (
        df.groupby(["brand", "predicted_sentiment"])
        .size()
        .unstack(fill_value=0)
    )
    by_brand["total"] = by_brand.sum(axis=1)
    for col in ["positive", "neutral", "negative"]:
        if col not in by_brand.columns:
            by_brand[col] = 0
    by_brand["positive_rate"] = (by_brand["positive"] / by_brand["total"] * 100).round(1)
    by_brand["negative_rate"] = (by_brand["negative"] / by_brand["total"] * 100).round(1)
    by_brand = by_brand.sort_values("negative_rate", ascending=False)

    brand_path = os.path.join(REPORTS_DIR, "sentiment_by_brand.csv")
    by_brand.to_csv(brand_path)
    print(f"\nSaved brand-level sentiment breakdown -> {brand_path}")

    print("\nBrands with the highest negative-sentiment rate (most in need of attention):")
    print(by_brand[["total", "positive_rate", "negative_rate"]].head(5).to_string())

    # --- Weekly sentiment trend (overall) ---
    df["week"] = df["created_at"].dt.to_period("W").dt.start_time
    weekly = df.groupby(["week", "predicted_sentiment"]).size().unstack(fill_value=0)
    weekly_pct = weekly.div(weekly.sum(axis=1), axis=0) * 100

    plt.figure(figsize=(10, 5))
    for sentiment, color in [("positive", "#16a34a"), ("neutral", "#64748b"), ("negative", "#dc2626")]:
        if sentiment in weekly_pct.columns:
            plt.plot(weekly_pct.index, weekly_pct[sentiment], label=sentiment.capitalize(), color=color, linewidth=2)

    plt.title("Weekly Sentiment Trend Across All Brands")
    plt.xlabel("Week")
    plt.ylabel("% of Tweets")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    trend_path = os.path.join(REPORTS_DIR, "sentiment_trend.png")
    plt.savefig(trend_path, dpi=150)
    print(f"Saved weekly sentiment trend chart -> {trend_path}")


if __name__ == "__main__":
    main()
