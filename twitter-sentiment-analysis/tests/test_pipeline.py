"""
Tests for the Twitter Sentiment Analysis pipeline.

Run with:
    pytest

Note: test_predict.py tests require a trained model in models/ (run
`python src/train.py` first if they're missing).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src.preprocess import clean_text

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_EXISTS = os.path.exists(os.path.join(MODELS_DIR, "sentiment_model.joblib"))


def test_clean_text_lowercases():
    assert clean_text("HELLO World") == clean_text("hello world")


def test_clean_text_removes_urls():
    cleaned = clean_text("Check this out https://example.com/page amazing")
    assert "http" not in cleaned
    assert "example" not in cleaned


def test_clean_text_removes_mentions():
    cleaned = clean_text("@someuser this is great")
    assert "someuser" not in cleaned


def test_clean_text_removes_hashtag_symbol_keeps_word():
    cleaned = clean_text("Loving this #sale right now")
    assert "#" not in cleaned
    # stemmed form of "sale" should still appear
    assert "sale" in cleaned or "sal" in cleaned


def test_clean_text_keeps_negation():
    cleaned = clean_text("This is not good at all")
    assert "not" in cleaned


def test_clean_text_empty_string():
    assert clean_text("") == ""


@pytest.mark.skipif(not MODEL_EXISTS, reason="Trained model not found; run `python src/train.py` first.")
def test_predict_positive_tweet():
    from src.predict import predict
    results = predict(["I absolutely love this, best experience ever, amazing!"])
    label, confidence = results[0]
    assert label == "positive"
    assert 0 <= confidence <= 1


@pytest.mark.skipif(not MODEL_EXISTS, reason="Trained model not found; run `python src/train.py` first.")
def test_predict_negative_tweet():
    from src.predict import predict
    results = predict(["Absolutely terrible experience, broken, disappointed, worst ever"])
    label, confidence = results[0]
    assert label == "negative"


@pytest.mark.skipif(not MODEL_EXISTS, reason="Trained model not found; run `python src/train.py` first.")
def test_predict_returns_valid_labels():
    from src.predict import predict
    results = predict(["Just a regular update about store hours"])
    label, confidence = results[0]
    assert label in {"positive", "negative", "neutral"}
