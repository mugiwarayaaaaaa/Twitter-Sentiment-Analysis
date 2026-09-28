"""
Lightweight Flask demo: paste in tweet text, get back a predicted sentiment
from the trained TF-IDF + Logistic Regression model.

Usage:
    python app/demo_app.py
Then visit http://127.0.0.1:5001
"""

import os
import sys

from flask import Flask, render_template, request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.predict import load_pipeline, predict

app = Flask(__name__)
_vectorizer, _model = None, None


def get_pipeline():
    global _vectorizer, _model
    if _vectorizer is None:
        _vectorizer, _model = load_pipeline()
    return _vectorizer, _model


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    text = ""
    if request.method == "POST":
        text = request.form.get("text", "").strip()
        if text:
            vectorizer, model = get_pipeline()
            label, confidence = predict([text], vectorizer, model)[0]
            result = {"label": label, "confidence": round(confidence * 100, 1)}

    return render_template("index.html", result=result, text=text)


if __name__ == "__main__":
    app.run(debug=True, port=5001)
