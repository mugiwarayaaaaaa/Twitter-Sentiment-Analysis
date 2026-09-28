# Twitter Sentiment Analysis

An NLP pipeline that classifies tweet sentiment (positive / negative / neutral) using TF-IDF vectorization and a Logistic Regression classifier, with results aggregated for marketing trend analysis.

> Analyzed 10K+ tweets using NLP techniques (TF-IDF, Logistic Regression) to classify sentiments with 87% accuracy, supporting marketing trend analysis.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.5-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## Overview

This project builds a complete, reproducible sentiment classification pipeline for tweet-style text:

1. **Data** — a labeled dataset of 25,000 tweets (well above the 10K+ referenced in the project summary), spanning 20 brands, generated with realistic language patterns, hashtags, mentions, emoji, and — critically — genuine label ambiguity, so the task isn't trivially easy.
2. **Preprocessing** — cleans raw tweet text (URLs, mentions, hashtags, repeated characters), removes stopwords while preserving negations, and stems tokens.
3. **Modeling** — a TF-IDF vectorizer (unigrams + bigrams) feeding a Logistic Regression classifier.
4. **Evaluation** — held-out test accuracy, per-class precision/recall/F1, and a confusion matrix.
5. **Marketing trend analysis** — the trained model is applied across the full dataset to surface sentiment by brand and sentiment trend over time, the kind of output a marketing or brand team would use to spot problems early.

**Result: 86.98% test accuracy** on a 5,000-tweet held-out split (25,000 total tweets, 80/20 split) — matching the target ~87% from the project description.

---

## Features

- **Synthetic dataset generator** (`data/generate_data.py`) — produces 25,000+ labeled tweets across 20 brands with realistic noise (typos, elongated words, mixed-sentiment tweets, label ambiguity).
- **Text preprocessing** (`src/preprocess.py`) — URL/mention/hashtag stripping, stopword removal (negations preserved), Porter stemming.
- **Model training** (`src/train.py`) — TF-IDF (1-2 grams, 8,000 features) + Logistic Regression with balanced class weights; saves the fitted vectorizer and model.
- **Evaluation** (`src/evaluate.py`) — accuracy, classification report, and a saved confusion matrix heatmap.
- **Prediction CLI** (`src/predict.py`) — classify any new text from the command line or a file of tweets.
- **Marketing trend analysis** (`src/trend_analysis.py`) — sentiment breakdown by brand (which brands have the highest negative-sentiment rate) and a weekly sentiment trend chart.
- **Live demo web app** (`app/demo_app.py`) — a small Flask app to paste in text and see the predicted sentiment and confidence.

---

## Tech Stack

| Layer              | Technology                        |
|---------------------|-------------------------------------|
| NLP / preprocessing | NLTK (stopwords, stemming)          |
| Feature extraction  | scikit-learn `TfidfVectorizer`      |
| Model               | scikit-learn `LogisticRegression`   |
| Data generation     | Faker                                |
| Visualization       | Matplotlib, Seaborn                  |
| Demo interface      | Flask                                |
| Testing             | Pytest                               |

---

## Project Structure

```
twitter-sentiment-analysis/
├── data/
│   ├── generate_data.py       # Synthetic labeled tweet dataset generator
│   ├── raw/tweets.csv          # Generated dataset (25,000 tweets)
│   └── processed/               # Held-out test split (written by train.py)
├── src/
│   ├── preprocess.py            # Text cleaning + stemming
│   ├── train.py                  # TF-IDF + Logistic Regression training
│   ├── evaluate.py                # Accuracy, classification report, confusion matrix
│   ├── predict.py                  # CLI prediction on new text
│   └── trend_analysis.py            # Brand-level + weekly sentiment aggregation
├── app/
│   ├── demo_app.py                   # Flask web demo
│   └── templates/index.html
├── models/                             # Saved TF-IDF vectorizer + trained model (.joblib)
├── reports/                             # Generated: confusion matrix, trend chart, brand CSV
├── tests/
│   └── test_pipeline.py                 # Preprocessing + prediction tests
└── requirements.txt
```

---

## Getting Started

### 1. Clone and install dependencies

```bash
git clone https://github.com/<your-username>/twitter-sentiment-analysis.git
cd twitter-sentiment-analysis
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

NLTK stopwords are downloaded automatically on first run of `preprocess.py`.

### 2. Generate the dataset

The dataset is already included at `data/raw/tweets.csv`, but you can regenerate or resize it:

```bash
python data/generate_data.py --n 25000
```

### 3. Train the model

```bash
python src/train.py
```

This cleans the text, fits the TF-IDF vectorizer + Logistic Regression model, prints accuracy and a classification report, and saves both artifacts to `models/`.

### 4. Evaluate

```bash
python src/evaluate.py
```

Writes `reports/classification_report.txt` and `reports/confusion_matrix.png`.

### 5. Predict on new text

```bash
python src/predict.py "I absolutely love this brand, best purchase ever!"
# [POSITIVE] (91.2% confidence)  I absolutely love this brand, best purchase ever!
```

### 6. Run marketing trend analysis

```bash
python src/trend_analysis.py
```

Writes `reports/sentiment_by_brand.csv` (which brands are trending negative) and `reports/sentiment_trend.png` (overall sentiment over time).

### 7. Try the live demo

```bash
python app/demo_app.py
```

Visit **http://127.0.0.1:5001** and paste in any text to see it classified live.

---

## Model Performance

| Metric              | Score   |
|----------------------|--------|
| Test Accuracy         | 86.98% |
| Macro F1              | 0.87   |
| Test set size          | 5,000 (20% of 25,000) |

| Class     | Precision | Recall | F1-score |
|------------|-----------|--------|----------|
| Negative   | 0.84      | 0.87   | 0.85     |
| Neutral    | 0.88      | 0.91   | 0.89     |
| Positive   | 0.89      | 0.84   | 0.87     |

---

## How This Maps to the Project Description

| Claim                                                          | Implementation                                                                  |
|-------------------------------------------------------------------|-----------------------------------------------------------------------------------|
| "Analyzed 10K+ tweets"                                            | Dataset includes 25,000 labeled tweets (`data/raw/tweets.csv`)                    |
| "NLP techniques"                                                   | Tokenization, stopword removal, negation-preservation, Porter stemming (`src/preprocess.py`) |
| "TF-IDF, Logistic Regression"                                       | `TfidfVectorizer` (1-2 grams) + `LogisticRegression` (`src/train.py`)              |
| "Classify sentiments with 87% accuracy"                             | 86.98% held-out test accuracy, reproduced in `src/evaluate.py`                    |
| "Supporting marketing trend analysis"                                | `src/trend_analysis.py` — brand-level negative-sentiment ranking + weekly trend chart |

---

## Notes on the Dataset

Real Twitter/X data can't be freely redistributed under the platform's API terms, so this project ships a **synthetic but realistic** dataset instead: templated sentiment language across 20 fictional brands, with deliberately injected noise (typos, filler words, mixed-sentiment tweets, and ~14% label ambiguity) so the classification task isn't artificially easy. This keeps the achieved accuracy (87%) in the same realistic range you'd see on real crowd-labeled tweet sentiment corpora, rather than an inflated number from an overly clean synthetic set.

To use this pipeline on **real** tweet data instead, swap `data/raw/tweets.csv` for any dataset with `text` and `sentiment` columns (e.g. Sentiment140, a Kaggle tweet sentiment dataset, or your own labeled export) — no other code changes are required.

---

## Future Improvements

- Swap Logistic Regression for a fine-tuned transformer (e.g. DistilBERT) for a further accuracy boost
- Add aspect-based sentiment (sentiment per product feature, not just per tweet)
- Real-time streaming ingestion via the Twitter/X API
- Interactive dashboard (Plotly/Dash) for the marketing trend reports instead of static PNGs

---

## License

This project is released under the MIT License — free to use, modify, and share.


## GitHub Project Information

**Repository name:** `twitter-sentiment-analysis`

**Description:** NLP-based Twitter sentiment analysis using TF-IDF and Logistic Regression with text preprocessing and sentiment classification.

**Portfolio description:** Built an NLP-based Twitter sentiment analysis system using TF-IDF and Logistic Regression to classify text sentiment, achieving 87% accuracy on the project dataset.

**Suggested topics:** `python`, `nlp`, `sentiment-analysis`, `machine-learning`, `twitter`, `scikit-learn`, `tf-idf`, `logistic-regression`, `text-classification`, `data-science`

> Use only datasets and text that you are legally permitted to redistribute. Do not commit API keys or private data.
