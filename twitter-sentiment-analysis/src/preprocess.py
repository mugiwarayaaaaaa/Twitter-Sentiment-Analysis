"""
Text preprocessing utilities for tweet cleaning.

Applies typical NLP preprocessing for social media text:
    - lowercasing
    - removing URLs, mentions (@user), and raw hashtag symbols (keeping the word)
    - removing punctuation/emoji/numbers
    - collapsing repeated characters ("sooooo" -> "soo")
    - stopword removal
    - stemming
"""

import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

for pkg in ["stopwords", "punkt"]:
    try:
        nltk.data.find(f"corpora/{pkg}" if pkg == "stopwords" else f"tokenizers/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

STEMMER = PorterStemmer()
STOPWORDS = set(stopwords.words("english"))
# Keep negation words -- they flip sentiment and are important signal
NEGATIONS = {"no", "not", "nor", "never", "n't", "cannot", "cant", "dont", "wont"}
STOPWORDS -= NEGATIONS

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_SYMBOL_RE = re.compile(r"#")
NON_ALPHA_RE = re.compile(r"[^a-zA-Z\s]")
REPEAT_CHAR_RE = re.compile(r"(.)\1{2,}")


def clean_text(text):
    """Full cleaning pipeline. Returns a cleaned, stemmed, whitespace-joined string."""
    text = text.lower()
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = HASHTAG_SYMBOL_RE.sub(" ", text)  # drop '#' but keep the word
    text = REPEAT_CHAR_RE.sub(r"\1\1", text)  # "amazzzing" -> "amazzing"
    text = NON_ALPHA_RE.sub(" ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))

    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    tokens = [STEMMER.stem(t) for t in tokens]

    return " ".join(tokens)


def clean_series(series):
    """Vectorized-friendly cleaning for a pandas Series of raw tweet text."""
    return series.astype(str).apply(clean_text)
