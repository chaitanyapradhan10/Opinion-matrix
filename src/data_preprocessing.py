"""
data_preprocessing.py
----------------------
Phase 3 (Data Collection checks) and Phase 5 (Data Cleaning) of the
Sentiment Analysis pipeline.

Provides:
    load_data(path)            -> pandas DataFrame, de-duplicated, no NaNs
    clean_text(text)           -> lowercased, punctuation/HTML/URL/emoji/number
                                   stripped string
    tokenize_and_normalize(text, use_lemmatization=True) -> cleaned token list
    preprocess_dataframe(df)   -> df with 'clean_review' and 'tokens' columns
"""

import re
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

from logging_config import get_logger

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "reviews.csv"

#data = pd.read_csv(DATA_PATH)

logger = get_logger(__name__)

# Ensure required NLTK resources are present (no-op if already downloaded)
'''for pkg in ["stopwords", "punkt", "punkt_tab", "wordnet", "omw-1.4"]:
    try:
        nltk.data.find(pkg)
    except LookupError:
        logger.info("NLTK resource '%s' not found locally; downloading...", pkg)
        nltk.download(pkg, quiet=True)
    except Exception:
        logger.warning("Unexpected error checking NLTK resource '%s'; attempting download.", pkg)
        nltk.download(pkg, quiet=True)'''

STOPWORDS = set(stopwords.words("english"))
STEMMER = PorterStemmer()
LEMMATIZER = WordNetLemmatizer()

URL_RE = re.compile(r"https?://\S+|www\.\S+")
HTML_RE = re.compile(r"<.*?>")
EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002700-\U000027BF"
    "\U0001F900-\U0001F9FF"
    "\U00002600-\U000026FF"
    "]+",
    flags=re.UNICODE,
)
NUMBER_RE = re.compile(r"\d+")
PUNCT_TABLE = str.maketrans("", "", string.punctuation)


# ---------------------------------------------------------------------------
# Phase 3: Data Collection
# ---------------------------------------------------------------------------
def load_data(path: str, text_col: str = "review", label_col: str = "sentiment") -> pd.DataFrame:
    """Load the raw CSV, report basic diagnostics, drop nulls/duplicates."""
    logger.info("Loading data from '%s'", path)
    try:
        df = pd.read_csv(path)
    except Exception:
        logger.exception("Failed to load CSV from '%s'", path)
        raise

    n_missing = df[text_col].isna().sum() + (df[text_col].astype(str).str.strip() == "").sum()
    n_dupes = df.duplicated(subset=[text_col]).sum()

    logger.info("Loaded %d rows.", len(df))
    logger.info("Missing/empty reviews: %d", n_missing)
    logger.info("Duplicate reviews: %d", n_dupes)
    logger.info("Class distribution:\n%s", df[label_col].value_counts())

    df = df.dropna(subset=[text_col])
    df = df[df[text_col].astype(str).str.strip() != ""]
    df = df.drop_duplicates(subset=[text_col]).reset_index(drop=True)

    logger.info("Rows after cleaning: %d", len(df))
    return df


# ---------------------------------------------------------------------------
# Phase 5: Data Cleaning
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Lowercase + strip HTML/URLs/emojis/numbers/punctuation/extra whitespace."""
    text = str(text).lower()
    text = HTML_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = EMOJI_RE.sub(" ", text)
    text = NUMBER_RE.sub(" ", text)
    text = text.translate(PUNCT_TABLE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_and_normalize(text: str, use_lemmatization: bool = True) -> list:
    """Tokenize, remove stopwords, and stem or lemmatize."""
    cleaned = clean_text(text)
    tokens = word_tokenize(cleaned)
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    if use_lemmatization:
        tokens = [LEMMATIZER.lemmatize(t) for t in tokens]
    else:
        tokens = [STEMMER.stem(t) for t in tokens]
    return tokens


def preprocess_dataframe(df: pd.DataFrame, text_col: str = "review",
                          use_lemmatization: bool = True) -> pd.DataFrame:
    """Add 'clean_review' (string) and 'tokens' (list) columns to the dataframe."""
    logger.info(
        "Preprocessing dataframe with %d rows (lemmatization=%s)",
        len(df), use_lemmatization,
    )
    df = df.copy()
    df["tokens"] = df[text_col].apply(lambda t: tokenize_and_normalize(t, use_lemmatization))
    df["clean_review"] = df["tokens"].apply(lambda toks: " ".join(toks))
    empty_after_clean = (df["clean_review"].str.strip() == "").sum()
    if empty_after_clean:
        logger.warning("%d rows became empty strings after cleaning.", empty_after_clean)
    logger.info("Preprocessing complete.")
    return df


if __name__ == "__main__":
    logger.info("Starting data preprocessing pipeline run.")
    data = pd.read_csv(DATA_PATH)
    data = preprocess_dataframe(data)
    logger.info("Pipeline run complete. Sample output:\n%s",
                data[["review", "clean_review"]].head())
