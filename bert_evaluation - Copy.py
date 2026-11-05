"""Unit III: BERT-family auxiliary evaluation (semantic similarity + sentiment classification).
Encoders are loaded lazily; if they cannot load (no internet, no torch) similarity falls back to TF-IDF
and is labelled as such. The sentiment classifier has NO fallback, so a BERT label is never faked."""
from functools import lru_cache
import numpy as np
import pandas as pd

SIM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"          # MiniLM: distilled BERT-style encoder
SENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"  # DistilBERT fine-tuned on SST-2 (binary sentiment)


@lru_cache(maxsize=1)
def _encoder():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(SIM_MODEL), None
    except Exception as e:
        return None, str(e)[:150]


@lru_cache(maxsize=1)
def _sentiment_pipe():
    try:
        from transformers import pipeline
        return pipeline("sentiment-analysis", model=SENT_MODEL), None
    except Exception as e:
        return None, str(e)[:150]


def embed(texts: list):
    """Return (unit-length matrix, method name)."""
    model, _ = _encoder()
    if model is not None:
        return np.asarray(model.encode(texts, normalize_embeddings=True)), "MiniLM sentence embeddings (BERT-family)"
    from sklearn.feature_extraction.text import TfidfVectorizer
    try:
        m = TfidfVectorizer().fit_transform(texts).toarray()
    except ValueError:
        m = np.zeros((len(texts), 1))
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(norms == 0, 1, norms), "TF-IDF fallback (NOT a BERT model)"


def similarity_to_reference(reference: str, outputs: dict) -> pd.DataFrame:
    names = [k for k, v in outputs.items() if v.strip()]
    if not reference.strip() or not names:
        return pd.DataFrame(columns=["Technique", "Similarity", "Method"])
    m, method = embed([reference] + [outputs[k] for k in names])
    sims = m[1:] @ m[0]
    return pd.DataFrame({"Technique": names, "Similarity": np.round(sims, 3), "Method": method})


def pairwise_matrix(outputs: dict) -> pd.DataFrame:
    names = [k for k, v in outputs.items() if v.strip()]
    if len(names) < 2:
        return pd.DataFrame()
    m, _ = embed([outputs[k] for k in names])
    return pd.DataFrame(np.round(m @ m.T, 3), index=names, columns=names)


def classify_sentiment(text: str) -> dict:
    pipe, err = _sentiment_pipe()
    if pipe is None:
        return {"label": None, "score": None, "model": SENT_MODEL, "error": err or "model unavailable"}
    r = pipe(text, truncation=True, max_length=512)[0]
    return {"label": r["label"].title(), "score": round(float(r["score"]), 3), "model": SENT_MODEL, "error": None}


def encoder_status() -> str:
    model, err = _encoder()
    return "MiniLM loaded" if model is not None else f"MiniLM unavailable ({err}); using TF-IDF fallback"
