from __future__ import annotations

from functools import lru_cache

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize


MODEL_NAME = "all-MiniLM-L6-v2"
FALLBACK_DIMENSIONS = 384


@lru_cache(maxsize=1)
def _load_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_NAME)


@lru_cache(maxsize=1)
def _fallback_vectorizer() -> HashingVectorizer:
    return HashingVectorizer(
        n_features=FALLBACK_DIMENSIONS,
        alternate_sign=False,
        analyzer="char_wb",
        ngram_range=(3, 5),
        norm=None,
    )


def _fallback_embeddings(texts: list[str]) -> np.ndarray:
    vectors = _fallback_vectorizer().transform(texts)
    return normalize(vectors, norm="l2").toarray()


def generate_embeddings(nodes: list[dict]) -> list[dict]:
    if not nodes:
        return nodes

    texts = [node["text"] for node in nodes]
    try:
        embeddings = _load_model().encode(texts, normalize_embeddings=True)
    except (ImportError, OSError):
        embeddings = _fallback_embeddings(texts)
    for node, embedding in zip(nodes, embeddings):
        node["embedding"] = np.asarray(embedding, dtype=float).tolist()
    return nodes


def cosine_similarity(left: list[float], right: list[float]) -> float:
    left_vector = np.asarray(left, dtype=float)
    right_vector = np.asarray(right, dtype=float)
    denominator = np.linalg.norm(left_vector) * np.linalg.norm(right_vector)
    if denominator == 0:
        return 0.0
    return float(np.dot(left_vector, right_vector) / denominator)
