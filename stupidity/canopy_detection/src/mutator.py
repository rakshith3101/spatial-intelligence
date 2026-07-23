from __future__ import annotations

import re


STOPWORDS = {
    # Articles & pronouns
    "a", "an", "and", "are", "as", "be", "from", "how", "into", "is", "may", "of", "the", "to", "was", "with",
    # Common verbs & auxiliaries
    "been", "being", "have", "has", "had", "do", "does", "did", "can", "could", "would", "should", "will", "shall",
    # Common words that appear everywhere
    "also", "about", "after", "all", "almost", "any", "back", "before", "between", "both", "by", "can", "come", "could",
    "down", "each", "early", "edit", "etc", "example", "first", "for", "get", "give", "go", "good", "got", "had", "has",
    "have", "he", "her", "here", "hers", "him", "his", "how", "i", "if", "in", "include", "including", "it", "its",
    "just", "know", "last", "like", "make", "many", "me", "might", "most", "much", "my", "no", "not", "now", "on",
    "only", "or", "other", "our", "out", "over", "same", "see", "should", "since", "so", "some", "such", "take", "than",
    "that", "the", "their", "them", "then", "there", "these", "they", "this", "through", "too", "under", "up", "use",
    "used", "very", "we", "what", "when", "where", "which", "while", "who", "whom", "why", "will", "with", "would",
    "you", "your",
    # Wiki/HTML artifacts & dates
    "category", "categories", "references", "see", "also", "links", "url", "cite", "citation",
    "january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december",
    "retrieved", "accessed", "archived", "format", "page",
    # Noise words that reduce signal
    "connect", "connects", "may", "pattern", "patterns", "shared", "latent", "relationship", "associate", "associated",
    "contain", "contains", "refer", "refers", "mention", "mentioned", "describe", "described", "relate", "relates",
    "removed", "removed", "challenged", "challenge",
    # Trivial/obvious connections
    "space", "earth", "world", "time", "year", "system", "data", "information", "process",
}


def _keywords(texts: list[str], limit: int = 6) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z-]+", " ".join(texts).lower())
    seen: dict[str, int] = {}
    for word in words:
        # Filter: not stopword, length > 4, not too common
        if word not in STOPWORDS and len(word) > 4 and not word.isdigit():
            seen[word] = seen.get(word, 0) + 1
    
    # Only keep words that appear at least 2 times (reduce noise)
    frequent_words = {word: count for word, count in seen.items() if count >= 2}
    
    if not frequent_words:
        # Fallback: use any non-stopword if frequency filter too strict
        frequent_words = {word: count for word, count in seen.items() if len(word) > 4}
    
    return [word for word, _ in sorted(frequent_words.items(), key=lambda item: (-item[1], item[0]))[:limit]]


def generate_hypothesis(path_texts: list[str]) -> str:
    if not path_texts:
        return "No semantic path was available for mutation."

    keywords = _keywords(path_texts)
    if len(keywords) >= 3:
        return (
            f"{keywords[0].title()} may connect {keywords[1]} and {keywords[2]} "
            "through a shared latent pattern."
        )
    if len(keywords) == 2:
        return f"{keywords[0].title()} may be associated with {keywords[1]} in this semantic graph."
    return f"{path_texts[0].rstrip('.')} may imply a broader hidden relationship."


def score_hypothesis(path_texts: list[str], generated_text: str) -> float:
    source_terms = set(_keywords(path_texts, limit=12))
    generated_terms = set(_keywords([generated_text], limit=12))
    if not source_terms:
        return 0.0
    overlap = len(source_terms & generated_terms) / len(source_terms)
    diversity = min(len(source_terms) / 8, 1.0)
    return round((0.65 * overlap) + (0.35 * diversity), 4)
