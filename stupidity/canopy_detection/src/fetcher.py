"""Web content fetcher with extraction capabilities."""
from __future__ import annotations

import re
from typing import Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup


def fetch_and_extract(url: str, max_length: int = 5000) -> Optional[str]:
    """
    Fetch a URL and extract meaningful text content.

    Args:
        url: URL to fetch
        max_length: Maximum characters to extract

    Returns:
        Extracted text or None if fetch fails
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        response = requests.get(url, timeout=10, headers=headers)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        # Extract main content areas
        main_content = None
        for selector in ["article", "main", "[role='main']", ".content", ".post", ".entry"]:
            main_content = soup.select_one(selector)
            if main_content:
                break

        if not main_content:
            main_content = soup.body or soup

        text = main_content.get_text(separator="\n", strip=True)

        # Clean up text
        text = re.sub(r"\n\s*\n", "\n", text)  # Remove multiple newlines
        text = re.sub(r"[ \t]+", " ", text)  # Normalize spaces
        text = text[:max_length]

        return text.strip() if text else None

    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None


def extract_sentences(text: str, min_length: int = 20) -> list[str]:
    """
    Extract sentences from fetched text.

    Args:
        text: Raw text to process
        min_length: Minimum sentence length

    Returns:
        List of sentences
    """
    # Split on common sentence endings
    sentences = re.split(r"(?<=[.!?])\s+", text)

    # Filter and clean
    sentences = [
        s.strip()
        for s in sentences
        if len(s.strip()) >= min_length and not s.strip().startswith(("http", "#", "©"))
    ]

    return sentences
