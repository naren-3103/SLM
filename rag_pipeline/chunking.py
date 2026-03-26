"""
Sentence-aware sliding-window chunker with overlap.

Each input page dict  →  list of chunk dicts:
    {
        "text":     str,   # chunk text
        "source":   str,   # original filename
        "page":     int,   # source page number
        "chunk_id": str,   # "<source>:p<page>:c<n>"
    }
"""

import re
import hashlib
from app.config import CHUNK_SIZE, CHUNK_OVERLAP


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _split_sentences(text: str) -> list[str]:
    """
    Split text into sentences using simple regex rules.
    Falls back gracefully if nltk is unavailable.
    """
    try:
        import nltk
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download("punkt_tab", quiet=True)
        sentences = nltk.sent_tokenize(text)
    except Exception:
        # Regex fallback: split on '. ', '! ', '? ' followed by capital letter or end
        sentences = re.split(r'(?<=[.!?])\s+', text)

    return [s.strip() for s in sentences if s.strip()]


def _build_chunks(sentences: list[str], chunk_size: int, overlap: int) -> list[str]:
    """
    Pack sentences into chunks up to `chunk_size` characters.
    Consecutive chunks share `overlap` characters of text.
    """
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        sentence_len = len(sentence)

        # If a single sentence exceeds chunk_size, hard-split it
        if sentence_len > chunk_size:
            # flush what we have
            if current:
                chunks.append(" ".join(current))
                current, current_len = [], 0
            # hard split the long sentence
            for start in range(0, sentence_len, chunk_size - overlap):
                sub = sentence[start : start + chunk_size]
                if sub.strip():
                    chunks.append(sub.strip())
            continue

        # Would adding this sentence exceed the limit?
        if current_len + sentence_len + (1 if current else 0) > chunk_size:
            chunks.append(" ".join(current))

            # Seed next chunk with overlap text from the tail of the current one
            overlap_text = " ".join(current)[-overlap:] if overlap > 0 else ""
            current = [overlap_text, sentence] if overlap_text else [sentence]
            current_len = sum(len(s) for s in current) + len(current) - 1
        else:
            current.append(sentence)
            current_len += sentence_len + (1 if len(current) > 1 else 0)

    if current:
        chunks.append(" ".join(current))

    return chunks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chunk_text(
    text: str,
    source: str = "unknown",
    page: int = 1,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """
    Chunk a single text string into overlapping, sentence-aware chunks.
    Returns a list of chunk metadata dicts.
    """
    sentences = _split_sentences(text)
    raw_chunks = _build_chunks(sentences, chunk_size, overlap)

    result: list[dict] = []
    for idx, chunk_text_str in enumerate(raw_chunks):
        chunk_id = f"{source}:p{page}:c{idx}"
        result.append({
            "text":     chunk_text_str,
            "source":   source,
            "page":     page,
            "chunk_id": chunk_id,
        })

    return result


def chunk_documents(pages: list[dict], chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """
    Chunk a list of page dicts (as returned by ``load_documents``).
    Returns a flat list of chunk dicts with deduplication by content hash.
    """
    all_chunks: list[dict] = []
    seen_hashes: set[str] = set()

    for page in pages:
        chunks = chunk_text(
            text=page["text"],
            source=page.get("source", "unknown"),
            page=page.get("page", 1),
            chunk_size=chunk_size,
            overlap=overlap,
        )
        for chunk in chunks:
            content_hash = hashlib.md5(chunk["text"].encode()).hexdigest()
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                all_chunks.append(chunk)

    return all_chunks