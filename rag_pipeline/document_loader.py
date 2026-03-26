import os
from pypdf import PdfReader


def load_documents(folder: str) -> list[dict]:
    """
    Load all PDF and TXT documents from a folder.

    Returns a list of page-level dicts:
        {
            "text":   str,   # raw extracted text for the page / file
            "source": str,   # original filename
            "page":   int,   # 1-based page number (TXT files always page 1)
        }
    """
    if not os.path.isdir(folder):
        print(f"[DocumentLoader] Folder not found: {folder}")
        return []

    docs: list[dict] = []

    for filename in sorted(os.listdir(folder)):
        filepath = os.path.join(folder, filename)

        # ── PDF ──────────────────────────────────────────────────────────────
        if filename.lower().endswith(".pdf"):
            try:
                reader = PdfReader(filepath)
                for page_num, page in enumerate(reader.pages, start=1):
                    text = page.extract_text() or ""
                    text = text.strip()
                    if text:          # skip blank / image-only pages
                        docs.append({
                            "text":   text,
                            "source": filename,
                            "page":   page_num,
                        })
            except Exception as exc:
                print(f"[DocumentLoader] Failed to read PDF '{filename}': {exc}")

        # ── TXT ──────────────────────────────────────────────────────────────
        elif filename.lower().endswith(".txt"):
            try:
                with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
                    text = fh.read().strip()
                if text:
                    docs.append({
                        "text":   text,
                        "source": filename,
                        "page":   1,
                    })
            except Exception as exc:
                print(f"[DocumentLoader] Failed to read TXT '{filename}': {exc}")

    print(f"[DocumentLoader] Loaded {len(docs)} page(s) from {folder!r}")
    return docs