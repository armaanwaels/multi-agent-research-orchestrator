"""Source 1: a folder of markdown documents (EIA "Energy Explained")."""

from __future__ import annotations

import re
from pathlib import Path

from .text_search import BM25, Passage

_SOURCE_LINE = re.compile(r"^Source: (\S+)", re.M)


def _split_sections(text: str) -> list[tuple[str, str]]:
    """Split a markdown document on headings into (heading, body) pairs."""
    sections, heading, buf = [], "", []
    for line in text.splitlines():
        if line.startswith("#"):
            if buf:
                sections.append((heading, "\n".join(buf).strip()))
            heading, buf = line.lstrip("# ").strip(), []
        else:
            buf.append(line)
    if buf:
        sections.append((heading, "\n".join(buf).strip()))
    return [(h, b) for h, b in sections if b]


class DocStore:
    def __init__(self, folder: Path) -> None:
        self.folder = folder
        self.docs: dict[str, str] = {p.stem: p.read_text() for p in sorted(folder.glob("*.md"))}
        passages = []
        for doc_id, text in self.docs.items():
            title = text.splitlines()[0].lstrip("# ").strip()
            m = _SOURCE_LINE.search(text)
            url = m.group(1) if m else ""
            for heading, body in _split_sections(text):
                if body.startswith("Source:"):
                    continue
                passages.append(Passage(doc_id, f"{title}: {heading}" if heading else title, url, body))
        self.index = BM25(passages)

    def list_docs(self) -> list[dict]:
        return [
            {"doc_id": doc_id, "title": text.splitlines()[0].lstrip("# ").strip(), "chars": len(text)}
            for doc_id, text in self.docs.items()
        ]

    def search(self, query: str, k: int = 4) -> list[dict]:
        return [
            {"doc_id": p.doc_id, "section": p.title, "url": p.url, "score": round(s, 2), "text": p.text}
            for s, p in self.index.search(query, k)
        ]

    def read(self, doc_id: str, start: int = 0, length: int = 4000) -> dict:
        if doc_id not in self.docs:
            raise KeyError(f"unknown doc_id {doc_id!r}; call list_docs to see what exists")
        text = self.docs[doc_id]
        return {"doc_id": doc_id, "start": start, "total_chars": len(text), "text": text[start : start + length]}
