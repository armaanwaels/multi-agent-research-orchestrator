"""A small BM25 index for the markdown docs and the web fixture.

Both corpora are under a few hundred passages, so an in-memory index is fast
enough and keeps the dependency list short.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "a an and are as at be by for from has have how in is it its of on or that the this to was "
    "were what when where which who why will with does did do".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


@dataclass
class Passage:
    doc_id: str
    title: str
    url: str
    text: str


class BM25:
    def __init__(self, passages: list[Passage], k1: float = 1.5, b: float = 0.75) -> None:
        self.passages = passages
        self.k1, self.b = k1, b
        self.tfs = [Counter(tokenize(f"{p.title} {p.text}")) for p in passages]
        self.lens = [sum(tf.values()) for tf in self.tfs]
        self.avg_len = sum(self.lens) / max(len(self.lens), 1)
        df: Counter[str] = Counter()
        for tf in self.tfs:
            df.update(tf.keys())
        n = len(passages)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def search(self, query: str, k: int = 5) -> list[tuple[float, Passage]]:
        terms = tokenize(query)
        scored = []
        for i, tf in enumerate(self.tfs):
            s = 0.0
            for t in terms:
                if t not in tf:
                    continue
                f = tf[t]
                s += (
                    self.idf[t]
                    * f
                    * (self.k1 + 1)
                    / (f + self.k1 * (1 - self.b + self.b * self.lens[i] / self.avg_len))
                )
            if s > 0:
                scored.append((s, self.passages[i]))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:k]
