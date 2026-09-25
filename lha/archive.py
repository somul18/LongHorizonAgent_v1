"""Cold storage: an append-only event log plus a searchable archive.

Everything the agent ever saw or did lands here, so nothing is truly lost, but
none of it is replayed into the prompt. The agent pulls specific items back
with the `recall` tool when it decides it needs them.

The log is a local JSONL file (source of truth, enables crash-resume and
audits) and is optionally mirrored to sinks such as Tinybird for analytics.
"""

from __future__ import annotations

import json
import math
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Protocol

_WORD = re.compile(r"[a-z0-9_\-\.]+")


_SPLIT = re.compile(r"[-_.]+")


def _terms(text: str) -> list[str]:
    """Whole tokens plus their parts, so "base-plate_width", "base-plate.width" and
    "base-plate width" all match each other (models do not name keys consistently)."""
    out = []
    for tok in _WORD.findall(text.lower()):
        out.append(tok)
        parts = [p for p in _SPLIT.split(tok) if p]
        if len(parts) > 1:
            out += parts
    return out


class Sink(Protocol):
    def emit(self, record: dict) -> None: ...
    def flush(self) -> None: ...


@dataclass
class ArchiveItem:
    id: int
    run_id: str
    step: int
    kind: str
    key: str
    payload: dict
    ts: float

    def text(self) -> str:
        return f"{self.kind} {self.key} {json.dumps(self.payload, default=str)}"


class Archive:
    def __init__(self, path: str | Path | None, run_id: str, sinks: Iterable[Sink] = ()):
        self.path = Path(path) if path else None
        self.run_id = run_id
        self.sinks = list(sinks)
        self.items: list[ArchiveItem] = []
        self._df: Counter[str] = Counter()
        if self.path and self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    self._index(ArchiveItem(**json.loads(line)))

    def _index(self, item: ArchiveItem) -> None:
        self.items.append(item)
        self._df.update(set(_terms(item.text())))

    def put(self, step: int, kind: str, key: str, payload: dict[str, Any]) -> ArchiveItem:
        item = ArchiveItem(len(self.items), self.run_id, step, kind, key, payload, time.time())
        self._index(item)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as f:
                f.write(json.dumps(item.__dict__, default=str) + "\n")
        for s in self.sinks:
            s.emit({"type": "archive", **item.__dict__, "payload": json.dumps(payload, default=str)})
        return item

    def recall(self, query: str, k: int = 5, kinds: Iterable[str] | None = None) -> list[ArchiveItem]:
        """BM25-flavoured lexical search, newest wins ties. Good enough for keys,
        ids and names; swap for embeddings if the domain needs it."""
        q = _terms(query)
        if not q:
            return []
        qs = set(_WORD.findall(query.lower()))
        kinds = set(kinds) if kinds else None
        n = max(1, len(self.items))
        scored = []
        for it in self.items:
            if kinds and it.kind not in kinds:
                continue
            tf = Counter(_terms(it.text()))
            score = sum(tf[t] / (tf[t] + 1.2) * math.log(1 + n / (1 + self._df[t])) for t in q if t in tf)
            if score > 0 and it.key.lower() in qs:
                score += 100.0  # the exact key the agent asked for beats partial word matches
            if score > 0:
                scored.append((score, it.id, it))
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return [it for _, _, it in scored[:k]]

    def flush(self) -> None:
        for s in self.sinks:
            s.flush()
