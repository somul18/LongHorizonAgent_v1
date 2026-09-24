"""Budget enforcement: keep the rendered state under a token budget.

When the working state grows past its budget, the compactor evicts the
least valuable items to the archive, cheapest-to-lose first:

    1. notes (volatile by definition), oldest first
    2. the last observation is truncated
    3. done/dropped tasks are collapsed into a one-line digest
    4. unpinned facts, lowest value first  (value = recency of last use x confidence)

Optionally a small, cheap model (e.g. a Liquid LFM) writes the digest of what
was evicted, so the gist survives even though the details moved to cold storage.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Callable

from .archive import Archive
from .state import WorkingState

Summarizer = Callable[[str], str]


class Compactor:
    def __init__(self, budget_tokens: int = 1500, obs_max_chars: int = 1200, summarizer: Summarizer | None = None):
        self.budget = budget_tokens
        self.obs_max_chars = obs_max_chars
        self.summarizer = summarizer
        self.evictions = 0

    def enforce(self, s: WorkingState, archive: Archive) -> list[str]:
        """Mutates state until it fits. Returns a log of what was evicted."""
        log: list[str] = []
        if len(s.observation) > self.obs_max_chars:
            s.observation = s.observation[: self.obs_max_chars] + " ...[truncated; full text archived]"
            log.append("truncated observation")

        while s.tokens() > self.budget and s.notes:
            n = s.notes.pop(0)
            archive.put(s.step, "note", f"note@{s.step}", {"text": n})
            log.append("evicted note")

        if s.tokens() > self.budget:
            closed = [t for t in s.tasks.values() if t.status in ("done", "dropped")]
            if closed:
                digest = "; ".join(f"{t.id}:{t.result or t.status}" for t in closed)
                for t in closed:
                    archive.put(s.step, "closed_task", t.id, asdict(t))
                    del s.tasks[t.id]
                if self.summarizer:
                    digest = self.summarizer(f"Summarize these completed tasks in one line:\n{digest}")
                s.notes.append(f"completed earlier: {digest[:300]}")
                log.append(f"collapsed {len(closed)} closed tasks")

        victims = sorted(
            (f for f in s.facts.values() if not f.pinned),
            key=lambda f: (f.last_used * f.confidence, f.step),
        )
        evicted = []
        while s.tokens() > self.budget and victims:
            f = victims.pop(0)
            del s.facts[f.key]
            archive.put(s.step, "evicted_fact", f.key, asdict(f))
            s.archived_keys.append(f.key)
            evicted.append(f.key)
        if evicted:
            log.append(f"evicted facts: {', '.join(evicted)}")

        # Breadcrumbs are themselves bounded; the archive still has everything.
        if len(s.archived_keys) > 200:
            s.archived_keys = s.archived_keys[-200:]

        self.evictions += len(log)
        return log
