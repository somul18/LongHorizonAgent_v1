"""Telemetry sinks. Every step emits one record: prompt tokens, state tokens,
number of facts, evictions, tool used, etc. Tinybird turns that stream into
live endpoints ("is context size flat over 10k steps?", "cost per run").
"""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path


class JsonlSink:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, record: dict) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def flush(self) -> None:
        pass


class TinybirdSink:
    """Batches records to the Tinybird Events API (NDJSON)."""

    def __init__(self, token: str | None = None, host: str | None = None, batch: int = 50):
        self.token = token or os.environ["TINYBIRD_TOKEN"]
        self.host = (host or os.environ.get("TINYBIRD_HOST", "https://api.tinybird.co")).rstrip("/")
        self.batch = batch
        self.buf: dict[str, list[dict]] = {}

    def emit(self, record: dict) -> None:
        ds = "agent_steps" if record.get("type") == "step" else "agent_archive"
        self.buf.setdefault(ds, []).append(record)
        if len(self.buf[ds]) >= self.batch:
            self._send(ds)

    def _send(self, ds: str) -> None:
        rows = self.buf.pop(ds, [])
        if not rows:
            return
        body = "\n".join(json.dumps(r, default=str) for r in rows).encode()
        req = urllib.request.Request(f"{self.host}/v0/events?name={ds}", data=body, method="POST")
        req.add_header("Authorization", f"Bearer {self.token}")
        try:
            urllib.request.urlopen(req, timeout=30).read()
        except Exception as e:  # telemetry must never kill a long run
            print(f"[tinybird] dropped {len(rows)} rows: {e}")

    def flush(self) -> None:
        for ds in list(self.buf):
            self._send(ds)


def sinks_from_env() -> list:
    return [TinybirdSink()] if os.environ.get("TINYBIRD_TOKEN") else []
