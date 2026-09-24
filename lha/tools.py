"""Tools the agent can call. Every tool returns a string observation.

Only the latest observation is shown in the working state; the full text of
every observation goes to the archive, so the agent must copy what matters
into facts with `set_fact` (or `recall` it later).
"""

from __future__ import annotations

import base64
import json
import os
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable

from .archive import Archive

ToolFn = Callable[[dict], str]


@dataclass
class Tool:
    name: str
    description: str
    fn: ToolFn


class ToolBox:
    def __init__(self, tools: list[Tool] | None = None):
        self.tools: dict[str, Tool] = {}
        for t in tools or []:
            self.add(t)

    def add(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    def help(self) -> str:
        return "\n".join(f"  {t.name}: {t.description}" for t in self.tools.values())

    def call(self, name: str, args: dict) -> str:
        if name not in self.tools:
            return f"ERROR: unknown tool {name!r}. Available: {', '.join(self.tools)}"
        try:
            return self.tools[name].fn(args or {})
        except Exception as e:  # tool failures are observations, not crashes
            return f"ERROR in {name}: {type(e).__name__}: {e}"


def recall_tool(archive: Archive) -> Tool:
    def fn(args: dict) -> str:
        hits = archive.recall(str(args.get("query", "")), k=int(args.get("k", 5)), kinds=args.get("kinds"))
        if not hits:
            return "recall: no matches"
        return "\n".join(f"[step {h.step}] {h.kind} {h.key}: {json.dumps(h.payload, default=str)[:400]}" for h in hits)

    return Tool("recall", 'search cold storage. args: {"query": str, "k": int, "kinds": [optional: evicted_fact|superseded_fact|dropped_fact|observation|note|closed_task]}', fn)


# ------------------------------------------------------------------ Nimble
# Nimble's realtime Web/SERP APIs. Endpoint and payload shape are configurable
# because they vary by plan; check your Nimble dashboard/docs and override via env.
NIMBLE_BASE = os.environ.get("LHA_NIMBLE_BASE_URL", "https://api.webit.live/api/v1/realtime")


def _nimble_post(path: str, payload: dict) -> Any:
    creds = os.environ["NIMBLE_CREDENTIALS"]  # "username:password" or a pre-encoded token
    token = creds if ":" not in creds else base64.b64encode(creds.encode()).decode()
    req = urllib.request.Request(f"{NIMBLE_BASE}/{path}", data=json.dumps(payload).encode(), method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Basic {token}")
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.loads(r.read())


def nimble_tools() -> list[Tool]:
    def search(args: dict) -> str:
        data = _nimble_post("serp", {"search_engine": "google_search", "query": args["query"], "parse": True})
        results = (data.get("parsing") or {}).get("entities", {}).get("OrganicResult", []) or data.get("results", [])
        lines = [f"- {r.get('title')} | {r.get('url')} | {r.get('snippet', '')[:200]}" for r in results[: int(args.get("k", 5))]]
        return "\n".join(lines) or json.dumps(data)[:1500]

    def fetch(args: dict) -> str:
        data = _nimble_post("web", {"url": args["url"], "render": True, "format": "markdown"})
        return str(data.get("markdown") or data.get("html_content") or data)[:6000]

    return [
        Tool("web_search", 'Nimble SERP search. args: {"query": str, "k": int}', search),
        Tool("web_fetch", 'Nimble page fetch (clean text). args: {"url": str}', fetch),
    ]
