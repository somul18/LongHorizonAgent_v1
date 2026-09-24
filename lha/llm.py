"""Model adapters. All return (text, usage) and share one tiny interface.

- BedrockLLM:  AWS Bedrock Converse API (the main reasoning model)
- OpenAICompatLLM: any OpenAI-compatible endpoint. Used for Liquid AI LFM
  models (served via Liquid's API, vLLM, llama.cpp or Ollama) as the cheap
  model that does compaction / summarization.
- ScriptedLLM: deterministic stand-in for tests and offline benchmarks.
"""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Protocol

from .tokens import estimate_tokens


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0


class LLM(Protocol):
    name: str

    def complete(self, system: str, prompt: str, max_tokens: int = 1024) -> tuple[str, Usage]: ...


class BedrockLLM:
    def __init__(self, model_id: str | None = None, region: str | None = None, temperature: float = 0.0):
        import boto3  # optional dependency

        self.model_id = model_id or os.environ["LHA_BEDROCK_MODEL_ID"]
        self.name = f"bedrock:{self.model_id}"
        self.temperature = temperature
        self.client = boto3.client("bedrock-runtime", region_name=region or os.environ.get("AWS_REGION", "us-east-1"))

    def complete(self, system: str, prompt: str, max_tokens: int = 1024) -> tuple[str, Usage]:
        resp = self.client.converse(
            modelId=self.model_id,
            system=[{"text": system}],
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": max_tokens, "temperature": self.temperature},
        )
        text = "".join(c.get("text", "") for c in resp["output"]["message"]["content"])
        u = resp.get("usage", {})
        return text, Usage(u.get("inputTokens", 0), u.get("outputTokens", 0))


class OpenAICompatLLM:
    """POST {base_url}/chat/completions. Defaults read LHA_LIQUID_* env vars."""

    def __init__(self, base_url: str | None = None, model: str | None = None, api_key: str | None = None):
        self.base_url = (base_url or os.environ["LHA_LIQUID_BASE_URL"]).rstrip("/")
        self.model = model or os.environ.get("LHA_LIQUID_MODEL", "LFM2-1.2B")
        self.api_key = api_key or os.environ.get("LHA_LIQUID_API_KEY", "")
        self.name = f"openai-compat:{self.model}"

    def complete(self, system: str, prompt: str, max_tokens: int = 512) -> tuple[str, Usage]:
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
        }).encode()
        req = urllib.request.Request(f"{self.base_url}/chat/completions", data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        if self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")
        with urllib.request.urlopen(req, timeout=120) as r:
            data = json.loads(r.read())
        u = data.get("usage", {})
        return data["choices"][0]["message"]["content"], Usage(u.get("prompt_tokens", 0), u.get("completion_tokens", 0))


@dataclass
class ScriptedLLM:
    """Calls a Python policy with the exact prompt a real model would see.
    Usage is estimated from prompt length so cost comparisons stay meaningful."""

    policy: Callable[[str, str], str]
    name: str = "scripted"
    calls: list[int] = field(default_factory=list)

    def complete(self, system: str, prompt: str, max_tokens: int = 1024) -> tuple[str, Usage]:
        out = self.policy(system, prompt)
        u = Usage(estimate_tokens(system) + estimate_tokens(prompt), estimate_tokens(out))
        self.calls.append(u.input_tokens)
        return out, u


def make_summarizer(llm: LLM) -> Callable[[str], str]:
    def summarize(text: str) -> str:
        out, _ = llm.complete("You compress agent memory. Reply with one short line, no preamble.", text, 120)
        return out.strip().splitlines()[0] if out.strip() else text[:200]

    return summarize


def from_env() -> LLM:
    """Pick the main model from the environment: Bedrock if configured, else Liquid."""
    if os.environ.get("LHA_BEDROCK_MODEL_ID"):
        return BedrockLLM()
    if os.environ.get("LHA_LIQUID_BASE_URL"):
        return OpenAICompatLLM()
    raise RuntimeError("Set LHA_BEDROCK_MODEL_ID (AWS) or LHA_LIQUID_BASE_URL (Liquid / OpenAI-compatible).")


def summarizer_from_env() -> Callable[[str], str] | None:
    """Compaction runs on the cheap model (Liquid LFM) when one is configured."""
    return make_summarizer(OpenAICompatLLM()) if os.environ.get("LHA_LIQUID_BASE_URL") else None
