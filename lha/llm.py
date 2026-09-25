"""Model adapters. All return (text, usage) and share one tiny interface.

- BedrockMantleLLM: Claude on Amazon Bedrock through the Messages API endpoint
  (bedrock-mantle), for current models with IDs like "anthropic.claude-sonnet-5"
- BedrockLLM:  AWS Bedrock Converse API, for versioned IDs such as
  "us.anthropic.claude-sonnet-4-5-20250929-v1:0" and non-Anthropic models
- OpenAICompatLLM: any OpenAI-compatible endpoint. Used for Liquid AI LFM
  models (served via Liquid's API, vLLM, llama.cpp or Ollama) as the cheap
  model that does compaction / summarization.
- ScriptedLLM: deterministic stand-in for tests and offline benchmarks.
"""

from __future__ import annotations

import json
import os
import re
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
        from botocore.config import Config

        # Fail loudly instead of sitting silently: boto3's defaults wait 60 s per attempt and retry quietly.
        timeout = float(os.environ.get("LHA_LLM_TIMEOUT", "60"))
        self.client = boto3.client("bedrock-runtime", region_name=region or os.environ.get("AWS_REGION", "us-east-1"),
                                   config=Config(connect_timeout=10, read_timeout=timeout,
                                                 retries={"mode": "standard", "max_attempts": 1}))

    def complete(self, system: str, prompt: str, max_tokens: int = 2048) -> tuple[str, Usage]:
        # 2048, not 1024: a cad_build action carries a whole build123d script inside the JSON reply,
        # and a truncated reply costs a step (parse error fed back) instead of a few tokens.
        try:
            resp = self.client.converse(
                modelId=self.model_id,
                system=[{"text": system}],
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={"maxTokens": max_tokens, "temperature": self.temperature},
            )
        except self.client.exceptions.ValidationException as e:
            if "inference profile" in str(e) and self.model_id.startswith("anthropic."):
                raise RuntimeError(f"{e}\nUse a cross-region inference profile ID, e.g. "
                                   f"LHA_BEDROCK_MODEL_ID=us.{self.model_id}") from e
            raise
        text = "".join(c.get("text", "") for c in resp["output"]["message"]["content"])
        u = resp.get("usage", {})
        return text, Usage(u.get("inputTokens", 0), u.get("outputTokens", 0))


class BedrockMantleLLM:
    """Current Claude models on Bedrock (Messages API shape, `anthropic.`-prefixed IDs).

    No `temperature`: current models reject sampling parameters. Thinking runs
    adaptively by default and its tokens count against max_tokens, hence the
    larger default. LHA_EFFORT (low|medium|high|xhigh|max) trades depth for
    cost and latency per step; unset uses the API default."""

    def __init__(self, model_id: str | None = None, region: str | None = None, effort: str | None = None,
                 client=None):
        self.model_id = model_id or os.environ["LHA_BEDROCK_MODEL_ID"]
        self.name = f"bedrock-mantle:{self.model_id}"
        self.effort = effort or os.environ.get("LHA_EFFORT") or None
        if client is None:
            from anthropic import AnthropicBedrockMantle  # optional dependency: pip install "anthropic[bedrock]"

            client = AnthropicBedrockMantle(aws_region=region or os.environ.get("AWS_REGION", "us-east-1"))
        self.client = client

    def complete(self, system: str, prompt: str, max_tokens: int = 16000) -> tuple[str, Usage]:
        extra = {"output_config": {"effort": self.effort}} if self.effort else {}
        resp = self.client.messages.create(
            model=self.model_id,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            **extra,
        )
        if resp.stop_reason == "refusal":
            return "", Usage(resp.usage.input_tokens, resp.usage.output_tokens)  # fed back as a parse error
        text = "".join(b.text for b in resp.content if b.type == "text")
        return text, Usage(resp.usage.input_tokens, resp.usage.output_tokens)


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
    """Pick the main model from the environment: Bedrock if configured, else Liquid.
    Versioned Bedrock IDs ("...-v1:0") go through Converse; current ones through Mantle."""
    if model_id := os.environ.get("LHA_BEDROCK_MODEL_ID"):
        return BedrockLLM() if re.search(r"-v\d+:\d+$", model_id) else BedrockMantleLLM()
    if os.environ.get("LHA_LIQUID_BASE_URL"):
        return OpenAICompatLLM()
    raise RuntimeError("Set LHA_BEDROCK_MODEL_ID (AWS) or LHA_LIQUID_BASE_URL (Liquid / OpenAI-compatible).")


def summarizer_from_env() -> Callable[[str], str] | None:
    """Compaction runs on the cheap model (Liquid LFM) when one is configured."""
    return make_summarizer(OpenAICompatLLM()) if os.environ.get("LHA_LIQUID_BASE_URL") else None
