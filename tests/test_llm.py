from types import SimpleNamespace as NS

import pytest

from lha import llm
from lha.llm import BedrockMantleLLM


class FakeMessages:
    def __init__(self):
        self.calls = []

    def create(self, **kw):
        self.calls.append(kw)
        return NS(stop_reason="end_turn", usage=NS(input_tokens=120, output_tokens=30),
                  content=[NS(type="thinking", thinking=""), NS(type="text", text='{"action": {"tool": "x"}}')])


def test_mantle_returns_text_blocks_and_sends_no_sampling_params():
    fake = NS(messages=FakeMessages())
    m = BedrockMantleLLM(model_id="anthropic.claude-sonnet-5", effort="medium", client=fake)
    text, usage = m.complete("sys", "prompt")
    assert text == '{"action": {"tool": "x"}}' and (usage.input_tokens, usage.output_tokens) == (120, 30)
    call = fake.messages.calls[0]
    assert call["output_config"] == {"effort": "medium"} and "temperature" not in call
    assert call["model"] == "anthropic.claude-sonnet-5" and call["system"] == "sys"


@pytest.mark.parametrize("model_id,cls", [
    ("anthropic.claude-sonnet-5", "BedrockMantleLLM"),
    ("anthropic.claude-opus-5", "BedrockMantleLLM"),
    ("us.anthropic.claude-sonnet-4-5-20250929-v1:0", "BedrockLLM"),
])
def test_from_env_routes_by_model_id(monkeypatch, model_id, cls):
    monkeypatch.setenv("LHA_BEDROCK_MODEL_ID", model_id)
    monkeypatch.setattr(llm, "BedrockLLM", lambda: "BedrockLLM")
    monkeypatch.setattr(llm, "BedrockMantleLLM", lambda: "BedrockMantleLLM")
    assert llm.from_env() == cls
