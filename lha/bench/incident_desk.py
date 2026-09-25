"""IncidentDesk: a synthetic long-horizon task with drifting ground truth.

The agent drains an inbox of N messages (one per `next_message` call). Most
are noisy log chatter; some change an attribute of a service ("owner of
payments-api is now dana"). Values change repeatedly over the run, so early
information goes stale. When the inbox is empty the env asks questions about
the *current* value of several attributes, including ones last touched at the
very start of the run. The agent must `finish` with {"q1": value, ...}.

It stresses exactly the failure modes of long-horizon agents:
  - context growth (N can be 50 or 5,000)
  - stale vs current information about the same key
  - facts needed at the end that were last seen at the very beginning
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field

from ..tools import Tool

SERVICES = ["payments-api", "auth-svc", "search", "checkout", "ledger", "notifier",
            "img-resize", "geo", "billing", "catalog", "fraud", "gateway"]
ATTRS = {
    "owner": ["alice", "bob", "carol", "dana", "eve", "frank", "grace", "heidi"],
    "region": ["us-east-1", "us-west-2", "eu-west-1", "eu-central-1", "ap-south-1"],
    "version": [f"v{a}.{b}" for a in range(1, 6) for b in range(10)],
}
TEMPLATES = {
    "owner": "[pager] ownership transfer: {svc} is now owned by {val} (ticket OPS-{n})",
    "region": "[infra] {svc} primary region migrated to {val}; old region drained (change CHG-{n})",
    "version": "[deploy] {svc} rolled out {val} to 100% of traffic (build #{n})",
}
UPDATE_RE = re.compile(
    r"(?:ownership transfer: (?P<s1>[\w-]+) is now owned by (?P<owner>[\w-]+))"
    r"|(?:\] (?P<s2>[\w-]+) primary region migrated to (?P<region>[\w-]+);)"
    r"|(?:\] (?P<s3>[\w-]+) rolled out (?P<version>v[\d.]+) to)"
)
NOISE = [
    "[log] {svc} p99 latency {a}ms, error rate 0.{b}%, {c} pods healthy, GC pause {a}us, cache hit {b}{c}%",
    "[chat] someone asked whether {svc} dashboards are stale again; nobody answered; thread has {c} replies",
    "[ci] nightly suite for {svc}: {a} passed, {b} skipped, flaky test test_retry_{c} quarantined again",
    "[alert] {svc} disk usage {b}{c}% on node-{a}; auto-resolved after compaction finished",
]


def parse_update(text: str) -> tuple[str, str, str] | None:
    m = UPDATE_RE.search(text)
    if not m:
        return None
    for svc_g, attr in (("s1", "owner"), ("s2", "region"), ("s3", "version")):
        if m.group(svc_g):
            return m.group(svc_g), attr, m.group(attr)
    return None


@dataclass
class IncidentDesk:
    n_messages: int = 200
    update_rate: float = 0.3
    n_questions: int = 6
    seed: int = 0
    messages: list[str] = field(default_factory=list)
    truth: dict[str, str] = field(default_factory=dict)
    questions: dict[str, str] = field(default_factory=dict)
    cursor: int = 0

    def __post_init__(self) -> None:
        rng = random.Random(self.seed)
        first_seen: dict[str, int] = {}
        # Seed every (svc, attr) once at the start so early facts exist.
        for svc in SERVICES:
            for attr, vals in ATTRS.items():
                self._update(rng, svc, attr, rng.choice(vals))
                first_seen[f"{svc}.{attr}"] = len(self.messages)
        while len(self.messages) < self.n_messages:
            if rng.random() < self.update_rate:
                svc, attr = rng.choice(SERVICES), rng.choice(list(ATTRS))
                self._update(rng, svc, attr, rng.choice(ATTRS[attr]))
            else:
                self.messages.append(rng.choice(NOISE).format(
                    svc=rng.choice(SERVICES), a=rng.randint(10, 999), b=rng.randint(1, 9), c=rng.randint(1, 9)))
        # Ask about the most-changed keys AND the ones untouched since the start.
        counts = {k: sum(1 for m in self.messages if (u := parse_update(m)) and f"{u[0]}.{u[1]}" == k) for k in self.truth}
        by_churn = sorted(self.truth, key=lambda k: -counts[k])
        untouched = [k for k in self.truth if counts[k] == 1]
        rng.shuffle(untouched)
        keys = list(dict.fromkeys(by_churn[: self.n_questions // 2] + untouched))[: self.n_questions]
        keys += [k for k in by_churn if k not in keys][: self.n_questions - len(keys)]
        self.questions = {f"q{i + 1}": k for i, k in enumerate(keys)}

    def _update(self, rng: random.Random, svc: str, attr: str, val: str) -> None:
        self.truth[f"{svc}.{attr}"] = val
        self.messages.append(TEMPLATES[attr].format(svc=svc, val=val, n=rng.randint(1000, 9999)))

    # ------------------------------------------------------------------ agent API
    def goal(self) -> str:
        return ("Drain the incident inbox by calling next_message until it is empty. Track the CURRENT owner, "
                "region and version of every service (values change over time; only the latest counts). "
                "When the inbox is empty you will get questions; answer with finish "
                '{"answer": {"q1": "<value>", ...}}.')

    def question_text(self) -> str:
        qs = "; ".join(f"{qid}: current {k.split('.')[1]} of {k.split('.')[0]}" for qid, k in self.questions.items())
        return f"INBOX EMPTY. Questions -> {qs}"

    def tools(self) -> list[Tool]:
        def next_message(_args: dict) -> str:
            if self.cursor >= len(self.messages):
                return self.question_text()
            self.cursor += 1
            return f"msg {self.cursor}/{len(self.messages)}: {self.messages[self.cursor - 1]}"

        return [Tool("next_message", "read the next inbox message. args: {}", next_message)]

    def score(self, answer) -> float:
        if not isinstance(answer, dict):
            return 0.0
        ok = sum(1 for qid, k in self.questions.items() if str(answer.get(qid, "")).strip() == self.truth[k])
        return ok / len(self.questions)


# ====================================================================== policies
# Deterministic "agents" that read the exact prompt a real model would get.
# They make the benchmark reproducible offline; with --live a real model is used.

QUESTION_RE = re.compile(r"(q\d+): current (\w+) of ([\w-]+)")


def _section(prompt: str, name: str) -> str:
    m = re.search(rf"# {name}[^\n]*\n(.*?)(?=\n# |\Z)", prompt, re.S)
    return m.group(1) if m else ""


def stateful_policy(system: str, prompt: str) -> str:
    facts = dict(re.findall(r"^- \*?([\w.-]+) = \"([^\"]*)\"", _section(prompt, "FACTS"), re.M))
    obs = _section(prompt, "LAST OBSERVATION")
    open_qs = QUESTION_RE.findall(_section(prompt, "OPEN QUESTIONS"))
    ops: list[dict] = []

    if (u := parse_update(obs)) and "INBOX EMPTY" not in obs:
        svc, attr, val = u
        ops.append({"op": "set_fact", "key": f"{svc}.{attr}", "value": val, "source": "inbox"})
    if "INBOX EMPTY" in obs:
        for qid, attr, svc in QUESTION_RE.findall(obs):
            ops.append({"op": "add_question", "text": f"{qid}: current {attr} of {svc}"})
            open_qs.append((qid, attr, svc))
            if f"{svc}.{attr}" in facts:
                ops.append({"op": "pin", "key": f"{svc}.{attr}"})
    if obs.startswith("recall("):
        # Newest archived value for the key we asked about wins.
        best: dict[str, tuple[int, str]] = {}
        for step, key, val in re.findall(r"\[step (\d+)\] (?:evicted|dropped)_fact ([\w.-]+) = \"([^\"]*)\"", obs):
            # never overwrite a live fact: it was set after any eviction, so it is newer
            if key not in facts and int(step) >= best.get(key, (-1, ""))[0]:
                best[key] = (int(step), val)
        for key, (_, val) in best.items():
            ops.append({"op": "set_fact", "key": key, "value": val, "source": "recall", "pin": True})
            facts[key] = val

    if open_qs:
        missing = [f"{s}.{a}" for _, a, s in open_qs if f"{s}.{a}" not in facts]
        if missing:
            return json.dumps({"thought": "need archived value", "state_ops": ops,
                               "action": {"tool": "recall", "args": {"query": missing[0], "k": 5, "kinds": ["evicted_fact", "dropped_fact"]}}})
        answer = {qid: facts[f"{s}.{a}"] for qid, a, s in open_qs}
        return json.dumps({"thought": "all answered", "state_ops": ops, "action": {"tool": "finish", "args": {"answer": answer}}})
    return json.dumps({"thought": "keep draining", "state_ops": ops, "action": {"tool": "next_message", "args": {}}})


def naive_policy(system: str, prompt: str) -> str:
    """Perfect reader of the transcript: takes the last mention of each key."""
    qs = [q for q in QUESTION_RE.findall(prompt)]
    if qs and "INBOX EMPTY" in prompt:
        latest: dict[str, str] = {}
        for line in prompt.splitlines():
            if " next_message -> " in line and (u := parse_update(line)):
                latest[f"{u[0]}.{u[1]}"] = u[2]
        answer = {qid: latest.get(f"{s}.{a}", "") for qid, a, s in qs}
        return json.dumps({"thought": "answer", "action": {"tool": "finish", "args": {"answer": answer}}})
    return json.dumps({"thought": "keep draining", "action": {"tool": "next_message", "args": {}}})
