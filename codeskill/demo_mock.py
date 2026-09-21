"""Deterministic mock LLM wiring shared by the `demo` CLI command and the
test suite. It's generic over *any* trajectory shaped like
`codeskill.extraction`'s prompt builders produce -- it picks a grounded
step's assistant action/tool result straight out of the prompt text rather
than hard-coding per-example answers -- so the demo genuinely exercises the
extraction/maintenance code paths (including the real appendix system
prompts) instead of faking their output. When a trajectory has more than
one grounded step (typically diagnose -> fix -> verify), it prefers the
second one, since that's conventionally the actual corrective action rather
than the initial diagnosis.
"""
from __future__ import annotations

import json
import re

from codeskill.llm import MockLLMClient
from codeskill.prompts import (
    EVENT_EXTRACTION_SYSTEM_PROMPT,
    SKILL_EVOLUTION_SYSTEM_PROMPT,
    SKILL_MAINTENANCE_SYSTEM_PROMPT,
    TASK_EXTRACTION_SYSTEM_PROMPT,
)

_GROUNDED_STEP_RE = re.compile(
    r"\[(\d+)\] \(observed\) action: (?P<action>.*?)\n(?:        tool: .*?\n)?        result: (?P<result>.*?)\n"
)
_TASK_CONTEXT_RE = re.compile(r"Task context: (?P<desc>.*)")


def _pick_grounded_step(prompt: str) -> tuple[int, str, str] | None:
    matches = list(_GROUNDED_STEP_RE.finditer(prompt + "\n"))
    if not matches:
        return None
    match = matches[1] if len(matches) > 1 else matches[0]
    return int(match.group(1)), match.group("action"), match.group("result")


def _task_desc(prompt: str) -> str:
    match = _TASK_CONTEXT_RE.search(prompt)
    return match.group("desc") if match else "unknown task"


def _extraction_response(granularity: str) -> callable:
    def responder(system: str, prompt: str) -> str:
        picked = _pick_grounded_step(prompt)
        if picked is None:
            return json.dumps({"action": "skip", "reason": "no grounded evidence found"})
        _, action, result = picked
        task_desc = _task_desc(prompt)
        return json.dumps(
            {
                "action": "generate",
                "skill": {
                    "title": task_desc[:60],
                    "granularity": granularity,
                    # Deliberately just the task text, not a templated phrase
                    # like "When encountering a situation like: ..." -- shared
                    # boilerplate wording would inflate lexical overlap between
                    # otherwise-unrelated skills once `_maintenance_response`
                    # compares them below.
                    "when_to_apply": task_desc,
                    "rules": [action, f"Expect/confirm: {result}"],
                },
            }
        )

    return responder


def _tokenize(text: str) -> set[str]:
    return {t.strip(".,:;()[]{}'\"").lower() for t in text.split() if len(t) > 2}


def _jaccard(a: str, b: str) -> float:
    tokens_a, tokens_b = _tokenize(a), _tokenize(b)
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def _evolution_response(system: str, prompt: str) -> str:
    header, _, _ = prompt.partition("Task context:")
    id_match = re.search(r'"id":\s*"([^"]+)"', header)
    picked = _pick_grounded_step(prompt)
    if id_match is None or picked is None:
        return json.dumps({"action": "skip", "reason": "no clearly aligned existing skill"})
    target_id = id_match.group(1)
    _, action, result = picked
    return json.dumps(
        {
            "action": "evolve",
            "target_skill_id": target_id,
            "reason": "new trajectory reveals an additional case",
            "skill": {
                "title": "revised skill",
                "granularity": "event-driven",
                "when_to_apply": "When encountering this recurring situation",
                "rules": [action, f"Expect/confirm: {result}"],
            },
        }
    )


_MERGE_OVERLAP_THRESHOLD = 0.3


def _maintenance_response(system: str, prompt: str) -> str:
    candidate_section, _, retrieved_section = prompt.partition("Retrieved similar skills:")
    candidate_json = candidate_section.split("Candidate skill:\n  ", 1)[-1].strip()
    candidate = json.loads(candidate_json)

    retrieved = [json.loads(line.strip()) for line in retrieved_section.strip().splitlines() if line.strip()]

    # Demo heuristic standing in for the real judgment call Figure 9 asks
    # for ("share the same capability identity"): only propose a merge when
    # the candidate's `when_to_apply` actually overlaps a retrieved skill's,
    # not merely because *some* similar-ish skill was retrieved -- bank
    # retrieval returns its top-k regardless of how weak the match is.
    best = max(retrieved, key=lambda s: _jaccard(candidate["when_to_apply"], s["when_to_apply"]), default=None)
    if best is None or _jaccard(candidate["when_to_apply"], best["when_to_apply"]) < _MERGE_OVERLAP_THRESHOLD:
        return json.dumps({"action": "add", "reason": "distinct from retrieved skills"})

    merged_rules = list(dict.fromkeys([*best.get("rules", []), *candidate.get("rules", [])]))
    return json.dumps(
        {
            "action": "merge",
            "merge_target_skill_id": best["id"],
            "reason": "candidate shares the same capability identity as a retrieved skill",
            "skill": {
                "title": best["title"],
                "granularity": best["granularity"],
                "when_to_apply": best["when_to_apply"],
                "rules": merged_rules,
            },
        }
    )


def build_demo_llm() -> MockLLMClient:
    client = MockLLMClient()
    client.register(lambda system, prompt: system == EVENT_EXTRACTION_SYSTEM_PROMPT, _extraction_response("event-driven"))
    client.register(lambda system, prompt: system == TASK_EXTRACTION_SYSTEM_PROMPT, _extraction_response("general"))
    client.register(lambda system, prompt: system == SKILL_EVOLUTION_SYSTEM_PROMPT, _evolution_response)
    client.register(lambda system, prompt: system == SKILL_MAINTENANCE_SYSTEM_PROMPT, _maintenance_response)
    return client
