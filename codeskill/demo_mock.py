"""Deterministic mock LLM wiring shared by the `demo` CLI command and the
integration test. It's generic over *any* trajectory shaped like
`build_extraction_prompt` produces -- it picks a grounded step's assistant
action/tool result straight out of the prompt text rather than hard-coding
per-example answers -- so the demo genuinely exercises the extraction/
grounding/rubric code paths instead of faking their output. When a
trajectory has more than one grounded step (typically diagnose -> fix ->
verify), it prefers the second one, since that's conventionally the actual
corrective action rather than the initial diagnosis.
"""
from __future__ import annotations

import re

from codeskill.extraction import SYSTEM_PROMPT
from codeskill.llm import MockLLMClient
from codeskill.rewards import RUBRIC_SYSTEM_PROMPT

_GROUNDED_STEP_RE = re.compile(
    r"\[(\d+)\] \(OBSERVED\) assistant: (?P<action>.*?)\n(?:      tool_call: .*?\n)?      tool_result: (?P<result>.*?)\n"
)


def _is_extraction_prompt(system: str, prompt: str) -> bool:
    return system == SYSTEM_PROMPT


def _is_rubric_prompt(system: str, prompt: str) -> bool:
    return system == RUBRIC_SYSTEM_PROMPT


def _extraction_response(system: str, prompt: str) -> str:
    matches = list(_GROUNDED_STEP_RE.finditer(prompt + "\n"))
    task_line = next((line for line in prompt.splitlines() if line.startswith("Task: ")), "Task: unknown")
    task_desc = task_line[len("Task: ") :]

    if not matches:
        return '{"operations": []}'
    match = matches[1] if len(matches) > 1 else matches[0]

    cited_index = int(match.group(1))
    action = match.group("action")
    result = match.group("result")

    op = {
        "op_type": "add",
        "name": task_desc[:60],
        "description": f"How to handle: {task_desc}",
        "steps": [action, f"Observed result: {result}"],
        "granularity": "event",
        "cited_step_indices": [cited_index],
        "preconditions": [],
        "tags": ["demo"],
        "rationale": "Extracted from a single grounded action/result pair.",
    }
    return f'{{"operations": [{__import__("json").dumps(op)}]}}'


def _rubric_response(system: str, prompt: str) -> str:
    import json

    return json.dumps(
        {
            "grounding": 0.9,
            "clarity": 0.8,
            "non_redundancy": 0.75,
            "alignment": 0.85,
            "explanation": "Deterministic demo score: steps are directly grounded in cited tool output.",
        }
    )


def build_demo_llm() -> MockLLMClient:
    client = MockLLMClient()
    client.register(_is_extraction_prompt, _extraction_response)
    client.register(_is_rubric_prompt, _rubric_response)
    return client
