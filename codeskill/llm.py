"""Pluggable LLM client interface.

Every learnable piece of CODESKILL (the skill-manager policy, the rubric
reward judge) talks to an LLM through this narrow interface so the rest of
the codebase never depends on a specific vendor SDK. Two implementations are
provided:

  - `MockLLMClient`: fully deterministic, rule-based stand-in used by the
    test suite and the `demo` CLI command so the whole pipeline is
    exercisable with no API key and no network access.
  - `AnthropicLLMClient`: thin wrapper around the `anthropic` SDK for real
    use. Import of the SDK is deferred so it's optional.
"""
from __future__ import annotations

import json
import re
from typing import Callable, Optional, Protocol


class LLMClient(Protocol):
    """Minimal interface: a system prompt + user prompt in, text out."""

    def complete(self, system: str, prompt: str, *, temperature: float = 0.0) -> str:
        ...


def extract_json(text: str) -> dict:
    """Best-effort extraction of a JSON object from raw LLM output.

    Handles the common case of a fenced ```json ... ``` block as well as
    plain JSON, and raises `ValueError` with the offending text on failure so
    callers can classify/log a bad manager output rather than crash blindly.
    """
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else text.strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        brace_match = re.search(r"\{.*\}", text, re.DOTALL)
        if brace_match:
            try:
                return json.loads(brace_match.group(0))
            except json.JSONDecodeError:
                pass
        raise ValueError(f"could not parse JSON from LLM output: {text!r}")


class MockLLMClient:
    """Deterministic stand-in LLM driven by a set of registered responders.

    A responder is `(predicate, response_fn)`; `complete()` returns the
    first responder whose predicate matches the prompt. This lets tests and
    the demo pipeline script exact, reproducible behavior for each of the
    manager/extractor/judge prompts without any network access.
    """

    def __init__(self) -> None:
        self._responders: list[tuple[Callable[[str, str], bool], Callable[[str, str], str]]] = []
        self.calls: list[tuple[str, str]] = []

    def register(
        self,
        predicate: Callable[[str, str], bool],
        response_fn: Callable[[str, str], str],
    ) -> None:
        self._responders.append((predicate, response_fn))

    def complete(self, system: str, prompt: str, *, temperature: float = 0.0) -> str:
        self.calls.append((system, prompt))
        for predicate, response_fn in self._responders:
            if predicate(system, prompt):
                return response_fn(system, prompt)
        raise LookupError(
            "MockLLMClient has no matching responder for this prompt; "
            "register one with .register(predicate, response_fn)"
        )


class AnthropicLLMClient:
    """Real backend using the `anthropic` Python SDK.

    Kept intentionally thin: this module has no other dependency on the
    `anthropic` package, so environments without it (or without an API key)
    can still import and use the rest of `codeskill` with `MockLLMClient`.
    """

    def __init__(self, model: str = "claude-sonnet-5", api_key: Optional[str] = None, max_tokens: int = 2048):
        import anthropic  # deferred import: optional dependency

        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens

    def complete(self, system: str, prompt: str, *, temperature: float = 0.0) -> str:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in response.content if block.type == "text")
