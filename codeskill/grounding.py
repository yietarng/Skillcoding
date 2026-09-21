"""A lexical-overlap groundedness heuristic -- NOT part of the CODESKILL
paper's prompts (see `codeskill/prompts.py` for what is).

The paper relies entirely on an LLM judge's qualitative "groundedness"
rubric dimension (Figures 10-12) to catch hallucinated rules; it does not
ask the extraction/evolution model itself to cite machine-checkable
evidence. That's a reasonable design for a paper measuring downstream
pass-rate, but a cheap automated check is a useful *additional* safety net
before a candidate skill even reaches the (expensive, LLM-call) rubric
judge -- similar in spirit to what the `codeskill-rebuild` third-party
reconstruction calls a `prompts/runtime/` addition layered on top of the
paper's own `prompts/paper/` prompts.

This module is that addition: a dependency-free lexical check of whether a
skill's `when_to_apply`/`rules` text actually overlaps with the trajectory
steps that were really observed (see `TrajectoryStep.is_grounded`), used to
flag (not silently reject) weakly-grounded candidates.
"""
from __future__ import annotations

from codeskill.schema import Skill, Trajectory


def _tokenize(text: str) -> set[str]:
    return {t.strip(".,:;()[]{}").lower() for t in text.split() if len(t) > 2}


def _grounded_evidence_texts(trajectories: list[Trajectory]) -> list[str]:
    texts = []
    for trajectory in trajectories:
        for step in trajectory.steps:
            if step.is_grounded():
                texts.append(f"{step.assistant_action} {step.tool_result}")
    return texts


def heuristic_groundedness(skill: Skill, trajectories: list[Trajectory]) -> float:
    """Mean, over the skill's `when_to_apply` and each rule, of that
    statement's best lexical overlap with any single observed trajectory
    step. 0.0 if there is no observed evidence at all, or the skill has no
    content to check.

    This is deliberately crude (no embeddings, no NLI) -- it is meant to
    catch outright hallucination (a rule about something never touched by
    any tool call), not to replace the paper's own LLM-judged rubric.
    """
    evidence_texts = _grounded_evidence_texts(trajectories)
    if not evidence_texts:
        return 0.0
    evidence_token_sets = [_tokenize(t) for t in evidence_texts]

    statements = [skill.when_to_apply, *skill.rules]
    statements = [s for s in statements if s.strip()]
    if not statements:
        return 0.0

    per_statement_scores = []
    for statement in statements:
        statement_tokens = _tokenize(statement)
        if not statement_tokens:
            per_statement_scores.append(0.0)
            continue
        best = 0.0
        for evidence_tokens in evidence_token_sets:
            if not evidence_tokens:
                continue
            overlap = len(statement_tokens & evidence_tokens) / len(statement_tokens)
            best = max(best, overlap)
        per_statement_scores.append(best)

    return sum(per_statement_scores) / len(per_statement_scores)


def is_weakly_grounded(skill: Skill, trajectories: list[Trajectory], threshold: float = 0.15) -> bool:
    return heuristic_groundedness(skill, trajectories) < threshold
