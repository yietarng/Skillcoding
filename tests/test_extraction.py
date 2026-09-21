import json

from codeskill.extraction import (
    EventSkillExtractor,
    SkillEvolver,
    SkillMaintainer,
    TaskSkillExtractor,
    parse_evolution_response,
    parse_extraction_response,
    parse_maintenance_response,
)
from codeskill.llm import MockLLMClient, classify_output
from codeskill.schema import Decision, Granularity, Skill, Trajectory, TrajectoryStep


def make_trajectory() -> Trajectory:
    return Trajectory(
        task_id="t1",
        task_description="fix ModuleNotFoundError for requests",
        steps=[
            TrajectoryStep(0, "run tests", tool_name="shell", tool_result="ModuleNotFoundError", observed=True),
            TrajectoryStep(1, "pip install requests", tool_name="shell", tool_result="installed", observed=True),
        ],
        success=True,
    )


def test_classify_output():
    assert classify_output("") == "empty"
    assert classify_output("no json here") == "not_json"
    assert classify_output('{"action": "skip"}') == "looks_json"


def test_parse_extraction_response_generate():
    raw = json.dumps(
        {
            "action": "generate",
            "skill": {
                "title": "install missing dependency",
                "granularity": "event-driven",
                "when_to_apply": "ModuleNotFoundError for a pip package",
                "rules": ["pip install the missing module", "re-run the tests"],
            },
        }
    )
    outcome = parse_extraction_response(raw)
    assert outcome.decision == Decision.GENERATE
    assert outcome.skill.title == "install missing dependency"
    assert outcome.skill.granularity == Granularity.EVENT_DRIVEN


def test_parse_extraction_response_skip():
    raw = json.dumps({"action": "skip", "reason": "too task-specific"})
    outcome = parse_extraction_response(raw)
    assert outcome.decision == Decision.SKIP
    assert outcome.skill is None
    assert outcome.reason == "too task-specific"


def test_parse_extraction_response_rejects_invalid_granularity():
    raw = json.dumps({"action": "generate", "skill": {"title": "x", "granularity": "bogus", "when_to_apply": "y", "rules": []}})
    outcome = parse_extraction_response(raw)
    assert outcome.decision == Decision.SKIP
    assert "invalid_granularity" in outcome.reason


def test_parse_extraction_response_handles_malformed_output():
    outcome = parse_extraction_response("not json at all")
    assert outcome.decision == Decision.SKIP
    assert outcome.reason.startswith("malformed_output")


def test_parse_evolution_response_evolve():
    raw = json.dumps(
        {
            "action": "evolve",
            "target_skill_id": "skill_abc123",
            "reason": "new caution discovered",
            "skill": {"title": "install missing dependency", "granularity": "event-driven", "when_to_apply": "x", "rules": ["y"]},
        }
    )
    outcome = parse_evolution_response(raw)
    assert outcome.decision == Decision.EVOLVE
    assert outcome.target_skill_id == "skill_abc123"
    assert outcome.skill.title == "install missing dependency"


def test_parse_maintenance_response_add():
    outcome = parse_maintenance_response(json.dumps({"action": "add", "reason": "distinct and reusable"}))
    assert outcome.decision == Decision.ADD


def test_parse_maintenance_response_merge():
    raw = json.dumps(
        {
            "action": "merge",
            "merge_target_skill_id": "skill_xyz789",
            "reason": "same capability identity",
            "skill": {"title": "merged", "granularity": "event-driven", "when_to_apply": "x", "rules": ["y"]},
        }
    )
    outcome = parse_maintenance_response(raw)
    assert outcome.decision == Decision.MERGE
    assert outcome.merge_target_skill_id == "skill_xyz789"


def test_parse_maintenance_response_drop():
    outcome = parse_maintenance_response(json.dumps({"action": "drop", "reason": "redundant"}))
    assert outcome.decision == Decision.DROP


def _llm_returning(payload: dict) -> MockLLMClient:
    llm = MockLLMClient()
    llm.register(lambda system, prompt: True, lambda system, prompt: json.dumps(payload))
    return llm


def test_event_skill_extractor_attaches_provenance():
    traj = make_trajectory()
    llm = _llm_returning(
        {
            "action": "generate",
            "skill": {"title": "n", "granularity": "event-driven", "when_to_apply": "d", "rules": ["r"]},
        }
    )
    outcome = EventSkillExtractor(llm).propose(traj)
    assert outcome.decision == Decision.GENERATE
    assert outcome.skill.provenance[0].trajectory_id == traj.id
    assert len(llm.calls) == 1


def test_task_skill_extractor_attaches_provenance_per_trajectory():
    trajs = [make_trajectory(), make_trajectory()]
    llm = _llm_returning(
        {"action": "generate", "skill": {"title": "n", "granularity": "general", "when_to_apply": "d", "rules": ["r"]}}
    )
    outcome = TaskSkillExtractor(llm).propose(trajs)
    assert outcome.decision == Decision.GENERATE
    assert len(outcome.skill.provenance) == 2


def test_skill_evolver_propose():
    traj = make_trajectory()
    existing = Skill(title="e", granularity=Granularity.EVENT_DRIVEN, when_to_apply="d", rules=["r"])
    llm = _llm_returning(
        {
            "action": "evolve",
            "target_skill_id": existing.id,
            "reason": "r",
            "skill": {"title": "e", "granularity": "event-driven", "when_to_apply": "d2", "rules": ["r", "r2"]},
        }
    )
    outcome = SkillEvolver(llm).propose([existing], traj)
    assert outcome.decision == Decision.EVOLVE
    assert outcome.target_skill_id == existing.id


def test_skill_maintainer_decide():
    candidate = Skill(title="c", granularity=Granularity.EVENT_DRIVEN, when_to_apply="d", rules=["r"])
    llm = _llm_returning({"action": "add", "reason": "distinct"})
    outcome = SkillMaintainer(llm).decide(candidate, [])
    assert outcome.decision == Decision.ADD
