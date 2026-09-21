import json

from codeskill.extraction import SkillExtractor, classify_output, parse_operations
from codeskill.llm import MockLLMClient
from codeskill.schema import OperationType, Trajectory, TrajectoryStep


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
    assert classify_output('{"operations": []}') == "looks_json"


def test_parse_operations_rejects_ungrounded_citation():
    traj = make_trajectory()
    raw = json.dumps(
        {
            "operations": [
                {
                    "op_type": "add",
                    "name": "bad skill",
                    "description": "cites a step that was never observed",
                    "steps": ["do the thing"],
                    "granularity": "event",
                    "cited_step_indices": [99],
                    "rationale": "hallucinated",
                }
            ]
        }
    )
    ops = parse_operations(raw, traj)
    assert len(ops) == 1
    assert ops[0].op_type == OperationType.NOOP
    assert "ungrounded_citation" in ops[0].exclusion_reason


def test_parse_operations_accepts_grounded_add():
    traj = make_trajectory()
    raw = json.dumps(
        {
            "operations": [
                {
                    "op_type": "add",
                    "name": "install missing dependency",
                    "description": "pip install the missing module",
                    "steps": ["pip install requests"],
                    "granularity": "event",
                    "cited_step_indices": [1],
                    "rationale": "grounded in step 1",
                }
            ]
        }
    )
    ops = parse_operations(raw, traj)
    assert len(ops) == 1
    op = ops[0]
    assert op.op_type == OperationType.ADD
    assert op.skill is not None
    assert op.skill.provenance[0].step_indices == [1]


def test_parse_operations_handles_malformed_output():
    traj = make_trajectory()
    ops = parse_operations("not json at all", traj)
    assert len(ops) == 1
    assert ops[0].op_type == OperationType.NOOP
    assert ops[0].exclusion_reason.startswith("malformed_output")


def test_skill_extractor_uses_llm_client():
    traj = make_trajectory()
    llm = MockLLMClient()
    llm.register(
        lambda system, prompt: True,
        lambda system, prompt: json.dumps(
            {
                "operations": [
                    {
                        "op_type": "add",
                        "name": "n",
                        "description": "d",
                        "steps": ["s"],
                        "granularity": "event",
                        "cited_step_indices": [0],
                        "rationale": "r",
                    }
                ]
            }
        ),
    )
    extractor = SkillExtractor(llm)
    ops = extractor.propose_operations(traj)
    assert len(ops) == 1
    assert ops[0].op_type == OperationType.ADD
    assert len(llm.calls) == 1
