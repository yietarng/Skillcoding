import json

from codeskill.extraction import EventSkillExtractor, SkillMaintainer
from codeskill.llm import MockLLMClient
from codeskill.schema import Granularity, Skill, Trajectory, TrajectoryStep
from codeskill.sft_data import build_event_extraction_dataset, build_maintenance_dataset, export_jsonl, load_jsonl


def make_trajectory(task_id="t1") -> Trajectory:
    return Trajectory(
        task_id=task_id,
        task_description="fix ModuleNotFoundError for requests",
        steps=[
            TrajectoryStep(0, "run tests", tool_name="shell", tool_result="ModuleNotFoundError", observed=True),
            TrajectoryStep(1, "pip install requests", tool_name="shell", tool_result="installed", observed=True),
        ],
        success=True,
    )


def teacher_llm_with_generate() -> MockLLMClient:
    llm = MockLLMClient()
    llm.register(
        lambda system, prompt: True,
        lambda system, prompt: json.dumps(
            {
                "action": "generate",
                "skill": {
                    "title": "install requests",
                    "granularity": "event-driven",
                    "when_to_apply": "ModuleNotFoundError for requests",
                    "rules": ["pip install requests"],
                },
            }
        ),
    )
    return llm


def teacher_llm_skip() -> MockLLMClient:
    llm = MockLLMClient()
    llm.register(lambda system, prompt: True, lambda system, prompt: json.dumps({"action": "skip", "reason": "too local"}))
    return llm


def test_build_event_extraction_dataset_includes_generate_examples():
    trajectories = [make_trajectory("t1"), make_trajectory("t2")]
    teacher = EventSkillExtractor(teacher_llm_with_generate())

    examples = build_event_extraction_dataset(trajectories, teacher)

    assert len(examples) == 2
    completion = json.loads(examples[0].completion)
    assert completion["action"] == "generate"
    assert completion["skill"]["title"] == "install requests"


def test_build_event_extraction_dataset_drops_pure_skip_by_default():
    trajectories = [make_trajectory("t1")]
    teacher = EventSkillExtractor(teacher_llm_skip())

    examples = build_event_extraction_dataset(trajectories, teacher)

    assert examples == []


def test_build_event_extraction_dataset_keeps_skip_when_requested():
    trajectories = [make_trajectory("t1")]
    teacher = EventSkillExtractor(teacher_llm_skip())

    examples = build_event_extraction_dataset(trajectories, teacher, drop_pure_skip=False)

    assert len(examples) == 1


def test_build_maintenance_dataset_keeps_drop_by_default():
    llm = MockLLMClient()
    llm.register(lambda system, prompt: True, lambda system, prompt: json.dumps({"action": "drop", "reason": "redundant"}))
    candidate = Skill(title="c", granularity=Granularity.EVENT_DRIVEN, when_to_apply="d", rules=["r"])
    existing = Skill(title="e", granularity=Granularity.EVENT_DRIVEN, when_to_apply="d", rules=["r"])

    examples = build_maintenance_dataset([(candidate, [existing])], SkillMaintainer(llm))

    assert len(examples) == 1
    assert json.loads(examples[0].completion)["action"] == "drop"


def test_export_and_load_jsonl_round_trip(tmp_path):
    trajectories = [make_trajectory("t1")]
    teacher = EventSkillExtractor(teacher_llm_with_generate())
    examples = build_event_extraction_dataset(trajectories, teacher)

    path = tmp_path / "sft.jsonl"
    export_jsonl(examples, path)
    restored = load_jsonl(path)

    assert len(restored) == len(examples)
    assert restored[0].prompt == examples[0].prompt
    assert restored[0].completion == examples[0].completion
