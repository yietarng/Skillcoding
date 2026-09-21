import json

from codeskill.extraction import SkillExtractor
from codeskill.llm import MockLLMClient
from codeskill.schema import Trajectory, TrajectoryStep
from codeskill.sft_data import build_warm_start_dataset, export_jsonl, generate_teacher_dataset, load_jsonl, operation_to_dict


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


def teacher_llm_with_op() -> MockLLMClient:
    llm = MockLLMClient()
    llm.register(
        lambda system, prompt: True,
        lambda system, prompt: json.dumps(
            {
                "operations": [
                    {
                        "op_type": "add",
                        "name": "install requests",
                        "description": "pip install the missing module",
                        "steps": ["pip install requests"],
                        "granularity": "event",
                        "cited_step_indices": [1],
                        "rationale": "grounded",
                    }
                ]
            }
        ),
    )
    return llm


def teacher_llm_noop() -> MockLLMClient:
    llm = MockLLMClient()
    llm.register(lambda system, prompt: True, lambda system, prompt: json.dumps({"operations": []}))
    return llm


def test_generate_teacher_dataset_and_build_warm_start_dataset():
    trajectories = [make_trajectory("t1"), make_trajectory("t2")]
    extractor = SkillExtractor(teacher_llm_with_op())

    labeled = generate_teacher_dataset(trajectories, extractor)
    examples = build_warm_start_dataset(labeled)

    assert len(labeled) == 2
    assert len(examples) == 2
    parsed_completion = json.loads(examples[0].completion)
    assert parsed_completion["operations"][0]["op_type"] == "add"
    assert parsed_completion["operations"][0]["cited_step_indices"] == [1]


def test_build_warm_start_dataset_drops_pure_noop_by_default():
    trajectories = [make_trajectory("t1")]
    extractor = SkillExtractor(teacher_llm_noop())

    labeled = generate_teacher_dataset(trajectories, extractor)
    examples = build_warm_start_dataset(labeled)

    assert examples == []


def test_build_warm_start_dataset_keeps_noop_when_requested():
    trajectories = [make_trajectory("t1")]
    extractor = SkillExtractor(teacher_llm_noop())

    labeled = generate_teacher_dataset(trajectories, extractor)
    examples = build_warm_start_dataset(labeled, drop_pure_noop=False)

    assert len(examples) == 1


def test_operation_to_dict_includes_skill_fields():
    trajectories = [make_trajectory("t1")]
    extractor = SkillExtractor(teacher_llm_with_op())
    ops = extractor.propose_operations(trajectories[0])

    d = operation_to_dict(ops[0])

    assert d["op_type"] == "add"
    assert d["name"] == "install requests"
    assert d["cited_step_indices"] == [1]


def test_export_and_load_jsonl_round_trip(tmp_path):
    trajectories = [make_trajectory("t1")]
    extractor = SkillExtractor(teacher_llm_with_op())
    labeled = generate_teacher_dataset(trajectories, extractor)
    examples = build_warm_start_dataset(labeled)

    path = tmp_path / "sft.jsonl"
    export_jsonl(examples, path)
    restored = load_jsonl(path)

    assert len(restored) == len(examples)
    assert restored[0].prompt == examples[0].prompt
    assert restored[0].completion == examples[0].completion
