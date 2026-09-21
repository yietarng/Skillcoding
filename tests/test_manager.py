import json

from codeskill.bank import SkillBank
from codeskill.extraction import SkillExtractor
from codeskill.llm import MockLLMClient
from codeskill.manager import SkillManagerPolicy
from codeskill.schema import OperationType, Trajectory, TrajectoryStep


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


def _llm_returning_add(name="install requests", cited=(1,)):
    llm = MockLLMClient()
    llm.register(
        lambda system, prompt: True,
        lambda system, prompt: json.dumps(
            {
                "operations": [
                    {
                        "op_type": "add",
                        "name": name,
                        "description": "pip install the missing dependency",
                        "steps": ["pip install requests"],
                        "granularity": "event",
                        "cited_step_indices": list(cited),
                        "rationale": "grounded",
                    }
                ]
            }
        ),
    )
    return llm


def test_process_trajectory_adds_skill_to_bank():
    bank = SkillBank()
    manager = SkillManagerPolicy(SkillExtractor(_llm_returning_add()), bank)

    outcomes = manager.process_trajectory(make_trajectory())

    assert len(outcomes) == 1
    assert outcomes[0].applied is True
    assert len(bank) == 1


def test_run_round_advances_bank_round_and_reports_sizes():
    bank = SkillBank()
    manager = SkillManagerPolicy(SkillExtractor(_llm_returning_add()), bank)

    report = manager.run_round([make_trajectory()])

    assert report.bank_size_before == 0
    assert report.bank_size_after == 1
    assert bank.current_round == 1


def test_run_round_second_trajectory_with_same_skill_is_deduplicated():
    bank = SkillBank()
    manager = SkillManagerPolicy(SkillExtractor(_llm_returning_add()), bank)

    manager.run_round([make_trajectory("t1")])
    report2 = manager.run_round([make_trajectory("t2")])

    # second identical skill is a content duplicate, so the bank does not grow
    assert report2.bank_size_after == 1
    outcome = report2.outcomes[0]
    assert outcome.applied is False
    assert "duplicate_of" in outcome.detail


def test_apply_operation_drop_deactivates_target():
    bank = SkillBank()
    manager = SkillManagerPolicy(SkillExtractor(_llm_returning_add()), bank)
    manager.process_trajectory(make_trajectory())
    skill_id = bank.all_skills()[0].id

    from codeskill.schema import SkillOperation

    outcome = manager.apply_operation(SkillOperation(op_type=OperationType.DROP, target_skill_id=skill_id, rationale="bad"))

    assert outcome.applied is True
    assert bank.get(skill_id).active is False
