import json

from codeskill.bank import SkillBank
from codeskill.extraction import EventSkillExtractor, SkillEvolver, SkillMaintainer
from codeskill.llm import MockLLMClient
from codeskill.manager import SkillManagerPolicy
from codeskill.prompts import EVENT_EXTRACTION_SYSTEM_PROMPT, SKILL_EVOLUTION_SYSTEM_PROMPT, SKILL_MAINTENANCE_SYSTEM_PROMPT
from codeskill.schema import Granularity, Skill, Trajectory, TrajectoryStep


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


GENERATED_SKILL = {
    "title": "install requests",
    "granularity": "event-driven",
    "when_to_apply": "ModuleNotFoundError for requests",
    "rules": ["pip install requests"],
}


def _manager_with(bank: SkillBank, *, generate=True, maintenance_response: dict) -> SkillManagerPolicy:
    llm = MockLLMClient()

    def extraction_responder(system, prompt):
        if generate:
            return json.dumps({"action": "generate", "skill": GENERATED_SKILL})
        return json.dumps({"action": "skip", "reason": "no strong reusable event"})

    llm.register(lambda system, prompt: system == EVENT_EXTRACTION_SYSTEM_PROMPT, extraction_responder)
    llm.register(lambda system, prompt: system == SKILL_MAINTENANCE_SYSTEM_PROMPT, lambda system, prompt: json.dumps(maintenance_response))
    return SkillManagerPolicy.from_llm(llm, bank)


def test_process_event_trajectory_adds_skill_to_bank():
    manager = _manager_with(SkillBank(), maintenance_response={"action": "add", "reason": "distinct"})

    outcomes = manager.process_event_trajectory(make_trajectory())

    assert [o.stage for o in outcomes] == ["extract_event", "maintain_add"]
    assert all(o.applied for o in outcomes)
    assert len(manager.bank) == 1


def test_process_event_trajectory_skip_short_circuits_maintenance():
    manager = _manager_with(SkillBank(), generate=False, maintenance_response={"action": "add", "reason": "unused"})

    outcomes = manager.process_event_trajectory(make_trajectory())

    assert len(outcomes) == 1
    assert outcomes[0].stage == "extract_event"
    assert outcomes[0].applied is False
    assert len(manager.bank) == 0


def test_process_event_trajectory_drop_leaves_bank_empty():
    manager = _manager_with(SkillBank(), maintenance_response={"action": "drop", "reason": "redundant"})

    outcomes = manager.process_event_trajectory(make_trajectory())

    assert outcomes[-1].stage == "maintain_drop"
    assert outcomes[-1].applied is False
    assert len(manager.bank) == 0


def test_process_event_trajectory_merge_replaces_existing_skill():
    bank = SkillBank()
    existing = Skill(
        title="install requests",
        granularity=Granularity.EVENT_DRIVEN,
        when_to_apply="ModuleNotFoundError for requests",
        rules=["pip install requests"],
    )
    bank.add(existing)

    manager = _manager_with(
        bank,
        maintenance_response={
            "action": "merge",
            "merge_target_skill_id": existing.id,
            "reason": "same capability identity",
            "skill": {
                "title": "install requests (merged)",
                "granularity": "event-driven",
                "when_to_apply": "ModuleNotFoundError for requests",
                "rules": ["pip install requests", "verify with pip show"],
            },
        },
    )

    outcomes = manager.process_event_trajectory(make_trajectory())

    assert outcomes[-1].stage == "maintain_merge"
    assert outcomes[-1].applied is True
    updated = bank.get(existing.id)
    assert updated.title == "install requests (merged)"
    assert "verify with pip show" in updated.rules
    assert len(bank) == 1  # still one active skill, not two


def test_run_round_advances_bank_round_and_reports_sizes():
    manager = _manager_with(SkillBank(), maintenance_response={"action": "add", "reason": "distinct"})

    report = manager.run_round([make_trajectory()])

    assert report.bank_size_before == 0
    assert report.bank_size_after == 1
    assert manager.bank.current_round == 1


def test_process_evolution_replaces_target_skill():
    bank = SkillBank()
    existing = Skill(
        title="install requests",
        granularity=Granularity.EVENT_DRIVEN,
        when_to_apply="ModuleNotFoundError for requests",
        rules=["pip install requests"],
    )
    bank.add(existing)

    llm = MockLLMClient()
    llm.register(
        lambda system, prompt: system == SKILL_EVOLUTION_SYSTEM_PROMPT,
        lambda system, prompt: json.dumps(
            {
                "action": "evolve",
                "target_skill_id": existing.id,
                "reason": "new caution",
                "skill": {
                    "title": "install requests",
                    "granularity": "event-driven",
                    "when_to_apply": "ModuleNotFoundError for requests",
                    "rules": ["pip install requests", "verify with pip show requests"],
                },
            }
        ),
    )
    manager = SkillManagerPolicy(
        bank=bank,
        event_extractor=EventSkillExtractor(llm),
        evolver=SkillEvolver(llm),
        maintainer=SkillMaintainer(llm),
    )

    outcomes = manager.process_evolution([existing], make_trajectory())

    assert outcomes[0].stage == "evolve"
    assert outcomes[0].applied is True
    updated = bank.get(existing.id)
    assert "verify with pip show requests" in updated.rules
    assert updated.version == 2


def test_process_evolution_skip_does_not_touch_bank():
    bank = SkillBank()
    existing = Skill(title="install requests", granularity=Granularity.EVENT_DRIVEN, when_to_apply="d", rules=["r"])
    bank.add(existing)

    llm = MockLLMClient()
    llm.register(
        lambda system, prompt: system == SKILL_EVOLUTION_SYSTEM_PROMPT,
        lambda system, prompt: json.dumps({"action": "skip", "reason": "evidence too weak"}),
    )
    manager = SkillManagerPolicy(bank=bank, event_extractor=EventSkillExtractor(llm), evolver=SkillEvolver(llm), maintainer=SkillMaintainer(llm))

    outcomes = manager.process_evolution([existing], make_trajectory())

    assert outcomes[0].applied is False
    assert bank.get(existing.id).version == 1
