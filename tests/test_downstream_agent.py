from codeskill.bank import SkillBank
from codeskill.downstream_agent import FrozenDownstreamAgent, SolveResult
from codeskill.schema import Granularity, Provenance, Skill


def make_skill(title, granularity, when_to_apply="fix ModuleNotFoundError requests", trajectory_id=None) -> Skill:
    provenance = [Provenance(trajectory_id=trajectory_id, step_indices=[])] if trajectory_id else []
    return Skill(title=title, granularity=granularity, when_to_apply=when_to_apply, rules=["pip install requests"], provenance=provenance)


def test_attempt_retrieves_both_granularities_with_independent_top_k():
    bank = SkillBank()
    # 2 task-level and 2 event-driven skills, all equally relevant to the query.
    for i in range(2):
        bank.add(make_skill(f"general-{i}", Granularity.GENERAL))
        bank.add(make_skill(f"event-{i}", Granularity.EVENT_DRIVEN))

    seen_skills = []

    def solve(task_description, skills):
        seen_skills.extend(skills)
        return SolveResult(success=True, trace="ok")

    # top_k=1 per granularity should still surface one of each kind, not
    # let one granularity's skills crowd the other out of a shared budget.
    agent = FrozenDownstreamAgent(bank, solve, top_k=1)
    agent.attempt("ModuleNotFoundError requests")

    granularities = {s.granularity for s in seen_skills}
    assert granularities == {Granularity.GENERAL, Granularity.EVENT_DRIVEN}
    assert len(seen_skills) == 2


def test_attempt_excludes_same_instance_skills():
    bank = SkillBank()
    bank.add(make_skill("from this instance", Granularity.EVENT_DRIVEN, trajectory_id="traj-self"))
    bank.add(make_skill("from another instance", Granularity.EVENT_DRIVEN, trajectory_id="traj-other"))

    seen_skills = []

    def solve(task_description, skills):
        seen_skills.extend(skills)
        return SolveResult(success=True, trace="ok")

    agent = FrozenDownstreamAgent(bank, solve, top_k=5)
    agent.attempt("ModuleNotFoundError requests", exclude_trajectory_ids={"traj-self"})

    titles = {s.title for s in seen_skills}
    assert titles == {"from another instance"}


def test_attempt_records_usage_by_default():
    bank = SkillBank()
    skill = make_skill("s", Granularity.EVENT_DRIVEN)
    bank.add(skill)
    agent = FrozenDownstreamAgent(bank, lambda t, s: SolveResult(success=True), top_k=5)

    agent.attempt("ModuleNotFoundError requests")

    assert skill.usage_count == 1
    assert skill.success_count == 1


def test_attempt_can_skip_recording_usage():
    bank = SkillBank()
    skill = make_skill("s", Granularity.EVENT_DRIVEN)
    bank.add(skill)
    agent = FrozenDownstreamAgent(bank, lambda t, s: SolveResult(success=True), top_k=5)

    agent.attempt("ModuleNotFoundError requests", record_usage=False)

    assert skill.usage_count == 0
