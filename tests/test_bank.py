from codeskill.bank import SkillBank
from codeskill.schema import Granularity, Skill


def make_skill(title="install requests", when_to_apply="pip install requests", rules=None) -> Skill:
    return Skill(title=title, granularity=Granularity.EVENT_DRIVEN, when_to_apply=when_to_apply, rules=rules or ["pip install requests"])


def test_add_accepts_new_skill():
    bank = SkillBank()
    accepted, reason = bank.add(make_skill())
    assert accepted is True
    assert reason is None
    assert len(bank) == 1


def test_add_rejects_duplicate_content():
    bank = SkillBank()
    bank.add(make_skill())
    accepted, reason = bank.add(make_skill(title="Install Requests", when_to_apply="Pip Install Requests"))
    assert accepted is False
    assert reason is not None and reason.startswith("duplicate_of:")
    assert len(bank) == 1


def test_replace_bumps_version_and_appends_provenance():
    bank = SkillBank()
    bank.add(make_skill())
    skill = bank.all_skills()[0]

    replacement = make_skill(title="install requests (revised)", rules=["pip install requests", "verify with pip show requests"])
    bank.replace(skill.id, replacement, rationale="merged with similar skill")

    updated = bank.get(skill.id)
    assert updated.version == 2
    assert updated.title == "install requests (revised)"
    assert "verify with pip show requests" in updated.rules
    assert updated.id == skill.id  # identity preserved


def test_drop_marks_inactive_and_logs_exclusion():
    bank = SkillBank()
    bank.add(make_skill())
    skill_id = bank.all_skills()[0].id

    bank.drop(skill_id, "unhelpful")

    assert bank.get(skill_id).active is False
    assert len(bank) == 0
    reasons = [e.reason for e in bank.exclusions()]
    assert any("dropped:unhelpful" in r for r in reasons)


def test_retrieve_ranks_by_lexical_overlap():
    bank = SkillBank()
    bank.add(make_skill(title="install requests", when_to_apply="fix ModuleNotFoundError requests", rules=["pip install requests"]))
    bank.add(make_skill(title="add fixture", when_to_apply="add a pytest fixture", rules=["edit conftest.py"]))

    results = bank.retrieve("ModuleNotFoundError requests missing package", top_k=2)

    assert results[0].skill.title == "install requests"
    assert results[0].score >= results[1].score


def test_retrieve_similar_queries_with_full_skill_content():
    bank = SkillBank()
    bank.add(make_skill(title="install requests", when_to_apply="fix ModuleNotFoundError requests", rules=["pip install requests"]))
    bank.add(make_skill(title="add fixture", when_to_apply="add a pytest fixture", rules=["edit conftest.py"]))

    candidate = make_skill(title="install requests candidate", when_to_apply="fix requests import error", rules=["pip install requests"])
    results = bank.retrieve_similar(candidate, top_k=2)

    assert results[0].skill.title == "install requests"


def test_compact_drops_low_success_rate_skills():
    bank = SkillBank()
    bank.add(make_skill())
    skill = bank.all_skills()[0]
    for _ in range(5):
        skill.record_usage(False)

    dropped = bank.compact(min_uses=3, min_success_rate=0.2)

    assert dropped == [skill.id]
    assert len(bank) == 0


def test_compact_keeps_skills_below_usage_threshold():
    bank = SkillBank()
    bank.add(make_skill())
    skill = bank.all_skills()[0]
    skill.record_usage(False)  # only 1 use, below min_uses

    dropped = bank.compact(min_uses=3, min_success_rate=0.2)

    assert dropped == []
    assert len(bank) == 1


def test_advance_round_snapshots_state():
    bank = SkillBank()
    bank.add(make_skill())
    snapshot = bank.advance_round()

    assert snapshot.round_num == 0
    assert bank.current_round == 1
    assert len(snapshot.skills) == 1
    assert bank.snapshot_at(0) is snapshot


def test_json_round_trip(tmp_path):
    bank = SkillBank()
    bank.add(make_skill())
    path = tmp_path / "bank.json"
    bank.save(path)

    restored = SkillBank.load(path)

    assert len(restored) == 1
    assert restored.all_skills()[0].title == "install requests"
