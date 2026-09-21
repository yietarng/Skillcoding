from codeskill.schema import Granularity, Skill, Trajectory, TrajectoryStep


def make_trajectory() -> Trajectory:
    return Trajectory(
        task_id="t1",
        task_description="fix the bug",
        steps=[
            TrajectoryStep(0, "look at the error", tool_name="shell", tool_result="Traceback...", observed=True),
            TrajectoryStep(1, "guess a fix", tool_name=None, tool_result=None, observed=False),
            TrajectoryStep(2, "apply the fix", tool_name="edit", tool_result="patched", observed=True),
        ],
        success=True,
    )


def test_grounded_step_indices():
    traj = make_trajectory()
    assert traj.grounded_step_indices() == [0, 2]


def test_is_grounded_span_accepts_only_observed_steps():
    traj = make_trajectory()
    assert traj.is_grounded_span([0, 2]) is True
    assert traj.is_grounded_span([0, 1]) is False
    assert traj.is_grounded_span([]) is False


def test_trajectory_round_trip():
    traj = make_trajectory()
    restored = Trajectory.from_dict(traj.to_dict())
    assert restored.task_description == traj.task_description
    assert [s.index for s in restored.steps] == [0, 1, 2]
    assert restored.grounded_step_indices() == [0, 2]


def test_skill_content_key_dedup_ignores_case_and_order_of_incidentals():
    a = Skill(name="Install requests", description="pip install requests", steps=["pip install requests"], granularity=Granularity.EVENT)
    b = Skill(name="install REQUESTS", description="Pip Install Requests", steps=["pip install requests"], granularity=Granularity.EVENT)
    assert a.content_key() == b.content_key()


def test_skill_success_rate_and_usage():
    skill = Skill(name="s", description="d", steps=["x"], granularity=Granularity.EVENT)
    assert skill.success_rate() == 0.0
    skill.record_usage(True)
    skill.record_usage(False)
    assert skill.usage_count == 2
    assert skill.success_count == 1
    assert skill.success_rate() == 0.5


def test_skill_round_trip():
    skill = Skill(name="s", description="d", steps=["x", "y"], granularity=Granularity.TASK, tags=["a"])
    restored = Skill.from_dict(skill.to_dict())
    assert restored.name == skill.name
    assert restored.granularity == Granularity.TASK
    assert restored.steps == ["x", "y"]
