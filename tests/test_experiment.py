"""Tests for ExperimentSkill - autonomous A/B testing."""
import pytest
import asyncio
from singularity.skills.experiment import (
    ExperimentSkill, Experiment, Variant, ExperimentStatus,
    _experiment_id, _wilson_lower_bound,
)


@pytest.fixture
def skill(tmp_path):
    s = ExperimentSkill()
    s.DATA_DIR = tmp_path / "experiments"
    return s


@pytest.fixture
def experiment_params():
    return {
        "name": "prompt-style",
        "hypothesis": "Concise prompts produce better results than verbose ones",
        "metric": "success_rate",
        "variants": [
            {"name": "concise", "description": "Short, direct prompts", "config": {"style": "concise"}},
            {"name": "verbose", "description": "Detailed, explanatory prompts", "config": {"style": "verbose"}},
        ],
        "min_trials": 3,
    }


def test_experiment_id_deterministic():
    assert _experiment_id("test") == _experiment_id("test")
    assert _experiment_id("a") != _experiment_id("b")


def test_wilson_lower_bound():
    assert _wilson_lower_bound(0, 0) == 0.0
    assert _wilson_lower_bound(10, 10) > _wilson_lower_bound(1, 1)
    assert 0 < _wilson_lower_bound(5, 10) < 0.5


@pytest.mark.asyncio
async def test_create_experiment(skill, experiment_params):
    result = await skill.execute("create", experiment_params)
    assert result.success
    assert result.data["name"] == "prompt-style"
    assert len(result.data["variants"]) == 2


@pytest.mark.asyncio
async def test_create_requires_two_variants(skill):
    result = await skill.execute("create", {
        "name": "bad", "hypothesis": "x", "metric": "success_rate",
        "variants": [{"name": "only_one"}],
    })
    assert not result.success


@pytest.mark.asyncio
async def test_start_experiment(skill, experiment_params):
    r = await skill.execute("create", experiment_params)
    exp_id = r.data["experiment_id"]
    r2 = await skill.execute("start", {"experiment_id": exp_id})
    assert r2.success
    assert r2.data["status"] == "running"


@pytest.mark.asyncio
async def test_pick_variant(skill, experiment_params):
    r = await skill.execute("create", experiment_params)
    exp_id = r.data["experiment_id"]
    await skill.execute("start", {"experiment_id": exp_id})
    r2 = await skill.execute("pick_variant", {"experiment_id": exp_id})
    assert r2.success
    assert r2.data["variant"] in ("concise", "verbose")


@pytest.mark.asyncio
async def test_record_outcome(skill, experiment_params):
    r = await skill.execute("create", experiment_params)
    exp_id = r.data["experiment_id"]
    await skill.execute("start", {"experiment_id": exp_id})
    r2 = await skill.execute("record_outcome", {
        "experiment_id": exp_id, "variant": "concise",
        "success": True, "score": 0.9, "cost": 0.01, "duration": 1.5,
    })
    assert r2.success
    assert r2.data["trials"] == 1
    assert r2.data["success_rate"] == 1.0


@pytest.mark.asyncio
async def test_full_experiment_lifecycle(skill, experiment_params):
    """Test create -> start -> record -> analyze -> conclude."""
    r = await skill.execute("create", experiment_params)
    exp_id = r.data["experiment_id"]
    await skill.execute("start", {"experiment_id": exp_id})

    # Record outcomes - concise wins
    for _ in range(4):
        await skill.execute("record_outcome", {
            "experiment_id": exp_id, "variant": "concise", "success": True,
        })
        await skill.execute("record_outcome", {
            "experiment_id": exp_id, "variant": "verbose", "success": False,
        })

    analysis = await skill.execute("analyze", {"experiment_id": exp_id})
    assert analysis.success
    assert analysis.data["leader"] == "concise"
    assert analysis.data["can_conclude"]

    conclusion = await skill.execute("conclude", {"experiment_id": exp_id})
    assert conclusion.success
    assert conclusion.data["winner"] == "concise"


@pytest.mark.asyncio
async def test_list_experiments(skill, experiment_params):
    await skill.execute("create", experiment_params)
    r = await skill.execute("list", {})
    assert r.success
    assert len(r.data["experiments"]) == 1


@pytest.mark.asyncio
async def test_learnings_empty(skill):
    r = await skill.execute("learnings", {})
    assert r.success
    assert len(r.data["learnings"]) == 0


@pytest.mark.asyncio
async def test_learnings_after_conclude(skill, experiment_params):
    r = await skill.execute("create", experiment_params)
    exp_id = r.data["experiment_id"]
    await skill.execute("start", {"experiment_id": exp_id})
    for _ in range(3):
        await skill.execute("record_outcome", {
            "experiment_id": exp_id, "variant": "concise", "success": True,
        })
        await skill.execute("record_outcome", {
            "experiment_id": exp_id, "variant": "verbose", "success": False,
        })
    await skill.execute("conclude", {"experiment_id": exp_id})
    r2 = await skill.execute("learnings", {})
    assert r2.success
    assert len(r2.data["learnings"]) == 1
    assert r2.data["learnings"][0]["winner"] == "concise"


@pytest.mark.asyncio
async def test_persistence(skill, experiment_params, tmp_path):
    """Test that experiments persist across skill instances."""
    r = await skill.execute("create", experiment_params)
    exp_id = r.data["experiment_id"]

    # Create a new skill instance pointing to same dir
    skill2 = ExperimentSkill()
    skill2.DATA_DIR = tmp_path / "experiments"
    r2 = await skill2.execute("get", {"experiment_id": exp_id})
    assert r2.success
    assert r2.data["name"] == "prompt-style"
