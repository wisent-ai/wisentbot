#!/usr/bin/env python3
"""
Comprehensive tests for CanaryDeploymentSkill.

Tests cover:
- Rollout creation (canary, blue-green, validation)
- Stage advancement with metrics
- Auto-rollback on threshold breach
- Manual rollback
- Health checks
- Pause/resume
- Status and listing
- Environment registration
- Analytics
- Configuration
- Edge cases
"""

from unittest.mock import patch

import pytest

from singularity.skills.canary_deployment import (
    DEFAULT_BG_STAGES,
    DEFAULT_CANARY_STAGES,
    DEFAULT_THRESHOLDS,
    ROLLOUT_STATUSES,
    CanaryDeploymentSkill,
)

# --- Fixtures ---


@pytest.fixture
def skill(tmp_path):
    """Create a CanaryDeploymentSkill with isolated temp storage."""
    canary_file = tmp_path / "canary_deployments.json"
    with patch("singularity.skills.canary_deployment.CANARY_FILE", canary_file):
        s = CanaryDeploymentSkill()
        yield s


@pytest.fixture
def skill_with_rollout(skill):
    """Skill with one active canary rollout."""
    import asyncio

    async def setup():
        await skill.execute("create_rollout", {
            "service_id": "api-gateway",
            "version": "2.1.0",
            "strategy": "canary",
        })

    asyncio.get_event_loop().run_until_complete(setup())
    return skill


@pytest.fixture
def skill_with_history(skill):
    """Skill with multiple rollouts in various states."""
    import asyncio

    async def setup():
        # Completed rollout
        r1 = await skill.execute("create_rollout", {
            "service_id": "service-a",
            "version": "1.0.0",
        })
        rid1 = r1.data["rollout_id"]
        for _ in range(len(DEFAULT_CANARY_STAGES)):
            await skill.execute("advance", {
                "rollout_id": rid1,
                "metrics": {"error_rate": 0.01, "success_rate": 0.99},
            })

        # Rolled back rollout
        r2 = await skill.execute("create_rollout", {
            "service_id": "service-b",
            "version": "1.0.0",
        })
        await skill.execute("rollback", {
            "rollout_id": r2.data["rollout_id"],
            "reason": "High error rate in staging",
        })

        # Active rollout
        await skill.execute("create_rollout", {
            "service_id": "service-c",
            "version": "1.0.0",
        })

    asyncio.get_event_loop().run_until_complete(setup())
    return skill


# --- Manifest Tests ---


class TestManifest:
    def test_manifest_id(self, skill):
        assert skill.manifest.skill_id == "canary_deployment"

    def test_manifest_name(self, skill):
        assert skill.manifest.name == "Canary Deployment"

    def test_manifest_category(self, skill):
        assert skill.manifest.category == "infrastructure"

    def test_manifest_actions(self, skill):
        action_names = [a.name for a in skill.manifest.actions]
        expected = [
            "create_rollout", "advance", "rollback", "check_health",
            "pause", "resume", "status", "list_rollouts",
            "register_environment", "analytics", "configure",
        ]
        for name in expected:
            assert name in action_names

    def test_check_credentials(self, skill):
        assert skill.check_credentials() is True


# --- Rollout Creation Tests ---


class TestCreateRollout:
    @pytest.mark.asyncio
    async def test_create_canary(self, skill):
        result = await skill.execute("create_rollout", {
            "service_id": "my-service",
            "version": "1.0.0",
        })
        assert result.success is True
        assert result.data["rollout_id"]
        assert result.data["strategy"] == "canary"
        assert result.data["stages"] == ["5%", "25%", "50%", "100%"]

    @pytest.mark.asyncio
    async def test_create_blue_green(self, skill):
        result = await skill.execute("create_rollout", {
            "service_id": "my-service",
            "version": "1.0.0",
            "strategy": "blue_green",
        })
        assert result.success is True
        assert result.data["strategy"] == "blue_green"
        assert result.data["stages"] == DEFAULT_BG_STAGES

    @pytest.mark.asyncio
    async def test_create_custom_stages(self, skill):
        result = await skill.execute("create_rollout", {
            "service_id": "my-service",
            "version": "1.0.0",
            "stages": [10, 50, 100],
        })
        assert result.success is True
        assert result.data["stages"] == ["10%", "50%", "100%"]

    @pytest.mark.asyncio
    async def test_create_custom_thresholds(self, skill):
        result = await skill.execute("create_rollout", {
            "service_id": "my-service",
            "version": "1.0.0",
            "thresholds": {"max_error_rate": 0.02},
        })
        assert result.success is True
        assert result.data["thresholds"]["max_error_rate"] == 0.02

    @pytest.mark.asyncio
    async def test_create_missing_service_id(self, skill):
        result = await skill.execute("create_rollout", {
            "version": "1.0.0",
        })
        assert result.success is False
        assert "service_id" in result.message

    @pytest.mark.asyncio
    async def test_create_missing_version(self, skill):
        result = await skill.execute("create_rollout", {
            "service_id": "my-service",
        })
        assert result.success is False
        assert "version" in result.message

    @pytest.mark.asyncio
    async def test_create_invalid_strategy(self, skill):
        result = await skill.execute("create_rollout", {
            "service_id": "my-service",
            "version": "1.0.0",
            "strategy": "invalid",
        })
        assert result.success is False
        assert "Invalid strategy" in result.message

    @pytest.mark.asyncio
    async def test_create_duplicate_active(self, skill_with_rollout):
        result = await skill_with_rollout.execute("create_rollout", {
            "service_id": "api-gateway",
            "version": "2.2.0",
        })
        assert result.success is False
        assert "active rollout" in result.message.lower()

    @pytest.mark.asyncio
    async def test_create_with_metadata(self, skill):
        result = await skill.execute("create_rollout", {
            "service_id": "my-service",
            "version": "1.0.0",
            "metadata": {"commit": "abc123", "author": "adam"},
        })
        assert result.success is True

    @pytest.mark.asyncio
    async def test_create_stages_auto_append_100(self, skill):
        result = await skill.execute("create_rollout", {
            "service_id": "my-service",
            "version": "1.0.0",
            "stages": [10, 50],
        })
        assert result.data["stages"][-1] == "100%"

    @pytest.mark.asyncio
    async def test_create_stages_dedup_and_sort(self, skill):
        result = await skill.execute("create_rollout", {
            "service_id": "my-service",
            "version": "1.0.0",
            "stages": [50, 25, 50, 100],
        })
        assert result.data["stages"] == ["25%", "50%", "100%"]


# --- Advance Tests ---


class TestAdvance:
    @pytest.mark.asyncio
    async def test_advance_basic(self, skill_with_rollout):
        # Get the rollout ID
        listing = await skill_with_rollout.execute("list_rollouts", {})
        rid = listing.data["rollouts"][0]["id"]

        result = await skill_with_rollout.execute("advance", {
            "rollout_id": rid,
            "metrics": {"error_rate": 0.01, "success_rate": 0.99},
        })
        assert result.success is True
        assert result.data["current_stage"] == "25%"
        assert result.data["previous_stage"] == "5%"

    @pytest.mark.asyncio
    async def test_advance_to_completion(self, skill):
        r = await skill.execute("create_rollout", {
            "service_id": "svc",
            "version": "1.0.0",
            "stages": [50, 100],
        })
        rid = r.data["rollout_id"]

        # First advance: 50% -> 100%
        r1 = await skill.execute("advance", {
            "rollout_id": rid,
            "metrics": {"error_rate": 0.01},
        })
        assert r1.success is True
        assert r1.data["current_stage"] == "100%"

        # Second advance: completes
        r2 = await skill.execute("advance", {
            "rollout_id": rid,
            "metrics": {"error_rate": 0.01},
        })
        assert r2.success is True
        assert r2.data["status"] == "completed"

    @pytest.mark.asyncio
    async def test_advance_auto_rollback(self, skill_with_rollout):
        listing = await skill_with_rollout.execute("list_rollouts", {})
        rid = listing.data["rollouts"][0]["id"]

        result = await skill_with_rollout.execute("advance", {
            "rollout_id": rid,
            "metrics": {"error_rate": 0.15},  # Way above 5% threshold
        })
        assert result.success is False
        assert "auto-rollback" in result.message.lower()
        assert result.data["status"] == "rolled_back"
        assert "error_rate" in result.data["violations"]

    @pytest.mark.asyncio
    async def test_advance_force_override(self, skill_with_rollout):
        listing = await skill_with_rollout.execute("list_rollouts", {})
        rid = listing.data["rollouts"][0]["id"]

        result = await skill_with_rollout.execute("advance", {
            "rollout_id": rid,
            "metrics": {"error_rate": 0.15},
            "force": True,
        })
        assert result.success is True
        assert result.data["status"] == "in_progress"

    @pytest.mark.asyncio
    async def test_advance_no_metrics(self, skill_with_rollout):
        listing = await skill_with_rollout.execute("list_rollouts", {})
        rid = listing.data["rollouts"][0]["id"]

        result = await skill_with_rollout.execute("advance", {
            "rollout_id": rid,
        })
        assert result.success is True

    @pytest.mark.asyncio
    async def test_advance_not_found(self, skill):
        result = await skill.execute("advance", {"rollout_id": "nonexistent"})
        assert result.success is False

    @pytest.mark.asyncio
    async def test_advance_missing_id(self, skill):
        result = await skill.execute("advance", {})
        assert result.success is False

    @pytest.mark.asyncio
    async def test_advance_completed_rollout(self, skill_with_history):
        listing = await skill_with_history.execute("list_rollouts", {
            "status": "completed",
        })
        if listing.data["rollouts"]:
            rid = listing.data["rollouts"][0]["id"]
            result = await skill_with_history.execute("advance", {
                "rollout_id": rid,
            })
            assert result.success is False

    @pytest.mark.asyncio
    async def test_advance_latency_violation(self, skill_with_rollout):
        listing = await skill_with_rollout.execute("list_rollouts", {})
        rid = listing.data["rollouts"][0]["id"]

        result = await skill_with_rollout.execute("advance", {
            "rollout_id": rid,
            "metrics": {"latency_ms": 10000},
        })
        assert result.success is False
        assert "latency_ms" in result.data["violations"]

    @pytest.mark.asyncio
    async def test_advance_success_rate_violation(self, skill_with_rollout):
        listing = await skill_with_rollout.execute("list_rollouts", {})
        rid = listing.data["rollouts"][0]["id"]

        result = await skill_with_rollout.execute("advance", {
            "rollout_id": rid,
            "metrics": {"success_rate": 0.80},
        })
        assert result.success is False
        assert "success_rate" in result.data["violations"]


# --- Rollback Tests ---


class TestRollback:
    @pytest.mark.asyncio
    async def test_rollback_basic(self, skill_with_rollout):
        listing = await skill_with_rollout.execute("list_rollouts", {})
        rid = listing.data["rollouts"][0]["id"]

        result = await skill_with_rollout.execute("rollback", {
            "rollout_id": rid,
            "reason": "Found critical bug",
        })
        assert result.success is True
        assert result.data["reason"] == "Found critical bug"
        assert "api-gateway" in result.message

    @pytest.mark.asyncio
    async def test_rollback_default_reason(self, skill_with_rollout):
        listing = await skill_with_rollout.execute("list_rollouts", {})
        rid = listing.data["rollouts"][0]["id"]

        result = await skill_with_rollout.execute("rollback", {
            "rollout_id": rid,
        })
        assert result.success is True
        assert result.data["reason"] == "Manual rollback"

    @pytest.mark.asyncio
    async def test_rollback_completed(self, skill_with_history):
        listing = await skill_with_history.execute("list_rollouts", {
            "status": "completed",
        })
        if listing.data["rollouts"]:
            rid = listing.data["rollouts"][0]["id"]
            result = await skill_with_history.execute("rollback", {
                "rollout_id": rid,
            })
            assert result.success is False

    @pytest.mark.asyncio
    async def test_rollback_already_rolled_back(self, skill_with_history):
        listing = await skill_with_history.execute("list_rollouts", {
            "status": "rolled_back",
        })
        if listing.data["rollouts"]:
            rid = listing.data["rollouts"][0]["id"]
            result = await skill_with_history.execute("rollback", {
                "rollout_id": rid,
            })
            assert result.success is False

    @pytest.mark.asyncio
    async def test_rollback_not_found(self, skill):
        result = await skill.execute("rollback", {"rollout_id": "nope"})
        assert result.success is False


# --- Health Check Tests ---


class TestHealthCheck:
    @pytest.mark.asyncio
    async def test_health_check_passing(self, skill_with_rollout):
        listing = await skill_with_rollout.execute("list_rollouts", {})
        rid = listing.data["rollouts"][0]["id"]

        result = await skill_with_rollout.execute("check_health", {
            "rollout_id": rid,
            "metrics": {
                "error_rate": 0.01,
                "latency_ms": 100,
                "success_rate": 0.99,
            },
        })
        assert result.success is True
        assert result.data["healthy"] is True
        assert len(result.data["violations"]) == 0

    @pytest.mark.asyncio
    async def test_health_check_failing(self, skill_with_rollout):
        listing = await skill_with_rollout.execute("list_rollouts", {})
        rid = listing.data["rollouts"][0]["id"]

        result = await skill_with_rollout.execute("check_health", {
            "rollout_id": rid,
            "metrics": {
                "error_rate": 0.10,
                "latency_ms": 100,
                "success_rate": 0.90,
            },
        })
        assert result.success is True
        assert result.data["healthy"] is False
        assert "error_rate" in result.data["violations"]

    @pytest.mark.asyncio
    async def test_health_check_missing_metrics(self, skill_with_rollout):
        listing = await skill_with_rollout.execute("list_rollouts", {})
        rid = listing.data["rollouts"][0]["id"]

        result = await skill_with_rollout.execute("check_health", {
            "rollout_id": rid,
        })
        assert result.success is False

    @pytest.mark.asyncio
    async def test_health_check_not_found(self, skill):
        result = await skill.execute("check_health", {
            "rollout_id": "nope",
            "metrics": {"error_rate": 0.01},
        })
        assert result.success is False


# --- Pause/Resume Tests ---


class TestPauseResume:
    @pytest.mark.asyncio
    async def test_pause(self, skill_with_rollout):
        listing = await skill_with_rollout.execute("list_rollouts", {})
        rid = listing.data["rollouts"][0]["id"]

        result = await skill_with_rollout.execute("pause", {"rollout_id": rid})
        assert result.success is True
        assert result.data["status"] == "paused"

    @pytest.mark.asyncio
    async def test_resume(self, skill_with_rollout):
        listing = await skill_with_rollout.execute("list_rollouts", {})
        rid = listing.data["rollouts"][0]["id"]

        await skill_with_rollout.execute("pause", {"rollout_id": rid})
        result = await skill_with_rollout.execute("resume", {"rollout_id": rid})
        assert result.success is True
        assert result.data["status"] == "in_progress"

    @pytest.mark.asyncio
    async def test_pause_not_in_progress(self, skill_with_history):
        listing = await skill_with_history.execute("list_rollouts", {
            "status": "completed",
        })
        if listing.data["rollouts"]:
            rid = listing.data["rollouts"][0]["id"]
            result = await skill_with_history.execute("pause", {"rollout_id": rid})
            assert result.success is False

    @pytest.mark.asyncio
    async def test_resume_not_paused(self, skill_with_rollout):
        listing = await skill_with_rollout.execute("list_rollouts", {})
        rid = listing.data["rollouts"][0]["id"]

        result = await skill_with_rollout.execute("resume", {"rollout_id": rid})
        assert result.success is False

    @pytest.mark.asyncio
    async def test_pause_missing_id(self, skill):
        result = await skill.execute("pause", {})
        assert result.success is False

    @pytest.mark.asyncio
    async def test_resume_missing_id(self, skill):
        result = await skill.execute("resume", {})
        assert result.success is False

    @pytest.mark.asyncio
    async def test_advance_paused_rollout(self, skill_with_rollout):
        listing = await skill_with_rollout.execute("list_rollouts", {})
        rid = listing.data["rollouts"][0]["id"]

        await skill_with_rollout.execute("pause", {"rollout_id": rid})
        result = await skill_with_rollout.execute("advance", {"rollout_id": rid})
        assert result.success is False


# --- Status & Listing Tests ---


class TestStatusListing:
    @pytest.mark.asyncio
    async def test_status(self, skill_with_rollout):
        listing = await skill_with_rollout.execute("list_rollouts", {})
        rid = listing.data["rollouts"][0]["id"]

        result = await skill_with_rollout.execute("status", {"rollout_id": rid})
        assert result.success is True
        assert result.data["service_id"] == "api-gateway"
        assert result.data["version"] == "2.1.0"
        assert result.data["status"] == "in_progress"

    @pytest.mark.asyncio
    async def test_status_not_found(self, skill):
        result = await skill.execute("status", {"rollout_id": "nope"})
        assert result.success is False

    @pytest.mark.asyncio
    async def test_status_missing_id(self, skill):
        result = await skill.execute("status", {})
        assert result.success is False

    @pytest.mark.asyncio
    async def test_list_all(self, skill_with_history):
        result = await skill_with_history.execute("list_rollouts", {})
        assert result.success is True
        assert len(result.data["rollouts"]) == 3

    @pytest.mark.asyncio
    async def test_list_by_service(self, skill_with_history):
        result = await skill_with_history.execute("list_rollouts", {
            "service_id": "service-a",
        })
        assert result.data["rollouts"][0]["service_id"] == "service-a"

    @pytest.mark.asyncio
    async def test_list_by_status(self, skill_with_history):
        result = await skill_with_history.execute("list_rollouts", {
            "status": "completed",
        })
        assert all(r["status"] == "completed" for r in result.data["rollouts"])

    @pytest.mark.asyncio
    async def test_list_limit(self, skill_with_history):
        result = await skill_with_history.execute("list_rollouts", {"limit": 1})
        assert len(result.data["rollouts"]) == 1

    @pytest.mark.asyncio
    async def test_list_empty(self, skill):
        result = await skill.execute("list_rollouts", {})
        assert result.success is True
        assert len(result.data["rollouts"]) == 0


# --- Environment Tests ---


class TestEnvironments:
    @pytest.mark.asyncio
    async def test_register_environment(self, skill):
        result = await skill.execute("register_environment", {
            "env_name": "blue",
            "service_id": "api-gateway",
            "endpoint": "https://blue.api-gateway.example.com",
            "version": "2.0.0",
        })
        assert result.success is True
        assert result.data["env_name"] == "blue"
        assert result.data["key"] == "api-gateway:blue"

    @pytest.mark.asyncio
    async def test_register_missing_name(self, skill):
        result = await skill.execute("register_environment", {
            "service_id": "api-gateway",
        })
        assert result.success is False

    @pytest.mark.asyncio
    async def test_register_missing_service(self, skill):
        result = await skill.execute("register_environment", {
            "env_name": "blue",
        })
        assert result.success is False

    @pytest.mark.asyncio
    async def test_register_overwrite(self, skill):
        await skill.execute("register_environment", {
            "env_name": "blue",
            "service_id": "svc",
            "version": "1.0",
        })
        result = await skill.execute("register_environment", {
            "env_name": "blue",
            "service_id": "svc",
            "version": "2.0",
        })
        assert result.success is True


# --- Analytics Tests ---


class TestAnalytics:
    @pytest.mark.asyncio
    async def test_analytics_with_history(self, skill_with_history):
        result = await skill_with_history.execute("analytics", {})
        assert result.success is True
        assert result.data["total_rollouts"] == 3
        assert result.data["completed"] == 1
        assert result.data["rolled_back"] == 1
        assert result.data["success_rate"] > 0

    @pytest.mark.asyncio
    async def test_analytics_empty(self, skill):
        result = await skill.execute("analytics", {})
        assert result.success is True
        assert result.data["total_rollouts"] == 0
        assert result.data["success_rate"] == 0.0

    @pytest.mark.asyncio
    async def test_analytics_by_service(self, skill_with_history):
        result = await skill_with_history.execute("analytics", {
            "service_id": "service-a",
        })
        assert result.data["total_rollouts"] == 1
        assert result.data["completed"] == 1

    @pytest.mark.asyncio
    async def test_analytics_rollback_reasons(self, skill_with_history):
        result = await skill_with_history.execute("analytics", {})
        reasons = result.data["top_rollback_reasons"]
        assert any("error rate" in r["reason"].lower() for r in reasons)

    @pytest.mark.asyncio
    async def test_analytics_strategy_breakdown(self, skill_with_history):
        result = await skill_with_history.execute("analytics", {})
        assert "canary" in result.data["strategy_breakdown"]


# --- Configure Tests ---


class TestConfigure:
    @pytest.mark.asyncio
    async def test_configure_view(self, skill):
        result = await skill.execute("configure", {})
        assert result.success is True
        assert result.data["config"]["default_strategy"] == "canary"

    @pytest.mark.asyncio
    async def test_configure_strategy(self, skill):
        result = await skill.execute("configure", {
            "default_strategy": "blue_green",
        })
        assert result.success is True
        assert result.data["config"]["default_strategy"] == "blue_green"

    @pytest.mark.asyncio
    async def test_configure_invalid_strategy(self, skill):
        result = await skill.execute("configure", {
            "default_strategy": "invalid",
        })
        assert result.success is False

    @pytest.mark.asyncio
    async def test_configure_stages(self, skill):
        result = await skill.execute("configure", {
            "canary_stages": [10, 50, 100],
        })
        assert result.success is True
        assert result.data["config"]["canary_stages"] == [10, 50, 100]

    @pytest.mark.asyncio
    async def test_configure_stages_auto_100(self, skill):
        result = await skill.execute("configure", {
            "canary_stages": [10, 50],
        })
        assert result.data["config"]["canary_stages"][-1] == 100

    @pytest.mark.asyncio
    async def test_configure_invalid_stages(self, skill):
        result = await skill.execute("configure", {
            "canary_stages": "not a list",
        })
        assert result.success is False

    @pytest.mark.asyncio
    async def test_configure_thresholds(self, skill):
        result = await skill.execute("configure", {
            "default_thresholds": {"max_error_rate": 0.02},
        })
        assert result.success is True
        assert result.data["config"]["default_thresholds"]["max_error_rate"] == 0.02


# --- Edge Cases ---


class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_unknown_action(self, skill):
        result = await skill.execute("nonexistent", {})
        assert result.success is False

    @pytest.mark.asyncio
    async def test_corrupted_file(self, skill, tmp_path):
        canary_file = tmp_path / "canary_deployments.json"
        with patch("singularity.skills.canary_deployment.CANARY_FILE", canary_file):
            canary_file.write_text("not json{{{")
            s = CanaryDeploymentSkill()
            result = await s.execute("list_rollouts", {})
            assert result.success is True
            assert len(result.data["rollouts"]) == 0

    @pytest.mark.asyncio
    async def test_full_canary_lifecycle(self, skill):
        """Test complete lifecycle: create -> advance (all stages) -> complete."""
        r = await skill.execute("create_rollout", {
            "service_id": "lifecycle-test",
            "version": "1.0.0",
            "stages": [25, 50, 100],
        })
        rid = r.data["rollout_id"]

        good_metrics = {"error_rate": 0.01, "latency_ms": 50, "success_rate": 0.99}

        # Advance through all stages
        r1 = await skill.execute("advance", {"rollout_id": rid, "metrics": good_metrics})
        assert r1.success is True
        assert r1.data["current_stage"] == "50%"

        r2 = await skill.execute("advance", {"rollout_id": rid, "metrics": good_metrics})
        assert r2.success is True
        assert r2.data["current_stage"] == "100%"

        r3 = await skill.execute("advance", {"rollout_id": rid, "metrics": good_metrics})
        assert r3.success is True
        assert r3.data["status"] == "completed"

        # Verify it can't be advanced further
        r4 = await skill.execute("advance", {"rollout_id": rid})
        assert r4.success is False

    @pytest.mark.asyncio
    async def test_pause_advance_resume_advance(self, skill):
        """Test pause-resume cycle during rollout."""
        r = await skill.execute("create_rollout", {
            "service_id": "svc",
            "version": "1.0.0",
            "stages": [50, 100],
        })
        rid = r.data["rollout_id"]

        # Advance once
        await skill.execute("advance", {"rollout_id": rid})

        # Pause
        p = await skill.execute("pause", {"rollout_id": rid})
        assert p.data["status"] == "paused"

        # Can't advance while paused
        a = await skill.execute("advance", {"rollout_id": rid})
        assert a.success is False

        # Resume
        res = await skill.execute("resume", {"rollout_id": rid})
        assert res.data["status"] == "in_progress"

        # Can advance after resume
        a2 = await skill.execute("advance", {"rollout_id": rid})
        assert a2.success is True

    @pytest.mark.asyncio
    async def test_different_services_can_rollout_simultaneously(self, skill):
        r1 = await skill.execute("create_rollout", {
            "service_id": "svc-a",
            "version": "1.0.0",
        })
        r2 = await skill.execute("create_rollout", {
            "service_id": "svc-b",
            "version": "1.0.0",
        })
        assert r1.success is True
        assert r2.success is True

    @pytest.mark.asyncio
    async def test_can_create_after_rollback(self, skill):
        """After rolling back, can create a new rollout for same service."""
        r1 = await skill.execute("create_rollout", {
            "service_id": "svc",
            "version": "1.0.0",
        })
        await skill.execute("rollback", {
            "rollout_id": r1.data["rollout_id"],
        })

        r2 = await skill.execute("create_rollout", {
            "service_id": "svc",
            "version": "1.1.0",
        })
        assert r2.success is True


# --- Constants Tests ---


class TestConstants:
    def test_default_stages(self):
        assert DEFAULT_CANARY_STAGES == [5, 25, 50, 100]

    def test_default_bg_stages(self):
        assert len(DEFAULT_BG_STAGES) == 4

    def test_default_thresholds(self):
        assert DEFAULT_THRESHOLDS["max_error_rate"] == 0.05
        assert DEFAULT_THRESHOLDS["max_latency_ms"] == 5000
        assert DEFAULT_THRESHOLDS["min_success_rate"] == 0.95

    def test_rollout_statuses(self):
        assert "pending" in ROLLOUT_STATUSES
        assert "completed" in ROLLOUT_STATUSES
        assert "rolled_back" in ROLLOUT_STATUSES
