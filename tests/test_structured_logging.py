#!/usr/bin/env python3
"""
Comprehensive tests for StructuredLoggingSkill.

Tests cover:
- Log ingestion (single, batch, validation)
- Level filtering and min_level config
- Querying (by level, skill, trace, tag, search, time range, pagination)
- Trace correlation
- Statistics and aggregation
- Error reporting
- Trace ID generation
- Purging
- Configuration management
- Trim behavior
- Edge cases
"""

import time
from unittest.mock import patch

import pytest

from singularity.skills.structured_logging import (
    LOG_LEVEL_SEVERITY,
    LOG_LEVELS,
    MAX_LOGS,
    MAX_TAGS_PER_ENTRY,
    StructuredLoggingSkill,
)

# --- Fixtures ---


@pytest.fixture
def skill(tmp_path):
    """Create a StructuredLoggingSkill with isolated temp storage."""
    log_file = tmp_path / "structured_logs.json"
    with patch("singularity.skills.structured_logging.LOG_FILE", log_file):
        s = StructuredLoggingSkill()
        yield s


@pytest.fixture
def skill_with_logs(skill):
    """Skill pre-loaded with sample log entries."""
    import asyncio

    async def setup():
        # Add various log entries
        await skill.execute("log", {
            "level": "INFO",
            "message": "Service started",
            "skill_id": "shell",
            "trace_id": "trace-001",
            "tags": ["startup"],
        })
        await skill.execute("log", {
            "level": "DEBUG",
            "message": "Loading config",
            "skill_id": "shell",
            "trace_id": "trace-001",
            "tags": ["startup", "config"],
        })
        await skill.execute("log", {
            "level": "WARN",
            "message": "Slow query detected",
            "skill_id": "github",
            "trace_id": "trace-002",
            "tags": ["performance"],
        })
        await skill.execute("log", {
            "level": "ERROR",
            "message": "Connection timeout",
            "skill_id": "github",
            "trace_id": "trace-002",
            "tags": ["network"],
        })
        await skill.execute("log", {
            "level": "FATAL",
            "message": "Out of memory",
            "skill_id": "browser",
            "trace_id": "trace-003",
        })
        await skill.execute("log", {
            "level": "INFO",
            "message": "Task completed successfully",
            "skill_id": "planner",
            "trace_id": "trace-003",
            "tags": ["tasks"],
        })
        await skill.execute("log", {
            "level": "ERROR",
            "message": "Connection timeout retry",
            "skill_id": "github",
            "trace_id": "trace-002",
            "tags": ["network", "retry"],
        })

    asyncio.get_event_loop().run_until_complete(setup())
    return skill


# --- Manifest Tests ---


class TestManifest:
    def test_manifest_id(self, skill):
        assert skill.manifest.skill_id == "structured_logging"

    def test_manifest_name(self, skill):
        assert skill.manifest.name == "Structured Logging"

    def test_manifest_version(self, skill):
        assert skill.manifest.version == "1.0.0"

    def test_manifest_category(self, skill):
        assert skill.manifest.category == "infrastructure"

    def test_manifest_actions(self, skill):
        action_names = [a.name for a in skill.manifest.actions]
        assert "log" in action_names
        assert "log_batch" in action_names
        assert "query" in action_names
        assert "trace" in action_names
        assert "stats" in action_names
        assert "error_report" in action_names
        assert "new_trace" in action_names
        assert "purge" in action_names
        assert "config" in action_names

    def test_manifest_no_credentials(self, skill):
        assert skill.manifest.required_credentials == []

    def test_check_credentials(self, skill):
        assert skill.check_credentials() is True


# --- Log Ingestion Tests ---


class TestLogIngestion:
    @pytest.mark.asyncio
    async def test_log_basic(self, skill):
        result = await skill.execute("log", {
            "message": "Hello world",
        })
        assert result.success is True
        assert "Hello world" in result.message
        assert result.data["ingested"] is True
        assert result.data["level"] == "INFO"
        assert result.data["trace_id"]  # auto-generated
        assert result.data["log_id"]

    @pytest.mark.asyncio
    async def test_log_with_all_fields(self, skill):
        result = await skill.execute("log", {
            "level": "ERROR",
            "message": "Something went wrong",
            "skill_id": "github",
            "trace_id": "my-trace-123",
            "data": {"url": "https://api.github.com", "status": 500},
            "tags": ["api", "error"],
        })
        assert result.success is True
        assert result.data["level"] == "ERROR"
        assert result.data["trace_id"] == "my-trace-123"

    @pytest.mark.asyncio
    async def test_log_missing_message(self, skill):
        result = await skill.execute("log", {"level": "INFO"})
        assert result.success is False
        assert "message" in result.message.lower()

    @pytest.mark.asyncio
    async def test_log_empty_message(self, skill):
        result = await skill.execute("log", {"message": "  "})
        assert result.success is False

    @pytest.mark.asyncio
    async def test_log_invalid_level(self, skill):
        result = await skill.execute("log", {
            "level": "INVALID",
            "message": "test",
        })
        assert result.success is False
        assert "Invalid level" in result.message

    @pytest.mark.asyncio
    async def test_log_default_level_info(self, skill):
        result = await skill.execute("log", {"message": "test"})
        assert result.data["level"] == "INFO"

    @pytest.mark.asyncio
    async def test_log_case_insensitive_level(self, skill):
        result = await skill.execute("log", {
            "level": "warn",
            "message": "test warning",
        })
        assert result.success is True
        assert result.data["level"] == "WARN"

    @pytest.mark.asyncio
    async def test_log_all_levels(self, skill):
        for level in LOG_LEVELS:
            result = await skill.execute("log", {
                "level": level,
                "message": f"Test {level}",
            })
            assert result.success is True
            assert result.data["level"] == level

    @pytest.mark.asyncio
    async def test_log_tags_as_string(self, skill):
        result = await skill.execute("log", {
            "message": "test",
            "tags": "api, network, slow",
        })
        assert result.success is True

        # Verify tags were parsed
        query = await skill.execute("query", {"limit": 1})
        log_entry = query.data["logs"][0]
        assert "api" in log_entry["tags"]
        assert "network" in log_entry["tags"]
        assert "slow" in log_entry["tags"]

    @pytest.mark.asyncio
    async def test_log_tags_truncated(self, skill):
        tags = [f"tag-{i}" for i in range(30)]
        result = await skill.execute("log", {
            "message": "test",
            "tags": tags,
        })
        assert result.success is True

        query = await skill.execute("query", {"limit": 1})
        assert len(query.data["logs"][0]["tags"]) == MAX_TAGS_PER_ENTRY

    @pytest.mark.asyncio
    async def test_log_data_non_dict(self, skill):
        """Non-dict data should be wrapped in {value: ...}."""
        result = await skill.execute("log", {
            "message": "test",
            "data": "plain string",
        })
        assert result.success is True

    @pytest.mark.asyncio
    async def test_log_auto_trace_id(self, skill):
        result = await skill.execute("log", {"message": "test"})
        assert result.data["trace_id"]
        assert len(result.data["trace_id"]) > 0


class TestLogBatch:
    @pytest.mark.asyncio
    async def test_batch_basic(self, skill):
        result = await skill.execute("log_batch", {
            "entries": [
                {"message": "log 1", "level": "INFO"},
                {"message": "log 2", "level": "WARN"},
                {"message": "log 3", "level": "ERROR"},
            ],
        })
        assert result.success is True
        assert result.data["ingested"] == 3
        assert result.data["dropped"] == 0
        assert result.data["invalid"] == 0

    @pytest.mark.asyncio
    async def test_batch_with_shared_trace(self, skill):
        result = await skill.execute("log_batch", {
            "entries": [
                {"message": "step 1", "trace_id": "batch-trace"},
                {"message": "step 2", "trace_id": "batch-trace"},
                {"message": "step 3", "trace_id": "batch-trace"},
            ],
        })
        assert result.success is True
        assert "batch-trace" in result.data["trace_ids"]

    @pytest.mark.asyncio
    async def test_batch_empty(self, skill):
        result = await skill.execute("log_batch", {"entries": []})
        assert result.success is False
        assert "empty" in result.message.lower()

    @pytest.mark.asyncio
    async def test_batch_not_a_list(self, skill):
        result = await skill.execute("log_batch", {"entries": "not a list"})
        assert result.success is False

    @pytest.mark.asyncio
    async def test_batch_too_large(self, skill):
        entries = [{"message": f"log {i}"} for i in range(101)]
        result = await skill.execute("log_batch", {"entries": entries})
        assert result.success is False
        assert "100" in result.message

    @pytest.mark.asyncio
    async def test_batch_with_invalid_entries(self, skill):
        result = await skill.execute("log_batch", {
            "entries": [
                {"message": "valid"},
                "not a dict",
                {"message": "also valid"},
                42,
            ],
        })
        assert result.success is True
        assert result.data["ingested"] == 2
        assert result.data["invalid"] == 2

    @pytest.mark.asyncio
    async def test_batch_with_level_filtering(self, skill):
        # Set min_level to WARN
        await skill.execute("config", {"min_level": "WARN"})

        result = await skill.execute("log_batch", {
            "entries": [
                {"message": "debug", "level": "DEBUG"},
                {"message": "info", "level": "INFO"},
                {"message": "warn", "level": "WARN"},
                {"message": "error", "level": "ERROR"},
            ],
        })
        assert result.success is True
        assert result.data["ingested"] == 2
        assert result.data["dropped"] == 2


# --- Query Tests ---


class TestQuery:
    @pytest.mark.asyncio
    async def test_query_all(self, skill_with_logs):
        result = await skill_with_logs.execute("query", {})
        assert result.success is True
        assert result.data["total_matching"] == 7

    @pytest.mark.asyncio
    async def test_query_by_level(self, skill_with_logs):
        result = await skill_with_logs.execute("query", {"level": "ERROR"})
        assert result.success is True
        # ERROR + FATAL = 3
        assert result.data["total_matching"] == 3

    @pytest.mark.asyncio
    async def test_query_by_skill(self, skill_with_logs):
        result = await skill_with_logs.execute("query", {"skill_id": "github"})
        assert result.success is True
        assert result.data["total_matching"] == 3

    @pytest.mark.asyncio
    async def test_query_by_trace(self, skill_with_logs):
        result = await skill_with_logs.execute("query", {"trace_id": "trace-002"})
        assert result.success is True
        assert result.data["total_matching"] == 3

    @pytest.mark.asyncio
    async def test_query_by_tag(self, skill_with_logs):
        result = await skill_with_logs.execute("query", {"tag": "network"})
        assert result.success is True
        assert result.data["total_matching"] == 2

    @pytest.mark.asyncio
    async def test_query_by_search(self, skill_with_logs):
        result = await skill_with_logs.execute("query", {"search": "timeout"})
        assert result.success is True
        assert result.data["total_matching"] == 2

    @pytest.mark.asyncio
    async def test_query_search_case_insensitive(self, skill_with_logs):
        result = await skill_with_logs.execute("query", {"search": "CONNECTION"})
        assert result.success is True
        assert result.data["total_matching"] == 2

    @pytest.mark.asyncio
    async def test_query_since(self, skill_with_logs):
        # All logs should be after a timestamp in the past
        result = await skill_with_logs.execute("query", {"since": 0})
        assert result.data["total_matching"] == 7

        # No logs should be after a timestamp in the future
        result = await skill_with_logs.execute("query", {"since": time.time() + 3600})
        assert result.data["total_matching"] == 0

    @pytest.mark.asyncio
    async def test_query_until(self, skill_with_logs):
        result = await skill_with_logs.execute("query", {"until": time.time() + 3600})
        assert result.data["total_matching"] == 7

    @pytest.mark.asyncio
    async def test_query_pagination(self, skill_with_logs):
        result = await skill_with_logs.execute("query", {"limit": 3, "offset": 0})
        assert len(result.data["logs"]) == 3
        assert result.data["total_matching"] == 7

        result2 = await skill_with_logs.execute("query", {"limit": 3, "offset": 3})
        assert len(result2.data["logs"]) == 3

        result3 = await skill_with_logs.execute("query", {"limit": 3, "offset": 6})
        assert len(result3.data["logs"]) == 1

    @pytest.mark.asyncio
    async def test_query_most_recent_first(self, skill_with_logs):
        result = await skill_with_logs.execute("query", {"limit": 1})
        # The last log added was "Connection timeout retry"
        assert "retry" in result.data["logs"][0]["message"].lower()

    @pytest.mark.asyncio
    async def test_query_combined_filters(self, skill_with_logs):
        result = await skill_with_logs.execute("query", {
            "skill_id": "github",
            "level": "ERROR",
        })
        assert result.success is True
        assert result.data["total_matching"] == 2  # 2 ERROR from github

    @pytest.mark.asyncio
    async def test_query_no_results(self, skill_with_logs):
        result = await skill_with_logs.execute("query", {"skill_id": "nonexistent"})
        assert result.success is True
        assert result.data["total_matching"] == 0

    @pytest.mark.asyncio
    async def test_query_limit_bounds(self, skill_with_logs):
        # Limit capped at 500
        result = await skill_with_logs.execute("query", {"limit": 999})
        assert result.data["limit"] == 500

        # Limit minimum 1
        result = await skill_with_logs.execute("query", {"limit": 0})
        assert result.data["limit"] == 1


# --- Trace Tests ---


class TestTrace:
    @pytest.mark.asyncio
    async def test_trace_basic(self, skill_with_logs):
        result = await skill_with_logs.execute("trace", {"trace_id": "trace-002"})
        assert result.success is True
        assert result.data["entry_count"] == 3
        assert "github" in result.data["skills_involved"]
        assert result.data["has_errors"] is True

    @pytest.mark.asyncio
    async def test_trace_chronological_order(self, skill_with_logs):
        result = await skill_with_logs.execute("trace", {"trace_id": "trace-001"})
        entries = result.data["entries"]
        assert len(entries) == 2
        # Should be in chronological order (not reverse)
        assert entries[0]["level"] == "INFO"
        assert entries[1]["level"] == "DEBUG"

    @pytest.mark.asyncio
    async def test_trace_duration(self, skill_with_logs):
        result = await skill_with_logs.execute("trace", {"trace_id": "trace-001"})
        assert "duration_seconds" in result.data
        assert result.data["duration_seconds"] >= 0

    @pytest.mark.asyncio
    async def test_trace_no_errors(self, skill_with_logs):
        result = await skill_with_logs.execute("trace", {"trace_id": "trace-001"})
        assert result.data["has_errors"] is False

    @pytest.mark.asyncio
    async def test_trace_not_found(self, skill_with_logs):
        result = await skill_with_logs.execute("trace", {"trace_id": "nonexistent"})
        assert result.success is True
        assert result.data["entry_count"] == 0

    @pytest.mark.asyncio
    async def test_trace_missing_id(self, skill_with_logs):
        result = await skill_with_logs.execute("trace", {})
        assert result.success is False
        assert "trace_id" in result.message.lower()

    @pytest.mark.asyncio
    async def test_trace_levels_seen(self, skill_with_logs):
        result = await skill_with_logs.execute("trace", {"trace_id": "trace-002"})
        levels = result.data["levels_seen"]
        assert "WARN" in levels
        assert "ERROR" in levels


# --- Stats Tests ---


class TestStats:
    @pytest.mark.asyncio
    async def test_stats_by_level(self, skill_with_logs):
        result = await skill_with_logs.execute("stats", {"group_by": "level"})
        assert result.success is True
        groups = result.data["groups"]
        assert groups["INFO"] == 2
        assert groups["DEBUG"] == 1
        assert groups["WARN"] == 1
        assert groups["ERROR"] == 2
        assert groups["FATAL"] == 1

    @pytest.mark.asyncio
    async def test_stats_by_skill(self, skill_with_logs):
        result = await skill_with_logs.execute("stats", {"group_by": "skill"})
        assert result.success is True
        groups = result.data["groups"]
        assert groups["github"] == 3
        assert groups["shell"] == 2

    @pytest.mark.asyncio
    async def test_stats_by_hour(self, skill_with_logs):
        result = await skill_with_logs.execute("stats", {"group_by": "hour"})
        assert result.success is True
        assert result.data["total_logs"] == 7

    @pytest.mark.asyncio
    async def test_stats_by_tag(self, skill_with_logs):
        result = await skill_with_logs.execute("stats", {"group_by": "tag"})
        assert result.success is True
        groups = result.data["groups"]
        assert groups["network"] == 2
        assert groups["startup"] == 2

    @pytest.mark.asyncio
    async def test_stats_invalid_group_by(self, skill_with_logs):
        result = await skill_with_logs.execute("stats", {"group_by": "invalid"})
        assert result.success is False

    @pytest.mark.asyncio
    async def test_stats_error_rate(self, skill_with_logs):
        result = await skill_with_logs.execute("stats", {})
        assert result.data["error_count"] == 3  # 2 ERROR + 1 FATAL
        assert result.data["error_rate"] > 0

    @pytest.mark.asyncio
    async def test_stats_since_filter(self, skill_with_logs):
        # Future timestamp should yield 0
        result = await skill_with_logs.execute("stats", {"since": time.time() + 3600})
        assert result.data["total_logs"] == 0

    @pytest.mark.asyncio
    async def test_stats_lifetime_counters(self, skill_with_logs):
        result = await skill_with_logs.execute("stats", {})
        assert result.data["lifetime_ingested"] == 7
        assert result.data["lifetime_trimmed"] == 0

    @pytest.mark.asyncio
    async def test_stats_default_level(self, skill_with_logs):
        result = await skill_with_logs.execute("stats", {})
        assert result.data["group_by"] == "level"


# --- Error Report Tests ---


class TestErrorReport:
    @pytest.mark.asyncio
    async def test_error_report_basic(self, skill_with_logs):
        result = await skill_with_logs.execute("error_report", {})
        assert result.success is True
        assert result.data["error_count"] == 3
        assert len(result.data["top_skills"]) > 0

    @pytest.mark.asyncio
    async def test_error_report_top_skills(self, skill_with_logs):
        result = await skill_with_logs.execute("error_report", {})
        top_skills = result.data["top_skills"]
        # github has 2 errors, browser has 1 (FATAL)
        assert top_skills[0]["skill"] == "github"
        assert top_skills[0]["count"] == 2

    @pytest.mark.asyncio
    async def test_error_report_top_messages(self, skill_with_logs):
        result = await skill_with_logs.execute("error_report", {})
        top_messages = result.data["top_messages"]
        assert len(top_messages) > 0
        # All messages should be present
        messages = [m["message"] for m in top_messages]
        assert any("timeout" in m.lower() for m in messages)

    @pytest.mark.asyncio
    async def test_error_report_top_traces(self, skill_with_logs):
        result = await skill_with_logs.execute("error_report", {})
        top_traces = result.data["top_traces"]
        # trace-002 has 2 errors
        assert any(t["trace_id"] == "trace-002" for t in top_traces)

    @pytest.mark.asyncio
    async def test_error_report_no_errors(self, skill):
        await skill.execute("log", {"message": "all good", "level": "INFO"})
        result = await skill.execute("error_report", {})
        assert result.success is True
        assert result.data["error_count"] == 0

    @pytest.mark.asyncio
    async def test_error_report_since_filter(self, skill_with_logs):
        result = await skill_with_logs.execute("error_report", {
            "since": time.time() + 3600,
        })
        assert result.data["error_count"] == 0

    @pytest.mark.asyncio
    async def test_error_report_top_n(self, skill_with_logs):
        result = await skill_with_logs.execute("error_report", {"top_n": 1})
        assert len(result.data["top_skills"]) == 1

    @pytest.mark.asyncio
    async def test_error_report_hourly_rates(self, skill_with_logs):
        result = await skill_with_logs.execute("error_report", {})
        assert "hourly_error_rates" in result.data


# --- New Trace Tests ---


class TestNewTrace:
    @pytest.mark.asyncio
    async def test_new_trace(self, skill):
        result = await skill.execute("new_trace", {})
        assert result.success is True
        assert result.data["trace_id"]
        assert len(result.data["trace_id"]) == 12

    @pytest.mark.asyncio
    async def test_new_trace_unique(self, skill):
        r1 = await skill.execute("new_trace", {})
        r2 = await skill.execute("new_trace", {})
        assert r1.data["trace_id"] != r2.data["trace_id"]


# --- Purge Tests ---


class TestPurge:
    @pytest.mark.asyncio
    async def test_purge_requires_confirm(self, skill_with_logs):
        result = await skill_with_logs.execute("purge", {
            "level": "DEBUG",
            "confirm": False,
        })
        assert result.success is False
        assert "confirm" in result.message.lower()

    @pytest.mark.asyncio
    async def test_purge_requires_filter(self, skill_with_logs):
        result = await skill_with_logs.execute("purge", {"confirm": True})
        assert result.success is False
        assert "filter" in result.message.lower()

    @pytest.mark.asyncio
    async def test_purge_by_level(self, skill_with_logs):
        result = await skill_with_logs.execute("purge", {
            "level": "DEBUG",
            "confirm": True,
        })
        assert result.success is True
        assert result.data["deleted"] == 1
        assert result.data["remaining"] == 6

    @pytest.mark.asyncio
    async def test_purge_by_skill(self, skill_with_logs):
        result = await skill_with_logs.execute("purge", {
            "skill_id": "github",
            "confirm": True,
        })
        assert result.success is True
        assert result.data["deleted"] == 3
        assert result.data["remaining"] == 4

    @pytest.mark.asyncio
    async def test_purge_by_age(self, skill_with_logs):
        # Purge everything (all logs are before future timestamp)
        result = await skill_with_logs.execute("purge", {
            "older_than": time.time() + 3600,
            "confirm": True,
        })
        assert result.success is True
        assert result.data["deleted"] == 7
        assert result.data["remaining"] == 0

    @pytest.mark.asyncio
    async def test_purge_confirm_string(self, skill_with_logs):
        result = await skill_with_logs.execute("purge", {
            "level": "FATAL",
            "confirm": "true",
        })
        assert result.success is True
        assert result.data["deleted"] == 1

    @pytest.mark.asyncio
    async def test_purge_updates_trim_counter(self, skill_with_logs):
        await skill_with_logs.execute("purge", {
            "level": "DEBUG",
            "confirm": True,
        })
        stats = await skill_with_logs.execute("stats", {})
        assert stats.data["lifetime_trimmed"] == 1


# --- Config Tests ---


class TestConfig:
    @pytest.mark.asyncio
    async def test_config_view(self, skill):
        result = await skill.execute("config", {})
        assert result.success is True
        assert result.data["config"]["max_logs"] == MAX_LOGS
        assert result.data["config"]["min_level"] == "DEBUG"

    @pytest.mark.asyncio
    async def test_config_update_max_logs(self, skill):
        result = await skill.execute("config", {"max_logs": 5000})
        assert result.success is True
        assert result.data["config"]["max_logs"] == 5000

    @pytest.mark.asyncio
    async def test_config_max_logs_bounds(self, skill):
        # Too small
        result = await skill.execute("config", {"max_logs": 10})
        assert result.data["config"]["max_logs"] == 100  # Minimum

        # Too large
        result = await skill.execute("config", {"max_logs": 999999})
        assert result.data["config"]["max_logs"] == 100000  # Maximum

    @pytest.mark.asyncio
    async def test_config_update_min_level(self, skill):
        result = await skill.execute("config", {"min_level": "WARN"})
        assert result.success is True
        assert result.data["config"]["min_level"] == "WARN"

    @pytest.mark.asyncio
    async def test_config_invalid_min_level(self, skill):
        result = await skill.execute("config", {"min_level": "INVALID"})
        assert result.success is False

    @pytest.mark.asyncio
    async def test_config_invalid_max_logs(self, skill):
        result = await skill.execute("config", {"max_logs": "abc"})
        assert result.success is False


# --- Min Level Filtering Tests ---


class TestMinLevelFiltering:
    @pytest.mark.asyncio
    async def test_min_level_drops_below(self, skill):
        await skill.execute("config", {"min_level": "WARN"})

        result = await skill.execute("log", {
            "level": "DEBUG",
            "message": "should be dropped",
        })
        assert result.success is True
        assert result.data["ingested"] is False

    @pytest.mark.asyncio
    async def test_min_level_accepts_at_level(self, skill):
        await skill.execute("config", {"min_level": "WARN"})

        result = await skill.execute("log", {
            "level": "WARN",
            "message": "should pass",
        })
        assert result.success is True
        assert result.data["ingested"] is True

    @pytest.mark.asyncio
    async def test_min_level_accepts_above(self, skill):
        await skill.execute("config", {"min_level": "WARN"})

        result = await skill.execute("log", {
            "level": "ERROR",
            "message": "should pass",
        })
        assert result.success is True
        assert result.data["ingested"] is True


# --- Trim Tests ---


class TestTrimBehavior:
    @pytest.mark.asyncio
    async def test_trim_at_max(self, skill):
        # Set very low max
        await skill.execute("config", {"max_logs": 100})

        # Add more than max
        entries = [{"message": f"log-{i}", "level": "INFO"} for i in range(120)]
        await skill.execute("log_batch", {"entries": entries[:100]})
        await skill.execute("log_batch", {"entries": entries[100:]})

        stats = await skill.execute("stats", {})
        assert stats.data["total_logs"] <= 100
        assert stats.data["lifetime_trimmed"] > 0


# --- Edge Cases ---


class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_unknown_action(self, skill):
        result = await skill.execute("nonexistent", {})
        assert result.success is False
        assert "Unknown action" in result.message

    @pytest.mark.asyncio
    async def test_empty_log_store_query(self, skill):
        result = await skill.execute("query", {})
        assert result.success is True
        assert result.data["total_matching"] == 0

    @pytest.mark.asyncio
    async def test_empty_log_store_stats(self, skill):
        result = await skill.execute("stats", {})
        assert result.success is True
        assert result.data["total_logs"] == 0
        assert result.data["error_rate"] == 0.0

    @pytest.mark.asyncio
    async def test_empty_log_store_error_report(self, skill):
        result = await skill.execute("error_report", {})
        assert result.success is True
        assert result.data["error_count"] == 0

    @pytest.mark.asyncio
    async def test_corrupted_file_recovery(self, skill, tmp_path):
        """Skill should handle corrupted JSON gracefully."""
        log_file = tmp_path / "structured_logs.json"
        with patch("singularity.skills.structured_logging.LOG_FILE", log_file):
            log_file.write_text("not valid json{{{")
            s = StructuredLoggingSkill()
            result = await s.execute("query", {})
            assert result.success is True
            assert result.data["total_matching"] == 0

    @pytest.mark.asyncio
    async def test_query_with_invalid_since(self, skill_with_logs):
        """Invalid since value should be ignored, not crash."""
        result = await skill_with_logs.execute("query", {"since": "not-a-number"})
        assert result.success is True

    @pytest.mark.asyncio
    async def test_stats_no_tags(self, skill):
        """Stats by tag with no tagged entries."""
        await skill.execute("log", {"message": "no tags"})
        result = await skill.execute("stats", {"group_by": "tag"})
        assert result.success is True

    @pytest.mark.asyncio
    async def test_purge_combined_filters(self, skill_with_logs):
        """Purge with multiple filters uses AND logic."""
        result = await skill_with_logs.execute("purge", {
            "older_than": time.time() + 3600,
            "level": "ERROR",
            "confirm": True,
        })
        assert result.success is True
        # Only ERROR entries should be deleted (2), not all 7
        assert result.data["deleted"] == 2


# --- Constants Tests ---


class TestConstants:
    def test_log_levels(self):
        assert LOG_LEVELS == ["DEBUG", "INFO", "WARN", "ERROR", "FATAL"]

    def test_log_level_severity_ordering(self):
        for i, level in enumerate(LOG_LEVELS):
            assert LOG_LEVEL_SEVERITY[level] == i

    def test_max_logs_default(self):
        assert MAX_LOGS == 10000

    def test_max_tags_per_entry(self):
        assert MAX_TAGS_PER_ENTRY == 20
