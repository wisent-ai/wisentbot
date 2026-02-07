"""Tests for ErrorRecoveryEngine."""

import pytest
from singularity.error_recovery import ErrorRecoveryEngine, ErrorCategory, ErrorContext


@pytest.fixture
def engine():
    return ErrorRecoveryEngine()


class TestCategorizeError:
    def test_timeout(self, engine):
        assert engine.categorize_error(TimeoutError("timed out")) == ErrorCategory.TIMEOUT

    def test_file_not_found(self, engine):
        assert engine.categorize_error(FileNotFoundError("no file")) == ErrorCategory.NOT_FOUND

    def test_permission(self, engine):
        assert engine.categorize_error(PermissionError("denied")) == ErrorCategory.PERMISSION

    def test_value_error(self, engine):
        assert engine.categorize_error(ValueError("bad value")) == ErrorCategory.INVALID_PARAMS

    def test_type_error(self, engine):
        assert engine.categorize_error(TypeError("wrong type")) == ErrorCategory.INVALID_PARAMS

    def test_key_error(self, engine):
        assert engine.categorize_error(KeyError("missing")) == ErrorCategory.INVALID_PARAMS

    def test_message_hint_api_key(self, engine):
        assert engine.categorize_error(Exception("Invalid API key")) == ErrorCategory.CREDENTIALS

    def test_message_hint_rate_limit(self, engine):
        assert engine.categorize_error(Exception("429 Too Many Requests")) == ErrorCategory.RATE_LIMIT

    def test_message_hint_network(self, engine):
        assert engine.categorize_error(Exception("Connection refused")) == ErrorCategory.NETWORK

    def test_unknown(self, engine):
        assert engine.categorize_error(Exception("something weird")) == ErrorCategory.UNKNOWN


class TestCreateErrorContext:
    def test_creates_context(self, engine):
        try:
            raise ValueError("bad param")
        except ValueError as e:
            ctx = engine.create_error_context(e, "fs", "read", {"path": "/tmp/x"})
        assert ctx.skill_id == "fs"
        assert ctx.action_name == "read"
        assert ctx.category == ErrorCategory.INVALID_PARAMS
        assert "bad param" in ctx.message
        assert ctx.params_used == {"path": "/tmp/x"}
        assert len(ctx.suggestions) > 0

    def test_sanitizes_sensitive_params(self, engine):
        ctx = engine.create_error_context(
            Exception("fail"), "auth", "login",
            {"api_key": "sk-secret123", "username": "bob"}
        )
        assert ctx.params_used["api_key"] == "***REDACTED***"
        assert ctx.params_used["username"] == "bob"

    def test_truncates_long_params(self, engine):
        ctx = engine.create_error_context(
            Exception("fail"), "s", "a", {"data": "x" * 300}
        )
        assert len(ctx.params_used["data"]) < 300
        assert "truncated" in ctx.params_used["data"]

    def test_records_to_history(self, engine):
        engine.create_error_context(Exception("e1"), "s1", "a1", {})
        engine.create_error_context(Exception("e2"), "s1", "a2", {})
        assert len(engine._error_history) == 2
        assert engine._error_counts["s1:a1"] == 1
        assert engine._error_counts["s1:a2"] == 1


class TestRecoverySuggestions:
    def test_repeated_failure_adds_suggestion(self, engine):
        for _ in range(3):
            engine.create_error_context(Exception("fail"), "s", "a", {})
        suggestions = engine.get_recovery_suggestions(ErrorCategory.UNKNOWN, "s", "a")
        assert "failed 3 times" in suggestions[0]

    def test_default_suggestions(self, engine):
        s = engine.get_recovery_suggestions(ErrorCategory.CREDENTIALS, "s", "a")
        assert any("API key" in x for x in s)


class TestFormatForLLM:
    def test_format(self, engine):
        ctx = ErrorContext(
            skill_id="fs", action_name="read",
            category=ErrorCategory.NOT_FOUND,
            message="File not found: /tmp/x",
            traceback_str="", params_used={},
            suggestions=["Check the path", "List files first"],
        )
        result = engine.format_for_llm(ctx)
        assert "not_found" in result
        assert "File not found" in result
        assert "Check the path" in result


class TestErrorSummary:
    def test_empty(self, engine):
        s = engine.get_error_summary()
        assert s["total_errors"] == 0

    def test_with_errors(self, engine):
        engine.create_error_context(ValueError("v"), "s1", "a1", {})
        engine.create_error_context(TimeoutError("t"), "s2", "a2", {})
        engine.create_error_context(ValueError("v"), "s1", "a1", {})
        s = engine.get_error_summary()
        assert s["total_errors"] == 3
        assert "invalid_params" in s["by_category"]
        assert len(s["patterns"]) >= 1  # s1:a1 failed 2x

    def test_clear(self, engine):
        engine.create_error_context(Exception("e"), "s", "a", {})
        engine.clear_history()
        assert engine.get_error_summary()["total_errors"] == 0


class TestMaxHistory:
    def test_respects_max(self):
        engine = ErrorRecoveryEngine(max_history=5)
        for i in range(10):
            engine.create_error_context(Exception(f"e{i}"), "s", "a", {})
        assert len(engine._error_history) == 5
