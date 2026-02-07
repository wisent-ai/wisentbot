"""Tests for TokenManager."""
from singularity.token_manager import TokenManager, estimate_tokens, get_context_limit, SessionTokenStats

def test_estimate_tokens_empty():
    assert estimate_tokens("") == 0

def test_estimate_tokens_basic():
    assert estimate_tokens("hello world") == 2
    assert estimate_tokens("a" * 400) == 100

def test_get_context_limit_known():
    assert get_context_limit("claude-sonnet-4-20250514") == 200_000
    assert get_context_limit("gpt-4o") == 128_000

def test_get_context_limit_unknown():
    assert get_context_limit("unknown-model") == 8_192

def test_token_manager_init():
    tm = TokenManager(model="gpt-4o")
    assert tm.context_limit == 128_000
    assert tm.effective_limit == int(128_000 * 0.95)

def test_compute_budget():
    tm = TokenManager(model="gpt-4o", output_reserve=1000)
    budget = tm.compute_budget("system prompt", "user prompt")
    assert budget.system_prompt > 0
    assert budget.available > 0

def test_fit_recent_actions_empty():
    tm = TokenManager()
    assert tm.fit_recent_actions([], 1000) == []

def test_fit_recent_actions_fits():
    tm = TokenManager()
    actions = [{"tool": "fs:view", "result": {"status": "ok"}}, {"tool": "sh:bash", "result": {"status": "ok"}}]
    assert len(tm.fit_recent_actions(actions, 10000)) == 2

def test_fit_recent_actions_truncates():
    tm = TokenManager()
    actions = [{"tool": f"s:a{i}", "result": {"status": "ok"}} for i in range(100)]
    result = tm.fit_recent_actions(actions, 50)
    assert len(result) < 100 and len(result) >= 2

def test_truncate_fits():
    tm = TokenManager()
    assert tm.truncate_to_budget("short", 1000) == "short"

def test_truncate_cuts():
    tm = TokenManager()
    result = tm.truncate_to_budget("x" * 10000, 100)
    assert len(result) < 10000 and "truncated" in result

def test_truncate_zero():
    tm = TokenManager()
    assert tm.truncate_to_budget("text", 0) == ""

def test_will_fit():
    tm = TokenManager(model="gpt-4o")
    assert tm.will_fit("short", "short") is True
    assert tm.will_fit("x" * 1000000, "x" * 1000000) is False

def test_utilization():
    tm = TokenManager(model="gpt-4o")
    assert 0 < tm.utilization("short", "short") < 0.1

def test_record_usage():
    tm = TokenManager()
    tm.record_usage(1000, 200, 0.01)
    tm.record_usage(1500, 300, 0.02)
    s = tm.get_stats()
    assert s["calls"] == 2
    assert s["total_input_tokens"] == 2500
    assert s["peak_input_tokens"] == 1500

def test_session_stats():
    s = SessionTokenStats()
    s.record(100, 50)
    s.record(200, 100)
    assert s.avg_input_tokens == 150 and s.total_tokens == 450

def test_update_model():
    tm = TokenManager(model="gpt-4o")
    tm.update_model("claude-sonnet-4-20250514")
    assert tm.context_limit == 200_000

def test_cognition_has_token_manager():
    from singularity.cognition import CognitionEngine
    engine = CognitionEngine(llm_provider="none")
    assert hasattr(engine, "token_manager")
    assert engine.token_manager.model == "claude-sonnet-4-20250514"
