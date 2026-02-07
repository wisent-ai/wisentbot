"""Tests for LocalMemorySkill."""
import asyncio
import json
import tempfile
import pytest
from singularity.skills.local_memory import LocalMemorySkill


@pytest.fixture
def skill(tmp_path):
    s = LocalMemorySkill()
    s.set_memory_dir(str(tmp_path / "memory"))
    s.set_agent_context("test_agent")
    return s


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def test_store_and_search(skill):
    r = run(skill.execute("store", {"content": "Python is great for AI", "category": "learnings", "tags": "python,ai"}))
    assert r.success
    assert r.data["category"] == "learnings"

    r = run(skill.execute("search", {"query": "python AI"}))
    assert r.success
    assert r.data["count"] >= 1
    assert "python" in r.data["results"][0]["content"].lower()


def test_recall_recent(skill):
    run(skill.execute("store", {"content": "First memory"}))
    run(skill.execute("store", {"content": "Second memory"}))
    r = run(skill.execute("recall_recent", {"limit": 5}))
    assert r.success
    assert r.data["count"] == 2


def test_record_outcome(skill):
    r = run(skill.execute("record_outcome", {
        "action": "deployed API",
        "outcome": "success",
        "lesson": "Always test before deploying"
    }))
    assert r.success
    assert r.data["outcome_type"] == "success"


def test_journal_entry(skill):
    r = run(skill.execute("journal_entry", {"entry": "Today I built a memory system", "session_id": "s1"}))
    assert r.success
    assert r.data["session_id"] == "s1"


def test_stats(skill):
    run(skill.execute("store", {"content": "test mem", "category": "learnings"}))
    r = run(skill.execute("stats", {}))
    assert r.success
    assert r.data["total_entries"] == 1


def test_get_lessons(skill):
    run(skill.execute("store", {"content": "Important lesson about testing", "category": "learnings", "relevance": 0.9}))
    run(skill.execute("record_outcome", {"action": "testing", "outcome": "success", "lesson": "Always test"}))
    r = run(skill.execute("get_lessons", {"topic": "testing"}))
    assert r.success
    assert r.data["count"] >= 1


def test_delete(skill):
    r = run(skill.execute("store", {"content": "delete me"}))
    mid = r.data["id"]
    r = run(skill.execute("delete", {"memory_id": mid}))
    assert r.success


def test_update_relevance(skill):
    r = run(skill.execute("store", {"content": "update me", "relevance": 0.3}))
    mid = r.data["id"]
    r = run(skill.execute("update_relevance", {"memory_id": mid, "relevance": 0.9}))
    assert r.success
    assert r.data["new_relevance"] == 0.9


def test_consolidate(skill):
    for i in range(10):
        run(skill.execute("store", {"content": f"Memory {i}", "category": "general"}))
    r = run(skill.execute("consolidate", {"category": "general", "keep_top": 3}))
    assert r.success
    assert r.data["consolidated"] == 7


def test_check_credentials(skill):
    assert skill.check_credentials() is True


def test_unknown_action(skill):
    r = run(skill.execute("nonexistent", {}))
    assert not r.success


def test_store_empty_content(skill):
    r = run(skill.execute("store", {"content": ""}))
    assert not r.success
