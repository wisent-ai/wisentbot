"""Tests for AdaptiveToolSelector."""
import json
import tempfile
import time
from pathlib import Path
from singularity.tool_filter import AdaptiveToolSelector, ToolStats, ESSENTIAL_TOOLS


def _make_tools(names):
    return [{"name": n, "description": f"desc for {n}", "parameters": {}} for n in names]


def test_tool_stats_defaults():
    ts = ToolStats()
    assert ts.total_uses == 0
    assert ts.success_rate == 0.5
    assert ts.avg_duration == 0.0


def test_tool_stats_success_rate():
    ts = ToolStats()
    ts.total_uses = 10
    ts.successes = 7
    ts.failures = 3
    assert ts.success_rate == 0.7


def test_tool_stats_serialization():
    ts = ToolStats()
    ts.total_uses = 5
    ts.successes = 3
    ts.failures = 2
    ts.last_used = 100.0
    d = ts.to_dict()
    ts2 = ToolStats.from_dict(d)
    assert ts2.total_uses == 5
    assert ts2.successes == 3


def test_selector_record_usage():
    sel = AdaptiveToolSelector()
    sel.record_usage("shell:bash", True)
    sel.record_usage("shell:bash", False)
    assert sel.stats["shell:bash"].total_uses == 2
    assert sel.stats["shell:bash"].successes == 1


def test_selector_score_unused_tool():
    sel = AdaptiveToolSelector()
    score = sel.score_tool("unknown:action")
    assert score == 0.5  # neutral for unused


def test_selector_score_essential_bonus():
    sel = AdaptiveToolSelector()
    score = sel.score_tool("filesystem:view_file")
    assert score == 1.0  # 0.5 neutral + 0.5 essential bonus


def test_prioritize_tools_sorting():
    sel = AdaptiveToolSelector()
    # Use tool_a a lot successfully
    for _ in range(10):
        sel.record_usage("tool_a", True)
    # Use tool_b rarely and failing
    sel.record_usage("tool_b", False)

    tools = _make_tools(["tool_b", "tool_a", "tool_c"])
    result = sel.prioritize_tools(tools)
    names = [t["name"] for t in result]
    # tool_a should be first (most used, highest success)
    assert names[0] == "tool_a"


def test_prioritize_preserves_all_tools_during_warmup():
    sel = AdaptiveToolSelector(max_tools=2, warmup_cycles=100)
    tools = _make_tools(["a", "b", "c", "d"])
    result = sel.prioritize_tools(tools)
    assert len(result) == 4  # all preserved during warmup


def test_prioritize_limits_after_warmup():
    sel = AdaptiveToolSelector(max_tools=2, warmup_cycles=0)
    sel.cycle_count = 5
    tools = _make_tools(["a", "b", "c", "d"])
    # Use 'a' a lot
    for _ in range(10):
        sel.record_usage("a", True)
    result = sel.prioritize_tools(tools)
    assert len(result) == 2


def test_essential_tools_always_kept():
    sel = AdaptiveToolSelector(max_tools=2, warmup_cycles=0)
    sel.cycle_count = 5
    tools = _make_tools(["filesystem:view_file", "shell:bash", "rare_tool", "other"])
    for _ in range(10):
        sel.record_usage("rare_tool", True)
    result = sel.prioritize_tools(tools)
    names = [t["name"] for t in result]
    assert "filesystem:view_file" in names
    assert "shell:bash" in names


def test_persistence():
    with tempfile.TemporaryDirectory() as td:
        path = str(Path(td) / "stats.json")
        sel1 = AdaptiveToolSelector(persistence_path=path)
        sel1.record_usage("shell:bash", True)
        sel1.record_usage("shell:bash", True)
        sel1.cycle_count = 42
        sel1.save()  # explicitly save cycle_count

        sel2 = AdaptiveToolSelector(persistence_path=path)
        assert sel2.stats["shell:bash"].total_uses == 2
        assert sel2.cycle_count == 42


def test_get_summary_empty():
    sel = AdaptiveToolSelector()
    s = sel.get_summary()
    assert s["total_actions"] == 0
    assert s["tools_used"] == 0


def test_get_summary_with_data():
    sel = AdaptiveToolSelector()
    for _ in range(5):
        sel.record_usage("a", True)
    for _ in range(3):
        sel.record_usage("b", False)
    s = sel.get_summary()
    assert s["total_actions"] == 8
    assert s["tools_used"] == 2
    assert len(s["top_tools"]) == 2
    assert len(s["failing_tools"]) == 1


def test_get_context_string():
    sel = AdaptiveToolSelector()
    assert sel.get_context_string() == ""
    for _ in range(5):
        sel.record_usage("shell:bash", True)
    ctx = sel.get_context_string()
    assert "shell:bash" in ctx
    assert "5x" in ctx
