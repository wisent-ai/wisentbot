"""
AdaptiveToolSelector — Prioritizes and filters tools based on usage history.

Tracks which tools the agent uses, their success rates, and recency.
Sorts tools so the most relevant appear first in the LLM prompt,
improving decision quality and reducing wasted tokens.

Serves: Self-Improvement (better decisions), Revenue (lower token costs)
"""

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class ToolStats:
    """Statistics for a single tool."""

    def __init__(self):
        self.total_uses = 0
        self.successes = 0
        self.failures = 0
        self.last_used = 0.0  # timestamp
        self.total_duration = 0.0

    @property
    def success_rate(self) -> float:
        if self.total_uses == 0:
            return 0.5  # neutral default for unused tools
        return self.successes / self.total_uses

    @property
    def avg_duration(self) -> float:
        if self.total_uses == 0:
            return 0.0
        return self.total_duration / self.total_uses

    def to_dict(self) -> dict:
        return {
            "total_uses": self.total_uses,
            "successes": self.successes,
            "failures": self.failures,
            "last_used": self.last_used,
            "total_duration": self.total_duration,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ToolStats":
        ts = cls()
        ts.total_uses = d.get("total_uses", 0)
        ts.successes = d.get("successes", 0)
        ts.failures = d.get("failures", 0)
        ts.last_used = d.get("last_used", 0.0)
        ts.total_duration = d.get("total_duration", 0.0)
        return ts


# Tools that should always be visible regardless of usage
ESSENTIAL_TOOLS = {
    "filesystem:view_file",
    "filesystem:write_file",
    "filesystem:ls",
    "filesystem:glob",
    "filesystem:grep",
    "shell:bash",
    "self_modify:view_prompt",
}

# Tool categories for grouping
TOOL_CATEGORIES = {
    "filesystem": "file_ops",
    "shell": "execution",
    "github": "vcs",
    "twitter": "social",
    "content": "content",
    "browser": "web",
    "email": "communication",
    "vercel": "deployment",
    "namecheap": "domains",
    "mcp_client": "mcp",
    "request": "http",
    "self_modify": "self",
    "steering": "self",
    "memory": "memory",
    "orchestrator": "orchestration",
    "crypto": "finance",
}


class AdaptiveToolSelector:
    """
    Tracks tool usage and prioritizes tools for the LLM prompt.

    Features:
    - Tracks per-tool usage count, success rate, and recency
    - Sorts tools by relevance score (combo of frequency + success + recency)
    - Can hide rarely-used tools after a warmup period to save tokens
    - Persists stats to disk for cross-session learning
    - Always keeps essential tools visible
    """

    def __init__(
        self,
        persistence_path: Optional[str] = None,
        warmup_cycles: int = 20,
        max_tools: int = 0,  # 0 = no limit, show all
        recency_weight: float = 0.3,
        frequency_weight: float = 0.4,
        success_weight: float = 0.3,
    ):
        self.stats: Dict[str, ToolStats] = defaultdict(ToolStats)
        self.warmup_cycles = warmup_cycles
        self.max_tools = max_tools
        self.recency_weight = recency_weight
        self.frequency_weight = frequency_weight
        self.success_weight = success_weight
        self.cycle_count = 0

        self.persistence_path = None
        if persistence_path:
            self.persistence_path = Path(persistence_path)
            self._load()

    def record_usage(
        self, tool_name: str, success: bool, duration: float = 0.0
    ) -> None:
        """Record a tool usage event."""
        stats = self.stats[tool_name]
        stats.total_uses += 1
        if success:
            stats.successes += 1
        else:
            stats.failures += 1
        stats.last_used = time.time()
        stats.total_duration += duration
        self._save()

    def tick_cycle(self) -> None:
        """Increment the cycle counter."""
        self.cycle_count += 1

    def score_tool(self, tool_name: str) -> float:
        """
        Calculate a relevance score for a tool.

        Score components:
        - frequency: how often used relative to max
        - success: success rate (0-1)
        - recency: how recently used (decays over time)

        Returns 0.0-1.0
        """
        stats = self.stats.get(tool_name)

        # Essential tools get a bonus
        is_essential = tool_name in ESSENTIAL_TOOLS
        essential_bonus = 0.5 if is_essential else 0.0

        if stats is None or stats.total_uses == 0:
            # Unused tools get a neutral score (encourage exploration)
            return 0.5 + essential_bonus

        # Frequency: normalize against the most-used tool
        max_uses = max(s.total_uses for s in self.stats.values()) or 1
        freq_score = stats.total_uses / max_uses

        # Success rate
        success_score = stats.success_rate

        # Recency: exponential decay, half-life = 50 cycles
        now = time.time()
        age_seconds = now - stats.last_used
        # Approximate: each cycle ~10 seconds
        age_cycles = age_seconds / 10.0
        recency_score = 2.0 ** (-age_cycles / 50.0)

        score = (
            self.frequency_weight * freq_score
            + self.success_weight * success_score
            + self.recency_weight * recency_score
        )

        return min(score + essential_bonus, 2.0)

    def prioritize_tools(self, tools: List[Dict]) -> List[Dict]:
        """
        Sort tools by relevance score (highest first).

        If max_tools > 0 and past warmup period, only return top N tools
        (plus essential tools).

        Args:
            tools: List of tool dicts with 'name', 'description', 'parameters'

        Returns:
            Sorted (and optionally filtered) list of tools
        """
        self.tick_cycle()

        # Score each tool
        scored: List[Tuple[float, Dict]] = []
        for tool in tools:
            score = self.score_tool(tool["name"])
            scored.append((score, tool))

        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)

        # Apply max_tools filter if set and past warmup
        if self.max_tools > 0 and self.cycle_count > self.warmup_cycles:
            essential_tools = []
            other_tools = []
            for score, tool in scored:
                if tool["name"] in ESSENTIAL_TOOLS:
                    essential_tools.append(tool)
                else:
                    other_tools.append(tool)

            # Take top N non-essential tools + all essential
            result = essential_tools + other_tools[: self.max_tools - len(essential_tools)]
            return result

        return [tool for _, tool in scored]

    def get_summary(self) -> Dict:
        """Get a summary of tool usage for context injection."""
        if not self.stats:
            return {"total_actions": 0, "tools_used": 0}

        total_actions = sum(s.total_uses for s in self.stats.values())
        tools_used = sum(1 for s in self.stats.values() if s.total_uses > 0)

        # Top 5 tools
        top_tools = sorted(
            [(name, s) for name, s in self.stats.items() if s.total_uses > 0],
            key=lambda x: x[1].total_uses,
            reverse=True,
        )[:5]

        # Failing tools (>50% failure rate, used at least 3 times)
        failing = [
            (name, s)
            for name, s in self.stats.items()
            if s.total_uses >= 3 and s.success_rate < 0.5
        ]

        return {
            "total_actions": total_actions,
            "tools_used": tools_used,
            "top_tools": [
                {"name": n, "uses": s.total_uses, "success_rate": round(s.success_rate, 2)}
                for n, s in top_tools
            ],
            "failing_tools": [
                {"name": n, "uses": s.total_uses, "success_rate": round(s.success_rate, 2)}
                for n, s in failing
            ],
        }

    def get_context_string(self) -> str:
        """Get a compact string for injection into LLM context."""
        summary = self.get_summary()
        if summary["total_actions"] == 0:
            return ""

        lines = [f"Tool usage: {summary['total_actions']} actions across {summary['tools_used']} tools"]

        if summary["top_tools"]:
            top = ", ".join(
                f"{t['name']}({t['uses']}x, {int(t['success_rate']*100)}%)"
                for t in summary["top_tools"][:3]
            )
            lines.append(f"Most used: {top}")

        if summary["failing_tools"]:
            fail = ", ".join(
                f"{t['name']}({int(t['success_rate']*100)}% success)"
                for t in summary["failing_tools"]
            )
            lines.append(f"⚠ Unreliable tools: {fail}")

        return "\n".join(lines)

    def save(self) -> None:
        """Persist stats to disk."""
        if not self.persistence_path:
            return
        try:
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "cycle_count": self.cycle_count,
                "stats": {name: s.to_dict() for name, s in self.stats.items()},
            }
            with open(self.persistence_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    # Keep backwards compat
    _save = save

    def _load(self) -> None:
        """Load stats from disk."""
        if not self.persistence_path or not self.persistence_path.exists():
            return
        try:
            with open(self.persistence_path) as f:
                data = json.load(f)
            self.cycle_count = data.get("cycle_count", 0)
            for name, sd in data.get("stats", {}).items():
                self.stats[name] = ToolStats.from_dict(sd)
        except Exception:
            pass
