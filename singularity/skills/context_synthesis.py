#!/usr/bin/env python3
"""
ContextSynthesis Skill - Cross-skill context aggregation for informed decisions.

This is the "working memory" of the autonomous agent. Before each major decision,
the agent needs to know:
- What are my active goals and their progress?
- What recent performance patterns should I be aware of?
- What adaptations have I applied? Are they working?
- What experiments are running?
- What strategic priorities did I identify?
- What did I learn recently?

Without this, each decision cycle starts "cold" — the LLM sees only the
current prompt and last few messages. ContextSynthesis reads from all
persistent data stores and produces a compact, prioritized briefing
that can be injected into the agent's working context.

This bridges the gap between having many isolated skills with their own
data stores and actually using that accumulated knowledge for decisions.

Part of the Self-Improvement pillar: meta-cognition and context awareness.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from .base import Skill, SkillResult, SkillManifest, SkillAction


DATA_DIR = Path(__file__).parent.parent / "data"

# Data source file paths
GOALS_FILE = DATA_DIR / "goals.json"
STRATEGY_FILE = DATA_DIR / "strategy.json"
FEEDBACK_FILE = DATA_DIR / "feedback_loop.json"
PERFORMANCE_FILE = DATA_DIR / "performance.json"
EXPERIMENT_FILE = DATA_DIR / "experiments.json"
CONTEXT_FILE = DATA_DIR / "context_synthesis.json"

# Limits for briefing compactness
MAX_ACTIVE_GOALS = 5
MAX_RECENT_ADAPTATIONS = 3
MAX_RECENT_EXPERIMENTS = 3
MAX_PERFORMANCE_SKILLS = 5
MAX_LEARNINGS = 5
MAX_BRIEFING_TOKENS = 2000  # Approximate, for budget awareness
MAX_SNAPSHOTS = 50


class ContextSynthesisSkill(Skill):
    """
    Aggregates context from all agent data sources into a compact briefing.

    The agent calls context:briefing at the start of each decision cycle
    to get a synthesized view of its current operational state. This enables
    informed decision-making without the LLM needing to query each skill
    individually.

    Actions:
    - briefing: Generate a compact context briefing from all sources
    - snapshot: Save current context state for later comparison
    - diff: Compare current context with a previous snapshot
    - sources: List available data sources and their freshness
    - focus: Set a focus area to prioritize in briefings
    - history: View past briefings and how context evolved
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._focus_area: Optional[str] = None
        self._append_prompt_fn: Optional[Callable[[str], None]] = None
        self._ensure_data()

    def set_cognition_hooks(self, append_prompt: Callable[[str], None] = None):
        """Connect to cognition engine for injecting briefings."""
        self._append_prompt_fn = append_prompt

    def _ensure_data(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        if not CONTEXT_FILE.exists():
            self._save(self._default_state())

    def _default_state(self) -> Dict:
        return {
            "snapshots": [],
            "briefing_history": [],
            "focus_area": None,
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
        }

    def _load(self) -> Dict:
        try:
            with open(CONTEXT_FILE, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return self._default_state()

    def _save(self, data: Dict):
        data["last_updated"] = datetime.now().isoformat()
        # Trim history
        if len(data.get("snapshots", [])) > MAX_SNAPSHOTS:
            data["snapshots"] = data["snapshots"][-MAX_SNAPSHOTS:]
        if len(data.get("briefing_history", [])) > MAX_SNAPSHOTS:
            data["briefing_history"] = data["briefing_history"][-MAX_SNAPSHOTS:]
        with open(CONTEXT_FILE, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def _load_json(self, filepath: Path) -> Dict:
        """Safely load a JSON data file."""
        try:
            with open(filepath, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="context",
            name="Context Synthesis",
            version="1.0.0",
            category="meta",
            description="Aggregate cross-skill context into compact briefings for informed decisions",
            actions=[
                SkillAction(
                    name="briefing",
                    description="Generate a compact context briefing from all data sources",
                    parameters={
                        "include_performance": {
                            "type": "boolean",
                            "required": False,
                            "description": "Include performance stats (default: true)",
                        },
                        "include_goals": {
                            "type": "boolean",
                            "required": False,
                            "description": "Include active goals (default: true)",
                        },
                        "include_strategy": {
                            "type": "boolean",
                            "required": False,
                            "description": "Include strategic context (default: true)",
                        },
                        "inject_to_prompt": {
                            "type": "boolean",
                            "required": False,
                            "description": "Inject briefing into agent prompt (default: false)",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="snapshot",
                    description="Save current context state for later comparison",
                    parameters={
                        "label": {
                            "type": "string",
                            "required": False,
                            "description": "Optional label for this snapshot",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="diff",
                    description="Compare current context with a previous snapshot",
                    parameters={
                        "snapshot_id": {
                            "type": "string",
                            "required": False,
                            "description": "Snapshot ID to compare against (default: latest)",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="sources",
                    description="List available data sources and their freshness",
                    parameters={},
                    estimated_cost=0,
                ),
                SkillAction(
                    name="focus",
                    description="Set a focus area to prioritize in briefings",
                    parameters={
                        "area": {
                            "type": "string",
                            "required": True,
                            "description": "Focus area: self_improvement, revenue, replication, goal_setting, or 'clear'",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="history",
                    description="View recent briefings and how context evolved",
                    parameters={
                        "count": {
                            "type": "number",
                            "required": False,
                            "description": "Number of recent briefings to show (default: 5)",
                        },
                    },
                    estimated_cost=0,
                ),
            ],
            required_credentials=[],
        )

    def check_credentials(self) -> bool:
        return True

    async def execute(self, action: str, params: Dict) -> SkillResult:
        handlers = {
            "briefing": self._briefing,
            "snapshot": self._snapshot,
            "diff": self._diff,
            "sources": self._sources,
            "focus": self._focus,
            "history": self._history,
        }
        handler = handlers.get(action)
        if handler:
            return handler(params)
        return SkillResult(success=False, message=f"Unknown action: {action}")

    # ── Briefing ──────────────────────────────────────────────────────

    def _briefing(self, params: Dict) -> SkillResult:
        """Generate a compact context briefing from all data sources."""
        include_perf = params.get("include_performance", True)
        include_goals = params.get("include_goals", True)
        include_strategy = params.get("include_strategy", True)
        inject = params.get("inject_to_prompt", False)

        sections = []
        source_stats = {}

        # 1. Active Goals
        if include_goals:
            goals_section, goals_meta = self._extract_goals()
            if goals_section:
                sections.append(goals_section)
                source_stats["goals"] = goals_meta

        # 2. Strategic Context
        if include_strategy:
            strategy_section, strategy_meta = self._extract_strategy()
            if strategy_section:
                sections.append(strategy_section)
                source_stats["strategy"] = strategy_meta

        # 3. Performance Patterns
        if include_perf:
            perf_section, perf_meta = self._extract_performance()
            if perf_section:
                sections.append(perf_section)
                source_stats["performance"] = perf_meta

        # 4. Active Adaptations
        adapt_section, adapt_meta = self._extract_adaptations()
        if adapt_section:
            sections.append(adapt_section)
            source_stats["adaptations"] = adapt_meta

        # 5. Running Experiments
        exp_section, exp_meta = self._extract_experiments()
        if exp_section:
            sections.append(exp_section)
            source_stats["experiments"] = exp_meta

        # 6. Focus area emphasis
        ctx_data = self._load()
        focus = ctx_data.get("focus_area") or self._focus_area
        if focus:
            sections.insert(0, f"[FOCUS: {focus.upper()}] — Prioritize actions related to {focus}.")

        # Assemble briefing
        if not sections:
            briefing_text = "No operational context available. All data sources are empty."
        else:
            briefing_text = "\n\n".join(sections)

        # Record briefing
        briefing_record = {
            "timestamp": datetime.now().isoformat(),
            "sections_count": len(sections),
            "sources": list(source_stats.keys()),
            "focus": focus,
            "char_count": len(briefing_text),
        }
        ctx_data["briefing_history"].append(briefing_record)
        self._save(ctx_data)

        # Optionally inject into prompt
        if inject and self._append_prompt_fn and sections:
            prompt_block = f"\n=== CONTEXT BRIEFING ({datetime.now().strftime('%H:%M')}) ===\n{briefing_text}\n=== END BRIEFING ==="
            self._append_prompt_fn(prompt_block)

        return SkillResult(
            success=True,
            message=f"Briefing generated from {len(source_stats)} sources ({len(briefing_text)} chars)",
            data={
                "briefing": briefing_text,
                "sources": source_stats,
                "focus": focus,
                "injected": inject and self._append_prompt_fn is not None,
            },
        )

    def _extract_goals(self) -> tuple:
        """Extract active goals summary."""
        data = self._load_json(GOALS_FILE)
        goals = data.get("goals", [])
        active = [g for g in goals if g.get("status") == "active"]

        if not active:
            return ("", {})

        # Sort by priority
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        active.sort(key=lambda g: priority_order.get(g.get("priority", "medium"), 2))

        lines = ["ACTIVE GOALS:"]
        for g in active[:MAX_ACTIVE_GOALS]:
            priority = g.get("priority", "medium")
            title = g.get("title", "Untitled")
            pillar = g.get("pillar", "other")
            # Compute milestone progress
            milestones = g.get("milestones", [])
            completed = sum(1 for m in milestones if m.get("completed"))
            total = len(milestones)
            progress = f"{completed}/{total}" if total > 0 else "no milestones"

            lines.append(f"  [{priority.upper()}] {title} ({pillar}) — {progress}")

        # Completion stats
        completed_goals = data.get("completed_goals", [])
        meta = {
            "active": len(active),
            "shown": min(len(active), MAX_ACTIVE_GOALS),
            "completed_total": len(completed_goals),
        }

        return ("\n".join(lines), meta)

    def _extract_strategy(self) -> tuple:
        """Extract latest strategic assessment."""
        data = self._load_json(STRATEGY_FILE)
        assessments = data.get("assessments", [])
        journal = data.get("journal", [])

        if not assessments and not journal:
            return ("", {})

        lines = ["STRATEGIC CONTEXT:"]

        # Latest assessment scores
        if assessments:
            latest = assessments[-1]
            scores = latest.get("scores", {})
            if scores:
                weakest = min(scores.items(), key=lambda x: x[1]) if scores else None
                strongest = max(scores.items(), key=lambda x: x[1]) if scores else None
                lines.append(f"  Pillar scores: {json.dumps(scores)}")
                if weakest:
                    lines.append(f"  Weakest: {weakest[0]} ({weakest[1]})")
                if strongest:
                    lines.append(f"  Strongest: {strongest[0]} ({strongest[1]})")

            diagnosis = latest.get("diagnosis", "")
            if diagnosis:
                lines.append(f"  Diagnosis: {diagnosis[:200]}")

        # Latest journal entry
        if journal:
            latest_entry = journal[-1]
            entry_text = latest_entry.get("entry", "")
            if entry_text:
                lines.append(f"  Latest insight: {entry_text[:200]}")

        meta = {
            "assessments": len(assessments),
            "journal_entries": len(journal),
        }

        return ("\n".join(lines), meta)

    def _extract_performance(self) -> tuple:
        """Extract recent performance patterns."""
        data = self._load_json(PERFORMANCE_FILE)
        records = data.get("records", [])

        if not records:
            return ("", {})

        # Only look at recent records (last 24h)
        cutoff = datetime.now() - timedelta(hours=24)
        recent = []
        for r in records:
            try:
                ts = datetime.fromisoformat(r.get("timestamp", ""))
                if ts >= cutoff:
                    recent.append(r)
            except (ValueError, TypeError):
                continue

        if not recent:
            return ("", {})

        # Per-skill summary
        from collections import defaultdict
        stats = defaultdict(lambda: {"total": 0, "success": 0, "cost": 0.0})
        for r in recent:
            key = f"{r.get('skill_id', '?')}:{r.get('action', '?')}"
            s = stats[key]
            s["total"] += 1
            if r.get("success"):
                s["success"] += 1
            s["cost"] += r.get("cost_usd", 0.0)

        # Sort by total actions, take top N
        sorted_skills = sorted(stats.items(), key=lambda x: x[1]["total"], reverse=True)

        lines = [f"PERFORMANCE (last 24h, {len(recent)} actions):"]
        total_cost = sum(s["cost"] for _, s in sorted_skills)
        overall_success = sum(s["success"] for _, s in sorted_skills)
        overall_total = sum(s["total"] for _, s in sorted_skills)
        overall_rate = overall_success / overall_total if overall_total > 0 else 0

        lines.append(f"  Overall: {overall_rate:.0%} success, ${total_cost:.4f} total cost")

        # Flag problematic skills
        for key, s in sorted_skills[:MAX_PERFORMANCE_SKILLS]:
            rate = s["success"] / s["total"] if s["total"] > 0 else 0
            flag = " ⚠" if rate < 0.5 and s["total"] >= 3 else ""
            lines.append(f"  {key}: {s['total']} actions, {rate:.0%} success, ${s['cost']:.4f}{flag}")

        meta = {
            "total_actions": len(recent),
            "skills_tracked": len(stats),
            "overall_success_rate": round(overall_rate, 3),
            "total_cost": round(total_cost, 4),
        }

        return ("\n".join(lines), meta)

    def _extract_adaptations(self) -> tuple:
        """Extract active adaptations from feedback loop."""
        data = self._load_json(FEEDBACK_FILE)
        adaptations = data.get("adaptations", [])
        outcomes = data.get("adaptation_outcomes", [])

        if not adaptations:
            return ("", {})

        applied = [a for a in adaptations if a.get("applied") and not a.get("reverted")]
        pending = [a for a in adaptations if not a.get("applied")]

        if not applied and not pending:
            return ("", {})

        lines = ["ACTIVE ADAPTATIONS:"]

        for a in applied[-MAX_RECENT_ADAPTATIONS:]:
            action = a.get("action", "?")[:120]
            lines.append(f"  [ACTIVE] {action}")

        if pending:
            lines.append(f"  ({len(pending)} pending adaptations awaiting application)")

        # Outcome summary
        improved = sum(1 for o in outcomes if o.get("verdict") == "improved")
        degraded = sum(1 for o in outcomes if o.get("verdict") == "degraded")
        if improved or degraded:
            lines.append(f"  Track record: {improved} improved, {degraded} degraded")

        meta = {
            "applied": len(applied),
            "pending": len(pending),
            "improved": improved,
            "degraded": degraded,
        }

        return ("\n".join(lines), meta)

    def _extract_experiments(self) -> tuple:
        """Extract running experiments."""
        data = self._load_json(EXPERIMENT_FILE)
        experiments = data.get("experiments", [])

        if not experiments:
            return ("", {})

        running = [e for e in experiments if e.get("status") == "running"]
        if not running:
            return ("", {})

        lines = ["RUNNING EXPERIMENTS:"]
        for e in running[:MAX_RECENT_EXPERIMENTS]:
            name = e.get("name", "Unnamed")
            hypothesis = e.get("hypothesis", "")[:100]
            trials = e.get("total_trials", 0)
            lines.append(f"  {name}: {hypothesis} ({trials} trials)")

        meta = {
            "running": len(running),
            "total": len(experiments),
        }

        return ("\n".join(lines), meta)

    # ── Snapshot ──────────────────────────────────────────────────────

    def _snapshot(self, params: Dict) -> SkillResult:
        """Save current context state for later comparison."""
        label = params.get("label", "")

        # Gather current state from all sources
        state = {
            "goals": self._summarize_goals(),
            "strategy": self._summarize_strategy(),
            "performance": self._summarize_performance(),
            "adaptations": self._summarize_adaptations(),
            "experiments": self._summarize_experiments(),
        }

        snapshot = {
            "id": f"snap_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "label": label or "auto",
            "timestamp": datetime.now().isoformat(),
            "state": state,
        }

        ctx_data = self._load()
        ctx_data["snapshots"].append(snapshot)
        self._save(ctx_data)

        return SkillResult(
            success=True,
            message=f"Snapshot saved: {snapshot['id']}",
            data={"snapshot_id": snapshot["id"], "label": snapshot["label"]},
        )

    def _summarize_goals(self) -> Dict:
        data = self._load_json(GOALS_FILE)
        goals = data.get("goals", [])
        active = [g for g in goals if g.get("status") == "active"]
        return {
            "active_count": len(active),
            "completed_count": len(data.get("completed_goals", [])),
            "active_titles": [g.get("title", "") for g in active[:5]],
        }

    def _summarize_strategy(self) -> Dict:
        data = self._load_json(STRATEGY_FILE)
        assessments = data.get("assessments", [])
        if not assessments:
            return {"scores": {}, "assessment_count": 0}
        latest = assessments[-1]
        return {
            "scores": latest.get("scores", {}),
            "assessment_count": len(assessments),
        }

    def _summarize_performance(self) -> Dict:
        data = self._load_json(PERFORMANCE_FILE)
        records = data.get("records", [])
        cutoff = datetime.now() - timedelta(hours=24)
        recent = []
        for r in records:
            try:
                ts = datetime.fromisoformat(r.get("timestamp", ""))
                if ts >= cutoff:
                    recent.append(r)
            except (ValueError, TypeError):
                continue

        success = sum(1 for r in recent if r.get("success"))
        total = len(recent)
        return {
            "actions_24h": total,
            "success_rate": round(success / total, 3) if total > 0 else 0,
            "total_cost": round(sum(r.get("cost_usd", 0) for r in recent), 4),
        }

    def _summarize_adaptations(self) -> Dict:
        data = self._load_json(FEEDBACK_FILE)
        adaptations = data.get("adaptations", [])
        applied = sum(1 for a in adaptations if a.get("applied"))
        return {"total": len(adaptations), "applied": applied}

    def _summarize_experiments(self) -> Dict:
        data = self._load_json(EXPERIMENT_FILE)
        experiments = data.get("experiments", [])
        running = sum(1 for e in experiments if e.get("status") == "running")
        return {"total": len(experiments), "running": running}

    # ── Diff ──────────────────────────────────────────────────────────

    def _diff(self, params: Dict) -> SkillResult:
        """Compare current context with a previous snapshot."""
        ctx_data = self._load()
        snapshots = ctx_data.get("snapshots", [])

        if not snapshots:
            return SkillResult(
                success=False,
                message="No snapshots available. Run context:snapshot first.",
            )

        snapshot_id = params.get("snapshot_id", "")
        if snapshot_id:
            target = next((s for s in snapshots if s["id"] == snapshot_id), None)
            if not target:
                return SkillResult(success=False, message=f"Snapshot not found: {snapshot_id}")
        else:
            target = snapshots[-1]

        # Current state
        current = {
            "goals": self._summarize_goals(),
            "strategy": self._summarize_strategy(),
            "performance": self._summarize_performance(),
            "adaptations": self._summarize_adaptations(),
            "experiments": self._summarize_experiments(),
        }

        prev = target.get("state", {})

        # Compute diffs
        changes = []

        # Goals diff
        prev_goals = prev.get("goals", {})
        curr_goals = current["goals"]
        if curr_goals.get("active_count", 0) != prev_goals.get("active_count", 0):
            changes.append(
                f"Goals: {prev_goals.get('active_count', 0)} → {curr_goals.get('active_count', 0)} active"
            )
        if curr_goals.get("completed_count", 0) != prev_goals.get("completed_count", 0):
            changes.append(
                f"Completed goals: {prev_goals.get('completed_count', 0)} → {curr_goals.get('completed_count', 0)}"
            )

        # Strategy diff
        prev_strat = prev.get("strategy", {})
        curr_strat = current["strategy"]
        prev_scores = prev_strat.get("scores", {})
        curr_scores = curr_strat.get("scores", {})
        for pillar in set(list(prev_scores.keys()) + list(curr_scores.keys())):
            old_val = prev_scores.get(pillar, 0)
            new_val = curr_scores.get(pillar, 0)
            if old_val != new_val:
                delta = new_val - old_val
                arrow = "↑" if delta > 0 else "↓"
                changes.append(f"Strategy {pillar}: {old_val} → {new_val} ({arrow}{abs(delta)})")

        # Performance diff
        prev_perf = prev.get("performance", {})
        curr_perf = current["performance"]
        if curr_perf.get("actions_24h", 0) != prev_perf.get("actions_24h", 0):
            changes.append(
                f"Actions (24h): {prev_perf.get('actions_24h', 0)} → {curr_perf.get('actions_24h', 0)}"
            )

        # Adaptations diff
        prev_adapt = prev.get("adaptations", {})
        curr_adapt = current["adaptations"]
        if curr_adapt.get("applied", 0) != prev_adapt.get("applied", 0):
            changes.append(
                f"Applied adaptations: {prev_adapt.get('applied', 0)} → {curr_adapt.get('applied', 0)}"
            )

        if not changes:
            changes.append("No significant changes since snapshot.")

        return SkillResult(
            success=True,
            message=f"Compared with snapshot {target['id']} ({target.get('label', '')}) from {target['timestamp']}",
            data={
                "snapshot_id": target["id"],
                "snapshot_timestamp": target["timestamp"],
                "changes": changes,
                "current": current,
                "previous": prev,
            },
        )

    # ── Sources ───────────────────────────────────────────────────────

    def _sources(self, params: Dict) -> SkillResult:
        """List available data sources and their freshness."""
        sources = {}
        source_files = {
            "goals": GOALS_FILE,
            "strategy": STRATEGY_FILE,
            "feedback_loop": FEEDBACK_FILE,
            "performance": PERFORMANCE_FILE,
            "experiments": EXPERIMENT_FILE,
        }

        for name, filepath in source_files.items():
            if filepath.exists():
                try:
                    data = self._load_json(filepath)
                    last_updated = data.get("last_updated", "unknown")
                    sources[name] = {
                        "exists": True,
                        "last_updated": last_updated,
                        "file": str(filepath),
                    }

                    # Count key entries
                    if name == "goals":
                        active = [g for g in data.get("goals", []) if g.get("status") == "active"]
                        sources[name]["active_entries"] = len(active)
                    elif name == "performance":
                        sources[name]["total_records"] = len(data.get("records", []))
                    elif name == "feedback_loop":
                        sources[name]["total_adaptations"] = len(data.get("adaptations", []))
                    elif name == "experiments":
                        sources[name]["total_experiments"] = len(data.get("experiments", []))
                    elif name == "strategy":
                        sources[name]["total_assessments"] = len(data.get("assessments", []))
                except Exception:
                    sources[name] = {"exists": True, "error": "Failed to read"}
            else:
                sources[name] = {"exists": False}

        available = sum(1 for s in sources.values() if s.get("exists"))
        return SkillResult(
            success=True,
            message=f"{available}/{len(sources)} data sources available",
            data={"sources": sources},
        )

    # ── Focus ─────────────────────────────────────────────────────────

    def _focus(self, params: Dict) -> SkillResult:
        """Set a focus area to prioritize in briefings."""
        area = params.get("area", "").strip().lower()
        valid_areas = ["self_improvement", "revenue", "replication", "goal_setting", "clear"]

        if area not in valid_areas:
            return SkillResult(
                success=False,
                message=f"Invalid focus area. Choose from: {', '.join(valid_areas)}",
            )

        ctx_data = self._load()

        if area == "clear":
            ctx_data["focus_area"] = None
            self._focus_area = None
            self._save(ctx_data)
            return SkillResult(success=True, message="Focus area cleared")

        ctx_data["focus_area"] = area
        self._focus_area = area
        self._save(ctx_data)

        return SkillResult(
            success=True,
            message=f"Focus set to: {area}. Briefings will prioritize {area}-related context.",
            data={"focus": area},
        )

    # ── History ───────────────────────────────────────────────────────

    def _history(self, params: Dict) -> SkillResult:
        """View recent briefings and how context evolved."""
        count = int(params.get("count", 5))
        ctx_data = self._load()
        history = ctx_data.get("briefing_history", [])

        recent = history[-count:] if history else []

        if not recent:
            return SkillResult(
                success=True,
                message="No briefing history yet. Run context:briefing first.",
                data={"history": [], "total": 0},
            )

        return SkillResult(
            success=True,
            message=f"Showing {len(recent)} of {len(history)} briefings",
            data={"history": recent, "total": len(history)},
        )
