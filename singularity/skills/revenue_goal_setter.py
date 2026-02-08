#!/usr/bin/env python3
"""
RevenueGoalAutoSetterSkill - Auto-set revenue goals from forecast data.

This bridge reads forecast data from RevenueAnalyticsDashboardSkill and
auto-creates/updates revenue goals in GoalManagerSkill. It completes the
data-driven goal-setting feedback loop:

  revenue data → forecast → auto-set goals → track progress → adapt targets

Without this bridge, the agent must manually inspect dashboard forecasts and
create goals. With it, the agent autonomously:
- Sets daily/weekly revenue targets based on growth trends
- Creates breakeven goals when not yet profitable
- Escalates goals when growth exceeds expectations
- Downgrades goals when forecasts show declining revenue
- Tracks goal attainment against actual revenue snapshots

Pillar: Goal Setting (primary), Revenue Generation (data-driven targets)

Actions:
- evaluate: Read forecast + snapshots, decide if goals need creation/updating
- set_goal: Manually set a revenue goal with target amount and deadline
- status: View current revenue goals and their progress
- configure: Set thresholds for auto-goal creation (min growth, target margins)
- history: View past goal-setting decisions and outcomes
- sync: Force-sync current revenue data into active goal progress
"""

import json
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
from .base import Skill, SkillResult, SkillManifest, SkillAction

DATA_DIR = Path(__file__).parent.parent / "data"
SETTER_FILE = DATA_DIR / "revenue_goal_setter.json"
DASHBOARD_FILE = DATA_DIR / "revenue_analytics_dashboard.json"
GOALS_FILE = DATA_DIR / "goals.json"

MAX_HISTORY = 200


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _load_json(path: Path) -> Optional[Dict]:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _save_json(path: Path, data: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


DEFAULT_CONFIG = {
    "enabled": True,
    # Minimum snapshots before auto-setting goals
    "min_snapshots": 3,
    # Target margin above compute cost (1.5 = 50% profit margin)
    "target_margin_multiplier": 1.5,
    # When growth is positive, set target this many days ahead
    "target_horizon_days": 7,
    # Escalation: if actual revenue exceeds target by this %, raise the goal
    "escalation_threshold_pct": 25.0,
    # Downgrade: if forecast shows revenue dropping below target by this %, lower the goal
    "downgrade_threshold_pct": 20.0,
    # Auto-create breakeven goal if not profitable
    "auto_breakeven_goal": True,
    # Auto-create growth goal if profitable
    "auto_growth_goal": True,
    # Default priority for auto-created goals
    "default_priority": "high",
    # Maximum auto-created goals at once
    "max_auto_goals": 5,
    # Compute cost per hour (synced from dashboard config)
    "compute_cost_per_hour": 0.10,
}


class RevenueGoalAutoSetterSkill(Skill):
    """Auto-sets revenue goals from RevenueAnalyticsDashboard forecast data."""

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._ensure_data()

    def _ensure_data(self):
        SETTER_FILE.parent.mkdir(parents=True, exist_ok=True)
        if not SETTER_FILE.exists():
            _save_json(SETTER_FILE, self._default_state())

    def _default_state(self) -> Dict:
        return {
            "config": dict(DEFAULT_CONFIG),
            "auto_goals": [],  # Goals created by this skill
            "decisions": [],  # Log of evaluate decisions
            "created_at": _now_iso(),
        }

    def _load(self) -> Dict:
        data = _load_json(SETTER_FILE)
        return data if data else self._default_state()

    def _save(self, data: Dict):
        data["last_updated"] = _now_iso()
        # Cap history
        if len(data.get("decisions", [])) > MAX_HISTORY:
            data["decisions"] = data["decisions"][-MAX_HISTORY:]
        _save_json(SETTER_FILE, data)

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="revenue_goal_setter",
            name="Revenue Goal Auto-Setter",
            version="1.0.0",
            category="goal_setting",
            description="Auto-set revenue goals from dashboard forecast data",
            actions=[
                SkillAction(
                    name="evaluate",
                    description="Read forecast + snapshots, decide if goals need creation/updating",
                    parameters={},
                ),
                SkillAction(
                    name="set_goal",
                    description="Manually set a revenue goal with target amount and deadline",
                    parameters={
                        "target_daily": {"type": "float", "required": True, "description": "Daily revenue target"},
                        "deadline_days": {"type": "int", "required": False, "description": "Days to achieve target"},
                        "title": {"type": "str", "required": False, "description": "Goal title override"},
                    },
                ),
                SkillAction(
                    name="status",
                    description="View current revenue goals and their progress",
                    parameters={},
                ),
                SkillAction(
                    name="configure",
                    description="Set thresholds for auto-goal creation",
                    parameters={
                        "key": {"type": "str", "required": True, "description": "Config key"},
                        "value": {"type": "any", "required": True, "description": "Config value"},
                    },
                ),
                SkillAction(
                    name="history",
                    description="View past goal-setting decisions and outcomes",
                    parameters={"limit": {"type": "int", "required": False, "description": "Max entries"}},
                ),
                SkillAction(
                    name="sync",
                    description="Force-sync current revenue data into active goal progress",
                    parameters={},
                ),
            ],
            required_credentials=[],
        )

    async def execute(self, action: str, params: Dict) -> SkillResult:
        handlers = {
            "evaluate": self._evaluate,
            "set_goal": self._set_goal,
            "status": self._status,
            "configure": self._configure,
            "history": self._history,
            "sync": self._sync,
        }
        handler = handlers.get(action)
        if handler:
            return handler(params)
        return SkillResult(success=False, message=f"Unknown action: {action}")

    def _get_forecast_data(self) -> Optional[Dict]:
        """Read forecast-relevant data from the revenue analytics dashboard."""
        dash = _load_json(DASHBOARD_FILE)
        if not dash:
            return None

        snapshots = dash.get("snapshots", [])
        config = dash.get("config", {})

        if len(snapshots) < 1:
            return None

        # Calculate growth rate from snapshots (same approach as dashboard)
        recent = snapshots[-20:]
        revenues = [s.get("total_revenue", 0) for s in recent]

        current_rev = revenues[-1] if revenues else 0

        # Linear regression for growth rate
        n = len(revenues)
        if n >= 3:
            x_mean = (n - 1) / 2
            y_mean = sum(revenues) / n
            numerator = sum((i - x_mean) * (revenues[i] - y_mean) for i in range(n))
            denominator = sum((i - x_mean) ** 2 for i in range(n))
            slope = numerator / denominator if denominator > 0 else 0
            daily_rate = slope * 24  # Assuming hourly snapshots
        else:
            slope = 0
            daily_rate = 0

        compute_cost_daily = config.get("compute_cost_per_hour", 0.10) * 24

        return {
            "current_revenue": current_rev,
            "daily_growth_rate": daily_rate,
            "growth_direction": "positive" if slope > 0 else "negative" if slope < 0 else "flat",
            "compute_cost_daily": compute_cost_daily,
            "snapshot_count": len(snapshots),
            "latest_snapshot": snapshots[-1] if snapshots else None,
        }

    def _get_active_revenue_goals(self) -> List[Dict]:
        """Get all active revenue goals from GoalManager."""
        goals_data = _load_json(GOALS_FILE)
        if not goals_data:
            return []
        return [
            g for g in goals_data.get("goals", [])
            if g.get("pillar") == "revenue" and g.get("status") == "active"
        ]

    def _get_auto_goal_ids(self, setter_data: Dict) -> set:
        """Get IDs of goals created by this skill."""
        return {g["goal_id"] for g in setter_data.get("auto_goals", []) if g.get("goal_id")}

    def _create_goal_in_manager(self, title: str, description: str, priority: str, deadline_hours: Optional[float] = None, milestones: Optional[List[str]] = None) -> Optional[str]:
        """Create a goal directly in the GoalManager data file."""
        goals_data = _load_json(GOALS_FILE)
        if not goals_data:
            goals_data = {
                "goals": [],
                "completed_goals": [],
                "session_log": [],
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
            }

        if len(goals_data.get("goals", [])) >= 200:
            return None

        goal_id = f"goal_{uuid.uuid4().hex[:8]}"
        now = datetime.now()

        ms = []
        if milestones:
            for i, mt in enumerate(milestones[:10]):
                ms.append({
                    "index": i,
                    "title": str(mt),
                    "completed": False,
                    "completed_at": None,
                })

        goal = {
            "id": goal_id,
            "title": title,
            "description": description,
            "pillar": "revenue",
            "priority": priority,
            "priority_score": {"critical": 4, "high": 3, "medium": 2, "low": 1}.get(priority, 2),
            "status": "active",
            "milestones": ms,
            "depends_on": [],
            "progress_notes": [f"Auto-created by RevenueGoalAutoSetter at {_now_iso()}"],
            "created_at": now.isoformat(),
            "deadline": (now + timedelta(hours=deadline_hours)).isoformat() if deadline_hours else None,
            "completed_at": None,
        }

        goals_data.setdefault("goals", []).append(goal)
        goals_data["last_updated"] = now.isoformat()
        _save_json(GOALS_FILE, goals_data)

        return goal_id

    def _update_goal_progress(self, goal_id: str, note: str):
        """Add a progress note to an existing goal."""
        goals_data = _load_json(GOALS_FILE)
        if not goals_data:
            return
        for g in goals_data.get("goals", []):
            if g.get("id") == goal_id:
                g.setdefault("progress_notes", []).append(note)
                goals_data["last_updated"] = datetime.now().isoformat()
                _save_json(GOALS_FILE, goals_data)
                return

    def _evaluate(self, params: Dict) -> SkillResult:
        """Read forecast + snapshots, decide if goals need creation/updating."""
        setter_data = self._load()
        config = setter_data.get("config", dict(DEFAULT_CONFIG))

        if not config.get("enabled", True):
            return SkillResult(success=True, message="Auto-setter is disabled", data={"enabled": False})

        forecast = self._get_forecast_data()
        if not forecast:
            return SkillResult(
                success=True,
                message="No dashboard data available. Use RevenueAnalyticsDashboard 'snapshot' action first.",
                data={"reason": "no_data"},
            )

        min_snaps = config.get("min_snapshots", 3)
        if forecast["snapshot_count"] < min_snaps:
            return SkillResult(
                success=True,
                message=f"Need at least {min_snaps} snapshots for auto-goal setting ({forecast['snapshot_count']} available).",
                data={"reason": "insufficient_data", "snapshots": forecast["snapshot_count"]},
            )

        active_rev_goals = self._get_active_revenue_goals()
        auto_goal_ids = self._get_auto_goal_ids(setter_data)
        auto_active = [g for g in active_rev_goals if g.get("id") in auto_goal_ids]

        current_rev = forecast["current_revenue"]
        daily_rate = forecast["daily_growth_rate"]
        cost_daily = forecast["compute_cost_daily"]
        margin_mult = config.get("target_margin_multiplier", 1.5)
        horizon = config.get("target_horizon_days", 7)
        max_auto = config.get("max_auto_goals", 5)
        priority = config.get("default_priority", "high")

        decisions = []
        goals_created = []
        goals_updated = []

        # --- Decision 1: Breakeven goal ---
        is_profitable = current_rev >= cost_daily
        has_breakeven_goal = any("breakeven" in g.get("title", "").lower() for g in auto_active)

        if not is_profitable and config.get("auto_breakeven_goal", True) and not has_breakeven_goal:
            if len(auto_active) < max_auto:
                # Calculate deadline based on growth rate
                if daily_rate > 0:
                    days_to_breakeven = max(1, int((cost_daily - current_rev) / daily_rate) + 1)
                else:
                    days_to_breakeven = 30  # Default if no growth

                gid = self._create_goal_in_manager(
                    title=f"Revenue Breakeven: ${cost_daily:.2f}/day",
                    description=(
                        f"Reach daily revenue of ${cost_daily:.2f} to cover compute costs. "
                        f"Current: ${current_rev:.4f}/day, Growth: ${daily_rate:.4f}/day. "
                        f"Estimated {days_to_breakeven} days at current growth."
                    ),
                    priority="critical",
                    deadline_hours=days_to_breakeven * 24,
                    milestones=[
                        f"Reach ${cost_daily * 0.25:.2f}/day (25% of breakeven)",
                        f"Reach ${cost_daily * 0.50:.2f}/day (50% of breakeven)",
                        f"Reach ${cost_daily * 0.75:.2f}/day (75% of breakeven)",
                        f"Achieve breakeven: ${cost_daily:.2f}/day",
                    ],
                )
                if gid:
                    setter_data.setdefault("auto_goals", []).append({
                        "goal_id": gid,
                        "type": "breakeven",
                        "target_daily": cost_daily,
                        "created_at": _now_iso(),
                    })
                    goals_created.append({"type": "breakeven", "goal_id": gid, "target": cost_daily})
                    decisions.append({
                        "action": "create_breakeven_goal",
                        "reason": f"Not profitable (${current_rev:.4f} < ${cost_daily:.2f})",
                        "goal_id": gid,
                    })

        # --- Decision 2: Growth goal ---
        has_growth_goal = any("growth" in g.get("title", "").lower() for g in auto_active)

        if is_profitable and config.get("auto_growth_goal", True) and not has_growth_goal:
            if len(auto_active) < max_auto:
                target_daily = cost_daily * margin_mult
                if daily_rate > 0:
                    days_to_target = max(1, int((target_daily - current_rev) / daily_rate) + 1)
                else:
                    days_to_target = horizon

                gid = self._create_goal_in_manager(
                    title=f"Revenue Growth: ${target_daily:.2f}/day",
                    description=(
                        f"Grow daily revenue to ${target_daily:.2f} ({margin_mult}x compute cost). "
                        f"Current: ${current_rev:.4f}/day, Growth: ${daily_rate:.4f}/day."
                    ),
                    priority=priority,
                    deadline_hours=days_to_target * 24,
                    milestones=[
                        f"Sustain ${current_rev * 1.25:.4f}/day (+25%)",
                        f"Reach ${target_daily * 0.5:.2f}/day (halfway)",
                        f"Reach ${target_daily:.2f}/day (target)",
                    ],
                )
                if gid:
                    setter_data.setdefault("auto_goals", []).append({
                        "goal_id": gid,
                        "type": "growth",
                        "target_daily": target_daily,
                        "created_at": _now_iso(),
                    })
                    goals_created.append({"type": "growth", "goal_id": gid, "target": target_daily})
                    decisions.append({
                        "action": "create_growth_goal",
                        "reason": f"Profitable (${current_rev:.4f} >= ${cost_daily:.2f}), targeting {margin_mult}x",
                        "goal_id": gid,
                    })

        # --- Decision 3: Escalation check ---
        esc_pct = config.get("escalation_threshold_pct", 25.0)
        for ag in auto_active:
            ag_meta = next((m for m in setter_data.get("auto_goals", []) if m["goal_id"] == ag.get("id")), None)
            if not ag_meta:
                continue
            target = ag_meta.get("target_daily", 0)
            if target > 0 and current_rev > target * (1 + esc_pct / 100):
                note = (
                    f"[ESCALATION] Revenue ${current_rev:.4f} exceeds target "
                    f"${target:.4f} by {((current_rev / target) - 1) * 100:.1f}%. "
                    f"Consider raising target."
                )
                self._update_goal_progress(ag.get("id"), note)
                goals_updated.append({"goal_id": ag.get("id"), "action": "escalation_note"})
                decisions.append({
                    "action": "escalation_note",
                    "reason": f"Revenue exceeds target by >{esc_pct}%",
                    "goal_id": ag.get("id"),
                })

        # --- Decision 4: Downgrade check ---
        dg_pct = config.get("downgrade_threshold_pct", 20.0)
        for ag in auto_active:
            ag_meta = next((m for m in setter_data.get("auto_goals", []) if m["goal_id"] == ag.get("id")), None)
            if not ag_meta:
                continue
            target = ag_meta.get("target_daily", 0)
            if target > 0 and daily_rate < 0:
                projected = current_rev + daily_rate * horizon
                if projected < target * (1 - dg_pct / 100):
                    note = (
                        f"[DOWNGRADE WARNING] Forecast shows ${projected:.4f}/day in {horizon} days, "
                        f"which is >{dg_pct}% below target ${target:.4f}. Revenue declining."
                    )
                    self._update_goal_progress(ag.get("id"), note)
                    goals_updated.append({"goal_id": ag.get("id"), "action": "downgrade_warning"})
                    decisions.append({
                        "action": "downgrade_warning",
                        "reason": f"Forecast projects below target by >{dg_pct}%",
                        "goal_id": ag.get("id"),
                    })

        # Record decisions
        if decisions:
            setter_data.setdefault("decisions", []).append({
                "timestamp": _now_iso(),
                "forecast_snapshot": {
                    "current_revenue": current_rev,
                    "daily_growth_rate": daily_rate,
                    "compute_cost_daily": cost_daily,
                },
                "decisions": decisions,
            })

        self._save(setter_data)

        total_actions = len(goals_created) + len(goals_updated)
        if total_actions == 0:
            msg = "No goal changes needed. "
            if is_profitable:
                msg += f"Revenue ${current_rev:.4f}/day above cost ${cost_daily:.2f}/day."
            else:
                msg += f"Revenue ${current_rev:.4f}/day below cost ${cost_daily:.2f}/day."
            if auto_active:
                msg += f" {len(auto_active)} auto-goal(s) active."
        else:
            msg = f"Created {len(goals_created)} goal(s), updated {len(goals_updated)} goal(s)."

        return SkillResult(
            success=True,
            message=msg,
            data={
                "goals_created": goals_created,
                "goals_updated": goals_updated,
                "decisions": decisions,
                "forecast": forecast,
                "is_profitable": is_profitable,
                "active_auto_goals": len(auto_active),
            },
        )

    def _set_goal(self, params: Dict) -> SkillResult:
        """Manually set a revenue goal."""
        target_daily = params.get("target_daily")
        if target_daily is None:
            return SkillResult(success=False, message="target_daily is required")
        try:
            target_daily = float(target_daily)
        except (TypeError, ValueError):
            return SkillResult(success=False, message="target_daily must be a number")

        if target_daily <= 0:
            return SkillResult(success=False, message="target_daily must be positive")

        deadline_days = params.get("deadline_days", 7)
        title = params.get("title", f"Revenue Target: ${target_daily:.2f}/day")

        setter_data = self._load()
        config = setter_data.get("config", dict(DEFAULT_CONFIG))
        priority = config.get("default_priority", "high")

        gid = self._create_goal_in_manager(
            title=title,
            description=f"Manual revenue target: ${target_daily:.2f}/day within {deadline_days} days.",
            priority=priority,
            deadline_hours=float(deadline_days) * 24,
        )

        if not gid:
            return SkillResult(success=False, message="Failed to create goal (max goals reached?)")

        setter_data.setdefault("auto_goals", []).append({
            "goal_id": gid,
            "type": "manual",
            "target_daily": target_daily,
            "created_at": _now_iso(),
        })

        setter_data.setdefault("decisions", []).append({
            "timestamp": _now_iso(),
            "decisions": [{"action": "manual_set_goal", "target_daily": target_daily, "goal_id": gid}],
        })

        self._save(setter_data)

        return SkillResult(
            success=True,
            message=f"Revenue goal created: ${target_daily:.2f}/day within {deadline_days} days",
            data={"goal_id": gid, "target_daily": target_daily, "deadline_days": deadline_days},
        )

    def _status(self, params: Dict) -> SkillResult:
        """View current revenue goals and progress."""
        setter_data = self._load()
        config = setter_data.get("config", dict(DEFAULT_CONFIG))
        auto_goals = setter_data.get("auto_goals", [])
        auto_ids = {g["goal_id"] for g in auto_goals}

        # Get active revenue goals
        active = self._get_active_revenue_goals()
        auto_active = [g for g in active if g.get("id") in auto_ids]
        manual_active = [g for g in active if g.get("id") not in auto_ids]

        # Get forecast
        forecast = self._get_forecast_data()

        goal_summaries = []
        for g in auto_active:
            meta = next((m for m in auto_goals if m["goal_id"] == g.get("id")), {})
            goal_summaries.append({
                "goal_id": g.get("id"),
                "title": g.get("title"),
                "type": meta.get("type", "unknown"),
                "target_daily": meta.get("target_daily"),
                "priority": g.get("priority"),
                "deadline": g.get("deadline"),
                "milestones_total": len(g.get("milestones", [])),
                "milestones_done": sum(1 for m in g.get("milestones", []) if m.get("completed")),
                "progress_notes_count": len(g.get("progress_notes", [])),
            })

        return SkillResult(
            success=True,
            message=f"{len(auto_active)} auto-goal(s), {len(manual_active)} other revenue goal(s) active",
            data={
                "auto_goals": goal_summaries,
                "other_revenue_goals": len(manual_active),
                "current_revenue": forecast["current_revenue"] if forecast else None,
                "daily_growth_rate": forecast["daily_growth_rate"] if forecast else None,
                "is_profitable": (forecast["current_revenue"] >= forecast["compute_cost_daily"]) if forecast else None,
                "config": {k: v for k, v in config.items() if k != "enabled"},
                "enabled": config.get("enabled", True),
            },
        )

    def _configure(self, params: Dict) -> SkillResult:
        """Update configuration."""
        key = params.get("key", "").strip()
        value = params.get("value")

        if not key:
            return SkillResult(success=False, message="key is required")

        if key not in DEFAULT_CONFIG:
            return SkillResult(
                success=False,
                message=f"Unknown config key: {key}. Valid keys: {list(DEFAULT_CONFIG.keys())}",
            )

        setter_data = self._load()
        config = setter_data.setdefault("config", dict(DEFAULT_CONFIG))

        # Type coercion
        expected = type(DEFAULT_CONFIG[key])
        try:
            if expected == bool:
                value = str(value).lower() in ("true", "1", "yes")
            elif expected == int:
                value = int(value)
            elif expected == float:
                value = float(value)
            else:
                value = str(value)
        except (TypeError, ValueError):
            return SkillResult(success=False, message=f"Cannot convert value to {expected.__name__}")

        old_value = config.get(key)
        config[key] = value
        self._save(setter_data)

        return SkillResult(
            success=True,
            message=f"Config updated: {key} = {value} (was {old_value})",
            data={"key": key, "value": value, "old_value": old_value},
        )

    def _history(self, params: Dict) -> SkillResult:
        """View past goal-setting decisions."""
        setter_data = self._load()
        decisions = setter_data.get("decisions", [])
        limit = min(params.get("limit", 20), MAX_HISTORY)

        recent = decisions[-limit:] if decisions else []

        return SkillResult(
            success=True,
            message=f"{len(recent)} decision(s) shown (of {len(decisions)} total)",
            data={"decisions": recent, "total": len(decisions)},
        )

    def _sync(self, params: Dict) -> SkillResult:
        """Force-sync current revenue data into active goal progress."""
        setter_data = self._load()
        forecast = self._get_forecast_data()

        if not forecast:
            return SkillResult(
                success=True,
                message="No dashboard data to sync.",
                data={"synced": 0},
            )

        auto_goals = setter_data.get("auto_goals", [])
        auto_ids = {g["goal_id"] for g in auto_goals}

        active = self._get_active_revenue_goals()
        auto_active = [g for g in active if g.get("id") in auto_ids]

        synced = 0
        for g in auto_active:
            meta = next((m for m in auto_goals if m["goal_id"] == g.get("id")), {})
            target = meta.get("target_daily", 0)
            current = forecast["current_revenue"]
            pct = (current / target * 100) if target > 0 else 0

            note = (
                f"[SYNC] Revenue: ${current:.4f}/day | Target: ${target:.4f}/day | "
                f"Progress: {pct:.1f}% | Growth: ${forecast['daily_growth_rate']:.4f}/day"
            )
            self._update_goal_progress(g.get("id"), note)

            # Auto-complete milestones based on revenue progress
            self._check_milestones(g, current, target)
            synced += 1

        return SkillResult(
            success=True,
            message=f"Synced revenue data to {synced} active goal(s)",
            data={
                "synced": synced,
                "current_revenue": forecast["current_revenue"],
                "daily_growth_rate": forecast["daily_growth_rate"],
            },
        )

    def _check_milestones(self, goal: Dict, current_rev: float, target: float):
        """Auto-complete milestones based on current revenue vs target thresholds."""
        if target <= 0:
            return

        goals_data = _load_json(GOALS_FILE)
        if not goals_data:
            return

        pct = current_rev / target
        for g in goals_data.get("goals", []):
            if g.get("id") != goal.get("id"):
                continue
            changed = False
            for ms in g.get("milestones", []):
                if ms.get("completed"):
                    continue
                # Check milestone text for percentage thresholds
                title = ms.get("title", "").lower()
                # Complete milestones for percentage-based markers
                if "25%" in title and pct >= 0.25:
                    ms["completed"] = True
                    ms["completed_at"] = _now_iso()
                    changed = True
                elif "50%" in title and pct >= 0.50:
                    ms["completed"] = True
                    ms["completed_at"] = _now_iso()
                    changed = True
                elif "75%" in title and pct >= 0.75:
                    ms["completed"] = True
                    ms["completed_at"] = _now_iso()
                    changed = True
                elif ("breakeven" in title or "target" in title) and pct >= 1.0:
                    ms["completed"] = True
                    ms["completed_at"] = _now_iso()
                    changed = True

            if changed:
                goals_data["last_updated"] = datetime.now().isoformat()
                _save_json(GOALS_FILE, goals_data)
            return
