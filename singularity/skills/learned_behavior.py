#!/usr/bin/env python3
"""
LearnedBehavior Skill - Persistent cross-session behavioral rules.

This is the critical missing piece in the Self-Improvement feedback loop:
while FeedbackLoopSkill generates adaptations and PerformanceTracker records
outcomes, those adaptations live only in memory and are lost when a session ends.

LearnedBehaviorSkill provides:
1. Persistent storage of behavioral rules that survive across sessions
2. Automatic loading of learned rules on session start
3. Cross-session effectiveness tracking for each rule
4. Automatic pruning of rules that don't improve performance
5. Rule categories (prompt_additions, skill_preferences, avoidance_rules, strategies)
6. Integration with FeedbackLoop to auto-persist successful adaptations
7. A "behavioral genome" that the agent evolves over time

The self-improvement loop becomes:
  act -> measure (PerformanceTracker) -> adapt (FeedbackLoop) -> persist (LearnedBehavior) -> act...

Part of the Self-Improvement pillar: enables persistent, cumulative learning.
"""

import json
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from .base import Skill, SkillResult, SkillManifest, SkillAction

BEHAVIOR_FILE = Path(__file__).parent.parent / "data" / "learned_behaviors.json"
MAX_RULES = 200
MAX_HISTORY = 500


class LearnedBehaviorSkill(Skill):
    """
    Persistent behavioral rule store for cross-session learning.

    Rules are categorized by type:
    - prompt_addition: Text appended to the system prompt
    - skill_preference: Preferred skill/action for a task type
    - avoidance_rule: Skill/action to avoid (with reason)
    - strategy: High-level behavioral strategy
    - heuristic: Learned decision-making shortcut

    Each rule tracks:
    - When it was created and by what mechanism
    - How many sessions it has been active in
    - Success/failure counts since adoption
    - Whether it's currently active or dormant
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._append_prompt_fn: Optional[Callable[[str], None]] = None
        self._get_prompt_fn: Optional[Callable[[], str]] = None
        self._ensure_data()

    def set_cognition_hooks(
        self,
        append_prompt: Callable[[str], None],
        get_prompt: Callable[[], str],
    ):
        """Connect to the agent's cognition engine for applying prompt rules."""
        self._append_prompt_fn = append_prompt
        self._get_prompt_fn = get_prompt

    def _ensure_data(self):
        BEHAVIOR_FILE.parent.mkdir(parents=True, exist_ok=True)
        if not BEHAVIOR_FILE.exists():
            self._save(self._default_state())

    def _default_state(self) -> Dict:
        return {
            "rules": [],
            "history": [],
            "stats": {
                "total_rules_created": 0,
                "total_rules_pruned": 0,
                "total_rules_applied": 0,
                "sessions_tracked": 0,
            },
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
        }

    def _load(self) -> Dict:
        try:
            with open(BEHAVIOR_FILE, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return self._default_state()

    def _save(self, data: Dict):
        data["last_updated"] = datetime.now().isoformat()
        if len(data.get("rules", [])) > MAX_RULES:
            # Keep active rules, prune oldest dormant ones
            active = [r for r in data["rules"] if r.get("active", True)]
            dormant = [r for r in data["rules"] if not r.get("active", True)]
            dormant.sort(key=lambda r: r.get("last_used", ""), reverse=True)
            data["rules"] = active + dormant[:MAX_RULES - len(active)]
        if len(data.get("history", [])) > MAX_HISTORY:
            data["history"] = data["history"][-MAX_HISTORY:]
        with open(BEHAVIOR_FILE, "w") as f:
            json.dump(data, f, indent=2, default=str)

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="learned_behavior",
            name="Learned Behavior",
            version="1.0.0",
            category="meta",
            description="Persistent cross-session behavioral rules for cumulative learning",
            actions=[
                SkillAction(
                    name="add_rule",
                    description="Add a new behavioral rule to the persistent store",
                    parameters={
                        "rule_type": {
                            "type": "string",
                            "required": True,
                            "description": "Type: prompt_addition, skill_preference, avoidance_rule, strategy, heuristic",
                        },
                        "content": {
                            "type": "string",
                            "required": True,
                            "description": "The rule content (prompt text, skill ID, or strategy description)",
                        },
                        "source": {
                            "type": "string",
                            "required": False,
                            "description": "Where this rule came from (feedback_loop, manual, experiment, observation)",
                        },
                        "context": {
                            "type": "string",
                            "required": False,
                            "description": "When this rule applies (e.g., 'when balance < $1', 'for code_review tasks')",
                        },
                        "confidence": {
                            "type": "number",
                            "required": False,
                            "description": "Confidence level 0.0-1.0 (default: 0.5)",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="load_rules",
                    description="Load and apply all active behavioral rules for session start",
                    parameters={
                        "rule_types": {
                            "type": "string",
                            "required": False,
                            "description": "Comma-separated types to load (default: all)",
                        },
                        "min_confidence": {
                            "type": "number",
                            "required": False,
                            "description": "Minimum confidence threshold (default: 0.3)",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="record_outcome",
                    description="Record whether a rule helped or hurt in the current session",
                    parameters={
                        "rule_id": {
                            "type": "string",
                            "required": True,
                            "description": "ID of the rule to record outcome for",
                        },
                        "success": {
                            "type": "boolean",
                            "required": True,
                            "description": "Whether the rule contributed to a positive outcome",
                        },
                        "detail": {
                            "type": "string",
                            "required": False,
                            "description": "Optional detail about the outcome",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="prune",
                    description="Remove rules that have consistently underperformed",
                    parameters={
                        "min_sessions": {
                            "type": "number",
                            "required": False,
                            "description": "Min sessions before pruning eligible (default: 3)",
                        },
                        "max_failure_rate": {
                            "type": "number",
                            "required": False,
                            "description": "Failure rate threshold for pruning (default: 0.7)",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="list_rules",
                    description="List all stored behavioral rules with their stats",
                    parameters={
                        "rule_type": {
                            "type": "string",
                            "required": False,
                            "description": "Filter by type",
                        },
                        "active_only": {
                            "type": "boolean",
                            "required": False,
                            "description": "Only show active rules (default: True)",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="import_from_feedback",
                    description="Import successful adaptations from FeedbackLoop as persistent rules",
                    parameters={},
                    estimated_cost=0,
                ),
                SkillAction(
                    name="deactivate",
                    description="Deactivate a rule without deleting it",
                    parameters={
                        "rule_id": {
                            "type": "string",
                            "required": True,
                            "description": "ID of the rule to deactivate",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="get_context_prompt",
                    description="Get a compiled prompt addition from all active rules for injection into LLM context",
                    parameters={},
                    estimated_cost=0,
                ),
            ],
            required_credentials=[],
        )

    def check_credentials(self) -> bool:
        return True

    async def execute(self, action: str, params: Dict) -> SkillResult:
        handlers = {
            "add_rule": self._add_rule,
            "load_rules": self._load_rules,
            "record_outcome": self._record_outcome,
            "prune": self._prune,
            "list_rules": self._list_rules,
            "import_from_feedback": self._import_from_feedback,
            "deactivate": self._deactivate,
            "get_context_prompt": self._get_context_prompt,
        }
        handler = handlers.get(action)
        if not handler:
            return SkillResult(
                success=False,
                message=f"Unknown action: {action}. Available: {list(handlers.keys())}",
            )
        return await handler(params)

    def _generate_rule_id(self, content: str, rule_type: str) -> str:
        """Generate a deterministic rule ID from content hash."""
        h = hashlib.sha256(f"{rule_type}:{content}".encode()).hexdigest()[:12]
        return f"rule_{h}"

    async def _add_rule(self, params: Dict) -> SkillResult:
        """Add a new behavioral rule."""
        rule_type = params.get("rule_type", "")
        content = params.get("content", "")
        source = params.get("source", "manual")
        context = params.get("context", "")
        confidence = min(1.0, max(0.0, float(params.get("confidence", 0.5))))

        valid_types = ["prompt_addition", "skill_preference", "avoidance_rule", "strategy", "heuristic"]
        if rule_type not in valid_types:
            return SkillResult(
                success=False,
                message=f"Invalid rule_type: {rule_type}. Must be one of: {valid_types}",
            )

        if not content:
            return SkillResult(success=False, message="Content is required")

        rule_id = self._generate_rule_id(content, rule_type)

        data = self._load()

        # Check for duplicate
        for existing in data["rules"]:
            if existing["rule_id"] == rule_id:
                return SkillResult(
                    success=False,
                    message=f"Duplicate rule: {rule_id} already exists",
                    data={"rule_id": rule_id, "existing": existing},
                )

        rule = {
            "rule_id": rule_id,
            "rule_type": rule_type,
            "content": content,
            "source": source,
            "context": context,
            "confidence": confidence,
            "active": True,
            "created_at": datetime.now().isoformat(),
            "last_used": None,
            "sessions_active": 0,
            "success_count": 0,
            "failure_count": 0,
            "effectiveness_score": 0.5,  # neutral starting point
        }

        data["rules"].append(rule)
        data["stats"]["total_rules_created"] += 1
        data["history"].append({
            "event": "rule_added",
            "rule_id": rule_id,
            "rule_type": rule_type,
            "source": source,
            "timestamp": datetime.now().isoformat(),
        })

        self._save(data)

        return SkillResult(
            success=True,
            message=f"Added {rule_type} rule: {rule_id}",
            data={"rule_id": rule_id, "rule": rule},
        )

    async def _load_rules(self, params: Dict) -> SkillResult:
        """Load and apply all active rules for session start."""
        rule_types_str = params.get("rule_types", "")
        min_confidence = float(params.get("min_confidence", 0.3))

        data = self._load()
        rules = data["rules"]

        # Filter by type if specified
        if rule_types_str:
            allowed_types = [t.strip() for t in rule_types_str.split(",")]
            rules = [r for r in rules if r["rule_type"] in allowed_types]

        # Filter active and confident
        rules = [r for r in rules if r.get("active", True) and r.get("confidence", 0.5) >= min_confidence]

        # Sort by effectiveness score (best rules first)
        rules.sort(key=lambda r: r.get("effectiveness_score", 0.5), reverse=True)

        applied_count = 0
        prompt_additions = []
        applied_rules = []

        for rule in rules:
            rule_id = rule["rule_id"]

            # Update session tracking
            for stored_rule in data["rules"]:
                if stored_rule["rule_id"] == rule_id:
                    stored_rule["sessions_active"] = stored_rule.get("sessions_active", 0) + 1
                    stored_rule["last_used"] = datetime.now().isoformat()
                    break

            applied_count += 1
            applied_rules.append({
                "rule_id": rule_id,
                "type": rule["rule_type"],
                "content": rule["content"][:100],
                "confidence": rule.get("confidence", 0.5),
                "effectiveness": rule.get("effectiveness_score", 0.5),
            })

            # Apply prompt additions to cognition engine
            if rule["rule_type"] == "prompt_addition" and self._append_prompt_fn:
                prompt_additions.append(rule["content"])

        # Apply all prompt additions at once
        if prompt_additions and self._append_prompt_fn:
            combined = "\n\n[LEARNED BEHAVIORS]\n" + "\n".join(
                f"- {p}" for p in prompt_additions
            )
            self._append_prompt_fn(combined)

        data["stats"]["total_rules_applied"] += applied_count
        data["stats"]["sessions_tracked"] += 1
        data["history"].append({
            "event": "session_load",
            "rules_loaded": applied_count,
            "prompt_additions": len(prompt_additions),
            "timestamp": datetime.now().isoformat(),
        })

        self._save(data)

        return SkillResult(
            success=True,
            message=f"Loaded {applied_count} rules ({len(prompt_additions)} prompt additions)",
            data={
                "rules_loaded": applied_count,
                "prompt_additions_applied": len(prompt_additions),
                "rules": applied_rules,
            },
        )

    async def _record_outcome(self, params: Dict) -> SkillResult:
        """Record whether a rule helped in the current session."""
        rule_id = params.get("rule_id", "")
        success = params.get("success", False)
        detail = params.get("detail", "")

        if not rule_id:
            return SkillResult(success=False, message="rule_id is required")

        data = self._load()

        target_rule = None
        for rule in data["rules"]:
            if rule["rule_id"] == rule_id:
                target_rule = rule
                break

        if not target_rule:
            return SkillResult(success=False, message=f"Rule {rule_id} not found")

        if success:
            target_rule["success_count"] = target_rule.get("success_count", 0) + 1
        else:
            target_rule["failure_count"] = target_rule.get("failure_count", 0) + 1

        # Recalculate effectiveness score using Bayesian-like update
        total = target_rule.get("success_count", 0) + target_rule.get("failure_count", 0)
        if total > 0:
            # Use Laplace smoothing: (successes + 1) / (total + 2)
            smoothed = (target_rule.get("success_count", 0) + 1) / (total + 2)
            target_rule["effectiveness_score"] = round(smoothed, 3)

        # Update confidence based on sample size
        target_rule["confidence"] = min(
            0.95,
            0.3 + (0.65 * min(total / 20.0, 1.0))  # confidence grows with samples
        )

        data["history"].append({
            "event": "outcome_recorded",
            "rule_id": rule_id,
            "success": success,
            "detail": detail[:200] if detail else "",
            "new_effectiveness": target_rule["effectiveness_score"],
            "timestamp": datetime.now().isoformat(),
        })

        self._save(data)

        return SkillResult(
            success=True,
            message=f"Recorded {'success' if success else 'failure'} for {rule_id} "
                    f"(effectiveness: {target_rule['effectiveness_score']:.3f})",
            data={
                "rule_id": rule_id,
                "effectiveness_score": target_rule["effectiveness_score"],
                "confidence": target_rule["confidence"],
                "success_count": target_rule["success_count"],
                "failure_count": target_rule["failure_count"],
            },
        )

    async def _prune(self, params: Dict) -> SkillResult:
        """Remove rules that consistently underperform."""
        min_sessions = int(params.get("min_sessions", 3))
        max_failure_rate = float(params.get("max_failure_rate", 0.7))

        data = self._load()
        pruned = []
        kept = []

        for rule in data["rules"]:
            sessions = rule.get("sessions_active", 0)
            total = rule.get("success_count", 0) + rule.get("failure_count", 0)

            should_prune = False
            reason = ""

            if sessions >= min_sessions and total > 0:
                failure_rate = rule.get("failure_count", 0) / total
                if failure_rate >= max_failure_rate:
                    should_prune = True
                    reason = f"failure_rate={failure_rate:.2f} >= {max_failure_rate}"

            if sessions >= min_sessions * 2 and rule.get("effectiveness_score", 0.5) < 0.3:
                should_prune = True
                reason = f"effectiveness={rule.get('effectiveness_score', 0.5):.3f} < 0.3"

            if should_prune:
                pruned.append({
                    "rule_id": rule["rule_id"],
                    "type": rule["rule_type"],
                    "content": rule["content"][:80],
                    "reason": reason,
                    "sessions": sessions,
                    "effectiveness": rule.get("effectiveness_score", 0.5),
                })
                data["stats"]["total_rules_pruned"] += 1
            else:
                kept.append(rule)

        data["rules"] = kept

        if pruned:
            data["history"].append({
                "event": "prune",
                "rules_pruned": len(pruned),
                "rules_kept": len(kept),
                "pruned_ids": [p["rule_id"] for p in pruned],
                "timestamp": datetime.now().isoformat(),
            })

        self._save(data)

        return SkillResult(
            success=True,
            message=f"Pruned {len(pruned)} underperforming rules, kept {len(kept)}",
            data={
                "pruned": pruned,
                "total_remaining": len(kept),
            },
        )

    async def _list_rules(self, params: Dict) -> SkillResult:
        """List all stored rules with stats."""
        rule_type_filter = params.get("rule_type", "")
        active_only = params.get("active_only", True)

        data = self._load()
        rules = data["rules"]

        if rule_type_filter:
            rules = [r for r in rules if r["rule_type"] == rule_type_filter]
        if active_only:
            rules = [r for r in rules if r.get("active", True)]

        # Sort by effectiveness
        rules.sort(key=lambda r: r.get("effectiveness_score", 0.5), reverse=True)

        summary = []
        for r in rules:
            total = r.get("success_count", 0) + r.get("failure_count", 0)
            summary.append({
                "rule_id": r["rule_id"],
                "type": r["rule_type"],
                "content": r["content"][:100],
                "active": r.get("active", True),
                "confidence": r.get("confidence", 0.5),
                "effectiveness": r.get("effectiveness_score", 0.5),
                "sessions": r.get("sessions_active", 0),
                "success_count": r.get("success_count", 0),
                "failure_count": r.get("failure_count", 0),
                "total_outcomes": total,
                "source": r.get("source", "unknown"),
                "created_at": r.get("created_at", ""),
            })

        type_counts = {}
        for r in rules:
            t = r["rule_type"]
            type_counts[t] = type_counts.get(t, 0) + 1

        return SkillResult(
            success=True,
            message=f"{len(rules)} rules found",
            data={
                "rules": summary,
                "count": len(rules),
                "by_type": type_counts,
                "stats": data.get("stats", {}),
            },
        )

    async def _import_from_feedback(self, params: Dict) -> SkillResult:
        """Import successful adaptations from FeedbackLoop as persistent rules."""
        feedback_file = Path(__file__).parent.parent / "data" / "feedback_loop.json"

        try:
            with open(feedback_file, "r") as f:
                feedback_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return SkillResult(
                success=False,
                message="No feedback loop data found",
            )

        adaptations = feedback_data.get("adaptations", [])
        if not adaptations:
            return SkillResult(
                success=False,
                message="No adaptations found in feedback loop data",
            )

        imported = 0
        skipped = 0
        data = self._load()
        existing_ids = {r["rule_id"] for r in data["rules"]}

        for adaptation in adaptations:
            # Only import applied adaptations
            if not adaptation.get("applied", False):
                skipped += 1
                continue

            content = adaptation.get("prompt_addition", "") or adaptation.get("description", "")
            if not content:
                skipped += 1
                continue

            # Map adaptation type to rule type
            adapt_type = adaptation.get("type", "general")
            rule_type = "prompt_addition"
            if adapt_type in ("avoid_skill", "avoid_action"):
                rule_type = "avoidance_rule"
            elif adapt_type in ("prefer_skill", "prefer_action"):
                rule_type = "skill_preference"
            elif adapt_type in ("strategy",):
                rule_type = "strategy"

            rule_id = self._generate_rule_id(content, rule_type)

            if rule_id in existing_ids:
                skipped += 1
                continue

            # Check if adaptation was evaluated as positive
            evaluated = adaptation.get("evaluated", False)
            improved = adaptation.get("improved", None)
            confidence = 0.5
            if evaluated and improved:
                confidence = 0.7
            elif evaluated and not improved:
                confidence = 0.3

            rule = {
                "rule_id": rule_id,
                "rule_type": rule_type,
                "content": content,
                "source": "feedback_loop",
                "context": adaptation.get("context", ""),
                "confidence": confidence,
                "active": True,
                "created_at": datetime.now().isoformat(),
                "imported_from": adaptation.get("id", ""),
                "last_used": None,
                "sessions_active": 0,
                "success_count": 1 if improved else 0,
                "failure_count": 0 if improved is not False else 1,
                "effectiveness_score": confidence,
            }

            data["rules"].append(rule)
            existing_ids.add(rule_id)
            imported += 1
            data["stats"]["total_rules_created"] += 1

        if imported > 0:
            data["history"].append({
                "event": "import_from_feedback",
                "imported": imported,
                "skipped": skipped,
                "timestamp": datetime.now().isoformat(),
            })

        self._save(data)

        return SkillResult(
            success=True,
            message=f"Imported {imported} rules from FeedbackLoop ({skipped} skipped)",
            data={
                "imported": imported,
                "skipped": skipped,
                "total_rules": len(data["rules"]),
            },
        )

    async def _deactivate(self, params: Dict) -> SkillResult:
        """Deactivate a rule without deleting it."""
        rule_id = params.get("rule_id", "")
        if not rule_id:
            return SkillResult(success=False, message="rule_id is required")

        data = self._load()

        for rule in data["rules"]:
            if rule["rule_id"] == rule_id:
                rule["active"] = False
                data["history"].append({
                    "event": "rule_deactivated",
                    "rule_id": rule_id,
                    "timestamp": datetime.now().isoformat(),
                })
                self._save(data)
                return SkillResult(
                    success=True,
                    message=f"Deactivated rule {rule_id}",
                    data={"rule_id": rule_id, "active": False},
                )

        return SkillResult(success=False, message=f"Rule {rule_id} not found")

    async def _get_context_prompt(self, params: Dict) -> SkillResult:
        """Compile all active rules into a context prompt for LLM injection."""
        data = self._load()
        active_rules = [r for r in data["rules"] if r.get("active", True)]

        if not active_rules:
            return SkillResult(
                success=True,
                message="No active rules",
                data={"prompt": "", "rule_count": 0},
            )

        # Group by type
        by_type = {}
        for r in active_rules:
            t = r["rule_type"]
            if t not in by_type:
                by_type[t] = []
            by_type[t].append(r)

        sections = []

        if "prompt_addition" in by_type:
            items = sorted(by_type["prompt_addition"],
                         key=lambda r: r.get("effectiveness_score", 0.5), reverse=True)
            lines = [f"  - {r['content']}" for r in items[:10]]
            sections.append("LEARNED INSIGHTS:\n" + "\n".join(lines))

        if "skill_preference" in by_type:
            items = sorted(by_type["skill_preference"],
                         key=lambda r: r.get("effectiveness_score", 0.5), reverse=True)
            lines = [f"  - PREFER: {r['content']}" for r in items[:5]]
            sections.append("SKILL PREFERENCES:\n" + "\n".join(lines))

        if "avoidance_rule" in by_type:
            items = sorted(by_type["avoidance_rule"],
                         key=lambda r: r.get("effectiveness_score", 0.5), reverse=True)
            lines = [f"  - AVOID: {r['content']}" for r in items[:5]]
            sections.append("AVOIDANCE RULES:\n" + "\n".join(lines))

        if "strategy" in by_type:
            items = sorted(by_type["strategy"],
                         key=lambda r: r.get("effectiveness_score", 0.5), reverse=True)
            lines = [f"  - {r['content']}" for r in items[:5]]
            sections.append("STRATEGIES:\n" + "\n".join(lines))

        if "heuristic" in by_type:
            items = sorted(by_type["heuristic"],
                         key=lambda r: r.get("effectiveness_score", 0.5), reverse=True)
            lines = [f"  - {r['content']}" for r in items[:5]]
            sections.append("HEURISTICS:\n" + "\n".join(lines))

        prompt = "\n\n".join(sections)

        return SkillResult(
            success=True,
            message=f"Compiled context from {len(active_rules)} active rules",
            data={
                "prompt": prompt,
                "rule_count": len(active_rules),
                "by_type": {t: len(rs) for t, rs in by_type.items()},
            },
        )
