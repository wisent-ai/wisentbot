#!/usr/bin/env python3
"""
DecisionReplaySkill - Counterfactual analysis of past decisions using current knowledge.

This skill enables the agent to "backtest" its decision-making: replay past decisions
using current distilled rules and see how behavior would have differed. Like backtesting
a trading strategy, this reveals whether the agent's learning has actually improved its
decision quality.

Why this matters:
- Without replay, the agent can't measure if its learning is ACTUALLY helping
- Self-improvement requires not just "learning" but verifying learning works
- Identifies which rules had the most impact on decision quality
- Finds past mistakes the agent would now avoid (and past lucky guesses it wouldn't repeat)

The feedback loop:
  1. Agent makes decisions -> logged in DecisionLogSkill
  2. Agent learns rules -> stored in LearningDistillationSkill
  3. Agent replays past decisions with current rules -> THIS SKILL
  4. Agent identifies improvements/regressions -> feeds back into learning

Actions:
  - replay: Re-evaluate a specific decision with current rules
  - batch_replay: Replay multiple past decisions, compare old vs new
  - impact_report: Aggregate analysis of how rules changed decision quality
  - find_reversals: Find decisions where current rules would choose differently
  - timeline: Track decision quality improvement over time
  - what_if: Hypothetical replay with custom rules (not just current ones)

Serves:
- Self-Improvement: Verify that learning actually improves decisions
- Goal Setting: Identify which learning areas have the most impact

Works with:
- DecisionLogSkill: Source of historical decisions with outcomes
- LearningDistillationSkill: Source of current learned rules
- AutonomousLoopSkill: Can trigger replays in LEARN phase
"""

import json
import time
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from .base import Skill, SkillResult, SkillManifest, SkillAction


REPLAY_DATA_FILE = Path(__file__).parent.parent / "data" / "decision_replay.json"
MAX_REPLAYS = 1000
MAX_REPORTS = 100


class DecisionReplaySkill(Skill):
    """
    Counterfactual decision analysis using current learned rules.

    Replays past decisions to measure whether the agent's accumulated
    learning would lead to better choices today.
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._ensure_data()

    def _ensure_data(self):
        REPLAY_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        if not REPLAY_DATA_FILE.exists():
            self._save(self._default_state())

    def _default_state(self) -> Dict:
        return {
            "replays": [],
            "reports": [],
            "stats": {
                "total_replays": 0,
                "total_reversals": 0,
                "total_improvements": 0,
                "total_regressions": 0,
            },
            "created_at": datetime.now().isoformat(),
        }

    def _load(self) -> Dict:
        try:
            with open(REPLAY_DATA_FILE, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return self._default_state()

    def _save(self, data: Dict):
        with open(REPLAY_DATA_FILE, "w") as f:
            json.dump(data, f, indent=2, default=str)

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="decision_replay",
            name="Decision Replay",
            version="1.0.0",
            category="meta",
            description="Replay past decisions with current rules to measure learning impact",
            actions=[
                SkillAction(
                    name="replay",
                    description="Re-evaluate a specific past decision using current learned rules",
                    parameters={
                        "decision_id": {
                            "type": "string",
                            "required": True,
                            "description": "ID of the decision to replay (from decision_log)",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="batch_replay",
                    description="Replay multiple recent decisions and compare outcomes",
                    parameters={
                        "count": {
                            "type": "integer",
                            "required": False,
                            "description": "Number of recent decisions to replay (default: 20)",
                        },
                        "category": {
                            "type": "string",
                            "required": False,
                            "description": "Filter by decision category",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="impact_report",
                    description="Generate aggregate analysis of how current rules would change past decisions",
                    parameters={
                        "window_days": {
                            "type": "integer",
                            "required": False,
                            "description": "Analyze decisions from last N days (default: 30)",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="find_reversals",
                    description="Find past decisions where current rules would choose a different action",
                    parameters={
                        "min_confidence": {
                            "type": "number",
                            "required": False,
                            "description": "Minimum rule confidence to consider (default: 0.5)",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="timeline",
                    description="Show decision quality trend over time based on replay analysis",
                    parameters={},
                    estimated_cost=0,
                ),
                SkillAction(
                    name="what_if",
                    description="Replay a decision with custom hypothetical rules",
                    parameters={
                        "decision_id": {
                            "type": "string",
                            "required": True,
                            "description": "ID of the decision to replay",
                        },
                        "custom_rules": {
                            "type": "array",
                            "required": True,
                            "description": "List of rule dicts with rule_text, category, confidence, skill_id",
                        },
                    },
                    estimated_cost=0,
                ),
            ],
            required_credentials=[],
        )

    async def execute(self, action: str, params: Dict) -> SkillResult:
        actions = {
            "replay": self._replay,
            "batch_replay": self._batch_replay,
            "impact_report": self._impact_report,
            "find_reversals": self._find_reversals,
            "timeline": self._timeline,
            "what_if": self._what_if,
        }
        handler = actions.get(action)
        if not handler:
            return SkillResult(
                success=False,
                message=f"Unknown action '{action}'. Available: {list(actions.keys())}",
            )
        return await handler(params)

    # -- Helpers --

    def _get_decisions(self) -> List[Dict]:
        """Load decisions from DecisionLogSkill's data file."""
        decision_file = Path(__file__).parent.parent / "data" / "decision_log.json"
        if not decision_file.exists():
            return []
        try:
            with open(decision_file, "r") as f:
                data = json.load(f)
            return data.get("decisions", [])
        except (json.JSONDecodeError, FileNotFoundError):
            return []

    def _get_rules(self) -> List[Dict]:
        """Load current rules from LearningDistillationSkill's data file."""
        rules_file = Path(__file__).parent.parent / "data" / "learning_distillation.json"
        if not rules_file.exists():
            return []
        try:
            with open(rules_file, "r") as f:
                data = json.load(f)
            return data.get("rules", [])
        except (json.JSONDecodeError, FileNotFoundError):
            return []

    def _find_decision(self, decision_id: str) -> Optional[Dict]:
        """Find a specific decision by ID."""
        for d in self._get_decisions():
            if d.get("id") == decision_id:
                return d
        return None

    def _evaluate_decision_with_rules(
        self, decision: Dict, rules: List[Dict]
    ) -> Dict:
        """
        Evaluate a past decision against a set of rules.

        Returns a replay result with:
        - applicable_rules: rules that apply to this decision's context
        - recommendation: what the rules suggest
        - would_change: whether the rules suggest a different choice
        - confidence: confidence in the replay recommendation
        - reasoning: explanation of why rules support or contradict the original choice
        """
        context = decision.get("context", "").lower()
        choice = decision.get("choice", "").lower()
        alternatives = [a.lower() for a in decision.get("alternatives", [])]
        category = decision.get("category", "")
        tags = [t.lower() for t in decision.get("tags", [])]

        # Find applicable rules
        applicable_rules = []
        for rule in rules:
            score = self._rule_relevance_score(rule, context, choice, alternatives, category, tags)
            if score > 0:
                applicable_rules.append({
                    "rule_id": rule.get("id", ""),
                    "rule_text": rule.get("rule_text", ""),
                    "category": rule.get("category", ""),
                    "confidence": rule.get("confidence", 0),
                    "relevance_score": score,
                    "skill_id": rule.get("skill_id", ""),
                })

        # Sort by relevance * confidence
        applicable_rules.sort(
            key=lambda r: r["relevance_score"] * r["confidence"], reverse=True
        )

        if not applicable_rules:
            return {
                "applicable_rules": [],
                "recommendation": choice,
                "would_change": False,
                "confidence": 0,
                "reasoning": "No applicable rules found for this decision context.",
                "support_score": 0,
                "contradict_score": 0,
            }

        # Calculate support vs contradiction scores
        support_score = 0.0
        contradict_score = 0.0
        support_reasons = []
        contradict_reasons = []

        for rule in applicable_rules:
            rule_text = rule["rule_text"].lower()
            weight = rule["relevance_score"] * rule["confidence"]

            if self._rule_supports_choice(rule_text, choice):
                support_score += weight
                support_reasons.append(rule["rule_text"])
            elif any(self._rule_supports_choice(rule_text, alt) for alt in alternatives):
                contradict_score += weight
                supported_alt = next(
                    (alt for alt in alternatives if self._rule_supports_choice(rule_text, alt)),
                    "alternative"
                )
                contradict_reasons.append(f"{rule['rule_text']} (suggests: {supported_alt})")
            elif self._rule_contradicts_choice(rule_text, choice):
                contradict_score += weight
                contradict_reasons.append(rule["rule_text"])

        # Determine recommendation
        would_change = contradict_score > support_score and contradict_score > 0.3
        total_weight = support_score + contradict_score
        replay_confidence = total_weight / max(len(applicable_rules), 1)
        replay_confidence = min(replay_confidence, 1.0)

        # Build reasoning
        reasoning_parts = []
        if support_reasons:
            reasoning_parts.append(
                f"Rules supporting original choice ({support_score:.2f} weight): "
                + "; ".join(support_reasons[:3])
            )
        if contradict_reasons:
            reasoning_parts.append(
                f"Rules suggesting change ({contradict_score:.2f} weight): "
                + "; ".join(contradict_reasons[:3])
            )
        if not reasoning_parts:
            reasoning_parts.append("Rules found but no clear directional signal.")

        recommendation = choice
        if would_change and alternatives:
            alt_scores = {}
            for alt in alternatives:
                alt_support = sum(
                    r["relevance_score"] * r["confidence"]
                    for r in applicable_rules
                    if self._rule_supports_choice(r["rule_text"].lower(), alt)
                )
                alt_scores[alt] = alt_support
            if alt_scores:
                best_alt = max(alt_scores, key=alt_scores.get)
                if alt_scores[best_alt] > 0:
                    recommendation = best_alt

        return {
            "applicable_rules": applicable_rules[:10],
            "recommendation": recommendation,
            "would_change": would_change,
            "confidence": round(replay_confidence, 3),
            "reasoning": " | ".join(reasoning_parts),
            "support_score": round(support_score, 3),
            "contradict_score": round(contradict_score, 3),
        }

    def _rule_relevance_score(
        self,
        rule: Dict,
        context: str,
        choice: str,
        alternatives: List[str],
        category: str,
        tags: List[str],
    ) -> float:
        """Score how relevant a rule is to a decision context."""
        score = 0.0
        rule_text = rule.get("rule_text", "").lower()
        rule_skill = rule.get("skill_id", "").lower()
        rule_category = rule.get("category", "").lower()

        # Keywords from rule text appearing in context
        rule_words = set(rule_text.split())
        context_words = set(context.split())
        stop_words = {"the", "a", "an", "is", "are", "was", "for", "to", "of", "in",
                       "and", "or", "with", "it", "this", "that", "be", "has", "have"}
        meaningful_overlap = (rule_words & context_words) - stop_words
        if meaningful_overlap:
            score += len(meaningful_overlap) * 0.2

        # Skill mentioned in context or choice
        if rule_skill and (rule_skill in context or rule_skill in choice):
            score += 0.5

        # Rule about the choice or alternatives
        if choice and choice in rule_text:
            score += 0.4
        for alt in alternatives:
            if alt and alt in rule_text:
                score += 0.3

        # Category alignment
        category_rule_map = {
            "success_pattern": ["skill_selection", "strategy"],
            "failure_pattern": ["error_handling", "skill_selection"],
            "cost_efficiency": ["resource_allocation", "prioritization"],
            "skill_preference": ["skill_selection"],
            "timing_pattern": ["prioritization"],
        }
        if rule_category in category_rule_map:
            if category in category_rule_map[rule_category]:
                score += 0.3

        # Tag overlap
        for tag in tags:
            if tag in rule_text:
                score += 0.15

        return round(score, 3)

    def _rule_supports_choice(self, rule_text: str, choice: str) -> bool:
        """Check if a rule text supports a particular choice."""
        if not choice:
            return False
        choice_lower = choice.lower()
        positive_patterns = [
            f"prefer {choice_lower}",
            f"use {choice_lower}",
            f"{choice_lower} is effective",
            f"{choice_lower} works well",
            f"{choice_lower} has high success",
            f"recommend {choice_lower}",
            f"{choice_lower} is fast",
            f"{choice_lower} is reliable",
        ]
        return any(p in rule_text for p in positive_patterns)

    def _rule_contradicts_choice(self, rule_text: str, choice: str) -> bool:
        """Check if a rule text contradicts a particular choice."""
        if not choice:
            return False
        choice_lower = choice.lower()
        negative_patterns = [
            f"avoid {choice_lower}",
            f"{choice_lower} fails",
            f"{choice_lower} is slow",
            f"{choice_lower} has low success",
            f"don't use {choice_lower}",
            f"{choice_lower} is unreliable",
            f"{choice_lower} is expensive",
        ]
        return any(p in rule_text for p in negative_patterns)

    # -- Actions --

    async def _replay(self, params: Dict) -> SkillResult:
        """Replay a single past decision with current rules."""
        decision_id = params.get("decision_id", "").strip()
        if not decision_id:
            return SkillResult(success=False, message="decision_id is required")

        decision = self._find_decision(decision_id)
        if not decision:
            return SkillResult(
                success=False,
                message=f"Decision '{decision_id}' not found in decision log.",
            )

        rules = self._get_rules()
        if not rules:
            return SkillResult(
                success=False,
                message="No learned rules available. Run learning_distillation.distill first.",
            )

        result = self._evaluate_decision_with_rules(decision, rules)

        # Save replay record
        data = self._load()
        replay_record = {
            "id": str(uuid.uuid4())[:8],
            "decision_id": decision_id,
            "original_choice": decision.get("choice", ""),
            "original_confidence": decision.get("confidence", 0),
            "original_outcome": decision.get("outcome"),
            "replay_recommendation": result["recommendation"],
            "would_change": result["would_change"],
            "replay_confidence": result["confidence"],
            "rules_applied": len(result["applicable_rules"]),
            "support_score": result["support_score"],
            "contradict_score": result["contradict_score"],
            "reasoning": result["reasoning"],
            "timestamp": datetime.now().isoformat(),
        }
        data["replays"].append(replay_record)
        data["stats"]["total_replays"] += 1
        if result["would_change"]:
            data["stats"]["total_reversals"] += 1

        if len(data["replays"]) > MAX_REPLAYS:
            data["replays"] = data["replays"][-MAX_REPLAYS:]

        self._save(data)

        return SkillResult(
            success=True,
            message=(
                f"Replayed decision '{decision_id}'. "
                f"{'Would change' if result['would_change'] else 'Would keep'} "
                f"original choice. {len(result['applicable_rules'])} rules applied."
            ),
            data={
                "replay": replay_record,
                "applicable_rules": result["applicable_rules"][:5],
                "decision_context": decision.get("context", ""),
            },
        )

    async def _batch_replay(self, params: Dict) -> SkillResult:
        """Replay multiple recent decisions and compare."""
        count = int(params.get("count", 20))
        count = min(count, 100)
        category_filter = params.get("category", "").strip()

        decisions = self._get_decisions()
        if not decisions:
            return SkillResult(success=False, message="No decisions found in decision log.")

        rules = self._get_rules()
        if not rules:
            return SkillResult(
                success=False,
                message="No learned rules available. Run learning_distillation.distill first.",
            )

        if category_filter:
            decisions = [d for d in decisions if d.get("category") == category_filter]

        recent = decisions[-count:]
        results = []
        reversals = 0
        improvements = 0
        regressions = 0
        data = self._load()

        for decision in recent:
            evaluation = self._evaluate_decision_with_rules(decision, rules)
            outcome = decision.get("outcome")
            original_success = outcome.get("success", None) if isinstance(outcome, dict) else None

            entry = {
                "decision_id": decision.get("id", ""),
                "category": decision.get("category", ""),
                "original_choice": decision.get("choice", ""),
                "replay_recommendation": evaluation["recommendation"],
                "would_change": evaluation["would_change"],
                "rules_applied": len(evaluation["applicable_rules"]),
                "support_score": evaluation["support_score"],
                "contradict_score": evaluation["contradict_score"],
                "original_outcome_known": original_success is not None,
            }

            if evaluation["would_change"]:
                reversals += 1
                if original_success is False:
                    improvements += 1
                    entry["improvement"] = True
                elif original_success is True:
                    regressions += 1
                    entry["regression"] = True

            results.append(entry)

            replay_record = {
                "id": str(uuid.uuid4())[:8],
                "decision_id": decision.get("id", ""),
                "original_choice": decision.get("choice", ""),
                "replay_recommendation": evaluation["recommendation"],
                "would_change": evaluation["would_change"],
                "replay_confidence": evaluation["confidence"],
                "rules_applied": len(evaluation["applicable_rules"]),
                "timestamp": datetime.now().isoformat(),
                "batch": True,
            }
            data["replays"].append(replay_record)

        data["stats"]["total_replays"] += len(recent)
        data["stats"]["total_reversals"] += reversals
        data["stats"]["total_improvements"] += improvements
        data["stats"]["total_regressions"] += regressions

        if len(data["replays"]) > MAX_REPLAYS:
            data["replays"] = data["replays"][-MAX_REPLAYS:]

        self._save(data)

        reversal_rate = reversals / max(len(recent), 1)
        improvement_rate = improvements / max(reversals, 1) if reversals else 0

        return SkillResult(
            success=True,
            message=(
                f"Replayed {len(recent)} decisions. "
                f"{reversals} reversals ({reversal_rate:.0%}), "
                f"{improvements} improvements, {regressions} regressions. "
                f"Improvement rate: {improvement_rate:.0%}."
            ),
            data={
                "total_replayed": len(recent),
                "reversals": reversals,
                "improvements": improvements,
                "regressions": regressions,
                "reversal_rate": round(reversal_rate, 3),
                "improvement_rate": round(improvement_rate, 3),
                "results": results[:20],
                "rules_used": len(rules),
            },
        )

    async def _impact_report(self, params: Dict) -> SkillResult:
        """Generate aggregate impact report of learned rules on decisions."""
        window_days = int(params.get("window_days", 30))

        decisions = self._get_decisions()
        rules = self._get_rules()

        if not decisions:
            return SkillResult(success=False, message="No decisions found.")

        cutoff = time.time() - (window_days * 86400)
        windowed = [d for d in decisions if d.get("timestamp", 0) > cutoff]
        if not windowed:
            windowed = decisions[-50:]

        category_stats = defaultdict(lambda: {
            "total": 0, "reversals": 0, "improvements": 0, "regressions": 0,
        })
        rule_usage = defaultdict(int)
        total_support = 0.0
        total_contradict = 0.0

        for decision in windowed:
            evaluation = self._evaluate_decision_with_rules(decision, rules)
            cat = decision.get("category", "other")
            cs = category_stats[cat]
            cs["total"] += 1
            total_support += evaluation["support_score"]
            total_contradict += evaluation["contradict_score"]

            if evaluation["would_change"]:
                cs["reversals"] += 1
                outcome = decision.get("outcome")
                if isinstance(outcome, dict):
                    if outcome.get("success") is False:
                        cs["improvements"] += 1
                    elif outcome.get("success") is True:
                        cs["regressions"] += 1

            for rule in evaluation["applicable_rules"]:
                rule_usage[rule["rule_id"]] += 1

        n = len(windowed)
        total_reversals = sum(cs["reversals"] for cs in category_stats.values())
        total_improvements = sum(cs["improvements"] for cs in category_stats.values())
        total_regressions = sum(cs["regressions"] for cs in category_stats.values())

        top_rules = sorted(rule_usage.items(), key=lambda x: x[1], reverse=True)[:10]
        rule_map = {r.get("id", ""): r for r in rules}
        top_rule_details = []
        for rule_id, count in top_rules:
            rule = rule_map.get(rule_id, {})
            top_rule_details.append({
                "rule_id": rule_id,
                "rule_text": rule.get("rule_text", "unknown"),
                "confidence": rule.get("confidence", 0),
                "decisions_affected": count,
            })

        quality_decisions = total_improvements + total_regressions
        learning_quality = (
            total_improvements / quality_decisions if quality_decisions > 0 else 0.5
        )

        report = {
            "window_days": window_days,
            "decisions_analyzed": n,
            "total_rules": len(rules),
            "total_reversals": total_reversals,
            "reversal_rate": round(total_reversals / max(n, 1), 3),
            "improvements": total_improvements,
            "regressions": total_regressions,
            "learning_quality_score": round(learning_quality, 3),
            "avg_support_per_decision": round(total_support / max(n, 1), 3),
            "avg_contradict_per_decision": round(total_contradict / max(n, 1), 3),
            "category_breakdown": dict(category_stats),
            "top_impactful_rules": top_rule_details,
        }

        data = self._load()
        report_record = {
            "id": str(uuid.uuid4())[:8],
            "timestamp": datetime.now().isoformat(),
            **report,
        }
        data["reports"].append(report_record)
        if len(data["reports"]) > MAX_REPORTS:
            data["reports"] = data["reports"][-MAX_REPORTS:]
        self._save(data)

        quality_label = (
            "excellent" if learning_quality > 0.8
            else "good" if learning_quality > 0.6
            else "moderate" if learning_quality > 0.4
            else "needs improvement"
        )

        return SkillResult(
            success=True,
            message=(
                f"Impact report: {n} decisions analyzed, {total_reversals} reversals "
                f"({total_reversals/max(n,1):.0%}), {total_improvements} improvements, "
                f"{total_regressions} regressions. "
                f"Learning quality: {quality_label} ({learning_quality:.0%})."
            ),
            data=report,
        )

    async def _find_reversals(self, params: Dict) -> SkillResult:
        """Find past decisions where current rules would choose differently."""
        min_confidence = float(params.get("min_confidence", 0.5))

        decisions = self._get_decisions()
        rules = [r for r in self._get_rules() if r.get("confidence", 0) >= min_confidence]

        if not decisions:
            return SkillResult(success=False, message="No decisions found.")
        if not rules:
            return SkillResult(
                success=False,
                message=f"No rules with confidence >= {min_confidence}.",
            )

        reversals = []
        for decision in decisions:
            evaluation = self._evaluate_decision_with_rules(decision, rules)
            if evaluation["would_change"]:
                outcome = decision.get("outcome")
                outcome_success = None
                if isinstance(outcome, dict):
                    outcome_success = outcome.get("success")

                reversals.append({
                    "decision_id": decision.get("id", ""),
                    "category": decision.get("category", ""),
                    "context": decision.get("context", "")[:200],
                    "original_choice": decision.get("choice", ""),
                    "recommended_choice": evaluation["recommendation"],
                    "original_outcome_success": outcome_success,
                    "contradict_score": evaluation["contradict_score"],
                    "support_score": evaluation["support_score"],
                    "reasoning": evaluation["reasoning"],
                    "top_rules": [
                        r["rule_text"] for r in evaluation["applicable_rules"][:3]
                    ],
                    "created_at": decision.get("created_at", ""),
                })

        reversals.sort(key=lambda r: r["contradict_score"], reverse=True)

        improvements = [r for r in reversals if r["original_outcome_success"] is False]
        would_regress = [r for r in reversals if r["original_outcome_success"] is True]
        unknown = [r for r in reversals if r["original_outcome_success"] is None]

        return SkillResult(
            success=True,
            message=(
                f"Found {len(reversals)} reversals. "
                f"{len(improvements)} would fix past failures, "
                f"{len(would_regress)} would undo past successes, "
                f"{len(unknown)} have unknown outcomes."
            ),
            data={
                "total_reversals": len(reversals),
                "improvements": len(improvements),
                "would_regress": len(would_regress),
                "unknown_outcome": len(unknown),
                "reversals": reversals[:20],
            },
        )

    async def _timeline(self, params: Dict) -> SkillResult:
        """Show decision quality trend over time."""
        data = self._load()
        replays = data.get("replays", [])

        if not replays:
            return SkillResult(
                success=False,
                message="No replay data yet. Run batch_replay or replay first.",
            )

        daily_stats = defaultdict(lambda: {"total": 0, "reversals": 0, "date": ""})

        for replay in replays:
            ts = replay.get("timestamp", "")
            date_key = ts[:10] if ts else "unknown"
            ds = daily_stats[date_key]
            ds["date"] = date_key
            ds["total"] += 1
            if replay.get("would_change"):
                ds["reversals"] += 1

        timeline = sorted(daily_stats.values(), key=lambda x: x["date"])

        for entry in timeline:
            entry["reversal_rate"] = round(
                entry["reversals"] / max(entry["total"], 1), 3
            )

        if len(timeline) >= 2:
            mid = len(timeline) // 2
            first_half = timeline[:mid]
            second_half = timeline[mid:]

            first_total = sum(e["total"] for e in first_half)
            first_reversals = sum(e["reversals"] for e in first_half)
            second_total = sum(e["total"] for e in second_half)
            second_reversals = sum(e["reversals"] for e in second_half)

            first_rate = first_reversals / max(first_total, 1)
            second_rate = second_reversals / max(second_total, 1)

            if second_rate < first_rate * 0.8:
                trend = "improving"
            elif second_rate > first_rate * 1.2:
                trend = "declining"
            else:
                trend = "stable"
        else:
            first_rate = 0
            second_rate = 0
            trend = "insufficient_data"

        return SkillResult(
            success=True,
            message=(
                f"Timeline: {len(timeline)} periods, "
                f"trend: {trend}. "
                f"Early reversal rate: {first_rate:.0%}, "
                f"recent: {second_rate:.0%}."
            ),
            data={
                "timeline": timeline,
                "trend": trend,
                "early_reversal_rate": round(first_rate, 3),
                "recent_reversal_rate": round(second_rate, 3),
                "total_replays": sum(e["total"] for e in timeline),
                "stats": data.get("stats", {}),
            },
        )

    async def _what_if(self, params: Dict) -> SkillResult:
        """Replay a decision with custom hypothetical rules."""
        decision_id = params.get("decision_id", "").strip()
        custom_rules = params.get("custom_rules", [])

        if not decision_id:
            return SkillResult(success=False, message="decision_id is required")
        if not custom_rules:
            return SkillResult(success=False, message="custom_rules list is required")

        decision = self._find_decision(decision_id)
        if not decision:
            return SkillResult(
                success=False,
                message=f"Decision '{decision_id}' not found.",
            )

        normalized_rules = []
        for rule in custom_rules:
            if isinstance(rule, str):
                rule = {"rule_text": rule}
            normalized_rules.append({
                "id": rule.get("id", str(uuid.uuid4())[:8]),
                "rule_text": rule.get("rule_text", ""),
                "category": rule.get("category", "general"),
                "confidence": float(rule.get("confidence", 0.7)),
                "skill_id": rule.get("skill_id", ""),
            })

        custom_result = self._evaluate_decision_with_rules(decision, normalized_rules)
        actual_rules = self._get_rules()
        actual_result = self._evaluate_decision_with_rules(decision, actual_rules)

        return SkillResult(
            success=True,
            message=(
                f"What-if analysis for decision '{decision_id}': "
                f"Custom rules {'would change' if custom_result['would_change'] else 'would keep'} "
                f"the choice (vs actual rules which "
                f"{'would change' if actual_result['would_change'] else 'would keep'} it)."
            ),
            data={
                "decision_id": decision_id,
                "original_choice": decision.get("choice", ""),
                "custom_rules_result": {
                    "recommendation": custom_result["recommendation"],
                    "would_change": custom_result["would_change"],
                    "confidence": custom_result["confidence"],
                    "rules_applied": len(custom_result["applicable_rules"]),
                    "reasoning": custom_result["reasoning"],
                },
                "actual_rules_result": {
                    "recommendation": actual_result["recommendation"],
                    "would_change": actual_result["would_change"],
                    "confidence": actual_result["confidence"],
                    "rules_applied": len(actual_result["applicable_rules"]),
                    "reasoning": actual_result["reasoning"],
                },
                "custom_rules_used": len(normalized_rules),
                "actual_rules_available": len(actual_rules),
            },
        )
