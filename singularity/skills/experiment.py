#!/usr/bin/env python3
"""
Experiment Skill - Autonomous A/B testing for agent self-improvement.

Enables agents to run controlled experiments on their own behavior,
measure outcomes, and automatically adopt winning strategies.

This is the core feedback loop: act → measure → adapt.

Example experiments an agent might run:
- "Does a more concise system prompt produce better results?"
- "Is model X faster/cheaper than model Y for code tasks?"
- "Does breaking tasks into subtasks improve success rate?"

The skill tracks experiments persistently so agents can learn
across sessions what strategies work best.
"""

import json
import time
import math
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
from .base import Skill, SkillManifest, SkillAction, SkillResult


class ExperimentStatus(str, Enum):
    """Status of an experiment."""
    DRAFT = "draft"           # Defined but not started
    RUNNING = "running"       # Actively collecting data
    PAUSED = "paused"         # Temporarily paused
    CONCLUDED = "concluded"   # Finished with a winner
    ABANDONED = "abandoned"   # Stopped without conclusion


@dataclass
class Variant:
    """A variant (arm) in an experiment."""
    name: str
    description: str
    config: Dict[str, Any] = field(default_factory=dict)
    # Tracking
    trials: int = 0
    successes: int = 0
    total_score: float = 0.0
    total_cost: float = 0.0
    total_time: float = 0.0

    @property
    def success_rate(self) -> float:
        return self.successes / self.trials if self.trials > 0 else 0.0

    @property
    def avg_score(self) -> float:
        return self.total_score / self.trials if self.trials > 0 else 0.0

    @property
    def avg_cost(self) -> float:
        return self.total_cost / self.trials if self.trials > 0 else 0.0

    @property
    def avg_time(self) -> float:
        return self.total_time / self.trials if self.trials > 0 else 0.0


@dataclass
class Experiment:
    """A controlled experiment with multiple variants."""
    id: str
    name: str
    hypothesis: str
    metric: str  # What we're optimizing: "success_rate", "score", "cost", "time"
    minimize: bool = False  # True if lower is better (cost, time)
    variants: Dict[str, Variant] = field(default_factory=dict)
    status: ExperimentStatus = ExperimentStatus.DRAFT
    min_trials_per_variant: int = 5
    created_at: float = field(default_factory=time.time)
    concluded_at: Optional[float] = None
    winner: Optional[str] = None
    conclusion: Optional[str] = None


def _experiment_id(name: str) -> str:
    """Generate a deterministic experiment ID from name."""
    return hashlib.md5(name.encode()).hexdigest()[:12]


def _wilson_lower_bound(successes: int, trials: int, z: float = 1.96) -> float:
    """
    Wilson score interval lower bound.
    Used for comparing success rates with different sample sizes.
    """
    if trials == 0:
        return 0.0
    p = successes / trials
    denominator = 1 + z * z / trials
    center = p + z * z / (2 * trials)
    spread = z * math.sqrt((p * (1 - p) + z * z / (4 * trials)) / trials)
    return (center - spread) / denominator


class ExperimentSkill(Skill):
    """
    Skill for running controlled experiments on agent behavior.

    Enables the self-improvement feedback loop:
    1. Define hypothesis and variants
    2. Run trials, recording outcomes
    3. Analyze results with statistical rigor
    4. Adopt the winning strategy
    """

    DATA_DIR = Path(__file__).parent.parent / "data" / "experiments"

    def __init__(self, credentials: Dict = None):
        super().__init__(credentials)
        self._experiments: Dict[str, Experiment] = {}
        self._loaded = False

    def _ensure_loaded(self):
        """Lazy-load experiments from disk."""
        if self._loaded:
            return
        self._loaded = True
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        index_file = self.DATA_DIR / "index.json"
        if index_file.exists():
            try:
                data = json.loads(index_file.read_text())
                for exp_data in data.get("experiments", []):
                    exp = self._deserialize_experiment(exp_data)
                    self._experiments[exp.id] = exp
            except (json.JSONDecodeError, KeyError):
                pass

    def _save(self):
        """Persist experiments to disk."""
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        index_file = self.DATA_DIR / "index.json"
        data = {
            "experiments": [
                self._serialize_experiment(exp)
                for exp in self._experiments.values()
            ]
        }
        index_file.write_text(json.dumps(data, indent=2))

    def _serialize_experiment(self, exp: Experiment) -> Dict:
        """Serialize an experiment to a dict."""
        return {
            "id": exp.id,
            "name": exp.name,
            "hypothesis": exp.hypothesis,
            "metric": exp.metric,
            "minimize": exp.minimize,
            "status": exp.status.value,
            "min_trials_per_variant": exp.min_trials_per_variant,
            "created_at": exp.created_at,
            "concluded_at": exp.concluded_at,
            "winner": exp.winner,
            "conclusion": exp.conclusion,
            "variants": {
                name: {
                    "name": v.name,
                    "description": v.description,
                    "config": v.config,
                    "trials": v.trials,
                    "successes": v.successes,
                    "total_score": v.total_score,
                    "total_cost": v.total_cost,
                    "total_time": v.total_time,
                }
                for name, v in exp.variants.items()
            },
        }

    def _deserialize_experiment(self, data: Dict) -> Experiment:
        """Deserialize an experiment from a dict."""
        variants = {}
        for name, v_data in data.get("variants", {}).items():
            variants[name] = Variant(
                name=v_data["name"],
                description=v_data["description"],
                config=v_data.get("config", {}),
                trials=v_data.get("trials", 0),
                successes=v_data.get("successes", 0),
                total_score=v_data.get("total_score", 0.0),
                total_cost=v_data.get("total_cost", 0.0),
                total_time=v_data.get("total_time", 0.0),
            )
        return Experiment(
            id=data["id"],
            name=data["name"],
            hypothesis=data["hypothesis"],
            metric=data["metric"],
            minimize=data.get("minimize", False),
            variants=variants,
            status=ExperimentStatus(data.get("status", "draft")),
            min_trials_per_variant=data.get("min_trials_per_variant", 5),
            created_at=data.get("created_at", time.time()),
            concluded_at=data.get("concluded_at"),
            winner=data.get("winner"),
            conclusion=data.get("conclusion"),
        )

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="experiment",
            name="Experiment",
            version="1.0.0",
            category="self-improvement",
            description="Run A/B experiments on your own behavior to find what works best",
            actions=[
                SkillAction(
                    name="create",
                    description="Create a new experiment with a hypothesis and variants to test",
                    parameters={
                        "name": {
                            "type": "string",
                            "required": True,
                            "description": "Short name for the experiment",
                        },
                        "hypothesis": {
                            "type": "string",
                            "required": True,
                            "description": "What you expect to happen and why",
                        },
                        "metric": {
                            "type": "string",
                            "required": True,
                            "description": "What to optimize: 'success_rate', 'score', 'cost', or 'time'",
                        },
                        "minimize": {
                            "type": "boolean",
                            "required": False,
                            "description": "True if lower metric is better (for cost/time). Default: False",
                        },
                        "variants": {
                            "type": "array",
                            "required": True,
                            "description": "List of variants, each with 'name', 'description', and optional 'config' dict",
                        },
                        "min_trials": {
                            "type": "integer",
                            "required": False,
                            "description": "Minimum trials per variant before concluding. Default: 5",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="start",
                    description="Start running an experiment (changes status from draft to running)",
                    parameters={
                        "experiment_id": {
                            "type": "string",
                            "required": True,
                            "description": "ID of the experiment to start",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="pick_variant",
                    description="Get the next variant to try (uses Thompson sampling for optimal exploration/exploitation)",
                    parameters={
                        "experiment_id": {
                            "type": "string",
                            "required": True,
                            "description": "ID of the running experiment",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="record_outcome",
                    description="Record the outcome of a trial for a specific variant",
                    parameters={
                        "experiment_id": {
                            "type": "string",
                            "required": True,
                            "description": "ID of the experiment",
                        },
                        "variant": {
                            "type": "string",
                            "required": True,
                            "description": "Name of the variant that was tested",
                        },
                        "success": {
                            "type": "boolean",
                            "required": True,
                            "description": "Whether the trial was successful",
                        },
                        "score": {
                            "type": "number",
                            "required": False,
                            "description": "Numeric score for this trial (e.g., quality rating 0-1)",
                        },
                        "cost": {
                            "type": "number",
                            "required": False,
                            "description": "Cost of this trial in USD",
                        },
                        "duration": {
                            "type": "number",
                            "required": False,
                            "description": "Duration of this trial in seconds",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="analyze",
                    description="Analyze current experiment results and check if a winner can be declared",
                    parameters={
                        "experiment_id": {
                            "type": "string",
                            "required": True,
                            "description": "ID of the experiment to analyze",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="conclude",
                    description="Conclude an experiment, declaring a winner and recording learnings",
                    parameters={
                        "experiment_id": {
                            "type": "string",
                            "required": True,
                            "description": "ID of the experiment to conclude",
                        },
                        "winner": {
                            "type": "string",
                            "required": False,
                            "description": "Override the winner (defaults to best performing variant)",
                        },
                        "conclusion": {
                            "type": "string",
                            "required": False,
                            "description": "Written conclusion about what was learned",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="list",
                    description="List all experiments and their status",
                    parameters={
                        "status": {
                            "type": "string",
                            "required": False,
                            "description": "Filter by status: 'draft', 'running', 'concluded', etc.",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="get",
                    description="Get detailed info about a specific experiment",
                    parameters={
                        "experiment_id": {
                            "type": "string",
                            "required": True,
                            "description": "ID of the experiment",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="learnings",
                    description="Get a summary of all learnings from concluded experiments",
                    parameters={},
                    estimated_cost=0,
                ),
            ],
            required_credentials=[],
        )

    def check_credentials(self) -> bool:
        return True  # No credentials needed

    async def execute(self, action: str, params: Dict) -> SkillResult:
        self._ensure_loaded()

        if action == "create":
            return self._create(params)
        elif action == "start":
            return self._start(params.get("experiment_id", ""))
        elif action == "pick_variant":
            return self._pick_variant(params.get("experiment_id", ""))
        elif action == "record_outcome":
            return self._record_outcome(params)
        elif action == "analyze":
            return self._analyze(params.get("experiment_id", ""))
        elif action == "conclude":
            return self._conclude(params)
        elif action == "list":
            return self._list(params.get("status"))
        elif action == "get":
            return self._get(params.get("experiment_id", ""))
        elif action == "learnings":
            return self._learnings()
        else:
            return SkillResult(success=False, message=f"Unknown action: {action}")

    def _create(self, params: Dict) -> SkillResult:
        name = params.get("name", "").strip()
        hypothesis = params.get("hypothesis", "").strip()
        metric = params.get("metric", "success_rate").strip()
        minimize = params.get("minimize", False)
        variant_list = params.get("variants", [])
        min_trials = params.get("min_trials", 5)

        if not name:
            return SkillResult(success=False, message="Experiment name is required")
        if not hypothesis:
            return SkillResult(success=False, message="Hypothesis is required")
        if metric not in ("success_rate", "score", "cost", "time"):
            return SkillResult(
                success=False,
                message=f"Invalid metric '{metric}'. Use: success_rate, score, cost, time"
            )
        if not variant_list or len(variant_list) < 2:
            return SkillResult(
                success=False,
                message="At least 2 variants are required for an experiment"
            )

        exp_id = _experiment_id(name)
        if exp_id in self._experiments:
            return SkillResult(
                success=False,
                message=f"Experiment '{name}' already exists (id: {exp_id})"
            )

        # Parse variants
        variants = {}
        for v in variant_list:
            if isinstance(v, str):
                v = {"name": v, "description": v}
            v_name = v.get("name", "").strip()
            if not v_name:
                continue
            variants[v_name] = Variant(
                name=v_name,
                description=v.get("description", v_name),
                config=v.get("config", {}),
            )

        if len(variants) < 2:
            return SkillResult(
                success=False,
                message="At least 2 valid variants are required"
            )

        exp = Experiment(
            id=exp_id,
            name=name,
            hypothesis=hypothesis,
            metric=metric,
            minimize=minimize,
            variants=variants,
            min_trials_per_variant=min_trials,
        )
        self._experiments[exp_id] = exp
        self._save()

        return SkillResult(
            success=True,
            message=f"Experiment '{name}' created with {len(variants)} variants",
            data={
                "experiment_id": exp_id,
                "name": name,
                "variants": list(variants.keys()),
                "metric": metric,
                "status": "draft",
            },
        )

    def _start(self, experiment_id: str) -> SkillResult:
        exp = self._experiments.get(experiment_id)
        if not exp:
            return SkillResult(success=False, message=f"Experiment '{experiment_id}' not found")
        if exp.status not in (ExperimentStatus.DRAFT, ExperimentStatus.PAUSED):
            return SkillResult(
                success=False,
                message=f"Cannot start experiment in '{exp.status.value}' status"
            )
        exp.status = ExperimentStatus.RUNNING
        self._save()
        return SkillResult(
            success=True,
            message=f"Experiment '{exp.name}' is now running",
            data={"experiment_id": exp.id, "status": "running"},
        )

    def _pick_variant(self, experiment_id: str) -> SkillResult:
        """
        Pick the next variant to test using Thompson Sampling.

        Thompson Sampling is a Bayesian approach that naturally balances
        exploration (trying under-tested variants) and exploitation
        (favoring variants that look good). It's optimal for multi-armed
        bandit problems.
        """
        import random

        exp = self._experiments.get(experiment_id)
        if not exp:
            return SkillResult(success=False, message=f"Experiment '{experiment_id}' not found")
        if exp.status != ExperimentStatus.RUNNING:
            return SkillResult(
                success=False,
                message=f"Experiment is not running (status: {exp.status.value})"
            )

        # Thompson Sampling: sample from Beta distribution for each variant
        best_sample = -1.0
        best_variant = None

        for name, v in exp.variants.items():
            # Beta(successes + 1, failures + 1) - uniform prior
            alpha = v.successes + 1
            beta_param = (v.trials - v.successes) + 1
            sample = random.betavariate(alpha, beta_param)

            # For minimize metrics, invert the sample
            if exp.minimize:
                sample = 1.0 - sample

            if sample > best_sample:
                best_sample = sample
                best_variant = name

        variant = exp.variants[best_variant]
        return SkillResult(
            success=True,
            message=f"Try variant '{best_variant}': {variant.description}",
            data={
                "variant": best_variant,
                "description": variant.description,
                "config": variant.config,
                "trials_so_far": variant.trials,
                "current_success_rate": variant.success_rate,
            },
        )

    def _record_outcome(self, params: Dict) -> SkillResult:
        experiment_id = params.get("experiment_id", "")
        variant_name = params.get("variant", "")
        success = params.get("success", False)
        score = params.get("score", 1.0 if success else 0.0)
        cost = params.get("cost", 0.0)
        duration = params.get("duration", 0.0)

        exp = self._experiments.get(experiment_id)
        if not exp:
            return SkillResult(success=False, message=f"Experiment '{experiment_id}' not found")
        if exp.status != ExperimentStatus.RUNNING:
            return SkillResult(
                success=False,
                message=f"Experiment is not running (status: {exp.status.value})"
            )

        variant = exp.variants.get(variant_name)
        if not variant:
            return SkillResult(
                success=False,
                message=f"Variant '{variant_name}' not found in experiment"
            )

        # Record the trial
        variant.trials += 1
        if success:
            variant.successes += 1
        variant.total_score += score
        variant.total_cost += cost
        variant.total_time += duration

        self._save()

        # Check if we have enough data to auto-analyze
        all_have_min = all(
            v.trials >= exp.min_trials_per_variant
            for v in exp.variants.values()
        )

        return SkillResult(
            success=True,
            message=f"Recorded {'success' if success else 'failure'} for '{variant_name}' "
                    f"(trial #{variant.trials}, rate: {variant.success_rate:.0%})",
            data={
                "variant": variant_name,
                "trials": variant.trials,
                "success_rate": variant.success_rate,
                "avg_score": variant.avg_score,
                "ready_to_conclude": all_have_min,
            },
        )

    def _analyze(self, experiment_id: str) -> SkillResult:
        """Analyze experiment results with statistical rigor."""
        exp = self._experiments.get(experiment_id)
        if not exp:
            return SkillResult(success=False, message=f"Experiment '{experiment_id}' not found")

        results = []
        for name, v in exp.variants.items():
            result = {
                "variant": name,
                "trials": v.trials,
                "successes": v.successes,
                "success_rate": v.success_rate,
                "avg_score": v.avg_score,
                "avg_cost": v.avg_cost,
                "avg_time": v.avg_time,
            }
            # Wilson score for confidence
            if v.trials > 0:
                result["wilson_lower"] = _wilson_lower_bound(v.successes, v.trials)
            else:
                result["wilson_lower"] = 0.0
            results.append(result)

        # Determine the metric-appropriate ranking
        metric = exp.metric
        if metric == "success_rate":
            # Use Wilson lower bound for fair comparison
            key_fn = lambda r: r["wilson_lower"]
            reverse = not exp.minimize
        elif metric == "score":
            key_fn = lambda r: r["avg_score"]
            reverse = not exp.minimize
        elif metric == "cost":
            key_fn = lambda r: r["avg_cost"] if r["trials"] > 0 else float('inf')
            reverse = exp.minimize  # Lower cost = better, so don't reverse when minimize=True
        elif metric == "time":
            key_fn = lambda r: r["avg_time"] if r["trials"] > 0 else float('inf')
            reverse = exp.minimize
        else:
            key_fn = lambda r: r["success_rate"]
            reverse = True

        results.sort(key=key_fn, reverse=reverse)

        # Check if we can declare a winner
        all_have_min = all(
            v.trials >= exp.min_trials_per_variant
            for v in exp.variants.values()
        )

        # Simple significance check: is the leader clearly ahead?
        can_conclude = False
        confidence = "low"
        if all_have_min and len(results) >= 2:
            if metric == "success_rate" and results[0]["trials"] >= 5:
                leader_wilson = results[0]["wilson_lower"]
                runner_up_rate = results[1]["success_rate"]
                if leader_wilson > runner_up_rate:
                    can_conclude = True
                    confidence = "high"
                elif results[0]["success_rate"] > results[1]["success_rate"]:
                    can_conclude = True
                    confidence = "moderate"
            else:
                # For other metrics, just check if leader is ahead
                can_conclude = True
                confidence = "moderate"

        return SkillResult(
            success=True,
            message=f"Analysis of '{exp.name}': {len(results)} variants, "
                    f"leader is '{results[0]['variant']}' "
                    f"({'ready to conclude' if can_conclude else 'need more data'})",
            data={
                "experiment_id": exp.id,
                "name": exp.name,
                "status": exp.status.value,
                "metric": metric,
                "results": results,
                "can_conclude": can_conclude,
                "confidence": confidence,
                "leader": results[0]["variant"] if results else None,
            },
        )

    def _conclude(self, params: Dict) -> SkillResult:
        experiment_id = params.get("experiment_id", "")
        override_winner = params.get("winner")
        conclusion_text = params.get("conclusion", "")

        exp = self._experiments.get(experiment_id)
        if not exp:
            return SkillResult(success=False, message=f"Experiment '{experiment_id}' not found")

        # Get analysis to find winner
        analysis = self._analyze(experiment_id)
        if not analysis.success:
            return analysis

        winner = override_winner or analysis.data.get("leader")
        if not winner or winner not in exp.variants:
            return SkillResult(
                success=False,
                message=f"Invalid winner '{winner}'. Must be one of: {list(exp.variants.keys())}"
            )

        exp.status = ExperimentStatus.CONCLUDED
        exp.concluded_at = time.time()
        exp.winner = winner
        exp.conclusion = conclusion_text or (
            f"Variant '{winner}' won on {exp.metric}. "
            f"Hypothesis: {exp.hypothesis}"
        )

        self._save()

        winner_variant = exp.variants[winner]
        return SkillResult(
            success=True,
            message=f"Experiment '{exp.name}' concluded. Winner: '{winner}' "
                    f"(rate: {winner_variant.success_rate:.0%}, "
                    f"trials: {winner_variant.trials})",
            data={
                "experiment_id": exp.id,
                "winner": winner,
                "winner_config": winner_variant.config,
                "winner_stats": {
                    "trials": winner_variant.trials,
                    "success_rate": winner_variant.success_rate,
                    "avg_score": winner_variant.avg_score,
                    "avg_cost": winner_variant.avg_cost,
                },
                "conclusion": exp.conclusion,
            },
        )

    def _list(self, status_filter: Optional[str] = None) -> SkillResult:
        experiments = []
        for exp in self._experiments.values():
            if status_filter and exp.status.value != status_filter:
                continue
            total_trials = sum(v.trials for v in exp.variants.values())
            experiments.append({
                "id": exp.id,
                "name": exp.name,
                "status": exp.status.value,
                "metric": exp.metric,
                "variants": len(exp.variants),
                "total_trials": total_trials,
                "winner": exp.winner,
            })

        return SkillResult(
            success=True,
            message=f"Found {len(experiments)} experiments"
                    + (f" (filtered: {status_filter})" if status_filter else ""),
            data={"experiments": experiments},
        )

    def _get(self, experiment_id: str) -> SkillResult:
        exp = self._experiments.get(experiment_id)
        if not exp:
            return SkillResult(success=False, message=f"Experiment '{experiment_id}' not found")

        return SkillResult(
            success=True,
            message=f"Experiment '{exp.name}' ({exp.status.value})",
            data=self._serialize_experiment(exp),
        )

    def _learnings(self) -> SkillResult:
        """Get a summary of all learnings from concluded experiments."""
        concluded = [
            exp for exp in self._experiments.values()
            if exp.status == ExperimentStatus.CONCLUDED
        ]

        if not concluded:
            return SkillResult(
                success=True,
                message="No concluded experiments yet. Run some experiments to learn!",
                data={"learnings": [], "total_experiments": len(self._experiments)},
            )

        learnings = []
        for exp in concluded:
            winner_v = exp.variants.get(exp.winner)
            learnings.append({
                "experiment": exp.name,
                "hypothesis": exp.hypothesis,
                "winner": exp.winner,
                "winner_config": winner_v.config if winner_v else {},
                "conclusion": exp.conclusion,
                "metric": exp.metric,
                "winner_success_rate": winner_v.success_rate if winner_v else 0,
                "total_trials": sum(v.trials for v in exp.variants.values()),
            })

        return SkillResult(
            success=True,
            message=f"{len(learnings)} learnings from concluded experiments",
            data={
                "learnings": learnings,
                "total_experiments": len(self._experiments),
                "concluded": len(concluded),
            },
        )
