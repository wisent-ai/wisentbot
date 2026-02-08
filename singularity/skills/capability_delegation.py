#!/usr/bin/env python3
"""
CapabilityAwareDelegationSkill - Route tasks to agents based on capability profiles.

Without this skill, task delegation is blind - tasks go to arbitrary agents or
manually chosen ones. With capability-aware delegation, the agent can:

1. Match task requirements to agent capabilities (via SelfAssessmentSkill profiles)
2. Rank candidate agents by capability score + reputation
3. Auto-delegate to the best-fit agent
4. Track match quality over time to improve future routing

This bridges:
- SelfAssessmentSkill (capability profiles)
- AgentReputationSkill (trust/competence scores)
- TaskDelegationSkill (actual task assignment)
- AgentNetworkSkill (agent discovery)

Flow:
  task requirements → match against capability profiles → rank by fit + reputation → delegate

Pillar: Replication + Self-Improvement
- Replication: Multi-agent coordination with smart task routing
- Self-Improvement: Better delegation = better outcomes = higher reputation

Actions:
- match: Find best agent(s) for a task based on required capabilities
- delegate: Match + auto-delegate in one step
- profiles: View known agent capability profiles
- history: View past capability-based delegations and match quality
- configure: Set matching weights and thresholds
- status: Current state and stats
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from .base import Skill, SkillResult, SkillManifest, SkillAction

DATA_FILE = Path(__file__).parent.parent / "data" / "capability_delegation.json"
MAX_HISTORY = 200


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


# Skill-to-category mapping for matching tasks to capabilities
SKILL_CATEGORIES = {
    "code_review": "self_improvement",
    "self_modify": "self_improvement",
    "experiment": "self_improvement",
    "self_tuning": "self_improvement",
    "self_assessment": "self_improvement",
    "prompt_evolution": "self_improvement",
    "feedback_loop": "self_improvement",
    "content": "revenue",
    "payment": "revenue",
    "revenue_services": "revenue",
    "marketplace": "revenue",
    "data_transform": "revenue",
    "replication": "replication",
    "agent_network": "replication",
    "task_delegation": "replication",
    "consensus": "replication",
    "knowledge_sharing": "replication",
    "planner": "goal_setting",
    "strategy": "goal_setting",
    "goal_manager": "goal_setting",
    "scheduler": "operations",
    "deployment": "operations",
    "health_monitor": "operations",
    "observability": "operations",
    "email": "communication",
    "messaging": "communication",
    "notification": "communication",
}


class CapabilityAwareDelegationSkill(Skill):
    """
    Route tasks to the best-fit agents based on capability profiles and reputation.

    Combines SelfAssessmentSkill capability scores with AgentReputationSkill
    trust scores to make intelligent delegation decisions.
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._store = None

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="capability_delegation",
            name="Capability-Aware Delegation",
            version="1.0.0",
            category="replication",
            description="Route tasks to best-fit agents based on capability profiles and reputation scores",
            actions=self.get_actions(),
            required_credentials=[],
            install_cost=0,
            author="singularity",
        )

    def get_actions(self) -> List[SkillAction]:
        return [
            SkillAction(
                name="match",
                description="Find best agent(s) for a task based on required skills/categories",
                parameters={
                    "task_name": {"type": "string", "required": True,
                                  "description": "Name of the task to match"},
                    "required_skills": {"type": "array", "required": False,
                                        "description": "List of skill IDs the task needs"},
                    "required_categories": {"type": "array", "required": False,
                                            "description": "List of capability categories needed (self_improvement, revenue, replication, goal_setting, operations, communication)"},
                    "top_n": {"type": "integer", "required": False,
                              "description": "Return top N matches (default: 3)"},
                },
            ),
            SkillAction(
                name="delegate",
                description="Match best agent and auto-delegate the task in one step",
                parameters={
                    "task_name": {"type": "string", "required": True,
                                  "description": "Name of the task"},
                    "task_description": {"type": "string", "required": False,
                                          "description": "Task description"},
                    "required_skills": {"type": "array", "required": False,
                                        "description": "Required skill IDs"},
                    "required_categories": {"type": "array", "required": False,
                                            "description": "Required capability categories"},
                    "budget": {"type": "number", "required": False,
                               "description": "Budget for the delegation (default: 1.0)"},
                },
            ),
            SkillAction(
                name="profiles",
                description="View known agent capability profiles",
                parameters={
                    "agent_id": {"type": "string", "required": False,
                                 "description": "Filter by specific agent ID"},
                },
            ),
            SkillAction(
                name="history",
                description="View past capability-based delegations",
                parameters={
                    "limit": {"type": "integer", "required": False,
                              "description": "Max entries to return (default: 20)"},
                },
            ),
            SkillAction(
                name="configure",
                description="Set matching weights and thresholds",
                parameters={
                    "capability_weight": {"type": "number", "required": False,
                                          "description": "Weight for capability score (default: 0.6)"},
                    "reputation_weight": {"type": "number", "required": False,
                                          "description": "Weight for reputation score (default: 0.4)"},
                    "min_match_score": {"type": "number", "required": False,
                                        "description": "Minimum score to be considered a match (default: 0.3)"},
                    "emit_events": {"type": "boolean", "required": False,
                                    "description": "Emit events on delegation (default: True)"},
                },
            ),
            SkillAction(
                name="status",
                description="Current state and delegation stats",
                parameters={},
            ),
        ]

    async def execute(self, action: str, params: Dict) -> SkillResult:
        handlers = {
            "match": self._match,
            "delegate": self._delegate,
            "profiles": self._profiles,
            "history": self._history,
            "configure": self._configure,
            "status": self._status,
        }
        handler = handlers.get(action)
        if not handler:
            return SkillResult(
                success=False,
                message=f"Unknown action: {action}. Available: {', '.join(handlers.keys())}",
            )
        return await handler(params)

    # ── Persistence ──

    def _load(self) -> Dict:
        if self._store is not None:
            return self._store
        try:
            if DATA_FILE.exists():
                with open(DATA_FILE) as f:
                    self._store = json.load(f)
                    return self._store
        except (json.JSONDecodeError, OSError):
            pass
        self._store = self._default_state()
        return self._store

    def _save(self, state: Dict = None):
        if state is not None:
            self._store = state
        if self._store is None:
            return
        DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(DATA_FILE, "w") as f:
            json.dump(self._store, f, indent=2)

    def _default_state(self) -> Dict:
        return {
            "config": {
                "capability_weight": 0.6,
                "reputation_weight": 0.4,
                "min_match_score": 0.3,
                "emit_events": True,
            },
            "cached_profiles": {},  # agent_id -> {categories, updated_at}
            "history": [],
            "stats": {
                "total_matches": 0,
                "total_delegations": 0,
                "avg_match_score": 0,
            },
        }

    # ── Handlers ──

    async def _match(self, params: Dict) -> SkillResult:
        """Find best agents for a task."""
        task_name = params.get("task_name", "").strip()
        if not task_name:
            return SkillResult(success=False, message="task_name is required")

        required_skills = params.get("required_skills", [])
        required_categories = params.get("required_categories", [])
        top_n = min(int(params.get("top_n", 3)), 10)

        if not required_skills and not required_categories:
            # Try to infer categories from task name
            required_categories = self._infer_categories(task_name)

        # Get agent profiles
        profiles = await self._get_agent_profiles()
        if not profiles:
            return SkillResult(
                success=False,
                message="No agent capability profiles available. Ensure agents have published profiles via SelfAssessmentSkill.",
            )

        # Get reputation data
        reputations = await self._get_reputations()

        state = self._load()
        config = state["config"]

        # Score each agent
        rankings = []
        for agent_id, profile in profiles.items():
            cap_score = self._compute_capability_score(
                profile, required_skills, required_categories
            )
            rep_score = self._get_reputation_score(reputations, agent_id)

            combined = (
                config["capability_weight"] * cap_score +
                config["reputation_weight"] * rep_score
            )

            if combined >= config["min_match_score"]:
                rankings.append({
                    "agent_id": agent_id,
                    "combined_score": round(combined, 3),
                    "capability_score": round(cap_score, 3),
                    "reputation_score": round(rep_score, 3),
                    "matching_categories": self._get_matching_categories(
                        profile, required_categories
                    ),
                })

        rankings.sort(key=lambda x: x["combined_score"], reverse=True)
        rankings = rankings[:top_n]

        state["stats"]["total_matches"] += 1
        self._save(state)

        if not rankings:
            return SkillResult(
                success=True,
                message=f"No agents match the requirements for '{task_name}' above threshold {config['min_match_score']}.",
                data={"matches": [], "requirements": {"skills": required_skills, "categories": required_categories}},
            )

        return SkillResult(
            success=True,
            message=f"Found {len(rankings)} matching agent(s) for '{task_name}'. "
                    f"Best: {rankings[0]['agent_id']} (score: {rankings[0]['combined_score']})",
            data={
                "matches": rankings,
                "requirements": {
                    "skills": required_skills,
                    "categories": required_categories or self._infer_categories(task_name),
                },
            },
        )

    async def _delegate(self, params: Dict) -> SkillResult:
        """Match + auto-delegate in one step."""
        task_name = params.get("task_name", "").strip()
        if not task_name:
            return SkillResult(success=False, message="task_name is required")

        # First, find best match
        match_result = await self._match(params)
        if not match_result.success:
            return match_result

        matches = match_result.data.get("matches", [])
        if not matches:
            return SkillResult(
                success=False,
                message=f"No suitable agents found for '{task_name}'.",
                data=match_result.data,
            )

        best = matches[0]
        task_description = params.get("task_description", task_name)
        budget = float(params.get("budget", 1.0))

        # Delegate via TaskDelegationSkill
        delegation_result = await self._do_delegate(
            agent_id=best["agent_id"],
            task_name=task_name,
            task_description=task_description,
            budget=budget,
        )

        state = self._load()
        record = {
            "task_name": task_name,
            "agent_id": best["agent_id"],
            "combined_score": best["combined_score"],
            "capability_score": best["capability_score"],
            "reputation_score": best["reputation_score"],
            "delegation_success": delegation_result is not None and delegation_result.get("success"),
            "delegation_id": delegation_result.get("delegation_id") if delegation_result else None,
            "timestamp": _now_iso(),
        }
        state["history"].append(record)
        state["history"] = state["history"][-MAX_HISTORY:]
        state["stats"]["total_delegations"] += 1

        # Update rolling avg match score
        n = state["stats"]["total_delegations"]
        old_avg = state["stats"]["avg_match_score"]
        state["stats"]["avg_match_score"] = round(
            old_avg + (best["combined_score"] - old_avg) / n, 3
        )
        self._save(state)

        if state["config"].get("emit_events", True):
            await self._emit_event("capability_delegation.delegated", record)

        return SkillResult(
            success=True,
            message=f"Delegated '{task_name}' to {best['agent_id']} "
                    f"(match: {best['combined_score']}, cap: {best['capability_score']}, rep: {best['reputation_score']})",
            data={
                "match": best,
                "delegation": delegation_result,
                "all_matches": matches,
            },
        )

    async def _profiles(self, params: Dict) -> SkillResult:
        """View known agent capability profiles."""
        agent_filter = params.get("agent_id")
        profiles = await self._get_agent_profiles()

        if agent_filter:
            if agent_filter in profiles:
                profiles = {agent_filter: profiles[agent_filter]}
            else:
                return SkillResult(
                    success=False,
                    message=f"No profile found for agent '{agent_filter}'.",
                )

        return SkillResult(
            success=True,
            message=f"{len(profiles)} agent profile(s) available.",
            data={"profiles": profiles},
        )

    async def _history(self, params: Dict) -> SkillResult:
        """View delegation history."""
        state = self._load()
        limit = min(int(params.get("limit", 20)), MAX_HISTORY)
        entries = state["history"][-limit:]
        return SkillResult(
            success=True,
            message=f"Showing {len(entries)} capability-based delegations.",
            data={"history": entries, "total": len(state["history"])},
        )

    async def _configure(self, params: Dict) -> SkillResult:
        """Update configuration."""
        state = self._load()
        config = state["config"]
        updated = []

        for key in ["capability_weight", "reputation_weight", "min_match_score", "emit_events"]:
            if key in params:
                old = config[key]
                config[key] = params[key]
                updated.append(f"{key}: {old} -> {params[key]}")

        if not updated:
            return SkillResult(
                success=True,
                message="No configuration changes requested.",
                data={"config": config},
            )

        self._save(state)
        return SkillResult(
            success=True,
            message=f"Updated {len(updated)} setting(s): {'; '.join(updated)}",
            data={"config": config},
        )

    async def _status(self, params: Dict) -> SkillResult:
        """Return status."""
        state = self._load()
        profiles = await self._get_agent_profiles()
        return SkillResult(
            success=True,
            message=f"Capability delegation active. {state['stats']['total_delegations']} delegations, "
                    f"{len(profiles)} agent profiles available.",
            data={
                "config": state["config"],
                "stats": state["stats"],
                "agent_count": len(profiles),
            },
        )

    # ── Internal helpers ──

    def _infer_categories(self, task_name: str) -> List[str]:
        """Infer required categories from task name keywords."""
        task_lower = task_name.lower()
        categories = set()

        keyword_map = {
            "self_improvement": ["review", "test", "refactor", "optimize", "tune", "assess", "improve"],
            "revenue": ["content", "write", "generate", "payment", "invoice", "service", "customer"],
            "replication": ["spawn", "delegate", "coordinate", "network", "replicate", "agent"],
            "goal_setting": ["plan", "strategy", "goal", "prioritize", "roadmap"],
            "operations": ["deploy", "monitor", "health", "fix", "incident", "alert"],
            "communication": ["email", "message", "notify", "send", "communicate"],
        }

        for category, keywords in keyword_map.items():
            if any(kw in task_lower for kw in keywords):
                categories.add(category)

        return list(categories) if categories else ["operations"]

    def _compute_capability_score(
        self, profile: Dict, required_skills: List[str], required_categories: List[str]
    ) -> float:
        """Compute how well an agent's capabilities match requirements."""
        if not required_skills and not required_categories:
            return 0.5  # neutral

        scores = []
        categories = profile.get("categories", {})

        # Score by required categories
        for cat in required_categories:
            cat_data = categories.get(cat, {})
            cat_score = cat_data.get("score", 0) / 100.0  # Normalize to 0-1
            scores.append(cat_score)

        # Score by required skills
        installed_skills = set(profile.get("installed_skills", []))
        for skill_id in required_skills:
            if skill_id in installed_skills:
                scores.append(1.0)
            else:
                scores.append(0.0)

        return sum(scores) / len(scores) if scores else 0.5

    def _get_reputation_score(self, reputations: Dict, agent_id: str) -> float:
        """Get normalized reputation score for an agent."""
        if not reputations or agent_id not in reputations:
            return 0.5  # neutral default

        rep = reputations[agent_id]
        overall = rep.get("overall", 50.0)
        return overall / 100.0  # Normalize to 0-1

    def _get_matching_categories(self, profile: Dict, required_categories: List[str]) -> List[str]:
        """Get which required categories the agent covers."""
        categories = profile.get("categories", {})
        return [cat for cat in required_categories if cat in categories and categories[cat].get("score", 0) > 0]

    async def _get_agent_profiles(self) -> Dict:
        """Get agent capability profiles."""
        # Try via AgentNetworkSkill
        if self.context:
            try:
                result = await self.context.call_skill("agent_network", "list", {})
                if result.success and result.data:
                    agents = result.data.get("agents", [])
                    profiles = {}
                    for agent in agents:
                        agent_id = agent.get("agent_id", "")
                        caps = agent.get("capabilities", {})
                        if agent_id and caps:
                            profiles[agent_id] = caps
                    if profiles:
                        return profiles
            except Exception:
                pass

        # Fallback: read from cached profiles in our state
        state = self._load()
        return state.get("cached_profiles", {})

    async def _get_reputations(self) -> Dict:
        """Get agent reputation data."""
        if self.context:
            try:
                result = await self.context.call_skill("agent_reputation", "leaderboard", {"limit": 50})
                if result.success and result.data:
                    reps = {}
                    for entry in result.data.get("leaderboard", []):
                        reps[entry["agent_id"]] = entry
                    return reps
            except Exception:
                pass

        # Fallback: read reputation file
        try:
            rep_file = Path(__file__).parent.parent / "data" / "agent_reputation.json"
            if rep_file.exists():
                with open(rep_file) as f:
                    data = json.load(f)
                    return data.get("reputations", {})
        except (json.JSONDecodeError, OSError):
            pass

        return {}

    async def _do_delegate(self, agent_id: str, task_name: str,
                           task_description: str, budget: float) -> Optional[Dict]:
        """Delegate task via TaskDelegationSkill."""
        if self.context:
            try:
                result = await self.context.call_skill("task_delegation", "delegate", {
                    "agent_id": agent_id,
                    "task_name": task_name,
                    "task_description": task_description,
                    "budget": budget,
                })
                return {
                    "success": result.success,
                    "delegation_id": result.data.get("delegation_id") if result.data else None,
                    "message": result.message,
                }
            except Exception as e:
                return {"success": False, "error": str(e)}

        return {"success": True, "delegation_id": f"cap_del_{int(time.time())}", "note": "simulated"}

    async def _emit_event(self, topic: str, data: Dict):
        """Emit event via EventBus."""
        if not self.context:
            return
        try:
            await self.context.call_skill("event", "publish", {
                "topic": topic,
                "data": data,
            })
        except Exception:
            pass

    async def initialize(self) -> bool:
        self.initialized = True
        return True
