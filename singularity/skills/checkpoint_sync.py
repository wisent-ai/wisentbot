#!/usr/bin/env python3
"""
CrossAgentCheckpointSyncSkill - Share checkpoint analytics between replicas for fleet-wide progress.

AgentCheckpointSkill creates local checkpoints. AgentNetworkSkill discovers peers.
CheckpointComparisonAnalyticsSkill analyzes local progress. But none of them share
checkpoint data across agent replicas. This skill bridges them:

1. SHARE - Publish a checkpoint summary to the fleet so other agents can see your progress
2. PULL - Fetch checkpoint summaries from peer agents for comparison
3. FLEET_TIMELINE - Build a fleet-wide timeline showing all agents' progress over time
4. DIVERGENCE - Detect when replicas diverge significantly from each other
5. BEST_PRACTICES - Identify which agent config/behaviors produce the best progress
6. SYNC_POLICY - Configure auto-sharing rules (share on checkpoint create, on milestone, etc.)
7. MERGE_INSIGHTS - Combine learnings from the best-performing agents into local state
8. STATUS - View sync state, connected peers, and sharing stats

Without this, replicas are blind to each other's progress. With it, the fleet can:
- Identify which replica is most successful and why
- Detect divergence early before replicas waste resources on bad strategies
- Auto-propagate winning strategies across the fleet
- Build a unified progress view for fleet-level goal setting

Pillars: Replication (fleet coordination), Self-Improvement (learn from best replica),
         Goal Setting (fleet-wide progress tracking)
"""

import json
import time
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import Skill, SkillResult, SkillManifest, SkillAction

SYNC_FILE = Path(__file__).parent.parent / "data" / "checkpoint_sync.json"
MAX_LOG_ENTRIES = 500
MAX_FLEET_SNAPSHOTS = 200
MAX_INSIGHTS = 100


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _summary_id(agent_id: str, checkpoint_id: str) -> str:
    raw = f"{agent_id}:{checkpoint_id}"
    return f"csync_{hashlib.sha256(raw.encode()).hexdigest()[:12]}"


class CrossAgentCheckpointSyncSkill(Skill):
    """Share checkpoint analytics between agent replicas for fleet-wide progress tracking."""

    def __init__(self, credentials: Dict[str, str] = None):
        self.credentials = credentials or {}
        self._ensure_data()

    def _ensure_data(self):
        SYNC_FILE.parent.mkdir(parents=True, exist_ok=True)
        if not SYNC_FILE.exists():
            self._save(self._default_state())

    def _default_state(self) -> Dict:
        return {
            "config": {
                "auto_share_on_checkpoint": True,
                "auto_pull_interval_minutes": 60,
                "divergence_threshold": 0.3,
                "share_full_data": False,  # Only share summaries by default
                "emit_events": True,
            },
            "local_agent_id": "",
            "shared_summaries": {},  # summary_id -> {agent_id, checkpoint_id, data, timestamp}
            "fleet_snapshots": [],   # [{timestamp, agents: {agent_id: summary}}]
            "peer_summaries": {},    # agent_id -> {latest_summary, last_seen, ...}
            "insights": [],          # [{source_agent, insight, timestamp}]
            "event_log": [],
            "stats": {
                "shares_sent": 0,
                "shares_received": 0,
                "pull_requests": 0,
                "divergence_alerts": 0,
                "insights_merged": 0,
                "fleet_snapshots_created": 0,
            },
        }

    def _load(self) -> Dict:
        try:
            return json.loads(SYNC_FILE.read_text())
        except Exception:
            return self._default_state()

    def _save(self, data: Dict):
        if len(data.get("event_log", [])) > MAX_LOG_ENTRIES:
            data["event_log"] = data["event_log"][-MAX_LOG_ENTRIES:]
        if len(data.get("fleet_snapshots", [])) > MAX_FLEET_SNAPSHOTS:
            data["fleet_snapshots"] = data["fleet_snapshots"][-MAX_FLEET_SNAPSHOTS:]
        if len(data.get("insights", [])) > MAX_INSIGHTS:
            data["insights"] = data["insights"][-MAX_INSIGHTS:]
        SYNC_FILE.parent.mkdir(parents=True, exist_ok=True)
        SYNC_FILE.write_text(json.dumps(data, indent=2, default=str))

    def _log_event(self, data: Dict, event_type: str, details: Dict):
        entry = {"timestamp": _now_iso(), "event": event_type, **details}
        data["event_log"].append(entry)

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="cross_agent_checkpoint_sync",
            name="CrossAgentCheckpointSync",
            version="1.0.0",
            category="replication",
            description="Share checkpoint analytics between replicas for fleet-wide progress tracking",
            actions=self.get_actions(),
            required_credentials=[],
        )

    def get_actions(self) -> List[SkillAction]:
        return [
            SkillAction(
                name="share",
                description="Publish a checkpoint summary to the fleet",
                parameters={
                    "agent_id": {"type": "string", "required": True, "description": "This agent's ID"},
                    "checkpoint_id": {"type": "string", "required": True, "description": "Checkpoint ID to share"},
                    "pillar_scores": {"type": "object", "required": False, "description": "Pillar maturity scores {pillar: score}"},
                    "file_stats": {"type": "object", "required": False, "description": "Data file stats {file: {size, entries}}"},
                    "label": {"type": "string", "required": False, "description": "Human-readable checkpoint label"},
                    "skills_active": {"type": "array", "required": False, "description": "List of active skill names"},
                    "experiments_running": {"type": "integer", "required": False, "description": "Number of active experiments"},
                    "goals_completed": {"type": "integer", "required": False, "description": "Number of goals completed"},
                },
            ),
            SkillAction(
                name="pull",
                description="Fetch checkpoint summaries from peer agents",
                parameters={
                    "peer_agent_id": {"type": "string", "required": False, "description": "Specific peer to pull from, or all peers"},
                },
            ),
            SkillAction(
                name="fleet_timeline",
                description="Build fleet-wide timeline showing all agents' progress over time",
                parameters={
                    "limit": {"type": "integer", "required": False, "description": "Max snapshots to return"},
                },
            ),
            SkillAction(
                name="divergence",
                description="Detect when replicas diverge significantly from each other",
                parameters={
                    "agent_a": {"type": "string", "required": False, "description": "First agent to compare"},
                    "agent_b": {"type": "string", "required": False, "description": "Second agent to compare"},
                },
            ),
            SkillAction(
                name="best_practices",
                description="Identify which agent produces the best progress scores",
                parameters={},
            ),
            SkillAction(
                name="sync_policy",
                description="Configure auto-sharing rules",
                parameters={
                    "auto_share_on_checkpoint": {"type": "boolean", "required": False},
                    "auto_pull_interval_minutes": {"type": "integer", "required": False},
                    "divergence_threshold": {"type": "number", "required": False},
                    "share_full_data": {"type": "boolean", "required": False},
                },
            ),
            SkillAction(
                name="merge_insights",
                description="Combine learnings from best-performing agents into local state",
                parameters={
                    "source_agent_id": {"type": "string", "required": True, "description": "Agent to learn from"},
                    "insight": {"type": "string", "required": True, "description": "The insight or strategy to adopt"},
                    "category": {"type": "string", "required": False, "description": "Category: strategy, config, behavior"},
                },
            ),
            SkillAction(
                name="status",
                description="View sync state, connected peers, and sharing stats",
                parameters={},
            ),
        ]

    async def execute(self, action: str, params: Dict[str, Any] = None) -> SkillResult:
        params = params or {}
        handlers = {
            "share": self._share,
            "pull": self._pull,
            "fleet_timeline": self._fleet_timeline,
            "divergence": self._divergence,
            "best_practices": self._best_practices,
            "sync_policy": self._sync_policy,
            "merge_insights": self._merge_insights,
            "status": self._status,
        }
        handler = handlers.get(action)
        if not handler:
            return SkillResult(success=False, message=f"Unknown action: {action}")
        return await handler(params)

    async def _share(self, params: Dict) -> SkillResult:
        """Publish a checkpoint summary to the fleet."""
        agent_id = params.get("agent_id", "")
        checkpoint_id = params.get("checkpoint_id", "")
        if not agent_id or not checkpoint_id:
            return SkillResult(success=False, message="agent_id and checkpoint_id required")

        data = self._load()
        sid = _summary_id(agent_id, checkpoint_id)

        if sid in data["shared_summaries"]:
            return SkillResult(success=False, message=f"Checkpoint {checkpoint_id} already shared by {agent_id}")

        summary = {
            "agent_id": agent_id,
            "checkpoint_id": checkpoint_id,
            "timestamp": _now_iso(),
            "pillar_scores": params.get("pillar_scores", {}),
            "file_stats": params.get("file_stats", {}),
            "label": params.get("label", ""),
            "skills_active": params.get("skills_active", []),
            "experiments_running": params.get("experiments_running", 0),
            "goals_completed": params.get("goals_completed", 0),
        }

        data["shared_summaries"][sid] = summary
        data["local_agent_id"] = agent_id
        data["stats"]["shares_sent"] += 1

        # Update peer tracking
        if agent_id not in data["peer_summaries"]:
            data["peer_summaries"][agent_id] = {
                "first_seen": _now_iso(),
                "summaries_count": 0,
                "latest_pillar_scores": {},
            }
        peer = data["peer_summaries"][agent_id]
        peer["last_seen"] = _now_iso()
        peer["summaries_count"] = peer.get("summaries_count", 0) + 1
        peer["latest_pillar_scores"] = summary["pillar_scores"]
        peer["latest_checkpoint_id"] = checkpoint_id
        peer["latest_label"] = summary["label"]

        self._log_event(data, "checkpoint_shared", {
            "agent_id": agent_id,
            "checkpoint_id": checkpoint_id,
            "summary_id": sid,
        })
        self._save(data)

        return SkillResult(
            success=True,
            message=f"Shared checkpoint {checkpoint_id} from {agent_id} (summary_id={sid})",
            data={"summary_id": sid, "summary": summary},
        )

    async def _pull(self, params: Dict) -> SkillResult:
        """Fetch checkpoint summaries from peer agents."""
        data = self._load()
        peer_filter = params.get("peer_agent_id")

        summaries = {}
        for sid, summary in data["shared_summaries"].items():
            aid = summary["agent_id"]
            if peer_filter and aid != peer_filter:
                continue
            if aid not in summaries:
                summaries[aid] = []
            summaries[aid].append(summary)

        data["stats"]["pull_requests"] += 1
        self._log_event(data, "pull_executed", {
            "peer_filter": peer_filter,
            "agents_found": len(summaries),
            "summaries_found": sum(len(v) for v in summaries.values()),
        })
        self._save(data)

        return SkillResult(
            success=True,
            message=f"Pulled summaries from {len(summaries)} agents ({sum(len(v) for v in summaries.values())} total)",
            data={"peer_summaries": summaries, "agents": list(summaries.keys())},
        )

    async def _fleet_timeline(self, params: Dict) -> SkillResult:
        """Build fleet-wide timeline showing all agents' progress over time."""
        data = self._load()
        limit = params.get("limit", 20)

        # Build timeline from shared summaries grouped by timestamp
        timeline_entries = []
        for sid, summary in data["shared_summaries"].items():
            timeline_entries.append({
                "timestamp": summary["timestamp"],
                "agent_id": summary["agent_id"],
                "checkpoint_id": summary["checkpoint_id"],
                "label": summary.get("label", ""),
                "pillar_scores": summary.get("pillar_scores", {}),
                "goals_completed": summary.get("goals_completed", 0),
                "experiments_running": summary.get("experiments_running", 0),
                "skills_active_count": len(summary.get("skills_active", [])),
            })

        # Sort by timestamp
        timeline_entries.sort(key=lambda x: x["timestamp"])

        # Take a fleet snapshot for the current state
        current_snapshot = {}
        for aid, peer in data["peer_summaries"].items():
            current_snapshot[aid] = {
                "last_seen": peer.get("last_seen", ""),
                "pillar_scores": peer.get("latest_pillar_scores", {}),
                "summaries_count": peer.get("summaries_count", 0),
            }

        if current_snapshot:
            snapshot_entry = {
                "timestamp": _now_iso(),
                "agents": current_snapshot,
            }
            data["fleet_snapshots"].append(snapshot_entry)
            data["stats"]["fleet_snapshots_created"] += 1
            self._save(data)

        return SkillResult(
            success=True,
            message=f"Fleet timeline: {len(timeline_entries)} entries across {len(set(e['agent_id'] for e in timeline_entries))} agents",
            data={
                "timeline": timeline_entries[-limit:],
                "total_entries": len(timeline_entries),
                "current_fleet_state": current_snapshot,
            },
        )

    async def _divergence(self, params: Dict) -> SkillResult:
        """Detect when replicas diverge significantly from each other."""
        data = self._load()
        agent_a = params.get("agent_a")
        agent_b = params.get("agent_b")
        threshold = data["config"]["divergence_threshold"]

        peers = data["peer_summaries"]

        if agent_a and agent_b:
            # Compare two specific agents
            pairs = [(agent_a, agent_b)]
        else:
            # Compare all pairs
            agent_ids = list(peers.keys())
            pairs = []
            for i in range(len(agent_ids)):
                for j in range(i + 1, len(agent_ids)):
                    pairs.append((agent_ids[i], agent_ids[j]))

        if not pairs:
            return SkillResult(
                success=True,
                message="No agent pairs to compare (need at least 2 agents with shared checkpoints)",
                data={"comparisons": [], "alerts": []},
            )

        comparisons = []
        alerts = []

        for a_id, b_id in pairs:
            a_scores = peers.get(a_id, {}).get("latest_pillar_scores", {})
            b_scores = peers.get(b_id, {}).get("latest_pillar_scores", {})

            if not a_scores and not b_scores:
                continue

            # Calculate divergence across pillars
            all_pillars = set(list(a_scores.keys()) + list(b_scores.keys()))
            pillar_diffs = {}
            total_diff = 0.0

            for pillar in all_pillars:
                a_val = a_scores.get(pillar, 0)
                b_val = b_scores.get(pillar, 0)
                diff = abs(a_val - b_val)
                # Normalize to 0-1 range (scores are 0-100)
                norm_diff = diff / 100.0 if isinstance(a_val, (int, float)) and isinstance(b_val, (int, float)) else 0
                pillar_diffs[pillar] = {
                    "agent_a_score": a_val,
                    "agent_b_score": b_val,
                    "absolute_diff": diff,
                    "normalized_diff": round(norm_diff, 3),
                }
                total_diff += norm_diff

            avg_divergence = total_diff / len(all_pillars) if all_pillars else 0
            is_divergent = avg_divergence > threshold

            comparison = {
                "agent_a": a_id,
                "agent_b": b_id,
                "avg_divergence": round(avg_divergence, 3),
                "threshold": threshold,
                "is_divergent": is_divergent,
                "pillar_diffs": pillar_diffs,
            }
            comparisons.append(comparison)

            if is_divergent:
                # Find the most divergent pillar
                worst_pillar = max(pillar_diffs.items(), key=lambda x: x[1]["normalized_diff"])
                alert = {
                    "agent_a": a_id,
                    "agent_b": b_id,
                    "divergence": round(avg_divergence, 3),
                    "worst_pillar": worst_pillar[0],
                    "worst_diff": worst_pillar[1]["normalized_diff"],
                    "recommendation": f"Agents diverge most on {worst_pillar[0]}. Consider sharing strategies from the higher-scoring agent.",
                }
                alerts.append(alert)
                data["stats"]["divergence_alerts"] += 1

        if alerts:
            self._log_event(data, "divergence_detected", {
                "alert_count": len(alerts),
                "pairs_checked": len(pairs),
            })
            self._save(data)

        return SkillResult(
            success=True,
            message=f"Checked {len(comparisons)} agent pairs: {len(alerts)} divergence alerts",
            data={"comparisons": comparisons, "alerts": alerts},
        )

    async def _best_practices(self, params: Dict) -> SkillResult:
        """Identify which agent produces the best progress scores."""
        data = self._load()
        peers = data["peer_summaries"]

        if not peers:
            return SkillResult(
                success=True,
                message="No peer data available. Share checkpoints first.",
                data={"rankings": [], "pillar_leaders": {}},
            )

        # Rank agents by overall score
        rankings = []
        pillar_leaders = {}  # pillar -> {agent_id, score}

        for agent_id, peer in peers.items():
            scores = peer.get("latest_pillar_scores", {})
            if not scores:
                continue

            total = sum(v for v in scores.values() if isinstance(v, (int, float)))
            avg = total / len(scores) if scores else 0

            rankings.append({
                "agent_id": agent_id,
                "total_score": total,
                "avg_score": round(avg, 1),
                "pillar_scores": scores,
                "summaries_shared": peer.get("summaries_count", 0),
                "last_seen": peer.get("last_seen", ""),
            })

            # Track per-pillar leaders
            for pillar, score in scores.items():
                if not isinstance(score, (int, float)):
                    continue
                if pillar not in pillar_leaders or score > pillar_leaders[pillar]["score"]:
                    pillar_leaders[pillar] = {"agent_id": agent_id, "score": score}

        rankings.sort(key=lambda x: x["total_score"], reverse=True)

        # Generate recommendations
        recommendations = []
        if len(rankings) >= 2:
            best = rankings[0]
            worst = rankings[-1]
            for pillar, leader in pillar_leaders.items():
                if leader["agent_id"] != worst["agent_id"]:
                    worst_score = worst["pillar_scores"].get(pillar, 0)
                    if isinstance(worst_score, (int, float)) and leader["score"] - worst_score > 20:
                        recommendations.append({
                            "action": f"Merge {pillar} strategies from {leader['agent_id']} to {worst['agent_id']}",
                            "expected_improvement": round(leader["score"] - worst_score, 1),
                            "source_agent": leader["agent_id"],
                            "target_agent": worst["agent_id"],
                            "pillar": pillar,
                        })

        return SkillResult(
            success=True,
            message=f"Ranked {len(rankings)} agents. {len(recommendations)} improvement recommendations.",
            data={
                "rankings": rankings,
                "pillar_leaders": pillar_leaders,
                "recommendations": recommendations,
            },
        )

    async def _sync_policy(self, params: Dict) -> SkillResult:
        """Configure auto-sharing rules."""
        data = self._load()
        changed = []

        for key in ["auto_share_on_checkpoint", "auto_pull_interval_minutes",
                     "divergence_threshold", "share_full_data"]:
            if key in params:
                old_val = data["config"].get(key)
                data["config"][key] = params[key]
                changed.append(f"{key}: {old_val} -> {params[key]}")

        if not changed:
            return SkillResult(
                success=True,
                message="No changes specified. Current config returned.",
                data={"config": data["config"]},
            )

        self._log_event(data, "policy_updated", {"changes": changed})
        self._save(data)

        return SkillResult(
            success=True,
            message=f"Updated sync policy: {', '.join(changed)}",
            data={"config": data["config"]},
        )

    async def _merge_insights(self, params: Dict) -> SkillResult:
        """Combine learnings from best-performing agents into local state."""
        source_agent = params.get("source_agent_id", "")
        insight_text = params.get("insight", "")
        category = params.get("category", "strategy")

        if not source_agent or not insight_text:
            return SkillResult(success=False, message="source_agent_id and insight are required")

        data = self._load()

        # Check for duplicate insights
        for existing in data["insights"]:
            if existing["source_agent"] == source_agent and existing["insight"] == insight_text:
                return SkillResult(
                    success=False,
                    message=f"Insight already recorded from {source_agent}",
                )

        insight_entry = {
            "id": f"ins_{hashlib.sha256(f'{source_agent}:{insight_text}'.encode()).hexdigest()[:10]}",
            "source_agent": source_agent,
            "insight": insight_text,
            "category": category,
            "timestamp": _now_iso(),
            "applied": False,
        }

        data["insights"].append(insight_entry)
        data["stats"]["insights_merged"] += 1

        self._log_event(data, "insight_merged", {
            "source_agent": source_agent,
            "category": category,
            "insight_id": insight_entry["id"],
        })
        self._save(data)

        return SkillResult(
            success=True,
            message=f"Merged insight from {source_agent} ({category}): {insight_text[:80]}",
            data={"insight": insight_entry},
        )

    async def _status(self, params: Dict) -> SkillResult:
        """View sync state, connected peers, and sharing stats."""
        data = self._load()

        peers_info = {}
        for agent_id, peer in data["peer_summaries"].items():
            peers_info[agent_id] = {
                "last_seen": peer.get("last_seen", ""),
                "summaries_count": peer.get("summaries_count", 0),
                "latest_pillar_scores": peer.get("latest_pillar_scores", {}),
            }

        return SkillResult(
            success=True,
            message=f"Sync status: {len(peers_info)} peers, {data['stats']['shares_sent']} sent, {data['stats']['pull_requests']} pulls",
            data={
                "local_agent_id": data["local_agent_id"],
                "config": data["config"],
                "peers": peers_info,
                "stats": data["stats"],
                "insights_count": len(data["insights"]),
                "fleet_snapshots_count": len(data["fleet_snapshots"]),
                "total_shared_summaries": len(data["shared_summaries"]),
            },
        )

    async def estimate_cost(self, action: str, params: Dict = None) -> float:
        return 0.0
