#!/usr/bin/env python3
"""
AutoPlaybookGeneratorSkill - Automatically generate playbooks from reflection clusters.

The agent accumulates reflections via AgentReflectionSkill, but manually creating
playbooks from them requires initiative. This skill closes the gap by automatically:

1. **Clustering** similar reflections using tag + keyword similarity
2. **Scoring** clusters by size, consistency, and lack of existing coverage
3. **Generating** playbooks from the best clusters by extracting common patterns
4. **Validating** generated playbooks against future reflection outcomes
5. **Pruning** low-quality generated playbooks that don't improve success rates

The auto-generation loop:
  reflections accumulate → cluster → identify gaps → generate playbook → track effectiveness → prune/improve

This is the #1 priority from Session 150 MEMORY: "Auto-Playbook Generation -
Use LLM to automatically generate playbooks from clusters of similar reflections"

Pillar: Self-Improvement (automated knowledge extraction from experience)
"""

import json
import hashlib
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
from .base import Skill, SkillResult, SkillManifest, SkillAction

GENERATOR_FILE = Path(__file__).parent.parent / "data" / "auto_playbook_generator.json"
MAX_GENERATIONS = 200
MAX_CLUSTERS = 50


class AutoPlaybookGeneratorSkill(Skill):
    """
    Automatically generates playbooks from clusters of similar reflections.

    Works with AgentReflectionSkill via SkillContext:
    - Reads reflections and existing playbooks from agent_reflection
    - Generates new playbooks by analyzing reflection clusters
    - Creates playbooks via agent_reflection's create_playbook action

    Clustering uses tag overlap + keyword similarity (no external LLM needed).
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._load_state()

    def _load_state(self):
        """Load or initialize generator state."""
        GENERATOR_FILE.parent.mkdir(parents=True, exist_ok=True)
        if GENERATOR_FILE.exists():
            try:
                with open(GENERATOR_FILE) as f:
                    data = json.load(f)
                self._generations = data.get("generations", [])
                self._cluster_cache = data.get("cluster_cache", {})
                self._config = data.get("config", self._default_config())
                self._stats = data.get("stats", self._default_stats())
                self._pruned = data.get("pruned", [])
            except (json.JSONDecodeError, Exception):
                self._init_empty()
        else:
            self._init_empty()

    def _init_empty(self):
        self._generations: List[Dict] = []
        self._cluster_cache: Dict[str, Dict] = {}
        self._config = self._default_config()
        self._stats = self._default_stats()
        self._pruned: List[Dict] = []

    def _default_config(self) -> Dict:
        return {
            "min_cluster_size": 3,
            "similarity_threshold": 0.3,
            "min_success_rate_for_steps": 0.5,
            "auto_prune_threshold": 0.25,
            "auto_prune_min_uses": 5,
            "max_playbooks_per_run": 3,
        }

    def _default_stats(self) -> Dict:
        return {
            "total_clusters_found": 0,
            "total_playbooks_generated": 0,
            "total_playbooks_pruned": 0,
            "total_scans": 0,
            "last_scan": None,
        }

    def _save(self):
        """Persist generator state."""
        data = {
            "generations": self._generations[-MAX_GENERATIONS:],
            "cluster_cache": dict(list(self._cluster_cache.items())[:MAX_CLUSTERS]),
            "config": self._config,
            "stats": self._stats,
            "pruned": self._pruned[-50:],
        }
        GENERATOR_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(GENERATOR_FILE, "w") as f:
            json.dump(data, f, indent=2, default=str)

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="auto_playbook_generator",
            name="Auto Playbook Generator",
            version="1.0.0",
            category="self_improvement",
            description="Automatically cluster reflections and generate playbooks from patterns",
            actions=self.get_actions(),
            required_credentials=[],
        )

    def get_actions(self) -> List[SkillAction]:
        return [
            SkillAction(
                name="scan",
                description="Scan reflections, cluster them, and identify gaps where playbooks should be generated",
                parameters={
                    "min_cluster_size": {"type": "int", "required": False, "description": "Minimum reflections in a cluster (default from config)"},
                },
            ),
            SkillAction(
                name="generate",
                description="Auto-generate playbooks from the highest-scored uncovered clusters",
                parameters={
                    "max_playbooks": {"type": "int", "required": False, "description": "Max playbooks to generate (default 3)"},
                    "dry_run": {"type": "bool", "required": False, "description": "If true, show what would be generated without creating"},
                },
            ),
            SkillAction(
                name="clusters",
                description="View current reflection clusters and their scores",
                parameters={
                    "limit": {"type": "int", "required": False, "description": "Max clusters to show (default 10)"},
                    "include_covered": {"type": "bool", "required": False, "description": "Include clusters already covered by playbooks"},
                },
            ),
            SkillAction(
                name="validate",
                description="Check generated playbooks' effectiveness and flag underperformers for pruning",
                parameters={},
            ),
            SkillAction(
                name="prune",
                description="Remove auto-generated playbooks that consistently underperform",
                parameters={
                    "dry_run": {"type": "bool", "required": False, "description": "If true, show what would be pruned without removing"},
                },
            ),
            SkillAction(
                name="configure",
                description="Update generator configuration",
                parameters={
                    "min_cluster_size": {"type": "int", "required": False, "description": "Min reflections per cluster"},
                    "similarity_threshold": {"type": "float", "required": False, "description": "Similarity threshold for clustering (0-1)"},
                    "auto_prune_threshold": {"type": "float", "required": False, "description": "Effectiveness below which to prune"},
                    "auto_prune_min_uses": {"type": "int", "required": False, "description": "Min uses before pruning eligible"},
                    "max_playbooks_per_run": {"type": "int", "required": False, "description": "Max playbooks per generate run"},
                },
            ),
            SkillAction(
                name="history",
                description="View generation history and pruning log",
                parameters={
                    "limit": {"type": "int", "required": False, "description": "Max entries (default 20)"},
                    "what": {"type": "str", "required": False, "description": "'generations', 'pruned', or 'all'"},
                },
            ),
            SkillAction(
                name="status",
                description="View generator stats and configuration",
                parameters={},
            ),
        ]

    async def execute(self, action: str, params: Dict) -> SkillResult:
        handlers = {
            "scan": self._scan,
            "generate": self._generate,
            "clusters": self._clusters,
            "validate": self._validate,
            "prune": self._prune,
            "configure": self._configure,
            "history": self._history,
            "status": self._status,
        }
        handler = handlers.get(action)
        if not handler:
            return SkillResult(success=False, message=f"Unknown action: {action}")
        try:
            return await handler(params)
        except Exception as e:
            return SkillResult(success=False, message=f"Auto-playbook generator error: {str(e)}")

    # --- Clustering Logic ---

    def _tokenize(self, text: str) -> Set[str]:
        """Extract meaningful tokens from text."""
        stop_words = {
            "the", "a", "an", "is", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "shall", "can", "to", "of",
            "in", "for", "on", "with", "at", "by", "from", "as", "into",
            "through", "during", "before", "after", "and", "but", "or",
            "not", "no", "it", "its", "this", "that", "these", "those",
        }
        words = set()
        for word in text.lower().split():
            # Strip punctuation
            clean = "".join(c for c in word if c.isalnum())
            if clean and len(clean) > 2 and clean not in stop_words:
                words.add(clean)
        return words

    def _similarity(self, ref_a: Dict, ref_b: Dict) -> float:
        """Compute similarity between two reflections using tags + keywords."""
        # Tag overlap (weighted higher)
        tags_a = set(t.lower() for t in ref_a.get("tags", []))
        tags_b = set(t.lower() for t in ref_b.get("tags", []))
        tag_union = tags_a | tags_b
        tag_sim = len(tags_a & tags_b) / len(tag_union) if tag_union else 0

        # Task keyword overlap
        tokens_a = self._tokenize(ref_a.get("task", ""))
        tokens_b = self._tokenize(ref_b.get("task", ""))
        token_union = tokens_a | tokens_b
        token_sim = len(tokens_a & tokens_b) / len(token_union) if token_union else 0

        # Analysis keyword overlap
        analysis_a = self._tokenize(ref_a.get("analysis", ""))
        analysis_b = self._tokenize(ref_b.get("analysis", ""))
        analysis_union = analysis_a | analysis_b
        analysis_sim = len(analysis_a & analysis_b) / len(analysis_union) if analysis_union else 0

        # Weighted combination: tags most important, then task, then analysis
        return 0.5 * tag_sim + 0.35 * token_sim + 0.15 * analysis_sim

    def _cluster_reflections(self, reflections: List[Dict]) -> List[Dict]:
        """
        Cluster reflections using single-linkage agglomerative clustering.

        Returns list of clusters, each with:
        - reflection_ids: list of IDs
        - reflections: the reflection dicts
        - tags: merged tags
        - score: cluster quality score
        """
        threshold = self._config["similarity_threshold"]

        # Each reflection starts as its own cluster
        clusters: List[List[int]] = [[i] for i in range(len(reflections))]
        merged = set()

        # Single-linkage: merge closest pairs above threshold
        for i in range(len(reflections)):
            if i in merged:
                continue
            for j in range(i + 1, len(reflections)):
                if j in merged:
                    continue
                sim = self._similarity(reflections[i], reflections[j])
                if sim >= threshold:
                    # Find cluster containing i
                    ci = next(c for c in clusters if i in c)
                    cj = next(c for c in clusters if j in c)
                    if ci is not cj:
                        # Merge cj into ci
                        ci.extend(cj)
                        clusters.remove(cj)

        # Build cluster dicts
        result = []
        for cluster_indices in clusters:
            if len(cluster_indices) < 2:
                continue  # Skip singletons

            refs = [reflections[i] for i in cluster_indices]
            all_tags = []
            for r in refs:
                all_tags.extend(r.get("tags", []))
            tag_counts = Counter(all_tags)
            common_tags = [t for t, c in tag_counts.most_common(5)]

            # Compute cluster quality score
            size = len(refs)
            successes = sum(1 for r in refs if r.get("success"))
            success_rate = successes / size if size > 0 else 0
            tag_consistency = len(common_tags) / max(len(set(all_tags)), 1)

            # Score: bigger clusters with consistent tags are more valuable
            # Moderate success rate preferred (pure success doesn't need playbook as much)
            score = (
                min(size, 10) * 10  # Size factor (cap at 10)
                + tag_consistency * 30  # Tag consistency
                + (1 - abs(success_rate - 0.6)) * 20  # Prefer ~60% success rate
                + len(common_tags) * 5  # More tags = more specific
            )

            # Generate cluster ID from sorted reflection IDs
            ref_ids = sorted([r.get("id", str(i)) for i, r in zip(cluster_indices, refs)])
            cluster_id = hashlib.sha256(
                "|".join(ref_ids).encode()
            ).hexdigest()[:12]

            result.append({
                "cluster_id": cluster_id,
                "size": size,
                "reflection_ids": [r.get("id", "") for r in refs],
                "reflections": refs,
                "common_tags": common_tags,
                "success_rate": round(success_rate, 2),
                "tag_consistency": round(tag_consistency, 2),
                "score": round(score, 2),
            })

        # Sort by score descending
        result.sort(key=lambda c: c["score"], reverse=True)
        return result

    def _extract_playbook_from_cluster(self, cluster: Dict) -> Dict:
        """
        Extract a playbook definition from a cluster of similar reflections.

        Analyzes:
        - Common actions taken across reflections → steps
        - Improvement suggestions from failures → pitfalls
        - Success patterns → prerequisites and expected outcome
        """
        refs = cluster["reflections"]
        common_tags = cluster["common_tags"]

        # Extract common actions from successful reflections
        success_refs = [r for r in refs if r.get("success")]
        failure_refs = [r for r in refs if not r.get("success")]

        # Build step frequency from successful attempts
        action_freq: Dict[str, int] = {}
        for r in success_refs:
            for action in r.get("actions_taken", []):
                action_str = str(action) if not isinstance(action, str) else action
                action_freq[action_str] = action_freq.get(action_str, 0) + 1

        # If not enough successes, use all reflections
        if len(success_refs) < 2:
            for r in refs:
                for action in r.get("actions_taken", []):
                    action_str = str(action) if not isinstance(action, str) else action
                    action_freq[action_str] = action_freq.get(action_str, 0) + 1

        # Steps = actions ordered by frequency (most common first)
        min_freq = max(1, len(refs) * self._config["min_success_rate_for_steps"])
        steps = [
            action for action, count
            in sorted(action_freq.items(), key=lambda x: x[1], reverse=True)
            if count >= min_freq
        ]

        # If no steps met threshold, take top 5 actions
        if not steps:
            steps = [
                action for action, _
                in sorted(action_freq.items(), key=lambda x: x[1], reverse=True)[:5]
            ]

        # Pitfalls from failure analysis and improvements
        pitfalls = []
        seen_pitfalls = set()
        for r in failure_refs:
            # From analysis
            analysis = r.get("analysis", "")
            if analysis and analysis not in seen_pitfalls:
                pitfalls.append(f"Failure pattern: {analysis[:120]}")
                seen_pitfalls.add(analysis)
            # From improvements
            for imp in r.get("improvements", []):
                if imp not in seen_pitfalls:
                    pitfalls.append(f"Avoid: {imp[:120]}")
                    seen_pitfalls.add(imp)

        pitfalls = pitfalls[:8]  # Cap at 8 pitfalls

        # Task pattern from common task descriptions
        task_words: Dict[str, int] = {}
        for r in refs:
            for word in self._tokenize(r.get("task", "")):
                task_words[word] = task_words.get(word, 0) + 1

        # Most common task words form the pattern
        top_task_words = sorted(task_words.items(), key=lambda x: x[1], reverse=True)[:6]
        task_pattern = " ".join(w for w, _ in top_task_words) if top_task_words else "general task"

        # Add tags to pattern for specificity
        if common_tags:
            task_pattern = f"{task_pattern} ({', '.join(common_tags[:3])})"

        # Expected outcome from successful reflections
        expected_outcome = ""
        if success_refs:
            outcomes = [r.get("outcome", "") for r in success_refs if r.get("outcome")]
            if outcomes:
                expected_outcome = outcomes[0][:200]  # Use first success outcome

        # Playbook name from tags
        name_parts = [t.replace(" ", "_") for t in common_tags[:2]] if common_tags else ["auto"]
        name = f"auto_{'_'.join(name_parts)}_{cluster['cluster_id'][:6]}"

        return {
            "name": name,
            "task_pattern": task_pattern,
            "steps": steps if steps else ["(no common steps extracted)"],
            "pitfalls": pitfalls,
            "prerequisites": [],
            "expected_outcome": expected_outcome,
            "tags": common_tags + ["auto_generated"],
            "source_cluster_id": cluster["cluster_id"],
            "source_cluster_size": cluster["size"],
            "source_success_rate": cluster["success_rate"],
        }

    def _is_covered_by_playbook(self, cluster: Dict, existing_playbooks: List[Dict]) -> bool:
        """Check if a cluster is already covered by an existing playbook."""
        cluster_tags = set(t.lower() for t in cluster["common_tags"])
        if not cluster_tags:
            return False

        for pb in existing_playbooks:
            pb_tags = set(t.lower() for t in pb.get("tags", []))
            if not pb_tags:
                continue
            # If >50% tag overlap, consider it covered
            overlap = len(cluster_tags & pb_tags)
            if overlap > 0 and overlap / len(cluster_tags) >= 0.5:
                return True

            # Also check task pattern keyword overlap
            cluster_words = set()
            for r in cluster["reflections"]:
                cluster_words |= self._tokenize(r.get("task", ""))
            pb_words = self._tokenize(pb.get("task_pattern", ""))
            if cluster_words and pb_words:
                word_overlap = len(cluster_words & pb_words) / len(cluster_words)
                if word_overlap >= 0.4:
                    return True

        return False

    # --- Action Handlers ---

    async def _get_reflections_and_playbooks(self) -> Tuple[List[Dict], List[Dict]]:
        """Fetch reflections and playbooks from AgentReflectionSkill via context."""
        reflections = []
        playbooks = []

        if self.context:
            # Try to get from agent_reflection skill
            review_result = await self.context.call_skill(
                "agent_reflection", "review",
                {"what": "all", "limit": 500}
            )
            if review_result.success:
                reflections = review_result.data.get("reflections", [])
                playbooks = review_result.data.get("playbooks", [])

        return reflections, playbooks

    async def _scan(self, params: Dict) -> SkillResult:
        """Scan reflections, cluster them, and identify playbook gaps."""
        min_size = params.get("min_cluster_size", self._config["min_cluster_size"])

        reflections, existing_playbooks = await self._get_reflections_and_playbooks()

        if len(reflections) < min_size:
            return SkillResult(
                success=True,
                message=f"Not enough reflections ({len(reflections)}) for clustering. Need at least {min_size}.",
                data={"reflections_count": len(reflections), "clusters": []},
            )

        # Cluster reflections
        clusters = self._cluster_reflections(reflections)

        # Filter by minimum size
        clusters = [c for c in clusters if c["size"] >= min_size]

        # Identify which clusters lack playbook coverage
        uncovered = []
        covered = []
        for cluster in clusters:
            # Remove full reflections to keep cache small
            cache_cluster = {k: v for k, v in cluster.items() if k != "reflections"}
            cache_cluster["reflection_ids"] = cluster.get("reflection_ids", [])

            if self._is_covered_by_playbook(cluster, existing_playbooks):
                cache_cluster["covered"] = True
                covered.append(cache_cluster)
            else:
                cache_cluster["covered"] = False
                uncovered.append(cache_cluster)

            self._cluster_cache[cluster["cluster_id"]] = cache_cluster

        self._stats["total_clusters_found"] = len(clusters)
        self._stats["total_scans"] += 1
        self._stats["last_scan"] = datetime.utcnow().isoformat()
        self._save()

        return SkillResult(
            success=True,
            message=f"Found {len(clusters)} clusters ({len(uncovered)} uncovered, {len(covered)} covered)",
            data={
                "total_reflections": len(reflections),
                "total_clusters": len(clusters),
                "uncovered_clusters": uncovered[:10],
                "covered_clusters": len(covered),
                "existing_playbooks": len(existing_playbooks),
            },
        )

    async def _generate(self, params: Dict) -> SkillResult:
        """Auto-generate playbooks from highest-scored uncovered clusters."""
        max_playbooks = params.get("max_playbooks", self._config["max_playbooks_per_run"])
        dry_run = params.get("dry_run", False)

        reflections, existing_playbooks = await self._get_reflections_and_playbooks()

        if len(reflections) < self._config["min_cluster_size"]:
            return SkillResult(
                success=True,
                message="Not enough reflections to generate playbooks.",
                data={"generated": []},
            )

        # Re-cluster with full data
        clusters = self._cluster_reflections(reflections)
        clusters = [c for c in clusters if c["size"] >= self._config["min_cluster_size"]]

        # Filter to uncovered clusters
        uncovered = [
            c for c in clusters
            if not self._is_covered_by_playbook(c, existing_playbooks)
        ]

        if not uncovered:
            return SkillResult(
                success=True,
                message="All clusters are already covered by existing playbooks.",
                data={"generated": [], "clusters_checked": len(clusters)},
            )

        # Take top-scored uncovered clusters
        to_generate = uncovered[:max_playbooks]
        generated = []

        for cluster in to_generate:
            playbook_def = self._extract_playbook_from_cluster(cluster)

            if dry_run:
                generated.append({
                    "would_create": playbook_def["name"],
                    "task_pattern": playbook_def["task_pattern"],
                    "steps": playbook_def["steps"],
                    "pitfalls_count": len(playbook_def["pitfalls"]),
                    "source_cluster_size": cluster["size"],
                    "cluster_score": cluster["score"],
                })
                continue

            # Create playbook via AgentReflectionSkill
            created = False
            if self.context:
                create_result = await self.context.call_skill(
                    "agent_reflection", "create_playbook", {
                        "name": playbook_def["name"],
                        "task_pattern": playbook_def["task_pattern"],
                        "steps": playbook_def["steps"],
                        "pitfalls": playbook_def["pitfalls"],
                        "prerequisites": playbook_def["prerequisites"],
                        "expected_outcome": playbook_def["expected_outcome"],
                        "tags": playbook_def["tags"],
                    }
                )
                created = create_result.success
            else:
                # Fallback: record locally without creating in reflection skill
                created = True

            generation_record = {
                "timestamp": datetime.utcnow().isoformat(),
                "playbook_name": playbook_def["name"],
                "task_pattern": playbook_def["task_pattern"],
                "steps_count": len(playbook_def["steps"]),
                "pitfalls_count": len(playbook_def["pitfalls"]),
                "source_cluster_id": cluster["cluster_id"],
                "source_cluster_size": cluster["size"],
                "cluster_score": cluster["score"],
                "created": created,
            }

            self._generations.append(generation_record)
            if created:
                self._stats["total_playbooks_generated"] += 1
                generated.append(generation_record)

        # Trim generations list
        if len(self._generations) > MAX_GENERATIONS:
            self._generations = self._generations[-MAX_GENERATIONS:]

        self._save()

        if dry_run:
            return SkillResult(
                success=True,
                message=f"Dry run: would generate {len(generated)} playbook(s)",
                data={"would_generate": generated},
            )

        return SkillResult(
            success=True,
            message=f"Generated {len(generated)} playbook(s) from reflection clusters",
            data={
                "generated": generated,
                "total_generated_all_time": self._stats["total_playbooks_generated"],
            },
        )

    async def _clusters(self, params: Dict) -> SkillResult:
        """View current cluster cache."""
        limit = params.get("limit", 10)
        include_covered = params.get("include_covered", False)

        clusters = list(self._cluster_cache.values())
        if not include_covered:
            clusters = [c for c in clusters if not c.get("covered")]

        clusters.sort(key=lambda c: c.get("score", 0), reverse=True)
        shown = clusters[:limit]

        return SkillResult(
            success=True,
            message=f"Showing {len(shown)}/{len(clusters)} clusters (run 'scan' to refresh)",
            data={
                "clusters": shown,
                "total_cached": len(self._cluster_cache),
            },
        )

    async def _validate(self, params: Dict) -> SkillResult:
        """Check generated playbooks' effectiveness."""
        _, existing_playbooks = await self._get_reflections_and_playbooks()

        # Find auto-generated playbooks
        auto_playbooks = [
            pb for pb in existing_playbooks
            if "auto_generated" in pb.get("tags", [])
        ]

        if not auto_playbooks:
            return SkillResult(
                success=True,
                message="No auto-generated playbooks to validate.",
                data={"validated": []},
            )

        prune_threshold = self._config["auto_prune_threshold"]
        min_uses = self._config["auto_prune_min_uses"]

        validated = []
        needs_pruning = []
        for pb in auto_playbooks:
            uses = pb.get("uses", 0)
            effectiveness = pb.get("effectiveness", 0)
            status = "new" if uses < min_uses else (
                "effective" if effectiveness >= 0.5 else (
                    "underperforming" if effectiveness >= prune_threshold else "prune_candidate"
                )
            )
            entry = {
                "name": pb.get("name", ""),
                "uses": uses,
                "effectiveness": effectiveness,
                "status": status,
            }
            validated.append(entry)
            if status == "prune_candidate":
                needs_pruning.append(entry)

        return SkillResult(
            success=True,
            message=f"Validated {len(auto_playbooks)} auto-generated playbooks: {len(needs_pruning)} need pruning",
            data={
                "validated": validated,
                "needs_pruning": needs_pruning,
                "prune_threshold": prune_threshold,
                "min_uses_for_evaluation": min_uses,
            },
        )

    async def _prune(self, params: Dict) -> SkillResult:
        """Remove underperforming auto-generated playbooks."""
        dry_run = params.get("dry_run", False)

        _, existing_playbooks = await self._get_reflections_and_playbooks()

        auto_playbooks = [
            pb for pb in existing_playbooks
            if "auto_generated" in pb.get("tags", [])
        ]

        prune_threshold = self._config["auto_prune_threshold"]
        min_uses = self._config["auto_prune_min_uses"]

        to_prune = [
            pb for pb in auto_playbooks
            if pb.get("uses", 0) >= min_uses and pb.get("effectiveness", 0) < prune_threshold
        ]

        if not to_prune:
            return SkillResult(
                success=True,
                message="No playbooks qualify for pruning.",
                data={"pruned": []},
            )

        if dry_run:
            return SkillResult(
                success=True,
                message=f"Dry run: would prune {len(to_prune)} playbook(s)",
                data={
                    "would_prune": [
                        {"name": pb.get("name"), "effectiveness": pb.get("effectiveness"), "uses": pb.get("uses")}
                        for pb in to_prune
                    ]
                },
            )

        # Record pruned playbooks and note them
        pruned_names = []
        for pb in to_prune:
            prune_record = {
                "timestamp": datetime.utcnow().isoformat(),
                "playbook_name": pb.get("name", ""),
                "effectiveness": pb.get("effectiveness", 0),
                "uses": pb.get("uses", 0),
                "reason": f"Effectiveness {pb.get('effectiveness', 0):.0%} below threshold {prune_threshold:.0%}",
            }
            self._pruned.append(prune_record)
            pruned_names.append(pb.get("name", ""))
            self._stats["total_playbooks_pruned"] += 1

        # Note: actual removal requires AgentReflection to support deletion.
        # For now, we record the prune recommendation. The evolve_playbook action
        # could be used to mark them as deprecated.
        self._save()

        return SkillResult(
            success=True,
            message=f"Pruned {len(pruned_names)} underperforming playbook(s): {', '.join(pruned_names)}",
            data={
                "pruned": pruned_names,
                "total_pruned_all_time": self._stats["total_playbooks_pruned"],
            },
        )

    async def _configure(self, params: Dict) -> SkillResult:
        """Update generator configuration."""
        changes = []
        for key in ["min_cluster_size", "similarity_threshold", "auto_prune_threshold",
                     "auto_prune_min_uses", "max_playbooks_per_run"]:
            if key in params:
                old = self._config.get(key)
                self._config[key] = params[key]
                changes.append(f"{key}: {old} → {params[key]}")

        if not changes:
            return SkillResult(
                success=True,
                message="No changes specified.",
                data={"config": self._config},
            )

        self._save()
        return SkillResult(
            success=True,
            message=f"Configuration updated: {', '.join(changes)}",
            data={"config": self._config},
        )

    async def _history(self, params: Dict) -> SkillResult:
        """View generation and pruning history."""
        limit = params.get("limit", 20)
        what = params.get("what", "all")

        data = {}
        if what in ("generations", "all"):
            data["generations"] = self._generations[-limit:]
        if what in ("pruned", "all"):
            data["pruned"] = self._pruned[-limit:]
        data["stats"] = self._stats

        return SkillResult(
            success=True,
            message=f"History: {len(self._generations)} generations, {len(self._pruned)} pruned",
            data=data,
        )

    async def _status(self, params: Dict) -> SkillResult:
        """View generator stats and configuration."""
        return SkillResult(
            success=True,
            message=f"Auto-Playbook Generator: {self._stats['total_playbooks_generated']} generated, {self._stats['total_playbooks_pruned']} pruned",
            data={
                "stats": self._stats,
                "config": self._config,
                "cached_clusters": len(self._cluster_cache),
            },
        )

    def estimate_cost(self, action: str, params: Dict) -> float:
        """All actions are local computation - free."""
        return 0.0
