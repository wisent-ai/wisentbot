#!/usr/bin/env python3
"""
Local Memory Skill - Zero-dependency persistent memory for agents.

Provides agents with file-based persistent memory that works without
any external services or API keys. Supports:
- Storing and retrieving memories by topic/category
- Keyword-based search across all memories
- Memory consolidation (auto-summarize old entries)
- Session journaling (track what happened each session)
- Skill/action outcome tracking (learn what works)
- Cross-session persistence via JSON files

This is the foundational memory layer that enables self-improvement:
without memory, the agent cannot learn from past experience.
"""

import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import Counter

from .base import Skill, SkillManifest, SkillAction, SkillResult

# Default memory directory
DEFAULT_MEMORY_DIR = Path(__file__).parent.parent / "data" / "memory"


class LocalMemorySkill(Skill):
    """
    Zero-dependency local memory system using JSON files.

    Memory is organized into categories, each stored as a separate JSON file:
    - learnings.json: Things the agent has learned
    - outcomes.json: Action outcomes (what worked, what didn't)
    - goals.json: Goals and their progress
    - journal.json: Session logs and reflections
    - general.json: Uncategorized memories

    Each memory entry has:
    - id: Unique identifier
    - content: The memory text
    - tags: Keywords for search
    - created_at: Timestamp
    - relevance_score: How important (0-1)
    - access_count: How often retrieved
    - last_accessed: When last retrieved
    """

    CATEGORIES = ["learnings", "outcomes", "goals", "journal", "general"]
    MAX_ENTRIES_PER_CATEGORY = 500

    def __init__(self, credentials: Dict = None):
        super().__init__(credentials)
        self._memory_dir = DEFAULT_MEMORY_DIR
        self._agent_name: Optional[str] = None
        self._memory_cache: Dict[str, List[Dict]] = {}

    def set_memory_dir(self, path: str):
        """Set custom memory directory."""
        self._memory_dir = Path(path)

    def set_agent_context(self, agent_name: str):
        """Set agent name for memory isolation."""
        self._agent_name = agent_name

    @property
    def _agent_dir(self) -> Path:
        """Get agent-specific memory directory."""
        if self._agent_name:
            return self._memory_dir / self._agent_name
        return self._memory_dir / "default"

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="local_memory",
            name="Local Memory",
            version="1.0.0",
            category="memory",
            description="Zero-dependency persistent memory using local JSON files",
            actions=[
                SkillAction(
                    name="store",
                    description="Store a memory with optional category and tags",
                    parameters={
                        "content": {
                            "type": "string",
                            "required": True,
                            "description": "The memory content to store"
                        },
                        "category": {
                            "type": "string",
                            "required": False,
                            "description": "Category: learnings, outcomes, goals, journal, general (default: general)"
                        },
                        "tags": {
                            "type": "string",
                            "required": False,
                            "description": "Comma-separated tags for search"
                        },
                        "relevance": {
                            "type": "number",
                            "required": False,
                            "description": "Importance score 0-1 (default: 0.5)"
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="search",
                    description="Search memories by keyword across all categories",
                    parameters={
                        "query": {
                            "type": "string",
                            "required": True,
                            "description": "Search query (keywords)"
                        },
                        "category": {
                            "type": "string",
                            "required": False,
                            "description": "Limit search to specific category"
                        },
                        "limit": {
                            "type": "integer",
                            "required": False,
                            "description": "Max results (default: 10)"
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="recall_recent",
                    description="Get the most recent memories, optionally filtered by category",
                    parameters={
                        "category": {
                            "type": "string",
                            "required": False,
                            "description": "Category to filter by"
                        },
                        "limit": {
                            "type": "integer",
                            "required": False,
                            "description": "Number of memories (default: 10)"
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="record_outcome",
                    description="Record the outcome of an action (what worked or didn't)",
                    parameters={
                        "action": {
                            "type": "string",
                            "required": True,
                            "description": "What action was taken"
                        },
                        "outcome": {
                            "type": "string",
                            "required": True,
                            "description": "What happened (success/failure/partial)"
                        },
                        "details": {
                            "type": "string",
                            "required": False,
                            "description": "Additional details about what happened"
                        },
                        "lesson": {
                            "type": "string",
                            "required": False,
                            "description": "What was learned from this outcome"
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="journal_entry",
                    description="Write a session journal entry (what happened, reflections)",
                    parameters={
                        "entry": {
                            "type": "string",
                            "required": True,
                            "description": "Journal entry text"
                        },
                        "session_id": {
                            "type": "string",
                            "required": False,
                            "description": "Session identifier"
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="consolidate",
                    description="Consolidate old memories in a category (keep important, summarize rest)",
                    parameters={
                        "category": {
                            "type": "string",
                            "required": True,
                            "description": "Category to consolidate"
                        },
                        "keep_top": {
                            "type": "integer",
                            "required": False,
                            "description": "Number of top memories to keep intact (default: 50)"
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="stats",
                    description="Get memory usage statistics",
                    parameters={},
                    estimated_cost=0,
                ),
                SkillAction(
                    name="get_lessons",
                    description="Get all learned lessons (high-relevance learnings and outcomes)",
                    parameters={
                        "topic": {
                            "type": "string",
                            "required": False,
                            "description": "Filter lessons by topic keyword"
                        },
                        "limit": {
                            "type": "integer",
                            "required": False,
                            "description": "Max results (default: 20)"
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="delete",
                    description="Delete a specific memory by its ID",
                    parameters={
                        "memory_id": {
                            "type": "string",
                            "required": True,
                            "description": "ID of the memory to delete"
                        },
                        "category": {
                            "type": "string",
                            "required": False,
                            "description": "Category to search in (searches all if not specified)"
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="update_relevance",
                    description="Update the relevance score of a memory",
                    parameters={
                        "memory_id": {
                            "type": "string",
                            "required": True,
                            "description": "ID of the memory"
                        },
                        "relevance": {
                            "type": "number",
                            "required": True,
                            "description": "New relevance score (0-1)"
                        },
                    },
                    estimated_cost=0,
                ),
            ],
            required_credentials=[],  # No credentials needed!
        )

    def check_credentials(self) -> bool:
        """Always returns True - no credentials needed."""
        return True

    async def execute(self, action: str, params: Dict) -> SkillResult:
        handlers = {
            "store": self._store,
            "search": self._search,
            "recall_recent": self._recall_recent,
            "record_outcome": self._record_outcome,
            "journal_entry": self._journal_entry,
            "consolidate": self._consolidate,
            "stats": self._stats,
            "get_lessons": self._get_lessons,
            "delete": self._delete,
            "update_relevance": self._update_relevance,
        }

        handler = handlers.get(action)
        if handler:
            return await handler(params)
        return SkillResult(success=False, message=f"Unknown action: {action}")

    # === Storage Helpers ===

    def _ensure_dir(self):
        """Create memory directory if it doesn't exist."""
        self._agent_dir.mkdir(parents=True, exist_ok=True)

    def _category_path(self, category: str) -> Path:
        """Get file path for a category."""
        cat = category if category in self.CATEGORIES else "general"
        return self._agent_dir / f"{cat}.json"

    def _load_category(self, category: str) -> List[Dict]:
        """Load all memories from a category file."""
        # Check cache first
        if category in self._memory_cache:
            return self._memory_cache[category]

        path = self._category_path(category)
        if not path.exists():
            return []

        try:
            with open(path, "r") as f:
                data = json.load(f)
            self._memory_cache[category] = data
            return data
        except (json.JSONDecodeError, IOError):
            return []

    def _save_category(self, category: str, entries: List[Dict]):
        """Save memories to a category file."""
        self._ensure_dir()
        path = self._category_path(category)

        # Enforce max entries
        if len(entries) > self.MAX_ENTRIES_PER_CATEGORY:
            # Keep the most relevant and most recent
            entries.sort(key=lambda e: (
                e.get("relevance_score", 0.5),
                e.get("created_at", "")
            ), reverse=True)
            entries = entries[:self.MAX_ENTRIES_PER_CATEGORY]

        with open(path, "w") as f:
            json.dump(entries, f, indent=2)

        # Update cache
        self._memory_cache[category] = entries

    def _generate_id(self) -> str:
        """Generate a unique memory ID."""
        return f"mem_{int(time.time() * 1000)}_{os.getpid()}"

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text for search indexing."""
        # Simple keyword extraction: lowercase, split, remove short/common words
        stop_words = {
            "the", "a", "an", "is", "was", "were", "are", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "can", "shall",
            "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "as", "into", "through", "during", "before", "after", "above",
            "below", "between", "and", "but", "or", "nor", "not", "so",
            "yet", "both", "either", "neither", "each", "every", "all",
            "any", "few", "more", "most", "other", "some", "such", "no",
            "than", "too", "very", "just", "about", "also", "it", "its",
            "this", "that", "these", "those", "i", "me", "my", "we",
            "our", "you", "your", "he", "him", "his", "she", "her",
            "they", "them", "their", "what", "which", "who", "when",
            "where", "why", "how", "if", "then", "else",
        }
        words = re.findall(r'[a-z0-9]+', text.lower())
        return list(set(w for w in words if len(w) > 2 and w not in stop_words))

    def _score_match(self, entry: Dict, query_keywords: List[str]) -> float:
        """Score how well an entry matches a query."""
        if not query_keywords:
            return 0.0

        # Check content, tags, and auto-keywords
        entry_text = (
            entry.get("content", "") + " " +
            " ".join(entry.get("tags", [])) + " " +
            " ".join(entry.get("keywords", []))
        ).lower()

        matches = sum(1 for kw in query_keywords if kw in entry_text)
        keyword_score = matches / len(query_keywords)

        # Boost by relevance score
        relevance = entry.get("relevance_score", 0.5)

        # Boost by access frequency (popular memories are likely useful)
        access_boost = min(entry.get("access_count", 0) / 10, 0.2)

        return keyword_score * 0.7 + relevance * 0.2 + access_boost * 0.1

    # === Action Handlers ===

    async def _store(self, params: Dict) -> SkillResult:
        """Store a new memory."""
        content = params.get("content", "").strip()
        if not content:
            return SkillResult(success=False, message="Content is required")

        category = params.get("category", "general")
        if category not in self.CATEGORIES:
            category = "general"

        tags_str = params.get("tags", "")
        tags = [t.strip() for t in tags_str.split(",") if t.strip()] if tags_str else []
        relevance = max(0.0, min(1.0, float(params.get("relevance", 0.5))))

        # Auto-extract keywords
        keywords = self._extract_keywords(content)

        entry = {
            "id": self._generate_id(),
            "content": content,
            "category": category,
            "tags": tags,
            "keywords": keywords,
            "relevance_score": relevance,
            "access_count": 0,
            "last_accessed": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        entries = self._load_category(category)
        entries.append(entry)
        self._save_category(category, entries)

        return SkillResult(
            success=True,
            message=f"Stored memory in '{category}' ({len(content)} chars, {len(keywords)} keywords)",
            data={
                "id": entry["id"],
                "category": category,
                "tags": tags,
                "keyword_count": len(keywords),
                "total_in_category": len(entries),
            }
        )

    async def _search(self, params: Dict) -> SkillResult:
        """Search memories by keyword."""
        query = params.get("query", "").strip()
        if not query:
            return SkillResult(success=False, message="Query is required")

        category = params.get("category")
        limit = int(params.get("limit", 10))
        query_keywords = self._extract_keywords(query)

        # Also include raw query words for better matching
        raw_words = [w.lower() for w in query.split() if len(w) > 2]
        all_keywords = list(set(query_keywords + raw_words))

        # Search across categories
        categories_to_search = [category] if category and category in self.CATEGORIES else self.CATEGORIES
        scored_results = []

        for cat in categories_to_search:
            entries = self._load_category(cat)
            for entry in entries:
                score = self._score_match(entry, all_keywords)
                if score > 0.05:  # Minimum relevance threshold
                    scored_results.append((score, entry))

        # Sort by score descending
        scored_results.sort(key=lambda x: x[0], reverse=True)
        top_results = scored_results[:limit]

        # Update access counts for retrieved memories
        for score, entry in top_results:
            entry["access_count"] = entry.get("access_count", 0) + 1
            entry["last_accessed"] = datetime.now(timezone.utc).isoformat()

        # Save updated access counts
        for cat in categories_to_search:
            entries = self._load_category(cat)
            if entries:
                self._save_category(cat, entries)

        # Format results
        results = [
            {
                "id": entry["id"],
                "content": entry["content"][:500],
                "category": entry.get("category", "general"),
                "tags": entry.get("tags", []),
                "relevance": entry.get("relevance_score", 0.5),
                "match_score": round(score, 3),
                "created_at": entry.get("created_at", ""),
            }
            for score, entry in top_results
        ]

        return SkillResult(
            success=True,
            message=f"Found {len(results)} memories matching '{query[:50]}'",
            data={
                "query": query,
                "results": results,
                "count": len(results),
                "keywords_used": all_keywords[:20],
            }
        )

    async def _recall_recent(self, params: Dict) -> SkillResult:
        """Get the most recent memories."""
        category = params.get("category")
        limit = int(params.get("limit", 10))

        categories_to_search = [category] if category and category in self.CATEGORIES else self.CATEGORIES
        all_entries = []

        for cat in categories_to_search:
            entries = self._load_category(cat)
            all_entries.extend(entries)

        # Sort by creation time, most recent first
        all_entries.sort(key=lambda e: e.get("created_at", ""), reverse=True)
        recent = all_entries[:limit]

        results = [
            {
                "id": entry["id"],
                "content": entry["content"][:500],
                "category": entry.get("category", "general"),
                "tags": entry.get("tags", []),
                "relevance": entry.get("relevance_score", 0.5),
                "created_at": entry.get("created_at", ""),
            }
            for entry in recent
        ]

        return SkillResult(
            success=True,
            message=f"Retrieved {len(results)} recent memories",
            data={
                "results": results,
                "count": len(results),
            }
        )

    async def _record_outcome(self, params: Dict) -> SkillResult:
        """Record an action outcome for learning."""
        action_desc = params.get("action", "").strip()
        outcome = params.get("outcome", "").strip()
        if not action_desc or not outcome:
            return SkillResult(success=False, message="Both 'action' and 'outcome' are required")

        details = params.get("details", "")
        lesson = params.get("lesson", "")

        content = f"ACTION: {action_desc}\nOUTCOME: {outcome}"
        if details:
            content += f"\nDETAILS: {details}"
        if lesson:
            content += f"\nLESSON: {lesson}"

        # Set relevance based on outcome
        outcome_lower = outcome.lower()
        if "success" in outcome_lower:
            relevance = 0.6
        elif "fail" in outcome_lower:
            relevance = 0.8  # Failures are more informative
        else:
            relevance = 0.5

        # If there's a lesson, boost relevance
        if lesson:
            relevance = min(1.0, relevance + 0.1)

        # Store as outcome
        entry = {
            "id": self._generate_id(),
            "content": content,
            "category": "outcomes",
            "tags": ["outcome", outcome_lower.split()[0] if outcome_lower else "unknown"],
            "keywords": self._extract_keywords(content),
            "relevance_score": relevance,
            "access_count": 0,
            "last_accessed": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "outcome_type": outcome_lower.split()[0] if outcome_lower else "unknown",
        }

        entries = self._load_category("outcomes")
        entries.append(entry)
        self._save_category("outcomes", entries)

        return SkillResult(
            success=True,
            message=f"Recorded {outcome} outcome for '{action_desc[:50]}'",
            data={
                "id": entry["id"],
                "outcome_type": entry["outcome_type"],
                "relevance": relevance,
                "has_lesson": bool(lesson),
            }
        )

    async def _journal_entry(self, params: Dict) -> SkillResult:
        """Write a session journal entry."""
        entry_text = params.get("entry", "").strip()
        if not entry_text:
            return SkillResult(success=False, message="Entry text is required")

        session_id = params.get("session_id", datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))

        content = f"SESSION {session_id}: {entry_text}"

        entry = {
            "id": self._generate_id(),
            "content": content,
            "category": "journal",
            "tags": ["journal", f"session_{session_id}"],
            "keywords": self._extract_keywords(content),
            "relevance_score": 0.4,  # Journal entries are context, not lessons
            "access_count": 0,
            "last_accessed": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id,
        }

        entries = self._load_category("journal")
        entries.append(entry)
        self._save_category("journal", entries)

        return SkillResult(
            success=True,
            message=f"Journal entry saved for session {session_id}",
            data={
                "id": entry["id"],
                "session_id": session_id,
                "total_entries": len(entries),
            }
        )

    async def _consolidate(self, params: Dict) -> SkillResult:
        """Consolidate old memories - keep important ones, summarize the rest."""
        category = params.get("category", "").strip()
        if not category or category not in self.CATEGORIES:
            return SkillResult(
                success=False,
                message=f"Valid category required. Options: {', '.join(self.CATEGORIES)}"
            )

        keep_top = int(params.get("keep_top", 50))
        entries = self._load_category(category)

        if len(entries) <= keep_top:
            return SkillResult(
                success=True,
                message=f"Category '{category}' has {len(entries)} entries, no consolidation needed (threshold: {keep_top})",
                data={"entry_count": len(entries), "threshold": keep_top}
            )

        # Sort by combined score: relevance + recency + access
        now = datetime.now(timezone.utc)
        for entry in entries:
            created = entry.get("created_at", "")
            try:
                age_days = (now - datetime.fromisoformat(created.replace("Z", "+00:00"))).days
            except (ValueError, TypeError):
                age_days = 365

            recency_score = max(0, 1 - age_days / 365)  # Newer = higher
            access_score = min(entry.get("access_count", 0) / 10, 1.0)
            entry["_sort_score"] = (
                entry.get("relevance_score", 0.5) * 0.5 +
                recency_score * 0.3 +
                access_score * 0.2
            )

        entries.sort(key=lambda e: e.get("_sort_score", 0), reverse=True)

        # Keep top entries
        kept = entries[:keep_top]
        consolidated = entries[keep_top:]

        # Create a summary of consolidated entries
        summary_parts = []
        tag_counter = Counter()
        for entry in consolidated:
            # Extract first line or first 100 chars
            first_line = entry["content"].split("\n")[0][:100]
            summary_parts.append(f"- {first_line}")
            for tag in entry.get("tags", []):
                tag_counter[tag] += 1

        summary_content = (
            f"CONSOLIDATED SUMMARY ({len(consolidated)} entries from '{category}')\n"
            f"Date range: {consolidated[-1].get('created_at', '?')[:10]} to "
            f"{consolidated[0].get('created_at', '?')[:10]}\n"
            f"Common tags: {', '.join(t for t, _ in tag_counter.most_common(10))}\n"
            f"Entries:\n" + "\n".join(summary_parts[:100])
        )

        # Add summary as a new high-relevance entry
        summary_entry = {
            "id": self._generate_id(),
            "content": summary_content,
            "category": category,
            "tags": ["consolidation_summary"] + [t for t, _ in tag_counter.most_common(5)],
            "keywords": self._extract_keywords(summary_content),
            "relevance_score": 0.7,
            "access_count": 0,
            "last_accessed": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        # Clean up sort scores
        for entry in kept:
            entry.pop("_sort_score", None)

        kept.append(summary_entry)
        self._save_category(category, kept)

        return SkillResult(
            success=True,
            message=f"Consolidated '{category}': kept {keep_top} entries, summarized {len(consolidated)}",
            data={
                "kept": len(kept),
                "consolidated": len(consolidated),
                "summary_id": summary_entry["id"],
            }
        )

    async def _stats(self, params: Dict) -> SkillResult:
        """Get memory usage statistics."""
        stats = {}
        total_entries = 0
        total_size = 0

        for category in self.CATEGORIES:
            entries = self._load_category(category)
            path = self._category_path(category)
            file_size = path.stat().st_size if path.exists() else 0

            cat_stats = {
                "entry_count": len(entries),
                "file_size_bytes": file_size,
            }

            if entries:
                relevances = [e.get("relevance_score", 0.5) for e in entries]
                cat_stats["avg_relevance"] = round(sum(relevances) / len(relevances), 3)
                cat_stats["oldest"] = entries[0].get("created_at", "unknown") if entries else None
                cat_stats["newest"] = entries[-1].get("created_at", "unknown") if entries else None

                # Tag distribution
                all_tags = []
                for e in entries:
                    all_tags.extend(e.get("tags", []))
                tag_counter = Counter(all_tags)
                cat_stats["top_tags"] = [t for t, _ in tag_counter.most_common(5)]

            stats[category] = cat_stats
            total_entries += len(entries)
            total_size += file_size

        return SkillResult(
            success=True,
            message=f"Memory stats: {total_entries} total entries across {len(self.CATEGORIES)} categories",
            data={
                "total_entries": total_entries,
                "total_size_bytes": total_size,
                "memory_dir": str(self._agent_dir),
                "categories": stats,
            }
        )

    async def _get_lessons(self, params: Dict) -> SkillResult:
        """Get all learned lessons."""
        topic = params.get("topic", "").strip()
        limit = int(params.get("limit", 20))

        # Gather high-relevance entries from learnings and outcomes
        lessons = []

        for category in ["learnings", "outcomes"]:
            entries = self._load_category(category)
            for entry in entries:
                if entry.get("relevance_score", 0.5) >= 0.5:
                    lessons.append(entry)

        # Filter by topic if provided
        if topic:
            topic_keywords = self._extract_keywords(topic)
            scored_lessons = []
            for lesson in lessons:
                score = self._score_match(lesson, topic_keywords)
                if score > 0.05:
                    scored_lessons.append((score, lesson))
            scored_lessons.sort(key=lambda x: x[0], reverse=True)
            lessons = [lesson for _, lesson in scored_lessons[:limit]]
        else:
            # Sort by relevance
            lessons.sort(key=lambda e: e.get("relevance_score", 0.5), reverse=True)
            lessons = lessons[:limit]

        results = [
            {
                "id": entry["id"],
                "content": entry["content"][:500],
                "category": entry.get("category", ""),
                "relevance": entry.get("relevance_score", 0.5),
                "created_at": entry.get("created_at", ""),
            }
            for entry in lessons
        ]

        return SkillResult(
            success=True,
            message=f"Found {len(results)} lessons" + (f" about '{topic}'" if topic else ""),
            data={
                "results": results,
                "count": len(results),
                "topic": topic or "all",
            }
        )

    async def _delete(self, params: Dict) -> SkillResult:
        """Delete a specific memory by ID."""
        memory_id = params.get("memory_id", "").strip()
        if not memory_id:
            return SkillResult(success=False, message="memory_id is required")

        category = params.get("category")
        categories_to_search = [category] if category and category in self.CATEGORIES else self.CATEGORIES

        for cat in categories_to_search:
            entries = self._load_category(cat)
            original_count = len(entries)
            entries = [e for e in entries if e.get("id") != memory_id]
            if len(entries) < original_count:
                self._save_category(cat, entries)
                return SkillResult(
                    success=True,
                    message=f"Deleted memory {memory_id} from '{cat}'",
                    data={"deleted_id": memory_id, "category": cat}
                )

        return SkillResult(success=False, message=f"Memory {memory_id} not found")

    async def _update_relevance(self, params: Dict) -> SkillResult:
        """Update a memory's relevance score."""
        memory_id = params.get("memory_id", "").strip()
        if not memory_id:
            return SkillResult(success=False, message="memory_id is required")

        relevance = float(params.get("relevance", 0.5))
        relevance = max(0.0, min(1.0, relevance))

        for cat in self.CATEGORIES:
            entries = self._load_category(cat)
            for entry in entries:
                if entry.get("id") == memory_id:
                    old_relevance = entry.get("relevance_score", 0.5)
                    entry["relevance_score"] = relevance
                    self._save_category(cat, entries)
                    return SkillResult(
                        success=True,
                        message=f"Updated relevance: {old_relevance:.2f} -> {relevance:.2f}",
                        data={
                            "memory_id": memory_id,
                            "old_relevance": old_relevance,
                            "new_relevance": relevance,
                        }
                    )

        return SkillResult(success=False, message=f"Memory {memory_id} not found")
