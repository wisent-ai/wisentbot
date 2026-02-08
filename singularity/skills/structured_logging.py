#!/usr/bin/env python3
"""
StructuredLoggingSkill - Centralized structured logging and log aggregation for autonomous agents.

The Singularity framework has many skills that produce logs, events, and traces
independently, but no centralized logging infrastructure to:
- Collect structured log entries from all skill executions
- Correlate logs across multi-skill workflows via trace IDs
- Query and filter log history for debugging and forensics
- Aggregate log statistics for observability
- Set retention policies and manage log lifecycle

This skill fills that gap. It provides:

1. **Structured Log Ingestion** (Self-Improvement pillar)
   - Every skill can emit structured log entries with severity, tags, and context
   - Auto-generated trace IDs for correlating logs across skill chains
   - Support for log levels: DEBUG, INFO, WARN, ERROR, FATAL

2. **Log Querying** (Revenue/debugging)
   - Filter by level, skill, trace_id, tags, time range
   - Full-text search across log messages
   - Pagination for large result sets

3. **Log Aggregation & Statistics** (Observability)
   - Count logs by level, skill, time bucket
   - Error rate tracking per skill
   - Hot-spot detection (which skills log most errors)

4. **Trace Correlation** (All pillars)
   - Group logs by trace_id to reconstruct execution flows
   - Visualize multi-skill request paths
   - Identify bottlenecks and failure points

5. **Retention Management** (Infrastructure)
   - Configurable max log count
   - Auto-trim oldest entries when limit reached
   - Manual purge by age or filter

Architecture:
  LogEntry = {trace_id, level, skill_id, message, data, tags, timestamp}
  Stored in JSON file with bounded size (configurable, default 10000 entries)
  In-memory index for fast querying by level and skill_id

Part of the Self-Improvement pillar: enables debugging, forensics, and
execution analytics that multiple other skills depend on.
"""

import json
import time
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from .base import Skill, SkillAction, SkillManifest, SkillResult

LOG_FILE = Path(__file__).parent.parent / "data" / "structured_logs.json"
MAX_LOGS = 10000
MAX_TAGS_PER_ENTRY = 20
LOG_LEVELS = ["DEBUG", "INFO", "WARN", "ERROR", "FATAL"]
LOG_LEVEL_SEVERITY = {level: i for i, level in enumerate(LOG_LEVELS)}


class StructuredLoggingSkill(Skill):
    """
    Centralized structured logging with trace correlation and aggregation.

    Provides a unified logging layer for all skills, enabling cross-skill
    log correlation via trace IDs, structured querying, and aggregation
    statistics for observability and debugging.
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._ensure_data()

    def _ensure_data(self):
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        if not LOG_FILE.exists():
            self._save(self._default_state())

    def _default_state(self) -> Dict:
        return {
            "logs": [],
            "config": {
                "max_logs": MAX_LOGS,
                "min_level": "DEBUG",
            },
            "stats": {
                "total_ingested": 0,
                "total_trimmed": 0,
            },
        }

    def _load(self) -> Dict:
        try:
            with open(LOG_FILE, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return self._default_state()

    def _save(self, state: Dict):
        with open(LOG_FILE, "w") as f:
            json.dump(state, f, indent=2)

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="structured_logging",
            name="Structured Logging",
            version="1.0.0",
            category="infrastructure",
            description=(
                "Centralized structured logging with trace correlation, "
                "querying, and aggregation for all skill executions"
            ),
            actions=[
                SkillAction(
                    name="log",
                    description="Emit a structured log entry",
                    parameters={
                        "level": {
                            "type": "string",
                            "required": False,
                            "description": (
                                f"Log level: {', '.join(LOG_LEVELS)} (default: INFO)"
                            ),
                        },
                        "message": {
                            "type": "string",
                            "required": True,
                            "description": "Log message",
                        },
                        "skill_id": {
                            "type": "string",
                            "required": False,
                            "description": "ID of the skill emitting the log",
                        },
                        "trace_id": {
                            "type": "string",
                            "required": False,
                            "description": (
                                "Trace ID for correlating logs across "
                                "skill chains (auto-generated if omitted)"
                            ),
                        },
                        "data": {
                            "type": "object",
                            "required": False,
                            "description": "Arbitrary structured data to attach",
                        },
                        "tags": {
                            "type": "array",
                            "required": False,
                            "description": f"Tags for categorization (max {MAX_TAGS_PER_ENTRY})",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="log_batch",
                    description="Emit multiple structured log entries at once",
                    parameters={
                        "entries": {
                            "type": "array",
                            "required": True,
                            "description": (
                                "Array of log entry objects, each with: "
                                "level, message, skill_id, trace_id, data, tags"
                            ),
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="query",
                    description="Query logs with filters",
                    parameters={
                        "level": {
                            "type": "string",
                            "required": False,
                            "description": "Filter by minimum level (e.g., WARN shows WARN+ERROR+FATAL)",
                        },
                        "skill_id": {
                            "type": "string",
                            "required": False,
                            "description": "Filter by skill ID",
                        },
                        "trace_id": {
                            "type": "string",
                            "required": False,
                            "description": "Filter by trace ID",
                        },
                        "tag": {
                            "type": "string",
                            "required": False,
                            "description": "Filter by tag",
                        },
                        "search": {
                            "type": "string",
                            "required": False,
                            "description": "Full-text search in message field",
                        },
                        "since": {
                            "type": "number",
                            "required": False,
                            "description": "Only logs after this Unix timestamp",
                        },
                        "until": {
                            "type": "number",
                            "required": False,
                            "description": "Only logs before this Unix timestamp",
                        },
                        "limit": {
                            "type": "integer",
                            "required": False,
                            "description": "Max results to return (default: 50, max: 500)",
                        },
                        "offset": {
                            "type": "integer",
                            "required": False,
                            "description": "Skip first N results for pagination",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="trace",
                    description="Get all logs for a specific trace ID, ordered chronologically",
                    parameters={
                        "trace_id": {
                            "type": "string",
                            "required": True,
                            "description": "The trace ID to look up",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="stats",
                    description="Get log aggregation statistics",
                    parameters={
                        "group_by": {
                            "type": "string",
                            "required": False,
                            "description": (
                                "Group stats by: 'level', 'skill', 'hour', 'tag' "
                                "(default: 'level')"
                            ),
                        },
                        "since": {
                            "type": "number",
                            "required": False,
                            "description": "Only include logs after this Unix timestamp",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="error_report",
                    description="Generate an error report: top error-producing skills, common messages",
                    parameters={
                        "since": {
                            "type": "number",
                            "required": False,
                            "description": "Only include logs after this Unix timestamp",
                        },
                        "top_n": {
                            "type": "integer",
                            "required": False,
                            "description": "Number of top items to return (default: 10)",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="new_trace",
                    description="Generate a new trace ID for correlating a multi-skill workflow",
                    parameters={},
                    estimated_cost=0,
                ),
                SkillAction(
                    name="purge",
                    description="Delete logs matching criteria",
                    parameters={
                        "older_than": {
                            "type": "number",
                            "required": False,
                            "description": "Delete logs older than this Unix timestamp",
                        },
                        "level": {
                            "type": "string",
                            "required": False,
                            "description": "Delete only logs at this exact level",
                        },
                        "skill_id": {
                            "type": "string",
                            "required": False,
                            "description": "Delete only logs from this skill",
                        },
                        "confirm": {
                            "type": "boolean",
                            "required": True,
                            "description": "Must be true to actually delete",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="config",
                    description="View or update logging configuration",
                    parameters={
                        "max_logs": {
                            "type": "integer",
                            "required": False,
                            "description": f"Maximum log entries to retain (default: {MAX_LOGS})",
                        },
                        "min_level": {
                            "type": "string",
                            "required": False,
                            "description": "Minimum log level to ingest (default: DEBUG)",
                        },
                    },
                    estimated_cost=0,
                ),
            ],
            required_credentials=[],
            author="adam",
        )

    def check_credentials(self) -> bool:
        return True

    async def execute(self, action: str, params: Dict) -> SkillResult:
        handlers = {
            "log": self._log,
            "log_batch": self._log_batch,
            "query": self._query,
            "trace": self._trace,
            "stats": self._stats,
            "error_report": self._error_report,
            "new_trace": self._new_trace,
            "purge": self._purge,
            "config": self._config,
        }
        handler = handlers.get(action)
        if not handler:
            return SkillResult(success=False, message=f"Unknown action: {action}")
        return handler(params)

    # --- Log Ingestion ---

    def _make_entry(self, params: Dict) -> Optional[Dict]:
        """Validate and construct a log entry from params. Returns None on error."""
        level = (params.get("level") or "INFO").upper().strip()
        if level not in LOG_LEVELS:
            return None

        message = (params.get("message") or "").strip()
        if not message:
            return None

        skill_id = (params.get("skill_id") or "").strip()
        trace_id = (params.get("trace_id") or str(uuid.uuid4())[:12]).strip()

        data = params.get("data")
        if data is not None and not isinstance(data, dict):
            data = {"value": data}

        tags = params.get("tags", [])
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",") if t.strip()]
        tags = tags[:MAX_TAGS_PER_ENTRY]

        return {
            "id": str(uuid.uuid4())[:12],
            "level": level,
            "message": message,
            "skill_id": skill_id,
            "trace_id": trace_id,
            "data": data or {},
            "tags": tags,
            "timestamp": time.time(),
            "created_at": datetime.now().isoformat(),
        }

    def _should_ingest(self, level: str, min_level: str) -> bool:
        """Check if a log entry meets the minimum level threshold."""
        return LOG_LEVEL_SEVERITY.get(level, 0) >= LOG_LEVEL_SEVERITY.get(min_level, 0)

    def _trim_logs(self, state: Dict):
        """Trim log list to configured max size, removing oldest entries."""
        max_logs = state.get("config", {}).get("max_logs", MAX_LOGS)
        if len(state["logs"]) > max_logs:
            excess = len(state["logs"]) - max_logs
            state["logs"] = state["logs"][excess:]
            state["stats"]["total_trimmed"] += excess

    def _log(self, params: Dict) -> SkillResult:
        """Emit a single structured log entry."""
        state = self._load()
        min_level = state.get("config", {}).get("min_level", "DEBUG")

        level = (params.get("level") or "INFO").upper().strip()
        if level not in LOG_LEVELS:
            return SkillResult(
                success=False,
                message=f"Invalid level '{level}'. Must be one of: {', '.join(LOG_LEVELS)}",
            )

        message = (params.get("message") or "").strip()
        if not message:
            return SkillResult(success=False, message="message is required")

        if not self._should_ingest(level, min_level):
            return SkillResult(
                success=True,
                message=f"Log dropped: level {level} below minimum {min_level}",
                data={"ingested": False, "reason": "below_min_level"},
            )

        entry = self._make_entry(params)
        if entry is None:
            return SkillResult(success=False, message="Failed to construct log entry")

        state["logs"].append(entry)
        state["stats"]["total_ingested"] += 1
        self._trim_logs(state)
        self._save(state)

        return SkillResult(
            success=True,
            message=f"[{entry['level']}] {entry['message'][:80]}",
            data={
                "log_id": entry["id"],
                "trace_id": entry["trace_id"],
                "level": entry["level"],
                "ingested": True,
            },
        )

    def _log_batch(self, params: Dict) -> SkillResult:
        """Emit multiple log entries at once for efficiency."""
        entries_raw = params.get("entries", [])
        if not isinstance(entries_raw, list):
            return SkillResult(success=False, message="entries must be a list")
        if not entries_raw:
            return SkillResult(success=False, message="entries list is empty")
        if len(entries_raw) > 100:
            return SkillResult(
                success=False,
                message="Maximum 100 entries per batch",
            )

        state = self._load()
        min_level = state.get("config", {}).get("min_level", "DEBUG")

        ingested = 0
        dropped = 0
        invalid = 0
        trace_ids = set()

        for raw in entries_raw:
            if not isinstance(raw, dict):
                invalid += 1
                continue

            level = (raw.get("level") or "INFO").upper().strip()
            if not self._should_ingest(level, min_level):
                dropped += 1
                continue

            entry = self._make_entry(raw)
            if entry is None:
                invalid += 1
                continue

            state["logs"].append(entry)
            trace_ids.add(entry["trace_id"])
            ingested += 1

        state["stats"]["total_ingested"] += ingested
        self._trim_logs(state)
        self._save(state)

        return SkillResult(
            success=True,
            message=f"Batch: {ingested} ingested, {dropped} dropped, {invalid} invalid",
            data={
                "ingested": ingested,
                "dropped": dropped,
                "invalid": invalid,
                "trace_ids": list(trace_ids),
            },
        )

    # --- Querying ---

    def _query(self, params: Dict) -> SkillResult:
        """Query logs with filters and pagination."""
        state = self._load()
        logs = state["logs"]

        # Apply filters
        level = (params.get("level") or "").upper().strip()
        if level and level in LOG_LEVELS:
            min_sev = LOG_LEVEL_SEVERITY[level]
            logs = [lg for lg in logs if LOG_LEVEL_SEVERITY.get(lg["level"], 0) >= min_sev]

        skill_id = (params.get("skill_id") or "").strip()
        if skill_id:
            logs = [lg for lg in logs if lg.get("skill_id") == skill_id]

        trace_id = (params.get("trace_id") or "").strip()
        if trace_id:
            logs = [lg for lg in logs if lg.get("trace_id") == trace_id]

        tag = (params.get("tag") or "").strip()
        if tag:
            logs = [lg for lg in logs if tag in lg.get("tags", [])]

        search = (params.get("search") or "").strip().lower()
        if search:
            logs = [lg for lg in logs if search in lg.get("message", "").lower()]

        since = params.get("since")
        if since is not None:
            try:
                since = float(since)
                logs = [lg for lg in logs if lg.get("timestamp", 0) >= since]
            except (ValueError, TypeError):
                pass

        until = params.get("until")
        if until is not None:
            try:
                until = float(until)
                logs = [lg for lg in logs if lg.get("timestamp", 0) <= until]
            except (ValueError, TypeError):
                pass

        total_matching = len(logs)

        # Pagination
        offset = max(0, int(params.get("offset", 0)))
        limit = max(1, min(500, int(params.get("limit", 50))))

        # Return most recent first
        logs = logs[::-1]
        page = logs[offset: offset + limit]

        # Format for readability
        results = []
        for lg in page:
            entry = {
                "id": lg["id"],
                "level": lg["level"],
                "message": lg["message"],
                "skill_id": lg.get("skill_id", ""),
                "trace_id": lg.get("trace_id", ""),
                "tags": lg.get("tags", []),
                "created_at": lg.get("created_at", ""),
            }
            if lg.get("data"):
                entry["data"] = lg["data"]
            results.append(entry)

        return SkillResult(
            success=True,
            message=f"Found {total_matching} logs, showing {len(results)} (offset {offset})",
            data={
                "total_matching": total_matching,
                "offset": offset,
                "limit": limit,
                "logs": results,
            },
        )

    def _trace(self, params: Dict) -> SkillResult:
        """Get all logs for a trace ID, ordered chronologically."""
        trace_id = (params.get("trace_id") or "").strip()
        if not trace_id:
            return SkillResult(success=False, message="trace_id is required")

        state = self._load()
        logs = [lg for lg in state["logs"] if lg.get("trace_id") == trace_id]

        # Chronological order for trace view
        logs.sort(key=lambda lg: lg.get("timestamp", 0))

        # Compute trace metadata
        skills_involved = list({lg.get("skill_id", "") for lg in logs if lg.get("skill_id")})
        levels_seen = list({lg["level"] for lg in logs})
        has_errors = any(
            LOG_LEVEL_SEVERITY.get(lg["level"], 0) >= LOG_LEVEL_SEVERITY["ERROR"]
            for lg in logs
        )

        if logs:
            duration = logs[-1].get("timestamp", 0) - logs[0].get("timestamp", 0)
        else:
            duration = 0

        entries = []
        for lg in logs:
            entry = {
                "id": lg["id"],
                "level": lg["level"],
                "message": lg["message"],
                "skill_id": lg.get("skill_id", ""),
                "created_at": lg.get("created_at", ""),
            }
            if lg.get("data"):
                entry["data"] = lg["data"]
            entries.append(entry)

        return SkillResult(
            success=True,
            message=f"Trace {trace_id}: {len(logs)} entries across {len(skills_involved)} skills",
            data={
                "trace_id": trace_id,
                "entry_count": len(logs),
                "skills_involved": skills_involved,
                "levels_seen": levels_seen,
                "has_errors": has_errors,
                "duration_seconds": round(duration, 3),
                "entries": entries,
            },
        )

    # --- Aggregation ---

    def _stats(self, params: Dict) -> SkillResult:
        """Compute aggregation statistics over logs."""
        state = self._load()
        logs = state["logs"]

        since = params.get("since")
        if since is not None:
            try:
                since = float(since)
                logs = [lg for lg in logs if lg.get("timestamp", 0) >= since]
            except (ValueError, TypeError):
                pass

        group_by = (params.get("group_by") or "level").strip().lower()

        groups = defaultdict(int)

        if group_by == "level":
            for lg in logs:
                groups[lg["level"]] += 1
        elif group_by == "skill":
            for lg in logs:
                sid = lg.get("skill_id") or "(none)"
                groups[sid] += 1
        elif group_by == "hour":
            for lg in logs:
                ts = lg.get("timestamp", 0)
                hour = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:00") if ts else "unknown"
                groups[hour] += 1
        elif group_by == "tag":
            for lg in logs:
                for t in lg.get("tags", []):
                    groups[t] += 1
            if not any(lg.get("tags") for lg in logs):
                groups["(no tags)"] = len(logs)
        else:
            return SkillResult(
                success=False,
                message=f"Invalid group_by: '{group_by}'. Use: level, skill, hour, tag",
            )

        # Sort by count descending
        sorted_groups = sorted(groups.items(), key=lambda x: x[1], reverse=True)

        # Compute totals
        total = len(logs)
        error_count = sum(
            1 for lg in logs
            if LOG_LEVEL_SEVERITY.get(lg["level"], 0) >= LOG_LEVEL_SEVERITY["ERROR"]
        )
        error_rate = error_count / total if total > 0 else 0.0

        return SkillResult(
            success=True,
            message=f"Stats ({group_by}): {total} logs, error rate {error_rate:.1%}",
            data={
                "total_logs": total,
                "error_count": error_count,
                "error_rate": round(error_rate, 4),
                "group_by": group_by,
                "groups": dict(sorted_groups),
                "lifetime_ingested": state["stats"]["total_ingested"],
                "lifetime_trimmed": state["stats"]["total_trimmed"],
            },
        )

    def _error_report(self, params: Dict) -> SkillResult:
        """Generate an error report: top error sources and common messages."""
        state = self._load()
        logs = state["logs"]

        since = params.get("since")
        if since is not None:
            try:
                since = float(since)
                logs = [lg for lg in logs if lg.get("timestamp", 0) >= since]
            except (ValueError, TypeError):
                pass

        top_n = max(1, min(50, int(params.get("top_n", 10))))

        # Filter to errors and above
        error_logs = [
            lg for lg in logs
            if LOG_LEVEL_SEVERITY.get(lg["level"], 0) >= LOG_LEVEL_SEVERITY["ERROR"]
        ]

        if not error_logs:
            return SkillResult(
                success=True,
                message="No errors found",
                data={
                    "error_count": 0,
                    "total_logs": len(logs),
                    "top_skills": [],
                    "top_messages": [],
                    "top_traces": [],
                },
            )

        # Top error-producing skills
        skill_errors = defaultdict(int)
        for lg in error_logs:
            sid = lg.get("skill_id") or "(none)"
            skill_errors[sid] += 1
        top_skills = sorted(skill_errors.items(), key=lambda x: x[1], reverse=True)[:top_n]

        # Top error messages (group by first 80 chars for dedup)
        msg_counts = defaultdict(int)
        for lg in error_logs:
            key = lg["message"][:80]
            msg_counts[key] += 1
        top_messages = sorted(msg_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]

        # Top traces with errors
        trace_errors = defaultdict(int)
        for lg in error_logs:
            tid = lg.get("trace_id", "")
            if tid:
                trace_errors[tid] += 1
        top_traces = sorted(trace_errors.items(), key=lambda x: x[1], reverse=True)[:top_n]

        # Error rate over time (hourly buckets)
        hourly_errors = defaultdict(lambda: {"errors": 0, "total": 0})
        for lg in logs:
            ts = lg.get("timestamp", 0)
            hour = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:00") if ts else "unknown"
            hourly_errors[hour]["total"] += 1
            if LOG_LEVEL_SEVERITY.get(lg["level"], 0) >= LOG_LEVEL_SEVERITY["ERROR"]:
                hourly_errors[hour]["errors"] += 1

        hourly_rates = {}
        for hour, counts in sorted(hourly_errors.items()):
            if counts["total"] > 0:
                hourly_rates[hour] = round(counts["errors"] / counts["total"], 4)

        return SkillResult(
            success=True,
            message=f"Error report: {len(error_logs)} errors from {len(skill_errors)} skills",
            data={
                "error_count": len(error_logs),
                "total_logs": len(logs),
                "error_rate": round(len(error_logs) / len(logs), 4) if logs else 0,
                "top_skills": [{"skill": s, "count": c} for s, c in top_skills],
                "top_messages": [{"message": m, "count": c} for m, c in top_messages],
                "top_traces": [{"trace_id": t, "errors": c} for t, c in top_traces],
                "hourly_error_rates": hourly_rates,
            },
        )

    # --- Utilities ---

    def _new_trace(self, params: Dict) -> SkillResult:
        """Generate a new trace ID."""
        trace_id = str(uuid.uuid4())[:12]
        return SkillResult(
            success=True,
            message=f"New trace ID: {trace_id}",
            data={"trace_id": trace_id},
        )

    def _purge(self, params: Dict) -> SkillResult:
        """Delete logs matching criteria."""
        confirm = params.get("confirm")
        if isinstance(confirm, str):
            confirm = confirm.lower() in ("true", "1", "yes")
        if not confirm:
            return SkillResult(
                success=False,
                message="Set confirm=true to actually delete logs",
            )

        state = self._load()
        original_count = len(state["logs"])
        logs = state["logs"]

        older_than = params.get("older_than")
        level = (params.get("level") or "").upper().strip()
        skill_id = (params.get("skill_id") or "").strip()

        if not older_than and not level and not skill_id:
            return SkillResult(
                success=False,
                message="At least one filter (older_than, level, skill_id) is required",
            )

        def should_keep(lg: Dict) -> bool:
            if older_than is not None:
                try:
                    if lg.get("timestamp", 0) < float(older_than):
                        return False
                except (ValueError, TypeError):
                    pass

            if level and lg.get("level") == level:
                return False

            if skill_id and lg.get("skill_id") == skill_id:
                return False

            return True

        # When multiple filters: remove entries matching ANY filter
        if older_than and (level or skill_id):
            # More precise: remove entries matching ALL specified filters
            def should_remove(lg: Dict) -> bool:
                matches = True
                if older_than is not None:
                    try:
                        if lg.get("timestamp", 0) >= float(older_than):
                            matches = False
                    except (ValueError, TypeError):
                        matches = False
                if level and lg.get("level") != level:
                    matches = False
                if skill_id and lg.get("skill_id") != skill_id:
                    matches = False
                return matches

            state["logs"] = [lg for lg in logs if not should_remove(lg)]
        else:
            state["logs"] = [lg for lg in logs if should_keep(lg)]

        deleted = original_count - len(state["logs"])
        state["stats"]["total_trimmed"] += deleted
        self._save(state)

        return SkillResult(
            success=True,
            message=f"Purged {deleted} logs (remaining: {len(state['logs'])})",
            data={
                "deleted": deleted,
                "remaining": len(state["logs"]),
            },
        )

    def _config(self, params: Dict) -> SkillResult:
        """View or update logging configuration."""
        state = self._load()
        config = state.get("config", {})
        changed = False

        max_logs = params.get("max_logs")
        if max_logs is not None:
            try:
                max_logs = int(max_logs)
                max_logs = max(100, min(100000, max_logs))
                config["max_logs"] = max_logs
                changed = True
            except (ValueError, TypeError):
                return SkillResult(
                    success=False,
                    message="max_logs must be an integer (100-100000)",
                )

        min_level = params.get("min_level")
        if min_level is not None:
            min_level = min_level.upper().strip()
            if min_level not in LOG_LEVELS:
                return SkillResult(
                    success=False,
                    message=f"Invalid min_level. Must be one of: {', '.join(LOG_LEVELS)}",
                )
            config["min_level"] = min_level
            changed = True

        state["config"] = config

        if changed:
            self._trim_logs(state)
            self._save(state)
            msg = "Configuration updated"
        else:
            msg = "Current configuration"

        return SkillResult(
            success=True,
            message=msg,
            data={
                "config": config,
                "current_log_count": len(state["logs"]),
                "lifetime_ingested": state["stats"]["total_ingested"],
                "lifetime_trimmed": state["stats"]["total_trimmed"],
            },
        )
