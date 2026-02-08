#!/usr/bin/env python3
"""
CrossDatabaseJoinSkill - Query across multiple SQLite databases with ATTACH.

Bridges multiple databases into a single query context using SQLite's
ATTACH DATABASE feature, enabling:

  1. Attach: Mount additional databases into a query session
  2. Detach: Remove an attached database from the session
  3. Cross-Query: Execute SQL that references tables from multiple databases
  4. List Sessions: Show active cross-database sessions and their attachments
  5. Discover: Find joinable columns across databases (FK/name matching)
  6. Federated Query: High-level action that auto-attaches needed databases,
     runs a cross-database query, and returns results
  7. Stats: Query history, performance metrics, and usage statistics

SQLite supports ATTACH DATABASE to open multiple database files simultaneously.
Tables are referenced as `alias.table_name` in queries. This skill manages
the session lifecycle and provides discovery of cross-database relationships.

Revenue (primary): Customers with data spread across multiple databases can
  get unified analysis without manual data merging. This is a premium service.

Self-Improvement: The agent can query across its own state stores (goals,
  performance, experiments, feedback) in a single query for richer insights.

Pillar: Revenue (primary), Self-Improvement (supporting)
"""

import json
import re
import sqlite3
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .base import Skill, SkillAction, SkillManifest, SkillResult

DATA_DIR = Path(__file__).parent.parent / "data"
DB_REGISTRY_FILE = DATA_DIR / "database_registry.json"
CROSS_DB_STATE_FILE = DATA_DIR / "cross_database_join.json"

# Safety limits
MAX_ATTACHED = 10  # SQLite limit is typically 10 attached databases
MAX_RESULT_ROWS = 10000
MAX_QUERY_LENGTH = 10000
QUERY_TIMEOUT_SECONDS = 30
MAX_HISTORY = 200


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


class CrossDatabaseJoinSkill(Skill):
    """
    Query across multiple SQLite databases using ATTACH DATABASE.

    Creates managed sessions where multiple databases are mounted under
    aliases, enabling cross-database JOINs, UNIONs, and subqueries.
    All queries are read-only for safety.
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._sessions: Dict[str, Dict] = {}  # session_id -> session info
        self._connections: Dict[str, sqlite3.Connection] = {}  # session_id -> connection
        self._ensure_data()

    def _ensure_data(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        if not CROSS_DB_STATE_FILE.exists():
            self._save_state(self._default_state())

    def _default_state(self) -> Dict:
        return {
            "created_at": _now_iso(),
            "query_history": [],
            "stats": {
                "total_queries": 0,
                "total_sessions": 0,
                "total_rows_returned": 0,
                "databases_queried": {},
                "avg_query_time_ms": 0,
            },
        }

    def _load_state(self) -> Dict:
        try:
            with open(CROSS_DB_STATE_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return self._default_state()

    def _save_state(self, state: Dict):
        state["last_updated"] = _now_iso()
        if len(state.get("query_history", [])) > MAX_HISTORY:
            state["query_history"] = state["query_history"][-MAX_HISTORY:]
        with open(CROSS_DB_STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)

    def _load_registry(self) -> Dict:
        try:
            with open(DB_REGISTRY_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {}

    def _resolve_db_path(self, db_name: str) -> Optional[str]:
        """Resolve a database name to its file path."""
        registry = self._load_registry()
        if db_name in registry:
            return registry[db_name].get("path")
        if db_name == "default":
            return str(DATA_DIR / "agent_data.db")
        # Check if it's a direct path
        candidate = DATA_DIR / f"{db_name}.db"
        if candidate.exists():
            return str(candidate)
        return None

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="cross_database_join",
            name="Cross-Database Join",
            version="1.0.0",
            category="data",
            description="Query across multiple SQLite databases with ATTACH — cross-DB JOINs, UNIONs, discovery",
            required_credentials=[],
            install_cost=0,
            actions=[
                SkillAction(
                    name="create_session",
                    description="Create a cross-database query session and attach databases",
                    parameters={
                        "databases": {
                            "type": "object",
                            "required": True,
                            "description": "Map of alias -> database name, e.g. {'goals': 'goals_db', 'perf': 'performance_db'}",
                        },
                    },
                    estimated_cost=0,
                    estimated_duration_seconds=1,
                    success_probability=0.95,
                ),
                SkillAction(
                    name="attach",
                    description="Attach an additional database to an existing session",
                    parameters={
                        "session_id": {"type": "string", "required": True, "description": "Session ID"},
                        "alias": {"type": "string", "required": True, "description": "Alias for the database in queries"},
                        "db_name": {"type": "string", "required": True, "description": "Database name from registry"},
                    },
                    estimated_cost=0,
                    estimated_duration_seconds=1,
                    success_probability=0.90,
                ),
                SkillAction(
                    name="detach",
                    description="Detach a database from a session",
                    parameters={
                        "session_id": {"type": "string", "required": True, "description": "Session ID"},
                        "alias": {"type": "string", "required": True, "description": "Alias of the database to detach"},
                    },
                    estimated_cost=0,
                    estimated_duration_seconds=1,
                    success_probability=0.95,
                ),
                SkillAction(
                    name="query",
                    description="Execute a cross-database SQL query within a session",
                    parameters={
                        "session_id": {"type": "string", "required": True, "description": "Session ID"},
                        "sql": {"type": "string", "required": True, "description": "SQL query using alias.table_name references"},
                        "params": {"type": "array", "required": False, "description": "Query parameters for ? placeholders"},
                        "limit": {"type": "integer", "required": False, "description": "Max rows (default: 1000)"},
                    },
                    estimated_cost=0,
                    estimated_duration_seconds=5,
                    success_probability=0.85,
                ),
                SkillAction(
                    name="federated_query",
                    description="One-shot cross-database query: auto-attach databases, run query, return results",
                    parameters={
                        "databases": {
                            "type": "object",
                            "required": True,
                            "description": "Map of alias -> database name",
                        },
                        "sql": {"type": "string", "required": True, "description": "SQL query using alias.table_name references"},
                        "params": {"type": "array", "required": False, "description": "Query parameters"},
                        "limit": {"type": "integer", "required": False, "description": "Max rows (default: 1000)"},
                    },
                    estimated_cost=0,
                    estimated_duration_seconds=5,
                    success_probability=0.85,
                ),
                SkillAction(
                    name="discover",
                    description="Find joinable columns across databases by matching column names and types",
                    parameters={
                        "databases": {
                            "type": "array",
                            "required": True,
                            "description": "List of database names to analyze for join candidates",
                        },
                    },
                    estimated_cost=0,
                    estimated_duration_seconds=3,
                    success_probability=0.90,
                ),
                SkillAction(
                    name="list_sessions",
                    description="List active cross-database query sessions",
                    parameters={},
                    estimated_cost=0,
                    estimated_duration_seconds=1,
                    success_probability=1.0,
                ),
                SkillAction(
                    name="close_session",
                    description="Close a cross-database query session and release connections",
                    parameters={
                        "session_id": {"type": "string", "required": True, "description": "Session ID to close"},
                    },
                    estimated_cost=0,
                    estimated_duration_seconds=1,
                    success_probability=1.0,
                ),
                SkillAction(
                    name="stats",
                    description="View query history, performance metrics, and usage statistics",
                    parameters={
                        "last_n": {"type": "integer", "required": False, "description": "Number of recent queries to show (default: 10)"},
                    },
                    estimated_cost=0,
                    estimated_duration_seconds=1,
                    success_probability=1.0,
                ),
            ],
        )

    def check_credentials(self) -> bool:
        return True  # No credentials needed for SQLite

    async def execute(self, action: str, params: Dict[str, Any] = None) -> SkillResult:
        params = params or {}
        dispatch = {
            "create_session": self._create_session,
            "attach": self._attach,
            "detach": self._detach,
            "query": self._query,
            "federated_query": self._federated_query,
            "discover": self._discover,
            "list_sessions": self._list_sessions,
            "close_session": self._close_session,
            "stats": self._stats,
        }
        handler = dispatch.get(action)
        if not handler:
            return SkillResult(success=False, message=f"Unknown action: {action}")
        try:
            return await handler(params) if asyncio.iscoroutinefunction(handler) else handler(params)
        except Exception as e:
            return SkillResult(success=False, message=f"{action} failed: {str(e)}")

    def _create_session(self, params: Dict) -> SkillResult:
        """Create a new cross-database session with attached databases."""
        databases = params.get("databases")
        if not databases or not isinstance(databases, dict):
            return SkillResult(success=False, message="'databases' required: map of alias -> db_name")

        if len(databases) > MAX_ATTACHED:
            return SkillResult(success=False, message=f"Max {MAX_ATTACHED} databases per session")

        session_id = str(uuid.uuid4())[:8]

        # Create an in-memory connection as the main database
        conn = sqlite3.connect(":memory:", timeout=QUERY_TIMEOUT_SECONDS)
        conn.row_factory = sqlite3.Row

        attached = {}
        errors = []
        for alias, db_name in databases.items():
            # Validate alias (must be a valid identifier)
            if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', alias):
                errors.append(f"Invalid alias '{alias}': must be a valid identifier")
                continue

            db_path = self._resolve_db_path(db_name)
            if not db_path:
                errors.append(f"Database '{db_name}' not found in registry")
                continue

            try:
                conn.execute(f"ATTACH DATABASE ? AS [{alias}]", (db_path,))
                attached[alias] = {"db_name": db_name, "path": db_path}
            except sqlite3.Error as e:
                errors.append(f"Failed to attach '{db_name}' as '{alias}': {str(e)}")

        if not attached:
            conn.close()
            return SkillResult(
                success=False,
                message=f"No databases attached. Errors: {'; '.join(errors)}",
            )

        session_info = {
            "session_id": session_id,
            "created_at": _now_iso(),
            "attached": attached,
            "query_count": 0,
        }
        self._sessions[session_id] = session_info
        self._connections[session_id] = conn

        # Update stats
        state = self._load_state()
        state["stats"]["total_sessions"] += 1
        for info in attached.values():
            db_name = info["db_name"]
            state["stats"]["databases_queried"][db_name] = (
                state["stats"]["databases_queried"].get(db_name, 0) + 1
            )
        self._save_state(state)

        result = {
            "session_id": session_id,
            "attached_databases": {
                alias: info["db_name"] for alias, info in attached.items()
            },
            "usage_hint": "Reference tables as alias.table_name in your SQL queries",
        }
        if errors:
            result["warnings"] = errors

        return SkillResult(success=True, data=result)

    def _attach(self, params: Dict) -> SkillResult:
        """Attach an additional database to an existing session."""
        session_id = params.get("session_id")
        alias = params.get("alias")
        db_name = params.get("db_name")

        if not all([session_id, alias, db_name]):
            return SkillResult(success=False, message="session_id, alias, and db_name required")

        if session_id not in self._sessions:
            return SkillResult(success=False, message=f"Session '{session_id}' not found")

        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', alias):
            return SkillResult(success=False, message=f"Invalid alias '{alias}'")

        session = self._sessions[session_id]
        if len(session["attached"]) >= MAX_ATTACHED:
            return SkillResult(success=False, message=f"Max {MAX_ATTACHED} databases per session")

        if alias in session["attached"]:
            return SkillResult(success=False, message=f"Alias '{alias}' already in use")

        db_path = self._resolve_db_path(db_name)
        if not db_path:
            return SkillResult(success=False, message=f"Database '{db_name}' not found")

        conn = self._connections[session_id]
        try:
            conn.execute(f"ATTACH DATABASE ? AS [{alias}]", (db_path,))
        except sqlite3.Error as e:
            return SkillResult(success=False, message=f"ATTACH failed: {str(e)}")

        session["attached"][alias] = {"db_name": db_name, "path": db_path}
        return SkillResult(
            success=True,
            data={
                "attached": alias,
                "db_name": db_name,
                "total_attached": len(session["attached"]),
            },
        )

    def _detach(self, params: Dict) -> SkillResult:
        """Detach a database from a session."""
        session_id = params.get("session_id")
        alias = params.get("alias")

        if not all([session_id, alias]):
            return SkillResult(success=False, message="session_id and alias required")

        if session_id not in self._sessions:
            return SkillResult(success=False, message=f"Session '{session_id}' not found")

        session = self._sessions[session_id]
        if alias not in session["attached"]:
            return SkillResult(success=False, message=f"Alias '{alias}' not attached")

        conn = self._connections[session_id]
        try:
            conn.execute(f"DETACH DATABASE [{alias}]")
        except sqlite3.Error as e:
            return SkillResult(success=False, message=f"DETACH failed: {str(e)}")

        del session["attached"][alias]
        return SkillResult(
            success=True,
            data={"detached": alias, "remaining": len(session["attached"])},
        )

    def _is_safe_query(self, sql: str) -> Tuple[bool, str]:
        """Check if a SQL query is read-only and safe to execute."""
        stripped = sql.strip().upper()
        if len(sql) > MAX_QUERY_LENGTH:
            return False, f"Query exceeds max length ({MAX_QUERY_LENGTH} chars)"

        # Allow SELECT, WITH (CTE), EXPLAIN, PRAGMA
        first_word = stripped.split()[0] if stripped.split() else ""
        if first_word in ("SELECT", "WITH", "EXPLAIN", "PRAGMA"):
            # Extra check: make sure WITH ... doesn't contain DML
            write_kw = {"INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "REPLACE", "TRUNCATE"}
            # Crude check: look for DML keywords outside of string literals
            # Remove string literals first
            no_strings = re.sub(r"'[^']*'", "", stripped)
            no_strings = re.sub(r'"[^"]*"', "", no_strings)
            for kw in write_kw:
                # Match as whole word
                if re.search(rf'\b{kw}\b', no_strings):
                    return False, f"Write operation '{kw}' not allowed in cross-database queries"
            return True, ""

        return False, f"Only SELECT/WITH/EXPLAIN/PRAGMA allowed, got '{first_word}'"

    def _query(self, params: Dict) -> SkillResult:
        """Execute a cross-database SQL query."""
        session_id = params.get("session_id")
        sql = params.get("sql", "").strip()
        query_params = params.get("params", [])
        limit = min(params.get("limit", 1000), MAX_RESULT_ROWS)

        if not session_id or not sql:
            return SkillResult(success=False, message="session_id and sql required")

        if session_id not in self._sessions:
            return SkillResult(success=False, message=f"Session '{session_id}' not found")

        safe, reason = self._is_safe_query(sql)
        if not safe:
            return SkillResult(success=False, message=reason)

        conn = self._connections[session_id]
        session = self._sessions[session_id]

        start = time.time()
        try:
            cursor = conn.execute(sql, query_params or [])
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            rows = cursor.fetchmany(limit)
            row_count = len(rows)

            # Convert to list of dicts
            results = [dict(zip(columns, row)) for row in rows]

            elapsed_ms = round((time.time() - start) * 1000, 2)

            # Update session stats
            session["query_count"] += 1

            # Update global stats
            state = self._load_state()
            state["stats"]["total_queries"] += 1
            state["stats"]["total_rows_returned"] += row_count

            # Running average of query time
            total_q = state["stats"]["total_queries"]
            old_avg = state["stats"].get("avg_query_time_ms", 0)
            state["stats"]["avg_query_time_ms"] = round(
                old_avg + (elapsed_ms - old_avg) / total_q, 2
            )

            # Record in history
            state["query_history"].append({
                "session_id": session_id,
                "sql": sql[:500],  # Truncate long queries
                "databases": list(session["attached"].keys()),
                "row_count": row_count,
                "elapsed_ms": elapsed_ms,
                "timestamp": _now_iso(),
            })
            self._save_state(state)

            return SkillResult(
                success=True,
                data={
                    "columns": columns,
                    "rows": results,
                    "row_count": row_count,
                    "truncated": row_count >= limit,
                    "elapsed_ms": elapsed_ms,
                    "databases_used": list(session["attached"].keys()),
                },
            )

        except sqlite3.Error as e:
            elapsed_ms = round((time.time() - start) * 1000, 2)
            return SkillResult(
                success=False,
                message=f"Query failed ({elapsed_ms}ms): {str(e)}",
            )

    def _federated_query(self, params: Dict) -> SkillResult:
        """One-shot: auto-attach databases, run query, return results, cleanup."""
        databases = params.get("databases")
        sql = params.get("sql", "").strip()
        query_params = params.get("params", [])
        limit = min(params.get("limit", 1000), MAX_RESULT_ROWS)

        if not databases or not sql:
            return SkillResult(success=False, message="'databases' and 'sql' required")

        # Create ephemeral session
        create_result = self._create_session({"databases": databases})
        if not create_result.success:
            return create_result

        session_id = create_result.data["session_id"]

        try:
            # Run the query
            query_result = self._query({
                "session_id": session_id,
                "sql": sql,
                "params": query_params,
                "limit": limit,
            })

            # Add federated context to result
            if query_result.success:
                query_result.data["federated"] = True
                query_result.data["auto_attached"] = create_result.data["attached_databases"]

            return query_result
        finally:
            # Always cleanup
            self._close_session({"session_id": session_id})

    def _discover(self, params: Dict) -> SkillResult:
        """Discover joinable columns across databases by matching names and types."""
        db_names = params.get("databases", [])
        if not db_names or len(db_names) < 2:
            return SkillResult(success=False, message="At least 2 database names required")

        # Collect schema info from each database
        db_schemas: Dict[str, List[Dict]] = {}
        errors = []

        for db_name in db_names:
            db_path = self._resolve_db_path(db_name)
            if not db_path:
                errors.append(f"Database '{db_name}' not found")
                continue

            try:
                conn = sqlite3.connect(db_path, timeout=10)
                cursor = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                )
                tables = [row[0] for row in cursor.fetchall()]

                columns = []
                for table in tables:
                    cursor = conn.execute(f"PRAGMA table_info([{table}])")
                    for row in cursor.fetchall():
                        columns.append({
                            "table": table,
                            "column": row[1],
                            "type": row[2],
                            "pk": bool(row[5]),
                        })
                db_schemas[db_name] = columns
                conn.close()
            except sqlite3.Error as e:
                errors.append(f"Error reading '{db_name}': {str(e)}")

        if len(db_schemas) < 2:
            return SkillResult(
                success=False,
                message=f"Need schemas from at least 2 databases. Errors: {'; '.join(errors)}",
            )

        # Find matching columns across databases
        join_candidates = []
        db_list = list(db_schemas.keys())

        for i, db_a in enumerate(db_list):
            for db_b in db_list[i + 1:]:
                for col_a in db_schemas[db_a]:
                    for col_b in db_schemas[db_b]:
                        # Match by exact column name
                        if col_a["column"] == col_b["column"]:
                            # Compute match quality
                            score = 1.0
                            if col_a["type"].upper() == col_b["type"].upper():
                                score += 0.5
                            if col_a["pk"] or col_b["pk"]:
                                score += 0.5
                            # Common join column names get a boost
                            name = col_a["column"].lower()
                            if name in ("id", "user_id", "session_id", "timestamp",
                                        "created_at", "name", "type", "status"):
                                score += 0.3

                            join_candidates.append({
                                "db_a": db_a,
                                "table_a": col_a["table"],
                                "db_b": db_b,
                                "table_b": col_b["table"],
                                "column": col_a["column"],
                                "type_a": col_a["type"],
                                "type_b": col_b["type"],
                                "score": round(score, 1),
                                "suggested_sql": (
                                    f"SELECT * FROM {db_a}.{col_a['table']} a "
                                    f"JOIN {db_b}.{col_b['table']} b "
                                    f"ON a.{col_a['column']} = b.{col_b['column']} "
                                    f"LIMIT 10"
                                ),
                            })

        # Sort by score descending
        join_candidates.sort(key=lambda x: x["score"], reverse=True)

        return SkillResult(
            success=True,
            data={
                "databases_analyzed": list(db_schemas.keys()),
                "join_candidates": join_candidates[:50],  # Top 50
                "total_candidates": len(join_candidates),
                "schemas": {
                    db: {
                        "tables": list({c["table"] for c in cols}),
                        "column_count": len(cols),
                    }
                    for db, cols in db_schemas.items()
                },
            },
        )

    def _list_sessions(self, params: Dict) -> SkillResult:
        """List all active cross-database sessions."""
        sessions = []
        for sid, info in self._sessions.items():
            sessions.append({
                "session_id": sid,
                "created_at": info["created_at"],
                "attached_databases": {
                    alias: details["db_name"]
                    for alias, details in info["attached"].items()
                },
                "query_count": info["query_count"],
            })

        return SkillResult(
            success=True,
            data={
                "active_sessions": sessions,
                "count": len(sessions),
            },
        )

    def _close_session(self, params: Dict) -> SkillResult:
        """Close a cross-database session and release connections."""
        session_id = params.get("session_id")
        if not session_id:
            return SkillResult(success=False, message="session_id required")

        if session_id not in self._sessions:
            return SkillResult(success=False, message=f"Session '{session_id}' not found")

        # Close the connection
        conn = self._connections.pop(session_id, None)
        if conn:
            try:
                conn.close()
            except sqlite3.Error:
                pass

        session = self._sessions.pop(session_id)
        return SkillResult(
            success=True,
            data={
                "closed": session_id,
                "databases_released": list(session["attached"].keys()),
                "queries_executed": session["query_count"],
            },
        )

    def _stats(self, params: Dict) -> SkillResult:
        """View query history and statistics."""
        last_n = params.get("last_n", 10)
        state = self._load_state()

        return SkillResult(
            success=True,
            data={
                "stats": state["stats"],
                "recent_queries": state["query_history"][-last_n:],
                "active_sessions": len(self._sessions),
            },
        )


# Need asyncio for the execute method's iscoroutinefunction check
import asyncio
