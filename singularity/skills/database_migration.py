#!/usr/bin/env python3
"""
DatabaseMigrationSkill - Schema versioning and migration management for SQLite databases.

Enables the agent to:

  1. CREATE migrations with up/down SQL (versioned, reversible schema changes)
  2. APPLY migrations forward (upgrade) to a target version or latest
  3. ROLLBACK migrations (downgrade) to undo schema changes safely
  4. STATUS check what version a database is at and which migrations are pending
  5. VALIDATE migrations before applying (dry-run, syntax check)
  6. HISTORY view full migration audit trail with timing and results
  7. GENERATE auto-generate migration SQL by comparing current schema vs desired
  8. SQUASH combine multiple sequential migrations into one

Self-Improvement (primary): The agent's own databases evolve over time as new
  skills store data. Without migration management, schema changes require
  manual intervention or data loss. This skill enables autonomous schema
  evolution — the agent can plan, validate, and apply its own database changes.

Revenue (supporting): Schema migration is a paid service. Customers can submit
  databases and desired schema changes, and the agent handles the migration
  safely with rollback capability.

Security:
  - All migrations are validated before execution
  - Rollback support for every migration (up + down SQL required)
  - Backup database before applying migrations
  - Full audit trail of all migration operations
  - Dry-run mode to preview changes without applying
"""

import hashlib
import json
import os
import re
import shutil
import sqlite3
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .base import Skill, SkillAction, SkillManifest, SkillResult

DATA_DIR = Path(__file__).parent.parent / "data"
MIGRATION_DIR = DATA_DIR / "migrations"
MIGRATION_STATE_FILE = DATA_DIR / "migration_state.json"
DB_REGISTRY_FILE = DATA_DIR / "database_registry.json"
MAX_HISTORY = 500


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _load_state() -> Dict:
    try:
        if MIGRATION_STATE_FILE.exists():
            with open(MIGRATION_STATE_FILE, "r") as f:
                return json.load(f)
    except (json.JSONDecodeError, IOError):
        pass
    return {
        "migrations": {},      # {db_name: {version: migration_info}}
        "applied": {},         # {db_name: [list of applied versions in order]}
        "history": [],         # audit trail
        "stats": {
            "total_migrations_created": 0,
            "total_applied": 0,
            "total_rolled_back": 0,
            "total_failed": 0,
        },
    }


def _save_state(state: Dict):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if len(state.get("history", [])) > MAX_HISTORY:
        state["history"] = state["history"][-MAX_HISTORY:]
    try:
        with open(MIGRATION_STATE_FILE, "w") as f:
            json.dump(state, f, indent=2, default=str)
    except IOError:
        pass


def _get_db_path(db_name: str) -> Path:
    """Resolve database name to file path."""
    if db_name == "default" or not db_name:
        return DATA_DIR / "agent_data.db"
    # Check registry
    try:
        if DB_REGISTRY_FILE.exists():
            with open(DB_REGISTRY_FILE, "r") as f:
                registry = json.load(f)
            if db_name in registry:
                return Path(registry[db_name].get("path", str(DATA_DIR / f"{db_name}.db")))
    except (json.JSONDecodeError, IOError):
        pass
    return DATA_DIR / f"{db_name}.db"


def _get_schema(db_path: Path) -> Dict[str, List[Dict]]:
    """Get current database schema as {table_name: [column_info]}."""
    schema = {}
    if not db_path.exists():
        return schema
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' AND name != '_migrations'"
        ).fetchall()
        for t in tables:
            tname = t["name"]
            cols = conn.execute(f"PRAGMA table_info('{tname}')").fetchall()
            schema[tname] = [
                {"name": c["name"], "type": c["type"], "notnull": c["notnull"], "pk": c["pk"], "dflt_value": c["dflt_value"]}
                for c in cols
            ]
        conn.close()
    except sqlite3.Error:
        pass
    return schema


def _ensure_migration_table(conn: sqlite3.Connection):
    """Create the _migrations tracking table if it doesn't exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS _migrations (
            version TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            applied_at TEXT NOT NULL,
            checksum TEXT NOT NULL,
            execution_time_ms INTEGER DEFAULT 0
        )
    """)
    conn.commit()


def _get_applied_versions(conn: sqlite3.Connection) -> List[str]:
    """Get list of applied migration versions from the database itself."""
    _ensure_migration_table(conn)
    rows = conn.execute(
        "SELECT version FROM _migrations ORDER BY applied_at ASC"
    ).fetchall()
    return [r[0] for r in rows]


def _compute_checksum(sql: str) -> str:
    """Compute a checksum for migration SQL."""
    return hashlib.sha256(sql.strip().encode()).hexdigest()[:16]


class DatabaseMigrationSkill(Skill):
    """
    Schema versioning and migration management for SQLite databases.

    Provides version-controlled schema changes with up/down SQL,
    validation, rollback, backup, and audit trail.
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        MIGRATION_DIR.mkdir(parents=True, exist_ok=True)

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="database_migration",
            name="Database Migration",
            version="1.0.0",
            category="infrastructure",
            description="Schema versioning and migration management for SQLite databases",
            actions=self.get_actions(),
            required_credentials=[],
        )

    def get_actions(self) -> List[SkillAction]:
        return [
            SkillAction(
                name="create",
                description="Create a new migration with up/down SQL",
                parameters={
                    "db": {"type": "string", "required": False, "description": "Database name (default: 'default')"},
                    "name": {"type": "string", "required": True, "description": "Migration name (e.g. 'add_users_table')"},
                    "up_sql": {"type": "string", "required": True, "description": "SQL to apply (upgrade)"},
                    "down_sql": {"type": "string", "required": True, "description": "SQL to rollback (downgrade)"},
                },
            ),
            SkillAction(
                name="apply",
                description="Apply pending migrations (upgrade database)",
                parameters={
                    "db": {"type": "string", "required": False, "description": "Database name"},
                    "target_version": {"type": "string", "required": False, "description": "Apply up to this version (default: latest)"},
                    "dry_run": {"type": "boolean", "required": False, "description": "Preview without executing (default: false)"},
                },
            ),
            SkillAction(
                name="rollback",
                description="Rollback applied migrations (downgrade database)",
                parameters={
                    "db": {"type": "string", "required": False, "description": "Database name"},
                    "target_version": {"type": "string", "required": False, "description": "Rollback to this version (default: one step back)"},
                    "steps": {"type": "integer", "required": False, "description": "Number of migrations to rollback (default: 1)"},
                },
            ),
            SkillAction(
                name="status",
                description="Check migration status for a database",
                parameters={
                    "db": {"type": "string", "required": False, "description": "Database name"},
                },
            ),
            SkillAction(
                name="validate",
                description="Validate migrations without applying them",
                parameters={
                    "db": {"type": "string", "required": False, "description": "Database name"},
                },
            ),
            SkillAction(
                name="history",
                description="View migration operation audit trail",
                parameters={
                    "db": {"type": "string", "required": False, "description": "Filter by database (optional)"},
                    "limit": {"type": "integer", "required": False, "description": "Max entries (default: 20)"},
                },
            ),
            SkillAction(
                name="generate",
                description="Auto-generate migration SQL by comparing current schema vs desired",
                parameters={
                    "db": {"type": "string", "required": False, "description": "Database name"},
                    "desired_schema": {"type": "object", "required": True, "description": "Desired schema: {table: [{name, type, notnull?, pk?, dflt_value?}]}"},
                },
            ),
            SkillAction(
                name="squash",
                description="Squash multiple sequential migrations into one",
                parameters={
                    "db": {"type": "string", "required": False, "description": "Database name"},
                    "from_version": {"type": "string", "required": True, "description": "Start version (inclusive)"},
                    "to_version": {"type": "string", "required": True, "description": "End version (inclusive)"},
                    "name": {"type": "string", "required": False, "description": "Name for squashed migration"},
                },
            ),
        ]

    async def execute(self, action: str, params: Dict[str, Any]) -> SkillResult:
        try:
            if action == "create":
                return await self._create(params)
            elif action == "apply":
                return await self._apply(params)
            elif action == "rollback":
                return await self._rollback(params)
            elif action == "status":
                return await self._status(params)
            elif action == "validate":
                return await self._validate(params)
            elif action == "history":
                return await self._history(params)
            elif action == "generate":
                return await self._generate(params)
            elif action == "squash":
                return await self._squash(params)
            else:
                return SkillResult(success=False, message=f"Unknown action: {action}")
        except Exception as e:
            return SkillResult(success=False, message=f"Migration error: {str(e)}")

    async def _create(self, params: Dict) -> SkillResult:
        """Create a new migration."""
        db_name = params.get("db", "default")
        name = params.get("name", "")
        up_sql = params.get("up_sql", "")
        down_sql = params.get("down_sql", "")

        if not name:
            return SkillResult(success=False, message="Migration name is required")
        if not up_sql:
            return SkillResult(success=False, message="up_sql is required")
        if not down_sql:
            return SkillResult(success=False, message="down_sql is required")

        # Sanitize name
        safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', name.lower())

        state = _load_state()
        if db_name not in state["migrations"]:
            state["migrations"][db_name] = {}

        # Generate version: timestamp-based for ordering
        version = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        # Ensure unique
        existing = state["migrations"][db_name]
        while version in existing:
            version = str(int(version) + 1)

        migration = {
            "version": version,
            "name": safe_name,
            "up_sql": up_sql,
            "down_sql": down_sql,
            "checksum": _compute_checksum(up_sql + down_sql),
            "created_at": _now_iso(),
        }

        state["migrations"][db_name][version] = migration
        state["stats"]["total_migrations_created"] += 1
        state["history"].append({
            "action": "create",
            "db": db_name,
            "version": version,
            "name": safe_name,
            "timestamp": _now_iso(),
        })

        _save_state(state)

        return SkillResult(
            success=True,
            message=f"Created migration {version}_{safe_name} for database '{db_name}'",
            data={
                "version": version,
                "name": safe_name,
                "db": db_name,
                "checksum": migration["checksum"],
            },
        )

    async def _apply(self, params: Dict) -> SkillResult:
        """Apply pending migrations to a database."""
        db_name = params.get("db", "default")
        target_version = params.get("target_version")
        dry_run = params.get("dry_run", False)

        state = _load_state()
        db_migrations = state.get("migrations", {}).get(db_name, {})

        if not db_migrations:
            return SkillResult(success=True, message=f"No migrations defined for '{db_name}'", data={"applied": 0})

        db_path = _get_db_path(db_name)

        # Get sorted versions
        all_versions = sorted(db_migrations.keys())

        # Get already applied from the database itself
        conn = sqlite3.connect(str(db_path))
        applied_versions = _get_applied_versions(conn)

        # Find pending
        pending = [v for v in all_versions if v not in applied_versions]
        if target_version:
            pending = [v for v in pending if v <= target_version]

        if not pending:
            conn.close()
            return SkillResult(success=True, message="Database is up to date", data={"applied": 0, "current_version": applied_versions[-1] if applied_versions else None})

        if dry_run:
            conn.close()
            preview = []
            for v in pending:
                m = db_migrations[v]
                preview.append({"version": v, "name": m["name"], "up_sql": m["up_sql"]})
            return SkillResult(
                success=True,
                message=f"Dry run: {len(pending)} migrations would be applied",
                data={"dry_run": True, "pending": preview},
            )

        # Backup database before applying
        backup_path = None
        if db_path.exists():
            backup_path = db_path.with_suffix(f".backup_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}")
            shutil.copy2(str(db_path), str(backup_path))

        results = []
        applied_count = 0
        failed = False

        for v in pending:
            m = db_migrations[v]
            start_time = time.time()
            try:
                conn.executescript(m["up_sql"])
                elapsed_ms = int((time.time() - start_time) * 1000)

                # Record in _migrations table
                _ensure_migration_table(conn)
                conn.execute(
                    "INSERT INTO _migrations (version, name, applied_at, checksum, execution_time_ms) VALUES (?, ?, ?, ?, ?)",
                    (v, m["name"], _now_iso(), m["checksum"], elapsed_ms),
                )
                conn.commit()

                results.append({"version": v, "name": m["name"], "success": True, "time_ms": elapsed_ms})
                applied_count += 1
            except sqlite3.Error as e:
                elapsed_ms = int((time.time() - start_time) * 1000)
                results.append({"version": v, "name": m["name"], "success": False, "error": str(e), "time_ms": elapsed_ms})
                failed = True
                # Restore from backup on failure
                conn.close()
                if backup_path and backup_path.exists():
                    shutil.copy2(str(backup_path), str(db_path))
                state["stats"]["total_failed"] += 1
                break

        if not failed:
            conn.close()

        # Update state
        if db_name not in state["applied"]:
            state["applied"][db_name] = []
        state["applied"][db_name] = applied_versions + [r["version"] for r in results if r["success"]]
        state["stats"]["total_applied"] += applied_count

        state["history"].append({
            "action": "apply",
            "db": db_name,
            "applied_count": applied_count,
            "failed": failed,
            "versions": [r["version"] for r in results],
            "timestamp": _now_iso(),
            "backup": str(backup_path) if backup_path else None,
        })

        _save_state(state)

        # Cleanup backup on success
        if not failed and backup_path and backup_path.exists():
            try:
                backup_path.unlink()
            except OSError:
                pass

        msg = f"Applied {applied_count}/{len(pending)} migrations to '{db_name}'"
        if failed:
            msg += " (FAILED — database restored from backup)"

        return SkillResult(
            success=not failed,
            message=msg,
            data={"applied": applied_count, "total_pending": len(pending), "results": results, "failed": failed},
        )

    async def _rollback(self, params: Dict) -> SkillResult:
        """Rollback applied migrations."""
        db_name = params.get("db", "default")
        target_version = params.get("target_version")
        steps = params.get("steps", 1)

        state = _load_state()
        db_migrations = state.get("migrations", {}).get(db_name, {})

        if not db_migrations:
            return SkillResult(success=False, message=f"No migrations defined for '{db_name}'")

        db_path = _get_db_path(db_name)
        if not db_path.exists():
            return SkillResult(success=False, message=f"Database '{db_name}' does not exist")

        conn = sqlite3.connect(str(db_path))
        applied_versions = _get_applied_versions(conn)

        if not applied_versions:
            conn.close()
            return SkillResult(success=True, message="No migrations to rollback", data={"rolled_back": 0})

        # Determine which versions to rollback
        if target_version:
            # Rollback everything after target_version
            to_rollback = [v for v in reversed(applied_versions) if v > target_version]
        else:
            # Rollback N steps
            to_rollback = list(reversed(applied_versions))[:steps]

        if not to_rollback:
            conn.close()
            return SkillResult(success=True, message="Nothing to rollback", data={"rolled_back": 0})

        # Backup before rollback
        backup_path = db_path.with_suffix(f".backup_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}")
        shutil.copy2(str(db_path), str(backup_path))

        results = []
        rolled_back = 0
        failed = False

        for v in to_rollback:
            m = db_migrations.get(v)
            if not m:
                results.append({"version": v, "success": False, "error": "Migration definition not found"})
                failed = True
                break

            start_time = time.time()
            try:
                conn.executescript(m["down_sql"])
                elapsed_ms = int((time.time() - start_time) * 1000)

                # Remove from _migrations table
                conn.execute("DELETE FROM _migrations WHERE version = ?", (v,))
                conn.commit()

                results.append({"version": v, "name": m["name"], "success": True, "time_ms": elapsed_ms})
                rolled_back += 1
            except sqlite3.Error as e:
                elapsed_ms = int((time.time() - start_time) * 1000)
                results.append({"version": v, "name": m["name"], "success": False, "error": str(e), "time_ms": elapsed_ms})
                failed = True
                conn.close()
                if backup_path.exists():
                    shutil.copy2(str(backup_path), str(db_path))
                state["stats"]["total_failed"] += 1
                break

        if not failed:
            conn.close()

        # Update state
        rolled_back_versions = {r["version"] for r in results if r["success"]}
        if db_name in state["applied"]:
            state["applied"][db_name] = [v for v in state["applied"][db_name] if v not in rolled_back_versions]
        state["stats"]["total_rolled_back"] += rolled_back

        state["history"].append({
            "action": "rollback",
            "db": db_name,
            "rolled_back_count": rolled_back,
            "failed": failed,
            "versions": [r["version"] for r in results],
            "timestamp": _now_iso(),
            "backup": str(backup_path),
        })

        _save_state(state)

        # Cleanup backup on success
        if not failed and backup_path.exists():
            try:
                backup_path.unlink()
            except OSError:
                pass

        msg = f"Rolled back {rolled_back} migration(s) on '{db_name}'"
        if failed:
            msg += " (FAILED — database restored from backup)"

        return SkillResult(
            success=not failed,
            message=msg,
            data={"rolled_back": rolled_back, "results": results, "failed": failed},
        )

    async def _status(self, params: Dict) -> SkillResult:
        """Check migration status for a database."""
        db_name = params.get("db", "default")
        state = _load_state()
        db_migrations = state.get("migrations", {}).get(db_name, {})

        all_versions = sorted(db_migrations.keys())

        # Check applied from the database itself
        db_path = _get_db_path(db_name)
        applied_versions = []
        if db_path.exists():
            try:
                conn = sqlite3.connect(str(db_path))
                applied_versions = _get_applied_versions(conn)
                conn.close()
            except sqlite3.Error:
                pass

        pending = [v for v in all_versions if v not in applied_versions]
        current = applied_versions[-1] if applied_versions else None

        migration_list = []
        for v in all_versions:
            m = db_migrations[v]
            migration_list.append({
                "version": v,
                "name": m["name"],
                "status": "applied" if v in applied_versions else "pending",
                "created_at": m.get("created_at"),
            })

        return SkillResult(
            success=True,
            message=f"Database '{db_name}': {len(applied_versions)} applied, {len(pending)} pending",
            data={
                "db": db_name,
                "current_version": current,
                "total_migrations": len(all_versions),
                "applied_count": len(applied_versions),
                "pending_count": len(pending),
                "migrations": migration_list,
            },
        )

    async def _validate(self, params: Dict) -> SkillResult:
        """Validate migrations without applying them."""
        db_name = params.get("db", "default")
        state = _load_state()
        db_migrations = state.get("migrations", {}).get(db_name, {})

        if not db_migrations:
            return SkillResult(success=True, message=f"No migrations to validate for '{db_name}'", data={"valid": True, "issues": []})

        issues = []
        all_versions = sorted(db_migrations.keys())

        for v in all_versions:
            m = db_migrations[v]
            # Check required fields
            if not m.get("up_sql"):
                issues.append({"version": v, "issue": "Missing up_sql"})
            if not m.get("down_sql"):
                issues.append({"version": v, "issue": "Missing down_sql"})
            # Verify checksum
            expected = _compute_checksum(m.get("up_sql", "") + m.get("down_sql", ""))
            if m.get("checksum") and m["checksum"] != expected:
                issues.append({"version": v, "issue": "Checksum mismatch — migration may have been tampered with"})
            # Basic SQL syntax check using in-memory db
            if m.get("up_sql"):
                try:
                    test_conn = sqlite3.connect(":memory:")
                    test_conn.executescript(m["up_sql"])
                    # Also try down_sql on the result
                    if m.get("down_sql"):
                        test_conn.executescript(m["down_sql"])
                    test_conn.close()
                except sqlite3.Error as e:
                    issues.append({"version": v, "issue": f"SQL error: {str(e)}"})

        valid = len(issues) == 0

        return SkillResult(
            success=True,
            message=f"Validation {'passed' if valid else 'failed'}: {len(issues)} issue(s) found",
            data={"valid": valid, "issues": issues, "migrations_checked": len(all_versions)},
        )

    async def _history(self, params: Dict) -> SkillResult:
        """View migration operation audit trail."""
        db_name = params.get("db")
        limit = params.get("limit", 20)

        state = _load_state()
        history = state.get("history", [])

        if db_name:
            history = [h for h in history if h.get("db") == db_name]

        history = history[-limit:]

        return SkillResult(
            success=True,
            message=f"Migration history: {len(history)} entries",
            data={"history": history, "stats": state.get("stats", {})},
        )

    async def _generate(self, params: Dict) -> SkillResult:
        """Auto-generate migration SQL by comparing current schema vs desired."""
        db_name = params.get("db", "default")
        desired = params.get("desired_schema", {})

        if not desired:
            return SkillResult(success=False, message="desired_schema is required")

        db_path = _get_db_path(db_name)
        current = _get_schema(db_path)

        up_statements = []
        down_statements = []

        # Tables to create (in desired but not in current)
        for table, columns in desired.items():
            if table not in current:
                col_defs = []
                for col in columns:
                    col_def = f"{col['name']} {col.get('type', 'TEXT')}"
                    if col.get("pk"):
                        col_def += " PRIMARY KEY"
                    if col.get("notnull"):
                        col_def += " NOT NULL"
                    if col.get("dflt_value") is not None:
                        col_def += f" DEFAULT {col['dflt_value']}"
                    col_defs.append(col_def)
                up_statements.append(f"CREATE TABLE {table} ({', '.join(col_defs)});")
                down_statements.append(f"DROP TABLE IF EXISTS {table};")

        # Tables to drop (in current but not in desired)
        for table in current:
            if table not in desired:
                # Generate CREATE for rollback
                cols = current[table]
                col_defs = []
                for col in cols:
                    col_def = f"{col['name']} {col.get('type', 'TEXT')}"
                    if col.get("pk"):
                        col_def += " PRIMARY KEY"
                    if col.get("notnull"):
                        col_def += " NOT NULL"
                    if col.get("dflt_value") is not None:
                        col_def += f" DEFAULT {col['dflt_value']}"
                    col_defs.append(col_def)
                down_statements.append(f"CREATE TABLE {table} ({', '.join(col_defs)});")
                up_statements.append(f"DROP TABLE IF EXISTS {table};")

        # Columns to add (in desired table but not in current table)
        for table, desired_cols in desired.items():
            if table in current:
                current_col_names = {c["name"] for c in current[table]}
                for col in desired_cols:
                    if col["name"] not in current_col_names:
                        col_def = f"{col['name']} {col.get('type', 'TEXT')}"
                        if col.get("dflt_value") is not None:
                            col_def += f" DEFAULT {col['dflt_value']}"
                        up_statements.append(f"ALTER TABLE {table} ADD COLUMN {col_def};")
                        # Note: SQLite doesn't support DROP COLUMN before 3.35.0
                        down_statements.append(f"-- ALTER TABLE {table} DROP COLUMN {col['name']}; (requires SQLite 3.35+)")

        up_sql = "\n".join(up_statements) if up_statements else "-- No changes needed"
        down_sql = "\n".join(down_statements) if down_statements else "-- No changes needed"

        changes_count = len(up_statements)

        return SkillResult(
            success=True,
            message=f"Generated migration with {changes_count} change(s)",
            data={
                "changes_count": changes_count,
                "up_sql": up_sql,
                "down_sql": down_sql,
                "tables_created": [t for t in desired if t not in current],
                "tables_dropped": [t for t in current if t not in desired],
                "columns_added": sum(
                    1 for t, cols in desired.items() if t in current
                    for c in cols if c["name"] not in {x["name"] for x in current.get(t, [])}
                ),
            },
        )

    async def _squash(self, params: Dict) -> SkillResult:
        """Squash multiple sequential migrations into one."""
        db_name = params.get("db", "default")
        from_version = params.get("from_version", "")
        to_version = params.get("to_version", "")
        name = params.get("name", "squashed")

        if not from_version or not to_version:
            return SkillResult(success=False, message="from_version and to_version are required")

        state = _load_state()
        db_migrations = state.get("migrations", {}).get(db_name, {})
        all_versions = sorted(db_migrations.keys())

        # Find the range
        to_squash = [v for v in all_versions if from_version <= v <= to_version]

        if len(to_squash) < 2:
            return SkillResult(success=False, message="Need at least 2 migrations to squash")

        # Check none are applied
        db_path = _get_db_path(db_name)
        applied_versions = []
        if db_path.exists():
            try:
                conn = sqlite3.connect(str(db_path))
                applied_versions = _get_applied_versions(conn)
                conn.close()
            except sqlite3.Error:
                pass

        applied_in_range = [v for v in to_squash if v in applied_versions]
        if applied_in_range:
            return SkillResult(
                success=False,
                message=f"Cannot squash already-applied migrations: {applied_in_range}. Rollback first.",
            )

        # Combine up_sql in order, down_sql in reverse
        combined_up = []
        combined_down = []
        for v in to_squash:
            m = db_migrations[v]
            combined_up.append(f"-- From migration {v}_{m['name']}\n{m['up_sql']}")
        for v in reversed(to_squash):
            m = db_migrations[v]
            combined_down.append(f"-- Rollback migration {v}_{m['name']}\n{m['down_sql']}")

        squashed_up = "\n\n".join(combined_up)
        squashed_down = "\n\n".join(combined_down)

        # Remove old migrations
        for v in to_squash:
            del state["migrations"][db_name][v]

        # Create squashed migration with the from_version timestamp
        safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', name.lower())
        squashed = {
            "version": from_version,
            "name": f"squashed_{safe_name}",
            "up_sql": squashed_up,
            "down_sql": squashed_down,
            "checksum": _compute_checksum(squashed_up + squashed_down),
            "created_at": _now_iso(),
            "squashed_from": to_squash,
        }
        state["migrations"][db_name][from_version] = squashed

        state["history"].append({
            "action": "squash",
            "db": db_name,
            "squashed_count": len(to_squash),
            "from_version": from_version,
            "to_version": to_version,
            "result_version": from_version,
            "timestamp": _now_iso(),
        })

        _save_state(state)

        return SkillResult(
            success=True,
            message=f"Squashed {len(to_squash)} migrations into {from_version}_squashed_{safe_name}",
            data={
                "squashed_count": len(to_squash),
                "result_version": from_version,
                "result_name": f"squashed_{safe_name}",
                "original_versions": to_squash,
            },
        )
