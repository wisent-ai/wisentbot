"""Tests for DatabaseMigrationSkill."""
import json
import sqlite3
import pytest
from pathlib import Path
from unittest.mock import patch
from singularity.skills.database_migration import (
    DatabaseMigrationSkill, _load_state, _save_state, _compute_checksum,
    _get_schema, DATA_DIR, MIGRATION_STATE_FILE, DB_REGISTRY_FILE, MIGRATION_DIR,
)


@pytest.fixture
def skill(tmp_path):
    with patch("singularity.skills.database_migration.DATA_DIR", tmp_path), \
         patch("singularity.skills.database_migration.MIGRATION_STATE_FILE", tmp_path / "mig_state.json"), \
         patch("singularity.skills.database_migration.DB_REGISTRY_FILE", tmp_path / "reg.json"), \
         patch("singularity.skills.database_migration.MIGRATION_DIR", tmp_path / "migrations"):
        s = DatabaseMigrationSkill()
        # Create a test database with a simple schema
        db_path = tmp_path / "default.db"
        # Patch _get_db_path to always return our test db
        s._test_db_path = db_path
        s._test_tmp = tmp_path
        yield s


def _patch_db(skill):
    """Return a patch context that routes db lookups to test db."""
    return patch(
        "singularity.skills.database_migration._get_db_path",
        return_value=skill._test_db_path,
    )


@pytest.mark.asyncio
async def test_create_migration(skill):
    with _patch_db(skill):
        result = await skill.execute("create", {
            "name": "add_users_table",
            "up_sql": "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT);",
            "down_sql": "DROP TABLE IF EXISTS users;",
        })
    assert result.success
    assert result.data["name"] == "add_users_table"
    assert result.data["version"]


@pytest.mark.asyncio
async def test_create_requires_fields(skill):
    with _patch_db(skill):
        r1 = await skill.execute("create", {"up_sql": "x", "down_sql": "y"})
        assert not r1.success  # missing name
        r2 = await skill.execute("create", {"name": "x", "down_sql": "y"})
        assert not r2.success  # missing up_sql
        r3 = await skill.execute("create", {"name": "x", "up_sql": "y"})
        assert not r3.success  # missing down_sql


@pytest.mark.asyncio
async def test_apply_migration(skill):
    with _patch_db(skill):
        await skill.execute("create", {
            "name": "create_items",
            "up_sql": "CREATE TABLE items (id INTEGER PRIMARY KEY, value TEXT);",
            "down_sql": "DROP TABLE IF EXISTS items;",
        })
        result = await skill.execute("apply", {})
    assert result.success
    assert result.data["applied"] == 1
    # Verify table was created
    conn = sqlite3.connect(str(skill._test_db_path))
    tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    table_names = [t[0] for t in tables]
    assert "items" in table_names
    conn.close()


@pytest.mark.asyncio
async def test_apply_dry_run(skill):
    with _patch_db(skill):
        await skill.execute("create", {
            "name": "test_dry",
            "up_sql": "CREATE TABLE dry_test (id INTEGER PRIMARY KEY);",
            "down_sql": "DROP TABLE IF EXISTS dry_test;",
        })
        result = await skill.execute("apply", {"dry_run": True})
    assert result.success
    assert result.data["dry_run"] is True
    assert len(result.data["pending"]) == 1
    # Table should NOT exist
    conn = sqlite3.connect(str(skill._test_db_path))
    tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='dry_test'").fetchall()
    assert len(tables) == 0
    conn.close()


@pytest.mark.asyncio
async def test_rollback_migration(skill):
    with _patch_db(skill):
        await skill.execute("create", {
            "name": "create_rollback_test",
            "up_sql": "CREATE TABLE rb_test (id INTEGER PRIMARY KEY);",
            "down_sql": "DROP TABLE IF EXISTS rb_test;",
        })
        await skill.execute("apply", {})
        result = await skill.execute("rollback", {"steps": 1})
    assert result.success
    assert result.data["rolled_back"] == 1
    # Verify table was dropped
    conn = sqlite3.connect(str(skill._test_db_path))
    tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='rb_test'").fetchall()
    assert len(tables) == 0
    conn.close()


@pytest.mark.asyncio
async def test_status(skill):
    with _patch_db(skill):
        await skill.execute("create", {
            "name": "status_test",
            "up_sql": "CREATE TABLE st (id INTEGER PRIMARY KEY);",
            "down_sql": "DROP TABLE IF EXISTS st;",
        })
        result = await skill.execute("status", {})
    assert result.success
    assert result.data["pending_count"] == 1
    assert result.data["applied_count"] == 0


@pytest.mark.asyncio
async def test_validate_good_migrations(skill):
    with _patch_db(skill):
        await skill.execute("create", {
            "name": "valid_test",
            "up_sql": "CREATE TABLE vt (id INTEGER PRIMARY KEY, name TEXT);",
            "down_sql": "DROP TABLE IF EXISTS vt;",
        })
        result = await skill.execute("validate", {})
    assert result.success
    assert result.data["valid"] is True


@pytest.mark.asyncio
async def test_generate_migration(skill):
    # Create a database with an existing table
    conn = sqlite3.connect(str(skill._test_db_path))
    conn.execute("CREATE TABLE existing (id INTEGER PRIMARY KEY, val TEXT)")
    conn.commit()
    conn.close()

    with _patch_db(skill):
        result = await skill.execute("generate", {
            "desired_schema": {
                "existing": [
                    {"name": "id", "type": "INTEGER", "pk": 1},
                    {"name": "val", "type": "TEXT"},
                    {"name": "new_col", "type": "REAL"},
                ],
                "new_table": [
                    {"name": "id", "type": "INTEGER", "pk": 1},
                    {"name": "data", "type": "TEXT"},
                ],
            }
        })
    assert result.success
    assert result.data["changes_count"] > 0
    assert "new_table" in result.data["tables_created"]
    assert result.data["columns_added"] == 1


@pytest.mark.asyncio
async def test_squash_migrations(skill):
    with _patch_db(skill):
        r1 = await skill.execute("create", {
            "name": "first", "up_sql": "CREATE TABLE a (id INTEGER PRIMARY KEY);", "down_sql": "DROP TABLE IF EXISTS a;",
        })
        import asyncio
        await asyncio.sleep(0.01)  # Ensure different timestamps
        r2 = await skill.execute("create", {
            "name": "second", "up_sql": "CREATE TABLE b (id INTEGER PRIMARY KEY);", "down_sql": "DROP TABLE IF EXISTS b;",
        })
        v1 = r1.data["version"]
        v2 = r2.data["version"]
        result = await skill.execute("squash", {"from_version": v1, "to_version": v2, "name": "combined"})
    assert result.success
    assert result.data["squashed_count"] == 2


@pytest.mark.asyncio
async def test_history(skill):
    with _patch_db(skill):
        await skill.execute("create", {
            "name": "hist_test", "up_sql": "CREATE TABLE ht (id INTEGER);", "down_sql": "DROP TABLE IF EXISTS ht;",
        })
        result = await skill.execute("history", {"limit": 10})
    assert result.success
    assert len(result.data["history"]) >= 1


@pytest.mark.asyncio
async def test_apply_idempotent(skill):
    """Applying when already up to date should succeed gracefully."""
    with _patch_db(skill):
        await skill.execute("create", {
            "name": "idem_test",
            "up_sql": "CREATE TABLE idem (id INTEGER PRIMARY KEY);",
            "down_sql": "DROP TABLE IF EXISTS idem;",
        })
        await skill.execute("apply", {})
        result = await skill.execute("apply", {})
    assert result.success
    assert result.data["applied"] == 0
