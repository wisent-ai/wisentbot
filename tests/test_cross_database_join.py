"""Tests for CrossDatabaseJoinSkill."""

import json
import os
import sqlite3
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch

from singularity.skills.cross_database_join import CrossDatabaseJoinSkill, DATA_DIR


@pytest.fixture
def temp_dir(tmp_path):
    """Use a temp directory for all data."""
    return tmp_path


@pytest.fixture
def two_dbs(temp_dir):
    """Create two SQLite databases with related data."""
    # Database A: users
    db_a = str(temp_dir / "users.db")
    conn_a = sqlite3.connect(db_a)
    conn_a.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, email TEXT)")
    conn_a.execute("INSERT INTO users VALUES (1, 'Alice', 'alice@test.com')")
    conn_a.execute("INSERT INTO users VALUES (2, 'Bob', 'bob@test.com')")
    conn_a.execute("INSERT INTO users VALUES (3, 'Charlie', 'charlie@test.com')")
    conn_a.commit()
    conn_a.close()

    # Database B: orders
    db_b = str(temp_dir / "orders.db")
    conn_b = sqlite3.connect(db_b)
    conn_b.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER, amount REAL, status TEXT)")
    conn_b.execute("INSERT INTO orders VALUES (1, 1, 99.99, 'completed')")
    conn_b.execute("INSERT INTO orders VALUES (2, 1, 49.50, 'pending')")
    conn_b.execute("INSERT INTO orders VALUES (3, 2, 199.00, 'completed')")
    conn_b.commit()
    conn_b.close()

    return db_a, db_b


@pytest.fixture
def skill(temp_dir, two_dbs):
    """Create skill with mocked data directory and registry."""
    db_a, db_b = two_dbs
    registry = {
        "users_db": {"path": db_a, "created": "2024-01-01"},
        "orders_db": {"path": db_b, "created": "2024-01-01"},
    }
    state_file = temp_dir / "cross_database_join.json"
    registry_file = temp_dir / "database_registry.json"
    with open(registry_file, "w") as f:
        json.dump(registry, f)

    with patch("singularity.skills.cross_database_join.DATA_DIR", temp_dir), \
         patch("singularity.skills.cross_database_join.DB_REGISTRY_FILE", registry_file), \
         patch("singularity.skills.cross_database_join.CROSS_DB_STATE_FILE", state_file):
        s = CrossDatabaseJoinSkill()
        yield s


@pytest.mark.asyncio
async def test_create_session(skill):
    result = skill._create_session({"databases": {"u": "users_db", "o": "orders_db"}})
    assert result.success
    assert "session_id" in result.data
    assert result.data["attached_databases"] == {"u": "users_db", "o": "orders_db"}


@pytest.mark.asyncio
async def test_cross_database_join_query(skill):
    create = skill._create_session({"databases": {"u": "users_db", "o": "orders_db"}})
    sid = create.data["session_id"]

    result = skill._query({
        "session_id": sid,
        "sql": "SELECT u.users.name, o.orders.amount FROM u.users JOIN o.orders ON u.users.id = o.orders.user_id",
    })
    assert result.success
    assert result.data["row_count"] == 3
    names = {r["name"] for r in result.data["rows"]}
    assert "Alice" in names
    assert "Bob" in names


@pytest.mark.asyncio
async def test_federated_query(skill):
    result = skill._federated_query({
        "databases": {"u": "users_db", "o": "orders_db"},
        "sql": "SELECT u.users.name, SUM(o.orders.amount) as total FROM u.users JOIN o.orders ON u.users.id = o.orders.user_id GROUP BY u.users.name",
    })
    assert result.success
    assert result.data["federated"] is True
    rows = result.data["rows"]
    alice_row = [r for r in rows if r["name"] == "Alice"][0]
    assert alice_row["total"] == pytest.approx(149.49)


@pytest.mark.asyncio
async def test_query_read_only_enforcement(skill):
    create = skill._create_session({"databases": {"u": "users_db"}})
    sid = create.data["session_id"]

    result = skill._query({
        "session_id": sid,
        "sql": "DELETE FROM u.users WHERE id = 1",
    })
    assert not result.success
    assert "not allowed" in result.message.lower() or "DELETE" in result.message


@pytest.mark.asyncio
async def test_attach_detach(skill):
    create = skill._create_session({"databases": {"u": "users_db"}})
    sid = create.data["session_id"]

    attach = skill._attach({"session_id": sid, "alias": "o", "db_name": "orders_db"})
    assert attach.success
    assert attach.data["total_attached"] == 2

    detach = skill._detach({"session_id": sid, "alias": "o"})
    assert detach.success
    assert detach.data["remaining"] == 1


@pytest.mark.asyncio
async def test_discover_join_candidates(skill):
    result = skill._discover({"databases": ["users_db", "orders_db"]})
    assert result.success
    candidates = result.data["join_candidates"]
    assert len(candidates) > 0
    # 'id' column exists in both - should be a candidate
    id_matches = [c for c in candidates if c["column"] == "id"]
    assert len(id_matches) > 0


@pytest.mark.asyncio
async def test_list_and_close_sessions(skill):
    create = skill._create_session({"databases": {"u": "users_db"}})
    sid = create.data["session_id"]

    listing = skill._list_sessions({})
    assert listing.success
    assert listing.data["count"] == 1

    close = skill._close_session({"session_id": sid})
    assert close.success
    assert close.data["queries_executed"] == 0

    listing2 = skill._list_sessions({})
    assert listing2.data["count"] == 0


@pytest.mark.asyncio
async def test_stats(skill):
    skill._federated_query({
        "databases": {"u": "users_db"},
        "sql": "SELECT * FROM u.users",
    })
    result = skill._stats({"last_n": 5})
    assert result.success
    assert result.data["stats"]["total_queries"] >= 1


@pytest.mark.asyncio
async def test_invalid_alias(skill):
    result = skill._create_session({"databases": {"123bad": "users_db"}})
    # Session might still be created if other dbs work, but this alias should fail
    assert not result.success or "warnings" in result.data


@pytest.mark.asyncio
async def test_session_not_found(skill):
    result = skill._query({"session_id": "nonexistent", "sql": "SELECT 1"})
    assert not result.success
    assert "not found" in result.message


@pytest.mark.asyncio
async def test_manifest(skill):
    m = skill.manifest
    assert m.skill_id == "cross_database_join"
    assert len(m.actions) == 9
