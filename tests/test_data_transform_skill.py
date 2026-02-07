"""Tests for DataTransformSkill."""
import pytest
from singularity.skills.data_transform import DataTransformSkill

@pytest.fixture
def skill():
    return DataTransformSkill({})

@pytest.fixture
def sample_data():
    return [
        {"name": "Alice", "age": 30, "dept": "eng"},
        {"name": "Bob", "age": 25, "dept": "eng"},
        {"name": "Carol", "age": 35, "dept": "sales"},
    ]

@pytest.mark.asyncio
async def test_json_to_csv(skill, sample_data):
    r = await skill.execute("json_to_csv", {"data": sample_data})
    assert r.success
    assert "Alice" in r.data["csv"]
    assert r.data["rows"] == 3

@pytest.mark.asyncio
async def test_json_to_csv_with_columns(skill, sample_data):
    r = await skill.execute("json_to_csv", {"data": sample_data, "columns": ["name", "age"]})
    assert r.success
    assert "dept" not in r.data["csv"].split("\n")[0]

@pytest.mark.asyncio
async def test_csv_to_json(skill):
    csv_text = "name,age\nAlice,30\nBob,25"
    r = await skill.execute("csv_to_json", {"csv_text": csv_text})
    assert r.success
    assert len(r.data["data"]) == 2
    assert r.data["data"][0]["name"] == "Alice"

@pytest.mark.asyncio
async def test_roundtrip_json_csv(skill, sample_data):
    r1 = await skill.execute("json_to_csv", {"data": sample_data})
    r2 = await skill.execute("csv_to_json", {"csv_text": r1.data["csv"]})
    assert r2.success
    assert len(r2.data["data"]) == 3

@pytest.mark.asyncio
async def test_flatten_json(skill):
    data = {"a": {"b": {"c": 1}, "d": 2}, "e": 3}
    r = await skill.execute("flatten_json", {"data": data})
    assert r.success
    assert r.data["data"]["a.b.c"] == 1
    assert r.data["data"]["a.d"] == 2
    assert r.data["data"]["e"] == 3

@pytest.mark.asyncio
async def test_unflatten_json(skill):
    data = {"a.b.c": 1, "a.d": 2, "e": 3}
    r = await skill.execute("unflatten_json", {"data": data})
    assert r.success
    assert r.data["data"]["a"]["b"]["c"] == 1

@pytest.mark.asyncio
async def test_filter_data_eq(skill, sample_data):
    r = await skill.execute("filter_data", {"data": sample_data, "conditions": {"dept": {"eq": "eng"}}})
    assert r.success
    assert r.data["matched"] == 2

@pytest.mark.asyncio
async def test_filter_data_gt(skill, sample_data):
    r = await skill.execute("filter_data", {"data": sample_data, "conditions": {"age": {"gt": 28}}})
    assert r.success
    assert r.data["matched"] == 2

@pytest.mark.asyncio
async def test_aggregate(skill, sample_data):
    r = await skill.execute("aggregate", {"data": sample_data, "field": "age"})
    assert r.success
    assert r.data["count"] == 3
    assert r.data["sum"] == 90
    assert r.data["avg"] == 30.0
    assert r.data["min"] == 25
    assert r.data["max"] == 35

@pytest.mark.asyncio
async def test_pick_fields(skill, sample_data):
    r = await skill.execute("pick_fields", {"data": sample_data, "fields": ["name"]})
    assert r.success
    assert all("age" not in row for row in r.data["data"])
    assert all("name" in row for row in r.data["data"])

@pytest.mark.asyncio
async def test_sort_data(skill, sample_data):
    r = await skill.execute("sort_data", {"data": sample_data, "field": "age"})
    assert r.success
    assert r.data["data"][0]["name"] == "Bob"

@pytest.mark.asyncio
async def test_sort_data_reverse(skill, sample_data):
    r = await skill.execute("sort_data", {"data": sample_data, "field": "age", "reverse": True})
    assert r.success
    assert r.data["data"][0]["name"] == "Carol"

@pytest.mark.asyncio
async def test_group_by(skill, sample_data):
    r = await skill.execute("group_by", {"data": sample_data, "field": "dept"})
    assert r.success
    assert r.data["total_groups"] == 2
    assert r.data["group_counts"]["eng"] == 2

@pytest.mark.asyncio
async def test_unknown_action(skill):
    r = await skill.execute("bogus", {})
    assert not r.success

@pytest.mark.asyncio
async def test_empty_data(skill):
    r = await skill.execute("json_to_csv", {"data": []})
    assert not r.success

@pytest.mark.asyncio
async def test_manifest(skill):
    m = skill.manifest
    assert m.skill_id == "data_transform"
    assert len(m.actions) == 9
    assert m.required_credentials == []
