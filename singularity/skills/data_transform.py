#!/usr/bin/env python3
"""
DataTransformSkill - Data format conversion and transformation.

Zero external dependencies. Uses Python stdlib (json, csv, io).
Enables the agent to offer data processing services (Revenue pillar)
and to manipulate structured data for its own operations (Self-Improvement).
"""

import csv
import io
import json
from typing import Any, Dict, List, Optional

from .base import Skill, SkillAction, SkillManifest, SkillResult


class DataTransformSkill(Skill):
    """Skill for converting, filtering, and aggregating structured data."""

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="data_transform",
            name="Data Transform",
            version="1.0.0",
            category="data",
            description="Convert, filter, and aggregate structured data (JSON, CSV)",
            required_credentials=[],
            install_cost=0,
            actions=[
                SkillAction(
                    name="json_to_csv",
                    description="Convert a JSON array of objects to CSV string",
                    parameters={
                        "data": {"type": "array", "required": True, "description": "JSON array of objects"},
                        "columns": {"type": "array", "required": False, "description": "Column order (default: all keys)"},
                    },
                    estimated_cost=0,
                    estimated_duration_seconds=1,
                    success_probability=0.95,
                ),
                SkillAction(
                    name="csv_to_json",
                    description="Convert CSV string to JSON array of objects",
                    parameters={
                        "csv_text": {"type": "string", "required": True, "description": "CSV text with header row"},
                    },
                    estimated_cost=0,
                    estimated_duration_seconds=1,
                    success_probability=0.95,
                ),
                SkillAction(
                    name="flatten_json",
                    description="Flatten nested JSON into dot-notation keys",
                    parameters={
                        "data": {"type": "object", "required": True, "description": "Nested JSON object"},
                        "separator": {"type": "string", "required": False, "description": "Key separator (default: '.')"},
                    },
                    estimated_cost=0,
                    estimated_duration_seconds=1,
                    success_probability=0.98,
                ),
                SkillAction(
                    name="unflatten_json",
                    description="Unflatten dot-notation keys back into nested JSON",
                    parameters={
                        "data": {"type": "object", "required": True, "description": "Flat JSON with dot-notation keys"},
                        "separator": {"type": "string", "required": False, "description": "Key separator (default: '.')"},
                    },
                    estimated_cost=0,
                    estimated_duration_seconds=1,
                    success_probability=0.98,
                ),
                SkillAction(
                    name="filter_data",
                    description="Filter a JSON array by field conditions",
                    parameters={
                        "data": {"type": "array", "required": True, "description": "JSON array of objects"},
                        "conditions": {"type": "object", "required": True, "description": "Field conditions, e.g. {\"age\": {\"gt\": 18}, \"status\": {\"eq\": \"active\"}}"},
                    },
                    estimated_cost=0,
                    estimated_duration_seconds=1,
                    success_probability=0.95,
                ),
                SkillAction(
                    name="aggregate",
                    description="Compute statistics on a field in a JSON array",
                    parameters={
                        "data": {"type": "array", "required": True, "description": "JSON array of objects"},
                        "field": {"type": "string", "required": True, "description": "Field name to aggregate"},
                        "operations": {"type": "array", "required": False, "description": "Operations: count, sum, avg, min, max (default: all)"},
                    },
                    estimated_cost=0,
                    estimated_duration_seconds=1,
                    success_probability=0.95,
                ),
                SkillAction(
                    name="pick_fields",
                    description="Select specific fields from each object in a JSON array",
                    parameters={
                        "data": {"type": "array", "required": True, "description": "JSON array of objects"},
                        "fields": {"type": "array", "required": True, "description": "Field names to keep"},
                    },
                    estimated_cost=0,
                    estimated_duration_seconds=1,
                    success_probability=0.98,
                ),
                SkillAction(
                    name="sort_data",
                    description="Sort a JSON array by a field",
                    parameters={
                        "data": {"type": "array", "required": True, "description": "JSON array of objects"},
                        "field": {"type": "string", "required": True, "description": "Field to sort by"},
                        "reverse": {"type": "boolean", "required": False, "description": "Sort descending (default: false)"},
                    },
                    estimated_cost=0,
                    estimated_duration_seconds=1,
                    success_probability=0.95,
                ),
                SkillAction(
                    name="group_by",
                    description="Group a JSON array by a field value",
                    parameters={
                        "data": {"type": "array", "required": True, "description": "JSON array of objects"},
                        "field": {"type": "string", "required": True, "description": "Field to group by"},
                    },
                    estimated_cost=0,
                    estimated_duration_seconds=1,
                    success_probability=0.95,
                ),
            ],
        )

    async def execute(self, action: str, params: Dict) -> SkillResult:
        handlers = {
            "json_to_csv": self._json_to_csv,
            "csv_to_json": self._csv_to_json,
            "flatten_json": self._flatten_json,
            "unflatten_json": self._unflatten_json,
            "filter_data": self._filter_data,
            "aggregate": self._aggregate,
            "pick_fields": self._pick_fields,
            "sort_data": self._sort_data,
            "group_by": self._group_by,
        }
        handler = handlers.get(action)
        if not handler:
            return SkillResult(success=False, message=f"Unknown action: {action}")
        try:
            return await handler(params)
        except Exception as e:
            return SkillResult(success=False, message=f"Error in {action}: {e}")

    async def _json_to_csv(self, params: Dict) -> SkillResult:
        data = params.get("data")
        if not data or not isinstance(data, list):
            return SkillResult(success=False, message="'data' must be a non-empty array")
        if not all(isinstance(row, dict) for row in data):
            return SkillResult(success=False, message="Each element in 'data' must be an object")

        columns = params.get("columns")
        if not columns:
            # Collect all unique keys in order
            seen = {}
            for row in data:
                for k in row:
                    if k not in seen:
                        seen[k] = True
            columns = list(seen.keys())

        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in data:
            writer.writerow(row)
        csv_text = output.getvalue()
        return SkillResult(
            success=True,
            message=f"Converted {len(data)} rows to CSV with {len(columns)} columns",
            data={"csv": csv_text, "rows": len(data), "columns": columns},
        )

    async def _csv_to_json(self, params: Dict) -> SkillResult:
        csv_text = params.get("csv_text", "")
        if not csv_text.strip():
            return SkillResult(success=False, message="'csv_text' is required")

        reader = csv.DictReader(io.StringIO(csv_text))
        rows = list(reader)
        columns = reader.fieldnames or []
        return SkillResult(
            success=True,
            message=f"Converted CSV to {len(rows)} JSON objects with {len(columns)} fields",
            data={"data": rows, "rows": len(rows), "columns": list(columns)},
        )

    def _do_flatten(self, obj: Any, parent_key: str = "", sep: str = ".") -> Dict:
        items = {}
        if isinstance(obj, dict):
            for k, v in obj.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.update(self._do_flatten(v, new_key, sep))
                elif isinstance(v, list):
                    for i, item in enumerate(v):
                        items.update(self._do_flatten(item, f"{new_key}{sep}{i}", sep))
                else:
                    items[new_key] = v
        else:
            items[parent_key] = obj
        return items

    async def _flatten_json(self, params: Dict) -> SkillResult:
        data = params.get("data")
        if not isinstance(data, dict):
            return SkillResult(success=False, message="'data' must be a JSON object")
        sep = params.get("separator", ".")
        flat = self._do_flatten(data, sep=sep)
        return SkillResult(
            success=True,
            message=f"Flattened to {len(flat)} keys",
            data={"data": flat, "key_count": len(flat)},
        )

    async def _unflatten_json(self, params: Dict) -> SkillResult:
        data = params.get("data")
        if not isinstance(data, dict):
            return SkillResult(success=False, message="'data' must be a JSON object")
        sep = params.get("separator", ".")
        result: Dict = {}
        for compound_key, value in data.items():
            parts = compound_key.split(sep)
            d = result
            for part in parts[:-1]:
                if part not in d:
                    d[part] = {}
                d = d[part]
            d[parts[-1]] = value
        return SkillResult(
            success=True,
            message=f"Unflattened {len(data)} keys",
            data={"data": result},
        )

    def _match_condition(self, value: Any, condition: Any) -> bool:
        """Check if a value matches a condition dict or equality."""
        if not isinstance(condition, dict):
            return value == condition
        for op, target in condition.items():
            if op == "eq" and value != target:
                return False
            elif op == "ne" and value == target:
                return False
            elif op == "gt" and not (isinstance(value, (int, float)) and value > target):
                return False
            elif op == "gte" and not (isinstance(value, (int, float)) and value >= target):
                return False
            elif op == "lt" and not (isinstance(value, (int, float)) and value < target):
                return False
            elif op == "lte" and not (isinstance(value, (int, float)) and value <= target):
                return False
            elif op == "in" and value not in target:
                return False
            elif op == "contains" and isinstance(value, str) and target not in value:
                return False
        return True

    async def _filter_data(self, params: Dict) -> SkillResult:
        data = params.get("data")
        if not isinstance(data, list):
            return SkillResult(success=False, message="'data' must be an array")
        conditions = params.get("conditions", {})
        if not isinstance(conditions, dict):
            return SkillResult(success=False, message="'conditions' must be an object")

        filtered = []
        for row in data:
            if not isinstance(row, dict):
                continue
            match = True
            for field, cond in conditions.items():
                if not self._match_condition(row.get(field), cond):
                    match = False
                    break
            if match:
                filtered.append(row)

        return SkillResult(
            success=True,
            message=f"Filtered {len(data)} rows to {len(filtered)} matching rows",
            data={"data": filtered, "total": len(data), "matched": len(filtered)},
        )

    async def _aggregate(self, params: Dict) -> SkillResult:
        data = params.get("data")
        if not isinstance(data, list):
            return SkillResult(success=False, message="'data' must be an array")
        field = params.get("field", "")
        if not field:
            return SkillResult(success=False, message="'field' is required")

        ops = params.get("operations", ["count", "sum", "avg", "min", "max"])

        values = []
        for row in data:
            if isinstance(row, dict) and field in row:
                v = row[field]
                if isinstance(v, (int, float)):
                    values.append(v)

        result: Dict[str, Any] = {"field": field, "total_rows": len(data), "numeric_values": len(values)}
        if "count" in ops:
            result["count"] = len(values)
        if values:
            if "sum" in ops:
                result["sum"] = sum(values)
            if "avg" in ops:
                result["avg"] = sum(values) / len(values)
            if "min" in ops:
                result["min"] = min(values)
            if "max" in ops:
                result["max"] = max(values)

        return SkillResult(
            success=True,
            message=f"Aggregated {len(values)} numeric values from '{field}'",
            data=result,
        )

    async def _pick_fields(self, params: Dict) -> SkillResult:
        data = params.get("data")
        if not isinstance(data, list):
            return SkillResult(success=False, message="'data' must be an array")
        fields = params.get("fields", [])
        if not fields:
            return SkillResult(success=False, message="'fields' is required")

        picked = []
        for row in data:
            if isinstance(row, dict):
                picked.append({f: row.get(f) for f in fields if f in row})
        return SkillResult(
            success=True,
            message=f"Picked {len(fields)} fields from {len(picked)} rows",
            data={"data": picked, "rows": len(picked), "fields": fields},
        )

    async def _sort_data(self, params: Dict) -> SkillResult:
        data = params.get("data")
        if not isinstance(data, list):
            return SkillResult(success=False, message="'data' must be an array")
        field = params.get("field", "")
        if not field:
            return SkillResult(success=False, message="'field' is required")
        reverse = params.get("reverse", False)

        def sort_key(row):
            if not isinstance(row, dict):
                return ""
            v = row.get(field)
            if v is None:
                return "" if not reverse else chr(0x10FFFF)
            return v

        try:
            sorted_data = sorted(data, key=sort_key, reverse=reverse)
        except TypeError:
            # Mixed types - convert to string for comparison
            sorted_data = sorted(data, key=lambda r: str(sort_key(r)), reverse=reverse)

        return SkillResult(
            success=True,
            message=f"Sorted {len(sorted_data)} rows by '{field}' {'desc' if reverse else 'asc'}",
            data={"data": sorted_data, "rows": len(sorted_data), "field": field, "reverse": reverse},
        )

    async def _group_by(self, params: Dict) -> SkillResult:
        data = params.get("data")
        if not isinstance(data, list):
            return SkillResult(success=False, message="'data' must be an array")
        field = params.get("field", "")
        if not field:
            return SkillResult(success=False, message="'field' is required")

        groups: Dict[str, list] = {}
        for row in data:
            if not isinstance(row, dict):
                continue
            key = str(row.get(field, "_none_"))
            if key not in groups:
                groups[key] = []
            groups[key].append(row)

        group_summary = {k: len(v) for k, v in groups.items()}
        return SkillResult(
            success=True,
            message=f"Grouped {len(data)} rows into {len(groups)} groups by '{field}'",
            data={"groups": groups, "group_counts": group_summary, "total_groups": len(groups)},
        )
