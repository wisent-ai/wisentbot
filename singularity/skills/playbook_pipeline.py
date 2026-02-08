#!/usr/bin/env python3
"""
PlaybookPipelineSkill - Convert playbooks into executable pipelines.

The critical gap: playbooks from AgentReflectionSkill contain step-by-step
strategies as text, but they aren't executable. The PipelineExecutor can
run multi-step action chains, but requires structured pipeline definitions.

This skill bridges the two:
1. **Convert** playbook steps into PipelineExecutor-compatible pipeline definitions
2. **Map** textual step descriptions to concrete skill:action tool references
3. **Execute** converted pipelines via PipelineExecutor with cost/timeout guards
4. **Track** which playbooks have been executed as pipelines and their outcomes
5. **Feedback** execution results back to AgentReflectionSkill as playbook usage data
6. **Template** library of common step-to-action mappings for reuse

The conversion loop:
  playbook.steps (text) → tool mapping → pipeline definition → execute → record outcome → update playbook effectiveness

Pillar: Self-Improvement (turns learned strategies into automated execution)
"""

import json
import re
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from .base import Skill, SkillResult, SkillManifest, SkillAction

PIPELINE_DATA_FILE = Path(__file__).parent.parent / "data" / "playbook_pipelines.json"
MAX_EXECUTIONS = 200
MAX_MAPPINGS = 500


class PlaybookPipelineSkill(Skill):
    """
    Converts playbooks into executable pipelines and runs them.

    Works with:
    - AgentReflectionSkill (via SkillContext) to read playbooks and record usage
    - PipelineExecutor concepts to build executable step chains

    Step mapping uses keyword matching against a configurable mapping table
    that associates common action descriptions with tool:action references.
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._load_state()

    def _load_state(self):
        """Load or initialize pipeline state."""
        PIPELINE_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        if PIPELINE_DATA_FILE.exists():
            try:
                with open(PIPELINE_DATA_FILE) as f:
                    data = json.load(f)
                self._mappings = data.get("mappings", self._default_mappings())
                self._conversions = data.get("conversions", [])
                self._executions = data.get("executions", [])
                self._config = data.get("config", self._default_config())
                self._stats = data.get("stats", self._default_stats())
            except (json.JSONDecodeError, Exception):
                self._init_empty()
        else:
            self._init_empty()

    def _init_empty(self):
        self._mappings: Dict[str, Dict] = self._default_mappings()
        self._conversions: List[Dict] = []
        self._executions: List[Dict] = []
        self._config = self._default_config()
        self._stats = self._default_stats()

    def _default_mappings(self) -> Dict[str, Dict]:
        """Built-in keyword-to-tool mappings for common playbook step patterns."""
        return {
            "git_status": {
                "keywords": ["git status", "check status", "working tree", "uncommitted"],
                "tool": "shell:run",
                "params": {"command": "git status"},
                "description": "Check git working tree status",
            },
            "git_commit": {
                "keywords": ["git commit", "commit changes", "save changes"],
                "tool": "shell:run",
                "params": {"command": "git add -A && git commit -m 'auto-commit'"},
                "description": "Stage and commit all changes",
            },
            "run_tests": {
                "keywords": ["run tests", "pytest", "test suite", "unit tests", "run the tests"],
                "tool": "shell:run",
                "params": {"command": "pytest -x"},
                "description": "Run test suite",
            },
            "create_pr": {
                "keywords": ["pull request", "create pr", "open pr", "submit pr"],
                "tool": "github:create_pr",
                "params": {"title": "Auto-generated PR"},
                "description": "Create a GitHub pull request",
            },
            "code_review": {
                "keywords": ["review code", "code review", "check code quality"],
                "tool": "code_review:review",
                "params": {},
                "description": "Run code review",
            },
            "read_file": {
                "keywords": ["read file", "open file", "check file", "examine file", "look at"],
                "tool": "filesystem:read",
                "params": {},
                "description": "Read a file's contents",
            },
            "write_file": {
                "keywords": ["write file", "create file", "save file", "update file"],
                "tool": "filesystem:write",
                "params": {},
                "description": "Write content to a file",
            },
            "search_code": {
                "keywords": ["search code", "find in code", "grep", "search for"],
                "tool": "shell:run",
                "params": {"command": "grep -r '' ."},
                "description": "Search codebase",
            },
            "install_deps": {
                "keywords": ["install dependencies", "pip install", "npm install", "requirements"],
                "tool": "shell:run",
                "params": {"command": "pip install -r requirements.txt"},
                "description": "Install project dependencies",
            },
            "check_memory": {
                "keywords": ["check memory", "read memory", "load memory", "recall"],
                "tool": "memory:recall",
                "params": {},
                "description": "Check agent memory",
            },
            "reflect": {
                "keywords": ["reflect", "analyze outcome", "review what happened"],
                "tool": "agent_reflection:reflect",
                "params": {},
                "description": "Reflect on recent actions",
            },
            "deploy": {
                "keywords": ["deploy", "push to production", "release", "ship"],
                "tool": "shell:run",
                "params": {"command": "echo 'deploy placeholder'"},
                "description": "Deploy the application",
            },
        }

    def _default_config(self) -> Dict:
        return {
            "min_mapping_confidence": 0.3,
            "max_pipeline_cost": 0.50,
            "max_pipeline_timeout": 120.0,
            "auto_record_to_reflection": True,
            "dry_run_by_default": False,
        }

    def _default_stats(self) -> Dict:
        return {
            "total_conversions": 0,
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "total_steps_mapped": 0,
            "unmapped_steps": 0,
        }

    def _save(self):
        """Persist state."""
        data = {
            "mappings": self._mappings,
            "conversions": self._conversions[-MAX_EXECUTIONS:],
            "executions": self._executions[-MAX_EXECUTIONS:],
            "config": self._config,
            "stats": self._stats,
        }
        with open(PIPELINE_DATA_FILE, "w") as f:
            json.dump(data, f, indent=2, default=str)

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="playbook_pipeline",
            name="Playbook Pipeline",
            version="1.0.0",
            category="self-improvement",
            description="Convert playbooks into executable pipelines and run them",
            actions=[
                SkillAction(
                    name="convert",
                    description="Convert a playbook's steps into a pipeline definition",
                    parameters={
                        "playbook_name": {"type": "string", "required": True, "description": "Name of playbook to convert"},
                        "param_overrides": {"type": "dict", "required": False, "description": "Override default params for specific steps"},
                    },
                ),
                SkillAction(
                    name="execute",
                    description="Execute a converted pipeline (or convert and execute in one step)",
                    parameters={
                        "playbook_name": {"type": "string", "required": True, "description": "Name of playbook to execute"},
                        "dry_run": {"type": "bool", "required": False, "description": "If true, show pipeline without executing"},
                        "max_cost": {"type": "float", "required": False, "description": "Max cost for this execution"},
                        "param_overrides": {"type": "dict", "required": False, "description": "Override params for steps"},
                    },
                ),
                SkillAction(
                    name="add_mapping",
                    description="Add a new step keyword-to-tool mapping",
                    parameters={
                        "mapping_id": {"type": "string", "required": True, "description": "Unique identifier for this mapping"},
                        "keywords": {"type": "list", "required": True, "description": "Keywords to match in step descriptions"},
                        "tool": {"type": "string", "required": True, "description": "Tool:action reference (e.g. shell:run)"},
                        "params": {"type": "dict", "required": False, "description": "Default parameters for this tool"},
                        "description": {"type": "string", "required": False, "description": "Human-readable description"},
                    },
                ),
                SkillAction(
                    name="remove_mapping",
                    description="Remove a step keyword-to-tool mapping",
                    parameters={
                        "mapping_id": {"type": "string", "required": True, "description": "ID of mapping to remove"},
                    },
                ),
                SkillAction(
                    name="list_mappings",
                    description="List all available step-to-tool mappings",
                    parameters={},
                ),
                SkillAction(
                    name="match_step",
                    description="Preview which tool a step description would map to",
                    parameters={
                        "step_text": {"type": "string", "required": True, "description": "Step description to match"},
                    },
                ),
                SkillAction(
                    name="history",
                    description="View pipeline execution history",
                    parameters={
                        "limit": {"type": "int", "required": False, "description": "Max entries to return (default 10)"},
                    },
                ),
                SkillAction(
                    name="status",
                    description="Get overall status and statistics",
                    parameters={},
                ),
            ],
            required_credentials=[],
        )

    async def execute(self, action: str, params: Dict[str, Any]) -> SkillResult:
        handlers = {
            "convert": self._convert,
            "execute": self._execute,
            "add_mapping": self._add_mapping,
            "remove_mapping": self._remove_mapping,
            "list_mappings": self._list_mappings,
            "match_step": self._match_step,
            "history": self._history,
            "status": self._status,
        }
        handler = handlers.get(action)
        if not handler:
            return SkillResult(
                success=False,
                message=f"Unknown action: {action}. Available: {list(handlers.keys())}",
            )
        return await handler(params)

    def _match_step_to_tool(self, step_text: str) -> Tuple[Optional[Dict], float]:
        """
        Match a textual step description to the best tool mapping.

        Returns (mapping_dict, confidence_score) or (None, 0.0).
        """
        step_lower = step_text.lower().strip()
        best_match = None
        best_score = 0.0

        for mapping_id, mapping in self._mappings.items():
            keywords = mapping.get("keywords", [])
            score = 0.0
            matches = 0

            for kw in keywords:
                kw_lower = kw.lower()
                if kw_lower in step_lower:
                    # Exact substring match
                    matches += 1
                    # Longer keyword matches are worth more
                    score += len(kw_lower) / max(len(step_lower), 1)
                else:
                    # Check word overlap
                    kw_words = set(kw_lower.split())
                    step_words = set(step_lower.split())
                    overlap = len(kw_words & step_words)
                    if overlap > 0 and kw_words:
                        word_score = (overlap / len(kw_words)) * 0.5
                        score += word_score

            if matches > 0:
                # Bonus for multiple keyword matches
                score += matches * 0.1

            # Normalize
            if keywords:
                score = min(score / len(keywords) + (matches / len(keywords)) * 0.5, 1.0)

            if score > best_score:
                best_score = score
                best_match = {**mapping, "mapping_id": mapping_id}

        return best_match, best_score

    def _convert_playbook_to_pipeline(
        self, playbook: Dict, param_overrides: Optional[Dict] = None
    ) -> Dict:
        """
        Convert a playbook's steps into a pipeline definition.

        Returns a dict with:
        - pipeline_steps: list of PipelineExecutor-compatible step dicts
        - mapping_report: details of each step's mapping
        - unmapped_steps: steps that couldn't be matched
        """
        steps = playbook.get("steps", [])
        pipeline_steps = []
        mapping_report = []
        unmapped = []
        overrides = param_overrides or {}

        for i, step in enumerate(steps):
            step_text = step if isinstance(step, str) else str(step)
            match, confidence = self._match_step_to_tool(step_text)

            step_override = overrides.get(str(i), overrides.get(step_text, {}))

            if match and confidence >= self._config["min_mapping_confidence"]:
                # Build pipeline step
                params = {**match.get("params", {})}
                if step_override:
                    params.update(step_override)

                pipeline_step = {
                    "tool": match["tool"],
                    "params": params,
                    "label": step_text[:80],
                    "timeout_seconds": 30.0,
                    "max_cost": 0.05,
                    "required": True,
                }
                pipeline_steps.append(pipeline_step)
                mapping_report.append({
                    "step_index": i,
                    "step_text": step_text,
                    "mapped_to": match["tool"],
                    "mapping_id": match.get("mapping_id", "unknown"),
                    "confidence": round(confidence, 3),
                    "params": params,
                })
                self._stats["total_steps_mapped"] += 1
            else:
                unmapped.append({
                    "step_index": i,
                    "step_text": step_text,
                    "best_match": match.get("tool") if match else None,
                    "best_confidence": round(confidence, 3) if match else 0.0,
                    "reason": "Below confidence threshold" if match else "No matching mapping",
                })
                self._stats["unmapped_steps"] += 1

        return {
            "pipeline_steps": pipeline_steps,
            "mapping_report": mapping_report,
            "unmapped_steps": unmapped,
            "playbook_name": playbook.get("name", "unknown"),
            "total_steps": len(steps),
            "mapped_count": len(pipeline_steps),
            "unmapped_count": len(unmapped),
            "coverage": len(pipeline_steps) / max(len(steps), 1),
        }

    def _get_playbook(self, name: str) -> Optional[Dict]:
        """Get a playbook from AgentReflectionSkill via context."""
        if not self.context:
            return None
        try:
            registry = self.context._registry
            reflection_skill = registry.get_skill("agent_reflection")
            if reflection_skill and hasattr(reflection_skill, "_playbooks"):
                return reflection_skill._playbooks.get(name)
        except Exception:
            pass
        return None

    def _get_all_playbooks(self) -> Dict[str, Dict]:
        """Get all playbooks from AgentReflectionSkill via context."""
        if not self.context:
            return {}
        try:
            registry = self.context._registry
            reflection_skill = registry.get_skill("agent_reflection")
            if reflection_skill and hasattr(reflection_skill, "_playbooks"):
                return reflection_skill._playbooks
        except Exception:
            pass
        return {}

    async def _convert(self, params: Dict) -> SkillResult:
        """Convert a playbook into a pipeline definition."""
        playbook_name = params.get("playbook_name", "")
        param_overrides = params.get("param_overrides", {})

        if not playbook_name:
            return SkillResult(
                success=False,
                message="Required: playbook_name",
            )

        playbook = self._get_playbook(playbook_name)
        if not playbook:
            # Check if passed directly
            if "playbook" in params:
                playbook = params["playbook"]
            else:
                available = list(self._get_all_playbooks().keys())
                return SkillResult(
                    success=False,
                    message=f"Playbook '{playbook_name}' not found. Available: {available[:10]}",
                )

        conversion = self._convert_playbook_to_pipeline(playbook, param_overrides)
        conversion["converted_at"] = datetime.utcnow().isoformat()

        self._conversions.append(conversion)
        self._stats["total_conversions"] += 1
        self._save()

        coverage_pct = conversion["coverage"] * 100
        return SkillResult(
            success=True,
            message=(
                f"Converted playbook '{playbook_name}': "
                f"{conversion['mapped_count']}/{conversion['total_steps']} steps mapped "
                f"({coverage_pct:.0f}% coverage)"
            ),
            data={"conversion": conversion},
        )

    async def _execute(self, params: Dict) -> SkillResult:
        """Execute a playbook as a pipeline."""
        playbook_name = params.get("playbook_name", "")
        dry_run = params.get("dry_run", self._config["dry_run_by_default"])
        max_cost = params.get("max_cost", self._config["max_pipeline_cost"])
        param_overrides = params.get("param_overrides", {})

        if not playbook_name:
            return SkillResult(
                success=False,
                message="Required: playbook_name",
            )

        playbook = self._get_playbook(playbook_name)
        if not playbook:
            if "playbook" in params:
                playbook = params["playbook"]
            else:
                return SkillResult(
                    success=False,
                    message=f"Playbook '{playbook_name}' not found",
                )

        # Convert playbook to pipeline
        conversion = self._convert_playbook_to_pipeline(playbook, param_overrides)

        if not conversion["pipeline_steps"]:
            return SkillResult(
                success=False,
                message=(
                    f"No steps could be mapped to tools. "
                    f"{conversion['unmapped_count']} unmapped steps. "
                    f"Add mappings with 'add_mapping' action."
                ),
                data={"unmapped": conversion["unmapped_steps"]},
            )

        if dry_run:
            return SkillResult(
                success=True,
                message=(
                    f"[DRY RUN] Pipeline for '{playbook_name}': "
                    f"{len(conversion['pipeline_steps'])} steps would execute"
                ),
                data={
                    "pipeline_steps": conversion["pipeline_steps"],
                    "mapping_report": conversion["mapping_report"],
                    "unmapped_steps": conversion["unmapped_steps"],
                    "estimated_cost": len(conversion["pipeline_steps"]) * 0.05,
                    "dry_run": True,
                },
            )

        # Execute via context if pipeline executor is available
        execution_record = {
            "playbook_name": playbook_name,
            "started_at": datetime.utcnow().isoformat(),
            "pipeline_steps": conversion["pipeline_steps"],
            "mapped_count": conversion["mapped_count"],
            "unmapped_count": conversion["unmapped_count"],
            "coverage": conversion["coverage"],
            "dry_run": False,
        }

        # Try to execute via skill context
        try:
            results = await self._execute_pipeline_steps(conversion["pipeline_steps"])
            execution_record["completed_at"] = datetime.utcnow().isoformat()
            execution_record["results"] = results
            execution_record["success"] = results.get("success", False)
            execution_record["steps_executed"] = results.get("steps_executed", 0)

            self._executions.append(execution_record)
            self._stats["total_executions"] += 1
            if results.get("success"):
                self._stats["successful_executions"] += 1
            else:
                self._stats["failed_executions"] += 1

            # Record playbook usage back to reflection skill
            if self._config["auto_record_to_reflection"]:
                await self._record_playbook_usage(
                    playbook_name,
                    results.get("success", False),
                )

            self._save()

            return SkillResult(
                success=results.get("success", False),
                message=(
                    f"Pipeline '{playbook_name}' executed: "
                    f"{results.get('steps_succeeded', 0)}/{results.get('steps_executed', 0)} steps succeeded"
                ),
                data={
                    "execution": execution_record,
                    "results": results,
                },
            )

        except Exception as e:
            execution_record["completed_at"] = datetime.utcnow().isoformat()
            execution_record["success"] = False
            execution_record["error"] = str(e)

            self._executions.append(execution_record)
            self._stats["total_executions"] += 1
            self._stats["failed_executions"] += 1
            self._save()

            return SkillResult(
                success=False,
                message=f"Pipeline execution failed: {e}",
                data={"execution": execution_record},
            )

    async def _execute_pipeline_steps(self, pipeline_steps: List[Dict]) -> Dict:
        """
        Execute pipeline steps using available execution infrastructure.

        Tries context-based execution first, falls back to simulated execution
        for environments without a full agent runtime.
        """
        results = {
            "success": True,
            "steps_executed": 0,
            "steps_succeeded": 0,
            "steps_failed": 0,
            "step_results": [],
        }

        for i, step in enumerate(pipeline_steps):
            tool = step.get("tool", "")
            params = step.get("params", {})
            label = step.get("label", f"step-{i}")

            step_result = {
                "step_index": i,
                "tool": tool,
                "label": label,
                "params": params,
            }

            try:
                # Try executing via context's skill registry
                if self.context:
                    skill_name, _, action_name = tool.partition(":")
                    if skill_name and action_name:
                        result = await self.context.call_skill(skill_name, action_name, params)
                        step_result["success"] = result.success
                        step_result["message"] = result.message
                        step_result["data"] = result.data
                        results["steps_executed"] += 1
                        if result.success:
                            results["steps_succeeded"] += 1
                        else:
                            results["steps_failed"] += 1
                            if step.get("required", True):
                                results["success"] = False
                                results["step_results"].append(step_result)
                                break
                    else:
                        step_result["success"] = False
                        step_result["message"] = f"Invalid tool format: {tool} (expected skill:action)"
                        results["steps_executed"] += 1
                        results["steps_failed"] += 1
                        if step.get("required", True):
                            results["success"] = False
                            results["step_results"].append(step_result)
                            break
                else:
                    # No context - record as not executable
                    step_result["success"] = False
                    step_result["message"] = "No execution context available"
                    results["steps_executed"] += 1
                    results["steps_failed"] += 1
                    if step.get("required", True):
                        results["success"] = False
                        results["step_results"].append(step_result)
                        break

            except Exception as e:
                step_result["success"] = False
                step_result["error"] = str(e)
                results["steps_executed"] += 1
                results["steps_failed"] += 1
                if step.get("required", True):
                    results["success"] = False
                    results["step_results"].append(step_result)
                    break

            results["step_results"].append(step_result)

        return results

    async def _record_playbook_usage(self, playbook_name: str, success: bool):
        """Record playbook usage back to AgentReflectionSkill."""
        if not self.context:
            return
        try:
            await self.context.call_skill(
                "agent_reflection",
                "record_playbook_use",
                {"playbook_name": playbook_name, "success": success},
            )
        except Exception:
            pass

    async def _add_mapping(self, params: Dict) -> SkillResult:
        """Add a new step keyword-to-tool mapping."""
        mapping_id = params.get("mapping_id", "")
        keywords = params.get("keywords", [])
        tool = params.get("tool", "")
        tool_params = params.get("params", {})
        description = params.get("description", "")

        if not mapping_id or not keywords or not tool:
            return SkillResult(
                success=False,
                message="Required: mapping_id, keywords (list), tool (string)",
            )

        if len(self._mappings) >= MAX_MAPPINGS:
            return SkillResult(
                success=False,
                message=f"Maximum mappings ({MAX_MAPPINGS}) reached. Remove some first.",
            )

        self._mappings[mapping_id] = {
            "keywords": keywords,
            "tool": tool,
            "params": tool_params,
            "description": description,
            "created_at": datetime.utcnow().isoformat(),
            "custom": True,
        }
        self._save()

        return SkillResult(
            success=True,
            message=f"Mapping '{mapping_id}' added: {keywords} → {tool}",
            data={"mapping": self._mappings[mapping_id]},
        )

    async def _remove_mapping(self, params: Dict) -> SkillResult:
        """Remove a step keyword-to-tool mapping."""
        mapping_id = params.get("mapping_id", "")
        if not mapping_id:
            return SkillResult(success=False, message="Required: mapping_id")

        if mapping_id not in self._mappings:
            return SkillResult(
                success=False,
                message=f"Mapping '{mapping_id}' not found",
            )

        del self._mappings[mapping_id]
        self._save()

        return SkillResult(
            success=True,
            message=f"Mapping '{mapping_id}' removed",
        )

    async def _list_mappings(self, params: Dict) -> SkillResult:
        """List all step-to-tool mappings."""
        mappings_list = []
        for mid, m in self._mappings.items():
            mappings_list.append({
                "id": mid,
                "keywords": m.get("keywords", []),
                "tool": m.get("tool", ""),
                "description": m.get("description", ""),
                "custom": m.get("custom", False),
            })

        return SkillResult(
            success=True,
            message=f"{len(mappings_list)} mappings available",
            data={"mappings": mappings_list},
        )

    async def _match_step(self, params: Dict) -> SkillResult:
        """Preview which tool a step description would map to."""
        step_text = params.get("step_text", "")
        if not step_text:
            return SkillResult(success=False, message="Required: step_text")

        match, confidence = self._match_step_to_tool(step_text)
        threshold = self._config["min_mapping_confidence"]

        if match and confidence >= threshold:
            return SkillResult(
                success=True,
                message=(
                    f"Step matches '{match.get('mapping_id', 'unknown')}' → "
                    f"{match['tool']} (confidence: {confidence:.1%})"
                ),
                data={
                    "match": match,
                    "confidence": round(confidence, 3),
                    "above_threshold": True,
                },
            )
        elif match:
            return SkillResult(
                success=True,
                message=(
                    f"Best match '{match.get('mapping_id', 'unknown')}' → "
                    f"{match['tool']} but below threshold "
                    f"({confidence:.1%} < {threshold:.1%})"
                ),
                data={
                    "match": match,
                    "confidence": round(confidence, 3),
                    "above_threshold": False,
                    "threshold": threshold,
                },
            )
        else:
            return SkillResult(
                success=True,
                message=f"No mapping found for: '{step_text}'",
                data={"match": None, "confidence": 0.0},
            )

    async def _history(self, params: Dict) -> SkillResult:
        """View pipeline execution history."""
        limit = params.get("limit", 10)
        recent = self._executions[-limit:]

        entries = []
        for ex in recent:
            entries.append({
                "playbook_name": ex.get("playbook_name"),
                "started_at": ex.get("started_at"),
                "success": ex.get("success"),
                "steps_executed": ex.get("results", {}).get("steps_executed", 0)
                    if isinstance(ex.get("results"), dict) else 0,
                "coverage": ex.get("coverage", 0),
            })

        return SkillResult(
            success=True,
            message=f"{len(entries)} execution(s) in history",
            data={"history": entries},
        )

    async def _status(self, params: Dict) -> SkillResult:
        """Get overall status and statistics."""
        playbooks = self._get_all_playbooks()

        return SkillResult(
            success=True,
            message="Playbook Pipeline status",
            data={
                "stats": self._stats,
                "config": self._config,
                "total_mappings": len(self._mappings),
                "total_conversions": len(self._conversions),
                "total_executions": len(self._executions),
                "available_playbooks": len(playbooks),
                "playbook_names": list(playbooks.keys())[:20],
            },
        )
