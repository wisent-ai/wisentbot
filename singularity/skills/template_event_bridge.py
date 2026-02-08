#!/usr/bin/env python3
"""
TemplateEventBridgeSkill - Wire WorkflowTemplateLibrary into EventDrivenWorkflowSkill.

This is the critical missing bridge between:
- WorkflowTemplateLibrarySkill (pre-built, parameterized workflow templates)
- EventDrivenWorkflowSkill (event-triggered autonomous execution)

Without this bridge, templates are static recipes and event workflows must be
hand-built. With this bridge, agents can:
1. Pick a template (e.g. "github_pr_review")
2. Fill in parameters (repo, review_depth, etc.)
3. Bind it to events (GitHub push webhook, Stripe payment, etc.)
4. Get a live, event-driven workflow that triggers automatically

Additional capabilities:
- Batch deploy: instantiate multiple templates at once for a use case
- Sync: re-sync a deployed workflow when its source template is updated
- Preview: see what the event-driven workflow would look like before deploying
- Catalog: list deployed template→workflow bindings with status

Actions:
1. DEPLOY       - Convert a template into a live event-driven workflow
2. DEPLOY_BATCH - Deploy multiple templates at once for a use case
3. PREVIEW      - Preview the conversion without deploying
4. SYNC         - Re-sync a deployed workflow from updated template
5. LIST         - List all template→workflow deployments
6. UNDEPLOY     - Remove a deployed template workflow
7. SUGGEST      - Suggest templates + event bindings for a use case

Pillars served:
- Revenue: Customers deploy automation with one command instead of building from scratch
- Self-Improvement: Agent learns which template+event combos work best
- Goal Setting: Suggests relevant automations for stated goals
"""

import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import Skill, SkillAction, SkillManifest, SkillResult

DATA_DIR = Path(__file__).parent.parent / "data"
BRIDGE_FILE = DATA_DIR / "template_event_bridge.json"


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _load_data(path: Path = None) -> Dict:
    p = path or BRIDGE_FILE
    if p.exists():
        try:
            with open(p) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {
        "deployments": {},
        "stats": {
            "total_deployments": 0,
            "total_syncs": 0,
            "total_batch_deploys": 0,
        },
    }


def _save_data(data: Dict, path: Path = None):
    p = path or BRIDGE_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(p) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, default=str)
    os.replace(tmp, str(p))


def _convert_template_steps(template: Dict, resolved_params: Dict) -> List[Dict]:
    """Convert WorkflowTemplateLibrary steps to EventDrivenWorkflow step format.

    Template steps look like:
        {"skill": "github", "action": "get_pr", "params_from": {"repo": "param.repo"}}

    EventDrivenWorkflow steps look like:
        {"name": "...", "skill_id": "...", "action": "...", "params": {...},
         "input_mapping": {...}, "event_mapping": {...}}
    """
    converted = []
    for i, step in enumerate(template.get("steps", [])):
        skill_id = step.get("skill", "")
        action = step.get("action", "")
        params_from = step.get("params_from", {})

        # Resolve parameter references
        static_params = {}
        input_mapping = {}
        event_mapping = {}

        for param_key, source in params_from.items():
            if not isinstance(source, str):
                static_params[param_key] = source
                continue

            if source.startswith("param."):
                # Reference to template parameter - resolve to actual value
                pkey = source[6:]
                if pkey in resolved_params:
                    static_params[param_key] = resolved_params[pkey]
                else:
                    static_params[param_key] = f"<unresolved:{pkey}>"
            elif source.startswith("step."):
                # Reference to previous step output: "step.0.diff" -> input_mapping
                parts = source.split(".", 2)
                if len(parts) >= 3:
                    step_idx = parts[1]
                    field_path = parts[2]
                    prev_step_id = f"step_{int(step_idx) + 1}"
                    input_mapping[f"{prev_step_id}.{field_path}"] = param_key
                else:
                    static_params[param_key] = source
            elif source.startswith("event.") or source.startswith("payload."):
                # Direct event payload reference
                event_mapping[source] = param_key
            else:
                static_params[param_key] = source

        step_id = f"step_{i + 1}"
        converted.append({
            "step_id": step_id,
            "name": f"{skill_id}:{action}",
            "skill_id": skill_id,
            "action": action,
            "params": static_params,
            "input_mapping": input_mapping,
            "event_mapping": event_mapping,
            "condition": {},
            "max_retries": 0,
            "retry_delay_seconds": 1.0,
            "continue_on_failure": False,
            "timeout_seconds": 0,
        })

    return converted


# Use-case suggestions: maps use case keywords to template+binding recommendations
USE_CASE_SUGGESTIONS = {
    "ci_cd": {
        "description": "Continuous integration and deployment automation",
        "templates": [
            {"template_id": "github_pr_review", "bindings": [{"source": "webhook", "pattern": "github-pr-*"}]},
            {"template_id": "deploy_on_merge", "bindings": [{"source": "webhook", "pattern": "github-push"}]},
        ],
    },
    "billing": {
        "description": "Payment processing and usage monitoring",
        "templates": [
            {"template_id": "stripe_payment_flow", "bindings": [{"source": "webhook", "pattern": "stripe-payment.*"}]},
            {"template_id": "usage_alert", "bindings": [{"source": "event_bus", "pattern": "usage.threshold.*"}]},
        ],
    },
    "monitoring": {
        "description": "Service health monitoring and incident response",
        "templates": [
            {"template_id": "health_check_pipeline", "bindings": [{"source": "event_bus", "pattern": "health.check.*"}]},
            {"template_id": "incident_response", "bindings": [{"source": "event_bus", "pattern": "incident.*"}]},
        ],
    },
    "onboarding": {
        "description": "Customer onboarding automation",
        "templates": [
            {"template_id": "customer_onboarding", "bindings": [{"source": "webhook", "pattern": "customer-signup"}]},
            {"template_id": "stripe_payment_flow", "bindings": [{"source": "webhook", "pattern": "stripe-payment.*"}]},
        ],
    },
    "content": {
        "description": "Content generation and publishing pipeline",
        "templates": [
            {"template_id": "content_pipeline", "bindings": [{"source": "event_bus", "pattern": "content.request.*"}]},
        ],
    },
    "devops": {
        "description": "DevOps automation: scaling, backups, and operations",
        "templates": [
            {"template_id": "scaling_decision", "bindings": [{"source": "event_bus", "pattern": "metrics.threshold.*"}]},
            {"template_id": "backup_verification", "bindings": [{"source": "event_bus", "pattern": "schedule.backup_check"}]},
        ],
    },
}


class TemplateEventBridgeSkill(Skill):
    """Bridge WorkflowTemplateLibrary templates into EventDrivenWorkflow execution."""

    def __init__(self, data_path: Path = None):
        super().__init__()
        self._data_path = data_path or BRIDGE_FILE

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="template_event_bridge",
            name="Template → Event Workflow Bridge",
            version="1.0.0",
            category="automation",
            description=(
                "Convert WorkflowTemplateLibrary templates into live "
                "EventDrivenWorkflow instances with event bindings. "
                "One-command deployment of pre-built automation templates."
            ),
            actions=[
                SkillAction(
                    name="deploy",
                    description="Deploy a template as a live event-driven workflow with event bindings",
                    parameters={
                        "template_id": {"type": "string", "required": True,
                                        "description": "Template ID from WorkflowTemplateLibrary"},
                        "params": {"type": "dict", "required": True,
                                   "description": "Template parameter values"},
                        "event_bindings": {"type": "list", "required": False,
                                           "description": "Event bindings: [{source, pattern}]"},
                        "workflow_name": {"type": "string", "required": False,
                                          "description": "Custom workflow name (default: template name)"},
                        "max_concurrent_runs": {"type": "integer", "required": False,
                                                "description": "Max simultaneous runs (default: 5)"},
                    },
                ),
                SkillAction(
                    name="deploy_batch",
                    description="Deploy multiple templates at once for a use case",
                    parameters={
                        "use_case": {"type": "string", "required": False,
                                     "description": "Use case: ci_cd, billing, monitoring, onboarding, content, devops"},
                        "deployments": {"type": "list", "required": False,
                                        "description": "Custom list of [{template_id, params, event_bindings}]"},
                    },
                ),
                SkillAction(
                    name="preview",
                    description="Preview the event-driven workflow that would be created without deploying",
                    parameters={
                        "template_id": {"type": "string", "required": True,
                                        "description": "Template ID to preview"},
                        "params": {"type": "dict", "required": False,
                                   "description": "Template parameter values"},
                        "event_bindings": {"type": "list", "required": False,
                                           "description": "Event bindings to preview"},
                    },
                ),
                SkillAction(
                    name="sync",
                    description="Re-sync a deployed workflow when its source template has been updated",
                    parameters={
                        "deployment_id": {"type": "string", "required": True,
                                          "description": "Deployment ID to sync"},
                        "params": {"type": "dict", "required": False,
                                   "description": "Updated parameter values (optional)"},
                    },
                ),
                SkillAction(
                    name="list",
                    description="List all template→workflow deployments",
                    parameters={
                        "template_id": {"type": "string", "required": False,
                                        "description": "Filter by source template"},
                        "status": {"type": "string", "required": False,
                                   "description": "Filter by status: active, paused, error"},
                    },
                ),
                SkillAction(
                    name="undeploy",
                    description="Remove a deployed template workflow",
                    parameters={
                        "deployment_id": {"type": "string", "required": True,
                                          "description": "Deployment ID to remove"},
                    },
                ),
                SkillAction(
                    name="suggest",
                    description="Suggest templates and event bindings for a use case or goal",
                    parameters={
                        "use_case": {"type": "string", "required": False,
                                     "description": "Use case keyword: ci_cd, billing, monitoring, etc."},
                        "goal": {"type": "string", "required": False,
                                 "description": "Free-text goal description"},
                    },
                ),
            ],
            required_credentials=[],
        )

    async def execute(self, action: str, params: Dict[str, Any]) -> SkillResult:
        data = _load_data(self._data_path)

        handlers = {
            "deploy": self._deploy,
            "deploy_batch": self._deploy_batch,
            "preview": self._preview,
            "sync": self._sync,
            "list": self._list,
            "undeploy": self._undeploy,
            "suggest": self._suggest,
        }

        handler = handlers.get(action)
        if not handler:
            return SkillResult(success=False, message=f"Unknown action: {action}")

        result = await handler(data, params)
        _save_data(data, self._data_path)
        return result

    async def _deploy(self, data: Dict, params: Dict) -> SkillResult:
        """Deploy a template as a live event-driven workflow."""
        template_id = params.get("template_id", "").strip()
        user_params = params.get("params", {})
        event_bindings = params.get("event_bindings", [])
        custom_name = params.get("workflow_name", "").strip()
        max_concurrent = params.get("max_concurrent_runs", 5)

        if not template_id:
            return SkillResult(success=False, message="template_id is required")

        # Fetch template from WorkflowTemplateLibrary
        template = await self._fetch_template(template_id)
        if not template:
            return SkillResult(
                success=False,
                message=f"Template '{template_id}' not found in WorkflowTemplateLibrary"
            )

        # Validate required parameters
        missing = []
        for pname, pdef in template.get("parameters", {}).items():
            if pdef.get("required", False) and pname not in user_params:
                missing.append(pname)

        if missing:
            return SkillResult(
                success=False,
                message=f"Missing required parameters: {missing}",
                data={"missing": missing, "template_parameters": template.get("parameters", {})},
            )

        # Resolve parameters with defaults
        resolved_params = {}
        for pname, pdef in template.get("parameters", {}).items():
            if pname in user_params:
                resolved_params[pname] = user_params[pname]
            elif "default" in pdef:
                resolved_params[pname] = pdef["default"]

        # Convert template steps to EventDrivenWorkflow format
        converted_steps = _convert_template_steps(template, resolved_params)

        workflow_name = custom_name or f"tpl:{template.get('name', template_id)}"

        # Create the event-driven workflow via skill context or record locally
        workflow_id = None
        if self.context:
            try:
                create_result = await self.context.call_skill(
                    "event_driven_workflow", "create_workflow", {
                        "name": workflow_name,
                        "description": f"Auto-deployed from template '{template_id}': {template.get('description', '')}",
                        "steps": converted_steps,
                        "event_bindings": event_bindings,
                        "max_concurrent_runs": max_concurrent,
                    }
                )
                if create_result.success:
                    workflow_id = (create_result.data or {}).get("workflow_id")
                else:
                    return SkillResult(
                        success=False,
                        message=f"Failed to create event workflow: {create_result.message}",
                    )
            except Exception as e:
                return SkillResult(
                    success=False,
                    message=f"Error creating event workflow: {str(e)[:150]}",
                )
        else:
            # No context - record locally with synthetic ID
            workflow_id = f"local_{uuid.uuid4().hex[:8]}"

        # Record deployment
        deployment_id = f"dep_{uuid.uuid4().hex[:10]}"
        deployment = {
            "id": deployment_id,
            "template_id": template_id,
            "template_name": template.get("name", template_id),
            "workflow_name": workflow_name,
            "workflow_id": workflow_id,
            "resolved_params": resolved_params,
            "event_bindings": event_bindings,
            "status": "active",
            "deployed_at": _now_iso(),
            "last_synced_at": _now_iso(),
            "template_version": template.get("version", "1.0.0"),
        }

        data["deployments"][deployment_id] = deployment
        data["stats"]["total_deployments"] += 1

        return SkillResult(
            success=True,
            message=f"Template '{template.get('name', template_id)}' deployed as event workflow '{workflow_name}' with {len(event_bindings)} binding(s)",
            data={
                "deployment_id": deployment_id,
                "workflow_name": workflow_name,
                "workflow_id": workflow_id,
                "steps_count": len(converted_steps),
                "event_bindings": event_bindings,
                "resolved_params": resolved_params,
            },
        )

    async def _deploy_batch(self, data: Dict, params: Dict) -> SkillResult:
        """Deploy multiple templates at once for a use case."""
        use_case = params.get("use_case", "").strip().lower()
        custom_deployments = params.get("deployments", [])

        if not use_case and not custom_deployments:
            return SkillResult(
                success=False,
                message="Provide either 'use_case' or 'deployments' list",
                data={"available_use_cases": list(USE_CASE_SUGGESTIONS.keys())},
            )

        deploy_specs = []
        if custom_deployments:
            deploy_specs = custom_deployments
        elif use_case in USE_CASE_SUGGESTIONS:
            suggestion = USE_CASE_SUGGESTIONS[use_case]
            deploy_specs = suggestion["templates"]
        else:
            return SkillResult(
                success=False,
                message=f"Unknown use case '{use_case}'. Available: {list(USE_CASE_SUGGESTIONS.keys())}",
            )

        results = []
        deployed = 0
        failed = 0

        for spec in deploy_specs:
            deploy_result = await self._deploy(data, {
                "template_id": spec.get("template_id", ""),
                "params": spec.get("params", {}),
                "event_bindings": spec.get("bindings", spec.get("event_bindings", [])),
                "workflow_name": spec.get("workflow_name", ""),
            })
            results.append({
                "template_id": spec.get("template_id", ""),
                "success": deploy_result.success,
                "message": deploy_result.message,
                "deployment_id": (deploy_result.data or {}).get("deployment_id"),
            })
            if deploy_result.success:
                deployed += 1
            else:
                failed += 1

        data["stats"]["total_batch_deploys"] += 1

        return SkillResult(
            success=deployed > 0,
            message=f"Batch deploy for '{use_case or 'custom'}': {deployed} succeeded, {failed} failed",
            data={
                "use_case": use_case,
                "total": len(deploy_specs),
                "deployed": deployed,
                "failed": failed,
                "results": results,
            },
        )

    async def _preview(self, data: Dict, params: Dict) -> SkillResult:
        """Preview the event-driven workflow that would be created."""
        template_id = params.get("template_id", "").strip()
        user_params = params.get("params", {})
        event_bindings = params.get("event_bindings", [])

        if not template_id:
            return SkillResult(success=False, message="template_id is required")

        template = await self._fetch_template(template_id)
        if not template:
            return SkillResult(
                success=False,
                message=f"Template '{template_id}' not found",
            )

        # Resolve parameters with defaults
        resolved_params = {}
        unresolved = []
        for pname, pdef in template.get("parameters", {}).items():
            if pname in user_params:
                resolved_params[pname] = user_params[pname]
            elif "default" in pdef:
                resolved_params[pname] = pdef["default"]
            else:
                unresolved.append(pname)
                resolved_params[pname] = f"<{pname}>"

        converted_steps = _convert_template_steps(template, resolved_params)

        preview = {
            "template_id": template_id,
            "template_name": template.get("name", template_id),
            "workflow_name": f"tpl:{template.get('name', template_id)}",
            "resolved_params": resolved_params,
            "unresolved_params": unresolved,
            "steps": converted_steps,
            "event_bindings": event_bindings,
            "required_skills": template.get("required_skills", []),
            "estimated_cost": template.get("estimated_cost", 0),
            "estimated_duration_seconds": template.get("estimated_duration_seconds", 0),
        }

        return SkillResult(
            success=True,
            message=f"Preview of '{template.get('name', template_id)}' → event workflow ({len(converted_steps)} steps)"
                    + (f", {len(unresolved)} unresolved params" if unresolved else ""),
            data={"preview": preview},
        )

    async def _sync(self, data: Dict, params: Dict) -> SkillResult:
        """Re-sync a deployed workflow when its source template has been updated."""
        deployment_id = params.get("deployment_id", "").strip()
        updated_params = params.get("params")

        if not deployment_id:
            return SkillResult(success=False, message="deployment_id is required")

        deployment = data["deployments"].get(deployment_id)
        if not deployment:
            return SkillResult(success=False, message=f"Deployment '{deployment_id}' not found")

        template_id = deployment["template_id"]
        template = await self._fetch_template(template_id)
        if not template:
            return SkillResult(
                success=False,
                message=f"Source template '{template_id}' no longer exists",
            )

        # Update parameters if provided
        resolved_params = dict(deployment["resolved_params"])
        if updated_params:
            resolved_params.update(updated_params)

        # Re-resolve with template defaults for any new parameters
        for pname, pdef in template.get("parameters", {}).items():
            if pname not in resolved_params and "default" in pdef:
                resolved_params[pname] = pdef["default"]

        # Convert steps
        converted_steps = _convert_template_steps(template, resolved_params)

        # Delete and recreate the event workflow
        workflow_name = deployment["workflow_name"]
        if self.context:
            try:
                # Delete old
                await self.context.call_skill(
                    "event_driven_workflow", "delete_workflow",
                    {"name": workflow_name}
                )
                # Create new
                create_result = await self.context.call_skill(
                    "event_driven_workflow", "create_workflow", {
                        "name": workflow_name,
                        "description": f"Auto-deployed from template '{template_id}' (synced): {template.get('description', '')}",
                        "steps": converted_steps,
                        "event_bindings": deployment.get("event_bindings", []),
                    }
                )
                if create_result.success:
                    deployment["workflow_id"] = (create_result.data or {}).get("workflow_id", deployment["workflow_id"])
            except Exception as e:
                return SkillResult(
                    success=False,
                    message=f"Error syncing workflow: {str(e)[:150]}",
                )

        # Update deployment record
        deployment["resolved_params"] = resolved_params
        deployment["last_synced_at"] = _now_iso()
        deployment["template_version"] = template.get("version", "1.0.0")
        data["stats"]["total_syncs"] += 1

        return SkillResult(
            success=True,
            message=f"Synced deployment '{deployment_id}' from template '{template_id}' v{template.get('version', '?')}",
            data={
                "deployment_id": deployment_id,
                "workflow_name": workflow_name,
                "steps_count": len(converted_steps),
                "params_updated": list(updated_params.keys()) if updated_params else [],
                "template_version": template.get("version", "1.0.0"),
            },
        )

    async def _list(self, data: Dict, params: Dict) -> SkillResult:
        """List all template→workflow deployments."""
        filter_template = params.get("template_id", "").strip()
        filter_status = params.get("status", "").strip()

        deployments = list(data["deployments"].values())

        if filter_template:
            deployments = [d for d in deployments if d["template_id"] == filter_template]
        if filter_status:
            deployments = [d for d in deployments if d.get("status") == filter_status]

        summaries = []
        for dep in sorted(deployments, key=lambda d: d.get("deployed_at", ""), reverse=True):
            summaries.append({
                "deployment_id": dep["id"],
                "template_id": dep["template_id"],
                "template_name": dep.get("template_name", ""),
                "workflow_name": dep["workflow_name"],
                "status": dep.get("status", "active"),
                "event_bindings": len(dep.get("event_bindings", [])),
                "deployed_at": dep.get("deployed_at", ""),
                "last_synced_at": dep.get("last_synced_at", ""),
                "template_version": dep.get("template_version", ""),
            })

        return SkillResult(
            success=True,
            message=f"Found {len(summaries)} deployment(s)",
            data={
                "deployments": summaries,
                "total": len(summaries),
                "stats": data["stats"],
            },
        )

    async def _undeploy(self, data: Dict, params: Dict) -> SkillResult:
        """Remove a deployed template workflow."""
        deployment_id = params.get("deployment_id", "").strip()

        if not deployment_id:
            return SkillResult(success=False, message="deployment_id is required")

        deployment = data["deployments"].get(deployment_id)
        if not deployment:
            return SkillResult(success=False, message=f"Deployment '{deployment_id}' not found")

        # Delete the event-driven workflow
        workflow_name = deployment["workflow_name"]
        if self.context:
            try:
                await self.context.call_skill(
                    "event_driven_workflow", "delete_workflow",
                    {"name": workflow_name}
                )
            except Exception:
                pass  # Best effort - it may already be deleted

        # Remove deployment record
        del data["deployments"][deployment_id]

        return SkillResult(
            success=True,
            message=f"Undeployed '{deployment.get('template_name', deployment_id)}' (workflow: {workflow_name})",
            data={
                "deployment_id": deployment_id,
                "template_id": deployment["template_id"],
                "workflow_name": workflow_name,
            },
        )

    async def _suggest(self, data: Dict, params: Dict) -> SkillResult:
        """Suggest templates and event bindings for a use case or goal."""
        use_case = params.get("use_case", "").strip().lower()
        goal = params.get("goal", "").strip().lower()

        if not use_case and not goal:
            return SkillResult(
                success=True,
                message="Available use cases for suggestions",
                data={
                    "use_cases": {
                        k: v["description"] for k, v in USE_CASE_SUGGESTIONS.items()
                    },
                    "hint": "Provide a 'use_case' key or a free-text 'goal'",
                },
            )

        suggestions = []

        if use_case and use_case in USE_CASE_SUGGESTIONS:
            suggestion = USE_CASE_SUGGESTIONS[use_case]
            suggestions.append({
                "use_case": use_case,
                "description": suggestion["description"],
                "templates": suggestion["templates"],
            })
        elif goal:
            # Match goal text against use case descriptions and template tags
            scored = []
            for uc_key, uc_data in USE_CASE_SUGGESTIONS.items():
                score = 0
                searchable = f"{uc_key} {uc_data['description']}".lower()
                for word in goal.split():
                    if len(word) > 2 and word in searchable:
                        score += 1
                # Also check template IDs
                for tpl in uc_data["templates"]:
                    if any(word in tpl["template_id"] for word in goal.split() if len(word) > 2):
                        score += 2
                if score > 0:
                    scored.append((score, uc_key, uc_data))

            scored.sort(key=lambda x: x[0], reverse=True)
            for score, uc_key, uc_data in scored[:3]:
                suggestions.append({
                    "use_case": uc_key,
                    "description": uc_data["description"],
                    "relevance_score": score,
                    "templates": uc_data["templates"],
                })

        if not suggestions:
            # Fallback: list all available use cases
            return SkillResult(
                success=True,
                message=f"No specific match for '{goal or use_case}'. Here are all available use cases:",
                data={
                    "use_cases": {
                        k: v["description"] for k, v in USE_CASE_SUGGESTIONS.items()
                    },
                    "templates_available": sum(
                        len(v["templates"]) for v in USE_CASE_SUGGESTIONS.values()
                    ),
                },
            )

        return SkillResult(
            success=True,
            message=f"Found {len(suggestions)} suggestion(s) for '{goal or use_case}'",
            data={
                "suggestions": suggestions,
                "deploy_hint": "Use deploy_batch with use_case to deploy all at once",
            },
        )

    async def _fetch_template(self, template_id: str) -> Optional[Dict]:
        """Fetch a template from WorkflowTemplateLibrary via SkillContext."""
        if self.context:
            try:
                result = await self.context.call_skill(
                    "workflow_templates", "get", {"template_id": template_id}
                )
                if result.success and result.data:
                    return result.data.get("template")
            except Exception:
                pass

        # Fallback: load templates directly from file
        from .workflow_templates import _load_data as load_templates, TEMPLATES_FILE
        tpl_data = load_templates(TEMPLATES_FILE)
        all_templates = dict(tpl_data.get("templates", {}))
        all_templates.update(tpl_data.get("custom_templates", {}))
        return all_templates.get(template_id)
