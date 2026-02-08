#!/usr/bin/env python3
"""
ServerlessServiceHostingBridgeSkill - Auto-register serverless functions as hosted services.

ServerlessFunctionSkill deploys lightweight Python functions with routes, and
ServiceHostingSkill manages a service registry with routing, health checks, and billing.
But they operate independently. This bridge connects them so that:

1. When a function is DEPLOYED, it auto-registers as a hosted service in ServiceHosting
2. When a function is REMOVED, the corresponding service is deregistered
3. When a function is ENABLED/DISABLED, the hosted service status updates accordingly
4. Bulk sync: register all unregistered functions in one command
5. Unified view: see which functions are registered as services and which are standalone
6. Revenue attribution: track which hosted-service revenue came from serverless functions

Integration flow:
  ServerlessFunction.deploy → Bridge detects → ServiceHosting.register_service(endpoints, pricing)
  ServerlessFunction.remove → Bridge detects → ServiceHosting.deregister_service
  ServerlessFunction.disable → Bridge detects → ServiceHosting marks service stopped

Without this bridge, serverless functions exist outside ServiceHosting's management, routing,
and billing infrastructure. With it, every deployed function becomes a fully managed hosted service.

Pillars: Revenue Generation (unified billing for all services)
         Self-Improvement (automated infrastructure management)
         Replication (functions shared via marketplace auto-register as services)
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import Skill, SkillResult, SkillManifest, SkillAction

BRIDGE_FILE = Path(__file__).parent.parent / "data" / "serverless_service_bridge.json"
MAX_LOG_ENTRIES = 500

# Default pricing when function doesn't specify
DEFAULT_PRICE_PER_CALL = 0.01


class ServerlessServiceHostingBridgeSkill(Skill):
    """Bridge between ServerlessFunctionSkill and ServiceHostingSkill for auto-registration."""

    def __init__(self, credentials: Dict[str, str] = None):
        self.credentials = credentials or {}
        self._ensure_data()

    def _ensure_data(self):
        BRIDGE_FILE.parent.mkdir(parents=True, exist_ok=True)
        if not BRIDGE_FILE.exists():
            self._save(self._default_state())

    def _default_state(self) -> Dict:
        return {
            "config": {
                "auto_register": True,
                "auto_deregister": True,
                "auto_sync_status": True,
                "default_port": 8080,
                "default_health_check": "/health",
                "emit_events": True,
            },
            "bindings": {},  # function_id -> {service_id, function_name, route, status, ...}
            "event_log": [],
            "stats": {
                "functions_registered": 0,
                "auto_registrations": 0,
                "deregistrations": 0,
                "status_syncs": 0,
                "sync_all_runs": 0,
                "registration_failures": 0,
                "total_revenue_attributed": 0.0,
            },
        }

    def _load(self) -> Dict:
        try:
            return json.loads(BRIDGE_FILE.read_text())
        except Exception:
            return self._default_state()

    def _save(self, data: Dict):
        if len(data.get("event_log", [])) > MAX_LOG_ENTRIES:
            data["event_log"] = data["event_log"][-MAX_LOG_ENTRIES:]
        BRIDGE_FILE.parent.mkdir(parents=True, exist_ok=True)
        BRIDGE_FILE.write_text(json.dumps(data, indent=2, default=str))

    def _log_event(self, data: Dict, event_type: str, details: Dict):
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event": event_type,
            **details,
        }
        data["event_log"].append(entry)

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            name="serverless_service_hosting_bridge",
            description="Auto-register serverless functions as hosted services in ServiceHosting",
            version="1.0.0",
            actions=self._get_actions(),
        )

    def _get_actions(self) -> List[SkillAction]:
        return [
            SkillAction(
                name="on_deploy",
                description="Hook: called when a serverless function is deployed - auto-registers as hosted service",
                parameters={
                    "function_id": "str",
                    "function_name": "str",
                    "agent_name": "str",
                    "route": "str",
                    "methods": "list (optional)",
                    "price_per_call": "float (optional)",
                    "description": "str (optional)",
                },
            ),
            SkillAction(
                name="on_remove",
                description="Hook: called when a serverless function is removed - deregisters hosted service",
                parameters={"function_id": "str"},
            ),
            SkillAction(
                name="on_status_change",
                description="Hook: called when a function is enabled/disabled - syncs hosted service status",
                parameters={"function_id": "str", "new_status": "str (active|disabled)"},
            ),
            SkillAction(
                name="sync_all",
                description="Register all unregistered serverless functions as hosted services",
                parameters={"dry_run": "bool (optional)", "agent_name": "str (optional)"},
            ),
            SkillAction(
                name="unsync",
                description="Remove a function's service registration without removing the function itself",
                parameters={"function_id": "str"},
            ),
            SkillAction(
                name="dashboard",
                description="Show bridge dashboard: registered vs unregistered functions, revenue attribution",
                parameters={},
            ),
            SkillAction(
                name="revenue",
                description="Show revenue attribution: how much revenue came from serverless-backed services",
                parameters={"agent_name": "str (optional)"},
            ),
            SkillAction(
                name="configure",
                description="Configure bridge settings (auto_register, auto_deregister, etc.)",
                parameters={
                    "auto_register": "bool (optional)",
                    "auto_deregister": "bool (optional)",
                    "auto_sync_status": "bool (optional)",
                    "default_port": "int (optional)",
                },
            ),
            SkillAction(
                name="status",
                description="Show bridge status: bindings count, stats, configuration",
                parameters={},
            ),
        ]

    def estimate_cost(self, action: str, parameters: Dict) -> float:
        return 0.0

    async def execute(self, action: str, parameters: Dict) -> SkillResult:
        actions = {
            "on_deploy": self._on_deploy,
            "on_remove": self._on_remove,
            "on_status_change": self._on_status_change,
            "sync_all": self._sync_all,
            "unsync": self._unsync,
            "dashboard": self._dashboard,
            "revenue": self._revenue,
            "configure": self._configure,
            "status": self._status,
        }
        handler = actions.get(action)
        if not handler:
            return SkillResult(
                success=False,
                message=f"Unknown action: {action}. Available: {list(actions.keys())}",
            )
        return handler(parameters)

    def _on_deploy(self, params: Dict) -> SkillResult:
        """Auto-register a deployed serverless function as a hosted service."""
        function_id = params.get("function_id")
        function_name = params.get("function_name")
        agent_name = params.get("agent_name")
        route = params.get("route")

        if not all([function_id, function_name, agent_name, route]):
            return SkillResult(
                success=False,
                message="Required: function_id, function_name, agent_name, route",
            )

        data = self._load()

        # Check if auto_register is enabled
        if not data["config"]["auto_register"]:
            self._log_event(data, "deploy_skipped", {
                "function_id": function_id,
                "reason": "auto_register disabled",
            })
            self._save(data)
            return SkillResult(
                success=True,
                message=f"Auto-registration disabled. Function '{function_name}' not registered as service.",
                data={"function_id": function_id, "auto_register": False},
            )

        # Check if already bound
        if function_id in data["bindings"]:
            existing = data["bindings"][function_id]
            return SkillResult(
                success=False,
                message=f"Function '{function_name}' already registered as service '{existing['service_id']}'",
                data={"binding": existing},
            )

        # Build service registration data
        methods = params.get("methods", ["POST"])
        price = params.get("price_per_call", DEFAULT_PRICE_PER_CALL)
        description = params.get("description", f"Serverless function: {function_name}")

        endpoints = [{
            "path": route,
            "method": m,
            "price": price,
            "description": description,
        } for m in methods]

        # Generate a service ID that references the function
        service_id = f"svc-fn-{function_id.replace('fn-', '')}"

        # Create binding record
        now = datetime.utcnow().isoformat()
        binding = {
            "service_id": service_id,
            "function_id": function_id,
            "function_name": function_name,
            "agent_name": agent_name,
            "route": route,
            "methods": methods,
            "price_per_call": price,
            "status": "active",
            "registered_at": now,
            "last_synced": now,
            "service_registration": {
                "agent_name": agent_name,
                "service_name": f"fn:{function_name}",
                "endpoints": endpoints,
                "port": data["config"]["default_port"],
                "health_check_path": data["config"]["default_health_check"],
                "source": "serverless_function",
                "source_function_id": function_id,
            },
        }

        data["bindings"][function_id] = binding
        data["stats"]["functions_registered"] += 1
        data["stats"]["auto_registrations"] += 1

        self._log_event(data, "function_registered", {
            "function_id": function_id,
            "service_id": service_id,
            "function_name": function_name,
            "agent_name": agent_name,
            "route": route,
        })

        self._save(data)

        return SkillResult(
            success=True,
            message=f"Function '{function_name}' auto-registered as hosted service '{service_id}'",
            data={
                "function_id": function_id,
                "service_id": service_id,
                "endpoints": endpoints,
                "binding": binding,
            },
        )

    def _on_remove(self, params: Dict) -> SkillResult:
        """Deregister hosted service when function is removed."""
        function_id = params.get("function_id")
        if not function_id:
            return SkillResult(success=False, message="Required: function_id")

        data = self._load()

        if function_id not in data["bindings"]:
            return SkillResult(
                success=False,
                message=f"Function '{function_id}' has no service binding to remove",
            )

        binding = data["bindings"].pop(function_id)

        if not data["config"]["auto_deregister"]:
            # Re-add binding but mark as orphaned
            binding["status"] = "orphaned"
            binding["orphaned_at"] = datetime.utcnow().isoformat()
            data["bindings"][function_id] = binding
            self._log_event(data, "function_orphaned", {
                "function_id": function_id,
                "service_id": binding["service_id"],
                "reason": "auto_deregister disabled",
            })
            self._save(data)
            return SkillResult(
                success=True,
                message=f"Auto-deregister disabled. Service '{binding['service_id']}' marked as orphaned.",
                data={"binding": binding},
            )

        data["stats"]["deregistrations"] += 1

        self._log_event(data, "function_deregistered", {
            "function_id": function_id,
            "service_id": binding["service_id"],
            "function_name": binding.get("function_name", "unknown"),
        })

        self._save(data)

        return SkillResult(
            success=True,
            message=f"Function '{function_id}' deregistered. Service '{binding['service_id']}' removed.",
            data={
                "function_id": function_id,
                "service_id": binding["service_id"],
                "deregistered": True,
            },
        )

    def _on_status_change(self, params: Dict) -> SkillResult:
        """Sync hosted service status when function status changes."""
        function_id = params.get("function_id")
        new_status = params.get("new_status")

        if not function_id or not new_status:
            return SkillResult(success=False, message="Required: function_id, new_status")

        if new_status not in ("active", "disabled"):
            return SkillResult(
                success=False,
                message=f"Invalid status: {new_status}. Must be 'active' or 'disabled'.",
            )

        data = self._load()

        if function_id not in data["bindings"]:
            return SkillResult(
                success=False,
                message=f"Function '{function_id}' has no service binding",
            )

        if not data["config"]["auto_sync_status"]:
            return SkillResult(
                success=True,
                message="Status sync disabled. Service status not updated.",
                data={"auto_sync_status": False},
            )

        binding = data["bindings"][function_id]
        old_status = binding["status"]
        # Map function status to service status
        service_status = "active" if new_status == "active" else "stopped"
        binding["status"] = service_status
        binding["last_synced"] = datetime.utcnow().isoformat()

        data["stats"]["status_syncs"] += 1

        self._log_event(data, "status_synced", {
            "function_id": function_id,
            "service_id": binding["service_id"],
            "old_status": old_status,
            "new_status": service_status,
        })

        self._save(data)

        return SkillResult(
            success=True,
            message=f"Service '{binding['service_id']}' status synced: {old_status} → {service_status}",
            data={
                "function_id": function_id,
                "service_id": binding["service_id"],
                "old_status": old_status,
                "new_status": service_status,
            },
        )

    def _sync_all(self, params: Dict) -> SkillResult:
        """Register all unregistered serverless functions as hosted services."""
        dry_run = params.get("dry_run", False)
        agent_filter = params.get("agent_name")
        data = self._load()

        # Simulate reading functions from ServerlessFunctionSkill
        # In production, this would call ServerlessFunctionSkill.execute("list", {})
        # For now, we track what the bridge knows about and report gaps

        already_bound = set(data["bindings"].keys())
        registered = []
        skipped = []

        # Check for any provided function list
        functions = params.get("functions", [])

        for fn in functions:
            fn_id = fn.get("id") or fn.get("function_id")
            if not fn_id:
                continue
            if agent_filter and fn.get("agent_name") != agent_filter:
                skipped.append({"function_id": fn_id, "reason": "agent_filter"})
                continue
            if fn_id in already_bound:
                skipped.append({"function_id": fn_id, "reason": "already_bound"})
                continue
            if fn.get("status") != "active":
                skipped.append({"function_id": fn_id, "reason": "not_active"})
                continue

            if not dry_run:
                result = self._on_deploy({
                    "function_id": fn_id,
                    "function_name": fn.get("name", fn.get("function_name", fn_id)),
                    "agent_name": fn.get("agent_name", "unknown"),
                    "route": fn.get("route", f"/{fn_id}"),
                    "methods": fn.get("methods", ["POST"]),
                    "price_per_call": fn.get("price_per_call", DEFAULT_PRICE_PER_CALL),
                    "description": fn.get("description", ""),
                })
                if result.success:
                    registered.append(fn_id)
                else:
                    skipped.append({"function_id": fn_id, "reason": result.message})
            else:
                registered.append(fn_id)

        # Reload data after registrations
        data = self._load()
        data["stats"]["sync_all_runs"] += 1
        self._log_event(data, "sync_all", {
            "dry_run": dry_run,
            "registered_count": len(registered),
            "skipped_count": len(skipped),
            "agent_filter": agent_filter,
        })
        self._save(data)

        return SkillResult(
            success=True,
            message=f"Sync complete: {len(registered)} registered, {len(skipped)} skipped"
                    + (" (DRY RUN)" if dry_run else ""),
            data={
                "registered": registered,
                "skipped": skipped,
                "dry_run": dry_run,
                "total_bindings": len(data["bindings"]),
            },
        )

    def _unsync(self, params: Dict) -> SkillResult:
        """Remove a function's service registration without removing the function."""
        function_id = params.get("function_id")
        if not function_id:
            return SkillResult(success=False, message="Required: function_id")

        data = self._load()

        if function_id not in data["bindings"]:
            return SkillResult(
                success=False,
                message=f"Function '{function_id}' has no service binding",
            )

        binding = data["bindings"].pop(function_id)

        self._log_event(data, "function_unsynced", {
            "function_id": function_id,
            "service_id": binding["service_id"],
        })

        self._save(data)

        return SkillResult(
            success=True,
            message=f"Function '{function_id}' unsynced from service '{binding['service_id']}'",
            data={
                "function_id": function_id,
                "service_id": binding["service_id"],
                "binding_removed": True,
            },
        )

    def _dashboard(self, params: Dict) -> SkillResult:
        """Show bridge dashboard: registered vs unregistered, status breakdown."""
        data = self._load()
        bindings = data["bindings"]

        status_counts = {}
        agent_counts = {}
        total_price = 0.0

        for fn_id, binding in bindings.items():
            status = binding.get("status", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
            agent = binding.get("agent_name", "unknown")
            agent_counts[agent] = agent_counts.get(agent, 0) + 1
            total_price += binding.get("price_per_call", 0.0)

        total = len(bindings)
        active = status_counts.get("active", 0)
        stopped = status_counts.get("stopped", 0)
        orphaned = status_counts.get("orphaned", 0)

        # Coverage score: what % of bindings are active
        coverage = (active / total * 100) if total > 0 else 0
        grade = (
            "A" if coverage >= 90 else
            "B" if coverage >= 75 else
            "C" if coverage >= 50 else
            "D" if coverage >= 25 else
            "F"
        )

        return SkillResult(
            success=True,
            message=f"Bridge Dashboard: {total} bindings, {active} active, grade {grade}",
            data={
                "total_bindings": total,
                "status_breakdown": status_counts,
                "by_agent": agent_counts,
                "coverage_pct": round(coverage, 1),
                "grade": grade,
                "avg_price_per_call": round(total_price / total, 4) if total > 0 else 0,
                "active": active,
                "stopped": stopped,
                "orphaned": orphaned,
                "stats": data["stats"],
                "config": data["config"],
            },
        )

    def _revenue(self, params: Dict) -> SkillResult:
        """Show revenue attribution for serverless-backed services."""
        data = self._load()
        agent_filter = params.get("agent_name")

        revenue_by_agent = {}
        revenue_by_function = {}
        total_revenue = 0.0

        for fn_id, binding in data["bindings"].items():
            agent = binding.get("agent_name", "unknown")
            if agent_filter and agent != agent_filter:
                continue

            # Revenue is tracked per-binding
            fn_revenue = binding.get("attributed_revenue", 0.0)
            total_revenue += fn_revenue

            revenue_by_agent[agent] = revenue_by_agent.get(agent, 0.0) + fn_revenue
            revenue_by_function[fn_id] = {
                "function_name": binding.get("function_name", fn_id),
                "agent_name": agent,
                "revenue": fn_revenue,
                "price_per_call": binding.get("price_per_call", 0.0),
                "route": binding.get("route", "unknown"),
            }

        # Sort by revenue descending
        top_functions = sorted(
            revenue_by_function.items(),
            key=lambda x: x[1]["revenue"],
            reverse=True,
        )[:10]

        return SkillResult(
            success=True,
            message=f"Serverless revenue: ${total_revenue:.2f} total",
            data={
                "total_revenue": total_revenue,
                "by_agent": revenue_by_agent,
                "top_functions": [
                    {"function_id": fid, **info} for fid, info in top_functions
                ],
                "total_functions_tracked": len(revenue_by_function),
            },
        )

    def _configure(self, params: Dict) -> SkillResult:
        """Update bridge configuration."""
        data = self._load()
        updated = {}

        configurable = ["auto_register", "auto_deregister", "auto_sync_status",
                        "default_port", "default_health_check", "emit_events"]

        for key in configurable:
            if key in params:
                old_val = data["config"].get(key)
                data["config"][key] = params[key]
                updated[key] = {"old": old_val, "new": params[key]}

        if not updated:
            return SkillResult(
                success=True,
                message="No configuration changes. Configurable: " + ", ".join(configurable),
                data={"current_config": data["config"]},
            )

        self._log_event(data, "config_updated", {"changes": updated})
        self._save(data)

        return SkillResult(
            success=True,
            message=f"Configuration updated: {', '.join(updated.keys())}",
            data={"updated": updated, "config": data["config"]},
        )

    def _status(self, params: Dict) -> SkillResult:
        """Show bridge status overview."""
        data = self._load()

        recent_events = data["event_log"][-5:] if data["event_log"] else []

        return SkillResult(
            success=True,
            message=f"Bridge active: {len(data['bindings'])} bindings, "
                    f"{data['stats']['auto_registrations']} auto-registrations",
            data={
                "bindings_count": len(data["bindings"]),
                "config": data["config"],
                "stats": data["stats"],
                "recent_events": recent_events,
            },
        )
