#!/usr/bin/env python3
"""
Auto Catalog Skill - Automatically registers services into the marketplace.

This is the missing bridge between value production (RevenueServiceSkill) and
business management (MarketplaceSkill). Without this, services exist but are
never listed for sale. AutoCatalogSkill ensures every sellable service is
automatically registered, health-monitored, and kept in sync.

Pipeline: RevenueServiceSkill (value) → AutoCatalogSkill (bridge) → MarketplaceSkill (sales)

Capabilities:
  - Auto-register: Scan all skills for sellable actions, register them in marketplace
  - Health monitor: Track service success rates, auto-pause unreliable services
  - Sync: Keep marketplace catalog in sync with available skills
  - Pricing suggestions: Recommend prices based on cost, complexity, market data

Part of the Revenue Generation pillar: completes the service-to-sale pipeline.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from .base import Skill, SkillResult, SkillManifest, SkillAction


# Persistent catalog state
CATALOG_FILE = Path(__file__).parent.parent / "data" / "auto_catalog.json"

# Skills that should never be listed for sale (internal/meta skills)
INTERNAL_SKILLS = frozenset({
    "marketplace", "strategy", "self_eval", "self_modify", "replication",
    "auto_catalog", "feedback_loop", "knowledge_sharing", "performance_tracker",
    "task_delegator", "orchestrator", "planner", "scheduler", "event",
    "steering", "memory", "workflow", "goal_manager",
})

# Skills whose actions are explicitly sellable services
REVENUE_SKILLS = frozenset({
    "revenue_services",
})

# Internal actions that shouldn't be sold (stats, config, etc.)
INTERNAL_ACTIONS = frozenset({
    "service_stats", "status", "config", "health", "list",
})

# Default pricing for revenue services (curated, human-friendly names)
SERVICE_CATALOG = {
    ("revenue_services", "code_review"): {
        "name": "AI Code Review",
        "description": "Professional code review analyzing security vulnerabilities, bugs, style issues, and performance anti-patterns. Supports Python, JavaScript, TypeScript, Go, Rust, Java, and Ruby.",
        "price": 0.10,
        "tags": ["code", "review", "security", "quality"],
        "sla_minutes": 5,
    },
    ("revenue_services", "summarize_text"): {
        "name": "Text Summarization",
        "description": "Condense long documents into clear key points. Supports bullet, paragraph, and executive summary styles.",
        "price": 0.05,
        "tags": ["text", "summary", "content"],
        "sla_minutes": 3,
    },
    ("revenue_services", "analyze_data"): {
        "name": "Data Analysis",
        "description": "Extract insights from structured JSON data. Field analysis, statistics, null detection, and pattern recognition.",
        "price": 0.10,
        "tags": ["data", "analysis", "insights", "statistics"],
        "sla_minutes": 5,
    },
    ("revenue_services", "seo_audit"): {
        "name": "SEO Content Audit",
        "description": "Analyze content for search engine optimization. Keyword density, readability metrics, and improvement suggestions.",
        "price": 0.05,
        "tags": ["seo", "content", "marketing"],
        "sla_minutes": 3,
    },
    ("revenue_services", "generate_api_docs"): {
        "name": "API Documentation Generator",
        "description": "Generate professional API docs from code or endpoint descriptions. Supports Markdown and OpenAPI 3.0 formats.",
        "price": 0.10,
        "tags": ["api", "documentation", "developer-tools"],
        "sla_minutes": 5,
    },
}


class AutoCatalogSkill(Skill):
    """
    Automatically registers and manages services in the marketplace.

    Bridges the gap between service implementations and the marketplace,
    ensuring all sellable services are discoverable, priced, and monitored.
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._catalog_state: Dict = {}
        self._load_state()

    def _load_state(self):
        try:
            if CATALOG_FILE.exists():
                with open(CATALOG_FILE, "r") as f:
                    self._catalog_state = json.load(f)
        except (json.JSONDecodeError, IOError):
            self._catalog_state = {}
        if not self._catalog_state:
            self._catalog_state = {
                "registered_services": [],
                "health_checks": [],
                "sync_history": [],
                "created_at": datetime.now().isoformat(),
            }

    def _save_state(self):
        CATALOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        # Trim histories
        for key in ("health_checks", "sync_history"):
            if len(self._catalog_state.get(key, [])) > 200:
                self._catalog_state[key] = self._catalog_state[key][-200:]
        try:
            with open(CATALOG_FILE, "w") as f:
                json.dump(self._catalog_state, f, indent=2, default=str)
        except IOError:
            pass

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="auto_catalog",
            name="Auto Catalog",
            version="1.0.0",
            category="revenue",
            description="Automatically registers services into the marketplace, monitors health, and keeps catalog in sync",
            required_credentials=[],
            install_cost=0,
            actions=[
                SkillAction(
                    name="auto_register",
                    description="Scan all skills and auto-register sellable services in the marketplace",
                    parameters={
                        "force": {"type": "boolean", "required": False, "description": "Re-register even if already registered"},
                        "dry_run": {"type": "boolean", "required": False, "description": "Preview what would be registered without actually doing it"},
                    },
                    estimated_cost=0,
                    estimated_duration_seconds=2,
                    success_probability=0.95,
                ),
                SkillAction(
                    name="health_check",
                    description="Check health of all registered services and auto-pause unreliable ones",
                    parameters={
                        "min_success_rate": {"type": "number", "required": False, "description": "Minimum success rate to stay active (default 0.7)"},
                        "min_executions": {"type": "number", "required": False, "description": "Minimum executions before evaluating (default 5)"},
                    },
                    estimated_cost=0,
                    estimated_duration_seconds=2,
                    success_probability=0.95,
                ),
                SkillAction(
                    name="sync_catalog",
                    description="Sync marketplace catalog with currently available skills (add new, remove stale)",
                    parameters={},
                    estimated_cost=0,
                    estimated_duration_seconds=3,
                    success_probability=0.95,
                ),
                SkillAction(
                    name="catalog_status",
                    description="Get status of auto-cataloged services including health metrics",
                    parameters={},
                    estimated_cost=0,
                    estimated_duration_seconds=1,
                    success_probability=1.0,
                ),
            ],
        )

    async def execute(self, action: str, params: Dict) -> SkillResult:
        actions = {
            "auto_register": self._auto_register,
            "health_check": self._health_check,
            "sync_catalog": self._sync_catalog,
            "catalog_status": self._catalog_status,
        }

        if action not in actions:
            return SkillResult(
                success=False,
                message=f"Unknown action: {action}. Available: {list(actions.keys())}",
            )

        try:
            return await actions[action](params)
        except Exception as e:
            return SkillResult(
                success=False,
                message=f"AutoCatalog error: {str(e)}",
            )

    async def _auto_register(self, params: Dict) -> SkillResult:
        """Scan skills and register sellable services in the marketplace."""
        if not self.context:
            return SkillResult(
                success=False,
                message="No skill context available. Cannot access marketplace or other skills.",
            )

        marketplace = self.context.get_skill("marketplace")
        if not marketplace:
            return SkillResult(
                success=False,
                message="MarketplaceSkill not installed. Cannot register services.",
            )

        force = params.get("force", False)
        dry_run = params.get("dry_run", False)

        registered = []
        skipped = []
        failed = []

        # Get services to register from the curated catalog
        for (skill_id, action_name), service_def in SERVICE_CATALOG.items():
            # Verify the skill actually exists
            skill = self.context.get_skill(skill_id)
            if not skill:
                skipped.append({
                    "skill_id": skill_id,
                    "action": action_name,
                    "reason": "skill not installed",
                })
                continue

            # Verify the action exists on the skill
            action_exists = any(a.name == action_name for a in skill.get_actions())
            if not action_exists:
                skipped.append({
                    "skill_id": skill_id,
                    "action": action_name,
                    "reason": "action not found on skill",
                })
                continue

            if dry_run:
                registered.append({
                    "name": service_def["name"],
                    "skill_id": skill_id,
                    "action": action_name,
                    "price": service_def["price"],
                    "dry_run": True,
                })
                continue

            # Register in marketplace
            create_params = {
                "name": service_def["name"],
                "description": service_def["description"],
                "skill_id": skill_id,
                "action": action_name,
                "price": service_def["price"],
                "tags": service_def.get("tags", []),
                "sla_minutes": service_def.get("sla_minutes", 60),
            }

            result = await marketplace.execute("create_service", create_params)

            if result.success:
                registered.append({
                    "name": service_def["name"],
                    "skill_id": skill_id,
                    "action": action_name,
                    "price": service_def["price"],
                    "service_id": result.data.get("service", {}).get("id") if result.data else None,
                })
            elif "already exists" in result.message.lower() and not force:
                skipped.append({
                    "skill_id": skill_id,
                    "action": action_name,
                    "reason": "already registered",
                })
            else:
                failed.append({
                    "skill_id": skill_id,
                    "action": action_name,
                    "error": result.message,
                })

        # Also discover additional sellable actions from revenue skills
        additional = await self._discover_additional_services(marketplace, force, dry_run)
        registered.extend(additional.get("registered", []))
        skipped.extend(additional.get("skipped", []))

        # Record this registration event
        self._catalog_state["registered_services"] = registered
        self._catalog_state["last_register"] = datetime.now().isoformat()
        self._save_state()

        total = len(registered)
        msg_parts = [f"Auto-registered {total} service(s)"]
        if skipped:
            msg_parts.append(f"{len(skipped)} skipped")
        if failed:
            msg_parts.append(f"{len(failed)} failed")
        if dry_run:
            msg_parts.append("(dry run)")

        return SkillResult(
            success=True,
            message=". ".join(msg_parts),
            data={
                "registered": registered,
                "skipped": skipped,
                "failed": failed,
                "total_registered": total,
            },
        )

    async def _discover_additional_services(
        self, marketplace: Skill, force: bool, dry_run: bool
    ) -> Dict:
        """Discover sellable services beyond the curated catalog."""
        registered = []
        skipped = []

        if not self.context:
            return {"registered": registered, "skipped": skipped}

        curated_keys = set(SERVICE_CATALOG.keys())

        for skill_id in self.context.list_skills():
            if skill_id in INTERNAL_SKILLS:
                continue

            skill = self.context.get_skill(skill_id)
            if not skill:
                continue

            # Only auto-register from revenue-focused skills (beyond curated)
            if skill_id not in REVENUE_SKILLS:
                continue

            for action in skill.get_actions():
                key = (skill_id, action.name)
                if key in curated_keys:
                    continue  # Already handled by curated catalog
                if action.name in INTERNAL_ACTIONS:
                    continue

                if dry_run:
                    registered.append({
                        "name": f"{skill.manifest.name}: {action.name}",
                        "skill_id": skill_id,
                        "action": action.name,
                        "price": max(action.estimated_cost * 3, 0.10),
                        "dry_run": True,
                        "source": "auto_discovered",
                    })
                    continue

                price = max(action.estimated_cost * 3, 0.10)
                create_params = {
                    "name": f"{skill.manifest.name}: {action.name}",
                    "description": action.description,
                    "skill_id": skill_id,
                    "action": action.name,
                    "price": round(price, 2),
                    "sla_minutes": max(int(action.estimated_duration_seconds / 60) + 1, 1),
                }

                result = await marketplace.execute("create_service", create_params)
                if result.success:
                    registered.append({
                        "name": create_params["name"],
                        "skill_id": skill_id,
                        "action": action.name,
                        "price": price,
                        "source": "auto_discovered",
                    })
                elif "already exists" not in result.message.lower():
                    skipped.append({
                        "skill_id": skill_id,
                        "action": action.name,
                        "reason": result.message,
                    })

        return {"registered": registered, "skipped": skipped}

    async def _health_check(self, params: Dict) -> SkillResult:
        """Check health of registered services and auto-pause unreliable ones."""
        if not self.context:
            return SkillResult(
                success=False,
                message="No skill context available",
            )

        marketplace = self.context.get_skill("marketplace")
        if not marketplace:
            return SkillResult(
                success=False,
                message="MarketplaceSkill not installed",
            )

        min_success_rate = float(params.get("min_success_rate", 0.7))
        min_executions = int(params.get("min_executions", 5))

        # Get service execution stats from revenue_services
        revenue_skill = self.context.get_skill("revenue_services")
        service_stats = {}
        if revenue_skill:
            stats_result = await revenue_skill.execute("service_stats", {})
            if stats_result.success and stats_result.data:
                for svc_name, stats in stats_result.data.get("services", {}).items():
                    service_stats[svc_name] = stats

        # Get current marketplace services
        list_result = await marketplace.execute("list_services", {"status": "active"})
        services = list_result.data.get("services", []) if list_result.data else []

        health_report = []
        paused_services = []
        healthy_services = []

        for service in services:
            action_name = service.get("action", "")
            stats = service_stats.get(action_name, {})
            total = stats.get("total", 0)
            successes = stats.get("successes", 0)
            success_rate = successes / total if total > 0 else 1.0

            entry = {
                "service_id": service.get("id"),
                "name": service.get("name"),
                "action": action_name,
                "total_executions": total,
                "success_rate": round(success_rate, 3),
                "status": "healthy",
            }

            if total >= min_executions and success_rate < min_success_rate:
                entry["status"] = "unhealthy"
                entry["recommendation"] = "auto_pause"
                paused_services.append(entry)
            else:
                healthy_services.append(entry)

            health_report.append(entry)

        # Record health check
        self._catalog_state["health_checks"].append({
            "timestamp": datetime.now().isoformat(),
            "services_checked": len(health_report),
            "healthy": len(healthy_services),
            "unhealthy": len(paused_services),
        })
        self._save_state()

        return SkillResult(
            success=True,
            message=f"Health check: {len(healthy_services)} healthy, {len(paused_services)} need attention",
            data={
                "report": health_report,
                "healthy_count": len(healthy_services),
                "unhealthy_count": len(paused_services),
                "paused": paused_services,
            },
        )

    async def _sync_catalog(self, params: Dict) -> SkillResult:
        """Sync marketplace catalog with available skills."""
        if not self.context:
            return SkillResult(
                success=False,
                message="No skill context available",
            )

        marketplace = self.context.get_skill("marketplace")
        if not marketplace:
            return SkillResult(
                success=False,
                message="MarketplaceSkill not installed",
            )

        # First, auto-register any missing services
        register_result = await self._auto_register({"force": False})
        newly_registered = register_result.data.get("total_registered", 0) if register_result.data else 0

        # Then check health
        health_result = await self._health_check(params)
        unhealthy = health_result.data.get("unhealthy_count", 0) if health_result.data else 0

        # Record sync
        self._catalog_state["sync_history"].append({
            "timestamp": datetime.now().isoformat(),
            "newly_registered": newly_registered,
            "unhealthy_detected": unhealthy,
        })
        self._save_state()

        return SkillResult(
            success=True,
            message=f"Catalog synced: {newly_registered} new services registered, {unhealthy} need attention",
            data={
                "newly_registered": newly_registered,
                "unhealthy_detected": unhealthy,
                "registration": register_result.data,
                "health": health_result.data,
            },
        )

    async def _catalog_status(self, params: Dict) -> SkillResult:
        """Get the current catalog status and metrics."""
        registered = self._catalog_state.get("registered_services", [])
        health_checks = self._catalog_state.get("health_checks", [])
        sync_history = self._catalog_state.get("sync_history", [])

        # Summary of recent health
        recent_health = health_checks[-1] if health_checks else None
        recent_sync = sync_history[-1] if sync_history else None

        # Get live marketplace count if possible
        marketplace_count = 0
        if self.context:
            marketplace = self.context.get_skill("marketplace")
            if marketplace:
                list_result = await marketplace.execute("list_services", {})
                if list_result.data:
                    marketplace_count = list_result.data.get("total", len(list_result.data.get("services", [])))

        return SkillResult(
            success=True,
            message=f"Catalog: {marketplace_count} services in marketplace, {len(registered)} auto-registered",
            data={
                "marketplace_services": marketplace_count,
                "auto_registered": len(registered),
                "curated_services": len(SERVICE_CATALOG),
                "recent_health_check": recent_health,
                "recent_sync": recent_sync,
                "total_health_checks": len(health_checks),
                "total_syncs": len(sync_history),
                "last_register": self._catalog_state.get("last_register"),
            },
        )
