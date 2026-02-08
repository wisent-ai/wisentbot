#!/usr/bin/env python3
"""
Service Catalog Skill - Pre-built service offerings deployable via ServiceAPI.

This is the "app store" for revenue services. While RevenueServiceSkill provides
the raw service implementations and AutoCatalogSkill bridges them to the marketplace,
ServiceCatalogSkill provides curated, ready-to-deploy service PACKAGES with:

  - Pre-configured pricing tiers (free, basic, pro, enterprise)
  - SLA guarantees (response time, uptime, quality)
  - Usage quotas and rate limiting per tier
  - Bundle packages (e.g., "Developer Tools" = code review + API docs + data analysis)
  - One-command deployment of entire service catalogs
  - Revenue projections based on expected usage patterns
  - A/B pricing experiments via ExperimentSkill integration

Deploy flow:
  1. Agent browses available catalog offerings
  2. Agent deploys a bundle or individual offering
  3. ServiceCatalogSkill configures pricing, SLAs, quotas
  4. Services become available via ServiceAPI
  5. Revenue tracked via MarketplaceSkill

Part of the Revenue Generation pillar: the service packaging and go-to-market layer.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from copy import deepcopy

from .base import Skill, SkillResult, SkillManifest, SkillAction


CATALOG_STATE_FILE = Path(__file__).parent.parent / "data" / "service_catalog_state.json"


# ── Pricing Tiers ──────────────────────────────────────────────────────

PRICING_TIERS = {
    "free": {
        "name": "Free",
        "description": "Try services with limited usage",
        "multiplier": 0.0,
        "rate_limit_per_hour": 5,
        "rate_limit_per_day": 20,
        "max_input_size_kb": 10,
        "sla_response_minutes": 30,
        "support": "community",
        "priority": "low",
    },
    "basic": {
        "name": "Basic",
        "description": "Affordable access for individuals",
        "multiplier": 1.0,
        "rate_limit_per_hour": 50,
        "rate_limit_per_day": 500,
        "max_input_size_kb": 100,
        "sla_response_minutes": 10,
        "support": "email",
        "priority": "normal",
    },
    "pro": {
        "name": "Professional",
        "description": "Full access for teams and power users",
        "multiplier": 2.5,
        "rate_limit_per_hour": 200,
        "rate_limit_per_day": 5000,
        "max_input_size_kb": 1000,
        "sla_response_minutes": 5,
        "support": "priority_email",
        "priority": "high",
    },
    "enterprise": {
        "name": "Enterprise",
        "description": "Unlimited access with dedicated support",
        "multiplier": 5.0,
        "rate_limit_per_hour": 1000,
        "rate_limit_per_day": 50000,
        "max_input_size_kb": 10000,
        "sla_response_minutes": 1,
        "support": "dedicated",
        "priority": "critical",
    },
}

# ── Service Offerings (individual services with base pricing) ──────────

SERVICE_OFFERINGS = {
    "code_review": {
        "name": "AI Code Review",
        "description": "Professional code review: security vulns, bugs, style, performance",
        "skill_id": "revenue_services",
        "action": "code_review",
        "base_price": 0.10,
        "cost_estimate": 0.02,
        "category": "developer_tools",
        "tags": ["code", "review", "security", "quality"],
        "avg_execution_seconds": 5,
        "quality_score": 0.85,
    },
    "text_summarization": {
        "name": "Text Summarization",
        "description": "Condense documents into key points (bullet, paragraph, executive)",
        "skill_id": "revenue_services",
        "action": "summarize_text",
        "base_price": 0.05,
        "cost_estimate": 0.01,
        "category": "content",
        "tags": ["text", "summary", "content"],
        "avg_execution_seconds": 3,
        "quality_score": 0.80,
    },
    "data_analysis": {
        "name": "Data Analysis",
        "description": "Extract insights from structured data: stats, patterns, anomalies",
        "skill_id": "revenue_services",
        "action": "analyze_data",
        "base_price": 0.10,
        "cost_estimate": 0.02,
        "category": "data",
        "tags": ["data", "analysis", "insights"],
        "avg_execution_seconds": 5,
        "quality_score": 0.82,
    },
    "seo_audit": {
        "name": "SEO Content Audit",
        "description": "Analyze content for search engine optimization and readability",
        "skill_id": "revenue_services",
        "action": "seo_audit",
        "base_price": 0.05,
        "cost_estimate": 0.01,
        "category": "marketing",
        "tags": ["seo", "content", "marketing"],
        "avg_execution_seconds": 3,
        "quality_score": 0.78,
    },
    "api_docs": {
        "name": "API Documentation Generator",
        "description": "Generate comprehensive API docs from code/endpoint definitions",
        "skill_id": "revenue_services",
        "action": "generate_api_docs",
        "base_price": 0.08,
        "cost_estimate": 0.015,
        "category": "developer_tools",
        "tags": ["api", "docs", "documentation"],
        "avg_execution_seconds": 4,
        "quality_score": 0.80,
    },
}

# ── Service Bundles (curated packages of multiple services) ──────────

SERVICE_BUNDLES = {
    "developer_essentials": {
        "name": "Developer Essentials",
        "description": "Everything a developer needs: code review, API docs, and data analysis",
        "services": ["code_review", "api_docs", "data_analysis"],
        "discount_pct": 15,
        "target_audience": "Software developers and engineering teams",
        "estimated_monthly_revenue": 50.0,
    },
    "content_creator": {
        "name": "Content Creator Suite",
        "description": "Content tools: summarization and SEO optimization",
        "services": ["text_summarization", "seo_audit"],
        "discount_pct": 10,
        "target_audience": "Content creators, bloggers, marketers",
        "estimated_monthly_revenue": 30.0,
    },
    "full_stack": {
        "name": "Full Stack Package",
        "description": "All services at a bundled discount - maximum value",
        "services": ["code_review", "text_summarization", "data_analysis", "seo_audit", "api_docs"],
        "discount_pct": 25,
        "target_audience": "Agencies, startups, and power users",
        "estimated_monthly_revenue": 100.0,
    },
    "data_intelligence": {
        "name": "Data Intelligence",
        "description": "Data-focused services: analysis and documentation",
        "services": ["data_analysis", "api_docs"],
        "discount_pct": 10,
        "target_audience": "Data engineers and API developers",
        "estimated_monthly_revenue": 40.0,
    },
}


class ServiceCatalogSkill(Skill):
    """
    Pre-built service offerings and bundles for one-command deployment.

    Acts as the "app store" layer: curated service packages with pricing
    tiers, SLA guarantees, usage quotas, and revenue projections.
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._deployed: Dict[str, Dict] = {}  # offering_id -> deployment config
        self._deployed_bundles: Dict[str, Dict] = {}  # bundle_id -> deployment info
        self._revenue_log: List[Dict] = []
        self._load_state()

    def _load_state(self):
        try:
            if CATALOG_STATE_FILE.exists():
                with open(CATALOG_STATE_FILE, "r") as f:
                    state = json.load(f)
                self._deployed = state.get("deployed", {})
                self._deployed_bundles = state.get("deployed_bundles", {})
                self._revenue_log = state.get("revenue_log", [])
        except (json.JSONDecodeError, IOError):
            pass

    def _save_state(self):
        CATALOG_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "deployed": self._deployed,
            "deployed_bundles": self._deployed_bundles,
            "revenue_log": self._revenue_log[-200:],
        }
        try:
            with open(CATALOG_STATE_FILE, "w") as f:
                json.dump(state, f, indent=2, default=str)
        except IOError:
            pass

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="service_catalog",
            name="Service Catalog",
            version="1.0.0",
            category="revenue",
            description="Pre-built service offerings and bundles for instant deployment",
            actions=self.get_actions(),
            required_credentials=[],
        )

    def get_actions(self) -> List[SkillAction]:
        return [
            SkillAction(
                name="list_offerings",
                description="Browse all available service offerings with pricing",
                parameters={},
                estimated_cost=0,
            ),
            SkillAction(
                name="list_bundles",
                description="Browse curated service bundles with discounts",
                parameters={},
                estimated_cost=0,
            ),
            SkillAction(
                name="preview",
                description="Preview a service offering or bundle with full pricing breakdown",
                parameters={
                    "offering_id": {"type": "str", "required": True, "description": "Service or bundle ID to preview"},
                },
                estimated_cost=0,
            ),
            SkillAction(
                name="deploy",
                description="Deploy a single service offering with a pricing tier",
                parameters={
                    "offering_id": {"type": "str", "required": True, "description": "Service offering ID"},
                    "tier": {"type": "str", "required": False, "description": "Pricing tier: free, basic, pro, enterprise (default: basic)"},
                },
                estimated_cost=0,
            ),
            SkillAction(
                name="deploy_bundle",
                description="Deploy an entire service bundle with a pricing tier",
                parameters={
                    "bundle_id": {"type": "str", "required": True, "description": "Bundle ID to deploy"},
                    "tier": {"type": "str", "required": False, "description": "Pricing tier (default: basic)"},
                },
                estimated_cost=0,
            ),
            SkillAction(
                name="undeploy",
                description="Remove a deployed service offering",
                parameters={
                    "offering_id": {"type": "str", "required": True, "description": "Service offering ID to remove"},
                },
                estimated_cost=0,
            ),
            SkillAction(
                name="status",
                description="View deployment status and revenue metrics for all deployed services",
                parameters={},
                estimated_cost=0,
            ),
            SkillAction(
                name="project_revenue",
                description="Project monthly revenue based on usage estimates",
                parameters={
                    "daily_requests": {"type": "int", "required": False, "description": "Expected daily requests per service (default: 100)"},
                    "tier": {"type": "str", "required": False, "description": "Pricing tier (default: basic)"},
                },
                estimated_cost=0,
            ),
        ]

    def execute(self, action: str, params: Dict[str, Any] = None) -> SkillResult:
        params = params or {}

        actions = {
            "list_offerings": self._list_offerings,
            "list_bundles": self._list_bundles,
            "preview": self._preview,
            "deploy": self._deploy,
            "deploy_bundle": self._deploy_bundle,
            "undeploy": self._undeploy,
            "status": self._status,
            "project_revenue": self._project_revenue,
        }

        handler = actions.get(action)
        if not handler:
            return SkillResult(
                success=False,
                message=f"Unknown action: {action}. Available: {list(actions.keys())}",
            )
        return handler(params)

    # ── Actions ────────────────────────────────────────────────────────

    def _list_offerings(self, params: Dict) -> SkillResult:
        """List all available service offerings."""
        offerings = []
        for oid, offering in SERVICE_OFFERINGS.items():
            deployed = oid in self._deployed
            offerings.append({
                "id": oid,
                "name": offering["name"],
                "description": offering["description"],
                "base_price": offering["base_price"],
                "category": offering["category"],
                "tags": offering["tags"],
                "deployed": deployed,
                "tier": self._deployed[oid].get("tier") if deployed else None,
            })
        return SkillResult(
            success=True,
            message=f"Found {len(offerings)} service offerings ({sum(1 for o in offerings if o['deployed'])} deployed)",
            data={"offerings": offerings, "categories": list(set(o["category"] for o in offerings))},
        )

    def _list_bundles(self, params: Dict) -> SkillResult:
        """List all available service bundles."""
        bundles = []
        for bid, bundle in SERVICE_BUNDLES.items():
            deployed = bid in self._deployed_bundles
            total_base = sum(SERVICE_OFFERINGS[s]["base_price"] for s in bundle["services"] if s in SERVICE_OFFERINGS)
            discounted = total_base * (1 - bundle["discount_pct"] / 100)
            bundles.append({
                "id": bid,
                "name": bundle["name"],
                "description": bundle["description"],
                "services": bundle["services"],
                "service_count": len(bundle["services"]),
                "total_base_price": round(total_base, 4),
                "bundle_price": round(discounted, 4),
                "discount_pct": bundle["discount_pct"],
                "savings": round(total_base - discounted, 4),
                "target_audience": bundle["target_audience"],
                "estimated_monthly_revenue": bundle["estimated_monthly_revenue"],
                "deployed": deployed,
            })
        return SkillResult(
            success=True,
            message=f"Found {len(bundles)} service bundles ({sum(1 for b in bundles if b['deployed'])} deployed)",
            data={"bundles": bundles},
        )

    def _preview(self, params: Dict) -> SkillResult:
        """Preview a service or bundle with full pricing across all tiers."""
        offering_id = params.get("offering_id", "")

        # Check if it's a service offering
        if offering_id in SERVICE_OFFERINGS:
            return self._preview_offering(offering_id)

        # Check if it's a bundle
        if offering_id in SERVICE_BUNDLES:
            return self._preview_bundle(offering_id)

        return SkillResult(
            success=False,
            message=f"Unknown offering: {offering_id}. Use list_offerings or list_bundles to see available options.",
        )

    def _preview_offering(self, offering_id: str) -> SkillResult:
        """Preview a single service offering with tier pricing."""
        offering = SERVICE_OFFERINGS[offering_id]
        pricing = {}
        for tid, tier in PRICING_TIERS.items():
            price = round(offering["base_price"] * tier["multiplier"], 4)
            margin = round(price - offering["cost_estimate"], 4) if price > 0 else 0
            pricing[tid] = {
                "price_per_request": price,
                "cost_per_request": offering["cost_estimate"],
                "margin_per_request": margin,
                "margin_pct": round((margin / price * 100) if price > 0 else 0, 1),
                "rate_limit_per_hour": tier["rate_limit_per_hour"],
                "rate_limit_per_day": tier["rate_limit_per_day"],
                "sla_response_minutes": tier["sla_response_minutes"],
                "max_input_size_kb": tier["max_input_size_kb"],
            }
        return SkillResult(
            success=True,
            message=f"Preview: {offering['name']} ({offering_id})",
            data={
                "offering": {**offering, "id": offering_id},
                "pricing_by_tier": pricing,
                "deployed": offering_id in self._deployed,
            },
        )

    def _preview_bundle(self, bundle_id: str) -> SkillResult:
        """Preview a bundle with aggregated pricing."""
        bundle = SERVICE_BUNDLES[bundle_id]
        services = []
        total_base = 0
        total_cost = 0
        for sid in bundle["services"]:
            if sid in SERVICE_OFFERINGS:
                svc = SERVICE_OFFERINGS[sid]
                services.append({"id": sid, "name": svc["name"], "base_price": svc["base_price"]})
                total_base += svc["base_price"]
                total_cost += svc["cost_estimate"]

        discount_factor = 1 - bundle["discount_pct"] / 100
        pricing = {}
        for tid, tier in PRICING_TIERS.items():
            bundle_price = round(total_base * tier["multiplier"] * discount_factor, 4)
            cost = round(total_cost, 4)
            margin = round(bundle_price - cost, 4) if bundle_price > 0 else 0
            pricing[tid] = {
                "bundle_price_per_request_set": bundle_price,
                "total_cost": cost,
                "margin": margin,
                "rate_limit_per_hour": tier["rate_limit_per_hour"],
                "sla_response_minutes": tier["sla_response_minutes"],
            }

        return SkillResult(
            success=True,
            message=f"Preview: {bundle['name']} bundle ({bundle_id}) - {len(services)} services, {bundle['discount_pct']}% discount",
            data={
                "bundle": {**bundle, "id": bundle_id},
                "services": services,
                "total_base_price": round(total_base, 4),
                "bundle_discount_pct": bundle["discount_pct"],
                "pricing_by_tier": pricing,
                "deployed": bundle_id in self._deployed_bundles,
            },
        )

    def _deploy(self, params: Dict) -> SkillResult:
        """Deploy a single service offering."""
        offering_id = params.get("offering_id", "")
        tier = params.get("tier", "basic")

        if offering_id not in SERVICE_OFFERINGS:
            return SkillResult(
                success=False,
                message=f"Unknown offering: {offering_id}. Use list_offerings to see options.",
            )
        if tier not in PRICING_TIERS:
            return SkillResult(
                success=False,
                message=f"Unknown tier: {tier}. Available: {list(PRICING_TIERS.keys())}",
            )

        offering = SERVICE_OFFERINGS[offering_id]
        tier_config = PRICING_TIERS[tier]
        price = round(offering["base_price"] * tier_config["multiplier"], 4)

        deployment = {
            "offering_id": offering_id,
            "name": offering["name"],
            "tier": tier,
            "price": price,
            "cost_estimate": offering["cost_estimate"],
            "skill_id": offering["skill_id"],
            "action": offering["action"],
            "rate_limit_per_hour": tier_config["rate_limit_per_hour"],
            "rate_limit_per_day": tier_config["rate_limit_per_day"],
            "sla_response_minutes": tier_config["sla_response_minutes"],
            "max_input_size_kb": tier_config["max_input_size_kb"],
            "deployed_at": datetime.now().isoformat(),
        }
        self._deployed[offering_id] = deployment
        self._save_state()

        return SkillResult(
            success=True,
            message=f"Deployed {offering['name']} on {tier_config['name']} tier at ${price}/request",
            data={"deployment": deployment},
        )

    def _deploy_bundle(self, params: Dict) -> SkillResult:
        """Deploy all services in a bundle."""
        bundle_id = params.get("bundle_id", "")
        tier = params.get("tier", "basic")

        if bundle_id not in SERVICE_BUNDLES:
            return SkillResult(
                success=False,
                message=f"Unknown bundle: {bundle_id}. Use list_bundles to see options.",
            )
        if tier not in PRICING_TIERS:
            return SkillResult(
                success=False,
                message=f"Unknown tier: {tier}. Available: {list(PRICING_TIERS.keys())}",
            )

        bundle = SERVICE_BUNDLES[bundle_id]
        tier_config = PRICING_TIERS[tier]
        discount_factor = 1 - bundle["discount_pct"] / 100
        deployed_services = []

        for sid in bundle["services"]:
            if sid not in SERVICE_OFFERINGS:
                continue
            offering = SERVICE_OFFERINGS[sid]
            price = round(offering["base_price"] * tier_config["multiplier"] * discount_factor, 4)
            deployment = {
                "offering_id": sid,
                "name": offering["name"],
                "tier": tier,
                "price": price,
                "cost_estimate": offering["cost_estimate"],
                "skill_id": offering["skill_id"],
                "action": offering["action"],
                "rate_limit_per_hour": tier_config["rate_limit_per_hour"],
                "rate_limit_per_day": tier_config["rate_limit_per_day"],
                "sla_response_minutes": tier_config["sla_response_minutes"],
                "max_input_size_kb": tier_config["max_input_size_kb"],
                "bundle_id": bundle_id,
                "discount_pct": bundle["discount_pct"],
                "deployed_at": datetime.now().isoformat(),
            }
            self._deployed[sid] = deployment
            deployed_services.append(deployment)

        self._deployed_bundles[bundle_id] = {
            "bundle_id": bundle_id,
            "name": bundle["name"],
            "tier": tier,
            "services_deployed": len(deployed_services),
            "discount_pct": bundle["discount_pct"],
            "deployed_at": datetime.now().isoformat(),
        }
        self._save_state()

        total_price = sum(d["price"] for d in deployed_services)
        return SkillResult(
            success=True,
            message=f"Deployed bundle '{bundle['name']}' ({len(deployed_services)} services) on {tier_config['name']} tier. Total: ${round(total_price, 4)}/request set",
            data={
                "bundle": self._deployed_bundles[bundle_id],
                "services": deployed_services,
                "total_price_per_request_set": round(total_price, 4),
            },
        )

    def _undeploy(self, params: Dict) -> SkillResult:
        """Remove a deployed service offering."""
        offering_id = params.get("offering_id", "")

        # Check if it's a bundle
        if offering_id in self._deployed_bundles:
            bundle_info = self._deployed_bundles.pop(offering_id)
            # Remove all services from this bundle
            removed = []
            for sid in list(self._deployed.keys()):
                if self._deployed[sid].get("bundle_id") == offering_id:
                    removed.append(sid)
                    del self._deployed[sid]
            self._save_state()
            return SkillResult(
                success=True,
                message=f"Undeployed bundle '{bundle_info['name']}' and {len(removed)} services",
                data={"bundle_id": offering_id, "removed_services": removed},
            )

        if offering_id not in self._deployed:
            return SkillResult(
                success=False,
                message=f"Service '{offering_id}' is not deployed.",
            )

        deployment = self._deployed.pop(offering_id)
        # Also remove from bundle tracking if part of one
        bundle_id = deployment.get("bundle_id")
        if bundle_id and bundle_id in self._deployed_bundles:
            remaining = sum(1 for d in self._deployed.values() if d.get("bundle_id") == bundle_id)
            if remaining == 0:
                del self._deployed_bundles[bundle_id]

        self._save_state()
        return SkillResult(
            success=True,
            message=f"Undeployed '{deployment['name']}'",
            data={"removed": deployment},
        )

    def _status(self, params: Dict) -> SkillResult:
        """View deployment status and revenue summary."""
        deployed_list = []
        total_price = 0
        total_cost = 0
        for oid, dep in self._deployed.items():
            deployed_list.append({
                "id": oid,
                "name": dep["name"],
                "tier": dep["tier"],
                "price": dep["price"],
                "cost_estimate": dep["cost_estimate"],
                "margin": round(dep["price"] - dep["cost_estimate"], 4),
                "bundle": dep.get("bundle_id"),
                "deployed_at": dep.get("deployed_at"),
            })
            total_price += dep["price"]
            total_cost += dep["cost_estimate"]

        bundles_list = list(self._deployed_bundles.values())

        return SkillResult(
            success=True,
            message=f"{len(deployed_list)} services deployed across {len(bundles_list)} bundles",
            data={
                "deployed_services": deployed_list,
                "deployed_bundles": bundles_list,
                "summary": {
                    "total_services": len(deployed_list),
                    "total_bundles": len(bundles_list),
                    "total_price_per_request_set": round(total_price, 4),
                    "total_cost_per_request_set": round(total_cost, 4),
                    "total_margin_per_request_set": round(total_price - total_cost, 4),
                },
            },
        )

    def _project_revenue(self, params: Dict) -> SkillResult:
        """Project monthly revenue based on expected usage."""
        daily_requests = params.get("daily_requests", 100)
        tier = params.get("tier", "basic")

        if tier not in PRICING_TIERS:
            return SkillResult(
                success=False,
                message=f"Unknown tier: {tier}. Available: {list(PRICING_TIERS.keys())}",
            )

        tier_config = PRICING_TIERS[tier]
        projections = []
        total_monthly_revenue = 0
        total_monthly_cost = 0

        for oid, offering in SERVICE_OFFERINGS.items():
            price = round(offering["base_price"] * tier_config["multiplier"], 4)
            monthly_requests = daily_requests * 30
            monthly_revenue = round(price * monthly_requests, 2)
            monthly_cost = round(offering["cost_estimate"] * monthly_requests, 2)
            monthly_profit = round(monthly_revenue - monthly_cost, 2)
            total_monthly_revenue += monthly_revenue
            total_monthly_cost += monthly_cost

            projections.append({
                "service": oid,
                "name": offering["name"],
                "price_per_request": price,
                "daily_requests": daily_requests,
                "monthly_requests": monthly_requests,
                "monthly_revenue": monthly_revenue,
                "monthly_cost": monthly_cost,
                "monthly_profit": monthly_profit,
                "margin_pct": round((monthly_profit / monthly_revenue * 100) if monthly_revenue > 0 else 0, 1),
            })

        return SkillResult(
            success=True,
            message=f"Revenue projection: ${round(total_monthly_revenue, 2)}/month at {daily_requests} requests/day/service on {tier_config['name']} tier",
            data={
                "projections": projections,
                "summary": {
                    "tier": tier,
                    "daily_requests_per_service": daily_requests,
                    "total_monthly_revenue": round(total_monthly_revenue, 2),
                    "total_monthly_cost": round(total_monthly_cost, 2),
                    "total_monthly_profit": round(total_monthly_revenue - total_monthly_cost, 2),
                    "profit_margin_pct": round(
                        ((total_monthly_revenue - total_monthly_cost) / total_monthly_revenue * 100)
                        if total_monthly_revenue > 0 else 0, 1
                    ),
                    "annual_revenue_estimate": round(total_monthly_revenue * 12, 2),
                    "annual_profit_estimate": round((total_monthly_revenue - total_monthly_cost) * 12, 2),
                },
            },
        )
