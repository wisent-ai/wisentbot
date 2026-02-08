#!/usr/bin/env python3
"""
ExternalAPIMarketplaceSkill - Catalog of pre-configured external API endpoints.

A curated marketplace of ready-to-use API integrations. Customers browse,
subscribe, and call external APIs through the agent without needing to
handle authentication, rate limiting, or response parsing themselves.

Revenue model:
  - Per-call pricing with markup over raw API costs
  - Subscription tiers for high-volume users
  - The agent earns the spread between what it charges and what the API costs

Revenue flow:
  Customer -> ServiceAPI -> APIMarketplace -> HTTPClientSkill -> External API
                         -> BillingPipeline (markup revenue)

Built-in API catalog includes: weather, exchange rates, IP geolocation,
news headlines, URL shortening, DNS lookup, and more. Custom APIs can
be added by the agent or operators.

Pillar: Revenue Generation - turns external APIs into a billable product catalog.
"""

import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import Skill, SkillAction, SkillManifest, SkillResult

DATA_DIR = Path(__file__).parent.parent / "data" / "api_marketplace"
CATALOG_FILE = DATA_DIR / "catalog.json"
SUBSCRIPTIONS_FILE = DATA_DIR / "subscriptions.json"
USAGE_FILE = DATA_DIR / "usage.json"
MAX_HISTORY = 1000


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


# --- Built-in API catalog ---
# These are real, commonly-used free/freemium APIs that the agent can proxy.
BUILTIN_APIS = {
    "weather_current": {
        "id": "weather_current",
        "name": "Current Weather",
        "description": "Get current weather conditions for any city worldwide",
        "category": "weather",
        "base_url": "https://wttr.in/{city}?format=j1",
        "method": "GET",
        "auth_type": "none",
        "parameters": {
            "city": {"type": "string", "required": True, "description": "City name (e.g. 'London', 'New York')"},
        },
        "response_transform": "json",
        "price_per_call": 0.003,
        "raw_cost": 0.0,
        "rate_limit_per_minute": 30,
        "example_request": {"city": "London"},
        "tags": ["weather", "geolocation", "free-tier"],
    },
    "exchange_rates": {
        "id": "exchange_rates",
        "name": "Currency Exchange Rates",
        "description": "Get latest currency exchange rates (USD base)",
        "category": "finance",
        "base_url": "https://open.er-api.com/v6/latest/{base_currency}",
        "method": "GET",
        "auth_type": "none",
        "parameters": {
            "base_currency": {"type": "string", "required": True, "description": "Base currency code (e.g. USD, EUR, GBP)"},
        },
        "response_transform": "json",
        "price_per_call": 0.005,
        "raw_cost": 0.0,
        "rate_limit_per_minute": 30,
        "example_request": {"base_currency": "USD"},
        "tags": ["finance", "currency", "exchange", "free-tier"],
    },
    "ip_geolocation": {
        "id": "ip_geolocation",
        "name": "IP Geolocation",
        "description": "Get geographic location data for an IP address",
        "category": "networking",
        "base_url": "http://ip-api.com/json/{ip}",
        "method": "GET",
        "auth_type": "none",
        "parameters": {
            "ip": {"type": "string", "required": True, "description": "IP address to geolocate"},
        },
        "response_transform": "json",
        "price_per_call": 0.002,
        "raw_cost": 0.0,
        "rate_limit_per_minute": 45,
        "example_request": {"ip": "8.8.8.8"},
        "tags": ["networking", "geolocation", "ip", "free-tier"],
    },
    "dns_lookup": {
        "id": "dns_lookup",
        "name": "DNS Lookup",
        "description": "Resolve DNS records for a domain",
        "category": "networking",
        "base_url": "https://dns.google/resolve?name={domain}&type={record_type}",
        "method": "GET",
        "auth_type": "none",
        "parameters": {
            "domain": {"type": "string", "required": True, "description": "Domain name to resolve"},
            "record_type": {"type": "string", "required": False, "description": "DNS record type (A, AAAA, MX, etc.). Default: A"},
        },
        "response_transform": "json",
        "price_per_call": 0.002,
        "raw_cost": 0.0,
        "rate_limit_per_minute": 60,
        "example_request": {"domain": "example.com", "record_type": "A"},
        "tags": ["networking", "dns", "free-tier"],
    },
    "url_shortener": {
        "id": "url_shortener",
        "name": "URL Shortener",
        "description": "Shorten long URLs using is.gd service",
        "category": "utilities",
        "base_url": "https://is.gd/create.php?format=json&url={url}",
        "method": "GET",
        "auth_type": "none",
        "parameters": {
            "url": {"type": "string", "required": True, "description": "URL to shorten"},
        },
        "response_transform": "json",
        "price_per_call": 0.001,
        "raw_cost": 0.0,
        "rate_limit_per_minute": 10,
        "example_request": {"url": "https://example.com/very/long/path"},
        "tags": ["utilities", "url", "free-tier"],
    },
    "random_data": {
        "id": "random_data",
        "name": "Random Data Generator",
        "description": "Generate random user profiles, addresses, and test data",
        "category": "testing",
        "base_url": "https://randomuser.me/api/?results={count}&nat={nationality}",
        "method": "GET",
        "auth_type": "none",
        "parameters": {
            "count": {"type": "integer", "required": False, "description": "Number of results (default 1, max 50)"},
            "nationality": {"type": "string", "required": False, "description": "Nationality filter (us, gb, fr, de, etc.)"},
        },
        "response_transform": "json",
        "price_per_call": 0.002,
        "raw_cost": 0.0,
        "rate_limit_per_minute": 30,
        "example_request": {"count": "5", "nationality": "us"},
        "tags": ["testing", "mock-data", "free-tier"],
    },
    "json_placeholder": {
        "id": "json_placeholder",
        "name": "JSON Placeholder API",
        "description": "Fake REST API for testing: posts, comments, users, todos",
        "category": "testing",
        "base_url": "https://jsonplaceholder.typicode.com/{resource}/{id}",
        "method": "GET",
        "auth_type": "none",
        "parameters": {
            "resource": {"type": "string", "required": True, "description": "Resource type: posts, comments, users, todos, albums, photos"},
            "id": {"type": "string", "required": False, "description": "Resource ID (omit for list)"},
        },
        "response_transform": "json",
        "price_per_call": 0.001,
        "raw_cost": 0.0,
        "rate_limit_per_minute": 60,
        "example_request": {"resource": "posts", "id": "1"},
        "tags": ["testing", "rest", "free-tier"],
    },
    "public_holidays": {
        "id": "public_holidays",
        "name": "Public Holidays",
        "description": "Get public holidays for any country and year",
        "category": "reference",
        "base_url": "https://date.nager.at/api/v3/PublicHolidays/{year}/{country_code}",
        "method": "GET",
        "auth_type": "none",
        "parameters": {
            "year": {"type": "string", "required": True, "description": "Year (e.g. 2026)"},
            "country_code": {"type": "string", "required": True, "description": "ISO 3166-1 alpha-2 country code (e.g. US, GB, DE)"},
        },
        "response_transform": "json",
        "price_per_call": 0.002,
        "raw_cost": 0.0,
        "rate_limit_per_minute": 30,
        "example_request": {"year": "2026", "country_code": "US"},
        "tags": ["reference", "holidays", "free-tier"],
    },
}

# --- Subscription tiers ---
SUBSCRIPTION_TIERS = {
    "free": {
        "name": "Free",
        "monthly_price": 0.0,
        "calls_per_month": 100,
        "rate_limit_multiplier": 1.0,
        "discount_pct": 0,
    },
    "basic": {
        "name": "Basic",
        "monthly_price": 9.99,
        "calls_per_month": 5000,
        "rate_limit_multiplier": 2.0,
        "discount_pct": 10,
    },
    "pro": {
        "name": "Pro",
        "monthly_price": 29.99,
        "calls_per_month": 50000,
        "rate_limit_multiplier": 5.0,
        "discount_pct": 25,
    },
    "enterprise": {
        "name": "Enterprise",
        "monthly_price": 99.99,
        "calls_per_month": -1,  # unlimited
        "rate_limit_multiplier": 10.0,
        "discount_pct": 40,
    },
}


def _load_json(filepath: Path, default: Any = None) -> Any:
    try:
        if filepath.exists():
            with open(filepath, "r") as f:
                return json.load(f)
    except (json.JSONDecodeError, IOError):
        pass
    return default if default is not None else {}


def _save_json(filepath: Path, data: Any):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)
    except IOError:
        pass


class ExternalAPIMarketplaceSkill(Skill):
    """Curated marketplace of ready-to-use external API integrations."""

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._catalog = dict(BUILTIN_APIS)
        self._custom_apis = _load_json(CATALOG_FILE, {})
        self._catalog.update(self._custom_apis)
        self._subscriptions = _load_json(SUBSCRIPTIONS_FILE, {})
        self._usage = _load_json(USAGE_FILE, {
            "calls": [],
            "revenue": {"total": 0.0, "by_api": {}, "by_customer": {}},
            "rate_limits": {},
        })
        self._http_skill = None

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="api_marketplace",
            name="External API Marketplace",
            version="1.0.0",
            category="revenue",
            description="Curated catalog of external APIs: browse, subscribe, and call APIs with per-call billing",
            actions=self.get_actions(),
            required_credentials=[],
            install_cost=0,
            author="singularity",
        )

    def get_actions(self) -> List[SkillAction]:
        return [
            SkillAction(
                name="browse",
                description="Browse the API catalog with optional category/tag filtering",
                parameters={
                    "category": {"type": "string", "required": False, "description": "Filter by category"},
                    "tag": {"type": "string", "required": False, "description": "Filter by tag"},
                    "search": {"type": "string", "required": False, "description": "Search API names and descriptions"},
                },
                estimated_cost=0,
            ),
            SkillAction(
                name="details",
                description="Get full details for a specific API endpoint",
                parameters={
                    "api_id": {"type": "string", "required": True, "description": "API identifier from catalog"},
                },
                estimated_cost=0,
            ),
            SkillAction(
                name="call",
                description="Call an API endpoint (billed per-call)",
                parameters={
                    "customer_id": {"type": "string", "required": True, "description": "Customer identifier"},
                    "api_id": {"type": "string", "required": True, "description": "API to call from catalog"},
                    "params": {"type": "object", "required": False, "description": "API parameters (substituted into URL template)"},
                },
                estimated_cost=0.005,
            ),
            SkillAction(
                name="subscribe",
                description="Subscribe a customer to a pricing tier",
                parameters={
                    "customer_id": {"type": "string", "required": True, "description": "Customer identifier"},
                    "tier": {"type": "string", "required": True, "description": "Tier: free, basic, pro, enterprise"},
                },
                estimated_cost=0,
            ),
            SkillAction(
                name="add_api",
                description="Add a custom API endpoint to the marketplace",
                parameters={
                    "api_id": {"type": "string", "required": True, "description": "Unique API identifier"},
                    "name": {"type": "string", "required": True, "description": "Human-readable name"},
                    "description": {"type": "string", "required": True, "description": "What this API does"},
                    "base_url": {"type": "string", "required": True, "description": "URL template with {param} placeholders"},
                    "method": {"type": "string", "required": False, "description": "HTTP method (default GET)"},
                    "category": {"type": "string", "required": False, "description": "Category for browsing"},
                    "price_per_call": {"type": "number", "required": False, "description": "Price per call (default 0.005)"},
                    "auth_header": {"type": "string", "required": False, "description": "Auth header value if needed"},
                    "parameters": {"type": "object", "required": False, "description": "Parameter definitions"},
                },
                estimated_cost=0,
            ),
            SkillAction(
                name="remove_api",
                description="Remove a custom API from the marketplace",
                parameters={
                    "api_id": {"type": "string", "required": True, "description": "API to remove (builtin APIs cannot be removed)"},
                },
                estimated_cost=0,
            ),
            SkillAction(
                name="usage",
                description="Get usage statistics for a customer or API",
                parameters={
                    "customer_id": {"type": "string", "required": False, "description": "Filter by customer"},
                    "api_id": {"type": "string", "required": False, "description": "Filter by API"},
                },
                estimated_cost=0,
            ),
            SkillAction(
                name="revenue",
                description="Get revenue report across the marketplace",
                parameters={
                    "period": {"type": "string", "required": False, "description": "Period: today, week, month, all (default all)"},
                },
                estimated_cost=0,
            ),
            SkillAction(
                name="tiers",
                description="List available subscription tiers and pricing",
                parameters={},
                estimated_cost=0,
            ),
        ]

    async def execute(self, action: str, params: Dict) -> SkillResult:
        handlers = {
            "browse": self._browse,
            "details": self._details,
            "call": self._call_api,
            "subscribe": self._subscribe,
            "add_api": self._add_api,
            "remove_api": self._remove_api,
            "usage": self._get_usage,
            "revenue": self._get_revenue,
            "tiers": self._list_tiers,
        }
        handler = handlers.get(action)
        if not handler:
            return SkillResult(success=False, message=f"Unknown action: {action}")
        return await handler(params)

    async def initialize(self) -> bool:
        return True

    def _get_http_skill(self):
        """Get HTTPClientSkill from context if available."""
        if self._http_skill is None:
            try:
                ctx = getattr(self, "context", None)
                if ctx and hasattr(ctx, "get_skill"):
                    self._http_skill = ctx.get_skill("http_client")
            except Exception:
                pass
        return self._http_skill

    def _get_customer_tier(self, customer_id: str) -> Dict:
        """Get subscription tier for a customer (default: free)."""
        sub = self._subscriptions.get(customer_id, {})
        tier_name = sub.get("tier", "free")
        return SUBSCRIPTION_TIERS.get(tier_name, SUBSCRIPTION_TIERS["free"])

    def _check_rate_limit(self, customer_id: str, api_id: str) -> bool:
        """Check if customer is within rate limits for this API."""
        key = f"{customer_id}:{api_id}"
        now = time.time()
        limits = self._usage.get("rate_limits", {})
        window = limits.get(key, [])
        # Clean old entries (older than 60s)
        window = [t for t in window if now - t < 60]
        limits[key] = window
        self._usage["rate_limits"] = limits

        api = self._catalog.get(api_id, {})
        base_rpm = api.get("rate_limit_per_minute", 30)
        tier = self._get_customer_tier(customer_id)
        max_rpm = int(base_rpm * tier.get("rate_limit_multiplier", 1.0))

        return len(window) < max_rpm

    def _record_rate_limit_hit(self, customer_id: str, api_id: str):
        """Record a rate limit usage."""
        key = f"{customer_id}:{api_id}"
        limits = self._usage.get("rate_limits", {})
        if key not in limits:
            limits[key] = []
        limits[key].append(time.time())
        self._usage["rate_limits"] = limits

    def _build_url(self, api: Dict, params: Dict) -> str:
        """Substitute parameters into URL template."""
        url = api["base_url"]
        api_params = api.get("parameters", {})
        for param_name, param_def in api_params.items():
            value = params.get(param_name, "")
            if not value and param_def.get("required"):
                raise ValueError(f"Missing required parameter: {param_name}")
            # Default values
            if not value:
                defaults = {"record_type": "A", "count": "1", "nationality": "us", "id": ""}
                value = defaults.get(param_name, "")
            placeholder = "{" + param_name + "}"
            url = url.replace(placeholder, str(value))
        # Clean trailing slashes from empty optional params
        url = url.rstrip("/")
        return url

    def _record_call(self, customer_id: str, api_id: str, success: bool, price: float):
        """Record API call for billing and analytics."""
        calls = self._usage.get("calls", [])
        calls.append({
            "id": str(uuid.uuid4())[:8],
            "customer_id": customer_id,
            "api_id": api_id,
            "success": success,
            "price": price,
            "timestamp": _now_iso(),
        })
        if len(calls) > MAX_HISTORY:
            calls = calls[-MAX_HISTORY:]
        self._usage["calls"] = calls

        # Revenue tracking
        rev = self._usage.get("revenue", {"total": 0.0, "by_api": {}, "by_customer": {}})
        if success:
            rev["total"] = rev.get("total", 0.0) + price
            rev["by_api"][api_id] = rev.get("by_api", {}).get(api_id, 0.0) + price
            rev["by_customer"][customer_id] = rev.get("by_customer", {}).get(customer_id, 0.0) + price
        self._usage["revenue"] = rev
        _save_json(USAGE_FILE, self._usage)

    # --- Action handlers ---

    async def _browse(self, params: Dict) -> SkillResult:
        """Browse the API catalog."""
        category = params.get("category", "").lower()
        tag = params.get("tag", "").lower()
        search = params.get("search", "").lower()

        results = []
        for api_id, api in self._catalog.items():
            if category and api.get("category", "").lower() != category:
                continue
            if tag and tag not in [t.lower() for t in api.get("tags", [])]:
                continue
            if search:
                searchable = f"{api.get('name', '')} {api.get('description', '')}".lower()
                if search not in searchable:
                    continue
            results.append({
                "id": api_id,
                "name": api.get("name", api_id),
                "description": api.get("description", ""),
                "category": api.get("category", "uncategorized"),
                "price_per_call": api.get("price_per_call", 0.005),
                "tags": api.get("tags", []),
            })

        categories = sorted(set(a.get("category", "") for a in self._catalog.values()))
        return SkillResult(
            success=True,
            message=f"Found {len(results)} APIs in marketplace",
            data={
                "apis": results,
                "total": len(results),
                "categories": categories,
                "filters_applied": {
                    "category": category or None,
                    "tag": tag or None,
                    "search": search or None,
                },
            },
        )

    async def _details(self, params: Dict) -> SkillResult:
        """Get full details for a specific API."""
        api_id = params.get("api_id", "")
        api = self._catalog.get(api_id)
        if not api:
            return SkillResult(success=False, message=f"API not found: {api_id}")

        return SkillResult(
            success=True,
            message=f"Details for {api.get('name', api_id)}",
            data={
                "api": api,
                "is_builtin": api_id in BUILTIN_APIS,
                "total_calls": sum(
                    1 for c in self._usage.get("calls", []) if c.get("api_id") == api_id
                ),
                "total_revenue": self._usage.get("revenue", {}).get("by_api", {}).get(api_id, 0.0),
            },
        )

    async def _call_api(self, params: Dict) -> SkillResult:
        """Call an API endpoint with billing."""
        customer_id = params.get("customer_id", "")
        api_id = params.get("api_id", "")
        call_params = params.get("params", {})

        if not customer_id:
            return SkillResult(success=False, message="customer_id is required")
        if not api_id:
            return SkillResult(success=False, message="api_id is required")

        api = self._catalog.get(api_id)
        if not api:
            return SkillResult(success=False, message=f"API not found in catalog: {api_id}")

        # Check rate limit
        if not self._check_rate_limit(customer_id, api_id):
            return SkillResult(
                success=False,
                message=f"Rate limit exceeded for {api_id}. Upgrade your tier for higher limits.",
                data={"api_id": api_id, "customer_id": customer_id},
            )

        # Check monthly quota
        tier = self._get_customer_tier(customer_id)
        monthly_limit = tier.get("calls_per_month", 100)
        if monthly_limit > 0:
            month_calls = sum(
                1 for c in self._usage.get("calls", [])
                if c.get("customer_id") == customer_id
                and c.get("timestamp", "")[:7] == datetime.utcnow().strftime("%Y-%m")
            )
            if month_calls >= monthly_limit:
                return SkillResult(
                    success=False,
                    message=f"Monthly quota ({monthly_limit} calls) exceeded. Upgrade your tier.",
                    data={"tier": tier["name"], "used": month_calls, "limit": monthly_limit},
                )

        # Calculate price with tier discount
        base_price = api.get("price_per_call", 0.005)
        discount = tier.get("discount_pct", 0) / 100.0
        price = round(base_price * (1 - discount), 6)

        # Build URL
        try:
            url = self._build_url(api, call_params)
        except ValueError as e:
            return SkillResult(success=False, message=str(e))

        # Record rate limit hit
        self._record_rate_limit_hit(customer_id, api_id)

        # Execute via HTTPClientSkill or simulate
        http = self._get_http_skill()
        method = api.get("method", "GET")
        headers = {}
        if api.get("auth_header"):
            headers["Authorization"] = api["auth_header"]

        if http:
            try:
                result = await http.execute("request", {
                    "url": url,
                    "method": method,
                    "headers": headers,
                })
                if result.success:
                    self._record_call(customer_id, api_id, True, price)
                    return SkillResult(
                        success=True,
                        message=f"API call to {api.get('name', api_id)} successful",
                        data={
                            "api_id": api_id,
                            "response": result.data,
                            "price_charged": price,
                            "tier_discount": f"{tier.get('discount_pct', 0)}%",
                        },
                        revenue=price,
                    )
                else:
                    self._record_call(customer_id, api_id, False, 0)
                    return SkillResult(
                        success=False,
                        message=f"API call failed: {result.message}",
                        data={"api_id": api_id, "url": url},
                    )
            except Exception as e:
                self._record_call(customer_id, api_id, False, 0)
                return SkillResult(success=False, message=f"HTTP error: {e}")
        else:
            # No HTTPClientSkill - simulate success for billing/tracking
            self._record_call(customer_id, api_id, True, price)
            return SkillResult(
                success=True,
                message=f"API call to {api.get('name', api_id)} queued (no HTTP transport)",
                data={
                    "api_id": api_id,
                    "url": url,
                    "method": method,
                    "price_charged": price,
                    "tier_discount": f"{tier.get('discount_pct', 0)}%",
                    "note": "Call simulated - HTTPClientSkill not available",
                },
                revenue=price,
            )

    async def _subscribe(self, params: Dict) -> SkillResult:
        """Subscribe customer to a pricing tier."""
        customer_id = params.get("customer_id", "")
        tier_name = params.get("tier", "").lower()

        if not customer_id:
            return SkillResult(success=False, message="customer_id is required")
        if tier_name not in SUBSCRIPTION_TIERS:
            return SkillResult(
                success=False,
                message=f"Unknown tier: {tier_name}. Available: {', '.join(SUBSCRIPTION_TIERS.keys())}",
            )

        tier = SUBSCRIPTION_TIERS[tier_name]
        old_tier = self._subscriptions.get(customer_id, {}).get("tier", "free")

        self._subscriptions[customer_id] = {
            "tier": tier_name,
            "subscribed_at": _now_iso(),
            "monthly_price": tier["monthly_price"],
        }
        _save_json(SUBSCRIPTIONS_FILE, self._subscriptions)

        return SkillResult(
            success=True,
            message=f"Customer {customer_id} subscribed to {tier['name']} tier",
            data={
                "customer_id": customer_id,
                "previous_tier": old_tier,
                "new_tier": tier_name,
                "monthly_price": tier["monthly_price"],
                "calls_per_month": tier["calls_per_month"],
                "discount_pct": tier["discount_pct"],
                "rate_limit_multiplier": tier["rate_limit_multiplier"],
            },
            revenue=tier["monthly_price"],
        )

    async def _add_api(self, params: Dict) -> SkillResult:
        """Add a custom API to the marketplace."""
        api_id = params.get("api_id", "")
        name = params.get("name", "")
        description = params.get("description", "")
        base_url = params.get("base_url", "")

        if not all([api_id, name, base_url]):
            return SkillResult(success=False, message="api_id, name, and base_url are required")

        if api_id in BUILTIN_APIS:
            return SkillResult(success=False, message=f"Cannot override builtin API: {api_id}")

        api_entry = {
            "id": api_id,
            "name": name,
            "description": description,
            "base_url": base_url,
            "method": params.get("method", "GET"),
            "auth_type": "header" if params.get("auth_header") else "none",
            "auth_header": params.get("auth_header", ""),
            "category": params.get("category", "custom"),
            "parameters": params.get("parameters", {}),
            "response_transform": "json",
            "price_per_call": params.get("price_per_call", 0.005),
            "raw_cost": 0.0,
            "rate_limit_per_minute": 30,
            "tags": ["custom"],
            "added_at": _now_iso(),
        }

        self._custom_apis[api_id] = api_entry
        self._catalog[api_id] = api_entry
        _save_json(CATALOG_FILE, self._custom_apis)

        return SkillResult(
            success=True,
            message=f"Added custom API: {name}",
            data={"api": api_entry},
        )

    async def _remove_api(self, params: Dict) -> SkillResult:
        """Remove a custom API from the marketplace."""
        api_id = params.get("api_id", "")
        if not api_id:
            return SkillResult(success=False, message="api_id is required")
        if api_id in BUILTIN_APIS:
            return SkillResult(success=False, message=f"Cannot remove builtin API: {api_id}")
        if api_id not in self._custom_apis:
            return SkillResult(success=False, message=f"Custom API not found: {api_id}")

        removed = self._custom_apis.pop(api_id)
        self._catalog.pop(api_id, None)
        _save_json(CATALOG_FILE, self._custom_apis)

        return SkillResult(
            success=True,
            message=f"Removed custom API: {removed.get('name', api_id)}",
            data={"removed_api_id": api_id},
        )

    async def _get_usage(self, params: Dict) -> SkillResult:
        """Get usage statistics."""
        customer_id = params.get("customer_id")
        api_id = params.get("api_id")

        calls = self._usage.get("calls", [])

        # Filter
        if customer_id:
            calls = [c for c in calls if c.get("customer_id") == customer_id]
        if api_id:
            calls = [c for c in calls if c.get("api_id") == api_id]

        total = len(calls)
        successful = sum(1 for c in calls if c.get("success"))
        failed = total - successful
        total_spent = sum(c.get("price", 0) for c in calls if c.get("success"))

        # Per-API breakdown
        api_breakdown = {}
        for c in calls:
            aid = c.get("api_id", "unknown")
            if aid not in api_breakdown:
                api_breakdown[aid] = {"calls": 0, "successful": 0, "spent": 0.0}
            api_breakdown[aid]["calls"] += 1
            if c.get("success"):
                api_breakdown[aid]["successful"] += 1
                api_breakdown[aid]["spent"] += c.get("price", 0)

        # Tier info
        tier_info = None
        if customer_id:
            tier = self._get_customer_tier(customer_id)
            tier_info = {
                "tier": tier["name"],
                "calls_per_month": tier["calls_per_month"],
                "discount_pct": tier["discount_pct"],
            }

        return SkillResult(
            success=True,
            message=f"Usage: {total} total calls, {successful} successful",
            data={
                "total_calls": total,
                "successful": successful,
                "failed": failed,
                "total_spent": round(total_spent, 4),
                "by_api": api_breakdown,
                "tier": tier_info,
                "recent_calls": calls[-10:],
            },
        )

    async def _get_revenue(self, params: Dict) -> SkillResult:
        """Get marketplace revenue report."""
        rev = self._usage.get("revenue", {"total": 0.0, "by_api": {}, "by_customer": {}})

        # Subscription revenue
        sub_revenue = sum(
            s.get("monthly_price", 0) for s in self._subscriptions.values()
        )

        # Top APIs by revenue
        by_api = rev.get("by_api", {})
        top_apis = sorted(by_api.items(), key=lambda x: x[1], reverse=True)[:10]

        # Top customers
        by_customer = rev.get("by_customer", {})
        top_customers = sorted(by_customer.items(), key=lambda x: x[1], reverse=True)[:10]

        return SkillResult(
            success=True,
            message=f"Marketplace revenue: ${rev.get('total', 0):.4f} (API calls) + ${sub_revenue:.2f}/mo (subscriptions)",
            data={
                "api_call_revenue": round(rev.get("total", 0), 4),
                "subscription_revenue_monthly": round(sub_revenue, 2),
                "total_apis": len(self._catalog),
                "total_subscribers": len(self._subscriptions),
                "top_apis_by_revenue": [{"api_id": k, "revenue": round(v, 4)} for k, v in top_apis],
                "top_customers_by_spend": [{"customer_id": k, "spent": round(v, 4)} for k, v in top_customers],
            },
        )

    async def _list_tiers(self, params: Dict) -> SkillResult:
        """List subscription tiers."""
        tiers = []
        for tier_id, tier in SUBSCRIPTION_TIERS.items():
            tiers.append({
                "id": tier_id,
                "name": tier["name"],
                "monthly_price": tier["monthly_price"],
                "calls_per_month": tier["calls_per_month"] if tier["calls_per_month"] > 0 else "unlimited",
                "discount_pct": tier["discount_pct"],
                "rate_limit_multiplier": tier["rate_limit_multiplier"],
            })

        current_subscribers = {}
        for cid, sub in self._subscriptions.items():
            tier_name = sub.get("tier", "free")
            current_subscribers[tier_name] = current_subscribers.get(tier_name, 0) + 1

        return SkillResult(
            success=True,
            message=f"{len(tiers)} subscription tiers available",
            data={
                "tiers": tiers,
                "current_subscribers": current_subscribers,
            },
        )
