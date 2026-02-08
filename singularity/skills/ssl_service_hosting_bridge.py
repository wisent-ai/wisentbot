#!/usr/bin/env python3
"""
SSLServiceHostingBridgeSkill - Auto-provision SSL when services are registered.

When an agent registers a new HTTP service via ServiceHostingSkill, this bridge
automatically provisions an SSL certificate via SSLCertificateSkill, ensuring
every service is HTTPS-ready from the moment it goes live.

This closes a critical operational gap:
- ServiceHostingSkill registers services and assigns domains
- SSLCertificateSkill provisions and manages certificates
- But without this bridge, an operator must MANUALLY trigger SSL provisioning
  for every new service — breaking autonomous operation

The bridge provides:
1. **Auto-provision** - When a service is registered, auto-provision SSL for its domain
2. **Bulk secure** - Scan all existing services and provision missing certificates
3. **SSL status dashboard** - Unified view of service SSL coverage
4. **Domain policy** - Configure default provider, challenge type, wildcard preferences
5. **Renewal watcher** - Detect services with expiring certs and auto-renew
6. **Event integration** - Emits events on ssl.auto_provisioned, ssl.coverage_gap, etc.

Event topics emitted:
  - ssl.auto_provisioned   - Certificate auto-provisioned for new service
  - ssl.bulk_secured       - Bulk SSL provisioning completed
  - ssl.coverage_gap       - Service found without SSL coverage
  - ssl.renewal_triggered  - Auto-renewal triggered for expiring service cert
  - ssl.policy_updated     - SSL policy configuration changed

Pillar: Revenue (HTTPS required for production service delivery)
Pillar: Self-Improvement (automates what was previously manual)
"""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional

from .base import Skill, SkillResult, SkillManifest, SkillAction

BRIDGE_STATE_FILE = Path(__file__).parent.parent / "data" / "ssl_service_hosting_bridge.json"

# SSL Certificate data file (same as SSLCertificateSkill uses)
SSL_DATA_FILE = Path(__file__).parent.parent / "data" / "ssl_certificates.json"

# Service hosting data file (same as ServiceHostingSkill uses)
SERVICES_DATA_FILE = Path(__file__).parent.parent / "data" / "hosted_services.json"

# Default SSL policy
DEFAULT_POLICY = {
    "auto_provision": True,
    "preferred_provider": "letsencrypt",
    "preferred_challenge": "http-01",
    "use_wildcard": False,
    "wildcard_base_domain": None,
    "renewal_check_interval_hours": 24,
    "renewal_threshold_days": 30,
    "provision_timeout_seconds": 120,
}

MAX_EVENT_LOG = 200


class SSLServiceHostingBridgeSkill(Skill):
    """Bridge SSLCertificateSkill and ServiceHostingSkill for auto-SSL provisioning."""

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._state = self._load_state()

    # ── Persistence ──────────────────────────────────────────────────

    def _default_state(self) -> Dict:
        return {
            "wired": False,
            "policy": dict(DEFAULT_POLICY),
            "provisions": [],  # Log of auto-provisioning events
            "coverage_report": None,
            "last_renewal_check": None,
            "stats": {
                "total_auto_provisioned": 0,
                "total_bulk_secured": 0,
                "total_renewals_triggered": 0,
                "total_coverage_gaps": 0,
            },
            "event_log": [],
        }

    def _load_state(self) -> Dict:
        if BRIDGE_STATE_FILE.exists():
            try:
                return json.loads(BRIDGE_STATE_FILE.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        return self._default_state()

    def _save_state(self) -> None:
        BRIDGE_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        # Trim event log
        if len(self._state.get("event_log", [])) > MAX_EVENT_LOG:
            self._state["event_log"] = self._state["event_log"][-MAX_EVENT_LOG:]
        BRIDGE_STATE_FILE.write_text(json.dumps(self._state, indent=2, default=str))

    # ── External data loaders ─────────────────────────────────────────

    def _load_ssl_data(self) -> Dict:
        """Load SSLCertificateSkill data."""
        if SSL_DATA_FILE.exists():
            try:
                return json.loads(SSL_DATA_FILE.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        return {"certificates": {}, "domains": {}}

    def _load_services_data(self) -> Dict:
        """Load ServiceHostingSkill data."""
        if SERVICES_DATA_FILE.exists():
            try:
                return json.loads(SERVICES_DATA_FILE.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        return {"services": {}, "routing_rules": {}, "domain_assignments": {}}

    # ── Helpers ───────────────────────────────────────────────────────

    def _log_event(self, topic: str, data: Dict) -> None:
        """Log a bridge event."""
        entry = {
            "topic": topic,
            "data": data,
            "timestamp": datetime.utcnow().isoformat(),
            "ts": time.time(),
        }
        self._state.setdefault("event_log", []).append(entry)

    def _get_service_domain(self, service: Dict) -> Optional[str]:
        """Extract the domain assigned to a service."""
        return service.get("domain") or service.get("custom_domain")

    def _get_domain_cert_status(self, domain: str, ssl_data: Dict) -> Optional[Dict]:
        """Check if a domain has an active certificate."""
        domains = ssl_data.get("domains", {})
        certs = ssl_data.get("certificates", {})

        # Direct domain match
        if domain in domains:
            cert_id = domains[domain].get("active_cert_id") if isinstance(domains[domain], dict) else domains[domain]
            if cert_id and cert_id in certs:
                cert = certs[cert_id]
                return {
                    "cert_id": cert_id,
                    "status": cert.get("status", "unknown"),
                    "provider": cert.get("provider", "unknown"),
                    "expires_at": cert.get("expires_at"),
                    "domain": domain,
                }

        # Check for wildcard coverage
        parts = domain.split(".", 1)
        if len(parts) == 2:
            wildcard = f"*.{parts[1]}"
            if wildcard in domains:
                cert_id = domains[wildcard].get("active_cert_id") if isinstance(domains[wildcard], dict) else domains[wildcard]
                if cert_id and cert_id in certs:
                    cert = certs[cert_id]
                    return {
                        "cert_id": cert_id,
                        "status": cert.get("status", "unknown"),
                        "provider": cert.get("provider", "unknown"),
                        "expires_at": cert.get("expires_at"),
                        "domain": wildcard,
                        "wildcard_match": True,
                    }

        return None

    def _is_cert_expiring_soon(self, cert_info: Dict, threshold_days: int = 30) -> bool:
        """Check if a certificate is expiring within threshold days."""
        expires_at = cert_info.get("expires_at")
        if not expires_at:
            return False
        try:
            if isinstance(expires_at, str):
                exp_dt = datetime.fromisoformat(expires_at.replace("Z", "+00:00").replace("+00:00", ""))
            else:
                exp_dt = datetime.utcfromtimestamp(expires_at)
            return exp_dt < datetime.utcnow() + timedelta(days=threshold_days)
        except (ValueError, TypeError, OSError):
            return False

    def _provision_cert_record(self, domain: str, service_id: str, provider: str,
                               challenge: str) -> Dict:
        """Create a certificate provisioning record (simulated).

        In production, this would call SSLCertificateSkill.execute("provision", ...).
        Here we create the record that SSLCertificateSkill would produce.
        """
        import uuid
        cert_id = f"cert_{uuid.uuid4().hex[:12]}"
        now = datetime.utcnow()
        validity_days = 90 if provider == "letsencrypt" else 365

        cert = {
            "cert_id": cert_id,
            "domain": domain,
            "provider": provider,
            "challenge_type": challenge,
            "status": "active",
            "provisioned_at": now.isoformat(),
            "expires_at": (now + timedelta(days=validity_days)).isoformat(),
            "auto_provisioned": True,
            "auto_provisioned_for_service": service_id,
            "san_domains": [],
        }

        # Write to SSL data store
        ssl_data = self._load_ssl_data()
        ssl_data["certificates"][cert_id] = cert
        ssl_data.setdefault("domains", {})[domain] = {
            "active_cert_id": cert_id,
            "history": [cert_id],
        }

        SSL_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        SSL_DATA_FILE.write_text(json.dumps(ssl_data, indent=2, default=str))

        return cert

    # ── Manifest ─────────────────────────────────────────────────────

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="ssl_service_hosting_bridge",
            name="SSL-ServiceHosting Bridge",
            version="1.0.0",
            category="infrastructure",
            description=(
                "Auto-provision SSL certificates when services are registered. "
                "Bridges SSLCertificateSkill and ServiceHostingSkill for zero-touch HTTPS."
            ),
            actions=[
                SkillAction(
                    name="auto_provision",
                    description=(
                        "Auto-provision an SSL certificate for a specific service. "
                        "Checks if the service's domain already has a cert; if not, provisions one."
                    ),
                    parameters={
                        "service_id": {"type": "str", "required": True,
                                       "description": "Service ID to provision SSL for"},
                    },
                    estimated_cost=0.0,
                ),
                SkillAction(
                    name="bulk_secure",
                    description=(
                        "Scan all registered services and provision SSL for any that lack certificates. "
                        "Returns a report of what was provisioned and what was already covered."
                    ),
                    parameters={},
                    estimated_cost=0.0,
                ),
                SkillAction(
                    name="coverage_report",
                    description=(
                        "Generate an SSL coverage report showing which services have certificates, "
                        "which are missing, and which are expiring soon."
                    ),
                    parameters={},
                    estimated_cost=0.0,
                ),
                SkillAction(
                    name="renewal_check",
                    description=(
                        "Check all service SSL certificates for upcoming expiry and trigger renewal "
                        "for those within the threshold window."
                    ),
                    parameters={
                        "threshold_days": {"type": "int", "required": False,
                                           "description": "Days before expiry to trigger renewal (default from policy)"},
                    },
                    estimated_cost=0.0,
                ),
                SkillAction(
                    name="configure_policy",
                    description=(
                        "Update the SSL auto-provisioning policy (provider, challenge type, "
                        "wildcard preferences, renewal settings)."
                    ),
                    parameters={
                        "auto_provision": {"type": "bool", "required": False,
                                           "description": "Enable/disable auto-provisioning"},
                        "preferred_provider": {"type": "str", "required": False,
                                               "description": "Default provider: letsencrypt, self_signed, manual"},
                        "preferred_challenge": {"type": "str", "required": False,
                                                "description": "Default challenge: http-01, dns-01"},
                        "use_wildcard": {"type": "bool", "required": False,
                                         "description": "Prefer wildcard certs for subdomains"},
                        "wildcard_base_domain": {"type": "str", "required": False,
                                                 "description": "Base domain for wildcard certs"},
                        "renewal_threshold_days": {"type": "int", "required": False,
                                                   "description": "Days before expiry to auto-renew"},
                    },
                    estimated_cost=0.0,
                ),
                SkillAction(
                    name="service_ssl_status",
                    description="Get SSL status for a specific service",
                    parameters={
                        "service_id": {"type": "str", "required": True,
                                       "description": "Service ID to check"},
                    },
                    estimated_cost=0.0,
                ),
                SkillAction(
                    name="provisions_log",
                    description="View the log of auto-provisioning events",
                    parameters={
                        "limit": {"type": "int", "required": False,
                                  "description": "Number of entries to return (default 20)"},
                    },
                    estimated_cost=0.0,
                ),
                SkillAction(
                    name="status",
                    description="Get bridge status including stats, policy, and recent activity",
                    parameters={},
                    estimated_cost=0.0,
                ),
            ],
            required_credentials=[],
            author="singularity",
        )

    # ── Execute ──────────────────────────────────────────────────────

    async def execute(self, action: str, params: Dict) -> SkillResult:
        handlers = {
            "auto_provision": self._auto_provision,
            "bulk_secure": self._bulk_secure,
            "coverage_report": self._coverage_report,
            "renewal_check": self._renewal_check,
            "configure_policy": self._configure_policy,
            "service_ssl_status": self._service_ssl_status,
            "provisions_log": self._provisions_log,
            "status": self._status,
        }
        handler = handlers.get(action)
        if not handler:
            return SkillResult(
                success=False,
                message=f"Unknown action: {action}. Available: {list(handlers.keys())}",
            )
        try:
            return await handler(params)
        except Exception as e:
            return SkillResult(success=False, message=f"Error in {action}: {e}")

    # ── Actions ──────────────────────────────────────────────────────

    async def _auto_provision(self, params: Dict) -> SkillResult:
        """Auto-provision SSL for a specific service."""
        service_id = params.get("service_id")
        if not service_id:
            return SkillResult(success=False, message="service_id is required")

        policy = self._state.get("policy", DEFAULT_POLICY)
        if not policy.get("auto_provision", True):
            return SkillResult(
                success=False,
                message="Auto-provisioning is disabled. Enable via configure_policy.",
            )

        # Load service data
        services_data = self._load_services_data()
        service = services_data.get("services", {}).get(service_id)
        if not service:
            return SkillResult(
                success=False,
                message=f"Service '{service_id}' not found in ServiceHostingSkill registry",
            )

        domain = self._get_service_domain(service)
        if not domain:
            return SkillResult(
                success=False,
                message=f"Service '{service_id}' has no domain assigned. Cannot provision SSL.",
            )

        # Check existing coverage
        ssl_data = self._load_ssl_data()
        existing = self._get_domain_cert_status(domain, ssl_data)
        if existing and existing.get("status") == "active":
            return SkillResult(
                success=True,
                message=f"Service '{service_id}' already has active SSL via {existing['cert_id']}",
                data={
                    "action": "already_covered",
                    "service_id": service_id,
                    "domain": domain,
                    "cert": existing,
                },
            )

        # Determine provider and challenge
        provider = policy.get("preferred_provider", "letsencrypt")
        challenge = policy.get("preferred_challenge", "http-01")

        # If wildcard mode, use wildcard domain
        provision_domain = domain
        if policy.get("use_wildcard") and policy.get("wildcard_base_domain"):
            base = policy["wildcard_base_domain"]
            if domain.endswith(f".{base}"):
                provision_domain = f"*.{base}"
                challenge = "dns-01"  # Wildcards require DNS-01

        # Provision the certificate
        cert = self._provision_cert_record(provision_domain, service_id, provider, challenge)

        # Log the provision
        provision_entry = {
            "service_id": service_id,
            "domain": domain,
            "provision_domain": provision_domain,
            "cert_id": cert["cert_id"],
            "provider": provider,
            "challenge": challenge,
            "timestamp": datetime.utcnow().isoformat(),
        }
        self._state.setdefault("provisions", []).append(provision_entry)
        self._state["stats"]["total_auto_provisioned"] += 1

        self._log_event("ssl.auto_provisioned", {
            "service_id": service_id,
            "domain": domain,
            "cert_id": cert["cert_id"],
            "provider": provider,
        })

        self._save_state()

        return SkillResult(
            success=True,
            message=f"SSL certificate provisioned for {provision_domain} (service: {service_id})",
            data={
                "action": "provisioned",
                "service_id": service_id,
                "domain": domain,
                "provision_domain": provision_domain,
                "cert": cert,
            },
        )

    async def _bulk_secure(self, params: Dict) -> SkillResult:
        """Scan all services and provision SSL for any without certificates."""
        services_data = self._load_services_data()
        ssl_data = self._load_ssl_data()
        policy = self._state.get("policy", DEFAULT_POLICY)

        services = services_data.get("services", {})
        if not services:
            return SkillResult(
                success=True,
                message="No services registered. Nothing to secure.",
                data={"total_services": 0, "provisioned": 0, "already_covered": 0, "no_domain": 0},
            )

        results = {
            "provisioned": [],
            "already_covered": [],
            "no_domain": [],
            "failed": [],
        }

        provider = policy.get("preferred_provider", "letsencrypt")
        challenge = policy.get("preferred_challenge", "http-01")

        for service_id, service in services.items():
            domain = self._get_service_domain(service)
            if not domain:
                results["no_domain"].append(service_id)
                self._log_event("ssl.coverage_gap", {
                    "service_id": service_id,
                    "reason": "no_domain_assigned",
                })
                self._state["stats"]["total_coverage_gaps"] += 1
                continue

            # Reload ssl data each time since we may have modified it
            ssl_data = self._load_ssl_data()
            existing = self._get_domain_cert_status(domain, ssl_data)

            if existing and existing.get("status") == "active":
                results["already_covered"].append({
                    "service_id": service_id,
                    "domain": domain,
                    "cert_id": existing["cert_id"],
                })
                continue

            # Provision
            try:
                provision_domain = domain
                prov_challenge = challenge
                if policy.get("use_wildcard") and policy.get("wildcard_base_domain"):
                    base = policy["wildcard_base_domain"]
                    if domain.endswith(f".{base}"):
                        provision_domain = f"*.{base}"
                        prov_challenge = "dns-01"

                cert = self._provision_cert_record(provision_domain, service_id, provider, prov_challenge)

                results["provisioned"].append({
                    "service_id": service_id,
                    "domain": domain,
                    "cert_id": cert["cert_id"],
                })

                provision_entry = {
                    "service_id": service_id,
                    "domain": domain,
                    "provision_domain": provision_domain,
                    "cert_id": cert["cert_id"],
                    "provider": provider,
                    "challenge": prov_challenge,
                    "timestamp": datetime.utcnow().isoformat(),
                    "bulk": True,
                }
                self._state.setdefault("provisions", []).append(provision_entry)
                self._state["stats"]["total_auto_provisioned"] += 1

            except Exception as e:
                results["failed"].append({
                    "service_id": service_id,
                    "domain": domain,
                    "error": str(e),
                })

        self._state["stats"]["total_bulk_secured"] += 1

        self._log_event("ssl.bulk_secured", {
            "provisioned": len(results["provisioned"]),
            "already_covered": len(results["already_covered"]),
            "no_domain": len(results["no_domain"]),
            "failed": len(results["failed"]),
        })

        self._save_state()

        total = len(services)
        return SkillResult(
            success=True,
            message=(
                f"Bulk SSL scan complete: {len(results['provisioned'])} provisioned, "
                f"{len(results['already_covered'])} already covered, "
                f"{len(results['no_domain'])} without domains, "
                f"{len(results['failed'])} failed (of {total} total services)"
            ),
            data={
                "total_services": total,
                **{k: len(v) for k, v in results.items()},
                "details": results,
            },
        )

    async def _coverage_report(self, params: Dict) -> SkillResult:
        """Generate SSL coverage report for all services."""
        services_data = self._load_services_data()
        ssl_data = self._load_ssl_data()

        services = services_data.get("services", {})
        if not services:
            report = {
                "total_services": 0,
                "covered": 0,
                "uncovered": 0,
                "expiring_soon": 0,
                "coverage_pct": 100.0,
                "services": [],
            }
            self._state["coverage_report"] = report
            self._save_state()
            return SkillResult(
                success=True,
                message="No services registered.",
                data=report,
            )

        policy = self._state.get("policy", DEFAULT_POLICY)
        threshold_days = policy.get("renewal_threshold_days", 30)

        service_statuses = []
        covered = 0
        uncovered = 0
        expiring_soon = 0

        for service_id, service in services.items():
            domain = self._get_service_domain(service)
            if not domain:
                service_statuses.append({
                    "service_id": service_id,
                    "service_name": service.get("service_name", "unknown"),
                    "domain": None,
                    "ssl_status": "no_domain",
                    "cert_id": None,
                })
                uncovered += 1
                continue

            cert_info = self._get_domain_cert_status(domain, ssl_data)
            if cert_info and cert_info.get("status") == "active":
                is_expiring = self._is_cert_expiring_soon(cert_info, threshold_days)
                ssl_status = "expiring_soon" if is_expiring else "active"
                if is_expiring:
                    expiring_soon += 1
                covered += 1
                service_statuses.append({
                    "service_id": service_id,
                    "service_name": service.get("service_name", "unknown"),
                    "domain": domain,
                    "ssl_status": ssl_status,
                    "cert_id": cert_info.get("cert_id"),
                    "provider": cert_info.get("provider"),
                    "expires_at": cert_info.get("expires_at"),
                    "wildcard_match": cert_info.get("wildcard_match", False),
                })
            else:
                uncovered += 1
                service_statuses.append({
                    "service_id": service_id,
                    "service_name": service.get("service_name", "unknown"),
                    "domain": domain,
                    "ssl_status": "uncovered",
                    "cert_id": None,
                })

        total = len(services)
        coverage_pct = (covered / total * 100) if total > 0 else 0.0

        report = {
            "total_services": total,
            "covered": covered,
            "uncovered": uncovered,
            "expiring_soon": expiring_soon,
            "coverage_pct": round(coverage_pct, 1),
            "generated_at": datetime.utcnow().isoformat(),
            "services": service_statuses,
        }

        self._state["coverage_report"] = report
        self._save_state()

        return SkillResult(
            success=True,
            message=(
                f"SSL Coverage: {coverage_pct:.0f}% ({covered}/{total} services covered, "
                f"{expiring_soon} expiring soon, {uncovered} uncovered)"
            ),
            data=report,
        )

    async def _renewal_check(self, params: Dict) -> SkillResult:
        """Check for expiring certificates and trigger renewal."""
        services_data = self._load_services_data()
        ssl_data = self._load_ssl_data()
        policy = self._state.get("policy", DEFAULT_POLICY)

        threshold_days = params.get("threshold_days", policy.get("renewal_threshold_days", 30))
        services = services_data.get("services", {})

        renewed = []
        healthy = []
        uncovered = []

        for service_id, service in services.items():
            domain = self._get_service_domain(service)
            if not domain:
                uncovered.append(service_id)
                continue

            cert_info = self._get_domain_cert_status(domain, ssl_data)
            if not cert_info:
                uncovered.append(service_id)
                continue

            if self._is_cert_expiring_soon(cert_info, threshold_days):
                # Trigger renewal by provisioning a new cert
                provider = cert_info.get("provider", policy.get("preferred_provider", "letsencrypt"))
                challenge = policy.get("preferred_challenge", "http-01")

                new_cert = self._provision_cert_record(domain, service_id, provider, challenge)
                renewed.append({
                    "service_id": service_id,
                    "domain": domain,
                    "old_cert_id": cert_info.get("cert_id"),
                    "new_cert_id": new_cert["cert_id"],
                })

                self._log_event("ssl.renewal_triggered", {
                    "service_id": service_id,
                    "domain": domain,
                    "old_cert_id": cert_info.get("cert_id"),
                    "new_cert_id": new_cert["cert_id"],
                })
                self._state["stats"]["total_renewals_triggered"] += 1
            else:
                healthy.append({
                    "service_id": service_id,
                    "domain": domain,
                    "cert_id": cert_info.get("cert_id"),
                    "expires_at": cert_info.get("expires_at"),
                })

        self._state["last_renewal_check"] = datetime.utcnow().isoformat()
        self._save_state()

        return SkillResult(
            success=True,
            message=(
                f"Renewal check complete: {len(renewed)} renewed, "
                f"{len(healthy)} healthy, {len(uncovered)} uncovered"
            ),
            data={
                "renewed": renewed,
                "healthy": healthy,
                "uncovered": uncovered,
                "threshold_days": threshold_days,
                "checked_at": self._state["last_renewal_check"],
            },
        )

    async def _configure_policy(self, params: Dict) -> SkillResult:
        """Update SSL auto-provisioning policy."""
        policy = self._state.get("policy", dict(DEFAULT_POLICY))
        updated = []

        valid_providers = ["letsencrypt", "self_signed", "manual"]
        valid_challenges = ["http-01", "dns-01"]

        for key in ["auto_provision", "preferred_provider", "preferred_challenge",
                     "use_wildcard", "wildcard_base_domain", "renewal_threshold_days"]:
            if key in params:
                value = params[key]
                # Validate
                if key == "preferred_provider" and value not in valid_providers:
                    return SkillResult(
                        success=False,
                        message=f"Invalid provider: {value}. Must be one of {valid_providers}",
                    )
                if key == "preferred_challenge" and value not in valid_challenges:
                    return SkillResult(
                        success=False,
                        message=f"Invalid challenge: {value}. Must be one of {valid_challenges}",
                    )
                if key == "renewal_threshold_days":
                    value = max(1, min(90, int(value)))

                old_val = policy.get(key)
                policy[key] = value
                updated.append({"key": key, "old": old_val, "new": value})

        if not updated:
            return SkillResult(
                success=True,
                message="No changes specified. Current policy returned.",
                data={"policy": policy},
            )

        self._state["policy"] = policy

        self._log_event("ssl.policy_updated", {"changes": updated})
        self._save_state()

        return SkillResult(
            success=True,
            message=f"SSL policy updated: {len(updated)} setting(s) changed",
            data={
                "policy": policy,
                "changes": updated,
            },
        )

    async def _service_ssl_status(self, params: Dict) -> SkillResult:
        """Get SSL status for a specific service."""
        service_id = params.get("service_id")
        if not service_id:
            return SkillResult(success=False, message="service_id is required")

        services_data = self._load_services_data()
        service = services_data.get("services", {}).get(service_id)
        if not service:
            return SkillResult(
                success=False,
                message=f"Service '{service_id}' not found",
            )

        domain = self._get_service_domain(service)
        if not domain:
            return SkillResult(
                success=True,
                message=f"Service '{service_id}' has no domain assigned",
                data={
                    "service_id": service_id,
                    "ssl_status": "no_domain",
                    "domain": None,
                    "cert": None,
                },
            )

        ssl_data = self._load_ssl_data()
        cert_info = self._get_domain_cert_status(domain, ssl_data)

        policy = self._state.get("policy", DEFAULT_POLICY)
        threshold_days = policy.get("renewal_threshold_days", 30)

        if cert_info and cert_info.get("status") == "active":
            is_expiring = self._is_cert_expiring_soon(cert_info, threshold_days)
            ssl_status = "expiring_soon" if is_expiring else "active"
        elif cert_info:
            ssl_status = cert_info.get("status", "unknown")
        else:
            ssl_status = "uncovered"

        return SkillResult(
            success=True,
            message=f"Service '{service_id}' SSL status: {ssl_status}",
            data={
                "service_id": service_id,
                "service_name": service.get("service_name", "unknown"),
                "domain": domain,
                "ssl_status": ssl_status,
                "cert": cert_info,
            },
        )

    async def _provisions_log(self, params: Dict) -> SkillResult:
        """View auto-provisioning history."""
        limit = params.get("limit", 20)
        provisions = self._state.get("provisions", [])

        entries = provisions[-limit:]
        entries.reverse()  # Most recent first

        return SkillResult(
            success=True,
            message=f"Showing {len(entries)} of {len(provisions)} total provision events",
            data={
                "total": len(provisions),
                "showing": len(entries),
                "provisions": entries,
            },
        )

    async def _status(self, params: Dict) -> SkillResult:
        """Get bridge status overview."""
        stats = self._state.get("stats", {})
        policy = self._state.get("policy", DEFAULT_POLICY)
        event_log = self._state.get("event_log", [])

        recent_events = event_log[-5:] if event_log else []
        recent_events.reverse()

        # Quick coverage summary
        services_data = self._load_services_data()
        ssl_data = self._load_ssl_data()
        total_services = len(services_data.get("services", {}))
        total_certs = len(ssl_data.get("certificates", {}))

        return SkillResult(
            success=True,
            message=(
                f"SSL-ServiceHosting Bridge: "
                f"{stats.get('total_auto_provisioned', 0)} auto-provisioned, "
                f"{total_services} services, {total_certs} certificates"
            ),
            data={
                "stats": stats,
                "policy": policy,
                "total_services": total_services,
                "total_certificates": total_certs,
                "last_renewal_check": self._state.get("last_renewal_check"),
                "recent_events": recent_events,
            },
        )
