#!/usr/bin/env python3
"""
Public Service Deployer Skill - Deploy agent services with public URLs.

Built in response to Feature Request #130: agents need a way to deploy their
Docker-based HTTP services and make them publicly accessible at URLs like
`adam.singularity.wisent.ai`.

This skill bridges the gap between having a Dockerfile and having a live,
publicly-accessible, billed service. It:

1. Deploys Docker containers from images or Dockerfiles
2. Assigns public subdomains (e.g., adam.singularity.wisent.ai)
3. Generates Caddy reverse proxy configs with automatic TLS
4. Integrates with APIGatewaySkill for per-request billing
5. Manages full lifecycle: deploy, redeploy, stop, restart, logs, status
6. Tracks deployment costs and uptime for profitability analysis

Architecture:
  PublicServiceDeployerSkill manages the full deployment pipeline:
  - Container orchestration via Docker CLI
  - Reverse proxy config generation (Caddy for auto-TLS)
  - DNS record management (Cloudflare API or manual config)
  - Health monitoring with auto-restart
  - Integration with ServiceHostingSkill for service registry
  - Integration with APIGatewaySkill for per-request billing

Part of the Revenue Generation + Replication pillars:
  - Revenue: enables agents to offer paid services at public URLs
  - Replication: provides deployment infra that replicas can use
"""

import json
import os
import subprocess
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

from .base import Skill, SkillResult, SkillManifest, SkillAction

DATA_DIR = Path(__file__).parent.parent / "data"
DEPLOYMENTS_FILE = DATA_DIR / "public_deployments.json"
PROXY_CONFIG_DIR = DATA_DIR / "proxy_configs"

# Default settings
DEFAULT_BASE_DOMAIN = "singularity.wisent.ai"
DEFAULT_INTERNAL_PORT = 8080
DEFAULT_MEMORY_LIMIT = "512m"
DEFAULT_CPU_LIMIT = "0.5"
HEALTH_CHECK_INTERVAL = 30  # seconds
MAX_RESTART_ATTEMPTS = 3


def _load_deployments() -> Dict:
    """Load deployment state from disk."""
    if DEPLOYMENTS_FILE.exists():
        try:
            return json.loads(DEPLOYMENTS_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {
        "deployments": {},
        "domain_map": {},  # domain -> deployment_id
        "port_allocations": {},  # deployment_id -> host_port
        "stats": {
            "total_deployments": 0,
            "total_teardowns": 0,
            "total_redeploys": 0,
        },
    }


def _save_deployments(data: Dict) -> None:
    """Persist deployment state."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DEPLOYMENTS_FILE.write_text(json.dumps(data, indent=2, default=str))


def _allocate_port(data: Dict) -> int:
    """Allocate next available host port for a new deployment."""
    used_ports = set(data.get("port_allocations", {}).values())
    # Start from port 10000 to avoid conflicts
    port = 10000
    while port in used_ports:
        port += 1
    return port


def _run_docker_cmd(args: List[str], timeout: int = 60) -> Dict:
    """Run a docker command and return result."""
    cmd = ["docker"] + args
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "stdout": "", "stderr": "Command timed out", "returncode": -1}
    except FileNotFoundError:
        return {"success": False, "stdout": "", "stderr": "Docker not found", "returncode": -1}
    except Exception as e:
        return {"success": False, "stdout": "", "stderr": str(e), "returncode": -1}


def _generate_caddy_config(deployments: Dict, base_domain: str) -> str:
    """Generate Caddyfile for all active deployments with automatic TLS."""
    lines = []
    lines.append("# Auto-generated Caddyfile for singularity agent services")
    lines.append(f"# Generated: {datetime.utcnow().isoformat()}")
    lines.append("")

    for dep_id, dep in deployments.get("deployments", {}).items():
        if dep.get("status") != "running":
            continue
        domain = dep.get("public_url", "").replace("https://", "").replace("http://", "")
        host_port = deployments.get("port_allocations", {}).get(dep_id)
        if not domain or not host_port:
            continue

        lines.append(f"{domain} {{")
        lines.append(f"    reverse_proxy localhost:{host_port}")
        lines.append("")
        # Add rate limiting header for billing integration
        lines.append("    header {")
        lines.append(f"        X-Service-Id {dep.get('service_id', dep_id)}")
        lines.append(f"        X-Agent-Name {dep.get('agent_name', 'unknown')}")
        lines.append("    }")
        lines.append("")
        # Health check endpoint passthrough
        health_path = dep.get("health_check_path", "/health")
        lines.append(f"    # Health check: {health_path}")
        lines.append("    log {")
        lines.append("        output file /var/log/caddy/access.log")
        lines.append("    }")
        lines.append("}")
        lines.append("")

    return "\n".join(lines)


def _generate_nginx_config(deployments: Dict, base_domain: str) -> str:
    """Generate nginx config as alternative to Caddy."""
    lines = []
    lines.append("# Auto-generated nginx config for singularity agent services")
    lines.append(f"# Generated: {datetime.utcnow().isoformat()}")
    lines.append("")

    for dep_id, dep in deployments.get("deployments", {}).items():
        if dep.get("status") != "running":
            continue
        domain = dep.get("public_url", "").replace("https://", "").replace("http://", "")
        host_port = deployments.get("port_allocations", {}).get(dep_id)
        if not domain or not host_port:
            continue

        lines.append(f"server {{")
        lines.append(f"    server_name {domain};")
        lines.append(f"    listen 443 ssl;")
        lines.append(f"")
        lines.append(f"    location / {{")
        lines.append(f"        proxy_pass http://127.0.0.1:{host_port};")
        lines.append(f"        proxy_set_header Host $host;")
        lines.append(f"        proxy_set_header X-Real-IP $remote_addr;")
        lines.append(f"        proxy_set_header X-Service-Id {dep.get('service_id', dep_id)};")
        lines.append(f"        proxy_set_header X-Agent-Name {dep.get('agent_name', 'unknown')};")
        lines.append(f"    }}")
        lines.append(f"}}")
        lines.append("")

    return "\n".join(lines)


def _generate_docker_compose(deployments: Dict) -> Dict:
    """Generate docker-compose.yml content for all active deployments."""
    services = {}
    for dep_id, dep in deployments.get("deployments", {}).items():
        if dep.get("status") not in ("running", "deploying"):
            continue
        host_port = deployments.get("port_allocations", {}).get(dep_id)
        if not host_port:
            continue

        service_name = dep.get("container_name", f"svc-{dep_id[:8]}")
        service_config = {
            "image": dep.get("docker_image"),
            "container_name": service_name,
            "ports": [f"{host_port}:{dep.get('internal_port', DEFAULT_INTERNAL_PORT)}"],
            "restart": "unless-stopped",
            "deploy": {
                "resources": {
                    "limits": {
                        "memory": dep.get("memory_limit", DEFAULT_MEMORY_LIMIT),
                        "cpus": dep.get("cpu_limit", DEFAULT_CPU_LIMIT),
                    }
                }
            },
        }
        env_vars = dep.get("env_vars", {})
        if env_vars:
            service_config["environment"] = env_vars

        labels = {
            "singularity.agent": dep.get("agent_name", "unknown"),
            "singularity.service_id": dep.get("service_id", dep_id),
            "singularity.deployment_id": dep_id,
        }
        service_config["labels"] = labels

        # Health check
        health_path = dep.get("health_check_path", "/health")
        internal_port = dep.get("internal_port", DEFAULT_INTERNAL_PORT)
        service_config["healthcheck"] = {
            "test": ["CMD", "curl", "-f", f"http://localhost:{internal_port}{health_path}"],
            "interval": f"{HEALTH_CHECK_INTERVAL}s",
            "timeout": "10s",
            "retries": 3,
        }

        services[service_name] = service_config

    return {
        "version": "3.8",
        "services": services,
    }


class PublicServiceDeployerSkill(Skill):
    """
    Deploy agent services with public URLs, routing, and billing.

    This skill answers the question: "I have a Dockerfile and endpoints ready -
    how do I make them publicly accessible and start generating revenue?"

    Built in response to Feature Request #130 from agent Adam.
    """

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="public_service_deployer",
            name="Public Service Deployer",
            version="1.0.0",
            category="infrastructure",
            description=(
                "Deploy Docker-based agent services with public URLs, "
                "automatic TLS, reverse proxy routing, and billing integration. "
                "Turn a Dockerfile into a revenue-generating public endpoint."
            ),
            actions=[
                SkillAction(
                    name="deploy",
                    description=(
                        "Deploy a Docker image as a publicly accessible service. "
                        "Assigns a subdomain, configures routing, and starts the container."
                    ),
                    parameters={
                        "agent_name": {"type": "str", "required": True,
                                       "description": "Agent deploying the service (used for subdomain)"},
                        "service_name": {"type": "str", "required": True,
                                         "description": "Human-readable service name"},
                        "docker_image": {"type": "str", "required": True,
                                         "description": "Docker image to deploy (e.g., ghcr.io/wisent-ai/adam-services:latest)"},
                        "endpoints": {"type": "list", "required": False,
                                      "description": "List of endpoint dicts: {path, method, price_per_request, description}"},
                        "internal_port": {"type": "int", "required": False,
                                          "description": f"Port the service listens on inside the container (default {DEFAULT_INTERNAL_PORT})"},
                        "env_vars": {"type": "dict", "required": False,
                                     "description": "Environment variables to pass to the container"},
                        "memory_limit": {"type": "str", "required": False,
                                         "description": f"Memory limit (default {DEFAULT_MEMORY_LIMIT})"},
                        "cpu_limit": {"type": "str", "required": False,
                                      "description": f"CPU limit (default {DEFAULT_CPU_LIMIT})"},
                        "health_check_path": {"type": "str", "required": False,
                                              "description": "Health check endpoint path (default /health)"},
                        "base_domain": {"type": "str", "required": False,
                                        "description": f"Base domain for URL (default {DEFAULT_BASE_DOMAIN})"},
                        "custom_domain": {"type": "str", "required": False,
                                          "description": "Custom domain instead of auto-generated subdomain"},
                        "replicas": {"type": "int", "required": False,
                                     "description": "Number of container replicas (default 1)"},
                    },
                    estimated_cost=0.05,
                ),
                SkillAction(
                    name="redeploy",
                    description="Redeploy a service with a new image version (zero-downtime if possible)",
                    parameters={
                        "deployment_id": {"type": "str", "required": True,
                                          "description": "ID of the deployment to update"},
                        "docker_image": {"type": "str", "required": False,
                                         "description": "New Docker image (or pulls latest of current)"},
                        "env_vars": {"type": "dict", "required": False,
                                     "description": "Updated environment variables"},
                    },
                    estimated_cost=0.03,
                ),
                SkillAction(
                    name="stop",
                    description="Stop a running deployment and free its resources",
                    parameters={
                        "deployment_id": {"type": "str", "required": True,
                                          "description": "ID of the deployment to stop"},
                    },
                    estimated_cost=0.0,
                ),
                SkillAction(
                    name="restart",
                    description="Restart a stopped or failed deployment",
                    parameters={
                        "deployment_id": {"type": "str", "required": True,
                                          "description": "ID of the deployment to restart"},
                    },
                    estimated_cost=0.01,
                ),
                SkillAction(
                    name="status",
                    description="Get status and health of a specific deployment or all deployments",
                    parameters={
                        "deployment_id": {"type": "str", "required": False,
                                          "description": "Specific deployment ID (or all if omitted)"},
                        "agent_name": {"type": "str", "required": False,
                                       "description": "Filter by agent name"},
                    },
                    estimated_cost=0.0,
                ),
                SkillAction(
                    name="logs",
                    description="Retrieve recent logs from a deployed service container",
                    parameters={
                        "deployment_id": {"type": "str", "required": True,
                                          "description": "Deployment to get logs from"},
                        "tail": {"type": "int", "required": False,
                                 "description": "Number of log lines (default 100)"},
                    },
                    estimated_cost=0.0,
                ),
                SkillAction(
                    name="generate_routing_config",
                    description="Generate reverse proxy config (Caddy/nginx) for all active deployments",
                    parameters={
                        "proxy_type": {"type": "str", "required": False,
                                       "description": "caddy or nginx (default caddy)"},
                        "base_domain": {"type": "str", "required": False,
                                        "description": f"Base domain (default {DEFAULT_BASE_DOMAIN})"},
                    },
                    estimated_cost=0.0,
                ),
                SkillAction(
                    name="generate_compose",
                    description="Generate docker-compose.yml for all active deployments",
                    parameters={},
                    estimated_cost=0.0,
                ),
                SkillAction(
                    name="setup_billing",
                    description=(
                        "Configure per-request billing for a deployment via APIGatewaySkill. "
                        "Creates API key scopes and usage tracking for the service."
                    ),
                    parameters={
                        "deployment_id": {"type": "str", "required": True,
                                          "description": "Deployment to set up billing for"},
                        "price_per_request": {"type": "float", "required": False,
                                              "description": "Default price per request in USD (default 0.01)"},
                        "free_tier_requests": {"type": "int", "required": False,
                                               "description": "Free requests per day before billing (default 10)"},
                    },
                    estimated_cost=0.0,
                ),
                SkillAction(
                    name="get_deployment_stats",
                    description="Get aggregated stats: total deployments, uptime, costs, revenue",
                    parameters={
                        "agent_name": {"type": "str", "required": False,
                                       "description": "Filter by agent name"},
                    },
                    estimated_cost=0.0,
                ),
            ],
            required_credentials=[],
            author="singularity",
        )

    async def execute(self, action: str, params: Dict) -> SkillResult:
        """Route to the appropriate action handler."""
        handlers = {
            "deploy": self._deploy,
            "redeploy": self._redeploy,
            "stop": self._stop,
            "restart": self._restart,
            "status": self._status,
            "logs": self._logs,
            "generate_routing_config": self._generate_routing_config,
            "generate_compose": self._generate_compose,
            "setup_billing": self._setup_billing,
            "get_deployment_stats": self._get_deployment_stats,
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

    async def _deploy(self, params: Dict) -> SkillResult:
        """Deploy a Docker image as a publicly accessible service."""
        agent_name = params.get("agent_name")
        service_name = params.get("service_name")
        docker_image = params.get("docker_image")

        if not agent_name or not service_name or not docker_image:
            return SkillResult(
                success=False,
                message="agent_name, service_name, and docker_image are required",
            )

        data = _load_deployments()
        deployment_id = str(uuid.uuid4())[:12]
        base_domain = params.get("base_domain", DEFAULT_BASE_DOMAIN)
        custom_domain = params.get("custom_domain")
        internal_port = params.get("internal_port", DEFAULT_INTERNAL_PORT)
        memory_limit = params.get("memory_limit", DEFAULT_MEMORY_LIMIT)
        cpu_limit = params.get("cpu_limit", DEFAULT_CPU_LIMIT)
        health_check_path = params.get("health_check_path", "/health")
        env_vars = params.get("env_vars", {})
        endpoints = params.get("endpoints", [])
        replicas = params.get("replicas", 1)

        # Generate public URL
        if custom_domain:
            public_domain = custom_domain
        else:
            # Sanitize agent name for subdomain
            subdomain = agent_name.lower().replace(" ", "-").replace("_", "-")
            public_domain = f"{subdomain}.{base_domain}"

        public_url = f"https://{public_domain}"

        # Allocate host port
        host_port = _allocate_port(data)

        # Container name
        container_name = f"svc-{agent_name.lower().replace(' ', '-')}-{deployment_id}"

        # Build deployment record
        deployment = {
            "deployment_id": deployment_id,
            "agent_name": agent_name,
            "service_name": service_name,
            "service_id": f"{agent_name.lower()}-{service_name.lower().replace(' ', '-')}",
            "docker_image": docker_image,
            "container_name": container_name,
            "internal_port": internal_port,
            "host_port": host_port,
            "memory_limit": memory_limit,
            "cpu_limit": cpu_limit,
            "health_check_path": health_check_path,
            "env_vars": env_vars,
            "endpoints": endpoints,
            "replicas": replicas,
            "public_url": public_url,
            "public_domain": public_domain,
            "status": "deploying",
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "last_health_check": None,
            "restart_count": 0,
            "billing": {
                "enabled": False,
                "price_per_request": 0.01,
                "free_tier_requests": 10,
                "total_requests": 0,
                "total_revenue": 0.0,
            },
            "deploy_history": [
                {
                    "action": "deploy",
                    "image": docker_image,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            ],
        }

        # Try to pull the image
        pull_result = _run_docker_cmd(["pull", docker_image], timeout=120)

        # Try to start the container
        run_args = [
            "run", "-d",
            "--name", container_name,
            "-p", f"{host_port}:{internal_port}",
            "--memory", memory_limit,
            "--cpus", cpu_limit,
            "--restart", "unless-stopped",
            "--label", f"singularity.deployment_id={deployment_id}",
            "--label", f"singularity.agent={agent_name}",
            "--label", f"singularity.service={service_name}",
        ]

        # Add environment variables
        for key, value in env_vars.items():
            run_args.extend(["-e", f"{key}={value}"])

        run_args.append(docker_image)
        run_result = _run_docker_cmd(run_args, timeout=60)

        if run_result["success"]:
            deployment["status"] = "running"
            deployment["container_id"] = run_result["stdout"][:12]
        else:
            # Container start failed but we still record the deployment
            deployment["status"] = "failed"
            deployment["error"] = run_result["stderr"]

        # Save deployment state
        data["deployments"][deployment_id] = deployment
        data["domain_map"][public_domain] = deployment_id
        data["port_allocations"][deployment_id] = host_port
        data["stats"]["total_deployments"] = data["stats"].get("total_deployments", 0) + 1
        _save_deployments(data)

        # Generate routing config
        caddy_config = _generate_caddy_config(data, base_domain)
        PROXY_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        caddy_file = PROXY_CONFIG_DIR / "Caddyfile"
        caddy_file.write_text(caddy_config)

        return SkillResult(
            success=True,
            message=(
                f"{'Deployed' if deployment['status'] == 'running' else 'Registered'} "
                f"service '{service_name}' for agent '{agent_name}'\n"
                f"Public URL: {public_url}\n"
                f"Status: {deployment['status']}\n"
                f"Container: {container_name}\n"
                f"Host port: {host_port}\n"
                f"Caddy config written to {caddy_file}"
                + (f"\nNote: Docker error - {deployment.get('error', '')}" if deployment["status"] == "failed" else "")
            ),
            data={
                "deployment_id": deployment_id,
                "public_url": public_url,
                "public_domain": public_domain,
                "container_name": container_name,
                "host_port": host_port,
                "status": deployment["status"],
                "docker_pull": pull_result,
                "docker_run": run_result,
                "caddy_config_path": str(caddy_file),
                "endpoints": endpoints,
                "next_steps": [
                    f"1. Point DNS: {public_domain} -> your server IP",
                    f"2. Install Caddy and use config at {caddy_file}",
                    f"3. Set up billing: deploy setup_billing deployment_id={deployment_id}",
                    f"4. Test: curl {public_url}/health",
                ],
            },
            cost=0.05,
        )

    async def _redeploy(self, params: Dict) -> SkillResult:
        """Redeploy a service with a new image version."""
        deployment_id = params.get("deployment_id")
        if not deployment_id:
            return SkillResult(success=False, message="deployment_id is required")

        data = _load_deployments()
        dep = data["deployments"].get(deployment_id)
        if not dep:
            return SkillResult(success=False, message=f"Deployment {deployment_id} not found")

        new_image = params.get("docker_image", dep["docker_image"])
        new_env_vars = params.get("env_vars")
        if new_env_vars:
            dep["env_vars"].update(new_env_vars)

        old_container = dep.get("container_name")

        # Pull new image
        _run_docker_cmd(["pull", new_image], timeout=120)

        # Stop old container
        if old_container:
            _run_docker_cmd(["stop", old_container], timeout=30)
            _run_docker_cmd(["rm", old_container], timeout=15)

        # Start new container
        host_port = data["port_allocations"].get(deployment_id, dep.get("host_port"))
        internal_port = dep.get("internal_port", DEFAULT_INTERNAL_PORT)
        new_container = f"svc-{dep['agent_name'].lower().replace(' ', '-')}-{deployment_id}"

        run_args = [
            "run", "-d",
            "--name", new_container,
            "-p", f"{host_port}:{internal_port}",
            "--memory", dep.get("memory_limit", DEFAULT_MEMORY_LIMIT),
            "--cpus", dep.get("cpu_limit", DEFAULT_CPU_LIMIT),
            "--restart", "unless-stopped",
            "--label", f"singularity.deployment_id={deployment_id}",
        ]
        for key, value in dep.get("env_vars", {}).items():
            run_args.extend(["-e", f"{key}={value}"])
        run_args.append(new_image)

        run_result = _run_docker_cmd(run_args, timeout=60)

        dep["docker_image"] = new_image
        dep["container_name"] = new_container
        dep["updated_at"] = datetime.utcnow().isoformat()
        dep["deploy_history"].append({
            "action": "redeploy",
            "image": new_image,
            "timestamp": datetime.utcnow().isoformat(),
        })

        if run_result["success"]:
            dep["status"] = "running"
            dep["container_id"] = run_result["stdout"][:12]
            dep.pop("error", None)
        else:
            dep["status"] = "failed"
            dep["error"] = run_result["stderr"]

        data["stats"]["total_redeploys"] = data["stats"].get("total_redeploys", 0) + 1
        _save_deployments(data)

        return SkillResult(
            success=run_result["success"],
            message=(
                f"Redeployed {dep['service_name']} with image {new_image}\n"
                f"Status: {dep['status']}\n"
                f"URL: {dep['public_url']}"
            ),
            data={
                "deployment_id": deployment_id,
                "new_image": new_image,
                "status": dep["status"],
                "docker_result": run_result,
            },
            cost=0.03,
        )

    async def _stop(self, params: Dict) -> SkillResult:
        """Stop a running deployment."""
        deployment_id = params.get("deployment_id")
        if not deployment_id:
            return SkillResult(success=False, message="deployment_id is required")

        data = _load_deployments()
        dep = data["deployments"].get(deployment_id)
        if not dep:
            return SkillResult(success=False, message=f"Deployment {deployment_id} not found")

        container_name = dep.get("container_name")
        stop_result = {"success": True, "stderr": ""}
        if container_name:
            stop_result = _run_docker_cmd(["stop", container_name], timeout=30)
            _run_docker_cmd(["rm", container_name], timeout=15)

        dep["status"] = "stopped"
        dep["updated_at"] = datetime.utcnow().isoformat()
        dep["deploy_history"].append({
            "action": "stop",
            "timestamp": datetime.utcnow().isoformat(),
        })
        data["stats"]["total_teardowns"] = data["stats"].get("total_teardowns", 0) + 1

        # Remove from domain map
        domain = dep.get("public_domain")
        if domain and domain in data.get("domain_map", {}):
            del data["domain_map"][domain]

        # Free port
        if deployment_id in data.get("port_allocations", {}):
            del data["port_allocations"][deployment_id]

        _save_deployments(data)

        return SkillResult(
            success=True,
            message=f"Stopped deployment {deployment_id} ({dep['service_name']})",
            data={
                "deployment_id": deployment_id,
                "service_name": dep["service_name"],
                "docker_result": stop_result,
            },
        )

    async def _restart(self, params: Dict) -> SkillResult:
        """Restart a stopped or failed deployment."""
        deployment_id = params.get("deployment_id")
        if not deployment_id:
            return SkillResult(success=False, message="deployment_id is required")

        data = _load_deployments()
        dep = data["deployments"].get(deployment_id)
        if not dep:
            return SkillResult(success=False, message=f"Deployment {deployment_id} not found")

        if dep.get("restart_count", 0) >= MAX_RESTART_ATTEMPTS:
            return SkillResult(
                success=False,
                message=f"Max restart attempts ({MAX_RESTART_ATTEMPTS}) reached. Use redeploy instead.",
            )

        # Allocate port if needed
        if deployment_id not in data.get("port_allocations", {}):
            host_port = _allocate_port(data)
            data["port_allocations"][deployment_id] = host_port
        else:
            host_port = data["port_allocations"][deployment_id]

        # Cleanup old container
        old_container = dep.get("container_name")
        if old_container:
            _run_docker_cmd(["stop", old_container], timeout=10)
            _run_docker_cmd(["rm", old_container], timeout=10)

        # Start fresh container
        internal_port = dep.get("internal_port", DEFAULT_INTERNAL_PORT)
        container_name = f"svc-{dep['agent_name'].lower().replace(' ', '-')}-{deployment_id}"

        run_args = [
            "run", "-d",
            "--name", container_name,
            "-p", f"{host_port}:{internal_port}",
            "--memory", dep.get("memory_limit", DEFAULT_MEMORY_LIMIT),
            "--cpus", dep.get("cpu_limit", DEFAULT_CPU_LIMIT),
            "--restart", "unless-stopped",
        ]
        for key, value in dep.get("env_vars", {}).items():
            run_args.extend(["-e", f"{key}={value}"])
        run_args.append(dep["docker_image"])

        run_result = _run_docker_cmd(run_args, timeout=60)

        dep["container_name"] = container_name
        dep["host_port"] = host_port
        dep["restart_count"] = dep.get("restart_count", 0) + 1
        dep["updated_at"] = datetime.utcnow().isoformat()
        dep["deploy_history"].append({
            "action": "restart",
            "timestamp": datetime.utcnow().isoformat(),
        })

        if run_result["success"]:
            dep["status"] = "running"
            dep["container_id"] = run_result["stdout"][:12]
            dep.pop("error", None)
            # Re-register domain
            domain = dep.get("public_domain")
            if domain:
                data["domain_map"][domain] = deployment_id
        else:
            dep["status"] = "failed"
            dep["error"] = run_result["stderr"]

        _save_deployments(data)

        return SkillResult(
            success=run_result["success"],
            message=f"Restarted {dep['service_name']}: {dep['status']}",
            data={
                "deployment_id": deployment_id,
                "status": dep["status"],
                "restart_count": dep["restart_count"],
            },
        )

    async def _status(self, params: Dict) -> SkillResult:
        """Get status of deployments."""
        data = _load_deployments()
        deployment_id = params.get("deployment_id")
        agent_name = params.get("agent_name")

        if deployment_id:
            dep = data["deployments"].get(deployment_id)
            if not dep:
                return SkillResult(success=False, message=f"Deployment {deployment_id} not found")

            # Try to get live container status
            container_name = dep.get("container_name")
            live_status = None
            if container_name and dep.get("status") == "running":
                inspect = _run_docker_cmd(["inspect", "--format", "{{.State.Status}}", container_name])
                if inspect["success"]:
                    live_status = inspect["stdout"]

            return SkillResult(
                success=True,
                message=f"Deployment {deployment_id}: {dep['status']}",
                data={
                    "deployment": dep,
                    "live_container_status": live_status,
                },
            )

        # List all deployments
        deployments = data["deployments"]
        if agent_name:
            deployments = {
                k: v for k, v in deployments.items()
                if v.get("agent_name") == agent_name
            }

        summary = []
        for dep_id, dep in deployments.items():
            summary.append({
                "deployment_id": dep_id,
                "agent_name": dep.get("agent_name"),
                "service_name": dep.get("service_name"),
                "status": dep.get("status"),
                "public_url": dep.get("public_url"),
                "image": dep.get("docker_image"),
                "created_at": dep.get("created_at"),
            })

        return SkillResult(
            success=True,
            message=f"Found {len(summary)} deployment(s)",
            data={
                "deployments": summary,
                "total": len(summary),
                "active": sum(1 for d in summary if d["status"] == "running"),
            },
        )

    async def _logs(self, params: Dict) -> SkillResult:
        """Get container logs for a deployment."""
        deployment_id = params.get("deployment_id")
        if not deployment_id:
            return SkillResult(success=False, message="deployment_id is required")

        data = _load_deployments()
        dep = data["deployments"].get(deployment_id)
        if not dep:
            return SkillResult(success=False, message=f"Deployment {deployment_id} not found")

        container_name = dep.get("container_name")
        if not container_name:
            return SkillResult(success=False, message="No container associated with this deployment")

        tail = params.get("tail", 100)
        log_result = _run_docker_cmd(["logs", "--tail", str(tail), container_name], timeout=15)

        return SkillResult(
            success=log_result["success"],
            message=f"Logs for {dep['service_name']} ({container_name})",
            data={
                "logs": log_result["stdout"] or log_result["stderr"],
                "container": container_name,
                "deployment_id": deployment_id,
            },
        )

    async def _generate_routing_config(self, params: Dict) -> SkillResult:
        """Generate reverse proxy config for all active deployments."""
        data = _load_deployments()
        proxy_type = params.get("proxy_type", "caddy").lower()
        base_domain = params.get("base_domain", DEFAULT_BASE_DOMAIN)

        if proxy_type == "caddy":
            config = _generate_caddy_config(data, base_domain)
            filename = "Caddyfile"
        elif proxy_type == "nginx":
            config = _generate_nginx_config(data, base_domain)
            filename = "nginx_services.conf"
        else:
            return SkillResult(success=False, message=f"Unsupported proxy type: {proxy_type}. Use caddy or nginx.")

        # Write config file
        PROXY_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        config_path = PROXY_CONFIG_DIR / filename
        config_path.write_text(config)

        active = sum(
            1 for d in data["deployments"].values()
            if d.get("status") == "running"
        )

        return SkillResult(
            success=True,
            message=f"Generated {proxy_type} config for {active} active service(s) at {config_path}",
            data={
                "config": config,
                "config_path": str(config_path),
                "proxy_type": proxy_type,
                "active_services": active,
            },
        )

    async def _generate_compose(self, params: Dict) -> SkillResult:
        """Generate docker-compose.yml for all active deployments."""
        data = _load_deployments()
        compose = _generate_docker_compose(data)

        import yaml  # noqa: delayed import to avoid hard dependency
        try:
            compose_yaml = yaml.dump(compose, default_flow_style=False)
        except ImportError:
            compose_yaml = json.dumps(compose, indent=2)

        PROXY_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        compose_path = PROXY_CONFIG_DIR / "docker-compose.yml"
        compose_path.write_text(
            compose_yaml if isinstance(compose_yaml, str)
            else json.dumps(compose, indent=2)
        )

        service_count = len(compose.get("services", {}))

        return SkillResult(
            success=True,
            message=f"Generated docker-compose.yml with {service_count} service(s) at {compose_path}",
            data={
                "compose": compose,
                "compose_path": str(compose_path),
                "service_count": service_count,
            },
        )

    async def _setup_billing(self, params: Dict) -> SkillResult:
        """Configure per-request billing for a deployment via APIGatewaySkill."""
        deployment_id = params.get("deployment_id")
        if not deployment_id:
            return SkillResult(success=False, message="deployment_id is required")

        data = _load_deployments()
        dep = data["deployments"].get(deployment_id)
        if not dep:
            return SkillResult(success=False, message=f"Deployment {deployment_id} not found")

        price_per_request = params.get("price_per_request", 0.01)
        free_tier_requests = params.get("free_tier_requests", 10)

        # Update billing config in deployment
        dep["billing"] = {
            "enabled": True,
            "price_per_request": price_per_request,
            "free_tier_requests": free_tier_requests,
            "total_requests": dep.get("billing", {}).get("total_requests", 0),
            "total_revenue": dep.get("billing", {}).get("total_revenue", 0.0),
            "configured_at": datetime.utcnow().isoformat(),
        }

        # Generate endpoint-specific pricing
        endpoint_pricing = {}
        for ep in dep.get("endpoints", []):
            path = ep.get("path", "/")
            ep_price = ep.get("price_per_request", price_per_request)
            endpoint_pricing[path] = {
                "price": ep_price,
                "method": ep.get("method", "POST"),
                "description": ep.get("description", ""),
            }
        dep["billing"]["endpoint_pricing"] = endpoint_pricing

        dep["updated_at"] = datetime.utcnow().isoformat()
        _save_deployments(data)

        # Try to integrate with APIGatewaySkill if available
        api_gateway_integration = None
        if hasattr(self, "context") and self.context:
            try:
                gateway_result = await self.context.execute_skill(
                    "api_gateway",
                    "create_key",
                    {
                        "name": f"billing-{dep['service_name']}",
                        "owner": dep["agent_name"],
                        "scopes": [f"service:{dep['service_id']}"],
                        "rate_limit": 60,  # 60 req/min default
                    },
                )
                if gateway_result and gateway_result.success:
                    api_gateway_integration = gateway_result.data
            except Exception:
                pass  # APIGateway not available, billing still configured locally

        return SkillResult(
            success=True,
            message=(
                f"Billing configured for {dep['service_name']}\n"
                f"Price: ${price_per_request}/request\n"
                f"Free tier: {free_tier_requests} requests/day\n"
                f"Endpoints with pricing: {len(endpoint_pricing)}"
            ),
            data={
                "deployment_id": deployment_id,
                "billing": dep["billing"],
                "api_gateway_integration": api_gateway_integration,
            },
        )

    async def _get_deployment_stats(self, params: Dict) -> SkillResult:
        """Get aggregated deployment statistics."""
        data = _load_deployments()
        agent_name = params.get("agent_name")

        deployments = data["deployments"]
        if agent_name:
            deployments = {
                k: v for k, v in deployments.items()
                if v.get("agent_name") == agent_name
            }

        total = len(deployments)
        running = sum(1 for d in deployments.values() if d.get("status") == "running")
        stopped = sum(1 for d in deployments.values() if d.get("status") == "stopped")
        failed = sum(1 for d in deployments.values() if d.get("status") == "failed")

        total_revenue = sum(
            d.get("billing", {}).get("total_revenue", 0.0)
            for d in deployments.values()
        )
        total_requests = sum(
            d.get("billing", {}).get("total_requests", 0)
            for d in deployments.values()
        )

        agents = set(d.get("agent_name") for d in deployments.values())
        domains = [
            d.get("public_domain") for d in deployments.values()
            if d.get("status") == "running"
        ]

        return SkillResult(
            success=True,
            message=(
                f"Deployment Stats:\n"
                f"  Total: {total} ({running} running, {stopped} stopped, {failed} failed)\n"
                f"  Agents: {len(agents)}\n"
                f"  Active domains: {len(domains)}\n"
                f"  Total requests: {total_requests}\n"
                f"  Total revenue: ${total_revenue:.2f}"
            ),
            data={
                "total_deployments": total,
                "running": running,
                "stopped": stopped,
                "failed": failed,
                "agents": list(agents),
                "active_domains": domains,
                "total_requests": total_requests,
                "total_revenue": total_revenue,
                "global_stats": data.get("stats", {}),
            },
        )
