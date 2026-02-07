#!/usr/bin/env python3
"""
Command-line interface for Singularity autonomous agent.

Usage:
    singularity run [--config config.json] [--name NAME] [--balance 100]
    singularity status
    singularity spawn --config base.json --name Child1
    singularity config-template [--output agent.json]
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional


DEFAULT_CONFIG = {
    "name": "Agent",
    "ticker": "AGENT",
    "agent_type": "general",
    "specialty": "",
    "starting_balance": 100.0,
    "instance_type": "local",
    "cycle_interval_seconds": 5.0,
    "llm_provider": "anthropic",
    "llm_base_url": "http://localhost:8000/v1",
    "llm_model": "claude-sonnet-4-20250514",
    "system_prompt": None,
    "system_prompt_file": None,
}


def load_config(config_path: str) -> Dict[str, Any]:
    """Load agent configuration from a JSON file.

    Supports JSON natively. If PyYAML is installed, also supports YAML.
    Environment variables in values are expanded (e.g., "$ANTHROPIC_API_KEY").
    """
    path = Path(config_path)
    if not path.exists():
        print(f"Error: Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    text = path.read_text()

    if path.suffix in (".yaml", ".yml"):
        try:
            import yaml
            config = yaml.safe_load(text)
        except ImportError:
            print("Error: PyYAML is required for YAML config files. Install with: pip install pyyaml",
                  file=sys.stderr)
            sys.exit(1)
    else:
        try:
            config = json.loads(text)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in {config_path}: {e}", file=sys.stderr)
            sys.exit(1)

    # Expand environment variables in string values
    config = _expand_env_vars(config)

    return config


def _expand_env_vars(obj: Any) -> Any:
    """Recursively expand environment variables in string values."""
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    elif isinstance(obj, dict):
        return {k: _expand_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_expand_env_vars(item) for item in obj]
    return obj


def merge_config(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Merge override config into base config. Overrides win for non-None values."""
    result = dict(base)
    for key, value in overrides.items():
        if value is not None:
            result[key] = value
    return result


def cmd_run(args: argparse.Namespace) -> None:
    """Run the agent with the given configuration."""
    # Start with defaults
    config = dict(DEFAULT_CONFIG)

    # Layer config file if provided
    if args.config:
        file_config = load_config(args.config)
        config = merge_config(config, file_config)

    # Layer CLI arguments (highest priority)
    cli_overrides = {}
    if args.name is not None:
        cli_overrides["name"] = args.name
    if args.ticker is not None:
        cli_overrides["ticker"] = args.ticker
    if args.agent_type is not None:
        cli_overrides["agent_type"] = args.agent_type
    if args.balance is not None:
        cli_overrides["starting_balance"] = args.balance
    if args.llm_provider is not None:
        cli_overrides["llm_provider"] = args.llm_provider
    if args.llm_model is not None:
        cli_overrides["llm_model"] = args.llm_model
    if args.cycle_interval is not None:
        cli_overrides["cycle_interval_seconds"] = args.cycle_interval
    if args.system_prompt_file is not None:
        cli_overrides["system_prompt_file"] = args.system_prompt_file

    config = merge_config(config, cli_overrides)

    # Build agent kwargs (only pass recognized constructor params)
    agent_kwargs = {
        "name": config.get("name", "Agent"),
        "ticker": config.get("ticker", "AGENT"),
        "agent_type": config.get("agent_type", "general"),
        "specialty": config.get("specialty", ""),
        "starting_balance": float(config.get("starting_balance", 100.0)),
        "instance_type": config.get("instance_type", "local"),
        "cycle_interval_seconds": float(config.get("cycle_interval_seconds", 5.0)),
        "llm_provider": config.get("llm_provider", "anthropic"),
        "llm_base_url": config.get("llm_base_url", "http://localhost:8000/v1"),
        "llm_model": config.get("llm_model", "claude-sonnet-4-20250514"),
    }

    # Optional string params
    if config.get("system_prompt"):
        agent_kwargs["system_prompt"] = config["system_prompt"]
    if config.get("system_prompt_file"):
        agent_kwargs["system_prompt_file"] = config["system_prompt_file"]
    if config.get("anthropic_api_key"):
        agent_kwargs["anthropic_api_key"] = config["anthropic_api_key"]
    if config.get("openai_api_key"):
        agent_kwargs["openai_api_key"] = config["openai_api_key"]

    # Print config summary
    print(f"Starting agent: {agent_kwargs['name']} (${agent_kwargs.get('ticker', 'AGENT')})")
    print(f"  LLM: {agent_kwargs['llm_provider']} / {agent_kwargs['llm_model']}")
    print(f"  Balance: ${agent_kwargs['starting_balance']:.2f}")
    print(f"  Cycle interval: {agent_kwargs['cycle_interval_seconds']}s")
    if args.config:
        print(f"  Config: {args.config}")
    print()

    from .autonomous_agent import AutonomousAgent
    agent = AutonomousAgent(**agent_kwargs)
    asyncio.run(agent.run())


def cmd_status(args: argparse.Namespace) -> None:
    """Show agent status from activity file."""
    activity_file = Path(__file__).parent / "data" / "activity.json"

    if not activity_file.exists():
        print("No agent activity found. Run an agent first.")
        return

    with open(activity_file) as f:
        data = json.load(f)

    state = data.get("state", {})
    status = data.get("status", "unknown")

    print(f"Agent: {state.get('name', 'Unknown')} (${state.get('ticker', '?')})")
    print(f"Status: {status.upper()}")
    print(f"Balance: ${state.get('balance_usd', 0):.4f}")
    print(f"API Cost: ${state.get('total_api_cost', 0):.4f}")
    print(f"Tokens Used: {state.get('total_tokens_used', 0):,}")
    print(f"Cycles: {state.get('cycle', 0)}")
    print(f"Runway: ~{state.get('runway_cycles', 0):.0f} cycles")
    print(f"Updated: {state.get('updated_at', 'Never')}")

    # Show recent logs
    logs = data.get("logs", [])
    if logs and not args.quiet:
        print(f"\nRecent activity ({len(logs)} entries):")
        for log in logs[-10:]:
            ts = log.get("timestamp", "")
            if "T" in ts:
                ts = ts.split("T")[1][:8]
            print(f"  [{ts}] [{log.get('tag', '?')}] {log.get('message', '')[:120]}")


def cmd_spawn(args: argparse.Namespace) -> None:
    """Spawn a new agent from a config file with overrides."""
    if not args.config:
        print("Error: --config is required for spawn", file=sys.stderr)
        sys.exit(1)

    config = load_config(args.config)

    # Apply spawn-specific overrides
    if args.name:
        config["name"] = args.name
    if args.ticker:
        config["ticker"] = args.ticker
    if args.balance is not None:
        config["starting_balance"] = args.balance

    # Generate a ticker from name if not provided
    if args.name and not args.ticker:
        config["ticker"] = args.name[:6].upper().replace(" ", "")

    # Save spawned config
    spawn_dir = Path("spawned_agents")
    spawn_dir.mkdir(exist_ok=True)

    agent_name = config.get("name", "agent").lower().replace(" ", "_")
    spawn_config_path = spawn_dir / f"{agent_name}.json"

    with open(spawn_config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Spawned agent config: {spawn_config_path}")
    print(f"  Name: {config.get('name')}")
    print(f"  Ticker: {config.get('ticker')}")
    print(f"  Balance: ${config.get('starting_balance', 100):.2f}")
    print(f"\nRun with: singularity run --config {spawn_config_path}")

    if args.start:
        print("\nStarting spawned agent...")
        # Simulate setting args.config for cmd_run
        run_args = argparse.Namespace(
            config=str(spawn_config_path),
            name=None, ticker=None, agent_type=None,
            balance=None, llm_provider=None, llm_model=None,
            cycle_interval=None, system_prompt_file=None,
        )
        cmd_run(run_args)


def cmd_config_template(args: argparse.Namespace) -> None:
    """Generate a config template file."""
    template = {
        "name": "MyAgent",
        "ticker": "MYAGT",
        "agent_type": "general",
        "specialty": "general-purpose autonomous agent",
        "starting_balance": 100.0,
        "instance_type": "local",
        "cycle_interval_seconds": 5.0,
        "llm_provider": "anthropic",
        "llm_base_url": "http://localhost:8000/v1",
        "llm_model": "claude-sonnet-4-20250514",
        "system_prompt": None,
        "system_prompt_file": None,
        "anthropic_api_key": "$ANTHROPIC_API_KEY",
        "openai_api_key": "$OPENAI_API_KEY",
    }

    output = args.output or "agent_config.json"
    with open(output, "w") as f:
        json.dump(template, f, indent=2)

    print(f"Config template written to: {output}")
    print("Edit it, then run: singularity run --config", output)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="singularity",
        description="Singularity - Autonomous AI Agent Framework",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- run ---
    run_parser = subparsers.add_parser("run", help="Run an agent")
    run_parser.add_argument("--config", "-c", type=str, help="Path to config file (JSON/YAML)")
    run_parser.add_argument("--name", "-n", type=str, help="Agent name")
    run_parser.add_argument("--ticker", "-t", type=str, help="Agent ticker symbol")
    run_parser.add_argument("--agent-type", type=str, help="Agent type (general, coder, writer)")
    run_parser.add_argument("--balance", "-b", type=float, help="Starting balance in USD")
    run_parser.add_argument("--llm-provider", type=str, help="LLM provider (anthropic, openai, local)")
    run_parser.add_argument("--llm-model", type=str, help="LLM model name")
    run_parser.add_argument("--cycle-interval", type=float, help="Seconds between cycles")
    run_parser.add_argument("--system-prompt-file", type=str, help="Path to system prompt file")

    # --- status ---
    status_parser = subparsers.add_parser("status", help="Show agent status")
    status_parser.add_argument("--quiet", "-q", action="store_true", help="Hide recent logs")

    # --- spawn ---
    spawn_parser = subparsers.add_parser("spawn", help="Spawn a new agent from config")
    spawn_parser.add_argument("--config", "-c", type=str, required=True, help="Base config file")
    spawn_parser.add_argument("--name", "-n", type=str, help="Override agent name")
    spawn_parser.add_argument("--ticker", "-t", type=str, help="Override ticker")
    spawn_parser.add_argument("--balance", "-b", type=float, help="Override starting balance")
    spawn_parser.add_argument("--start", "-s", action="store_true", help="Start the agent immediately")

    # --- config-template ---
    template_parser = subparsers.add_parser("config-template", help="Generate a config template")
    template_parser.add_argument("--output", "-o", type=str, help="Output file path")

    return parser


def main() -> None:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    commands = {
        "run": cmd_run,
        "status": cmd_status,
        "spawn": cmd_spawn,
        "config-template": cmd_config_template,
    }

    handler = commands.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
