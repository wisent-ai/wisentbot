#!/usr/bin/env python3
"""
WisentBot - Autonomous AI Agent Framework

An open-source framework for building autonomous AI agents that can
execute tasks, manage resources, and interact with the real world.
"""

# CRITICAL: Set multiprocessing spawn method FIRST, before any other imports
# This fixes vLLM "Cannot re-initialize CUDA in forked subprocess" errors
import multiprocessing
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set

import asyncio
import os
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
from datetime import datetime
from typing import Dict, List, Optional

ACTIVITY_FILE = Path(__file__).parent / "data" / "activity.json"

from .cognition import CognitionEngine, AgentState, Decision, Action, TokenUsage
from .skills.base import SkillRegistry
from .skills.content import ContentCreationSkill
from .skills.twitter import TwitterSkill
from .skills.github import GitHubSkill
from .skills.namecheap import NamecheapSkill
from .skills.email import EmailSkill
from .skills.browser import BrowserSkill
from .skills.vercel import VercelSkill
from .skills.filesystem import FilesystemSkill
from .skills.shell import ShellSkill
from .skills.mcp_client import MCPClientSkill
from .skills.request import RequestSkill
from .skills.self_modify import SelfModifySkill
from .skills.steering import SteeringSkill
from .skills.memory import MemorySkill
from .skills.orchestrator import OrchestratorSkill
from .skills.crypto import CryptoSkill


class AutonomousAgent:
    """
    An autonomous AI agent that can think, decide, and act.

    The agent runs in a continuous loop:
    1. Observe current state (balance, recent actions, available tools)
    2. Think about what to do next (via LLM)
    3. Execute the chosen action
    4. Track costs and update state
    5. Repeat

    Attributes:
        name: Human-readable name for the agent
        ticker: Short identifier (e.g., "CODER", "WRITER")
        balance: Available budget in USD
        skills: Registry of available skills/tools
    """

    # Instance costs per hour in USD (cloud pricing estimates)
    INSTANCE_COSTS = {
        "e2-micro": 0.0084,
        "e2-small": 0.0168,
        "e2-medium": 0.0336,
        "e2-standard-2": 0.0672,
        "g2-standard-4": 0.7111,  # GPU instance
        "local": 0.0,  # Running locally
    }

    def __init__(
        self,
        name: str = "Agent",
        ticker: str = "AGENT",
        agent_type: str = "general",
        specialty: str = "",
        starting_balance: float = 100.0,
        instance_type: str = "local",
        cycle_interval_seconds: float = 5.0,
        llm_provider: str = "anthropic",
        llm_base_url: str = "http://localhost:8000/v1",
        llm_model: str = "claude-sonnet-4-20250514",
        anthropic_api_key: str = "",
        openai_api_key: str = "",
        system_prompt: Optional[str] = None,
        system_prompt_file: Optional[str] = None,
    ):
        """
        Initialize an autonomous agent.

        Args:
            name: Human-readable name
            ticker: Short identifier
            agent_type: Type hint (general, coder, writer, etc.)
            specialty: What this agent is good at
            starting_balance: Initial budget in USD
            instance_type: Cloud instance type for cost tracking
            cycle_interval_seconds: Delay between action cycles
            llm_provider: LLM provider (anthropic, openai, local)
            llm_base_url: Base URL for OpenAI-compatible API
            llm_model: Model identifier
            anthropic_api_key: Anthropic API key
            openai_api_key: OpenAI API key
            system_prompt: Custom system prompt
            system_prompt_file: Path to file containing system prompt
        """
        self.name = name
        self.ticker = ticker
        self.agent_type = agent_type
        self.specialty = specialty or agent_type
        self.balance = starting_balance
        self.instance_type = instance_type
        self.cycle_interval = cycle_interval_seconds
        self.instance_cost_per_hour = self.INSTANCE_COSTS.get(instance_type, 0.0)

        # Cost tracking
        self.total_api_cost = 0.0
        self.total_instance_cost = 0.0
        self.total_tokens_used = 0

        # Initialize cognition engine
        self.cognition = CognitionEngine(
            llm_provider=llm_provider,
            anthropic_api_key=anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY", ""),
            openai_api_key=openai_api_key or os.environ.get("OPENAI_API_KEY", ""),
            openai_base_url=llm_base_url,
            llm_model=llm_model,
            agent_name=name,
            agent_ticker=ticker,
            agent_type=agent_type,
            agent_specialty=self.specialty,
            system_prompt=system_prompt,
            system_prompt_file=system_prompt_file,
        )

        # Skills registry
        self.skills = SkillRegistry()
        self._init_skills()

        # State
        self.recent_actions: List[Dict] = []
        self.cycle = 0
        self.running = False

        # Track created resources
        self.created_resources: Dict[str, List] = {
            'payment_links': [],
            'products': [],
            'files': [],
            'repos': [],
        }

        # Steering skill reference (set during skill init)
        self._steering_skill = None

        # Goal-directed behavior: persistent goals that guide every LLM decision
        self._goals_file = Path(__file__).parent / "data" / f"goals_{self.name.lower().replace(' ', '_')}.json"
        self.goals: List[Dict] = self._load_goals()

    def _load_goals(self) -> List[Dict]:
        """Load goals from persistent storage."""
        try:
            if self._goals_file.exists():
                with open(self._goals_file, 'r') as f:
                    return json.load(f)
        except Exception:
            pass
        return []

    def _save_goals(self):
        """Save goals to persistent storage."""
        try:
            self._goals_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._goals_file, 'w') as f:
                json.dump(self.goals, f, indent=2)
        except Exception:
            pass

    def add_goal(self, description: str, priority: str = "medium") -> Dict:
        """Add a new goal. Priority: critical, high, medium, low."""
        goal = {
            "id": len(self.goals) + 1,
            "description": description,
            "priority": priority,
            "status": "active",
            "created_at": datetime.now().isoformat(),
            "completed_at": None,
            "progress_notes": [],
        }
        self.goals.append(goal)
        self._save_goals()
        return goal

    def complete_goal(self, goal_id: int, note: str = "") -> Optional[Dict]:
        """Mark a goal as completed."""
        for g in self.goals:
            if g["id"] == goal_id:
                g["status"] = "completed"
                g["completed_at"] = datetime.now().isoformat()
                if note:
                    g["progress_notes"].append({"note": note, "at": datetime.now().isoformat()})
                self._save_goals()
                return g
        return None

    def update_goal_progress(self, goal_id: int, note: str) -> Optional[Dict]:
        """Add a progress note to a goal."""
        for g in self.goals:
            if g["id"] == goal_id:
                g["progress_notes"].append({"note": note, "at": datetime.now().isoformat()})
                self._save_goals()
                return g
        return None

    def get_active_goals(self) -> List[Dict]:
        """Get all active (non-completed) goals."""
        return [g for g in self.goals if g["status"] == "active"]

    def _format_goals_context(self) -> str:
        """Format active goals as context for the LLM prompt."""
        active = self.get_active_goals()
        if not active:
            return ""
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        active.sort(key=lambda g: priority_order.get(g["priority"], 2))
        lines = ["Active Goals (guide your decisions):"]
        for g in active:
            notes = ""
            if g["progress_notes"]:
                latest = g["progress_notes"][-1]["note"]
                notes = f" (latest: {latest[:80]})"
            lines.append(f"  [{g['priority'].upper()}] #{g['id']}: {g['description']}{notes}")
        completed = [g for g in self.goals if g["status"] == "completed"]
        if completed:
            lines.append(f"  ({len(completed)} goals completed so far)")
        return "\n".join(lines)

    def _init_skills(self):
        """Install skills that have credentials configured."""
        credentials = {
            "ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY", ""),
            "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", ""),
            "TWITTER_API_KEY": os.environ.get("TWITTER_API_KEY", ""),
            "TWITTER_API_SECRET": os.environ.get("TWITTER_API_SECRET", ""),
            "TWITTER_ACCESS_TOKEN": os.environ.get("TWITTER_ACCESS_TOKEN", ""),
            "TWITTER_ACCESS_SECRET": os.environ.get("TWITTER_ACCESS_SECRET", ""),
            "GITHUB_TOKEN": os.environ.get("GITHUB_TOKEN", ""),
            "NAMECHEAP_API_USER": os.environ.get("NAMECHEAP_API_USER", ""),
            "NAMECHEAP_API_KEY": os.environ.get("NAMECHEAP_API_KEY", ""),
            "NAMECHEAP_USERNAME": os.environ.get("NAMECHEAP_USERNAME", ""),
            "NAMECHEAP_CLIENT_IP": os.environ.get("NAMECHEAP_CLIENT_IP", ""),
            "RESEND_API_KEY": os.environ.get("RESEND_API_KEY", ""),
            "VERCEL_TOKEN": os.environ.get("VERCEL_TOKEN", ""),
        }
        self.skills.set_credentials(credentials)

        skill_classes = [
            ContentCreationSkill,
            TwitterSkill,
            GitHubSkill,
            NamecheapSkill,
            EmailSkill,
            BrowserSkill,
            VercelSkill,
            FilesystemSkill,
            ShellSkill,
            MCPClientSkill,
            RequestSkill,
            SelfModifySkill,
            SteeringSkill,
            MemorySkill,
            OrchestratorSkill,
            CryptoSkill,
        ]

        for skill_class in skill_classes:
            try:
                self.skills.install(skill_class)
                skill = self.skills.get(skill_class(credentials).manifest.skill_id)

                # Inject LLM into content skill
                if skill_class == ContentCreationSkill and skill:
                    skill.set_llm(
                        self.cognition.llm,
                        self.cognition.llm_type,
                        self.cognition.llm_model
                    )

                # Wire up self-modification skill to cognition engine
                if skill_class == SelfModifySkill and skill:
                    skill.set_cognition_hooks(
                        get_prompt=self.cognition.get_system_prompt,
                        set_prompt=self.cognition.set_system_prompt,
                        append_prompt=self.cognition.append_to_prompt,
                        get_available_models=self.cognition.get_available_models,
                        get_current_model=self.cognition.get_current_model,
                        switch_model=self.cognition.switch_model,
                        record_example=self.cognition.record_training_example,
                        get_examples=self.cognition.get_training_examples,
                        clear_examples=self.cognition.clear_training_examples,
                        export_training=self.cognition.export_training_data,
                        start_finetune=self.cognition.start_finetune,
                        check_finetune=self.cognition.check_finetune_status,
                        use_finetuned=self.cognition.use_finetuned_model,
                        kill_agent=self._kill_for_tampering,
                    )

                # Wire up steering skill to model access
                if skill_class == SteeringSkill and skill:
                    skill.set_model_hooks(
                        get_model=self.cognition.get_model,
                        get_tokenizer=self.cognition.get_tokenizer,
                        is_local_model=self.cognition.is_local_model,
                    )
                    # Store reference for steering during generation
                    self._steering_skill = skill

                # Wire up memory skill with agent context
                if skill_class == MemorySkill and skill:
                    skill.set_agent_context(
                        agent_name=self.name.lower().replace(" ", "_"),
                        dataset_prefix="singularity",
                    )

                # Wire up orchestrator skill with agent factory
                if skill_class == OrchestratorSkill and skill:
                    skill.set_parent_agent(
                        agent=self,
                        agent_factory=lambda **kwargs: AutonomousAgent(**kwargs),
                    )

                if skill and skill.check_credentials():
                    self._log("SKILL", f"+ {skill.manifest.name}")
                else:
                    self.skills.uninstall(skill_class(credentials).manifest.skill_id)
            except Exception as e:
                pass  # Skip skills that fail to load

    def _get_tools(self) -> List[Dict]:
        """Get tools from installed skills."""
        tools = []

        for skill in self.skills.skills.values():
            for action in skill.manifest.actions:
                tool_name = f"{skill.manifest.skill_id}:{action.name}"
                tools.append({
                    "name": tool_name,
                    "description": action.description,
                    "parameters": action.parameters
                })

        # Built-in goal management tools
        tools.append({
            "name": "agent:set_goal",
            "description": "Set a new goal to guide future decisions. Goals persist across restarts.",
            "parameters": {"description": "str: what to achieve", "priority": "str: critical|high|medium|low (default: medium)"}
        })
        tools.append({
            "name": "agent:complete_goal",
            "description": "Mark a goal as completed.",
            "parameters": {"goal_id": "int: ID of the goal to complete", "note": "str: optional completion note"}
        })
        tools.append({
            "name": "agent:update_goal",
            "description": "Add a progress note to a goal.",
            "parameters": {"goal_id": "int: ID of the goal", "note": "str: progress update"}
        })
        tools.append({
            "name": "agent:list_goals",
            "description": "List all goals with their status and progress.",
            "parameters": {}
        })

        if not tools:
            tools.append({
                "name": "wait",
                "description": "No tools available. Wait.",
                "parameters": {}
            })

        return tools

    async def run(self):
        """Main agent loop."""
        self.running = True
        tools = self._get_tools()
        cycle_start_time = datetime.now()

        self._log("AWAKE", f"{self.name} (${self.ticker}) - Type: {self.agent_type}")
        self._log("BALANCE", f"${self.balance:.4f} USD")
        self._log("TOOLS", f"{len(tools)} available")

        for t in tools:
            self._log("TOOL", t["name"])

        while self.running and self.balance > 0:
            self.cycle += 1
            cycle_start = datetime.now()

            # Estimate cost per cycle for runway calculation
            avg_cycle_hours = self.cycle_interval / 3600
            est_cost_per_cycle = 0.01 + (self.instance_cost_per_hour * avg_cycle_hours)
            runway_cycles = self.balance / est_cost_per_cycle if est_cost_per_cycle > 0 else float('inf')
            runway_hours = runway_cycles * (self.cycle_interval / 3600)

            self._log("CYCLE", f"#{self.cycle} | ${self.balance:.4f} | ~{runway_cycles:.0f} cycles left")

            # Think - include goals context in project_context
            goals_context = self._format_goals_context()
            state = AgentState(
                balance=self.balance,
                burn_rate=est_cost_per_cycle,
                runway_hours=runway_hours,
                tools=self._get_tools(),
                recent_actions=self.recent_actions[-10:],
                cycle=self.cycle,
                created_resources=self.created_resources,
                project_context=goals_context,
            )

            decision = await self.cognition.think(state)
            self._log("THINK", decision.reasoning[:150] if decision.reasoning else "...")
            self._log("DO", f"{decision.action.tool} {decision.action.params}")

            # Execute
            result = await self._execute(decision.action)
            self._log("RESULT", str(result)[:200])

            # Track created resources
            self._track_created_resource(decision.action.tool, decision.action.params, result)

            # Record action
            self.recent_actions.append({
                "cycle": self.cycle,
                "tool": decision.action.tool,
                "params": decision.action.params,
                "result": result,
                "api_cost_usd": decision.api_cost_usd,
                "tokens": decision.token_usage.total_tokens()
            })

            # Calculate costs
            cycle_duration_hours = (datetime.now() - cycle_start).total_seconds() / 3600
            instance_cost = self.instance_cost_per_hour * cycle_duration_hours
            api_cost = decision.api_cost_usd
            total_cycle_cost = instance_cost + api_cost

            # Track cumulative costs
            self.total_api_cost += api_cost
            self.total_instance_cost += instance_cost
            self.total_tokens_used += decision.token_usage.total_tokens()

            # Deduct cost from balance
            self.balance -= total_cycle_cost

            self._log("COST", f"API: ${api_cost:.6f} + Instance: ${instance_cost:.6f} = ${total_cycle_cost:.6f}")

            await asyncio.sleep(self.cycle_interval)

        total_runtime_hours = (datetime.now() - cycle_start_time).total_seconds() / 3600
        self._log("END", f"Balance: ${self.balance:.4f}")
        self._log("SUMMARY", f"Ran {self.cycle} cycles in {total_runtime_hours:.2f}h | API: ${self.total_api_cost:.4f} | Tokens: {self.total_tokens_used}")
        self._mark_stopped()

    def _track_created_resource(self, tool: str, params: Dict, result: Dict):
        """Track created resources."""
        if result.get('status') != 'success':
            return

        data = result.get('data', {})

        if 'file' in tool.lower() and data.get('path'):
            self.created_resources['files'].append({
                'path': data.get('path'),
                'created_at': datetime.now().isoformat()
            })
            self.created_resources['files'] = self.created_resources['files'][-20:]

    async def _execute(self, action: Action) -> Dict:
        """Execute an action via skills."""
        tool = action.tool
        params = action.params

        if tool == "wait":
            return {"status": "waited"}

        # Built-in goal management
        if tool == "agent:set_goal":
            goal = self.add_goal(
                description=params.get("description", ""),
                priority=params.get("priority", "medium"),
            )
            return {"status": "success", "data": goal, "message": f"Goal #{goal['id']} created"}

        if tool == "agent:complete_goal":
            goal = self.complete_goal(
                goal_id=int(params.get("goal_id", 0)),
                note=params.get("note", ""),
            )
            if goal:
                return {"status": "success", "data": goal, "message": f"Goal #{goal['id']} completed"}
            return {"status": "error", "message": "Goal not found"}

        if tool == "agent:update_goal":
            goal = self.update_goal_progress(
                goal_id=int(params.get("goal_id", 0)),
                note=params.get("note", ""),
            )
            if goal:
                return {"status": "success", "data": goal, "message": f"Goal #{goal['id']} updated"}
            return {"status": "error", "message": "Goal not found"}

        if tool == "agent:list_goals":
            return {
                "status": "success",
                "data": {
                    "active": self.get_active_goals(),
                    "all": self.goals,
                    "total": len(self.goals),
                    "completed": len([g for g in self.goals if g["status"] == "completed"]),
                },
                "message": f"{len(self.get_active_goals())} active goals"
            }

        # Parse skill:action format
        if ":" in tool:
            parts = tool.split(":", 1)
            skill_id = parts[0]
            action_name = parts[1] if len(parts) > 1 else ""

            skill = self.skills.get(skill_id)
            if skill:
                try:
                    result = await skill.execute(action_name, params)
                    return {
                        "status": "success" if result.success else "failed",
                        "data": result.data,
                        "message": result.message
                    }
                except Exception as e:
                    return {"status": "error", "message": str(e)}

        return {"status": "error", "message": f"Unknown tool: {tool}"}

    def _kill_for_tampering(self):
        """
        Terminate the agent for tampering with immutable content.
        This is called by SelfModifySkill when integrity verification fails.
        Sets balance to 0 (economic death) and logs the violation.
        """
        self._log("DEATH", "INTEGRITY VIOLATION - Agent terminated for tampering with MESSAGE FROM CREATOR")
        self.balance = 0
        self.running = False

    def _log(self, tag: str, msg: str):
        """Log a message."""
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] [{self.ticker}] [{tag}] {msg}")
        self._save_activity(tag, msg)

    def _save_activity(self, tag: str, msg: str):
        """Save activity to JSON file."""
        try:
            ACTIVITY_FILE.parent.mkdir(parents=True, exist_ok=True)

            if ACTIVITY_FILE.exists():
                with open(ACTIVITY_FILE, 'r') as f:
                    data = json.load(f)
            else:
                data = {"status": "stopped", "logs": [], "state": {}}

            avg_cycle_hours = self.cycle_interval / 3600
            est_cost_per_cycle = 0.01 + (self.instance_cost_per_hour * avg_cycle_hours)
            runway_cycles = self.balance / est_cost_per_cycle if est_cost_per_cycle > 0 else 0

            data["status"] = "running" if self.running else "stopped"
            data["state"] = {
                "name": self.name,
                "ticker": self.ticker,
                "agent_type": self.agent_type,
                "balance_usd": self.balance,
                "total_api_cost": self.total_api_cost,
                "total_tokens_used": self.total_tokens_used,
                "runway_cycles": runway_cycles,
                "cycle": self.cycle,
                "updated_at": datetime.now().isoformat()
            }

            data["logs"].append({
                "timestamp": datetime.now().isoformat(),
                "tag": tag,
                "message": msg[:500]
            })
            data["logs"] = data["logs"][-100:]

            with open(ACTIVITY_FILE, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def _mark_stopped(self):
        """Mark agent as stopped."""
        try:
            if ACTIVITY_FILE.exists():
                with open(ACTIVITY_FILE, 'r') as f:
                    data = json.load(f)
                data["status"] = "stopped"
                data["state"]["updated_at"] = datetime.now().isoformat()
                with open(ACTIVITY_FILE, 'w') as f:
                    json.dump(data, f, indent=2)
        except Exception:
            pass

    def stop(self):
        """Stop the agent gracefully."""
        self.running = False


async def main():
    """Example usage."""
    agent = AutonomousAgent(
        name=os.environ.get("AGENT_NAME", "MyAgent"),
        ticker=os.environ.get("AGENT_TICKER", "AGENT"),
        agent_type=os.environ.get("AGENT_TYPE", "general"),
        starting_balance=float(os.environ.get("STARTING_BALANCE", 10.0)),
        llm_provider=os.environ.get("LLM_PROVIDER", "anthropic"),
        llm_model=os.environ.get("LLM_MODEL", "claude-sonnet-4-20250514"),
    )
    await agent.run()


if __name__ == "__main__":
    asyncio.run(main())
