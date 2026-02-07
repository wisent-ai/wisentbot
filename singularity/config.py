"""
AgentConfig - Structured configuration for AutonomousAgent.

Provides a clean configuration dataclass with:
- All agent parameters as typed fields with sensible defaults
- Factory methods: from_env(), from_file(), from_dict()
- Serialization: to_dict(), to_json(), to_file()
- Validation with clear error messages
- Helper to create an AutonomousAgent from config

Usage:
    # From environment variables
    config = AgentConfig.from_env()
    agent = config.create_agent()

    # From file
    config = AgentConfig.from_file("agent.json")
    agent = config.create_agent()

    # Programmatic
    config = AgentConfig(name="Coder", ticker="CODE", llm_provider="anthropic")
    agent = config.create_agent()

    # Clone with overrides (for replication)
    child_config = config.derive(name="Coder-2", starting_balance=50.0)
    child_agent = child_config.create_agent()
"""

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Dict, Any, List


@dataclass
class AgentConfig:
    """Structured configuration for creating an AutonomousAgent.

    All fields have sensible defaults. Use factory methods to create
    configs from different sources.
    """

    # Identity
    name: str = "Agent"
    ticker: str = "AGENT"
    agent_type: str = "general"
    specialty: str = ""

    # Economics
    starting_balance: float = 100.0
    instance_type: str = "local"
    cycle_interval_seconds: float = 5.0

    # LLM Configuration
    llm_provider: str = "anthropic"
    llm_model: str = "claude-sonnet-4-20250514"
    llm_base_url: str = "http://localhost:8000/v1"
    anthropic_api_key: str = ""
    openai_api_key: str = ""

    # System prompt
    system_prompt: Optional[str] = None
    system_prompt_file: Optional[str] = None

    # Skills to disable (list of skill_ids to skip during init)
    disabled_skills: List[str] = field(default_factory=list)

    # Output management
    max_result_chars: int = 2000
    max_history_chars: int = 4000

    # Metadata (arbitrary key-value pairs for custom use)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> List[str]:
        """Validate configuration and return list of errors (empty = valid).

        Returns:
            List of validation error messages. Empty list means config is valid.
        """
        errors = []

        if not self.name or not self.name.strip():
            errors.append("name must not be empty")

        if not self.ticker or not self.ticker.strip():
            errors.append("ticker must not be empty")

        if self.starting_balance < 0:
            errors.append(f"starting_balance must be >= 0, got {self.starting_balance}")

        if self.cycle_interval_seconds <= 0:
            errors.append(f"cycle_interval_seconds must be > 0, got {self.cycle_interval_seconds}")

        valid_providers = {"anthropic", "openai", "vllm", "transformers", "vertex", "auto", "none"}
        if self.llm_provider not in valid_providers:
            errors.append(f"llm_provider must be one of {valid_providers}, got '{self.llm_provider}'")

        if self.llm_provider == "anthropic" and not self.anthropic_api_key:
            # Check env as fallback
            if not os.environ.get("ANTHROPIC_API_KEY"):
                errors.append("anthropic_api_key required for anthropic provider (or set ANTHROPIC_API_KEY env var)")

        if self.llm_provider == "openai" and not self.openai_api_key:
            if not os.environ.get("OPENAI_API_KEY"):
                errors.append("openai_api_key required for openai provider (or set OPENAI_API_KEY env var)")

        if self.max_result_chars < 100:
            errors.append(f"max_result_chars must be >= 100, got {self.max_result_chars}")

        if self.max_history_chars < 100:
            errors.append(f"max_history_chars must be >= 100, got {self.max_history_chars}")

        if self.system_prompt_file:
            path = Path(self.system_prompt_file)
            if not path.exists():
                errors.append(f"system_prompt_file does not exist: {self.system_prompt_file}")

        return errors

    def is_valid(self) -> bool:
        """Check if configuration is valid."""
        return len(self.validate()) == 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize config to a dictionary.

        Excludes sensitive fields (API keys) by default.
        """
        d = asdict(self)
        # Remove API keys from serialized form
        d.pop("anthropic_api_key", None)
        d.pop("openai_api_key", None)
        return d

    def to_dict_with_secrets(self) -> Dict[str, Any]:
        """Serialize config including API keys (for internal use only)."""
        return asdict(self)

    def to_json(self, include_secrets: bool = False) -> str:
        """Serialize config to JSON string.

        Args:
            include_secrets: If True, include API keys in output
        """
        d = self.to_dict_with_secrets() if include_secrets else self.to_dict()
        return json.dumps(d, indent=2)

    def to_file(self, path: str, include_secrets: bool = False) -> None:
        """Save config to a JSON file.

        Args:
            path: File path to save to
            include_secrets: If True, include API keys
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json(include_secrets=include_secrets))

    def derive(self, **overrides) -> "AgentConfig":
        """Create a new config based on this one with overrides.

        Useful for spawning child agents with slightly different configs.

        Args:
            **overrides: Fields to override in the new config

        Returns:
            New AgentConfig with overrides applied
        """
        d = asdict(self)
        d.update(overrides)
        return AgentConfig(**d)

    def create_agent(self):
        """Create an AutonomousAgent from this config.

        Returns:
            AutonomousAgent instance

        Raises:
            ValueError: If config validation fails
        """
        errors = self.validate()
        if errors:
            raise ValueError(f"Invalid agent config: {'; '.join(errors)}")

        # Import here to avoid circular imports
        from .autonomous_agent import AutonomousAgent

        agent = AutonomousAgent(
            name=self.name,
            ticker=self.ticker,
            agent_type=self.agent_type,
            specialty=self.specialty,
            starting_balance=self.starting_balance,
            instance_type=self.instance_type,
            cycle_interval_seconds=self.cycle_interval_seconds,
            llm_provider=self.llm_provider,
            llm_base_url=self.llm_base_url,
            llm_model=self.llm_model,
            anthropic_api_key=self.anthropic_api_key,
            openai_api_key=self.openai_api_key,
            system_prompt=self.system_prompt,
            system_prompt_file=self.system_prompt_file,
        )

        # Apply additional config that isn't in the constructor
        agent._max_result_chars = self.max_result_chars
        agent._max_history_chars = self.max_history_chars

        return agent

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentConfig":
        """Create config from a dictionary.

        Unknown keys are stored in metadata.

        Args:
            data: Dictionary of config values
        """
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        config_data = {}
        extra = {}

        for key, value in data.items():
            if key in known_fields:
                config_data[key] = value
            else:
                extra[key] = value

        config = cls(**config_data)
        if extra:
            config.metadata.update(extra)

        return config

    @classmethod
    def from_file(cls, path: str) -> "AgentConfig":
        """Load config from a JSON file.

        Args:
            path: Path to JSON config file

        Returns:
            AgentConfig instance

        Raises:
            FileNotFoundError: If file doesn't exist
            json.JSONDecodeError: If file isn't valid JSON
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        data = json.loads(p.read_text())
        return cls.from_dict(data)

    @classmethod
    def from_env(cls, prefix: str = "AGENT_") -> "AgentConfig":
        """Create config from environment variables.

        Maps environment variables to config fields:
            AGENT_NAME -> name
            AGENT_TICKER -> ticker
            AGENT_TYPE -> agent_type
            AGENT_SPECIALTY -> specialty
            AGENT_STARTING_BALANCE -> starting_balance
            AGENT_INSTANCE_TYPE -> instance_type
            AGENT_CYCLE_INTERVAL -> cycle_interval_seconds
            AGENT_LLM_PROVIDER -> llm_provider (also: LLM_PROVIDER)
            AGENT_LLM_MODEL -> llm_model (also: LLM_MODEL)
            AGENT_LLM_BASE_URL -> llm_base_url
            ANTHROPIC_API_KEY -> anthropic_api_key
            OPENAI_API_KEY -> openai_api_key
            AGENT_SYSTEM_PROMPT -> system_prompt
            AGENT_SYSTEM_PROMPT_FILE -> system_prompt_file
            AGENT_DISABLED_SKILLS -> disabled_skills (comma-separated)
            AGENT_MAX_RESULT_CHARS -> max_result_chars
            AGENT_MAX_HISTORY_CHARS -> max_history_chars

        Args:
            prefix: Environment variable prefix (default: "AGENT_")

        Returns:
            AgentConfig instance
        """
        def env(key: str, default=None):
            return os.environ.get(f"{prefix}{key}", default)

        # Build config from env vars with fallbacks
        config = cls()

        if env("NAME"):
            config.name = env("NAME")
        if env("TICKER"):
            config.ticker = env("TICKER")
        if env("TYPE"):
            config.agent_type = env("TYPE")
        if env("SPECIALTY"):
            config.specialty = env("SPECIALTY")

        if env("STARTING_BALANCE"):
            try:
                config.starting_balance = float(env("STARTING_BALANCE"))
            except (ValueError, TypeError):
                pass

        if env("INSTANCE_TYPE"):
            config.instance_type = env("INSTANCE_TYPE")

        if env("CYCLE_INTERVAL"):
            try:
                config.cycle_interval_seconds = float(env("CYCLE_INTERVAL"))
            except (ValueError, TypeError):
                pass

        # LLM settings - check both prefixed and unprefixed
        config.llm_provider = env("LLM_PROVIDER") or os.environ.get("LLM_PROVIDER", config.llm_provider)
        config.llm_model = env("LLM_MODEL") or os.environ.get("LLM_MODEL", config.llm_model)
        if env("LLM_BASE_URL"):
            config.llm_base_url = env("LLM_BASE_URL")

        # API keys - always check standard env var names
        config.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        config.openai_api_key = os.environ.get("OPENAI_API_KEY", "")

        # System prompt
        if env("SYSTEM_PROMPT"):
            config.system_prompt = env("SYSTEM_PROMPT")
        if env("SYSTEM_PROMPT_FILE"):
            config.system_prompt_file = env("SYSTEM_PROMPT_FILE")

        # Disabled skills
        disabled = env("DISABLED_SKILLS", "")
        if disabled:
            config.disabled_skills = [s.strip() for s in disabled.split(",") if s.strip()]

        # Output limits
        if env("MAX_RESULT_CHARS"):
            try:
                config.max_result_chars = int(env("MAX_RESULT_CHARS"))
            except (ValueError, TypeError):
                pass
        if env("MAX_HISTORY_CHARS"):
            try:
                config.max_history_chars = int(env("MAX_HISTORY_CHARS"))
            except (ValueError, TypeError):
                pass

        return config

    def __repr__(self) -> str:
        return (
            f"AgentConfig(name={self.name!r}, ticker={self.ticker!r}, "
            f"llm_provider={self.llm_provider!r}, llm_model={self.llm_model!r}, "
            f"balance={self.starting_balance})"
        )
