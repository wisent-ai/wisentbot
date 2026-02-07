"""Tests for AgentConfig."""

import json
import os
import pytest
from singularity.config import AgentConfig


class TestAgentConfig:
    def test_defaults(self):
        config = AgentConfig()
        assert config.name == "Agent"
        assert config.ticker == "AGENT"
        assert config.starting_balance == 100.0
        assert config.llm_provider == "anthropic"

    def test_custom_values(self):
        config = AgentConfig(name="Coder", ticker="CODE", starting_balance=50.0)
        assert config.name == "Coder"
        assert config.ticker == "CODE"
        assert config.starting_balance == 50.0

    def test_validate_valid(self):
        config = AgentConfig(llm_provider="none")
        assert config.is_valid()
        assert config.validate() == []

    def test_validate_invalid_balance(self):
        config = AgentConfig(starting_balance=-10, llm_provider="none")
        errors = config.validate()
        assert any("starting_balance" in e for e in errors)

    def test_validate_invalid_provider(self):
        config = AgentConfig(llm_provider="invalid")
        errors = config.validate()
        assert any("llm_provider" in e for e in errors)

    def test_validate_empty_name(self):
        config = AgentConfig(name="", llm_provider="none")
        errors = config.validate()
        assert any("name" in e for e in errors)

    def test_validate_cycle_interval(self):
        config = AgentConfig(cycle_interval_seconds=0, llm_provider="none")
        errors = config.validate()
        assert any("cycle_interval" in e for e in errors)

    def test_to_dict_excludes_secrets(self):
        config = AgentConfig(anthropic_api_key="sk-secret")
        d = config.to_dict()
        assert "anthropic_api_key" not in d
        assert "openai_api_key" not in d

    def test_to_dict_with_secrets(self):
        config = AgentConfig(anthropic_api_key="sk-secret")
        d = config.to_dict_with_secrets()
        assert d["anthropic_api_key"] == "sk-secret"

    def test_to_json(self):
        config = AgentConfig(name="Test")
        j = config.to_json()
        data = json.loads(j)
        assert data["name"] == "Test"
        assert "anthropic_api_key" not in data

    def test_from_dict(self):
        config = AgentConfig.from_dict({"name": "FromDict", "ticker": "FD"})
        assert config.name == "FromDict"
        assert config.ticker == "FD"

    def test_from_dict_unknown_keys(self):
        config = AgentConfig.from_dict({"name": "X", "custom_field": "value"})
        assert config.name == "X"
        assert config.metadata["custom_field"] == "value"

    def test_from_file(self, tmp_path):
        f = tmp_path / "config.json"
        f.write_text(json.dumps({"name": "FileAgent", "ticker": "FA", "llm_provider": "none"}))
        config = AgentConfig.from_file(str(f))
        assert config.name == "FileAgent"
        assert config.ticker == "FA"

    def test_from_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            AgentConfig.from_file("/nonexistent/config.json")

    def test_to_file(self, tmp_path):
        config = AgentConfig(name="SaveMe", llm_provider="none")
        path = str(tmp_path / "out.json")
        config.to_file(path)
        loaded = AgentConfig.from_file(path)
        assert loaded.name == "SaveMe"

    def test_derive(self):
        parent = AgentConfig(name="Parent", ticker="P", starting_balance=100.0)
        child = parent.derive(name="Child", ticker="C", starting_balance=50.0)
        assert child.name == "Child"
        assert child.ticker == "C"
        assert child.starting_balance == 50.0
        assert child.llm_provider == parent.llm_provider  # inherited

    def test_from_env(self, monkeypatch):
        monkeypatch.setenv("AGENT_NAME", "EnvAgent")
        monkeypatch.setenv("AGENT_TICKER", "ENV")
        monkeypatch.setenv("AGENT_STARTING_BALANCE", "42.5")
        monkeypatch.setenv("LLM_PROVIDER", "none")
        config = AgentConfig.from_env()
        assert config.name == "EnvAgent"
        assert config.ticker == "ENV"
        assert config.starting_balance == 42.5

    def test_from_env_disabled_skills(self, monkeypatch):
        monkeypatch.setenv("AGENT_DISABLED_SKILLS", "twitter,email,browser")
        monkeypatch.setenv("LLM_PROVIDER", "none")
        config = AgentConfig.from_env()
        assert config.disabled_skills == ["twitter", "email", "browser"]

    def test_from_env_llm_settings(self, monkeypatch):
        monkeypatch.setenv("AGENT_LLM_PROVIDER", "none")
        monkeypatch.setenv("AGENT_LLM_MODEL", "gpt-4o")
        config = AgentConfig.from_env()
        assert config.llm_provider == "none"
        assert config.llm_model == "gpt-4o"

    def test_repr(self):
        config = AgentConfig(name="Test", ticker="T")
        r = repr(config)
        assert "Test" in r
        assert "T" in r

    def test_roundtrip_json(self):
        original = AgentConfig(name="RT", ticker="RT", starting_balance=77.0, llm_provider="none")
        j = original.to_json()
        restored = AgentConfig.from_dict(json.loads(j))
        assert restored.name == original.name
        assert restored.ticker == original.ticker
        assert restored.starting_balance == original.starting_balance
