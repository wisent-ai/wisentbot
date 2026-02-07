"""Tests for singularity CLI module."""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import argparse

import pytest

from singularity.cli import (
    load_config,
    merge_config,
    build_parser,
    cmd_status,
    cmd_config_template,
    cmd_spawn,
    _expand_env_vars,
    DEFAULT_CONFIG,
    main,
)


class TestLoadConfig:
    def test_load_json_config(self, tmp_path):
        config_file = tmp_path / "test.json"
        config_file.write_text(json.dumps({"name": "TestBot", "balance": 50.0}))
        config = load_config(str(config_file))
        assert config["name"] == "TestBot"
        assert config["balance"] == 50.0

    def test_load_missing_file_exits(self):
        with pytest.raises(SystemExit):
            load_config("/nonexistent/config.json")

    def test_load_invalid_json_exits(self, tmp_path):
        config_file = tmp_path / "bad.json"
        config_file.write_text("{not valid json")
        with pytest.raises(SystemExit):
            load_config(str(config_file))

    def test_env_var_expansion(self, tmp_path):
        os.environ["TEST_CLI_VAR"] = "expanded_value"
        config_file = tmp_path / "env.json"
        config_file.write_text(json.dumps({"api_key": "$TEST_CLI_VAR"}))
        config = load_config(str(config_file))
        assert config["api_key"] == "expanded_value"
        del os.environ["TEST_CLI_VAR"]


class TestExpandEnvVars:
    def test_string(self):
        os.environ["TEST_EXP"] = "hello"
        assert _expand_env_vars("$TEST_EXP") == "hello"
        del os.environ["TEST_EXP"]

    def test_dict(self):
        os.environ["TEST_EXP2"] = "world"
        result = _expand_env_vars({"key": "$TEST_EXP2", "num": 42})
        assert result["key"] == "world"
        assert result["num"] == 42
        del os.environ["TEST_EXP2"]

    def test_list(self):
        result = _expand_env_vars([1, "plain", None])
        assert result == [1, "plain", None]

    def test_none(self):
        assert _expand_env_vars(None) is None


class TestMergeConfig:
    def test_basic_merge(self):
        base = {"a": 1, "b": 2}
        overrides = {"b": 3, "c": 4}
        result = merge_config(base, overrides)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_none_values_not_applied(self):
        base = {"a": 1, "b": 2}
        overrides = {"b": None}
        result = merge_config(base, overrides)
        assert result["b"] == 2

    def test_empty_override(self):
        base = {"x": 10}
        result = merge_config(base, {})
        assert result == {"x": 10}


class TestBuildParser:
    def test_parser_created(self):
        parser = build_parser()
        assert parser is not None

    def test_run_command(self):
        parser = build_parser()
        args = parser.parse_args(["run", "--name", "TestBot", "--balance", "50"])
        assert args.command == "run"
        assert args.name == "TestBot"
        assert args.balance == 50.0

    def test_status_command(self):
        parser = build_parser()
        args = parser.parse_args(["status", "--quiet"])
        assert args.command == "status"
        assert args.quiet is True

    def test_spawn_command(self):
        parser = build_parser()
        args = parser.parse_args(["spawn", "--config", "test.json", "--name", "Child"])
        assert args.command == "spawn"
        assert args.config == "test.json"
        assert args.name == "Child"

    def test_config_template_command(self):
        parser = build_parser()
        args = parser.parse_args(["config-template", "--output", "out.json"])
        assert args.command == "config-template"
        assert args.output == "out.json"


class TestCmdStatus:
    def test_status_with_activity(self, capsys):
        activity = {
            "status": "running",
            "state": {
                "name": "TestBot",
                "ticker": "TEST",
                "balance_usd": 95.5,
                "total_api_cost": 0.05,
                "total_tokens_used": 1000,
                "cycle": 5,
                "runway_cycles": 100,
                "updated_at": "2024-01-01T12:00:00",
            },
            "logs": [
                {"timestamp": "2024-01-01T12:00:00", "tag": "CYCLE", "message": "Test log"}
            ],
        }
        import singularity.cli as cli_mod
        real_path = Path(cli_mod.__file__).parent / "data" / "activity.json"
        real_path.parent.mkdir(parents=True, exist_ok=True)
        existed_before = real_path.exists()
        old_content = real_path.read_text() if existed_before else None
        try:
            real_path.write_text(json.dumps(activity))
            args = argparse.Namespace(quiet=False)
            cmd_status(args)
            output = capsys.readouterr().out
            assert "TestBot" in output
            assert "RUNNING" in output
        finally:
            if existed_before and old_content:
                real_path.write_text(old_content)
            elif real_path.exists():
                real_path.unlink()


class TestCmdConfigTemplate:
    def test_generates_template(self, tmp_path):
        output_file = tmp_path / "template.json"
        args = argparse.Namespace(output=str(output_file))
        cmd_config_template(args)
        assert output_file.exists()
        config = json.loads(output_file.read_text())
        assert "name" in config
        assert "llm_provider" in config
        assert config["name"] == "MyAgent"


class TestCmdSpawn:
    def test_spawn_creates_config(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        base_config = tmp_path / "base.json"
        base_config.write_text(json.dumps({"name": "Parent", "starting_balance": 100}))

        args = argparse.Namespace(
            config=str(base_config),
            name="Child1",
            ticker="CH1",
            balance=25.0,
            start=False,
        )
        cmd_spawn(args)

        spawned = tmp_path / "spawned_agents" / "child1.json"
        assert spawned.exists()
        config = json.loads(spawned.read_text())
        assert config["name"] == "Child1"
        assert config["starting_balance"] == 25.0


class TestMain:
    def test_no_command_shows_help(self, capsys):
        with patch("sys.argv", ["singularity"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0

    def test_default_config_has_required_keys(self):
        assert "name" in DEFAULT_CONFIG
        assert "llm_provider" in DEFAULT_CONFIG
        assert "starting_balance" in DEFAULT_CONFIG
