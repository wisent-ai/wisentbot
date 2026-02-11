"""Tests for singularity.autonomous_agent — the main agent class."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch, PropertyMock
from singularity.cognition.types import Action, TokenUsage, Decision, AgentState
from singularity.skills.base.types import SkillResult


# We need to import AutonomousAgent carefully since it calls load_dotenv at import time
# The conftest.py mocks handle this


class TestAutonomousAgentInit:
    """Test agent initialization without loading real skills."""

    @patch("singularity.autonomous_agent.PluginLoader")
    @patch("singularity.autonomous_agent.CognitionEngine")
    def test_basic_init(self, MockCognition, MockLoader):
        from singularity.autonomous_agent import AutonomousAgent
        agent = AutonomousAgent(
            name="TestAgent", ticker="TEST",
            starting_balance=50.0, llm_provider="none",
        )
        assert agent.name == "TestAgent"
        assert agent.ticker == "TEST"
        assert agent.balance == 50.0

    @patch("singularity.autonomous_agent.PluginLoader")
    @patch("singularity.autonomous_agent.CognitionEngine")
    def test_default_values(self, MockCognition, MockLoader):
        from singularity.autonomous_agent import AutonomousAgent
        agent = AutonomousAgent(llm_provider="none")
        assert agent.name == "Agent"
        assert agent.ticker == "AGENT"
        assert agent.balance == 100.0
        assert agent.cycle == 0
        assert agent.running is False
        assert agent.conversation == []

    @patch("singularity.autonomous_agent.PluginLoader")
    @patch("singularity.autonomous_agent.CognitionEngine")
    def test_instance_costs(self, MockCognition, MockLoader):
        from singularity.autonomous_agent import AutonomousAgent
        agent = AutonomousAgent(instance_type="e2-micro", llm_provider="none")
        assert agent.instance_cost_per_hour == 0.0084

    @patch("singularity.autonomous_agent.PluginLoader")
    @patch("singularity.autonomous_agent.CognitionEngine")
    def test_local_instance_free(self, MockCognition, MockLoader):
        from singularity.autonomous_agent import AutonomousAgent
        agent = AutonomousAgent(instance_type="local", llm_provider="none")
        assert agent.instance_cost_per_hour == 0.0

    @patch("singularity.autonomous_agent.PluginLoader")
    @patch("singularity.autonomous_agent.CognitionEngine")
    def test_unknown_instance_free(self, MockCognition, MockLoader):
        from singularity.autonomous_agent import AutonomousAgent
        agent = AutonomousAgent(instance_type="unknown", llm_provider="none")
        assert agent.instance_cost_per_hour == 0.0

    @patch("singularity.autonomous_agent.PluginLoader")
    @patch("singularity.autonomous_agent.CognitionEngine")
    def test_created_resources_initialized(self, MockCognition, MockLoader):
        from singularity.autonomous_agent import AutonomousAgent
        agent = AutonomousAgent(llm_provider="none")
        assert "payment_links" in agent.created_resources
        assert "products" in agent.created_resources
        assert "files" in agent.created_resources
        assert "repos" in agent.created_resources


class TestExecute:
    """Test the _execute method."""

    @patch("singularity.autonomous_agent.PluginLoader")
    @patch("singularity.autonomous_agent.CognitionEngine")
    def _make_agent(self, MockCognition, MockLoader):
        from singularity.autonomous_agent import AutonomousAgent
        return AutonomousAgent(llm_provider="none")

    @pytest.mark.asyncio
    async def test_execute_wait(self):
        agent = self._make_agent()
        result = await agent._execute(Action(tool="wait"))
        assert result["status"] == "waited"

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self):
        agent = self._make_agent()
        result = await agent._execute(Action(tool="nonexistent"))
        assert result["status"] == "error"
        assert "Unknown tool" in result["message"]

    @pytest.mark.asyncio
    async def test_execute_skill_action(self):
        agent = self._make_agent()
        mock_skill = MagicMock()
        mock_skill.execute = AsyncMock(return_value=SkillResult(success=True, message="Done", data={"key": "val"}))
        agent.skills.skills["mock"] = mock_skill
        result = await agent._execute(Action(tool="mock:do_thing", params={"p": "v"}))
        assert result["status"] == "success"
        assert result["data"]["key"] == "val"
        mock_skill.execute.assert_called_once_with("do_thing", {"p": "v"})

    @pytest.mark.asyncio
    async def test_execute_skill_failure(self):
        agent = self._make_agent()
        mock_skill = MagicMock()
        mock_skill.execute = AsyncMock(return_value=SkillResult(success=False, message="API error"))
        agent.skills.skills["mock"] = mock_skill
        result = await agent._execute(Action(tool="mock:action"))
        assert result["status"] == "failed"

    @pytest.mark.asyncio
    async def test_execute_skill_exception(self):
        agent = self._make_agent()
        mock_skill = MagicMock()
        mock_skill.execute = AsyncMock(side_effect=RuntimeError("Boom"))
        agent.skills.skills["mock"] = mock_skill
        result = await agent._execute(Action(tool="mock:action"))
        assert result["status"] == "error"
        assert "Boom" in result["message"]

    @pytest.mark.asyncio
    async def test_execute_skill_not_installed(self):
        agent = self._make_agent()
        result = await agent._execute(Action(tool="missing_skill:action"))
        assert result["status"] == "error"


class TestGetTools:
    """Test the _get_tools method."""

    @patch("singularity.autonomous_agent.PluginLoader")
    @patch("singularity.autonomous_agent.CognitionEngine")
    def test_no_skills_returns_wait(self, MockCognition, MockLoader):
        from singularity.autonomous_agent import AutonomousAgent
        agent = AutonomousAgent(llm_provider="none")
        agent.skills.skills = {}
        tools = agent._get_tools()
        assert len(tools) == 1
        assert tools[0]["name"] == "wait"

    @patch("singularity.autonomous_agent.PluginLoader")
    @patch("singularity.autonomous_agent.CognitionEngine")
    def test_with_skills(self, MockCognition, MockLoader):
        from singularity.autonomous_agent import AutonomousAgent
        from tests.test_skill_system import MockSkill
        agent = AutonomousAgent(llm_provider="none")
        agent.skills.skills = {"mock": MockSkill()}
        tools = agent._get_tools()
        assert len(tools) == 2  # do_thing + do_other
        names = [t["name"] for t in tools]
        assert "mock:do_thing" in names
        assert "mock:do_other" in names


class TestStop:
    @patch("singularity.autonomous_agent.PluginLoader")
    @patch("singularity.autonomous_agent.CognitionEngine")
    def test_stop(self, MockCognition, MockLoader):
        from singularity.autonomous_agent import AutonomousAgent
        agent = AutonomousAgent(llm_provider="none")
        agent.running = True
        agent.stop()
        assert agent.running is False


class TestInstanceCosts:
    def test_all_instance_types_defined(self):
        from singularity.autonomous_agent import AutonomousAgent
        assert "e2-micro" in AutonomousAgent.INSTANCE_COSTS
        assert "e2-small" in AutonomousAgent.INSTANCE_COSTS
        assert "local" in AutonomousAgent.INSTANCE_COSTS
        assert AutonomousAgent.INSTANCE_COSTS["local"] == 0.0
