"""
Comprehensive tests for singularity.skills.base.registry — the SkillRegistry
that manages skill installation, credentials, and execution.
"""

import asyncio
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
exec(open(str(Path(__file__).resolve().parent / "conftest.py")).read())

from singularity.skills.base.types import SkillResult, SkillAction, SkillManifest
from singularity.skills.base.skill import Skill
from singularity.skills.base.registry import SkillRegistry


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ── Test Skills ─────────────────────────────────────────────────────


class EchoSkill(Skill):
    """Test skill that echoes parameters."""

    @property
    def manifest(self):
        return SkillManifest(
            skill_id="echo",
            name="Echo Skill",
            version="1.0.0",
            category="test",
            description="Echoes parameters back",
            actions=[
                SkillAction(name="echo", description="Echo params",
                           parameters={"text": {"type": "string"}}),
                SkillAction(name="reverse", description="Reverse text",
                           parameters={"text": {"type": "string"}}),
            ],
            required_credentials=[],
        )

    async def execute(self, action, params):
        if action == "echo":
            text = params.get("text", "")
            return SkillResult(success=True, message=text, data={"text": text})
        elif action == "reverse":
            text = params.get("text", "")
            rev = text[::-1]
            return SkillResult(success=True, message=rev, data={"text": rev})
        return SkillResult(success=False, message=f"Unknown: {action}")


class SecureSkill(Skill):
    """Test skill that requires credentials."""

    @property
    def manifest(self):
        return SkillManifest(
            skill_id="secure",
            name="Secure Skill",
            version="1.0.0",
            category="test",
            description="Needs API key",
            actions=[
                SkillAction(name="auth_action", description="Requires auth",
                           parameters={}, estimated_cost=0.01),
            ],
            required_credentials=["SECRET_KEY"],
        )

    async def execute(self, action, params):
        return SkillResult(success=True, message="Authorized")


class CostlySkill(Skill):
    """Test skill that tracks costs and revenue."""

    @property
    def manifest(self):
        return SkillManifest(
            skill_id="costly",
            name="Costly Skill",
            version="1.0.0",
            category="test",
            description="Has costs and revenue",
            actions=[
                SkillAction(name="earn", description="Earn money", parameters={}),
                SkillAction(name="spend", description="Spend money", parameters={}),
            ],
            required_credentials=[],
        )

    async def execute(self, action, params):
        if action == "earn":
            return SkillResult(success=True, message="Earned", revenue=10.0)
        elif action == "spend":
            return SkillResult(success=True, message="Spent", cost=5.0)
        return SkillResult(success=False, message="Unknown")


# ── Registry Tests ──────────────────────────────────────────────────


class TestRegistryInit(unittest.TestCase):
    """Test SkillRegistry initialization."""

    def test_empty_registry(self):
        reg = SkillRegistry()
        self.assertEqual(len(reg.skills), 0)
        self.assertEqual(len(reg.credentials), 0)
        self.assertIsNone(reg.loader)

    def test_with_loader(self):
        loader = MagicMock()
        reg = SkillRegistry(loader=loader)
        self.assertEqual(reg.loader, loader)


class TestRegistryInstall(unittest.TestCase):
    """Test skill installation."""

    def test_install_by_class(self):
        reg = SkillRegistry()
        result = reg.install(EchoSkill)
        self.assertTrue(result)
        self.assertIn("echo", reg.skills)

    def test_install_multiple(self):
        reg = SkillRegistry()
        reg.install(EchoSkill)
        reg.install(CostlySkill)
        self.assertEqual(len(reg.skills), 2)
        self.assertIn("echo", reg.skills)
        self.assertIn("costly", reg.skills)

    def test_install_with_credentials(self):
        reg = SkillRegistry()
        reg.install(SecureSkill, skill_credentials={"SECRET_KEY": "mykey"})
        skill = reg.get("secure")
        self.assertIsNotNone(skill)
        self.assertEqual(skill.credentials["SECRET_KEY"], "mykey")

    def test_install_by_id_no_loader(self):
        reg = SkillRegistry()
        result = reg.install("some_skill")
        self.assertFalse(result)


class TestRegistryUninstall(unittest.TestCase):
    """Test skill uninstallation."""

    def test_uninstall_existing(self):
        reg = SkillRegistry()
        reg.install(EchoSkill)
        result = reg.uninstall("echo")
        self.assertTrue(result)
        self.assertNotIn("echo", reg.skills)

    def test_uninstall_nonexistent(self):
        reg = SkillRegistry()
        result = reg.uninstall("nonexistent")
        self.assertFalse(result)


class TestRegistryGet(unittest.TestCase):
    """Test skill retrieval."""

    def test_get_existing(self):
        reg = SkillRegistry()
        reg.install(EchoSkill)
        skill = reg.get("echo")
        self.assertIsNotNone(skill)
        self.assertIsInstance(skill, EchoSkill)

    def test_get_nonexistent(self):
        reg = SkillRegistry()
        skill = reg.get("nonexistent")
        self.assertIsNone(skill)


class TestRegistryCredentials(unittest.TestCase):
    """Test credential management."""

    def test_set_credentials(self):
        reg = SkillRegistry()
        reg.set_credentials({"API_KEY": "test123"})
        self.assertEqual(reg.credentials["API_KEY"], "test123")

    def test_credentials_propagate_to_skills(self):
        reg = SkillRegistry()
        reg.install(EchoSkill)
        reg.set_credentials({"API_KEY": "test123"})
        skill = reg.get("echo")
        self.assertEqual(skill.credentials["API_KEY"], "test123")

    def test_credentials_available_at_install(self):
        reg = SkillRegistry()
        reg.set_credentials({"SECRET_KEY": "mykey"})
        reg.install(SecureSkill)
        skill = reg.get("secure")
        self.assertEqual(skill.credentials["SECRET_KEY"], "mykey")


class TestRegistryListSkills(unittest.TestCase):
    """Test skill listing."""

    def test_list_empty(self):
        reg = SkillRegistry()
        self.assertEqual(reg.list_skills(), [])

    def test_list_with_skills(self):
        reg = SkillRegistry()
        reg.install(EchoSkill)
        reg.install(CostlySkill)
        skills = reg.list_skills()
        self.assertEqual(len(skills), 2)
        ids = {s["skill_id"] for s in skills}
        self.assertEqual(ids, {"echo", "costly"})


class TestRegistryListActions(unittest.TestCase):
    """Test action listing."""

    def test_list_empty(self):
        reg = SkillRegistry()
        self.assertEqual(reg.list_all_actions(), [])

    def test_list_actions(self):
        reg = SkillRegistry()
        reg.install(EchoSkill)
        actions = reg.list_all_actions()
        self.assertEqual(len(actions), 2)  # echo + reverse
        names = {a["action"] for a in actions}
        self.assertEqual(names, {"echo", "reverse"})

    def test_action_structure(self):
        reg = SkillRegistry()
        reg.install(EchoSkill)
        actions = reg.list_all_actions()
        for action in actions:
            self.assertIn("skill_id", action)
            self.assertIn("skill_name", action)
            self.assertIn("action", action)
            self.assertIn("description", action)
            self.assertIn("parameters", action)
            self.assertIn("estimated_cost", action)

    def test_multiple_skills(self):
        reg = SkillRegistry()
        reg.install(EchoSkill)
        reg.install(CostlySkill)
        actions = reg.list_all_actions()
        self.assertEqual(len(actions), 4)  # 2 echo + 2 costly


class TestRegistryExecute(unittest.TestCase):
    """Test skill execution through registry."""

    def test_execute_success(self):
        reg = SkillRegistry()
        reg.install(EchoSkill)
        result = run(reg.execute("echo", "echo", {"text": "hello"}))
        self.assertTrue(result.success)
        self.assertEqual(result.data["text"], "hello")

    def test_execute_reverse(self):
        reg = SkillRegistry()
        reg.install(EchoSkill)
        result = run(reg.execute("echo", "reverse", {"text": "hello"}))
        self.assertTrue(result.success)
        self.assertEqual(result.data["text"], "olleh")

    def test_execute_nonexistent_skill(self):
        reg = SkillRegistry()
        result = run(reg.execute("nonexistent", "action", {}))
        self.assertFalse(result.success)
        self.assertIn("not found", result.message.lower())

    def test_execute_tracks_usage(self):
        reg = SkillRegistry()
        reg.install(CostlySkill)
        run(reg.execute("costly", "earn", {}))
        skill = reg.get("costly")
        self.assertEqual(skill._usage_count, 1)
        self.assertEqual(skill._total_revenue, 10.0)

    def test_execute_tracks_cost(self):
        reg = SkillRegistry()
        reg.install(CostlySkill)
        run(reg.execute("costly", "spend", {}))
        skill = reg.get("costly")
        self.assertEqual(skill._total_cost, 5.0)

    def test_execute_multiple(self):
        reg = SkillRegistry()
        reg.install(CostlySkill)
        run(reg.execute("costly", "earn", {}))
        run(reg.execute("costly", "earn", {}))
        run(reg.execute("costly", "spend", {}))
        skill = reg.get("costly")
        self.assertEqual(skill._usage_count, 3)
        self.assertEqual(skill._total_revenue, 20.0)
        self.assertEqual(skill._total_cost, 5.0)

    def test_execute_missing_credentials(self):
        reg = SkillRegistry()
        reg.install(SecureSkill)  # No credentials provided
        result = run(reg.execute("secure", "auth_action", {}))
        # Should fail due to missing credentials during initialize()
        self.assertFalse(result.success)
        self.assertIn("credentials", result.message.lower())


class TestRegistryGetSkillsForLLM(unittest.TestCase):
    """Test LLM-friendly skill listing."""

    def test_empty_registry(self):
        reg = SkillRegistry()
        output = reg.get_skills_for_llm()
        self.assertIn("INSTALLED SKILLS", output)

    def test_with_skills(self):
        reg = SkillRegistry()
        reg.install(EchoSkill)
        output = reg.get_skills_for_llm()
        self.assertIn("echo", output)
        self.assertIn("Echo Skill", output)
        self.assertIn("reverse", output)
        self.assertIn("text", output)

    def test_cost_in_output(self):
        reg = SkillRegistry()
        reg.install(CostlySkill)
        output = reg.get_skills_for_llm()
        self.assertIn("$", output)

    def test_multiple_skills(self):
        reg = SkillRegistry()
        reg.install(EchoSkill)
        reg.install(CostlySkill)
        output = reg.get_skills_for_llm()
        self.assertIn("[echo]", output)
        self.assertIn("[costly]", output)


class TestRegistrySetAgent(unittest.TestCase):
    """Test agent reference setting."""

    def test_set_agent(self):
        reg = SkillRegistry()
        mock_agent = MagicMock()
        reg.set_agent(mock_agent)
        self.assertEqual(reg._agent, mock_agent)


class TestRegistryInstallAllAvailable(unittest.TestCase):
    """Test install_all_available with mock loader."""

    def test_no_loader(self):
        reg = SkillRegistry()
        result = reg.install_all_available()
        self.assertEqual(result, [])

    def test_with_loader(self):
        mock_loader = MagicMock()
        mock_metadata = MagicMock()
        mock_metadata.skill_id = "test_skill"
        mock_loader.list_available.return_value = [mock_metadata]
        mock_loader.check_credentials.return_value = True
        mock_loader.load.return_value = EchoSkill()

        reg = SkillRegistry(loader=mock_loader)
        installed = reg.install_all_available()
        # Should attempt to install the available skill
        self.assertIsInstance(installed, list)


if __name__ == "__main__":
    unittest.main()
