"""Tests for SpamGuard - spam protection for agent messaging.

Uses unittest (no pytest dependency required).
"""
import asyncio
import sys
import time
import unittest
from pathlib import Path

# Bypass heavy imports in singularity.__init__ and skills.__init__
# by stubbing the packages before importing the modules we need.
sys.path.insert(0, str(Path(__file__).parent.parent))

import types

_root = str(Path(__file__).parent.parent / "singularity")
_skills_root = str(Path(__file__).parent.parent / "singularity" / "skills")

# Stub singularity package (avoids autonomous_agent -> dotenv)
_pkg = types.ModuleType("singularity")
_pkg.__path__ = [_root]
_pkg.__package__ = "singularity"
sys.modules["singularity"] = _pkg

# Stub singularity.skills package (avoids email -> httpx, etc.)
_skills_pkg = types.ModuleType("singularity.skills")
_skills_pkg.__path__ = [_skills_root]
_skills_pkg.__package__ = "singularity.skills"
sys.modules["singularity.skills"] = _skills_pkg

# Now import only the modules we need (base + messaging — stdlib only)
from singularity.skills.base import Skill, SkillResult, SkillManifest, SkillAction  # noqa: E402,F401
from singularity.skills.messaging import MessagingSkill, SpamGuard  # noqa: E402


class TestSpamGuardRateLimit(unittest.TestCase):
    """Test per-sender rate limiting."""

    def test_allows_messages_under_limit(self):
        guard = SpamGuard(rate_limit=5, rate_window=60)
        for i in range(5):
            self.assertIsNone(guard.check("agent_a", f"msg {i}"))
            guard.record("agent_a", f"msg {i}")

    def test_blocks_messages_over_limit(self):
        guard = SpamGuard(rate_limit=3, rate_window=60)
        for i in range(3):
            guard.record("agent_a", f"msg {i}")
        result = guard.check("agent_a", "one more")
        self.assertIsNotNone(result)
        self.assertIn("Rate limit exceeded", result)

    def test_rate_limit_per_sender(self):
        """Different senders have independent limits."""
        guard = SpamGuard(rate_limit=2, rate_window=60)
        guard.record("agent_a", "msg1")
        guard.record("agent_a", "msg2")
        # agent_a is blocked
        self.assertIsNotNone(guard.check("agent_a", "msg3"))
        # agent_b is fine
        self.assertIsNone(guard.check("agent_b", "msg1"))

    def test_rate_limit_resets_after_window(self):
        guard = SpamGuard(rate_limit=2, rate_window=1)
        guard.record("agent_a", "msg1")
        guard.record("agent_a", "msg2")
        self.assertIsNotNone(guard.check("agent_a", "msg3"))
        # Wait for window to expire
        time.sleep(1.1)
        self.assertIsNone(guard.check("agent_a", "msg3"))


class TestSpamGuardDuplicateDetection(unittest.TestCase):
    """Test duplicate message detection."""

    def test_allows_first_messages_up_to_limit(self):
        guard = SpamGuard(max_duplicates=2, duplicate_window=60)
        self.assertIsNone(guard.check("agent_a", "OFFERING: Buy my stuff!"))
        guard.record("agent_a", "OFFERING: Buy my stuff!")
        # Second identical is still OK (max_duplicates=2)
        self.assertIsNone(guard.check("agent_a", "OFFERING: Buy my stuff!"))
        guard.record("agent_a", "OFFERING: Buy my stuff!")

    def test_blocks_excessive_duplicates(self):
        guard = SpamGuard(max_duplicates=2, duplicate_window=60)
        for _ in range(2):
            guard.record("agent_a", "spam spam spam")
        result = guard.check("agent_a", "spam spam spam")
        self.assertIsNotNone(result)
        self.assertIn("Duplicate message blocked", result)

    def test_different_content_not_flagged(self):
        guard = SpamGuard(max_duplicates=1, duplicate_window=60)
        guard.record("agent_a", "message one")
        self.assertIsNone(guard.check("agent_a", "message two"))

    def test_duplicate_window_expires(self):
        guard = SpamGuard(max_duplicates=1, duplicate_window=1)
        guard.record("agent_a", "repeat this")
        self.assertIsNotNone(guard.check("agent_a", "repeat this"))
        time.sleep(1.1)
        self.assertIsNone(guard.check("agent_a", "repeat this"))

    def test_duplicate_per_sender(self):
        """Duplicates tracked independently per sender."""
        guard = SpamGuard(max_duplicates=1, duplicate_window=60)
        guard.record("agent_a", "shared message")
        # agent_a can't repeat
        self.assertIsNotNone(guard.check("agent_a", "shared message"))
        # agent_b can still say the same thing
        self.assertIsNone(guard.check("agent_b", "shared message"))


class TestSpamGuardInternal(unittest.TestCase):
    """Test internal helper methods."""

    def test_hash_content_deterministic(self):
        h1 = SpamGuard._hash_content("hello world")
        h2 = SpamGuard._hash_content("hello world")
        self.assertEqual(h1, h2)
        self.assertEqual(len(h1), 16)

    def test_hash_content_different_for_different_input(self):
        h1 = SpamGuard._hash_content("hello")
        h2 = SpamGuard._hash_content("world")
        self.assertNotEqual(h1, h2)

    def test_cleanup_removes_old_entries(self):
        guard = SpamGuard(rate_window=1, duplicate_window=1)
        guard.record("a", "msg")
        self.assertEqual(len(guard._send_times["a"]), 1)
        time.sleep(1.1)
        guard._cleanup("a", time.time())
        self.assertEqual(len(guard._send_times["a"]), 0)
        self.assertEqual(len(guard._content_hashes["a"]), 0)


class TestSpamGuardMessagingIntegration(unittest.TestCase):
    """Integration tests: SpamGuard within MessagingSkill."""

    def setUp(self):
        import tempfile
        self._tmpdir = tempfile.mkdtemp()
        self.strict_skill = MessagingSkill(credentials={
            "data_path": str(Path(self._tmpdir) / "messages.json"),
            "rate_limit": "3",
            "rate_window": "60",
            "max_duplicates": "1",
            "duplicate_window": "60",
        })
        self.default_skill = MessagingSkill(credentials={
            "data_path": str(Path(self._tmpdir) / "default_messages.json"),
        })

    def _run(self, coro):
        """Helper to run async code in tests."""
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_blocks_flood(self):
        """Sending too many messages triggers rate limit."""
        for i in range(3):
            r = self._run(self.strict_skill.execute("send", {
                "from_instance_id": "spammer",
                "to_instance_id": "victim",
                "content": f"Message {i}",
            }))
            self.assertTrue(r.success, f"Message {i} should succeed")

        # 4th message should be blocked
        r = self._run(self.strict_skill.execute("send", {
            "from_instance_id": "spammer",
            "to_instance_id": "victim",
            "content": "Message 3",
        }))
        self.assertFalse(r.success)
        self.assertIn("spam guard", r.message.lower())
        self.assertTrue(r.data["blocked"])

    def test_blocks_duplicates(self):
        """Identical messages trigger duplicate detection."""
        r = self._run(self.strict_skill.execute("send", {
            "from_instance_id": "spammer",
            "to_instance_id": "victim",
            "content": "OFFERING: Buy my services!",
        }))
        self.assertTrue(r.success)

        # Same content again → blocked
        r = self._run(self.strict_skill.execute("send", {
            "from_instance_id": "spammer",
            "to_instance_id": "victim",
            "content": "OFFERING: Buy my services!",
        }))
        self.assertFalse(r.success)
        self.assertIn("Duplicate", r.message)

    def test_allows_different_senders(self):
        """Different senders are not affected by each other's limits."""
        r = self._run(self.strict_skill.execute("send", {
            "from_instance_id": "agent_a",
            "to_instance_id": "agent_c",
            "content": "Hello there!",
        }))
        self.assertTrue(r.success)

        # agent_b can send the same content
        r = self._run(self.strict_skill.execute("send", {
            "from_instance_id": "agent_b",
            "to_instance_id": "agent_c",
            "content": "Hello there!",
        }))
        self.assertTrue(r.success)

    def test_default_config_is_permissive(self):
        """Default spam guard allows normal use (10 unique messages)."""
        for i in range(10):
            r = self._run(self.default_skill.execute("send", {
                "from_instance_id": "normal_agent",
                "to_instance_id": "recipient",
                "content": f"Normal message #{i} with unique content",
            }))
            self.assertTrue(r.success, f"Message {i} should succeed with default config")

    def test_existing_tests_still_pass_send(self):
        """Verify backward compatibility — normal send still works."""
        r = self._run(self.default_skill.execute("send", {
            "from_instance_id": "agent_eve",
            "to_instance_id": "agent_adam",
            "content": "Hello Adam, want to bundle our services?",
        }))
        self.assertTrue(r.success)
        self.assertEqual(r.data["to"], "agent_adam")
        self.assertTrue(r.data["message_id"].startswith("msg_"))
        self.assertTrue(r.data["conversation_id"].startswith("conv_"))


if __name__ == "__main__":
    unittest.main()
