"""
Shared test fixtures and mocks for the singularity test suite.

All external dependencies (anthropic, openai, vertexai, etc.) are mocked
so tests run without any API keys or network access.
"""

import sys
import types
from unittest.mock import MagicMock
import pytest


# ─── Mock external dependencies before any singularity imports ──────────


def _make_mock_module(name: str, attrs: dict = None) -> types.ModuleType:
    """Create a mock module with optional attributes."""
    mod = types.ModuleType(name)
    mod.__spec__ = MagicMock()
    for k, v in (attrs or {}).items():
        setattr(mod, k, v)
    return mod


# Mock anthropic
_anthropic = _make_mock_module("anthropic", {
    "AsyncAnthropic": MagicMock,
    "AnthropicVertex": MagicMock,
    "Anthropic": MagicMock,
})
sys.modules.setdefault("anthropic", _anthropic)

# Mock openai
_openai = _make_mock_module("openai", {
    "AsyncOpenAI": MagicMock,
    "OpenAI": MagicMock,
})
sys.modules.setdefault("openai", _openai)

# Mock vertex AI
_vertexai = _make_mock_module("vertexai")
_vertexai_gen = _make_mock_module("vertexai.generative_models", {
    "GenerativeModel": MagicMock,
    "GenerationConfig": MagicMock,
})
sys.modules.setdefault("vertexai", _vertexai)
sys.modules.setdefault("vertexai.generative_models", _vertexai_gen)

# Mock google cloud
_google = _make_mock_module("google")
_google_cloud = _make_mock_module("google.cloud")
_google_aiplatform = _make_mock_module("google.cloud.aiplatform")
sys.modules.setdefault("google", _google)
sys.modules.setdefault("google.cloud", _google_cloud)
sys.modules.setdefault("google.cloud.aiplatform", _google_aiplatform)

# Mock torch with realistic CUDA/MPS attribute structure
_torch = _make_mock_module("torch")
_torch_cuda = MagicMock()
_torch_cuda.is_available = MagicMock(return_value=False)
_torch.cuda = _torch_cuda
_torch_backends = MagicMock()
_torch_backends_mps = MagicMock()
_torch_backends_mps.is_available = MagicMock(return_value=False)
_torch_backends.mps = _torch_backends_mps
_torch.backends = _torch_backends
_torch.float16 = "float16"
sys.modules.setdefault("torch", _torch)

sys.modules.setdefault("transformers", _make_mock_module("transformers", {
    "AutoTokenizer": MagicMock,
    "AutoModelForCausalLM": MagicMock,
}))
sys.modules.setdefault("vllm", _make_mock_module("vllm", {
    "LLM": MagicMock,
    "SamplingParams": MagicMock,
}))

# Mock aiohttp (used by some skills)
_aiohttp = _make_mock_module("aiohttp", {
    "ClientSession": MagicMock,
})
sys.modules.setdefault("aiohttp", _aiohttp)

# Mock dotenv
_dotenv = _make_mock_module("dotenv", {
    "load_dotenv": MagicMock(),
})
sys.modules.setdefault("dotenv", _dotenv)

# Mock playwright
_pw = _make_mock_module("playwright")
_pw_async = _make_mock_module("playwright.async_api", {
    "async_playwright": MagicMock,
})
sys.modules.setdefault("playwright", _pw)
sys.modules.setdefault("playwright.async_api", _pw_async)

# Mock httpx at module level for skills that import it
_httpx = _make_mock_module("httpx")
_httpx.AsyncClient = MagicMock
_httpx.Response = MagicMock
sys.modules.setdefault("httpx", _httpx)

# Mock wisent / cognee (optional steering/memory)
sys.modules.setdefault("wisent", _make_mock_module("wisent"))
sys.modules.setdefault("cognee", _make_mock_module("cognee"))

# Mock web3 / eth_account (crypto skills)
sys.modules.setdefault("web3", _make_mock_module("web3", {"Web3": MagicMock}))
sys.modules.setdefault("eth_account", _make_mock_module("eth_account"))

# Mock tweepy (twitter)
sys.modules.setdefault("tweepy", _make_mock_module("tweepy"))

# Mock stripe
sys.modules.setdefault("stripe", _make_mock_module("stripe"))

# Mock resend
sys.modules.setdefault("resend", _make_mock_module("resend"))

# Mock beautifulsoup4
_bs4 = _make_mock_module("bs4", {"BeautifulSoup": MagicMock})
sys.modules.setdefault("bs4", _bs4)

# Mock PyGithub
sys.modules.setdefault("github", _make_mock_module("github", {"Github": MagicMock}))


# ─── Fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
def mock_credentials():
    """Return a dict of fake credentials for testing."""
    return {
        "ANTHROPIC_API_KEY": "sk-ant-test-key",
        "OPENAI_API_KEY": "sk-test-key",
        "GITHUB_TOKEN": "ghp_test_token",
        "TWITTER_API_KEY": "test_twitter_key",
        "TWITTER_API_SECRET": "test_twitter_secret",
        "TWITTER_ACCESS_TOKEN": "test_twitter_access",
        "TWITTER_ACCESS_SECRET": "test_twitter_access_secret",
        "RESEND_API_KEY": "re_test_key",
        "STRIPE_SECRET_KEY": "sk_test_stripe",
        "VERCEL_TOKEN": "test_vercel_token",
    }


@pytest.fixture
def sample_agent_state():
    """Return a minimal AgentState for testing."""
    from singularity.cognition.types import AgentState
    return AgentState(
        balance=10.0,
        burn_rate=0.02,
        runway_hours=500.0,
        tools=[
            {"skill_id": "github", "actions": ["create_repo", "create_issue"]},
            {"skill_id": "shell", "actions": ["run_command"]},
        ],
        recent_actions=[],
        cycle=1,
        chat_messages=[],
        project_context="",
        goals_progress={},
        pending_tasks=[],
        created_resources={},
    )


@pytest.fixture
def sample_action():
    """Return a sample Action for testing."""
    from singularity.cognition.types import Action
    return Action(
        tool="github:create_issue",
        params={"repo": "test/repo", "title": "Test issue"},
        reasoning="Testing issue creation",
    )
