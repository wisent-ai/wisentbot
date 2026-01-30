# WisentBot

An open-source framework for building autonomous AI agents that can execute tasks, manage resources, and interact with the real world.

## Features

- **Multi-LLM Support**: Anthropic Claude, OpenAI GPT, Google Vertex AI (Gemini), local models via vLLM/Transformers
- **Self-Modification**: Agents can edit their prompts, switch models, and fine-tune themselves
- **Activation Steering**: Integration with [wisent](https://github.com/wisent-ai/wisent) for representation engineering
- **Modular Skills**: Extensible skill system for adding new capabilities
- **Cost Tracking**: Built-in API cost and resource tracking
- **Async First**: Fully asynchronous for high performance

## Installation

```bash
pip install wisentbot
```

With optional dependencies:

```bash
# All features
pip install wisentbot[all]

# Specific features
pip install wisentbot[twitter,github,browser]

# Local GPU inference
pip install wisentbot[gpu]

# Activation steering (wisent integration)
pip install wisentbot[steering]
```

## Quick Start

```python
import asyncio
from wisentbot import AutonomousAgent

async def main():
    agent = AutonomousAgent(
        name="MyAgent",
        ticker="AGENT",
        starting_balance=10.0,  # USD budget
        llm_provider="anthropic",
        llm_model="claude-sonnet-4-20250514",
    )
    await agent.run()

asyncio.run(main())
```

## Environment Variables

```bash
# Required (at least one LLM provider)
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...

# Optional - for specific skills
TWITTER_API_KEY=...
TWITTER_API_SECRET=...
TWITTER_ACCESS_TOKEN=...
TWITTER_ACCESS_SECRET=...
GITHUB_TOKEN=ghp_...
RESEND_API_KEY=re_...
VERCEL_TOKEN=...
```

## Available Skills

| Skill | Description | Required Env Vars |
|-------|-------------|-------------------|
| `content` | Generate text content | LLM API key |
| `twitter` | Post tweets, read timeline | `TWITTER_*` |
| `github` | Manage repos, issues, PRs | `GITHUB_TOKEN` |
| `email` | Send emails via Resend | `RESEND_API_KEY` |
| `browser` | Web automation | None (uses Playwright) |
| `filesystem` | Read/write files | None |
| `shell` | Execute shell commands | None |
| `request` | Make HTTP requests | None |
| `vercel` | Deploy to Vercel | `VERCEL_TOKEN` |
| `namecheap` | Manage domains | `NAMECHEAP_*` |
| `mcp` | MCP protocol client | None |
| `self` | Self-modify prompts, switch models, fine-tune | `OPENAI_API_KEY` (for fine-tuning) |
| `steering` | Activation steering via wisent | Local model required |

## Creating Custom Skills

```python
from wisentbot.skills import Skill, SkillManifest, SkillAction, SkillResult

class MySkill(Skill):
    def __init__(self, credentials: dict):
        manifest = SkillManifest(
            skill_id="myskill",
            name="My Custom Skill",
            description="Does something useful",
            actions=[
                SkillAction(
                    name="do_thing",
                    description="Does the thing",
                    parameters={"input": {"type": "string", "required": True}}
                )
            ]
        )
        super().__init__(manifest, credentials)

    def check_credentials(self) -> bool:
        return True  # Or check required credentials

    async def execute(self, action: str, params: dict) -> SkillResult:
        if action == "do_thing":
            # Do the thing
            return SkillResult(success=True, data={"result": "done"})
        return SkillResult(success=False, message=f"Unknown action: {action}")
```

## LLM Providers

### Anthropic (Default)

```python
agent = AutonomousAgent(
    llm_provider="anthropic",
    llm_model="claude-sonnet-4-20250514",
)
```

### OpenAI

```python
agent = AutonomousAgent(
    llm_provider="openai",
    llm_model="gpt-4o",
)
```

### Local (vLLM on CUDA)

```python
agent = AutonomousAgent(
    llm_provider="vllm",
    llm_model="Qwen/Qwen2.5-7B-Instruct",
)
```

### Local (Transformers on Apple Silicon)

```python
agent = AutonomousAgent(
    llm_provider="transformers",
    llm_model="Qwen/Qwen2.5-1.5B-Instruct",
)
```

### Google Vertex AI

```python
agent = AutonomousAgent(
    llm_provider="vertex",
    llm_model="gemini-2.0-flash-001",
)
```

## Self-Modification

Agents can modify their own behavior at runtime using the `self` skill:

```python
# Actions available to the agent:
# self:get_prompt - View current system prompt
# self:set_prompt - Replace entire system prompt
# self:append_prompt - Add to system prompt
# self:add_rule - Add a behavioral rule
# self:add_goal - Add a personal goal
# self:add_learning - Record something learned

# Model switching:
# self:list_models - List available models
# self:current_model - Get current model info
# self:switch_model - Switch to different model

# Fine-tuning (requires OpenAI API):
# self:record_experience - Record prompt/response pair
# self:training_stats - View collected examples
# self:start_finetune - Start fine-tuning job
# self:check_finetune - Check job status
# self:use_finetuned - Switch to fine-tuned model
```

## Activation Steering (Wisent Integration)

For agents running on local models, you can use [wisent](https://github.com/wisent-ai/wisent) for activation-level behavior control:

```bash
pip install wisentbot[steering]
```

```python
# Steering actions available to the agent:
# steering:add_contrastive_pair - Add good/bad example pair
# steering:train_steering_vector - Train from collected pairs
# steering:apply_steering - Apply steering to modify behavior
# steering:remove_steering - Remove active steering
# steering:detect_issue - Check if response is problematic
```

This allows agents to:
- Create steering vectors from their experiences
- Modify behavior without retraining weights
- Detect potentially harmful or hallucinatory outputs
- Self-correct at the activation level

## Architecture

```
┌─────────────────────────────────────────────┐
│              AutonomousAgent                │
│  ┌─────────────────────────────────────┐    │
│  │         CognitionEngine             │    │
│  │  ┌─────────┐  ┌─────────────────┐   │    │
│  │  │   LLM   │  │  System Prompt  │   │    │
│  │  └─────────┘  └─────────────────┘   │    │
│  └─────────────────────────────────────┘    │
│  ┌─────────────────────────────────────┐    │
│  │          SkillRegistry              │    │
│  │  ┌───────┐ ┌───────┐ ┌───────┐     │    │
│  │  │Twitter│ │GitHub │ │Browser│ ... │    │
│  │  └───────┘ └───────┘ └───────┘     │    │
│  └─────────────────────────────────────┘    │
│  ┌─────────────────────────────────────┐    │
│  │         State & Resources           │    │
│  │   Balance, Actions, Created Items   │    │
│  └─────────────────────────────────────┘    │
└─────────────────────────────────────────────┘
```

## Agent Loop

1. **Observe**: Gather current state (balance, recent actions, available tools)
2. **Think**: LLM decides what action to take
3. **Act**: Execute the chosen skill action
4. **Record**: Track costs, results, and created resources
5. **Repeat**: Continue until balance depleted or stopped

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please read our [Contributing Guide](CONTRIBUTING.md) first.

## Links

- [Documentation](https://github.com/wisent-ai/wisentbot)
- [Issues](https://github.com/wisent-ai/wisentbot/issues)
- [Wisent AI](https://wisent.ai)
