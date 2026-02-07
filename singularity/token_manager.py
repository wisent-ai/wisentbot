"""Token Manager - Context window management for LLM interactions."""
from dataclasses import dataclass, field
from typing import Dict, List, Any

MODEL_CONTEXT_LIMITS = {
    "claude-sonnet-4-20250514": 200_000, "claude-3-5-sonnet-20241022": 200_000,
    "claude-3-haiku-20240307": 200_000, "claude-3-5-haiku-20241022": 200_000,
    "gpt-4o": 128_000, "gpt-4o-mini": 128_000, "gpt-4-turbo": 128_000,
    "gpt-4": 8_192, "gpt-3.5-turbo": 16_385,
    "gemini-2.0-flash-001": 1_048_576, "gemini-1.5-pro-002": 2_097_152,
    "claude-3-5-sonnet-v2@20241022": 200_000, "_default": 8_192,
}
DEFAULT_OUTPUT_RESERVE = 2_048

def estimate_tokens(text):
    if not text: return 0
    return max(1, len(text) // 4)

def get_context_limit(model):
    if model in MODEL_CONTEXT_LIMITS: return MODEL_CONTEXT_LIMITS[model]
    for key, limit in MODEL_CONTEXT_LIMITS.items():
        if model.startswith(key.split("-")[0]): return limit
    return MODEL_CONTEXT_LIMITS["_default"]

@dataclass
class TokenBudget:
    total_limit: int; output_reserve: int; system_prompt: int = 0
    user_prompt_fixed: int = 0; recent_actions: int = 0; project_context: int = 0; available: int = 0
    @property
    def input_limit(self): return self.total_limit - self.output_reserve
    def compute_available(self):
        self.available = self.input_limit - self.system_prompt - self.user_prompt_fixed
        return self.available

@dataclass
class SessionTokenStats:
    total_input_tokens: int = 0; total_output_tokens: int = 0
    total_cost_usd: float = 0.0; call_count: int = 0; peak_input_tokens: int = 0
    def record(self, input_tokens, output_tokens, cost_usd=0.0):
        self.total_input_tokens += input_tokens; self.total_output_tokens += output_tokens
        self.total_cost_usd += cost_usd; self.call_count += 1
        if input_tokens > self.peak_input_tokens: self.peak_input_tokens = input_tokens
    @property
    def total_tokens(self): return self.total_input_tokens + self.total_output_tokens
    @property
    def avg_input_tokens(self): return self.total_input_tokens // self.call_count if self.call_count else 0
    def summary(self):
        return {"calls": self.call_count, "total_input_tokens": self.total_input_tokens,
                "total_output_tokens": self.total_output_tokens, "total_tokens": self.total_tokens,
                "peak_input_tokens": self.peak_input_tokens, "avg_input_tokens": self.avg_input_tokens,
                "total_cost_usd": round(self.total_cost_usd, 6)}

class TokenManager:
    def __init__(self, model="claude-sonnet-4-20250514", output_reserve=DEFAULT_OUTPUT_RESERVE, safety_margin=0.95):
        self.model = model; self.context_limit = get_context_limit(model)
        self.output_reserve = output_reserve; self.safety_margin = safety_margin
        self.session_stats = SessionTokenStats()
    @property
    def effective_limit(self): return int(self.context_limit * self.safety_margin)
    def update_model(self, model):
        self.model = model; self.context_limit = get_context_limit(model)
    def compute_budget(self, system_prompt, user_prompt_base):
        budget = TokenBudget(total_limit=self.effective_limit, output_reserve=self.output_reserve,
                             system_prompt=estimate_tokens(system_prompt), user_prompt_fixed=estimate_tokens(user_prompt_base))
        budget.compute_available(); return budget
    def fit_recent_actions(self, actions, token_budget, min_actions=2):
        if not actions: return []
        if token_budget <= 0: return actions[-min_actions:] if len(actions) >= min_actions else actions
        result, tokens_used = [], 0
        for action in reversed(actions):
            t = f"- {action.get('tool','?')}: {action.get('result',{}).get('status','?')}"
            at = estimate_tokens(t)
            if tokens_used + at > token_budget and len(result) >= min_actions: break
            result.append(action); tokens_used += at
        result.reverse(); return result
    def truncate_to_budget(self, text, token_budget, suffix="\n... [truncated to fit context window]"):
        if not text: return text
        if estimate_tokens(text) <= token_budget: return text
        if token_budget <= 0: return ""
        char_limit = max(0, token_budget - estimate_tokens(suffix)) * 4
        return text[:char_limit] + suffix if char_limit > 0 else ""
    def will_fit(self, system_prompt, user_prompt):
        return estimate_tokens(system_prompt) + estimate_tokens(user_prompt) + self.output_reserve <= self.effective_limit
    def utilization(self, system_prompt, user_prompt):
        total = estimate_tokens(system_prompt) + estimate_tokens(user_prompt) + self.output_reserve
        return total / self.effective_limit if self.effective_limit > 0 else 1.0
    def record_usage(self, input_tokens, output_tokens, cost_usd=0.0):
        self.session_stats.record(input_tokens, output_tokens, cost_usd)
    def get_stats(self):
        return {"model": self.model, "context_limit": self.context_limit,
                "effective_limit": self.effective_limit, "output_reserve": self.output_reserve,
                **self.session_stats.summary()}
