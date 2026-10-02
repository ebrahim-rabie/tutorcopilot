from __future__ import annotations

import time
from dataclasses import dataclass, field

from app.generate import Generator, Judge
from app.settings import Settings


@dataclass
class RateLimitBucket:
    """Token bucket for rate limiting."""

    capacity: int
    refill_per_sec: float
    tokens: float = field(default=None, init=False)
    last_refill: float = field(default=None, init=False)

    def __post_init__(self):
        self.tokens = float(self.capacity)
        self.last_refill = time.time()

    def consume(self, n: int = 1) -> bool:
        self._refill()
        if self.tokens >= n:
            self.tokens -= n
            return True
        return False

    def _refill(self):
        now = time.time()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_per_sec)
        self.last_refill = now

    def available(self) -> float:
        self._refill()
        return self.tokens


class RateLimiter:
    """Per-key rate limiter with token buckets."""

    def __init__(self, default_capacity: int = 100, default_refill: float = 10.0):
        self.default_capacity = default_capacity
        self.default_refill = default_refill
        self.buckets: dict[str, RateLimitBucket] = {}

    def get_bucket(self, key: str) -> RateLimitBucket:
        if key not in self.buckets:
            self.buckets[key] = RateLimitBucket(self.default_capacity, self.default_refill)
        return self.buckets[key]

    def check(self, key: str, n: int = 1) -> bool:
        return self.get_bucket(key).consume(n)


class GuardrailManager:
    """Central guardrails for the agent."""

    def __init__(self, settings: Settings, generator: Generator, judge: Judge):
        self.settings = settings
        self.generator = generator
        self.judge = judge
        self.rate_limiter = RateLimiter(
            default_capacity=settings.rate_limit_requests
            if hasattr(settings, "rate_limit_requests")
            else 100,
            default_refill=settings.rate_limit_refill
            if hasattr(settings, "rate_limit_refill")
            else 10.0,
        )

    def check_rate_limit(self, user_id: str = "default") -> tuple[bool, str | None]:
        """Check if request is allowed. Returns (allowed, error_message)."""
        if not self.rate_limiter.check(user_id):
            return False, f"Rate limit exceeded for {user_id}. Please wait."
        return True, None

    def validate_input(self, text: str) -> tuple[bool, str | None]:
        """Basic input sanitization."""
        if (
            len(text) > self.settings.max_input_chars
            if hasattr(self.settings, "max_input_chars")
            else 5000
        ):
            return False, "Input too long."
        # Block obvious injection patterns
        suspicious = ["<script", "javascript:", "data:", "eval(", "exec("]
        if any(s in text.lower() for s in suspicious):
            return False, "Suspicious input detected."
        return True, None

    async def check_faithfulness(
        self, question: str, answer: str, context: str
    ) -> tuple[bool, float, str]:
        """
        Use LLM judge to check if answer is grounded in context.
        Returns (passed, score, reason).
        """
        if not self.judge.available():
            return True, 1.0, "Judge unavailable, skipping faithfulness check"

        verdict = self.judge.rate_grounded(question, answer, context)
        faithfulness = verdict.get("faithfulness", 5)
        reason = verdict.get("faithfulness_reason", "")

        threshold = getattr(self.settings, "faithfulness_threshold", 3)
        passed = faithfulness >= threshold
        return passed, float(faithfulness), reason

    async def validate_output(
        self, question: str, answer: str, context: str
    ) -> tuple[bool, str | None]:
        """Run all output validations."""
        # 1. Faithfulness
        passed, score, reason = await self.check_faithfulness(question, answer, context)
        if not passed:
            return False, f"Faithfulness check failed (score={score}): {reason}"

        # 2. Required citations (if context was provided)
        if context and "[1]" not in answer and "[2]" not in answer:
            return False, "Answer missing citations."

        # 3. No refusal when context has answer
        refusal_phrases = ["i don't know", "i cannot", "unable to", "context does not"]
        if context and any(p in answer.lower() for p in refusal_phrases):
            # Could be legitimate if context truly lacks answer
            # For now, just warn
            pass

        return True, None


def build_guardrails(settings: Settings, generator: Generator) -> GuardrailManager:
    from app.generate import Judge

    judge = Judge(settings)
    return GuardrailManager(settings, generator, judge)
