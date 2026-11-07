"""Request-scoped deterministic token and synthetic-cost budgeting."""

import math
import os
import threading
from collections import deque
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional


class RuntimeBudgetExceeded(RuntimeError):
    """Raised when a request would exceed a configured local runtime budget."""

    def __init__(self, limit: str) -> None:
        super().__init__(f"Request exceeded the configured {limit} budget.")
        self.limit = limit


@dataclass(frozen=True)
class RuntimeBudgetLimits:
    """Limits applied independently to each incoming request."""

    max_input_tokens: int = 512
    max_total_tokens: int = 16384
    max_estimated_cost: Decimal = Decimal("0.05")
    synthetic_cost_per_1000_tokens: Decimal = Decimal("0.002")

    @classmethod
    def from_environment(cls) -> "RuntimeBudgetLimits":
        return cls(
            max_input_tokens=int(
                os.environ.get("CRED_REQUEST_MAX_INPUT_TOKENS", "512")
            ),
            max_total_tokens=int(
                os.environ.get("CRED_REQUEST_MAX_TOTAL_TOKENS", "16384")
            ),
            max_estimated_cost=Decimal(
                os.environ.get("CRED_REQUEST_MAX_ESTIMATED_COST", "0.05")
            ),
            synthetic_cost_per_1000_tokens=Decimal(
                os.environ.get("CRED_SYNTHETIC_COST_PER_1000_TOKENS", "0.002")
            ),
        )

    def __post_init__(self) -> None:
        if self.max_input_tokens < 1:
            raise ValueError("max_input_tokens must be positive")
        if self.max_total_tokens < self.max_input_tokens:
            raise ValueError("max_total_tokens must be >= max_input_tokens")
        if self.max_estimated_cost <= 0:
            raise ValueError("max_estimated_cost must be positive")
        if self.synthetic_cost_per_1000_tokens < 0:
            raise ValueError("synthetic cost rate cannot be negative")


class RuntimeBudget:
    """Atomic token accounting shared by all model stages for one request.

    Token estimation is deterministic: one estimated token per four UTF-8
    bytes, rounded up, with at least one token for non-empty text. Synthetic
    cost is an illustrative local estimate, not an API charge or bill.
    """

    def __init__(self, limits: Optional[RuntimeBudgetLimits] = None) -> None:
        self.limits = limits or RuntimeBudgetLimits.from_environment()
        self.input_tokens = 0
        self.total_tokens = 0
        self.estimated_cost = Decimal("0")
        self.outcome = "within_budget"
        self.workflow_started = False
        self._input_registered = False
        self._reserved_call_tokens: deque[int] = deque()
        self._lock = threading.Lock()

    @staticmethod
    def estimate_tokens(text: str) -> int:
        if not text:
            return 0
        return max(1, math.ceil(len(text.encode("utf-8")) / 4))

    def register_user_input(self, text: str) -> int:
        """Check and account the user input once, before workflow execution."""
        estimated = self.estimate_tokens(text)
        with self._lock:
            if self._input_registered:
                return estimated
            self.input_tokens = estimated
            if estimated > self.limits.max_input_tokens:
                self.outcome = "input_limit_exceeded"
                raise RuntimeBudgetExceeded("input token")
            self._apply_charge(estimated)
            self._input_registered = True
            return estimated

    @property
    def input_registered(self) -> bool:
        with self._lock:
            return self._input_registered

    def mark_workflow_started(self) -> None:
        with self._lock:
            self.workflow_started = True

    def raise_if_exceeded(self) -> None:
        """Raise the safe budget exception after a bounded stage completes."""
        with self._lock:
            outcome = self.outcome
        if outcome == "input_limit_exceeded":
            raise RuntimeBudgetExceeded("input token")
        if outcome == "total_token_limit_exceeded":
            raise RuntimeBudgetExceeded("total token")
        if outcome == "estimated_cost_limit_exceeded":
            raise RuntimeBudgetExceeded("synthetic cost")

    def charge_model_call(self, prompt: str, completion: str) -> tuple[int, int]:
        """Charge one local model invocation before any resulting tool dispatch."""
        prompt_tokens = self.estimate_tokens(prompt)
        completion_tokens = self.estimate_tokens(completion)
        with self._lock:
            if not self._input_registered:
                raise RuntimeError("User input must be budgeted before model execution.")
            actual_call_tokens = prompt_tokens + completion_tokens
            if self._reserved_call_tokens:
                reserved_tokens = self._reserved_call_tokens.popleft()
                if actual_call_tokens > reserved_tokens:
                    self._apply_charge(actual_call_tokens - reserved_tokens)
            else:
                self._apply_charge(actual_call_tokens)
        return prompt_tokens, completion_tokens

    def reserve_model_calls(self, call_token_limits: list[int]) -> None:
        """Preflight a bounded multi-agent stage before starting its execution.

        Reserved amounts are conservative estimates for known deterministic
        turns. Later actual calls consume reservations; any overage is charged
        before the model response can trigger tool dispatch or another turn.
        """
        if not call_token_limits or any(tokens < 1 for tokens in call_token_limits):
            raise ValueError("model call reservations must be positive")
        with self._lock:
            if not self._input_registered:
                raise RuntimeError("User input must be budgeted before model execution.")
            proposed_total = self.total_tokens + sum(call_token_limits)
            proposed_cost = (
                Decimal(proposed_total)
                * self.limits.synthetic_cost_per_1000_tokens
                / Decimal(1000)
            )
            if proposed_total > self.limits.max_total_tokens:
                self.outcome = "total_token_limit_exceeded"
                raise RuntimeBudgetExceeded("total token")
            if proposed_cost > self.limits.max_estimated_cost:
                self.outcome = "estimated_cost_limit_exceeded"
                raise RuntimeBudgetExceeded("synthetic cost")
            self.total_tokens = proposed_total
            self.estimated_cost = proposed_cost
            self._reserved_call_tokens.extend(call_token_limits)

    def _apply_charge(self, tokens: int) -> None:
        proposed_total = self.total_tokens + tokens
        proposed_cost = (
            Decimal(proposed_total)
            * self.limits.synthetic_cost_per_1000_tokens
            / Decimal(1000)
        )
        if proposed_total > self.limits.max_total_tokens:
            self.outcome = "total_token_limit_exceeded"
            raise RuntimeBudgetExceeded("total token")
        if proposed_cost > self.limits.max_estimated_cost:
            self.outcome = "estimated_cost_limit_exceeded"
            raise RuntimeBudgetExceeded("synthetic cost")
        self.total_tokens = proposed_total
        self.estimated_cost = proposed_cost

    def snapshot(self) -> dict[str, int | str]:
        """Return safe accounting metadata without request content."""
        with self._lock:
            return {
                "outcome": self.outcome,
                "input_tokens": self.input_tokens,
                "total_tokens": self.total_tokens,
                "synthetic_estimated_cost": format(self.estimated_cost, "f"),
            }
