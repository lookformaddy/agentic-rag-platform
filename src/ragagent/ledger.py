"""Per-request cost / latency / token accounting with fail-closed budgets.

Budgets are checked *before* a request runs: if a tenant has already
spent its cost cap, hit its request cap, or exceeded its token cap, the
request is refused with a ``BudgetExceeded`` error and nothing is
executed. Usage is recorded after each request, successful or not.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ragagent.models import LedgerEntry


class BudgetExceeded(Exception):
    """Raised when a tenant budget would be exceeded (fail-closed)."""


@dataclass
class TenantUsage:
    cost_usd: float = 0.0
    requests: int = 0
    tokens: int = 0


class Ledger:
    """In-memory usage ledger with per-tenant fail-closed caps."""

    def __init__(self, cost_cap_usd: float, request_cap: int,
                 token_cap: int):
        self.cost_cap_usd = cost_cap_usd
        self.request_cap = request_cap
        self.token_cap = token_cap
        self.entries: list[LedgerEntry] = []
        self.usage: dict[str, TenantUsage] = {}

    def check(self, tenant: str, est_cost_usd: float = 0.0,
              est_tokens: int = 0) -> None:
        """Refuse before execution if any cap is already met or would be."""
        used = self.usage.get(tenant, TenantUsage())
        if used.cost_usd + est_cost_usd > self.cost_cap_usd:
            raise BudgetExceeded(
                f"tenant {tenant}: cost cap ${self.cost_cap_usd:.2f} would "
                f"be exceeded (used ${used.cost_usd:.4f})")
        if used.requests + 1 > self.request_cap:
            raise BudgetExceeded(
                f"tenant {tenant}: request cap {self.request_cap} reached")
        if used.tokens + est_tokens > self.token_cap:
            raise BudgetExceeded(
                f"tenant {tenant}: token cap {self.token_cap} would be "
                f"exceeded (used {used.tokens})")

    def record(self, entry: LedgerEntry) -> LedgerEntry:
        used = self.usage.setdefault(entry.tenant, TenantUsage())
        used.cost_usd += entry.cost_usd
        used.requests += 1
        used.tokens += entry.tokens_in + entry.tokens_out
        self.entries.append(entry)
        return entry

    def tenant_usage(self, tenant: str) -> TenantUsage:
        return self.usage.get(tenant, TenantUsage())

    def total_cost(self) -> float:
        return sum(e.cost_usd for e in self.entries)


def estimate_tokens(text: str) -> int:
    """Deterministic token estimate: ~1.3 tokens per word, minimum 1."""
    words = len(text.split())
    return max(1, int(words * 1.3))
