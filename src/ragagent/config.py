"""Settings for the Agentic RAG Platform.

Every setting is overridable from the environment (``ARP_`` prefix).
No secrets are needed: retrieval, generation and tools are deterministic
and simulated, so the platform runs fully offline.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass
class Settings:
    """Runtime configuration for the agentic RAG platform."""

    # Retrieval defaults
    embedding_dim: int = field(
        default_factory=lambda: _env_int("ARP_EMBEDDING_DIM", 128))
    default_top_k: int = field(
        default_factory=lambda: _env_int("ARP_DEFAULT_TOP_K", 5))
    default_vector_weight: float = field(
        default_factory=lambda: _env_float("ARP_DEFAULT_VECTOR_WEIGHT", 0.5))
    default_max_rounds: int = field(
        default_factory=lambda: _env_int("ARP_DEFAULT_MAX_ROUNDS", 3))
    default_max_subqueries: int = field(
        default_factory=lambda: _env_int("ARP_DEFAULT_MAX_SUBQUERIES", 4))

    # Sufficiency / abstention thresholds (hybrid scores in [0, 1]).
    # Calibrated against the demo golden set: grounded sub-queries score
    # ~0.33+, out-of-corpus questions ~0.18, so the thresholds sit
    # between the two with margin on both sides.
    default_sufficiency_threshold: float = field(
        default_factory=lambda: _env_float("ARP_SUFFICIENCY_THRESHOLD", 0.30))
    default_abstain_threshold: float = field(
        default_factory=lambda: _env_float("ARP_ABSTAIN_THRESHOLD", 0.25))
    default_min_evidence_score: float = field(
        default_factory=lambda: _env_float("ARP_MIN_EVIDENCE_SCORE", 0.20))

    # Tool defaults
    reference_date: str = field(
        default_factory=lambda: _env_str("ARP_REFERENCE_DATE", "2026-11-17"))
    tool_timeout_ms: int = field(
        default_factory=lambda: _env_int("ARP_TOOL_TIMEOUT_MS", 2_000))

    # Cost model (USD; deterministic, per 1k tokens / per tool call)
    cost_per_1k_tokens: float = field(
        default_factory=lambda: _env_float("ARP_COST_PER_1K_TOKENS", 0.002))
    cost_per_tool_call: float = field(
        default_factory=lambda: _env_float("ARP_COST_PER_TOOL_CALL", 0.0005))
    cost_per_retrieval_round: float = field(
        default_factory=lambda: _env_float("ARP_COST_PER_RETRIEVAL_ROUND", 0.0002))

    # Tenant budgets (fail-closed caps, checked *before* execution)
    tenant_cost_cap_usd: float = field(
        default_factory=lambda: _env_float("ARP_TENANT_COST_CAP_USD", 1.00))
    tenant_request_cap: int = field(
        default_factory=lambda: _env_int("ARP_TENANT_REQUEST_CAP", 1_000))
    tenant_token_cap: int = field(
        default_factory=lambda: _env_int("ARP_TENANT_TOKEN_CAP", 1_000_000))

    seed: int = field(default_factory=lambda: _env_int("ARP_SEED", 42))
    host: str = field(default_factory=lambda: _env_str("ARP_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: _env_int("ARP_PORT", 8000))
