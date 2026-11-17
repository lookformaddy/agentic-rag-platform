"""Core data models for the Agentic RAG Platform.

All models are plain dataclasses — no pydantic dependency in the core, so
the engine runs on pure stdlib. The FastAPI service layer adds its own
pydantic request/response models on top of these.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class PolicyStatus(str, Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    DEPRECATED = "deprecated"


class AnswerStatus(str, Enum):
    ANSWERED = "answered"
    ABSTAINED = "abstained"
    BLOCKED = "blocked"  # fail-closed budget / policy refusal


@dataclass
class Document:
    """A source document in the corpus (synthetic, generic domain)."""

    doc_id: str
    title: str
    text: str
    source: str = "corpus"
    tenant: str = "default"
    tags: tuple[str, ...] = ()


@dataclass
class Chunk:
    """One retrievable chunk of a document."""

    chunk_id: str
    doc_id: str
    text: str
    source: str
    chunk_index: int = 0


@dataclass
class RetrievalHit:
    chunk: Chunk
    vector_score: float
    keyword_score: float
    score: float  # hybrid combination


@dataclass
class SubQuery:
    """One decomposed sub-question from the planner."""

    text: str
    kind: str = "factual"  # factual | keyword | arithmetic | date | definition
    origin: str = "planner"  # planner | followup | tool


@dataclass
class ToolDefinition:
    name: str
    description: str
    args_schema: dict
    cost_usd: float = 0.0005
    latency_ms: int = 5


@dataclass
class ToolCallResult:
    tool: str
    args: dict
    ok: bool
    result: str = ""
    error: str = ""
    latency_ms: int = 0
    cost_usd: float = 0.0


@dataclass
class Evidence:
    """A piece of evidence backing an answer sentence."""

    source: str  # doc_id or "tool:<name>"
    text: str
    score: float


@dataclass
class Answer:
    request_id: str
    question: str
    status: AnswerStatus
    text: str = ""
    citations: list[Evidence] = field(default_factory=list)
    sub_queries: list[SubQuery] = field(default_factory=list)
    rounds: int = 0
    tool_calls: list[ToolCallResult] = field(default_factory=list)
    sufficiency: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0
    abstain_reason: str = ""
    policy_version: str = ""


@dataclass
class RetrievalPolicy:
    """A versioned retrieval policy (weights, limits, thresholds)."""

    version: str
    name: str
    vector_weight: float = 0.5
    top_k: int = 5
    sufficiency_threshold: float = 0.30
    abstain_threshold: float = 0.25
    min_evidence_score: float = 0.20
    max_rounds: int = 3
    status: PolicyStatus = PolicyStatus.DRAFT
    notes: str = ""


@dataclass
class EvalCase:
    """One golden-set case: question, expected answer keywords, sources."""

    case_id: str
    question: str
    expected_keywords: list[str]
    expected_sources: list[str]
    expect_abstain: bool = False
    tenant: str = "default"


@dataclass
class EvalCaseResult:
    case_id: str
    answer_accuracy: float
    citation_precision: float
    abstention_correct: bool
    retrieval_recall_at_k: float
    abstained: bool
    top_k_sources: list[str] = field(default_factory=list)


@dataclass
class EvalReport:
    policy_version: str
    n_cases: int
    mean_answer_accuracy: float
    mean_citation_precision: float
    abstention_correctness: float
    mean_recall_at_k: float
    cases: list[EvalCaseResult] = field(default_factory=list)


@dataclass
class LedgerEntry:
    request_id: str
    tenant: str
    tokens_in: int
    tokens_out: int
    cost_usd: float
    latency_ms: int
    rounds: int
    tool_calls: int
    status: str


@dataclass
class AuditEntry:
    seq: int
    event: str
    request_id: str
    tenant: str
    detail: dict
    prev_hash: str
    hash: str
