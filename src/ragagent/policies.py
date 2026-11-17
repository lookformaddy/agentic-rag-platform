"""Versioned retrieval policies with A/B comparison and gated promotion.

Policies are immutable versions: only one is ACTIVE at a time. A candidate
is promoted only if, on the golden set, it clears every gate — recall@k
floor, citation-precision floor, abstention-correctness floor — and does
not regress the active policy on any of them by more than the allowed
margin. A failed promotion leaves the active policy untouched and
records the reasons.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ragagent.models import EvalReport, PolicyStatus, RetrievalPolicy


@dataclass
class PromotionResult:
    promoted: bool
    reasons: list[str] = field(default_factory=list)


class PolicyRegistry:
    """Registry of versioned retrieval policies (one ACTIVE at a time)."""

    def __init__(self, initial: RetrievalPolicy | None = None):
        self._policies: dict[str, RetrievalPolicy] = {}
        if initial is not None:
            self.register(initial)
            self.activate(initial.version)

    def register(self, policy: RetrievalPolicy) -> None:
        if policy.version in self._policies:
            raise ValueError(f"policy version already exists: {policy.version}")
        self._policies[policy.version] = policy

    def get(self, version: str) -> RetrievalPolicy | None:
        return self._policies.get(version)

    def active(self) -> RetrievalPolicy | None:
        for policy in self._policies.values():
            if policy.status == PolicyStatus.ACTIVE:
                return policy
        return None

    def list(self) -> list[RetrievalPolicy]:
        return sorted(self._policies.values(), key=lambda p: p.version)

    def activate(self, version: str) -> RetrievalPolicy:
        policy = self._policies.get(version)
        if policy is None:
            raise ValueError(f"unknown policy version: {version}")
        current = self.active()
        if current is not None:
            current.status = PolicyStatus.DEPRECATED
        policy.status = PolicyStatus.ACTIVE
        return policy


def compare_reports(candidate: EvalReport, baseline: EvalReport) -> dict[str, float]:
    """Per-metric deltas of candidate minus baseline."""
    return {
        "answer_accuracy": candidate.mean_answer_accuracy - baseline.mean_answer_accuracy,
        "citation_precision": candidate.mean_citation_precision - baseline.mean_citation_precision,
        "abstention_correctness": candidate.abstention_correctness - baseline.abstention_correctness,
        "recall_at_k": candidate.mean_recall_at_k - baseline.mean_recall_at_k,
    }


def gated_promote(registry: PolicyRegistry, candidate_version: str,
                  candidate_report: EvalReport,
                  baseline_report: EvalReport | None = None,
                  min_recall_at_k: float = 0.70,
                  min_citation_precision: float = 0.70,
                  min_abstention_correctness: float = 0.80,
                  max_regression: float = 0.05) -> PromotionResult:
    """Promote a candidate only if it clears every gate.

    Gates: absolute floors on recall@k, citation precision and
    abstention correctness; plus a no-regression rule vs the baseline
    report (each metric may drop by at most ``max_regression``).
    """
    reasons: list[str] = []
    if candidate_report.mean_recall_at_k < min_recall_at_k:
        reasons.append(
            f"recall@k {candidate_report.mean_recall_at_k:.3f} below floor {min_recall_at_k}")
    if candidate_report.mean_citation_precision < min_citation_precision:
        reasons.append(
            f"citation precision {candidate_report.mean_citation_precision:.3f} "
            f"below floor {min_citation_precision}")
    if candidate_report.abstention_correctness < min_abstention_correctness:
        reasons.append(
            f"abstention correctness {candidate_report.abstention_correctness:.3f} "
            f"below floor {min_abstention_correctness}")
    if baseline_report is not None:
        deltas = compare_reports(candidate_report, baseline_report)
        for metric, delta in deltas.items():
            if delta < -max_regression:
                reasons.append(
                    f"{metric} regressed by {-delta:.3f} vs baseline "
                    f"(max allowed {max_regression})")
    if reasons:
        return PromotionResult(promoted=False, reasons=reasons)
    registry.activate(candidate_version)
    return PromotionResult(promoted=True, reasons=[
        f"all gates passed: recall@k {candidate_report.mean_recall_at_k:.3f}, "
        f"citation precision {candidate_report.mean_citation_precision:.3f}, "
        f"abstention {candidate_report.abstention_correctness:.3f}"])
