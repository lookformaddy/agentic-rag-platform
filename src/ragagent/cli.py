"""CLI for the Agentic RAG Platform.

Usage:
    python -m ragagent.cli ask "What is the return window?"
    python -m ragagent.cli ask "What is the total?" --tenant team-a
    python -m ragagent.cli eval
    python -m ragagent.cli policies
    python -m ragagent.cli demo
"""
from __future__ import annotations

import argparse
import json
import sys

from ragagent.demo import build_engine, run_demo
from ragagent.demo_data import golden_set
from ragagent.eval import evaluate
from ragagent.models import AnswerStatus


def _print_answer(answer) -> None:
    print(f"status: {answer.status.value}")
    if answer.status == AnswerStatus.ANSWERED:
        print(f"answer: {answer.text}")
        print("citations:")
        for c in answer.citations:
            print(f"  - [{c.source}] {c.text}")
    else:
        print(f"reason: {answer.abstain_reason}")
    print(f"sub-queries: {[sq.text for sq in answer.sub_queries]}")
    print(f"rounds: {answer.rounds}  tools: "
          f"{[(t.tool, t.ok) for t in answer.tool_calls]}")
    print(f"sufficiency: {answer.sufficiency:.2f}  "
          f"tokens: {answer.tokens_in}+{answer.tokens_out}  "
          f"cost: ${answer.cost_usd:.6f}  latency: {answer.latency_ms}ms  "
          f"policy: {answer.policy_version}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ragagent",
                                     description="Agentic RAG Platform CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    ask_p = sub.add_parser("ask", help="Ask a question")
    ask_p.add_argument("question")
    ask_p.add_argument("--tenant", default="default")
    ask_p.add_argument("--json", action="store_true")

    sub.add_parser("eval", help="Run the golden-set eval harness")

    sub.add_parser("policies", help="List retrieval policies")

    sub.add_parser("demo", help="Run the six-scenario demo")

    args = parser.parse_args(argv)
    if args.command == "demo":
        results = run_demo()
        return 0 if results.get("all_ok") else 1

    engine = build_engine()
    if args.command == "ask":
        answer = engine.ask(args.question, tenant=args.tenant)
        if args.json:
            print(json.dumps({
                "status": answer.status.value, "text": answer.text,
                "citations": [c.source for c in answer.citations],
                "sub_queries": [sq.text for sq in answer.sub_queries],
                "rounds": answer.rounds,
                "tool_calls": [{"tool": t.tool, "ok": t.ok,
                                "result": t.result} for t in answer.tool_calls],
                "sufficiency": round(answer.sufficiency, 3),
                "cost_usd": answer.cost_usd,
                "latency_ms": answer.latency_ms,
                "abstain_reason": answer.abstain_reason}, indent=2))
        else:
            _print_answer(answer)
        return 0 if answer.status != AnswerStatus.BLOCKED else 2
    if args.command == "eval":
        report = evaluate(engine, golden_set())
        print(f"policy: {report.policy_version}  cases: {report.n_cases}")
        print(f"answer accuracy:      {report.mean_answer_accuracy:.3f}")
        print(f"citation precision:   {report.mean_citation_precision:.3f}")
        print(f"abstention correct:   {report.abstention_correctness:.3f}")
        print(f"recall@k:             {report.mean_recall_at_k:.3f}")
        for case in report.cases:
            print(f"  {case.case_id:22s} acc={case.answer_accuracy:.2f} "
                  f"cit={case.citation_precision:.2f} "
                  f"abst_ok={case.abstention_correct} "
                  f"recall={case.retrieval_recall_at_k:.2f}")
        return 0
    if args.command == "policies":
        for policy in engine.policies.list():
            print(f"{policy.version:8s} {policy.name:14s} "
                  f"w={policy.vector_weight} top_k={policy.top_k} "
                  f"{policy.status.value}")
        return 0
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
