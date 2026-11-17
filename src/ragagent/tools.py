"""Schema-validated tool registry (scripted tools, no network).

Three tools ship with the platform:

- ``calculator`` — safe arithmetic over an expression string (AST-eval,
  numbers and + - * / % ** only; no names, calls or attribute access).
- ``date_lookup`` — deterministic date arithmetic against the configured
  reference date (``today`` / day-of-week / days-until / days-between).
- ``definition_lookup`` — glossary lookup over the platform's synthetic
  term dictionary.

Every call is validated against a JSON-schema subset *before* execution
(fail-closed), and every outcome is recorded with cost and latency.
"""
from __future__ import annotations

import ast
import operator
import re
from datetime import date, timedelta

from ragagent.models import ToolCallResult, ToolDefinition

_ALLOWED_BINOPS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
}
_ALLOWED_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _eval_expr(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_expr(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        return _ALLOWED_BINOPS[type(node.op)](
            _eval_expr(node.left), _eval_expr(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY:
        return _ALLOWED_UNARY[type(node.op)](_eval_expr(node.operand))
    raise ValueError("unsupported expression element")


def _validate(args: dict, schema: dict) -> str | None:
    """Validate ``args`` against a small JSON-schema subset. None = ok."""
    if not isinstance(args, dict):
        return "args must be an object"
    props = schema.get("properties", {})
    for req in schema.get("required", []):
        if req not in args:
            return f"missing required argument: {req}"
    if schema.get("additionalProperties") is False:
        for key in args:
            if key not in props:
                return f"unexpected argument: {key}"
    for key, value in args.items():
        spec = props.get(key)
        if spec is None:
            continue
        if spec.get("type") == "string" and not isinstance(value, str):
            return f"argument {key} must be a string"
        if spec.get("type") == "number" and not isinstance(value, (int, float)):
            return f"argument {key} must be a number"
        if isinstance(value, str):
            if "minLength" in spec and len(value) < spec["minLength"]:
                return f"argument {key} is too short"
            if "maxLength" in spec and len(value) > spec["maxLength"]:
                return f"argument {key} is too long"
            if "enum" in spec and value not in spec["enum"]:
                return f"argument {key} must be one of {spec['enum']}"
            if "pattern" in spec and not re.search(spec["pattern"], value):
                return f"argument {key} does not match {spec['pattern']}"
    return None


class ToolRegistry:
    """Versioned tool registry with fail-closed argument validation."""

    def __init__(self, reference_date: str = "2026-11-17",
                 glossary: dict[str, str] | None = None):
        self.reference_date = reference_date
        self.glossary = {k.lower(): v for k, v in (glossary or {}).items()}
        self._tools: dict[str, ToolDefinition] = {}
        self._handlers: dict[str, object] = {}
        self.register(ToolDefinition(
            name="calculator",
            description="Evaluate a deterministic arithmetic expression.",
            args_schema={
                "type": "object",
                "properties": {"expression": {
                    "type": "string", "minLength": 1, "maxLength": 256}},
                "required": ["expression"], "additionalProperties": False,
            }), self._calculator)
        self.register(ToolDefinition(
            name="date_lookup",
            description="Date facts relative to the reference date.",
            args_schema={
                "type": "object",
                "properties": {"query": {
                    "type": "string", "minLength": 1, "maxLength": 256}},
                "required": ["query"], "additionalProperties": False,
            }), self._date_lookup)
        self.register(ToolDefinition(
            name="definition_lookup",
            description="Look up a term in the platform glossary.",
            args_schema={
                "type": "object",
                "properties": {"term": {
                    "type": "string", "minLength": 1, "maxLength": 128}},
                "required": ["term"], "additionalProperties": False,
            }), self._definition_lookup)

    def register(self, definition: ToolDefinition, handler) -> None:
        self._tools[definition.name] = definition
        self._handlers[definition.name] = handler

    def list(self) -> list[ToolDefinition]:
        return sorted(self._tools.values(), key=lambda t: t.name)

    def get(self, name: str) -> ToolDefinition | None:
        return self._tools.get(name)

    def call(self, name: str, args: dict) -> ToolCallResult:
        """Validate, then execute. Unknown tools and bad args fail closed."""
        definition = self._tools.get(name)
        if definition is None:
            return ToolCallResult(tool=name, args=args, ok=False,
                                 error=f"unknown tool: {name}")
        error = _validate(args, definition.args_schema)
        if error:
            return ToolCallResult(tool=name, args=args, ok=False, error=error)
        try:
            result = self._handlers[name](args)
        except Exception as exc:  # scripted tools must never raise
            return ToolCallResult(tool=name, args=args, ok=False,
                                 error=f"tool error: {exc}")
        return ToolCallResult(tool=name, args=args, ok=True, result=result,
                              latency_ms=definition.latency_ms,
                              cost_usd=definition.cost_usd)

    # -- handlers --------------------------------------------------------
    def _calculator(self, args: dict) -> str:
        expr = args["expression"].strip()
        if len(expr) > 256 or not re.fullmatch(r"[0-9+\-*/%().\s]+", expr):
            raise ValueError("expression contains unsupported characters")
        value = _eval_expr(ast.parse(expr, mode="eval"))
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("result is not a finite number")
        return f"{expr} = {value:g}"

    def _date_lookup(self, args: dict) -> str:
        ref = date.fromisoformat(self.reference_date)
        query = args["query"].strip().lower()
        if query in ("today", "current date", "what is today's date",
                     "what is the date today"):
            return f"Today is {ref.isoformat()} ({ref.strftime('%A')})."
        m = re.match(r"day of week (\d{4}-\d{2}-\d{2})$", query)
        if m:
            d = date.fromisoformat(m.group(1))
            return f"{d.isoformat()} is a {d.strftime('%A')}."
        m = re.match(r"days until (\d{4}-\d{2}-\d{2})$", query)
        if m:
            d = date.fromisoformat(m.group(1))
            return f"There are {(d - ref).days} days from {ref.isoformat()} until {d.isoformat()}."
        m = re.match(r"days between (\d{4}-\d{2}-\d{2}) and (\d{4}-\d{2}-\d{2})$", query)
        if m:
            d1, d2 = date.fromisoformat(m.group(1)), date.fromisoformat(m.group(2))
            return f"There are {abs((d2 - d1).days)} days between {d1.isoformat()} and {d2.isoformat()}."
        raise ValueError(
            "unsupported date query — use 'today', 'day of week YYYY-MM-DD', "
            "'days until YYYY-MM-DD', or 'days between YYYY-MM-DD and YYYY-MM-DD'")

    def _definition_lookup(self, args: dict) -> str:
        term = args["term"].strip().lower()
        definition = self.glossary.get(term)
        if definition is None:
            raise ValueError(f"no glossary entry for term: {term}")
        return f"{term}: {definition}"


def detect_tool_need(sub_query_text: str,
                     glossary: dict[str, str] | None = None):
    """Decide which scripted tool (if any) can fill a sub-query's gap.

    Returns (tool_name, args) or None. Arithmetic needs numbers present;
    date needs a date-shaped query; definition needs a 'what is X' form
    whose term is actually in the glossary (when one is supplied).
    """
    low = sub_query_text.lower()
    if re.search(r"\d", low) and any(
            h in low for h in ("total", "sum", "add", "plus", "difference",
                               "altogether", "combined", "how much")):
        nums = re.findall(r"\$?(\d+(?:\.\d{1,2})?)", sub_query_text)
        if len(nums) >= 2:
            return "calculator", {"expression": " + ".join(nums)}
    dates = re.findall(r"\d{4}-\d{2}-\d{2}", sub_query_text)
    if "days between" in low and len(dates) >= 2:
        return "date_lookup", {"query": f"days between {dates[0]} and {dates[1]}"}
    if "days until" in low and dates:
        return "date_lookup", {"query": f"days until {dates[0]}"}
    if ("what day" in low or "which day" in low) and dates:
        return "date_lookup", {"query": f"day of week {dates[0]}"}
    if "today" in low and ("date" in low or "day" in low):
        return "date_lookup", {"query": "today"}
    m = re.match(r"what (?:is|does) (.+?) mean\??$", low)
    if m:
        term = m.group(1).strip()
        if glossary is None or term in {k.lower() for k in glossary}:
            return "definition_lookup", {"term": term}
    m = re.match(r"what is (?:a |an |the )?(.+?)\??$", low)
    if m and len(m.group(1).split()) <= 3:
        term = m.group(1).strip()
        if glossary is None or term in {k.lower() for k in glossary}:
            return "definition_lookup", {"term": term}
    return None
