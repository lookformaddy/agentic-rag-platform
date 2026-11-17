from ragagent.tools import ToolRegistry, detect_tool_need


def test_calculator_evaluates_expression(tools):
    result = tools.call("calculator", {"expression": "129.99 + 159.99"})
    assert result.ok and "289.98" in result.result


def test_calculator_rejects_code_injection(tools):
    result = tools.call("calculator", {"expression": "__import__('os')"})
    assert not result.ok


def test_calculator_rejects_missing_arg(tools):
    result = tools.call("calculator", {})
    assert not result.ok and "missing required" in result.error


def test_calculator_rejects_unknown_tool(tools):
    result = tools.call("shell_run", {"command": "rm -rf /"})
    assert not result.ok and "unknown tool" in result.error


def test_date_lookup_today_and_weekday(tools):
    today = tools.call("date_lookup", {"query": "today"})
    assert today.ok and "2026-11-17" in today.result
    assert "Tuesday" in today.result
    weekday = tools.call("date_lookup", {"query": "day of week 2026-11-17"})
    assert weekday.ok and "Tuesday" in weekday.result


def test_date_lookup_days_between(tools):
    result = tools.call("date_lookup",
                        {"query": "days between 2026-11-17 and 2026-12-25"})
    assert result.ok and "38" in result.result


def test_date_lookup_rejects_freeform(tools):
    result = tools.call("date_lookup", {"query": "when is the heat death?"})
    assert not result.ok


def test_definition_lookup_hit_and_miss(tools):
    hit = tools.call("definition_lookup", {"term": "warranty"})
    assert hit.ok and "1-year" in hit.result
    miss = tools.call("definition_lookup", {"term": "quantum"})
    assert not miss.ok


def test_detect_tool_need_arithmetic():
    assert detect_tool_need(
        "If I buy the trail jacket at $129.99 and the hiking boots at "
        "$159.99, what is the total before tax?") == (
        "calculator", {"expression": "129.99 + 159.99"})


def test_detect_tool_need_date_and_definition():
    assert detect_tool_need("What day of the week is 2026-11-17?") == (
        "date_lookup", {"query": "day of week 2026-11-17"})
    from ragagent.demo_data import GLOSSARY
    assert detect_tool_need("What is a warranty?",
                            glossary=GLOSSARY) == (
        "definition_lookup", {"term": "warranty"})
    # "return window" is itself a glossary term, so it triggers a lookup.
    assert detect_tool_need("What is the return window?",
                            glossary=GLOSSARY) == (
        "definition_lookup", {"term": "return window"})
    # A term outside the glossary must not trigger a tool call.
    assert detect_tool_need("What is the refund policy?",
                            glossary=GLOSSARY) is None


def test_detect_tool_need_returns_none_for_plain_factual():
    assert detect_tool_need("How much does standard shipping cost?") is None
