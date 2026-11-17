from ragagent.audit import AuditLog


def test_audit_chain_verifies():
    log = AuditLog()
    log.append("request", "r1", "team-a", {"q": "x"})
    log.append("answer", "r1", "team-a", {"status": "answered"})
    ok, message = log.verify()
    assert ok and "2 entries" in message


def test_audit_detects_tampered_detail():
    log = AuditLog()
    log.append("request", "r1", "team-a", {"q": "x"})
    log.entries[0].detail["q"] = "forged"
    ok, message = log.verify()
    assert not ok and "hash mismatch" in message


def test_audit_detects_broken_link():
    log = AuditLog()
    log.append("request", "r1", "team-a", {})
    log.append("answer", "r1", "team-a", {})
    log.entries[1].prev_hash = "deadbeef"
    ok, message = log.verify()
    assert not ok and "prev_hash mismatch" in message


def test_audit_events_filter():
    log = AuditLog()
    log.append("request", "r1", "team-a", {})
    log.append("tool_call", "r1", "team-a", {"tool": "calculator"})
    assert len(log.events("tool_call")) == 1
    assert len(log.events()) == 2
