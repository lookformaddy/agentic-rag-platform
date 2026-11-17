import pytest

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from ragagent.demo import build_engine
from ragagent.service import create_app


@pytest.fixture()
def client():
    return TestClient(create_app(build_engine()))


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert resp.json()["chunks"] > 0


def test_ask_endpoint_answers(client):
    resp = client.post("/ask", json={"question": "What is the return window?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "answered"
    assert "30" in body["text"]
    assert body["citations"]


def test_ask_endpoint_abstains(client):
    resp = client.post(
        "/ask", json={"question": "What is the quarterly revenue of "
                                 "Northwind Outfitters?"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "abstained"


def test_ingest_and_policies_endpoints(client):
    resp = client.post("/ingest", json={
        "doc_id": "test-doc", "title": "Test",
        "text": "The test widget costs $9.99 and ships free."})
    assert resp.status_code == 200
    assert resp.json()["chunks"] >= 1
    resp = client.get("/policies")
    assert resp.status_code == 200
    assert any(p["version"] == "v1.0.0" for p in resp.json()["policies"])


def test_eval_endpoint(client):
    resp = client.post("/eval")
    assert resp.status_code == 200
    body = resp.json()
    assert body["n_cases"] == 10
    assert body["mean_recall_at_k"] >= 0.8
    assert body["abstention_correctness"] == 1.0


def test_ledger_and_audit_endpoints(client):
    client.post("/ask", json={"question": "What is the return window?",
                              "tenant": "team-a"})
    resp = client.get("/ledger/team-a")
    assert resp.status_code == 200
    assert resp.json()["requests"] >= 1
    resp = client.get("/audit/verify")
    assert resp.status_code == 200
    assert resp.json()["valid"] is True
