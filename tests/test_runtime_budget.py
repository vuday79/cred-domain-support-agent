"""Tests for request-scoped deterministic token and synthetic-cost budgets."""

from decimal import Decimal
import json

import pytest
from fastapi.testclient import TestClient

from src.cred_agent.api import PublicSupportResponse, create_app
from src.cred_agent.services.runtime_budget import (
    RuntimeBudget,
    RuntimeBudgetExceeded,
    RuntimeBudgetLimits,
)


def _limits(
    *,
    input_tokens: int = 64,
    total_tokens: int = 2048,
    cost: str = "1",
) -> RuntimeBudgetLimits:
    return RuntimeBudgetLimits(
        max_input_tokens=input_tokens,
        max_total_tokens=total_tokens,
        max_estimated_cost=Decimal(cost),
        synthetic_cost_per_1000_tokens=Decimal("0.01"),
    )


def test_token_estimation_and_synthetic_cost_arithmetic_are_deterministic():
    budget = RuntimeBudget(_limits())

    assert budget.estimate_tokens("abcd") == 1
    assert budget.estimate_tokens("abcde") == 2
    assert budget.estimate_tokens("é") == 1
    assert budget.register_user_input("abcd") == 1
    budget.charge_model_call("abcdefgh", "1234")

    assert budget.snapshot() == {
        "outcome": "within_budget",
        "input_tokens": 1,
        "total_tokens": 4,
        "synthetic_estimated_cost": "0.00004",
    }


def test_input_total_and_synthetic_cost_limits_reject_without_partial_charge():
    too_long = RuntimeBudget(_limits(input_tokens=2))
    with pytest.raises(RuntimeBudgetExceeded, match="input token"):
        too_long.register_user_input("123456789")
    assert too_long.snapshot()["total_tokens"] == 0

    total_limited = RuntimeBudget(_limits(input_tokens=8, total_tokens=8))
    total_limited.register_user_input("1234")
    with pytest.raises(RuntimeBudgetExceeded, match="total token"):
        total_limited.charge_model_call("x" * 40, "")
    assert total_limited.snapshot()["total_tokens"] == 1

    cost_limited = RuntimeBudget(_limits(input_tokens=8, total_tokens=100, cost="0.00002"))
    cost_limited.register_user_input("1234")
    with pytest.raises(RuntimeBudgetExceeded, match="synthetic cost"):
        cost_limited.charge_model_call("x" * 20, "")
    assert cost_limited.snapshot()["total_tokens"] == 1


def test_budgets_are_isolated_per_request():
    first = RuntimeBudget(_limits())
    second = RuntimeBudget(_limits())
    first.register_user_input("first query")
    first.charge_model_call("x" * 80, "done")
    second.register_user_input("second query")

    assert first.snapshot()["total_tokens"] > second.snapshot()["total_tokens"]
    assert second.snapshot()["total_tokens"] == second.snapshot()["input_tokens"]


def test_oversized_api_input_is_rejected_before_session_or_workflow_creation(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("CRED_REQUEST_MAX_INPUT_TOKENS", "8")
    called = False

    def forbidden_workflow(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("Oversized input must not invoke CrewAI.")

    monkeypatch.setattr("src.cred_agent.api.run_customer_query", forbidden_workflow)
    with TestClient(create_app(log_path=tmp_path / "budget.jsonl")) as client:
        response = client.post(
            "/ask",
            json={
                "query": "x" * 40,
                "session_id": "oversized-session",
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body["budget_outcome"] == "input_limit_exceeded"
        assert body["workflow_invoked"] is False
        assert body["history_messages"] == 0
        assert body["structured_response"]["response_type"] == "safe_error"
        PublicSupportResponse.model_validate(body["structured_response"])
        assert client.app.state.sessions == {}
        log_record = json.loads((tmp_path / "budget.jsonl").read_text().splitlines()[-1])
        assert log_record["budget_outcome"] == "input_limit_exceeded"
    assert called is False


def test_pii_is_not_returned_when_oversized_request_is_rejected(tmp_path, monkeypatch):
    pan = "ABCDE1234F"
    monkeypatch.setenv("CRED_REQUEST_MAX_INPUT_TOKENS", "8")
    with TestClient(create_app(log_path=tmp_path / "budget.jsonl")) as client:
        response = client.post(
            "/ask",
            json={
                "query": f"PAN {pan} " + ("x" * 40),
                "session_id": "pii-budget-session",
            },
        )

    assert response.status_code == 200
    assert pan not in response.text
    assert response.json()["input_pii_categories"] == ["pan"]
    assert response.json()["budget_outcome"] == "input_limit_exceeded"
    assert pan not in (tmp_path / "budget.jsonl").read_text()


def test_combined_intent_request_fits_default_budget(tmp_path, monkeypatch):
    monkeypatch.delenv("CRED_REQUEST_MAX_INPUT_TOKENS", raising=False)
    monkeypatch.delenv("CRED_REQUEST_MAX_TOTAL_TOKENS", raising=False)
    monkeypatch.delenv("CRED_REQUEST_MAX_ESTIMATED_COST", raising=False)
    with TestClient(create_app(log_path=tmp_path / "combined.jsonl")) as client:
        response = client.post(
            "/ask",
            json={
                "query": (
                    "What is the status of APP-1001 and how is EMI calculated?"
                ),
                "session_id": "combined-default-budget",
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["budget_outcome"] == "within_budget"
    assert body["structured_response"]["response_type"] == "combined"
    assert body["estimated_total_tokens"] < 16384
    assert float(body["synthetic_estimated_cost"]) <= 0.05


def test_cumulative_crewai_and_autogen_budget_rejection_is_safe_and_not_memorized(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("CRED_REQUEST_MAX_INPUT_TOKENS", "512")
    monkeypatch.setenv("CRED_REQUEST_MAX_TOTAL_TOKENS", "8192")
    monkeypatch.setenv("CRED_REQUEST_MAX_ESTIMATED_COST", "1")
    with TestClient(create_app(log_path=tmp_path / "budget.jsonl")) as client:
        response = client.post(
            "/ask",
            json={
                "query": (
                    "What is the status of APP-1001 and how is EMI calculated?"
                ),
                "session_id": "cumulative-budget-session",
            },
        )
        body = response.json()
        assert response.status_code == 200
        assert body["budget_outcome"] == "total_token_limit_exceeded"
        assert body["workflow_invoked"] is True
        assert body["history_messages"] == 0
        assert body["structured_response"]["response_type"] == "safe_error"
        assert body["estimated_total_tokens"] <= 8192
        assert "APP-1001" not in body["response_text"]
        session = client.app.state.sessions["cumulative-budget-session"]
        assert session.memory.get_messages("cumulative-budget-session") == []
        log_record = json.loads((tmp_path / "budget.jsonl").read_text().splitlines()[-1])
        assert log_record["budget_outcome"] == "total_token_limit_exceeded"
        assert "APP-1001" not in (tmp_path / "budget.jsonl").read_text()


def test_crew_model_budget_rejection_is_safe_without_history_mutation(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("CRED_REQUEST_MAX_INPUT_TOKENS", "512")
    monkeypatch.setenv("CRED_REQUEST_MAX_TOTAL_TOKENS", "512")
    monkeypatch.setenv("CRED_REQUEST_MAX_ESTIMATED_COST", "1")
    with TestClient(create_app(log_path=tmp_path / "crew-budget.jsonl")) as client:
        response = client.post(
            "/ask",
            json={
                "query": "How is EMI calculated?",
                "session_id": "crew-budget-session",
            },
        )

    body = response.json()
    assert response.status_code == 200
    assert body["budget_outcome"] == "total_token_limit_exceeded"
    assert body["workflow_invoked"] is True
    assert body["structured_response"]["response_type"] == "safe_error"
    assert body["history_messages"] == 0
    assert client.app.state.sessions["crew-budget-session"].memory.get_messages(
        "crew-budget-session"
    ) == []
