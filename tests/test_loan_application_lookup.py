"""Tests for independent loan application lookup and escalation scoring."""

from typing import Dict

import pytest

from data.dataset import LOAN_APPLICATIONS, get_loan_application_by_id
from src.cred_agent.services import loan_application_lookup
from src.cred_agent.services.loan_application_lookup import (
    ESCALATION_THRESHOLD,
    check_loan_application_status,
)


def _record(flagged: bool, days: int) -> Dict[str, object]:
    return {
        "record_id": "APP-TEST",
        "category": "Personal Loan",
        "status": "Under Review",
        "loan_amount_inr": 250_000,
        "days_since_created": days,
        "flagged_for_fraud_review": flagged,
    }


def test_existing_flagged_application_returns_dataset_fields_and_escalates():
    application = next(
        record for record in LOAN_APPLICATIONS
        if record["flagged_for_fraud_review"]
    )

    result = check_loan_application_status(application["record_id"])
    expected_score = round(
        0.70 + 0.30 * (30 - application["days_since_created"]) / 30,
        4,
    )

    assert result["found"]
    assert result["record_id"] == application["record_id"]
    assert result["category"] == application["category"]
    assert result["status"] == application["status"]
    assert result["loan_amount_inr"] == application["loan_amount_inr"]
    assert result["days_since_created"] == application["days_since_created"]
    assert result["flagged_for_fraud_review"] is True
    assert result["escalation_score"] == expected_score
    assert result["escalation_required"] is True
    assert "Fraud review is flagged" in result["escalation_reason"]


def test_existing_non_flagged_application_is_looked_up_without_fraud_escalation():
    application = next(
        record for record in LOAN_APPLICATIONS
        if not record["flagged_for_fraud_review"]
    )

    result = check_loan_application_status(application["record_id"])

    assert result["found"]
    assert result["flagged_for_fraud_review"] is False
    assert result["escalation_score"] < ESCALATION_THRESHOLD
    assert result["escalation_required"] is False
    assert "No fraud review flag" in result["escalation_reason"]


def test_recent_application_has_higher_recency_component_than_old_application():
    recent = next(
        record for record in LOAN_APPLICATIONS
        if not record["flagged_for_fraud_review"]
        and record["days_since_created"] == min(
            candidate["days_since_created"]
            for candidate in LOAN_APPLICATIONS
            if not candidate["flagged_for_fraud_review"]
        )
    )
    old = next(
        record for record in LOAN_APPLICATIONS
        if not record["flagged_for_fraud_review"]
        and record["days_since_created"] == max(
            candidate["days_since_created"]
            for candidate in LOAN_APPLICATIONS
            if not candidate["flagged_for_fraud_review"]
        )
    )

    recent_result = check_loan_application_status(recent["record_id"])
    old_result = check_loan_application_status(old["record_id"])

    assert recent_result["days_since_created"] < old_result["days_since_created"]
    assert recent_result["escalation_score"] > old_result["escalation_score"]


@pytest.mark.parametrize(
    ("flagged", "days", "expected"),
    [(True, 0, 1.0), (False, 30, 0.0)],
)
def test_maximum_and_minimum_score_boundaries(
    monkeypatch: pytest.MonkeyPatch,
    flagged: bool,
    days: int,
    expected: float,
):
    monkeypatch.setattr(
        loan_application_lookup,
        "get_loan_application_by_id",
        lambda record_id: _record(flagged, days),
    )

    result = check_loan_application_status("APP-TEST")

    assert result["escalation_score"] == expected
    assert 0.0 <= result["escalation_score"] <= 1.0
    assert result["escalation_required"] is (expected >= ESCALATION_THRESHOLD)


@pytest.mark.parametrize("record_id", ["", "  ", None, 123])
def test_invalid_record_id_returns_explicit_error(record_id):
    result = check_loan_application_status(record_id)  # type: ignore[arg-type]

    assert result["found"] is False
    assert result["error"] == "invalid_record_id"
    assert result["message"] == "record_id must be a non-empty string."
    assert "category" not in result


def test_lookup_normalizes_record_id_like_dataset_helper():
    application = LOAN_APPLICATIONS[0]
    result = check_loan_application_status(f"  {application['record_id'].lower()} ")

    assert result["found"] is True
    assert result["record_id"] == application["record_id"]
    assert result["record_id"] == get_loan_application_by_id(
        application["record_id"]
    )["record_id"]


def test_unknown_record_returns_predictable_not_found_result():
    result = check_loan_application_status("APP-DOES-NOT-EXIST")

    assert result == {
        "found": False,
        "record_id": "APP-DOES-NOT-EXIST",
        "error": "not_found",
        "message": (
            "No loan application was found for record_id 'APP-DOES-NOT-EXIST'."
        ),
    }


def test_scores_for_entire_generated_dataset_stay_within_bounds():
    results = [
        check_loan_application_status(application["record_id"])
        for application in LOAN_APPLICATIONS
    ]

    assert all(
        0.0 <= result["escalation_score"] <= 1.0
        for result in results
    )


@pytest.mark.parametrize("invalid_days", [-1, 31, 1.5, True, None])
def test_invalid_dataset_recency_is_rejected_safely(
    monkeypatch: pytest.MonkeyPatch,
    invalid_days,
):
    monkeypatch.setattr(
        loan_application_lookup,
        "get_loan_application_by_id",
        lambda record_id: _record(False, invalid_days),
    )

    with pytest.raises(ValueError, match="expected an integer from 0 to 30"):
        check_loan_application_status("APP-TEST")
