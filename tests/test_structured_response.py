"""Tests for schema validation of structured Crew responses."""

import pytest
from pydantic import ValidationError

from src.cred_agent.crew import validate_composer_response
from src.cred_agent.mock_llm import CredMockLLM
from src.cred_agent.schemas import (
    LoanApplicationDetails,
    LoanApplicationStatus,
    RetrievedContext,
    SimilarityMeasurement,
    SupportResponse,
    SupportResponseType,
    safe_support_response,
    validate_support_response,
)


def _valid_knowledge_response() -> dict:
    return {
        "response": "EMIs use the reducing balance method.\nSource: emi_calculation.",
        "response_type": "knowledge_base",
        "is_grounded": True,
        "source_documents": ["emi_calculation"],
        "retrieved_context": [
            {
                "chunk_id": "fixed_emi_calculation_0",
                "parent_document_id": "emi_calculation",
                "source": "emi_calculation.md",
                "text": "EMIs use the reducing balance method.",
                "cosine_similarity": 0.8,
            }
        ],
        "similarity_measurements": [
            {
                "chunk_id": "fixed_emi_calculation_0",
                "cosine_similarity": 0.8,
                "distance": 0.2,
            }
        ],
    }


def test_valid_structured_response_passes_and_preserves_nested_context():
    response = validate_support_response(_valid_knowledge_response())

    assert isinstance(response, SupportResponse)
    assert response.response_type is SupportResponseType.KNOWLEDGE_BASE
    assert isinstance(response.retrieved_context[0], RetrievedContext)
    assert isinstance(response.similarity_measurements[0], SimilarityMeasurement)
    assert response.source_documents == ["emi_calculation"]


def test_missing_required_fields_are_rejected():
    with pytest.raises(ValidationError):
        SupportResponse.model_validate({"response_type": "knowledge_base"})


@pytest.mark.parametrize(
    "field,value",
    [
        ("response", 42),
        ("is_grounded", "not-a-boolean"),
        ("source_documents", "emi_calculation"),
        ("retrieved_context", [{"chunk_id": "chunk", "cosine_similarity": "high"}]),
    ],
)
def test_invalid_field_types_are_rejected(field, value):
    payload = _valid_knowledge_response()
    payload[field] = value

    with pytest.raises(ValidationError):
        SupportResponse.model_validate(payload)


def test_invalid_application_status_enum_is_rejected():
    with pytest.raises(ValidationError):
        LoanApplicationDetails.model_validate(
            {
                "record_id": "APP-1001",
                "category": "Personal Loan",
                "status": "Pending Review",
                "loan_amount_inr": 750000,
                "days_since_created": 7,
                "flagged_for_fraud_review": False,
                "escalation_score": 0.23,
                "escalation_required": False,
                "escalation_reason": "No escalation required.",
            }
        )

    assert LoanApplicationStatus.SUBMITTED.value == "Submitted"


def test_application_response_retains_all_lookup_and_escalation_fields():
    payload = {
        "response": "Application APP-1001 status: Submitted.",
        "response_type": "loan_application",
        "is_grounded": True,
        "source_documents": [],
        "retrieved_context": [],
        "similarity_measurements": [],
        "record_id": "APP-1001",
        "loan_application": {
            "record_id": "APP-1001",
            "category": "Personal Loan",
            "status": "Submitted",
            "loan_amount_inr": 750000,
            "days_since_created": 7,
            "flagged_for_fraud_review": False,
            "escalation_score": 0.23,
            "escalation_required": False,
            "escalation_reason": "No fraud review flag.",
        },
    }

    response = SupportResponse.model_validate(payload)

    assert response.loan_application is not None
    assert response.loan_application.status is LoanApplicationStatus.SUBMITTED
    assert response.loan_application.escalation_score == 0.23
    assert response.loan_application.escalation_required is False
    assert response.loan_application.escalation_reason == "No fraud review flag."


def test_malformed_mock_llm_findings_return_schema_valid_safe_response():
    response = CredMockLLM().call(
        messages=[{"role": "user", "content": "not structured specialist findings"}],
        response_model=SupportResponse,
    )

    assert isinstance(response, SupportResponse)
    assert response == safe_support_response()
    assert response.response_type is SupportResponseType.SAFE_ERROR
    assert not response.is_grounded


def test_invalid_composer_output_is_replaced_with_safe_schema_response():
    response, error = validate_composer_response(
        {
            "response_type": "loan_application",
            "is_grounded": True,
            "source_documents": [],
            "retrieved_context": [],
            "similarity_measurements": [],
        }
    )

    assert error
    assert isinstance(response, SupportResponse)
    assert response == safe_support_response()
    assert response.response_type is SupportResponseType.SAFE_ERROR
    assert response.response
