"""Validated structured response models for customer-support Crew output."""

from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class LoanApplicationStatus(str, Enum):
    """Allowed statuses present in the generated loan application dataset."""

    APPROVED = "Approved"
    DISBURSED = "Disbursed"
    REJECTED = "Rejected"
    SUBMITTED = "Submitted"
    UNDER_REVIEW = "Under Review"


class RetrievedContext(BaseModel):
    """A retained, attributed chunk returned by policy retrieval."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(min_length=1)
    parent_document_id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    text: str = Field(min_length=1)
    cosine_similarity: float = Field(ge=-1.0, le=1.0)


class SimilarityMeasurement(BaseModel):
    """Cosine similarity and original Chroma distance for one retrieved chunk."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(min_length=1)
    cosine_similarity: float = Field(ge=-1.0, le=1.0)
    distance: float = Field(ge=0.0, le=2.0)


class LoanApplicationDetails(BaseModel):
    """Verified application fields returned by the authorized lookup tool."""

    model_config = ConfigDict(extra="forbid")

    record_id: str = Field(min_length=1)
    category: str = Field(min_length=1)
    status: LoanApplicationStatus
    loan_amount_inr: int = Field(ge=0)
    days_since_created: int = Field(ge=0)
    flagged_for_fraud_review: bool
    escalation_score: float = Field(ge=0.0, le=1.0)
    escalation_required: bool
    escalation_reason: str


class SupportResponseType(str, Enum):
    """Supported classifications for generated customer-facing responses."""

    KNOWLEDGE_BASE = "knowledge_base"
    KNOWLEDGE_BASE_FALLBACK = "knowledge_base_fallback"
    LOAN_APPLICATION = "loan_application"
    APPLICATION_NOT_FOUND = "application_not_found"
    COMBINED = "combined"
    SAFE_ERROR = "safe_error"


class SupportResponse(BaseModel):
    """Complete, validated result returned by the integrated support Crew."""

    model_config = ConfigDict(extra="forbid")

    response: str = Field(min_length=1)
    response_type: SupportResponseType
    is_grounded: bool
    source_documents: list[str]
    retrieved_context: list[RetrievedContext]
    similarity_measurements: list[SimilarityMeasurement]
    record_id: Optional[str] = None
    loan_application: Optional[LoanApplicationDetails] = None

    @model_validator(mode="after")
    def validate_response_shape(self) -> "SupportResponse":
        if self.response_type is SupportResponseType.KNOWLEDGE_BASE:
            if not self.is_grounded or not self.source_documents:
                raise ValueError(
                    "grounded knowledge-base responses require source documents"
                )
            if self.loan_application is not None or self.record_id is not None:
                raise ValueError(
                    "knowledge-base responses cannot include loan application data"
                )
        elif self.response_type is SupportResponseType.KNOWLEDGE_BASE_FALLBACK:
            if self.is_grounded or self.source_documents:
                raise ValueError(
                    "knowledge-base fallback responses cannot claim grounded sources"
                )
            if self.loan_application is not None or self.record_id is not None:
                raise ValueError(
                    "knowledge-base fallback responses cannot include application data"
                )
        elif self.response_type is SupportResponseType.LOAN_APPLICATION:
            if self.loan_application is None or not self.is_grounded:
                raise ValueError(
                    "loan application responses require verified application details"
                )
            if self.record_id != self.loan_application.record_id:
                raise ValueError("record_id must match the verified application")
        elif self.response_type is SupportResponseType.APPLICATION_NOT_FOUND:
            if not self.record_id or self.loan_application is not None:
                raise ValueError(
                    "not-found responses require a record ID and no application details"
                )
            if not self.is_grounded:
                raise ValueError("authorized not-found results are verified outcomes")
        elif self.response_type is SupportResponseType.COMBINED:
            if not self.is_grounded and not self.loan_application:
                raise ValueError(
                    "combined responses require verified application or policy evidence"
                )
            if self.loan_application and self.record_id != self.loan_application.record_id:
                raise ValueError("record_id must match the verified application")
        elif self.response_type is SupportResponseType.SAFE_ERROR:
            if (
                self.is_grounded
                or self.source_documents
                or self.retrieved_context
                or self.similarity_measurements
                or self.record_id is not None
                or self.loan_application is not None
            ):
                raise ValueError(
                    "safe-error responses cannot claim retrieved or application facts"
                )
        return self


SAFE_STRUCTURED_RESPONSE = SupportResponse(
    response=(
        "I couldn't safely validate a response from the available verified "
        "information. Please try again."
    ),
    response_type=SupportResponseType.SAFE_ERROR,
    is_grounded=False,
    source_documents=[],
    retrieved_context=[],
    similarity_measurements=[],
)


def validate_support_response(value: Any) -> SupportResponse:
    """Strictly validate a Crew response from JSON text, mapping, or model."""
    if isinstance(value, str):
        return SupportResponse.model_validate_json(value)
    return SupportResponse.model_validate(value)


def safe_support_response() -> SupportResponse:
    """Return a fresh safe schema-valid response after output validation fails."""
    return SAFE_STRUCTURED_RESPONSE.model_copy(deep=True)
