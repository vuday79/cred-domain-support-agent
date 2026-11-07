"""Deterministic customer-facing composition of verified specialist findings."""

import json
from typing import Any, Dict, List, Optional

from src.cred_agent.rag.retrieval import FALLBACK_RESPONSE
from src.cred_agent.schemas import (
    LoanApplicationDetails,
    RetrievedContext,
    SimilarityMeasurement,
    SupportResponse,
    SupportResponseType,
)


def compose_specialist_findings(prompt: str) -> Optional[str]:
    """Compose only recognized structured specialist results from a task prompt.

    CrewAI includes task instructions and expected-output text in its user
    message. The composer therefore requires an explicit ``Specialist findings:``
    section and decodes the JSON value following that marker; it never treats
    surrounding instructions as evidence.
    """
    marker = "Specialist findings:"
    marker_index = prompt.lower().find(marker.lower())
    if marker_index < 0:
        return None

    findings_text = prompt[marker_index + len(marker) :]
    parsed = _decode_json_value(findings_text)
    if parsed is None:
        return None

    results = _result_mappings(parsed)
    composed: List[str] = []
    for result in results:
        if isinstance(result.get("is_grounded"), bool):
            composed_result = _compose_retrieval_result(result)
        elif isinstance(result.get("found"), bool):
            composed_result = _compose_lookup_result(result)
        else:
            continue
        if composed_result and composed_result not in composed:
            composed.append(composed_result)

    if not composed:
        return None
    return "\n\n".join(composed)


def compose_structured_specialist_findings(prompt: str) -> SupportResponse:
    """Build the response schema from verified retrieval and lookup outputs."""
    marker = "Specialist findings:"
    marker_index = prompt.lower().find(marker.lower())
    if marker_index < 0:
        raise ValueError("Specialist findings were not present in the prompt.")

    parsed = _decode_json_value(prompt[marker_index + len(marker) :])
    if parsed is None:
        raise ValueError("Specialist findings were not valid JSON.")

    results = _result_mappings(parsed)
    if not results:
        raise ValueError("Specialist findings contain no recognized tool results.")

    composed_response = compose_specialist_findings(prompt)
    if not composed_response:
        raise ValueError("Specialist findings could not be composed.")

    retrieval_results = [
        result for result in results if isinstance(result.get("is_grounded"), bool)
    ]
    lookup_results = [
        result for result in results if isinstance(result.get("found"), bool)
    ]
    sources = list(
        dict.fromkeys(
            source
            for result in retrieval_results
            if result.get("is_grounded")
            for source in result.get("source_documents", [])
            if isinstance(source, str) and source
        )
    )
    chunks: List[RetrievedContext] = []
    measurements: List[SimilarityMeasurement] = []
    for result in retrieval_results:
        for chunk in result.get("retrieved_chunks", []):
            if not isinstance(chunk, dict):
                continue
            context_fields = {
                key: chunk.get(key)
                for key in (
                    "chunk_id",
                    "parent_document_id",
                    "source",
                    "text",
                    "cosine_similarity",
                )
            }
            chunks.append(RetrievedContext.model_validate(context_fields))
            distance = chunk.get("distance")
            if distance is None:
                distance = 1.0 - float(chunk["cosine_similarity"])
            measurements.append(
                SimilarityMeasurement(
                    chunk_id=chunk["chunk_id"],
                    cosine_similarity=chunk["cosine_similarity"],
                    distance=distance,
                )
            )

    loan_application: Optional[LoanApplicationDetails] = None
    record_id: Optional[str] = None
    for result in lookup_results:
        record_id = result.get("record_id")
        if result.get("found"):
            loan_application = LoanApplicationDetails.model_validate(
                {
                    field: result.get(field)
                    for field in (
                        "record_id",
                        "category",
                        "status",
                        "loan_amount_inr",
                        "days_since_created",
                        "flagged_for_fraud_review",
                        "escalation_score",
                        "escalation_required",
                        "escalation_reason",
                    )
                }
            )
            record_id = loan_application.record_id
            break

    if retrieval_results and lookup_results:
        response_type = SupportResponseType.COMBINED
        grounded = all(result.get("is_grounded", False) for result in retrieval_results)
        grounded = grounded and all(result.get("found", False) for result in lookup_results)
        if any(not result.get("found") for result in lookup_results):
            grounded = True
    elif retrieval_results:
        grounded = bool(retrieval_results[0].get("is_grounded"))
        response_type = (
            SupportResponseType.KNOWLEDGE_BASE
            if grounded
            else SupportResponseType.KNOWLEDGE_BASE_FALLBACK
        )
    elif lookup_results and lookup_results[0].get("found"):
        response_type = SupportResponseType.LOAN_APPLICATION
        grounded = True
    elif lookup_results:
        response_type = SupportResponseType.APPLICATION_NOT_FOUND
        grounded = True
    else:
        raise ValueError("No supported specialist result was provided.")

    return SupportResponse(
        response=composed_response,
        response_type=response_type,
        is_grounded=grounded,
        source_documents=sources,
        retrieved_context=chunks,
        similarity_measurements=measurements,
        record_id=record_id,
        loan_application=loan_application,
    )


def _decode_json_value(text: str) -> Any:
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character not in "{[":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        return value
    return None


def _result_mappings(value: Any) -> List[Dict[str, Any]]:
    if isinstance(value, list):
        mappings: List[Dict[str, Any]] = []
        for item in value:
            mappings.extend(_result_mappings(item))
        return mappings
    if not isinstance(value, dict):
        return []

    if isinstance(value.get("is_grounded"), bool) or isinstance(
        value.get("found"), bool
    ):
        return [value]

    nested_results: List[Dict[str, Any]] = []
    for item in value.values():
        if isinstance(item, (dict, list)):
            nested_results.extend(_result_mappings(item))
        elif isinstance(item, str):
            nested = _decode_json_value(item)
            if nested is not None:
                nested_results.extend(_result_mappings(nested))
    return nested_results


def _compose_retrieval_result(result: Dict[str, Any]) -> str:
    answer = result.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        return FALLBACK_RESPONSE
    if not result.get("is_grounded") or answer.strip() == FALLBACK_RESPONSE:
        return FALLBACK_RESPONSE

    sources: List[str] = []
    source_documents = result.get("source_documents")
    if isinstance(source_documents, list):
        sources.extend(
            source for source in source_documents if isinstance(source, str) and source
        )
    chunks = result.get("retrieved_chunks")
    if isinstance(chunks, list):
        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue
            parent_id = chunk.get("parent_document_id")
            source = chunk.get("source")
            if (
                isinstance(parent_id, str)
                and parent_id in sources
                and isinstance(source, str)
                and source
                and source not in sources
            ):
                filename = source.replace("\\", "/").rsplit("/", 1)[-1]
                sources[sources.index(parent_id)] = f"{parent_id} ({filename})"
    response = answer.strip()
    if sources:
        response += "\nSource: " + "; ".join(dict.fromkeys(sources)) + "."
    return response


def _compose_lookup_result(result: Dict[str, Any]) -> str:
    record_id = result.get("record_id")
    if not isinstance(record_id, str) or not record_id:
        return "I couldn't verify the application ID from the lookup result."

    if not result.get("found"):
        message = result.get("message")
        if isinstance(message, str) and message.strip():
            return message.strip()
        return f"No loan application was found for record_id '{record_id}'."

    parts = [f"Application {record_id}"]
    category = result.get("category")
    status = result.get("status")
    if isinstance(category, str) and category:
        parts.append(f"Category: {category}")
    if isinstance(status, str) and status:
        parts.append(f"Status: {status}")
    response = ". ".join(parts) + "."

    amount = result.get("loan_amount_inr")
    if isinstance(amount, (int, float)) and not isinstance(amount, bool):
        response += f" Loan amount: INR {amount:,.0f}."
    days_since_created = result.get("days_since_created")
    if isinstance(days_since_created, int) and not isinstance(
        days_since_created, bool
    ):
        response += f" Application age: {days_since_created} days."

    flagged = result.get("flagged_for_fraud_review")
    if isinstance(flagged, bool):
        response += f" Fraud review flagged: {'Yes' if flagged else 'No'}."

    escalation_score = result.get("escalation_score")
    escalation_required = result.get("escalation_required")
    if isinstance(escalation_score, (int, float)) and not isinstance(
        escalation_score, bool
    ):
        response += f" Escalation score: {escalation_score:.4f}."
    if isinstance(escalation_required, bool):
        response += (
            " Escalation is required."
            if escalation_required
            else " Escalation is not required."
        )
        if escalation_required:
            response += (
                " The lookup indicates that escalation is required; it does not "
                "confirm that escalation has been completed."
            )
    reason = result.get("escalation_reason")
    if isinstance(reason, str) and reason.strip():
        response += f" Reason: {reason.strip()}"
    return response
