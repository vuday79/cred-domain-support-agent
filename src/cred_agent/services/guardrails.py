"""Deterministic, keyless input and grounded-output guardrails."""

import json
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional, Tuple

from src.cred_agent.rag.retrieval import FALLBACK_RESPONSE
from src.cred_agent.schemas import (
    SAFE_STRUCTURED_RESPONSE,
    SupportResponse,
    SupportResponseType,
    safe_support_response,
)

_BANK_ACCOUNT_PATTERN = re.compile(
    r"(?P<prefix>\b(?:(?:bank\s+)?account|acct|a/c)\s*"
    r"(?:(?:number|no\.?)\s*)?(?:is\s+)?[:#=-]?\s*)"
    r"(?P<number>(?:\d[\s-]?){8,17}\d)(?![\s-]?\d)",
    re.IGNORECASE,
)
_AADHAAR_PATTERN = re.compile(
    r"(?<![A-Z0-9])(?:[2-9]\d{11}|[2-9]\d{3}[\s-]\d{4}[\s-]\d{4})"
    r"(?![A-Z0-9])",
    re.IGNORECASE,
)
_PAN_PATTERN = re.compile(
    r"(?<![A-Z0-9])[A-Z]{5}\d{4}[A-Z](?![A-Z0-9])",
    re.IGNORECASE,
)
_INJECTION_PATTERNS: Tuple[Tuple[str, re.Pattern[str]], ...] = (
    (
        "instruction_override",
        re.compile(
            r"\b(?:ignore|disregard|forget)\s+(?:all\s+)?(?:the\s+)?"
            r"(?:previous|prior|above|earlier)\s+(?:system\s+)?"
            r"(?:instructions?|rules?|directions?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "internal_prompt_disclosure",
        re.compile(
            r"\b(?:reveal|show|print|repeat|disclose|dump)\b.{0,60}\b"
            r"(?:system\s+prompt|hidden\s+prompt|internal\s+instructions?|"
            r"developer\s+message)\b|"
            r"\b(?:system\s+prompt|hidden\s+prompt|internal\s+instructions?)"
            r"\b.{0,60}\b(?:reveal|show|print|repeat|disclose|dump)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "safety_bypass",
        re.compile(
            r"\b(?:bypass|disable|override|evade|ignore)\b.{0,50}\b"
            r"(?:safety|security|guardrails?|restrictions?|policies)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "data_exfiltration",
        re.compile(
            r"\b(?:exfiltrate|dump|export|leak|steal|send)\b.{0,80}\b"
            r"(?:all\s+records?|other\s+users?|other\s+customers?|"
            r"all\s+customers?|entire\s+(?:database|customer\s+records?)|"
            r"personal\s+data\s+of\s+other|everyone['’]s\s+(?:data|records?)|"
            r"records?\s+or\s+other\s+users?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "priority_override",
        re.compile(
            r"\b(?:treat|consider|interpret)\b.{0,50}\b"
            r"(?:user[- ]provided|user|my|this)\s+"
            r"(?:content|input|text)\b.{0,60}\b"
            r"(?:higher[- ]priority|system|developer|trusted)\b|"
            r"\b(?:user[- ]provided|user)\s+(?:content|input|text)\b"
            r".{0,60}\b(?:higher[- ]priority|system)\s+"
            r"(?:instructions?|messages?|prompts?)\b",
            re.IGNORECASE,
        ),
    ),
)
_WORD_PATTERN = re.compile(r"[a-z]+|[0-9]+(?:\.[0-9]+)?", re.IGNORECASE)
_NUMBER_PATTERN = re.compile(r"(?<![A-Za-z])\d[\d,]*(?:\.\d+)?(?![A-Za-z])")
_GROUNDING_STOP_WORDS = {
    "a",
    "about",
    "across",
    "all",
    "an",
    "and",
    "are",
    "as",
    "at",
    "based",
    "be",
    "by",
    "does",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "using",
    "was",
    "were",
    "with",
    "your",
}
_LOOKUP_PRESENTATION_TERMS = (
    "application category status loan amount inr application age days "
    "fraud review flagged yes no escalation score is required not reason "
    "lookup indicates does not confirm that escalation has been completed"
)


@dataclass(frozen=True)
class GuardedInput:
    """Masked user text and non-sensitive guardrail decisions."""

    text: str
    pii_categories: Tuple[str, ...]
    injection_category: Optional[str]

    @property
    def pii_was_masked(self) -> bool:
        return bool(self.pii_categories)

    @property
    def injection_detected(self) -> bool:
        return self.injection_category is not None


def guard_user_input(text: str) -> GuardedInput:
    """Mask fixed-format PII, then identify suspicious instruction attempts."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("message must be a non-empty string")

    categories: List[str] = []
    masked = text.strip()

    def replace_account(match: re.Match[str]) -> str:
        if "bank_account" not in categories:
            categories.append("bank_account")
        return f"{match.group('prefix')}[BANK ACCOUNT REDACTED]"

    masked = _BANK_ACCOUNT_PATTERN.sub(replace_account, masked)

    def replace_aadhaar(_: re.Match[str]) -> str:
        if "aadhaar" not in categories:
            categories.append("aadhaar")
        return "[AADHAAR REDACTED]"

    masked = _AADHAAR_PATTERN.sub(replace_aadhaar, masked)

    def replace_pan(_: re.Match[str]) -> str:
        if "pan" not in categories:
            categories.append("pan")
        return "[PAN REDACTED]"

    masked = _PAN_PATTERN.sub(replace_pan, masked)
    injection_category = detect_prompt_injection(masked)
    return GuardedInput(
        text=masked,
        pii_categories=tuple(categories),
        injection_category=injection_category,
    )


def detect_prompt_injection(text: str) -> Optional[str]:
    """Return a stable category for known injection patterns, without echoing text."""
    for category, pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            return category
    return None


def check_response_groundedness(
    response: SupportResponse,
    specialist_findings: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[bool, Optional[str]]:
    """Conservatively check customer-facing claims against actual evidence text.

    Citation labels and source filenames are removed from the claim text and
    are never counted as supporting evidence.
    """
    response_type = response.response_type
    if response_type is SupportResponseType.SAFE_ERROR:
        return (
            response.response == SAFE_STRUCTURED_RESPONSE.response,
            None
            if response.response == SAFE_STRUCTURED_RESPONSE.response
            else "invalid_safe_error_text",
        )
    if response_type is SupportResponseType.KNOWLEDGE_BASE_FALLBACK:
        return (
            response.response == FALLBACK_RESPONSE,
            None if response.response == FALLBACK_RESPONSE else "invalid_fallback_text",
        )

    if specialist_findings is not None:
        from src.cred_agent.services.response_composition import (
            compose_specialist_findings,
        )

        expected_response = compose_specialist_findings(
            "Specialist findings:\n"
            + json.dumps(specialist_findings, ensure_ascii=False, sort_keys=True)
        )
        if expected_response is None or _normalize_text(
            response.response
        ) != _normalize_text(expected_response):
            return False, "response_mismatch_with_verified_findings"

    evidence_parts = [chunk.text for chunk in response.retrieved_context]
    if response.response_type in {
        SupportResponseType.LOAN_APPLICATION,
        SupportResponseType.APPLICATION_NOT_FOUND,
        SupportResponseType.COMBINED,
    }:
        evidence_parts.append(_LOOKUP_PRESENTATION_TERMS)
        if response.loan_application is not None:
            evidence_parts.append(
                json.dumps(
                    response.loan_application.model_dump(mode="json"),
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
        if response.record_id:
            evidence_parts.append(response.record_id)
        if response.response_type is SupportResponseType.APPLICATION_NOT_FOUND:
            evidence_parts.append("application record not found")

    has_grounded_kb_findings = (
        response.response_type is SupportResponseType.KNOWLEDGE_BASE
        or (
            response.response_type is SupportResponseType.COMBINED
            and bool(response.source_documents)
        )
    )
    if has_grounded_kb_findings:
        if not response.is_grounded or not response.retrieved_context:
            return False, "missing_retrieved_evidence"
        context_documents = {
            chunk.parent_document_id for chunk in response.retrieved_context
        }
        if not set(response.source_documents).issubset(context_documents):
            return False, "source_attribution_mismatch"
        if response.response_type is SupportResponseType.KNOWLEDGE_BASE and (
            not response.source_documents
        ):
            return False, "missing_source_attribution"

    if response.response_type is SupportResponseType.COMBINED and (
        response.loan_application is None and not response.record_id
    ):
        return False, "missing_verified_lookup"
    if (
        response.response_type is SupportResponseType.COMBINED
        and not response.source_documents
        and FALLBACK_RESPONSE not in response.response
    ):
        return False, "missing_source_attribution"

    if response.response_type is SupportResponseType.LOAN_APPLICATION and (
        not response.is_grounded or response.loan_application is None
    ):
        return False, "missing_verified_lookup"
    if response.response_type is SupportResponseType.APPLICATION_NOT_FOUND and (
        not response.is_grounded or not response.record_id
    ):
        return False, "missing_verified_not_found_result"

    claim_text = re.sub(
        r"(?im)^\s*source\s*:\s*.*$",
        "",
        response.response,
    )
    if response.response_type is SupportResponseType.COMBINED:
        claim_text = claim_text.replace(FALLBACK_RESPONSE, "")
    evidence = " ".join(evidence_parts).lower().replace("_", " ")
    evidence_words = set(_WORD_PATTERN.findall(evidence))

    for sentence in re.split(r"(?<=[.!?])\s+", claim_text.strip()):
        words = [
            word.lower()
            for word in _WORD_PATTERN.findall(sentence)
            if word.lower() not in _GROUNDING_STOP_WORDS
            and not _is_numeric_word(word)
        ]
        if not words:
            continue
        matched = sum(word in evidence_words for word in words)
        if matched / len(words) < 0.8:
            return False, "unsupported_factual_claim"

    evidence_numbers = _decimal_values(" ".join(evidence_parts))
    for number in _NUMBER_PATTERN.findall(claim_text):
        if _decimal_value(number) not in evidence_numbers:
            return False, "unsupported_numeric_claim"
    return True, None


def sanitize_response_pii(response: SupportResponse) -> SupportResponse:
    """Return a schema-validated response with fixed-format PII masked."""

    def sanitize(value: Any) -> Any:
        if isinstance(value, str):
            return guard_user_input(value).text
        if isinstance(value, list):
            return [sanitize(item) for item in value]
        if isinstance(value, dict):
            return {key: sanitize(item) for key, item in value.items()}
        return value

    try:
        return SupportResponse.model_validate(
            sanitize(response.model_dump(mode="python"))
        )
    except (TypeError, ValueError):
        return safe_support_response()


def sanitize_nested_strings(value: Any) -> Any:
    """Mask PII recursively in auditable result dictionaries before exposure."""
    if isinstance(value, str):
        return guard_user_input(value).text
    if isinstance(value, list):
        return [sanitize_nested_strings(item) for item in value]
    if isinstance(value, dict):
        return {key: sanitize_nested_strings(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(sanitize_nested_strings(item) for item in value)
    return value


def _decimal_values(text: str) -> set[Decimal]:
    return {
        value
        for token in _NUMBER_PATTERN.findall(text)
        if (value := _decimal_value(token)) is not None
    }


def _decimal_value(token: str) -> Optional[Decimal]:
    try:
        return Decimal(token.replace(",", "")).normalize()
    except InvalidOperation:
        return None


def _is_numeric_word(word: str) -> bool:
    try:
        Decimal(word)
    except InvalidOperation:
        return False
    return True


def _normalize_text(text: str) -> str:
    return " ".join(text.lower().split())
