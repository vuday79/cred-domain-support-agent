"""Lookup generated loan applications and calculate a demonstrative escalation score.

The score is an educational capstone rule, not a real credit decision or
recommendation. Fraud review receives 70% of the score because it is the
strongest explicit risk signal in this synthetic dataset. The remaining 30%
is recency: newer applications have larger normalized recency values because
recent operational work is more time-sensitive.

The 0.65 threshold is deliberately between the maximum possible non-flagged
score (0.30) and the minimum flagged score (0.70). Thus recency alone cannot
trigger escalation, while every fraud-flagged application does; recency
increases priority within either group. This boundary is chosen to preserve
the intended dominance of the fraud signal, not to represent a lender policy.
"""

from typing import Any, Dict, Optional

from data.dataset import get_loan_application_by_id

ESCALATION_THRESHOLD = 0.65
MIN_DAYS_SINCE_CREATED = 0
MAX_DAYS_SINCE_CREATED = 30


def _not_found_result(
    record_id: Optional[str],
    error: str,
    message: str,
) -> Dict[str, Any]:
    return {
        "found": False,
        "record_id": record_id,
        "error": error,
        "message": message,
    }


def check_loan_application_status(record_id: str) -> Dict[str, Any]:
    """Look up an existing dataset record and calculate its escalation score.

    The dataset's identifier lookup is case-insensitive and ignores surrounding
    whitespace. Invalid and unknown IDs return explicit results without
    fabricating an application. Invalid ``days_since_created`` values in a
    found dataset record raise ``ValueError`` instead of silently scoring bad
    data.
    """
    if not isinstance(record_id, str) or not record_id.strip():
        return _not_found_result(
            record_id.strip().upper() if isinstance(record_id, str) else None,
            "invalid_record_id",
            "record_id must be a non-empty string.",
        )

    normalized_id = record_id.strip().upper()
    application = get_loan_application_by_id(normalized_id)
    if application is None:
        return _not_found_result(
            normalized_id,
            "not_found",
            f"No loan application was found for record_id '{normalized_id}'.",
        )

    days_since_created: Any = application.get("days_since_created")
    if (
        isinstance(days_since_created, bool)
        or not isinstance(days_since_created, int)
        or not MIN_DAYS_SINCE_CREATED <= days_since_created <= MAX_DAYS_SINCE_CREATED
    ):
        raise ValueError(
            f"Application '{normalized_id}' has invalid days_since_created "
            f"{days_since_created!r}; expected an integer from 0 to 30."
        )

    flagged_for_fraud_review: Any = application.get("flagged_for_fraud_review")
    if not isinstance(flagged_for_fraud_review, bool):
        raise ValueError(
            f"Application '{normalized_id}' has invalid "
            "flagged_for_fraud_review; expected a boolean."
        )

    normalized_recency = (MAX_DAYS_SINCE_CREATED - days_since_created) / 30
    raw_score = (
        (0.70 if flagged_for_fraud_review else 0.0)
        + (0.30 * normalized_recency)
    )
    escalation_score = min(1.0, max(0.0, round(raw_score, 4)))
    escalation_required = escalation_score >= ESCALATION_THRESHOLD

    if flagged_for_fraud_review:
        escalation_reason = (
            "Fraud review is flagged (0.70 weight); "
            f"recency contributes {0.30 * normalized_recency:.4f} "
            f"({days_since_created} days old). Score {escalation_score:.4f} "
            f"{'meets' if escalation_required else 'does not meet'} the "
            f"{ESCALATION_THRESHOLD:.2f} escalation threshold."
        )
    else:
        escalation_reason = (
            "No fraud review flag; only recency contributes "
            f"{0.30 * normalized_recency:.4f} ({days_since_created} days old). "
            f"Score {escalation_score:.4f} "
            f"{'meets' if escalation_required else 'does not meet'} the "
            f"{ESCALATION_THRESHOLD:.2f} escalation threshold."
        )

    return {
        "found": True,
        "record_id": application["record_id"],
        "category": application["category"],
        "status": application["status"],
        "loan_amount_inr": application["loan_amount_inr"],
        "days_since_created": days_since_created,
        "flagged_for_fraud_review": flagged_for_fraud_review,
        "escalation_score": escalation_score,
        "escalation_required": escalation_required,
        "escalation_reason": escalation_reason,
    }
