"""Cred Domain Support Agent - Loan Application Dataset Generator.

Track: Banking & FinTech (Cred)
Module: data.dataset

This module produces a deterministic, validated synthetic dataset of loan applications
used by the Cred Domain Support Agent for policy checks, account lookup tools,
and multi-agent escalation workflows.

Design Reasoning for Loan Amount Range (INR 50,000 to INR 50,00,000):
"Loan amounts range from INR 50,000 to INR 50,00,000 (50 Lakhs), capturing the full spectrum
of Cred's prime lending portfolio—from entry-level unsecured personal credit lines up to
high-value collateral-backed home and business loans."
"""

import math
import random
from typing import Any, Dict, List, Optional, Tuple, TypedDict

# ============================================================================
# Domain Vocabularies & Specification Constants
# ============================================================================

# The five required loan categories (Cred Domain Vocabulary)
LOAN_CATEGORIES: Tuple[str, ...] = (
    "Personal Loan",
    "Home Loan",
    "Auto Loan",
    "Education Loan",
    "Business Loan",
)

# The five required loan statuses (Cred Domain Vocabulary)
LOAN_STATUSES: Tuple[str, ...] = (
    "Submitted",
    "Under Review",
    "Approved",
    "Rejected",
    "Disbursed",
)

# Realistic amount boundaries per loan category (all strictly within INR 50k - INR 50L)
CATEGORY_AMOUNT_RANGES: Dict[str, Tuple[int, int]] = {
    "Personal Loan": (50_000, 15_00_000),      # Unsecured personal credit
    "Auto Loan": (1_00_000, 25_00_000),         # Vehicle financing
    "Education Loan": (2_00_000, 35_00_000),    # Higher education domestic & overseas
    "Business Loan": (5_00_000, 50_00_000),     # Merchant & SME working capital
    "Home Loan": (10_00_000, 50_00_000),        # Residential mortgage financing
}

# Core dataset validation thresholds
DEFAULT_NUM_RECORDS: int = 50
INITIAL_SEED: int = 42
MIN_LOAN_AMOUNT_INR: int = 50_000
MAX_LOAN_AMOUNT_INR: int = 5_000_000
MIN_CATEGORY_COUNT: int = 3
MIN_STATUS_COUNT: int = 1
MIN_FRAUD_PCT: float = 10.0
MAX_FRAUD_PCT: float = 30.0
MIN_DAYS_SINCE_CREATED: int = 0
MAX_DAYS_SINCE_CREATED: int = 30
DEFAULT_FRAUD_PROBABILITY: float = 0.20

LOAN_AMOUNT_REASONING: str = (
    "Loan amounts range from INR 50,000 to INR 50,00,000 (50 Lakhs), capturing the full spectrum "
    "of Cred's prime lending portfolio—from entry-level unsecured personal credit lines up to "
    "high-value collateral-backed home and business loans."
)


# ============================================================================
# Type Definitions
# ============================================================================

class LoanApplication(TypedDict):
    """Schema for a single loan application record."""
    record_id: str
    category: str
    status: str
    loan_amount_inr: int
    days_since_created: int
    flagged_for_fraud_review: bool


# ============================================================================
# Generator Implementation
# ============================================================================

def _generate_candidate_records(
    num_records: int,
    seed: int,
    fraud_prob: float,
) -> List[LoanApplication]:
    """Deterministically generate a batch of loan application records.

    Args:
        num_records: Number of records to generate (>= 50).
        seed: Random seed for deterministic reproducibility.
        fraud_prob: Probability weight for setting flagged_for_fraud_review=True.

    Returns:
        List of LoanApplication dictionaries.
    """
    rng = random.Random(seed)
    records: List[LoanApplication] = []

    for idx in range(1, num_records + 1):
        record_id = f"APP-{1000 + idx}"
        category = rng.choice(LOAN_CATEGORIES)
        status = rng.choice(LOAN_STATUSES)

        # Draw loan amount within the category-specific realistic band
        min_amt, max_amt = CATEGORY_AMOUNT_RANGES[category]
        # Align loan amounts to realistic 10,000 INR increments
        amount = rng.randrange(min_amt, max_amt + 1, 10_000)

        # Enforce hard global boundaries
        amount = max(MIN_LOAN_AMOUNT_INR, min(MAX_LOAN_AMOUNT_INR, amount))

        # Recency: integer between 0 and 30 days
        days = rng.randint(MIN_DAYS_SINCE_CREATED, MAX_DAYS_SINCE_CREATED)

        # Fraud review flag
        flagged_for_fraud = rng.random() < fraud_prob

        records.append({
            "record_id": record_id,
            "category": category,
            "status": status,
            "loan_amount_inr": amount,
            "days_since_created": days,
            "flagged_for_fraud_review": flagged_for_fraud,
        })

    return records


# ============================================================================
# Validation Implementation
# ============================================================================

def validate_loan_dataset(
    records: List[LoanApplication],
    min_records: int = DEFAULT_NUM_RECORDS,
) -> Tuple[bool, Dict[str, Any], List[str]]:
    """Validate a loan application dataset against all capstone specification rules.

    Rules checked:
        1. Total records >= min_records (default 50).
        2. Unique record_ids across all records.
        3. All 5 required categories present with >= 3 records each.
        4. All 5 required statuses present with >= 1 record each.
        5. Every loan amount within [50,000, 50,00,000] INR.
        6. Every days_since_created within [0, 30].
        7. Percentage of flagged_for_fraud_review records between 10.0% and 30.0%.

    Args:
        records: List of loan application records.
        min_records: Minimum acceptable record count (default 50).

    Returns:
        A tuple of:
            - is_valid (bool): True if all rules passed.
            - stats (dict): Computed descriptive metrics for the dataset.
            - error_messages (list[str]): List of violation descriptions if invalid.
    """
    errors: List[str] = []
    total_count = len(records)

    # 1. Total record count check
    if total_count < min_records:
        errors.append(f"Record count ({total_count}) is less than required minimum ({min_records}).")

    # 2. Record ID uniqueness check
    record_ids = [r["record_id"] for r in records]
    if len(record_ids) != len(set(record_ids)):
        errors.append("Duplicate record_id values detected.")

    # 3. Category distribution check
    category_counts = {cat: 0 for cat in LOAN_CATEGORIES}
    for r in records:
        cat = r["category"]
        category_counts[cat] = category_counts.get(cat, 0) + 1

    for cat in LOAN_CATEGORIES:
        count = category_counts.get(cat, 0)
        if count < MIN_CATEGORY_COUNT:
            errors.append(
                f"Category '{cat}' has {count} records (minimum required: {MIN_CATEGORY_COUNT})."
            )

    # 4. Status distribution check
    status_counts = {st: 0 for st in LOAN_STATUSES}
    for r in records:
        st = r["status"]
        status_counts[st] = status_counts.get(st, 0) + 1

    for st in LOAN_STATUSES:
        count = status_counts.get(st, 0)
        if count < MIN_STATUS_COUNT:
            errors.append(
                f"Status '{st}' has {count} records (minimum required: {MIN_STATUS_COUNT})."
            )

    # 5. Loan amount boundary check
    amounts = [r["loan_amount_inr"] for r in records]
    min_amount = min(amounts) if amounts else 0
    max_amount = max(amounts) if amounts else 0

    if min_amount < MIN_LOAN_AMOUNT_INR:
        errors.append(
            f"Minimum loan amount INR {min_amount:,} is below allowed INR {MIN_LOAN_AMOUNT_INR:,}."
        )
    if max_amount > MAX_LOAN_AMOUNT_INR:
        errors.append(
            f"Maximum loan amount INR {max_amount:,} exceeds allowed INR {MAX_LOAN_AMOUNT_INR:,}."
        )

    # 6. Days since created boundary check
    days_list = [r["days_since_created"] for r in records]
    min_days = min(days_list) if days_list else 0
    max_days = max(days_list) if days_list else 0

    if min_days < MIN_DAYS_SINCE_CREATED or max_days > MAX_DAYS_SINCE_CREATED:
        errors.append(
            f"Days since created range [{min_days}, {max_days}] violates allowed bounds "
            f"[{MIN_DAYS_SINCE_CREATED}, {MAX_DAYS_SINCE_CREATED}]."
        )

    # 7. Fraud percentage check
    fraud_count = sum(1 for r in records if r["flagged_for_fraud_review"])
    fraud_pct = (fraud_count / total_count * 100.0) if total_count > 0 else 0.0

    if not (MIN_FRAUD_PCT <= fraud_pct <= MAX_FRAUD_PCT):
        errors.append(
            f"Fraud review percentage {fraud_pct:.2f}% is outside the allowed band "
            f"[{MIN_FRAUD_PCT}%, {MAX_FRAUD_PCT}%]."
        )

    # Descriptive statistics aggregation
    sorted_amounts = sorted(amounts) if amounts else [0]
    sorted_days = sorted(days_list) if days_list else [0]

    stats: Dict[str, Any] = {
        "total_records": total_count,
        "category_counts": category_counts,
        "status_counts": status_counts,
        "fraud_count": fraud_count,
        "fraud_percentage": round(fraud_pct, 2),
        "amount_stats": {
            "min": min_amount,
            "max": max_amount,
            "mean": sum(amounts) / total_count if total_count else 0,
            "median": sorted_amounts[total_count // 2] if total_count else 0,
        },
        "days_stats": {
            "min": min_days,
            "max": max_days,
            "mean": sum(days_list) / total_count if total_count else 0,
            "p50": sorted_days[total_count // 2] if total_count else 0,
            "p80": sorted_days[int(total_count * 0.80)] if total_count else 0,
        },
    }

    is_valid = len(errors) == 0
    return is_valid, stats, errors


# ============================================================================
# Deterministic Generation & Auto-Regeneration Loop
# ============================================================================

def generate_loan_dataset(
    num_records: int = DEFAULT_NUM_RECORDS,
    initial_seed: int = INITIAL_SEED,
    initial_fraud_prob: float = DEFAULT_FRAUD_PROBABILITY,
    max_attempts: int = 50,
) -> Tuple[List[LoanApplication], Dict[str, Any]]:
    """Generate and validate a loan application dataset with deterministic regeneration.

    Requirement 10:
    "If the fraud percentage does not satisfy the requirement, change the seed or weights
    deterministically and regenerate the entire dataset. Never manually modify individual records."

    Args:
        num_records: Number of loan applications to generate (>= 50).
        initial_seed: Seed to attempt first (default 42).
        initial_fraud_prob: Initial probability weight for fraud flagging (default 0.20).
        max_attempts: Maximum deterministic attempts before raising an error.

    Returns:
        Tuple of (validated_records, validation_stats).

    Raises:
        RuntimeError: If valid dataset cannot be generated within max_attempts.
    """
    current_seed = initial_seed
    current_fraud_prob = initial_fraud_prob

    for attempt in range(1, max_attempts + 1):
        # 1. Generate full candidate dataset from current deterministic parameters
        candidate = _generate_candidate_records(
            num_records=num_records,
            seed=current_seed,
            fraud_prob=current_fraud_prob,
        )

        # 2. Validate the entire candidate dataset
        is_valid, stats, errors = validate_loan_dataset(candidate, min_records=num_records)

        # 3. If fully compliant, return the generated dataset
        if is_valid:
            stats["generation_metadata"] = {
                "initial_seed": initial_seed,
                "final_seed": current_seed,
                "final_fraud_prob": current_fraud_prob,
                "attempts": attempt,
            }
            return candidate, stats

        # 4. If invalid, adapt deterministically and regenerate ENTIRE dataset
        # Never mutate individual records
        fraud_pct = stats["fraud_percentage"]
        if fraud_pct < MIN_FRAUD_PCT:
            # Deterministically nudge fraud probability upward
            current_fraud_prob = round(min(0.28, current_fraud_prob + 0.02), 4)
        elif fraud_pct > MAX_FRAUD_PCT:
            # Deterministically nudge fraud probability downward
            current_fraud_prob = round(max(0.12, current_fraud_prob - 0.02), 4)

        # Increment seed deterministically to advance pseudorandom trajectory
        current_seed += 1

    raise RuntimeError(
        f"Failed to generate compliant dataset after {max_attempts} deterministic attempts. "
        f"Last errors: {errors}"
    )


# ============================================================================
# Console Output & Visualization
# ============================================================================

def print_dataset_summary(stats: Dict[str, Any], errors: Optional[List[str]] = None) -> None:
    """Print a structured, readable console report showing dataset distributions and validation.

    Args:
        stats: Computed statistics dictionary from validate_loan_dataset.
        errors: Optional list of validation errors.
    """
    total = stats.get("total_records", 0)
    meta = stats.get("generation_metadata", {})
    final_seed = meta.get("final_seed", INITIAL_SEED)
    attempts = meta.get("attempts", 1)

    print("\n" + "=" * 76)
    print(" CRED DOMAIN SUPPORT AGENT — LOAN APPLICATION DATASET VALIDATION REPORT")
    print("=" * 76)
    print(f" Status                : {'[VALID]' if not errors else '[INVALID]'}")
    print(f" Total Records         : {total} (Requirement: >= 50)")
    print(f" Active Seed           : {final_seed} (Initial seed: {INITIAL_SEED}, Attempts: {attempts})")
    print(f" Amount Design Bounds  : INR {MIN_LOAN_AMOUNT_INR:,} to INR {MAX_LOAN_AMOUNT_INR:,}")
    print("-" * 76)

    # Category Counts
    print(" LOAN CATEGORY DISTRIBUTION (Requirement: every category >= 3 records):")
    cat_counts = stats.get("category_counts", {})
    for cat, count in cat_counts.items():
        pct = (count / total * 100) if total else 0
        status_marker = "[PASS]" if count >= MIN_CATEGORY_COUNT else "[FAIL]"
        print(f"   * {cat:<18}: {count:>2} records ({pct:>5.1f}%) {status_marker}")

    print("-" * 76)

    # Status Counts
    print(" LOAN APPLICATION STATUS DISTRIBUTION (Requirement: every status >= 1 record):")
    stat_counts = stats.get("status_counts", {})
    for st, count in stat_counts.items():
        pct = (count / total * 100) if total else 0
        status_marker = "[PASS]" if count >= MIN_STATUS_COUNT else "[FAIL]"
        print(f"   * {st:<18}: {count:>2} records ({pct:>5.1f}%) {status_marker}")

    print("-" * 76)

    # Fraud Review Percentage
    fraud_count = stats.get("fraud_count", 0)
    fraud_pct = stats.get("fraud_percentage", 0.0)
    fraud_marker = "[PASS]" if (MIN_FRAUD_PCT <= fraud_pct <= MAX_FRAUD_PCT) else "[FAIL]"
    print(" FRAUD REVIEW FLAG DISTRIBUTION (Requirement: 10.0% to 30.0%):")
    print(
        f"   * Flagged for Fraud : {fraud_count} / {total} records "
        f"({fraud_pct:.1f}%) {fraud_marker} (Target: [{MIN_FRAUD_PCT}%, {MAX_FRAUD_PCT}%])"
    )

    print("-" * 76)

    # Financial & Operational Metrics
    amt = stats.get("amount_stats", {})
    days = stats.get("days_stats", {})
    print(" FINANCIAL & OPERATIONAL DISTRIBUTIONS:")
    print(
        f"   * Loan Amounts (INR): Min=INR {amt.get('min', 0):,}, "
        f"Max=INR {amt.get('max', 0):,}, Median=INR {amt.get('median', 0):,}, "
        f"Mean=INR {int(amt.get('mean', 0)):,}"
    )
    print(
        f"   * Days Since Created: Min={days.get('min', 0)}d, "
        f"Max={days.get('max', 0)}d, Median (p50)={days.get('p50', 0)}d, "
        f"80th Percentile (p80)={days.get('p80', 0)}d"
    )

    if errors:
        print("-" * 76)
        print(" VALIDATION FAILURES:")
        for err in errors:
            print(f"   [!] {err}")

    print("=" * 76)


# ============================================================================
# Module Exports & Lookup Helpers
# ============================================================================

# Pre-generated, validated dataset for direct import across capstone tasks
LOAN_APPLICATIONS, DATASET_STATS = generate_loan_dataset(
    num_records=DEFAULT_NUM_RECORDS,
    initial_seed=INITIAL_SEED,
    initial_fraud_prob=DEFAULT_FRAUD_PROBABILITY,
)

# In-memory dictionary for O(1) lookup by record_id (used in Task 6 lookup tool)
LOAN_APPLICATIONS_BY_ID: Dict[str, LoanApplication] = {
    rec["record_id"]: rec for rec in LOAN_APPLICATIONS
}


def get_loan_application_by_id(record_id: str) -> Optional[LoanApplication]:
    """Retrieve a loan application record by its unique identifier.

    Args:
        record_id: e.g. 'APP-1001'

    Returns:
        The LoanApplication record dict if found, or None.
    """
    return LOAN_APPLICATIONS_BY_ID.get(record_id.strip().upper())


# ============================================================================
# Main Execution Block
# ============================================================================

if __name__ == "__main__":
    # Validate the pre-generated module dataset
    valid, summary_stats, errs = validate_loan_dataset(LOAN_APPLICATIONS)
    print_dataset_summary(DATASET_STATS, errors=errs)

    # Print first 5 sample records as formatted preview
    print("\nSAMPLE RECORDS PREVIEW (First 5 records):")
    print(f"{'Record ID':<10} | {'Category':<16} | {'Status':<14} | {'Amount (INR)':<14} | {'Days':<6} | {'Fraud Review'}")
    print("-" * 76)
    for app in LOAN_APPLICATIONS[:5]:
        print(
            f"{app['record_id']:<10} | "
            f"{app['category']:<16} | "
            f"{app['status']:<14} | "
            f"INR {app['loan_amount_inr']:>10,} | "
            f"{app['days_since_created']:>4}d | "
            f"{str(app['flagged_for_fraud_review']):<12}"
        )
    print("-" * 76)

