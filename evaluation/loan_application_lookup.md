# Loan Application Lookup and Escalation Score

`check_loan_application_status(record_id: str)` is a reusable, CrewAI-independent
lookup over the seeded `data.dataset.LOAN_APPLICATIONS` dataset. Successful results
include the dataset's record fields, an escalation score, the escalation decision,
and an explanation. Invalid IDs and unknown IDs return explicit `found: false`
results; neither case fabricates application data.

## Scoring rule

The scoring rule is a capstone demonstration, **not a real credit decision or
recommendation**.

```text
normalized_recency = (30 - days_since_created) / 30

escalation_score = clamp(
    round(
        (0.70 if flagged_for_fraud_review else 0.0)
        + 0.30 * normalized_recency,
        4,
    ),
    0.0,
    1.0,
)

escalation_required = escalation_score >= 0.65
```

Fraud review receives 70% because it is the explicit high-priority risk marker
in the synthetic application schema. Recency receives 30%; a newly created
application has a greater normalized recency value than an older one, reflecting
the demonstration's operational preference for timely review. A 0.65 threshold
is above the largest possible non-flagged score (0.30) and below the smallest
flagged score (0.70): recency alone cannot trigger escalation, while a fraud
flag always does. Within both groups, newer applications receive a higher score.

## Worked examples from the generated dataset

- `APP-1011` is flagged for fraud review and 22 days old:
  `0.70 + 0.30 * ((30 - 22) / 30) = 0.7800`; escalation is required.
- `APP-1009` is not flagged and 29 days old:
  `0.00 + 0.30 * ((30 - 29) / 30) = 0.0100`; escalation is not required.

At the boundaries, a flagged application created today scores `1.0000`, and an
unflagged application 30 days old scores `0.0000`.

## Validation

Run the lookup tests with:

```powershell
python -m pytest tests\test_loan_application_lookup.py -q
```
