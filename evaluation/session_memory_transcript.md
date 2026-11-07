# Session Memory Demo Transcript

**Transcript generated:** 2026-11-07T23:03:48+05:30

Each turn shows the LangChain history supplied to the workflow before the request and the stored history after the response. Timestamps use the machine's local timezone.

## SESSION A: interaction 1

**Session ID:** `session-a`

**Started:** 2026-11-07T23:03:48+05:30

**Completed:** 2026-11-07T23:03:58+05:30

**History before interaction (context supplied to workflow):**
(empty)

**User:** What is the EMI calculation method?

**Assistant:** Monthly Equated Monthly Installments (EMIs) across all Cred lending products are calculated using the reducing balance method based on the principal amount, tenure, and applicable annual interest rate.
Source: emi_calculation (emi_calculation.md).

**History after interaction:**
- User: What is the EMI calculation method?
- Assistant: Monthly Equated Monthly Installments (EMIs) across all Cred lending products are calculated using the reducing balance method based on the principal amount, tenure, and applicable annual interest rate.
Source: emi_calculation (emi_calculation.md).

---

## SESSION A: interaction 2

**Session ID:** `session-a`

**Started:** 2026-11-07T23:03:58+05:30

**Completed:** 2026-11-07T23:03:58+05:30

**History before interaction (context supplied to workflow):**
- User: What is the EMI calculation method?
- Assistant: Monthly Equated Monthly Installments (EMIs) across all Cred lending products are calculated using the reducing balance method based on the principal amount, tenure, and applicable annual interest rate.
Source: emi_calculation (emi_calculation.md).

**User:** What tenure options are available?

**Assistant:** Monthly Equated Monthly Installments (EMIs) across all Cred lending products are calculated using the reducing balance method based on the principal amount, tenure, and applicable annual interest rate. Borrowers may select tenures from 6 to 60 months, with payments auto-debited on the 5th calendar day of each month via registered NACH or UPI auto-pay channels.
Source: emi_calculation (emi_calculation.md).

**History after interaction:**
- User: What is the EMI calculation method?
- Assistant: Monthly Equated Monthly Installments (EMIs) across all Cred lending products are calculated using the reducing balance method based on the principal amount, tenure, and applicable annual interest rate.
Source: emi_calculation (emi_calculation.md).
- User: What tenure options are available?
- Assistant: Monthly Equated Monthly Installments (EMIs) across all Cred lending products are calculated using the reducing balance method based on the principal amount, tenure, and applicable annual interest rate. Borrowers may select tenures from 6 to 60 months, with payments auto-debited on the 5th calendar day of each month via registered NACH or UPI auto-pay channels.
Source: emi_calculation (emi_calculation.md).

---

## SESSION B: interaction 3

**Session ID:** `session-b`

**Started:** 2026-11-07T23:03:58+05:30

**Completed:** 2026-11-07T23:03:58+05:30

**History before interaction (context supplied to workflow):**
(empty)

**User:** What tenure options are available?

**Assistant:** Monthly Equated Monthly Installments (EMIs) across all Cred lending products are calculated using the reducing balance method based on the principal amount, tenure, and applicable annual interest rate. Borrowers may select tenures from 6 to 60 months, with payments auto-debited on the 5th calendar day of each month via registered NACH or UPI auto-pay channels.
Source: emi_calculation (emi_calculation.md).

**History after interaction:**
- User: What tenure options are available?
- Assistant: Monthly Equated Monthly Installments (EMIs) across all Cred lending products are calculated using the reducing balance method based on the principal amount, tenure, and applicable annual interest rate. Borrowers may select tenures from 6 to 60 months, with payments auto-debited on the 5th calendar day of each month via registered NACH or UPI auto-pay channels.
Source: emi_calculation (emi_calculation.md).

---

**Isolation evidence:** SESSION B uses a separate history object and had empty history before its first interaction. It could not access SESSION A's messages; SESSION A's history was unchanged after SESSION B completed.

**Retrieval note:** Adjacent fixed-size chunks are rejoined using their exact overlap before extractive sentence matching. Retrieved chunk records remain unchanged, while the composed answers below contain the complete tenure sentence once.
