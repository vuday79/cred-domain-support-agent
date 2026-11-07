# Cred Domain Support Agent — Knowledge Base

This directory contains the domain policy knowledge base for the **Cred Domain Support Agent** certification capstone project.

## Purpose
The documents in this knowledge base provide the grounded context needed by the Retrieval-Augmented Generation (RAG) system, CrewAI agents, and AutoGen governance review stages. They cover essential banking and FinTech policy topics such as loan eligibility criteria, interest rate slabs, EMI computation, KYC compliance, fraud dispute resolution, account closure, credit card fee structures, and NRI banking.

## Simulated Policy Disclaimer
> [!IMPORTANT]
> **Academic & Certification Purpose Only:**
> All policies, terms, fee structures, interest slabs, and operational rules contained in this knowledge base are **simulated policies** created exclusively for this academic capstone project.
> They **do not represent** real-world policies, fees, terms, or guidelines of **CRED** (Dreamplug Technologies Pvt. Ltd.) or any affiliated banking partner or financial institution.
> These simulated policies are designed to test AI agent retrieval accuracy, grounded reasoning, guardrail compliance, and multi-agent coordination under deterministic conditions.

## Knowledge Base Inventory & Topic Mapping

| # | File Name | Policy Topic | Section Coverage |
|---|---|---|---|
| 1 | `loan_eligibility.md` | Loan eligibility criteria by loan type | Personal, Home, Auto, Education, and Business loan requirements, income thresholds, co-applicant criteria, and underwriting escalation. |
| 2 | `emi_calculation.md` | EMI calculation rules | Reducing balance methodology, amortization trajectory, NACH auto-debit dates, 3-day grace period, and restructuring exceptions. |
| 3 | `credit_card_fees.md` | Credit-card fee structure | Annual membership fee & spend waiver, 45-day interest-free grace period, cash advance fees, forex markup, and dispute escalation. |
| 4 | `kyc_requirements.md` | KYC document requirements | RBI compliance, PAN & Aadhaar/Passport OVDs, address verification, 72-hour video-KYC requirement, and AML escalation. |
| 5 | `fraud_disputes.md` | Fraud-dispute resolution process | 24-hour reporting window, immediate protective card/account freeze, 7–10 day forensic audit SLA, provisional credit, and Nodal Officer escalation. |
| 6 | `account_closure.md` | Account-closure process | Pre-conditions (zero outstanding balance), mandate deactivations within 2 days, 5-day NOC issuance, and lien/dispute handling. |
| 7 | `interest_rates.md` | Interest-rate slabs | Credit score-linked tiering (Tier-1: 780+, Tier-2: 720-779, Tier-3: 680-719), quarterly repo rate adjustments, and sub-prime escalation. |
| 8 | `prepayment_rules.md` | Prepayment-penalty rules | Zero penalty on floating rate retail loans (RBI directive), 2%/1% rules for fixed-rate loans, partial prepayment rules, and hardship waivers. |
| 9 | `minimum_balance.md` | Minimum-balance requirements | Average Monthly Balance (AMB) thresholds (Metro/Urban: INR 10,000; Semi-urban/Rural: INR 5,000), shortfalls, INR 250 penalty, and salary account exemptions. |
| 10 | `credit_score.md` | Credit-score impact factors | Five CIBIL scoring dimensions (35% payment history, 30% utilization, 15% age, 10% mix, 10% inquiries), hard inquiry impacts, and bureau correction. |
| 11 | `joint_accounts.md` | Joint-account rules | Mandates (Either-or-Survivor, Jointly, Former-or-Survivor), joint & several loan liability, unanimous consent for closure, and deceased holder survivorship. |
| 12 | `nri_eligibility.md` | NRI-account eligibility | NRE vs. NRO account distinctions, tax-free repatriation, USD 1M statutory limits, mandatory NRI KYC, and FEMA compliance escalation. |

