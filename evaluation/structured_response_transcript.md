# Structured Crew Response Demo

**Generated:** 2026-11-07T23:09:59+05:30

CrewAI 1.15.37 exposes `Task.response_model` for structured LLM output; Crew and `Crew.kickoff()` do not expose a `response_format` parameter. The Response Composer task sets both `response_model` and `output_pydantic` to the Pydantic model below. MOCK_LLM returns a deterministic local model instance; the workflow validates it again before returning the response.

## Pydantic schema

Model: `src.cred_agent.schemas.SupportResponse`

```json
{
  "$defs": {
    "LoanApplicationDetails": {
      "additionalProperties": false,
      "description": "Verified application fields returned by the authorized lookup tool.",
      "properties": {
        "record_id": {
          "minLength": 1,
          "title": "Record Id",
          "type": "string"
        },
        "category": {
          "minLength": 1,
          "title": "Category",
          "type": "string"
        },
        "status": {
          "$ref": "#/$defs/LoanApplicationStatus"
        },
        "loan_amount_inr": {
          "minimum": 0,
          "title": "Loan Amount Inr",
          "type": "integer"
        },
        "days_since_created": {
          "minimum": 0,
          "title": "Days Since Created",
          "type": "integer"
        },
        "flagged_for_fraud_review": {
          "title": "Flagged For Fraud Review",
          "type": "boolean"
        },
        "escalation_score": {
          "maximum": 1.0,
          "minimum": 0.0,
          "title": "Escalation Score",
          "type": "number"
        },
        "escalation_required": {
          "title": "Escalation Required",
          "type": "boolean"
        },
        "escalation_reason": {
          "title": "Escalation Reason",
          "type": "string"
        }
      },
      "required": [
        "record_id",
        "category",
        "status",
        "loan_amount_inr",
        "days_since_created",
        "flagged_for_fraud_review",
        "escalation_score",
        "escalation_required",
        "escalation_reason"
      ],
      "title": "LoanApplicationDetails",
      "type": "object"
    },
    "LoanApplicationStatus": {
      "description": "Allowed statuses present in the generated loan application dataset.",
      "enum": [
        "Approved",
        "Disbursed",
        "Rejected",
        "Submitted",
        "Under Review"
      ],
      "title": "LoanApplicationStatus",
      "type": "string"
    },
    "RetrievedContext": {
      "additionalProperties": false,
      "description": "A retained, attributed chunk returned by policy retrieval.",
      "properties": {
        "chunk_id": {
          "minLength": 1,
          "title": "Chunk Id",
          "type": "string"
        },
        "parent_document_id": {
          "minLength": 1,
          "title": "Parent Document Id",
          "type": "string"
        },
        "source": {
          "minLength": 1,
          "title": "Source",
          "type": "string"
        },
        "text": {
          "minLength": 1,
          "title": "Text",
          "type": "string"
        },
        "cosine_similarity": {
          "maximum": 1.0,
          "minimum": -1.0,
          "title": "Cosine Similarity",
          "type": "number"
        }
      },
      "required": [
        "chunk_id",
        "parent_document_id",
        "source",
        "text",
        "cosine_similarity"
      ],
      "title": "RetrievedContext",
      "type": "object"
    },
    "SimilarityMeasurement": {
      "additionalProperties": false,
      "description": "Cosine similarity and original Chroma distance for one retrieved chunk.",
      "properties": {
        "chunk_id": {
          "minLength": 1,
          "title": "Chunk Id",
          "type": "string"
        },
        "cosine_similarity": {
          "maximum": 1.0,
          "minimum": -1.0,
          "title": "Cosine Similarity",
          "type": "number"
        },
        "distance": {
          "maximum": 2.0,
          "minimum": 0.0,
          "title": "Distance",
          "type": "number"
        }
      },
      "required": [
        "chunk_id",
        "cosine_similarity",
        "distance"
      ],
      "title": "SimilarityMeasurement",
      "type": "object"
    },
    "SupportResponseType": {
      "description": "Supported classifications for generated customer-facing responses.",
      "enum": [
        "knowledge_base",
        "knowledge_base_fallback",
        "loan_application",
        "application_not_found",
        "combined",
        "safe_error"
      ],
      "title": "SupportResponseType",
      "type": "string"
    }
  },
  "additionalProperties": false,
  "description": "Complete, validated result returned by the integrated support Crew.",
  "properties": {
    "response": {
      "minLength": 1,
      "title": "Response",
      "type": "string"
    },
    "response_type": {
      "$ref": "#/$defs/SupportResponseType"
    },
    "is_grounded": {
      "title": "Is Grounded",
      "type": "boolean"
    },
    "source_documents": {
      "items": {
        "type": "string"
      },
      "title": "Source Documents",
      "type": "array"
    },
    "retrieved_context": {
      "items": {
        "$ref": "#/$defs/RetrievedContext"
      },
      "title": "Retrieved Context",
      "type": "array"
    },
    "similarity_measurements": {
      "items": {
        "$ref": "#/$defs/SimilarityMeasurement"
      },
      "title": "Similarity Measurements",
      "type": "array"
    },
    "record_id": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "title": "Record Id"
    },
    "loan_application": {
      "anyOf": [
        {
          "$ref": "#/$defs/LoanApplicationDetails"
        },
        {
          "type": "null"
        }
      ],
      "default": null
    }
  },
  "required": [
    "response",
    "response_type",
    "is_grounded",
    "source_documents",
    "retrieved_context",
    "similarity_measurements"
  ],
  "title": "SupportResponse",
  "type": "object"
}
```

## Successful grounded RAG response

**Query:** How is EMI calculated?

**Pydantic validation:** passed (`SupportResponse`)

```json
{
  "response": "Monthly Equated Monthly Installments (EMIs) across all Cred lending products are calculated using the reducing balance method based on the principal amount, tenure, and applicable annual interest rate.\nSource: emi_calculation (emi_calculation.md).",
  "response_type": "knowledge_base",
  "is_grounded": true,
  "source_documents": [
    "emi_calculation"
  ],
  "retrieved_context": [
    {
      "chunk_id": "fixed_emi_calculation_0",
      "parent_document_id": "emi_calculation",
      "source": "D:/Uday Verma/Personal/upskill/Agentic Systems Design/repos/cred-domain-support-agent/data/knowledge_base/emi_calculation.md",
      "text": "Monthly Equated Monthly Installments (EMIs) across all Cred lending products are calculated using the reducing balance method based on the principal amount, tenure, and applicable annual interest rate. The repayment schedule is amortized so earlier installments allocate a higher interest proportion, which systematically transitions toward principal liquidation as the balance declines. Borrowers may select tenures from 6 to 60 months, with payments auto-debited on the 5th calendar day of each mon",
      "cosine_similarity": 0.4502679109573364
    },
    {
      "chunk_id": "fixed_emi_calculation_1",
      "parent_document_id": "emi_calculation",
      "source": "D:/Uday Verma/Personal/upskill/Agentic Systems Design/repos/cred-domain-support-agent/data/knowledge_base/emi_calculation.md",
      "text": "y select tenures from 6 to 60 months, with payments auto-debited on the 5th calendar day of each month via registered NACH or UPI auto-pay channels. If an auto-debit attempt fails due to insufficient balance, an automated 3-business-day grace period is provided before late payment fees apply. Requests for EMI restructuring or repayment re-amortization due to financial hardship must be escalated to the Lending Operations Committee.",
      "cosine_similarity": 0.36431145668029785
    },
    {
      "chunk_id": "fixed_minimum_balance_0",
      "parent_document_id": "minimum_balance",
      "source": "D:/Uday Verma/Personal/upskill/Agentic Systems Design/repos/cred-domain-support-agent/data/knowledge_base/minimum_balance.md",
      "text": "Cred Smart Savings accounts require account holders to maintain an Average Monthly Balance (AMB) of INR 10,000 across metropolitan and urban branches, or INR 5,000 in semi-urban and rural locations. Average Monthly Balance is computed by summing the end-of-day ledger balances across all calendar days of the month and dividing by the total number of days in that month. Account holders falling below the mandated threshold receive automated balance shortfall warning notifications on the 15th and 25",
      "cosine_similarity": 0.28202348947525024
    }
  ],
  "similarity_measurements": [
    {
      "chunk_id": "fixed_emi_calculation_0",
      "cosine_similarity": 0.4502679109573364,
      "distance": 0.5497320890426636
    },
    {
      "chunk_id": "fixed_emi_calculation_1",
      "cosine_similarity": 0.36431145668029785,
      "distance": 0.6356885433197021
    },
    {
      "chunk_id": "fixed_minimum_balance_0",
      "cosine_similarity": 0.28202348947525024,
      "distance": 0.7179765105247498
    }
  ],
  "record_id": null,
  "loan_application": null
}
```

**Customer-facing response:**

Monthly Equated Monthly Installments (EMIs) across all Cred lending products are calculated using the reducing balance method based on the principal amount, tenure, and applicable annual interest rate.
Source: emi_calculation (emi_calculation.md).

## Deliberately invalid response

Input omits required `response` and does not provide a valid `loan_application` object:

```json
{
  "response_type": "loan_application",
  "is_grounded": true,
  "source_documents": [],
  "retrieved_context": [],
  "similarity_measurements": []
}
```

**Validation result:** rejected by Pydantic.

```text
1 validation error for SupportResponse
response
  Field required [type=missing, input_value={'response_type': 'loan_a...arity_measurements': []}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.12/v/missing
```

**Safe handling:** the customer-facing result is replaced with the schema-valid safe error response; no unvalidated content is returned.

```json
{
  "response": "I couldn't safely validate a response from the available verified information. Please try again.",
  "response_type": "safe_error",
  "is_grounded": false,
  "source_documents": [],
  "retrieved_context": [],
  "similarity_measurements": [],
  "record_id": null,
  "loan_application": null
}
```

**Offline execution:** the retrieval model runs from its local cache; MOCK_LLM makes no external LLM API calls.
