"""Generate a structured-response evaluation transcript using offline MOCK_LLM."""

from datetime import datetime
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from src.cred_agent.crew import run_customer_query, validate_composer_response
from src.cred_agent.schemas import SupportResponse, validate_support_response


def run_structured_response_demo() -> str:
    """Run a real RAG query and record successful and rejected validations."""
    result = run_customer_query("How is EMI calculated?")
    validated = validate_support_response(result.structured_response)

    invalid_payload = {
        "response_type": "loan_application",
        "is_grounded": True,
        "source_documents": [],
        "retrieved_context": [],
        "similarity_measurements": [],
    }
    safe_response, validation_error = validate_composer_response(invalid_payload)

    lines = [
        "# Structured Crew Response Demo",
        "",
        f"**Generated:** {datetime.now().astimezone().isoformat(timespec='seconds')}",
        "",
        "CrewAI 1.15.37 exposes `Task.response_model` for structured LLM output; "
        "Crew and `Crew.kickoff()` do not expose a `response_format` parameter. "
        "The Response Composer task sets both `response_model` and "
        "`output_pydantic` to the Pydantic model below. MOCK_LLM returns a "
        "deterministic local model instance; the workflow validates it again "
        "before returning the response.",
        "",
        "## Pydantic schema",
        "",
        "Model: `src.cred_agent.schemas.SupportResponse`",
        "",
        "```json",
        json.dumps(SupportResponse.model_json_schema(), indent=2),
        "```",
        "",
        "## Successful grounded RAG response",
        "",
        f"**Query:** {result.query}",
        "",
        f"**Pydantic validation:** passed (`{type(validated).__name__}`)",
        "",
        "```json",
        validated.model_dump_json(indent=2),
        "```",
        "",
        "**Customer-facing response:**",
        "",
        validated.response,
        "",
        "## Deliberately invalid response",
        "",
        "Input omits required `response` and does not provide a valid "
        "`loan_application` object:",
        "",
        "```json",
        json.dumps(invalid_payload, indent=2),
        "```",
        "",
        "**Validation result:** rejected by Pydantic.",
        "",
        "```text",
        validation_error or "No validation error was reported.",
        "```",
        "",
        "**Safe handling:** the customer-facing result is replaced with the "
        "schema-valid safe error response; no unvalidated content is returned.",
        "",
        "```json",
        safe_response.model_dump_json(indent=2),
        "```",
        "",
        "**Offline execution:** the retrieval model runs from its local cache; "
        "MOCK_LLM makes no external LLM API calls.",
    ]
    transcript = "\n".join(lines) + "\n"
    report_path = PROJECT_ROOT / "evaluation" / "structured_response_transcript.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(transcript, encoding="utf-8")
    print(transcript)
    print(f"Transcript saved to: {report_path}")
    return transcript


if __name__ == "__main__":
    run_structured_response_demo()
