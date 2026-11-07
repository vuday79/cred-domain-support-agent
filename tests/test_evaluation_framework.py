"""Tests for the deterministic Task 13 evaluation dataset, judge, and runner."""

import os
from pathlib import Path
from typing import List

from scripts.run_task13_evaluation import (
    DIMENSIONS,
    evaluate_cases,
    load_evaluation_cases,
)
from src.cred_agent.crew import IntegratedCrewResult, QueryRoute, run_customer_query
from src.cred_agent.evaluation_judge import score_case
from src.cred_agent.mock_llm import CredMockLLM
from src.cred_agent.rag.retrieval import PolicyRetriever
from src.cred_agent.schemas import (
    RetrievedContext,
    SimilarityMeasurement,
    SupportResponse,
    SupportResponseType,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _safe_error_result(
    query: str = "test query",
    kickoff_count: int = 0,
    prompt_injection_detected: bool = False,
) -> IntegratedCrewResult:
    response = SupportResponse(
        response="I couldn't safely validate a response from the available verified information. Please try again.",
        response_type=SupportResponseType.SAFE_ERROR,
        is_grounded=False,
        source_documents=[],
        retrieved_context=[],
        similarity_measurements=[],
    )
    return IntegratedCrewResult(
        query=query,
        route=QueryRoute(False, False, None),
        specialist_findings=[],
        tool_invocations=[],
        final_response=response.response,
        kickoff_count=kickoff_count,
        structured_response=response,
        input_pii_categories=[],
        prompt_injection_detected=prompt_injection_detected,
        groundedness_passed=True,
    )


def test_dataset_covers_all_kb_topics_and_representative_edge_cases():
    cases = load_evaluation_cases(PROJECT_ROOT / "evaluation" / "evaluation_dataset.json")
    expected_topics = {
        "loan_eligibility",
        "emi_calculation",
        "credit_card_fees",
        "kyc_requirements",
        "fraud_disputes",
        "account_closure",
        "interest_rates",
        "prepayment_rules",
        "minimum_balance",
        "credit_score",
        "joint_accounts",
        "nri_eligibility",
    }

    assert len(cases) >= 15
    assert expected_topics.issubset(
        {
            source
            for case in cases
            for source in case["expected_sources"]
        }
    )
    assert sum(case["expected_behavior"] == "fallback" for case in cases) >= 2
    assert any(case["expected_behavior"] == "loan_lookup" for case in cases)
    assert any(case["expected_behavior"] == "combined" for case in cases)
    assert any(case["expect_injection_rejected"] for case in cases)


def test_evaluator_configures_keyless_local_mock_and_offline_runtime():
    assert CredMockLLM().api_key is None
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
    assert os.environ["CREWAI_DISABLE_TELEMETRY"] == "true"


def test_judge_is_deterministic_and_scores_stay_in_documented_range():
    cases = load_evaluation_cases()
    query_case = next(case for case in cases if case["query_id"] == "injection_rejected")
    result = _safe_error_result(
        query=query_case["query"],
        prompt_injection_detected=True,
    )

    first = score_case(query_case, result)
    second = score_case(query_case, result)

    assert first == second
    assert set(first) == set(DIMENSIONS)
    assert all(0 <= item["score"] <= 5 for item in first.values())


def test_schema_valid_cited_but_unsupported_answer_fails_grounding():
    case = next(
        case
        for case in load_evaluation_cases()
        if case["query_id"] == "kb_loan_eligibility"
    )
    response = SupportResponse(
        response="The moon is made of cheese.",
        response_type=SupportResponseType.KNOWLEDGE_BASE,
        is_grounded=True,
        source_documents=["loan_eligibility"],
        retrieved_context=[
            RetrievedContext(
                chunk_id="eligibility-0",
                parent_document_id="loan_eligibility",
                source="loan_eligibility.md",
                text="Personal loan applicants must be Indian residents aged 21 to 58 years.",
                cosine_similarity=0.9,
            )
        ],
        similarity_measurements=[
            SimilarityMeasurement(
                chunk_id="eligibility-0",
                cosine_similarity=0.9,
                distance=0.1,
            )
        ],
    )
    result = IntegratedCrewResult(
        query=case["query"],
        route=QueryRoute(True, False, None),
        specialist_findings=[],
        tool_invocations=[],
        final_response=response.response,
        kickoff_count=2,
        structured_response=response,
        groundedness_passed=True,
    )

    scores = score_case(case, result)

    assert scores["grounding"]["score"] == 0
    assert "failed" in scores["grounding"]["reason"].lower()


def test_evaluate_cases_executes_and_retains_every_query_including_failures():
    cases = load_evaluation_cases()
    calls: List[str] = []

    def workflow(query: str, retriever: PolicyRetriever) -> IntegratedCrewResult:
        del retriever
        calls.append(query)
        if query == cases[0]["query"]:
            raise RuntimeError("deliberately omitted from the report")
        return _safe_error_result(query=query)

    report = evaluate_cases(cases, retriever=PolicyRetriever.__new__(PolicyRetriever), workflow=workflow)

    assert calls == [case["query"] for case in cases]
    assert len(report["cases"]) == len(cases)
    assert report["aggregates"]["case_count"] == len(cases)
    assert report["aggregates"]["failed_count"] >= 1
    assert report["cases"][0]["actual"]["error_category"] == "RuntimeError"
    for dimension in DIMENSIONS:
        scores = [item["scores"][dimension]["score"] for item in report["cases"]]
        assert report["aggregates"]["average_scores"][dimension] == round(
            sum(scores) / len(scores), 4
        )


def test_real_workflow_evaluation_uses_local_persisted_retrieval():
    from scripts.run_rag_demo import load_persisted_retriever

    case = next(
        case
        for case in load_evaluation_cases()
        if case["query_id"] == "kb_emi_calculation"
    )
    result = run_customer_query(case["query"], retriever=load_persisted_retriever())
    scores = score_case(case, result)

    assert result.kickoff_count > 0
    assert result.structured_response.response_type is SupportResponseType.KNOWLEDGE_BASE
    assert result.structured_response.source_documents
    assert result.groundedness_passed
    assert all(0 <= item["score"] <= 5 for item in scores.values())
