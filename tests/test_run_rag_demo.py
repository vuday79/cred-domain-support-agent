"""Tests for the runnable RAG demonstration and document-level evaluation."""

from pathlib import Path
from typing import Any, Dict, List

import pytest

from scripts import run_rag_demo
from src.cred_agent.rag.retrieval import FALLBACK_RESPONSE
from src.cred_agent.rag.vector_store import COLLECTION_FIXED, COLLECTION_SENTENCE


class DemoRetriever:
    """Deterministic fake retriever with duplicate parent chunks for evaluation."""

    def __init__(self, overlap: bool = False):
        self.overlap = overlap

    def retrieve(
        self,
        query: str,
        collection_name: str,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        query_info = next(
            (
                item for item in run_rag_demo.IN_SCOPE_QUERIES
                if item["query"] == query
            ),
            None,
        )
        if query_info is None:
            similarity = 0.4 if self.overlap else (
                0.10 if "IPL" in query else 0.12
            )
            parent_id = "loan_eligibility"
            text = "Personal loan eligibility requires Indian residency."
            chunk_prefix = "oos"
        else:
            score_by_query = {
                "personal_loan_eligibility": 0.72,
                "emi_calculation": 0.61,
                "kyc_documents": 0.68,
                "prepayment_penalty": 0.75,
                "credit_scores": 0.57,
            }
            similarity = (
                0.4 if self.overlap
                else score_by_query[query_info["query_id"]]
            )
            parent_id = next(iter(query_info["ground_truth_docs"]))
            text_by_doc = {
                "loan_eligibility":
                    "Personal loan eligibility requires Indian residency.",
                "emi_calculation":
                    "EMI is calculated using the reducing balance method.",
                "kyc_requirements":
                    "KYC requires verified PAN and address documents.",
                "prepayment_rules":
                    "Fixed loan prepayment has a 2% penalty.",
                "credit_score":
                    "Payment history affects the credit score.",
            }
            text = text_by_doc[parent_id]
            chunk_prefix = query_info["query_id"]

        hits = [
            {
                "chunk_id": f"{chunk_prefix}_0",
                "text": text,
                "parent_document_id": parent_id,
                "doc_id": parent_id,
                "source": f"data/knowledge_base/{parent_id}.md",
                "cosine_similarity": similarity,
                "distance": 1.0 - similarity,
            },
            {
                "chunk_id": f"{chunk_prefix}_1",
                "text": text,
                "parent_document_id": parent_id,
                "doc_id": parent_id,
                "source": f"data/knowledge_base/{parent_id}.md",
                "cosine_similarity": similarity - 0.05,
                "distance": 1.0 - (similarity - 0.05),
            },
            {
                "chunk_id": f"{chunk_prefix}_2",
                "text": "Account information is held in the local policy system.",
                "parent_document_id": "account_closure",
                "doc_id": "account_closure",
                "source": "data/knowledge_base/account_closure.md",
                "cosine_similarity": 0.2,
                "distance": 0.8,
            },
        ]
        return hits[:top_k]


def test_run_demo_saves_report_and_compares_both_collections(tmp_path: Path):
    report_path = tmp_path / "evaluation" / "rag_evaluation_report.md"
    result = run_rag_demo.run_rag_demo(DemoRetriever(), report_path=report_path)

    assert report_path.is_file()
    report = report_path.read_text(encoding="utf-8")
    assert "Part 1, Task 4" in report
    assert "Part 1, Task 5" in report
    assert "cosine similarity" in report
    assert "parent document" in report
    assert "source" in report
    assert "| 1 / 2 = 0.5000 | 1 / 1 = 1.0000 |" in report
    assert len(result["demonstrations"]) == 14
    assert set(result["evaluations"]) == {COLLECTION_FIXED, COLLECTION_SENTENCE}
    assert result["calibration"][COLLECTION_FIXED]["chosen_threshold"] == pytest.approx(0.345)
    assert all(
        row["precision"] == pytest.approx(0.5)
        and row["recall"] == pytest.approx(1.0)
        for collection in result["evaluations"].values()
        for row in collection["queries"]
    )
    assert all(
        demo["result"]["answer"] == FALLBACK_RESPONSE
        for demo in result["demonstrations"]
        if demo["scope"] == "Out-of-scope"
    )


def test_calibration_failure_does_not_select_threshold_or_answer(tmp_path: Path):
    result = run_rag_demo.run_rag_demo(
        DemoRetriever(overlap=True),
        report_path=tmp_path / "overlap.md",
    )

    for collection in (COLLECTION_FIXED, COLLECTION_SENTENCE):
        report = result["calibration"][collection]
        assert not report["separation_found"]
        assert report["chosen_threshold"] is None
    assert all(
        demo["result"]["answer"] == FALLBACK_RESPONSE
        and demo["result"]["threshold"] is None
        for demo in result["demonstrations"]
    )
    assert "calibration failure" in (tmp_path / "overlap.md").read_text(
        encoding="utf-8"
    ).lower()


def test_missing_collection_is_reported_without_creating_a_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    database_dir = tmp_path / "chroma"
    database_dir.mkdir()
    (database_dir / "chroma.sqlite3").touch()

    class ExistingCollection:
        name = COLLECTION_FIXED

    class ExistingClient:
        def list_collections(self):
            return [ExistingCollection()]

    monkeypatch.setattr(
        run_rag_demo.chromadb,
        "PersistentClient",
        lambda **kwargs: ExistingClient(),
    )

    with pytest.raises(
        run_rag_demo.MissingVectorCollectionsError,
        match=COLLECTION_SENTENCE,
    ):
        run_rag_demo.load_persisted_retriever(database_dir)


def test_missing_storage_path_is_reported(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="persistence directory"):
        run_rag_demo.load_persisted_retriever(tmp_path / "missing")


def test_strategy_recommendation_uses_macro_f1_then_tiebreaks():
    recommendation, text = run_rag_demo._recommend_strategy({
        COLLECTION_FIXED: {
            "collection_name": COLLECTION_FIXED,
            "average_precision": 0.6,
            "average_recall": 0.6,
            "average_f1": 0.6,
        },
        COLLECTION_SENTENCE: {
            "collection_name": COLLECTION_SENTENCE,
            "average_precision": 0.4,
            "average_recall": 0.9,
            "average_f1": 0.5,
        },
    })

    assert recommendation == COLLECTION_FIXED
    assert "macro f1 0.6000 versus 0.5000" in text
