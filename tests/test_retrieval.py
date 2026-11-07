"""Tests for cosine retrieval, attribution, and extractive grounded answers."""

import json
from pathlib import Path
from typing import List

import pytest

from src.cred_agent.rag.chunking import Chunk, chunk_fixed_size
from src.cred_agent.rag.evaluation import evaluate_similarity_threshold, print_similarity_threshold_report
from src.cred_agent.rag.retrieval import (
    FALLBACK_RESPONSE,
    GroundedGenerator,
    PolicyRetriever,
    calibrate_empirical_threshold,
)
from src.cred_agent.rag.vector_store import (
    COLLECTION_FIXED,
    COLLECTION_SENTENCE,
    ChromaVectorStore,
)
from src.cred_agent.services.response_composition import compose_specialist_findings


class KeywordEmbedder:
    """Tiny deterministic embedder for exercising Chroma without model downloads."""

    dimension = 4
    model_name = "test-keyword-embedder"

    def encode_text(self, text: str) -> List[float]:
        lowered = text.lower()
        if "kyc" in lowered:
            return [0.0, 0.0, 1.0, 0.0]
        if "emi" in lowered or "calculation" in lowered or "calculated" in lowered:
            return [0.0, 1.0, 0.0, 0.0]
        if "loan" in lowered or "eligibility" in lowered:
            return [1.0, 0.0, 0.0, 0.0]
        return [0.0, 0.0, 0.0, 1.0]

    def encode_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        return [self.encode_text(text) for text in texts]


def _indexed_store(path: Path) -> ChromaVectorStore:
    store = ChromaVectorStore(persist_directory=path, embedder=KeywordEmbedder())
    fixed_chunks = [
        Chunk("fixed_loan_0", "loan_eligibility", "loan_eligibility.md", "fixed_size", 0,
              "Personal loan eligibility requires Indian residency."),
        Chunk("fixed_emi_0", "emi_calculation", "emi_calculation.md", "fixed_size", 0,
              "EMI is calculated using the reducing balance method."),
        Chunk("fixed_kyc_0", "kyc_requirements", "kyc_requirements.md", "fixed_size", 0,
              "KYC requires PAN and address documents."),
    ]
    sentence_chunks = [
        Chunk("sent_loan_0", "loan_eligibility", "loan_eligibility.md", "sentence_based", 0,
              "Personal loan eligibility requires Indian residency."),
        Chunk("sent_emi_0", "emi_calculation", "emi_calculation.md", "sentence_based", 0,
              "EMI is calculated using the reducing balance method."),
        Chunk("sent_kyc_0", "kyc_requirements", "kyc_requirements.md", "sentence_based", 0,
              "KYC requires PAN and address documents."),
    ]
    store.index_fixed_size_chunks(fixed_chunks)
    store.index_sentence_based_chunks(sentence_chunks)
    return store


def test_retrieves_relevant_chunks_with_attribution_from_both_collections(tmp_path: Path):
    retriever = PolicyRetriever(_indexed_store(tmp_path / "chroma"))

    for collection in (COLLECTION_FIXED, COLLECTION_SENTENCE):
        hits = retriever.retrieve("What are the eligibility criteria for a personal loan?",
                                  collection_name=collection, top_k=1)

        assert len(hits) == 1
        assert hits[0]["text"] == "Personal loan eligibility requires Indian residency."
        assert hits[0]["parent_document_id"] == "loan_eligibility"
        assert hits[0]["doc_id"] == "loan_eligibility"
        assert hits[0]["chunk_id"].startswith("fixed_" if collection == COLLECTION_FIXED else "sent_")
        assert hits[0]["source"] == "loan_eligibility.md"
        assert hits[0]["cosine_similarity"] == pytest.approx(1.0)


def test_top_k_returns_requested_number_in_relevance_order(tmp_path: Path):
    retriever = PolicyRetriever(_indexed_store(tmp_path / "chroma"))
    hits = retriever.retrieve("personal loan eligibility", collection_name=COLLECTION_FIXED, top_k=2)

    assert len(hits) == 2
    assert hits[0]["parent_document_id"] == "loan_eligibility"
    assert hits[0]["cosine_similarity"] >= hits[1]["cosine_similarity"]


class FakeCollection:
    name = COLLECTION_FIXED
    metadata = {"hnsw:space": "cosine"}

    def count(self):
        return 3

    def query(self, **kwargs):
        return {
            "ids": [["chunk-a", "chunk-b", "chunk-c"]],
            "documents": [["A.", "B.", "C."]],
            "metadatas": [
                [
                    {"parent_document_id": "doc-a", "source": "a.md"},
                    {"parent_document_id": "doc-b", "source": "b.md"},
                    {"parent_document_id": "doc-c", "source": "c.md"},
                ]
            ],
            "distances": [[0.8, 0.25, 0.5]],
        }


def test_cosine_distance_is_converted_to_similarity_and_sorted():
    store = ChromaVectorStore.__new__(ChromaVectorStore)
    store.embedder = KeywordEmbedder()
    store.collection_fixed = FakeCollection()

    hits = store.query("loan eligibility", COLLECTION_FIXED, top_k=3)

    assert [hit["chunk_id"] for hit in hits] == ["chunk-b", "chunk-c", "chunk-a"]
    assert [hit["cosine_similarity"] for hit in hits] == pytest.approx([0.75, 0.5, 0.2])
    assert hits[0]["distance"] == pytest.approx(0.25)
    assert hits[0]["similarity_score"] == pytest.approx(0.75)
    assert hits[0]["source"] == "b.md"
    assert hits[0]["parent_document_id"] == "doc-b"


def test_non_cosine_collection_is_rejected_with_safe_reindex_guidance():
    collection = FakeCollection()
    collection.metadata = {"hnsw:space": "l2"}

    with pytest.raises(RuntimeError, match="new collection or persistence directory"):
        ChromaVectorStore._require_cosine_metric(collection)


def test_out_of_scope_query_uses_exact_fallback(tmp_path: Path):
    retriever = PolicyRetriever(_indexed_store(tmp_path / "chroma"))
    generator = GroundedGenerator(retriever=retriever, threshold=0.5)

    result = generator.generate("Who won the latest IPL season?", COLLECTION_SENTENCE, top_k=2)

    assert result["answer"] == FALLBACK_RESPONSE
    assert not result["is_grounded"]
    assert result["source_documents"] == []
    assert result["retrieved_context"] == result["retrieved_chunks"]
    assert result["similarity_measurements"]


def test_answer_is_extractive_and_cites_its_source(tmp_path: Path):
    retriever = PolicyRetriever(_indexed_store(tmp_path / "chroma"))
    generator = GroundedGenerator(retriever=retriever, threshold=0.5)
    result = generator.generate(
        "How is EMI calculated?", COLLECTION_SENTENCE, top_k=2
    )

    retrieved_text = " ".join(hit["text"] for hit in result["retrieved_chunks"])
    assert result["is_grounded"]
    assert result["answer"] in retrieved_text
    assert result["answer"] == "EMI is calculated using the reducing balance method."
    assert result["source_documents"] == ["emi_calculation"]
    assert result["retrieved_chunks"][0]["source"] == "emi_calculation.md"
    assert result["top_1_similarity"] == pytest.approx(1.0)


def test_overlapping_fixed_chunks_compose_one_complete_grounded_sentence():
    sentence = (
        "Borrowers may select tenures from 6 to 60 months, with payments "
        "auto-debited on the 5th calendar day of each month via registered "
        "NACH or UPI auto-pay channels."
    )
    document_text = ("General policy detail. " * 20) + sentence
    chunks = chunk_fixed_size(
        document_text,
        parent_document_id="emi_calculation",
        source="emi_calculation.md",
    )
    assert len(chunks) == 2
    assert not chunks[0].text.endswith(".")
    assert not chunks[1].text.startswith("Borrowers")

    class OverlappingChunkRetriever:
        def retrieve(self, query, collection_name, top_k):
            return [
                {
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "parent_document_id": chunk.parent_document_id,
                    "source": chunk.source,
                    "chunking_strategy": chunk.chunking_strategy,
                    "metadata": {"chunk_index": chunk.chunk_index},
                    "cosine_similarity": 0.9,
                    "distance": 0.1,
                }
                for chunk in reversed(chunks)
            ][:top_k]

    result = GroundedGenerator(
        retriever=OverlappingChunkRetriever(),
        threshold=0.286211,
    ).generate(
        "What tenure options are available?",
        collection_name=COLLECTION_FIXED,
        top_k=2,
    )

    assert result["is_grounded"]
    assert result["answer"] == sentence
    assert result["answer"].count("Borrowers may select tenures") == 1
    assert result["source_documents"] == ["emi_calculation"]
    assert len(result["retrieved_chunks"]) == 2
    assert result["retrieved_chunks"][0]["text"] == chunks[1].text

    composed = compose_specialist_findings(
        "Specialist findings: "
        + json.dumps(
            {
                "answer": result["answer"],
                "is_grounded": result["is_grounded"],
                "source_documents": result["source_documents"],
                "retrieved_chunks": result["retrieved_chunks"],
            }
        )
    )
    assert composed == (
        sentence + "\nSource: emi_calculation (emi_calculation.md)."
    )


@pytest.mark.parametrize(
    ("query", "source_id", "required_answer_details"),
    [
        (
            "What annual fee and interest-free period apply to Cred credit cards?",
            "credit_card_fees",
            ("45 days", "interest-free"),
        ),
        (
            "What identity and address documents are required for KYC?",
            "kyc_requirements",
            ("PAN", "Aadhaar", "Passport"),
        ),
        (
            "How should I report an unauthorized transaction and what is the investigation timeframe?",
            "fraud_disputes",
            ("7 to 10 business days",),
        ),
        (
            "Which factors influence a CIBIL score, especially repayment history?",
            "credit_score",
            ("30%", "Credit utilization"),
        ),
        (
            "How many people can open a joint account and how are loan repayments shared?",
            "joint_accounts",
            ("up to three",),
        ),
        (
            "Can NRIs open NRE and NRO accounts, and what are they used for?",
            "nri_eligibility",
            ("foreign income", "rupee income"),
        ),
    ],
)
def test_retrieved_supporting_sentences_are_not_dropped_during_composition(
    query: str,
    source_id: str,
    required_answer_details: tuple[str, ...],
):
    from scripts.run_rag_demo import load_persisted_retriever

    retriever = load_persisted_retriever()
    result = GroundedGenerator(
        retriever=retriever,
        threshold=0.286211,
    ).generate(
        query,
        collection_name=COLLECTION_FIXED,
        top_k=3,
    )

    assert result["is_grounded"]
    assert source_id in result["source_documents"]
    eligible_source_chunks = [
        hit
        for hit in result["retrieved_chunks"]
        if hit["parent_document_id"] == source_id
        and hit["cosine_similarity"] >= 0.286211
    ]
    assert eligible_source_chunks
    for detail in required_answer_details:
        assert detail.casefold() in result["answer"].casefold()
        assert any(
            detail.casefold() in hit["text"].casefold()
            for hit in eligible_source_chunks
        )

    composed = compose_specialist_findings(
        "Specialist findings: "
        + json.dumps(
            {
                "answer": result["answer"],
                "is_grounded": result["is_grounded"],
                "source_documents": result["source_documents"],
                "retrieved_chunks": result["retrieved_chunks"],
            },
            ensure_ascii=False,
        )
    )
    assert composed is not None
    for detail in required_answer_details:
        assert detail.casefold() in composed.casefold()
    assert source_id in composed


def test_similarity_evaluation_prints_scores_and_empirical_threshold(
    tmp_path: Path, capsys
):
    retriever = PolicyRetriever(_indexed_store(tmp_path / "chroma"))
    reports = evaluate_similarity_threshold(retriever)
    print_similarity_threshold_report(reports)

    captured = capsys.readouterr().out
    assert "TOP-1 COSINE SIMILARITY CALIBRATION" in captured
    assert "0.000000" in captured
    assert reports[COLLECTION_FIXED]["separation_found"]
    assert reports[COLLECTION_FIXED]["chosen_threshold"] == pytest.approx(0.5)
    assert reports[COLLECTION_SENTENCE]["separation_found"]


def test_calibration_does_not_invent_threshold_without_separation():
    class OverlappingRetriever:
        def retrieve(self, query, collection_name, top_k):
            return [{
                "cosine_similarity": 0.4,
                "parent_document_id": "some_policy",
            }]

    report = calibrate_empirical_threshold(
        ["in-scope query"],
        ["out-of-scope query"],
        retriever=OverlappingRetriever(),
    )

    assert not report["separation_found"]
    assert report["separation_gap"] == pytest.approx(0.0)
    assert report["chosen_threshold"] is None
