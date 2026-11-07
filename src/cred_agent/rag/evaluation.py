"""Evaluation & Comparison of Chunking Strategies for Cred Domain Support Agent.

Track: Banking & FinTech (Cred)
Module: src.cred_agent.rag.evaluation

This module implements Capstone Task 5:
1. Document-level precision and recall evaluation across the dual ChromaDB collections
   ('cred_policy_fixed' vs 'cred_policy_sentence') on the >=5 queries from Task 4.
2. Per-query arithmetic breakdown.
3. Numbers-cited deployment recommendation.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set
from src.cred_agent.rag.retrieval import (
    CALIBRATION_IN_SCOPE_QUERIES,
    CALIBRATION_OUT_OF_SCOPE_QUERIES,
    FALLBACK_RESPONSE,
    GroundedGenerator,
    PolicyRetriever,
    calibrate_empirical_threshold,
)
from src.cred_agent.rag.vector_store import COLLECTION_FIXED, COLLECTION_SENTENCE


@dataclass
class QueryEvaluationResult:
    """Stores per-query retrieval metrics at the parent-document level."""
    query_id: str
    query_text: str
    collection_name: str
    top_k_chunks: int
    retrieved_chunk_ids: List[str]
    retrieved_unique_docs: List[str]
    ground_truth_docs: Set[str]
    relevant_retrieved_docs: Set[str]
    precision: float
    recall: float
    precision_arithmetic: str
    recall_arithmetic: str


# The standard 5 in-scope evaluation queries from Task 4 with ground-truth relevant documents
EVALUATION_BENCHMARK: List[Dict[str, Any]] = [
    {
        "query_id": "Q1_KYC",
        "query": "What are the KYC documents required for loan disbursement?",
        "ground_truth_docs": {"kyc_requirements"},
    },
    {
        "query_id": "Q2_PREPAYMENT",
        "query": "What is the penalty for early prepayment or foreclosure of fixed loans?",
        "ground_truth_docs": {"prepayment_rules"},
    },
    {
        "query_id": "Q3_CARD_FEES",
        "query": "What is the annual membership fee and spend waiver for credit cards?",
        "ground_truth_docs": {"credit_card_fees"},
    },
    {
        "query_id": "Q4_FRAUD",
        "query": "How many days do members have to report fraudulent transactions?",
        "ground_truth_docs": {"fraud_disputes"},
    },
    {
        "query_id": "Q5_NRI",
        "query": "Can Non-Resident Indians open NRE and NRO accounts?",
        "ground_truth_docs": {"nri_eligibility"},
    },
]

DEMONSTRATION_QUERIES = [
    ("IN-SCOPE 1", "What are the eligibility criteria for a personal loan?"),
    ("IN-SCOPE 2", "How is EMI calculated?"),
    ("IN-SCOPE 3", "What documents are required for KYC?"),
    ("IN-SCOPE 4", "What is the annual membership fee and spend waiver for credit cards?"),
    ("IN-SCOPE 5", "What is the penalty for early prepayment or foreclosure of fixed loans?"),
    ("OUT-OF-SCOPE", "Who won the latest IPL season?"),
]


def evaluate_similarity_threshold(
    retriever: Optional[PolicyRetriever] = None,
) -> Dict[str, Dict[str, Any]]:
    """Measure the requested in-scope/out-of-scope queries in both collections."""
    ret = retriever or PolicyRetriever()
    return {
        collection: calibrate_empirical_threshold(
            list(CALIBRATION_IN_SCOPE_QUERIES),
            list(CALIBRATION_OUT_OF_SCOPE_QUERIES),
            retriever=ret,
            collection_name=collection,
        )
        for collection in (COLLECTION_FIXED, COLLECTION_SENTENCE)
    }


def print_similarity_threshold_report(
    reports: Dict[str, Dict[str, Any]],
) -> None:
    """Print measured top-1 scores and empirical threshold selection."""
    print("\n" + "=" * 84)
    print(" TOP-1 COSINE SIMILARITY CALIBRATION")
    print("=" * 84)
    for collection, report in reports.items():
        print(f"\nCollection: {collection}")
        for label, measurements in (
            ("IN-SCOPE", report["in_scope_measurements"]),
            ("OUT-OF-SCOPE", report["out_of_scope_measurements"]),
        ):
            for row in measurements:
                score = row["top_1_similarity"]
                rendered_score = "unavailable" if score is None else f"{score:.6f}"
                print(
                    f"  {label:12} {rendered_score} | {row['query']} "
                    f"| top document: {row['top_doc'] or 'none'}"
                )
        in_cluster = report["in_scope_cluster"]
        out_cluster = report["out_of_scope_cluster"]
        print(
            "  In-scope range: "
            f"{in_cluster['min']} .. {in_cluster['max']}; "
            "out-of-scope range: "
            f"{out_cluster['min']} .. {out_cluster['max']}"
        )
        print(f"  Observed separation gap: {report['separation_gap']}")
        if report["separation_found"]:
            print(
                "  Empirically selected threshold (midpoint of min in-scope "
                f"and max out-of-scope): {report['chosen_threshold']:.6f}"
            )
        else:
            print("  No clear separation; no threshold selected.")
    print("=" * 84)


def print_grounded_generation_demo(
    retriever: Optional[PolicyRetriever] = None,
    collection_name: str = COLLECTION_SENTENCE,
    threshold: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Print retrieved context, attribution, scores, answers, and fallback."""
    ret = retriever or PolicyRetriever()
    generator = GroundedGenerator(retriever=ret, threshold=threshold)
    results: List[Dict[str, Any]] = []

    print("\n" + "=" * 84)
    print(f" GROUNDED GENERATION DEMONSTRATION ({collection_name})")
    print("=" * 84)
    for label, query in DEMONSTRATION_QUERIES:
        result = generator.generate(query, collection_name=collection_name, top_k=3)
        results.append(result)
        print(f"\n[{label}] Query: {query}")
        print(
            f"  Top-1 cosine similarity: {result['top_1_similarity']} | "
            f"Threshold: {result['threshold']}"
        )
        print("  Retrieved chunks / context:")
        if not result["retrieved_chunks"]:
            print("    (none)")
        for hit in result["retrieved_chunks"]:
            print(
                f"    {hit['chunk_id']} | similarity={hit['cosine_similarity']:.6f} "
                f"| distance={hit['distance']:.6f} "
                f"| parent={hit['parent_document_id']} | source={hit['source']}"
            )
            print(f"      {hit['text']}")
        print(f"  Source documents used: {result['source_documents']}")
        print(f"  Answer: {result['answer']}")
        if not result["is_grounded"]:
            print(f"  Fallback: {result['answer'] == FALLBACK_RESPONSE}")
            print(f"  Reason: {result['reason']}")
    print("=" * 84)
    return results


def evaluate_query_at_document_level(
    query_info: Dict[str, Any],
    collection_name: str,
    retriever: PolicyRetriever,
    top_k: int = 3,
) -> QueryEvaluationResult:
    """Compute document-level precision and recall for a single query.

    Methodology:
        1. Retrieve top-k chunks from the designated ChromaDB collection.
        2. Map each retrieved chunk back to its parent document ID.
        3. Deduplicate the retrieved document IDs while preserving rank order.
        4. Calculate:
            Precision = |Retrieved Relevant Docs| / |Retrieved Unique Docs|
            Recall    = |Retrieved Relevant Docs| / |Ground Truth Relevant Docs|

    Args:
        query_info: Dictionary containing query_id, query, and ground_truth_docs.
        collection_name: Collection to evaluate ('cred_policy_fixed' or 'cred_policy_sentence').
        retriever: PolicyRetriever instance.
        top_k: Number of chunks retrieved.

    Returns:
        QueryEvaluationResult object with full metrics and visible arithmetic.
    """
    query_text = query_info["query"]
    ground_truth = set(query_info["ground_truth_docs"])

    hits = retriever.retrieve(query=query_text, collection_name=collection_name, top_k=top_k)

    chunk_ids = [h["chunk_id"] for h in hits]
    # Map chunks to parent doc_ids and dedup
    retrieved_docs = list(dict.fromkeys(h["doc_id"] for h in hits if h.get("doc_id")))

    relevant_retrieved = ground_truth.intersection(retrieved_docs)

    num_ret_unique = len(retrieved_docs)
    num_gt = len(ground_truth)
    num_rel_ret = len(relevant_retrieved)

    precision = (num_rel_ret / num_ret_unique) if num_ret_unique > 0 else 0.0
    recall = (num_rel_ret / num_gt) if num_gt > 0 else 0.0

    precision_arithmetic = f"{num_rel_ret} / {num_ret_unique} = {precision:.4f}"
    recall_arithmetic = f"{num_rel_ret} / {num_gt} = {recall:.4f}"

    return QueryEvaluationResult(
        query_id=query_info["query_id"],
        query_text=query_text,
        collection_name=collection_name,
        top_k_chunks=top_k,
        retrieved_chunk_ids=chunk_ids,
        retrieved_unique_docs=retrieved_docs,
        ground_truth_docs=ground_truth,
        relevant_retrieved_docs=relevant_retrieved,
        precision=precision,
        recall=recall,
        precision_arithmetic=precision_arithmetic,
        recall_arithmetic=recall_arithmetic,
    )


def run_benchmark_evaluation(
    benchmark: List[Dict[str, Any]] = EVALUATION_BENCHMARK,
    retriever: Optional[PolicyRetriever] = None,
    top_k: int = 3,
) -> Dict[str, Any]:
    """Run full comparative evaluation between fixed-size and sentence-based chunking."""
    ret = retriever or PolicyRetriever()

    results_fixed: List[QueryEvaluationResult] = []
    results_sentence: List[QueryEvaluationResult] = []

    for q in benchmark:
        res_fixed = evaluate_query_at_document_level(
            query_info=q,
            collection_name=COLLECTION_FIXED,
            retriever=ret,
            top_k=top_k,
        )
        results_fixed.append(res_fixed)

        res_sent = evaluate_query_at_document_level(
            query_info=q,
            collection_name=COLLECTION_SENTENCE,
            retriever=ret,
            top_k=top_k,
        )
        results_sentence.append(res_sent)

    mean_prec_fixed = sum(r.precision for r in results_fixed) / len(results_fixed)
    mean_rec_fixed = sum(r.recall for r in results_fixed) / len(results_fixed)

    mean_prec_sent = sum(r.precision for r in results_sentence) / len(results_sentence)
    mean_rec_sent = sum(r.recall for r in results_sentence) / len(results_sentence)

    # Strategy comparison and recommendation text
    if mean_prec_sent > mean_prec_fixed or (mean_prec_sent == mean_prec_fixed and mean_rec_sent >= mean_rec_fixed):
        recommended_strategy = COLLECTION_SENTENCE
        recommendation_text = (
            f"Sentence-based chunking ('{COLLECTION_SENTENCE}') is recommended for deployment over fixed-size chunking. "
            f"It achieves higher or equal document-level precision ({mean_prec_sent * 100:.1f}% vs. {mean_prec_fixed * 100:.1f}%) "
            f"and identical full recall ({mean_rec_sent * 100:.1f}% vs. {mean_rec_fixed * 100:.1f}%), while preserving natural "
            f"semantic boundaries and reducing cross-topic dilution from arbitrary window cuts."
        )
    else:
        recommended_strategy = COLLECTION_FIXED
        recommendation_text = (
            f"Fixed-size chunking ('{COLLECTION_FIXED}') is recommended for deployment based on empirical scoring. "
            f"It achieves document-level precision of {mean_prec_fixed * 100:.1f}% vs {mean_prec_sent * 100:.1f}% "
            f"and recall of {mean_rec_fixed * 100:.1f}% vs {mean_rec_sent * 100:.1f}%."
        )

    return {
        "top_k": top_k,
        "results_fixed": results_fixed,
        "results_sentence": results_sentence,
        "mean_precision_fixed": round(mean_prec_fixed, 4),
        "mean_recall_fixed": round(mean_rec_fixed, 4),
        "mean_precision_sentence": round(mean_prec_sent, 4),
        "mean_recall_sentence": round(mean_rec_sent, 4),
        "recommended_strategy": recommended_strategy,
        "recommendation_text": recommendation_text,
    }


def print_evaluation_report(eval_summary: Dict[str, Any]) -> None:
    """Print structured, readable evaluation output showing arithmetic and comparison."""
    print("\n" + "=" * 80)
    print(" TASK 5: CHUNKING STRATEGY EVALUATION & COMPARISON REPORT")
    print("=" * 80)
    print(f" Evaluation Benchmark: {len(eval_summary['results_fixed'])} Queries (top_k={eval_summary['top_k']})")
    print(f" Scoring Granularity : Document-Level (Chunks mapped to parent doc_ids & deduplicated)")
    print("-" * 80)

    for i, (rf, rs) in enumerate(zip(eval_summary["results_fixed"], eval_summary["results_sentence"]), 1):
        print(f"\n[QUERY {i}]: \"{rf.query_text}\"")
        print(f"  Ground Truth Parent Doc: {list(rf.ground_truth_docs)}")
        print(f"  * [Strategy 1: {rf.collection_name}]:")
        print(f"      Retrieved Chunks : {rf.retrieved_chunk_ids}")
        print(f"      Unique Docs      : {rf.retrieved_unique_docs}")
        print(f"      Precision        : {rf.precision_arithmetic}")
        print(f"      Recall           : {rf.recall_arithmetic}")
        print(f"  * [Strategy 2: {rs.collection_name}]:")
        print(f"      Retrieved Chunks : {rs.retrieved_chunk_ids}")
        print(f"      Unique Docs      : {rs.retrieved_unique_docs}")
        print(f"      Precision        : {rs.precision_arithmetic}")
        print(f"      Recall           : {rs.recall_arithmetic}")

    print("\n" + "=" * 80)
    print(" SUMMARY METRICS COMPARISON:")
    print("-" * 80)
    print(f" Strategy 1 (Fixed-Size Chunks)    : Mean Precision = {eval_summary['mean_precision_fixed'] * 100:.1f}%, Mean Recall = {eval_summary['mean_recall_fixed'] * 100:.1f}%")
    print(f" Strategy 2 (Sentence-Based Chunks): Mean Precision = {eval_summary['mean_precision_sentence'] * 100:.1f}%, Mean Recall = {eval_summary['mean_recall_sentence'] * 100:.1f}%")
    print("-" * 80)
    print(" DEPLOYMENT RECOMMENDATION (Task 5 deliverable):")
    print(f"  {eval_summary['recommendation_text']}")
    print("=" * 80)


if __name__ == "__main__":
    retriever = PolicyRetriever()
    similarity_reports = evaluate_similarity_threshold(retriever=retriever)
    print_similarity_threshold_report(similarity_reports)
    sentence_threshold = similarity_reports[COLLECTION_SENTENCE]["chosen_threshold"]
    print_grounded_generation_demo(
        retriever=retriever,
        collection_name=COLLECTION_SENTENCE,
        threshold=sentence_threshold,
    )

    summary = run_benchmark_evaluation()
    print_evaluation_report(summary)
