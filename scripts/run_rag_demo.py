"""Run deterministic RAG demonstrations and document-level evaluation."""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import chromadb
from chromadb.config import Settings

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cred_agent.config import DEFAULT_CHROMA_PATH
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

logger = logging.getLogger(__name__)

IN_SCOPE_QUERIES: List[Dict[str, Any]] = [
    {
        "query_id": "personal_loan_eligibility",
        "query": "What are the eligibility criteria for a personal loan?",
        "ground_truth_docs": {"loan_eligibility"},
    },
    {
        "query_id": "emi_calculation",
        "query": "How is EMI calculated?",
        "ground_truth_docs": {"emi_calculation"},
    },
    {
        "query_id": "kyc_documents",
        "query": "What documents are required for KYC?",
        "ground_truth_docs": {"kyc_requirements"},
    },
    {
        "query_id": "prepayment_penalty",
        "query": "What is the penalty for early prepayment or foreclosure of fixed loans?",
        "ground_truth_docs": {"prepayment_rules"},
    },
    {
        "query_id": "credit_scores",
        "query": "How does payment history affect credit scores?",
        "ground_truth_docs": {"credit_score"},
    },
]
OUT_OF_SCOPE_QUERIES = [
    "Who won the latest IPL season?",
    "What is the weather forecast for Delhi?",
]
COLLECTIONS = (COLLECTION_FIXED, COLLECTION_SENTENCE)
DEFAULT_REPORT_PATH = PROJECT_ROOT / "evaluation" / "rag_evaluation_report.md"
DEFAULT_PERSIST_DIRECTORY = (PROJECT_ROOT / DEFAULT_CHROMA_PATH).resolve()


class MissingVectorCollectionsError(RuntimeError):
    """Raised when persisted Chroma storage lacks required RAG collections."""


def load_persisted_retriever(
    persist_directory: Path = DEFAULT_PERSIST_DIRECTORY,
) -> PolicyRetriever:
    """Open existing collections without silently creating missing indexes."""
    directory = Path(persist_directory).resolve()
    sqlite_path = directory / "chroma.sqlite3"
    if not directory.is_dir() or not sqlite_path.is_file():
        raise FileNotFoundError(
            f"ChromaDB persistence directory or database is missing: {directory}"
        )

    client = chromadb.PersistentClient(
        path=str(directory),
        settings=Settings(anonymized_telemetry=False),
    )
    existing_names = {
        collection.name if hasattr(collection, "name") else str(collection)
        for collection in client.list_collections()
    }
    missing = sorted(set(COLLECTIONS) - existing_names)
    if missing:
        raise MissingVectorCollectionsError(
            f"Missing required ChromaDB collection(s) in {directory}: "
            f"{', '.join(missing)}. Run scripts/build_index.py to create them."
        )

    store = ChromaVectorStore(persist_directory=directory)
    return PolicyRetriever(vector_store=store)


def _document_metrics(
    query_info: Dict[str, Any],
    hits: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Calculate precision and recall after deduplicating retrieved parents."""
    retrieved_documents = list(dict.fromkeys(
        hit.get("parent_document_id") or hit.get("doc_id")
        for hit in hits
        if hit.get("parent_document_id") or hit.get("doc_id")
    ))
    ground_truth = set(query_info["ground_truth_docs"])
    relevant = sorted(ground_truth.intersection(retrieved_documents))
    relevant_count = len(relevant)
    retrieved_count = len(retrieved_documents)
    ground_truth_count = len(ground_truth)
    precision = relevant_count / retrieved_count if retrieved_count else 0.0
    recall = relevant_count / ground_truth_count if ground_truth_count else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "query_id": query_info["query_id"],
        "query": query_info["query"],
        "ground_truth_docs": sorted(ground_truth),
        "retrieved_chunk_ids": [hit["chunk_id"] for hit in hits],
        "retrieved_unique_docs": retrieved_documents,
        "relevant_retrieved_docs": relevant,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "precision_arithmetic": f"{relevant_count} / {retrieved_count} = {precision:.4f}",
        "recall_arithmetic": f"{relevant_count} / {ground_truth_count} = {recall:.4f}",
    }


def _evaluate_collection(
    retriever: PolicyRetriever,
    collection_name: str,
) -> Dict[str, Any]:
    """Calculate document-level Precision@3, Recall@3, and macro averages."""
    rows = []
    for query_info in IN_SCOPE_QUERIES:
        hits = retriever.retrieve(
            query_info["query"],
            collection_name=collection_name,
            top_k=3,
        )
        rows.append(_document_metrics(query_info, hits))
    count = len(rows)
    average_precision = sum(row["precision"] for row in rows) / count
    average_recall = sum(row["recall"] for row in rows) / count
    average_f1 = sum(row["f1"] for row in rows) / count
    return {
        "collection_name": collection_name,
        "queries": rows,
        "average_precision": average_precision,
        "average_recall": average_recall,
        "average_f1": average_f1,
    }


def _recommend_strategy(
    evaluations: Dict[str, Dict[str, Any]],
) -> Tuple[str, str]:
    """Recommend by macro F1; use recall then precision as deterministic tie-breaks."""
    fixed = evaluations[COLLECTION_FIXED]
    sentence = evaluations[COLLECTION_SENTENCE]
    ordered = sorted(
        (fixed, sentence),
        key=lambda result: (
            result["average_f1"],
            result["average_recall"],
            result["average_precision"],
        ),
        reverse=True,
    )
    winner = ordered[0]
    runner_up = ordered[1]
    metric_names = ("average_f1", "average_recall", "average_precision")
    comparison = next(
        (
            metric
            for metric in metric_names
            if abs(winner[metric] - runner_up[metric]) > 1e-12
        ),
        None,
    )
    if comparison is None:
        return (
            winner["collection_name"],
            "Both strategies tie on macro F1, recall, and precision; "
            "there is no metric-based preference.",
        )
    metric_label = comparison.removeprefix("average_")
    return (
        winner["collection_name"],
        f"{winner['collection_name']} is recommended: macro {metric_label} "
        f"{winner[comparison]:.4f} versus {runner_up[comparison]:.4f} for "
        f"{runner_up['collection_name']}. Macro Precision@3 is "
        f"{winner['average_precision']:.4f} versus "
        f"{runner_up['average_precision']:.4f}, and macro Recall@3 is "
        f"{winner['average_recall']:.4f} versus "
        f"{runner_up['average_recall']:.4f}.",
    )


def _fallback_result(
    query: str,
    collection_name: str,
    hits: List[Dict[str, Any]],
    reason: str,
) -> Dict[str, Any]:
    """Return a fallback when calibration failed, without triggering auto-calibration."""
    return {
        "query": query,
        "answer": FALLBACK_RESPONSE,
        "is_grounded": False,
        "top_1_similarity": hits[0]["cosine_similarity"] if hits else None,
        "threshold": None,
        "collection_used": collection_name,
        "retrieved_chunks": hits,
        "retrieved_context": hits,
        "similarity_measurements": [
            {
                "chunk_id": hit["chunk_id"],
                "cosine_similarity": hit["cosine_similarity"],
                "distance": hit["distance"],
            }
            for hit in hits
        ],
        "source_documents": [],
        "retrieved_source_documents": list(dict.fromkeys(
            hit["parent_document_id"]
            for hit in hits
            if hit.get("parent_document_id")
        )),
        "reason": reason,
    }


def _format_hit(hit: Dict[str, Any]) -> str:
    source_name = Path(hit["source"]).name if hit.get("source") else "unknown"
    return (
        f"- **{hit['chunk_id']}** — cosine similarity "
        f"`{hit['cosine_similarity']:.6f}`; parent document "
        f"`{hit['parent_document_id']}`; source `{source_name}`\n"
        f"  - Retrieved context: {hit['text']}"
    )


def render_report(result: Dict[str, Any]) -> str:
    """Render the complete calibration, demo, and document-level evaluation."""
    lines = [
        "# RAG Retrieval and Grounded Generation Evaluation",
        "",
        "This report was generated by `scripts/run_rag_demo.py` using the local "
        "SentenceTransformer model and persisted ChromaDB collections. No external "
        "LLM API is used; generated answers are deterministic and extractive.",
        "",
        f"Persistent index: `{result['persist_directory']}`.",
        "",
        "## Part 1, Task 4: Grounded-generation demonstration",
        "",
        "Five in-scope questions and two out-of-scope questions are measured and "
        "run against both chunking collections. Each question shows the top three "
        "retrieved chunks, cosine similarity, parent document, source filename, "
        "retrieved text, and the grounded answer or fallback.",
        "",
        "### Empirical threshold calibration",
        "",
        "Calibration uses all five in-scope demo questions and both out-of-scope "
        "questions per collection. A threshold is the midpoint between the lowest "
        "in-scope and highest out-of-scope top-1 cosine similarities only when "
        "the in-scope minimum is strictly greater than the out-of-scope maximum. "
        "Overlapping or incomplete measurements are reported as calibration "
        "failure and do not produce a threshold.",
        "",
    ]
    for collection_name, calibration in result["calibration"].items():
        lines.extend([
            f"#### `{collection_name}`",
            "",
            "| Scope | Query | Top-1 cosine similarity | Top parent document |",
            "|---|---|---:|---|",
        ])
        for scope, rows in (
            ("In-scope", calibration["in_scope_measurements"]),
            ("Out-of-scope", calibration["out_of_scope_measurements"]),
        ):
            for row in rows:
                score = row["top_1_similarity"]
                rendered_score = "unavailable" if score is None else f"{score:.6f}"
                lines.append(
                    f"| {scope} | {row['query']} | {rendered_score} | "
                    f"{row['top_doc'] or 'none'} |"
                )
        lines.extend([
            "",
            f"- In-scope range: `{calibration['in_scope_cluster']['min']}` to "
            f"`{calibration['in_scope_cluster']['max']}`.",
            f"- Out-of-scope range: `{calibration['out_of_scope_cluster']['min']}` "
            f"to `{calibration['out_of_scope_cluster']['max']}`.",
            f"- Separation gap: `{calibration['separation_gap']}`.",
            "- Selection: "
            + (
                f"threshold `{calibration['chosen_threshold']:.6f}` "
                "(empirical midpoint)."
                if calibration["separation_found"]
                else "**calibration failure; no threshold selected.**"
            ),
            "",
        ])

    for demo in result["demonstrations"]:
        status = "grounded answer" if demo["result"]["is_grounded"] else "fallback"
        lines.extend([
            f"### {demo['scope']} — {demo['query']}",
            "",
            f"Collection: `{demo['collection_name']}`; top-1 similarity: "
            f"`{demo['result']['top_1_similarity']}`; threshold: "
            f"`{demo['result']['threshold']}`; outcome: **{status}**.",
            "",
        ])
        lines.extend(_format_hit(hit) for hit in demo["result"]["retrieved_chunks"])
        if not demo["result"]["retrieved_chunks"]:
            lines.append("- No chunks retrieved.")
        lines.extend([
            "",
            f"**Answer:** {demo['result']['answer']}",
            "",
            f"**Source documents used:** "
            f"{', '.join(demo['result']['source_documents']) or 'none'}",
            "",
        ])

    lines.extend([
        "## Part 1, Task 5: Document-level Precision@3 and Recall@3",
        "",
        "Ground truth is declared per question below. Retrieved chunks are mapped "
        "to parent document IDs and deduplicated before precision is calculated. "
        "Precision@3 is relevant retrieved parent documents divided by unique "
        "retrieved parent documents; Recall@3 is relevant retrieved parent "
        "documents divided by ground-truth parent documents.",
        "",
    ])
    for collection_name, evaluation in result["evaluations"].items():
        lines.extend([
            f"### `{collection_name}`",
            "",
            "| Query | Ground-truth parent document(s) | Retrieved chunk IDs | "
            "Unique retrieved parents | Relevant parents | Precision@3 arithmetic | "
            "Recall@3 arithmetic |",
            "|---|---|---|---|---|---:|---:|",
        ])
        for row in evaluation["queries"]:
            lines.append(
                f"| {row['query']} | {', '.join(row['ground_truth_docs'])} | "
                f"{', '.join(row['retrieved_chunk_ids']) or 'none'} | "
                f"{', '.join(row['retrieved_unique_docs']) or 'none'} | "
                f"{', '.join(row['relevant_retrieved_docs']) or 'none'} | "
                f"{row['precision_arithmetic']} | {row['recall_arithmetic']} |"
            )
        lines.extend([
            "",
            f"- **Macro Precision@3:** `{evaluation['average_precision']:.4f}`",
            f"- **Macro Recall@3:** `{evaluation['average_recall']:.4f}`",
            f"- **Macro F1:** `{evaluation['average_f1']:.4f}`",
            "",
        ])
    recommended, explanation = result["recommendation"]
    lines.extend([
        "### Strategy recommendation",
        "",
        f"**Recommended:** `{recommended}`.",
        "",
        explanation,
        "",
        "Recommendation ranking uses macro F1; ties are broken by macro recall "
        "and then macro precision. A tie across all three metrics is reported "
        "without claiming a metric-based winner.",
        "",
        "## Method and limitations",
        "",
        "Similarity is cosine similarity (`1 - Chroma cosine distance`). Grounded "
        "answers use the existing deterministic `GroundedGenerator` and only "
        "retrieved context. Threshold calibration is specific to these sampled "
        "queries and should not be treated as a general guarantee of domain "
        "classification.",
        "",
    ])
    return "\n".join(lines)


def run_rag_demo(
    retriever: PolicyRetriever,
    report_path: Path = DEFAULT_REPORT_PATH,
) -> Dict[str, Any]:
    """Calibrate, demonstrate, compare both collections, and save a Markdown report."""
    calibration: Dict[str, Dict[str, Any]] = {}
    evaluations: Dict[str, Dict[str, Any]] = {}
    demonstrations: List[Dict[str, Any]] = []

    in_queries = [item["query"] for item in IN_SCOPE_QUERIES]
    for collection_name in COLLECTIONS:
        calibration[collection_name] = calibrate_empirical_threshold(
            in_queries,
            OUT_OF_SCOPE_QUERIES,
            retriever=retriever,
            collection_name=collection_name,
        )
        evaluations[collection_name] = _evaluate_collection(
            retriever,
            collection_name,
        )
        threshold = calibration[collection_name]["chosen_threshold"]
        generator = GroundedGenerator(retriever=retriever, threshold=threshold)
        for query_info in IN_SCOPE_QUERIES:
            query = query_info["query"]
            if threshold is None:
                hits = retriever.retrieve(query, collection_name, top_k=3)
                generated = _fallback_result(
                    query,
                    collection_name,
                    hits,
                    "Calibration did not show clear separation; threshold remains unset.",
                )
            else:
                generated = generator.generate(query, collection_name, top_k=3)
            demonstrations.append({
                "scope": "In-scope",
                "query": query,
                "collection_name": collection_name,
                "result": generated,
            })
        for query in OUT_OF_SCOPE_QUERIES:
            if threshold is None:
                hits = retriever.retrieve(query, collection_name, top_k=3)
                generated = _fallback_result(
                    query,
                    collection_name,
                    hits,
                    "Calibration did not show clear separation; threshold remains unset.",
                )
            else:
                generated = generator.generate(query, collection_name, top_k=3)
            demonstrations.append({
                "scope": "Out-of-scope",
                "query": query,
                "collection_name": collection_name,
                "result": generated,
            })

    recommendation = _recommend_strategy(evaluations)
    result = {
        "persist_directory": str(
            getattr(
                getattr(retriever, "vector_store", None),
                "persist_directory",
                DEFAULT_PERSIST_DIRECTORY,
            )
        ),
        "calibration": calibration,
        "demonstrations": demonstrations,
        "evaluations": evaluations,
        "recommendation": recommendation,
    }
    report = render_report(result)
    destination = Path(report_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nReport saved to: {destination.resolve()}")
    return result


def write_missing_collections_report(
    error: Exception,
    report_path: Path = DEFAULT_REPORT_PATH,
) -> None:
    """Persist an explicit report if the requested persisted index is unavailable."""
    report = (
        "# RAG Retrieval and Grounded Generation Evaluation\n\n"
        "## Unable to run evaluation\n\n"
        f"{type(error).__name__}: {error}\n\n"
        "No collection was created or modified. Build the indexes with "
        "`python scripts/build_index.py`, then rerun "
        "`python scripts/run_rag_demo.py`.\n"
    )
    destination = Path(report_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(report, encoding="utf-8")
    print(report)
    print(f"Failure report saved to: {destination.resolve()}")


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    try:
        retriever = load_persisted_retriever()
    except (FileNotFoundError, MissingVectorCollectionsError) as error:
        logger.error("%s", error)
        write_missing_collections_report(error)
        return 1
    run_rag_demo(retriever)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
