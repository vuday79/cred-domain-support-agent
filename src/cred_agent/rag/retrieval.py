"""Cosine retrieval and deterministic grounded generation for Cred policies."""

import re
from typing import Any, Dict, List, Optional, Set, Tuple

from src.cred_agent.rag.vector_store import (
    COLLECTION_FIXED,
    COLLECTION_SENTENCE,
    ChromaVectorStore,
)

# None means no threshold has been calibrated for the active collection.
CALIBRATED_FALLBACK_THRESHOLD: Optional[float] = None
FALLBACK_RESPONSE = (
    "I don't have sufficient information in the available knowledge base "
    "to answer this question."
)

CALIBRATION_IN_SCOPE_QUERIES = (
    "What are the eligibility criteria for a personal loan?",
    "How is EMI calculated?",
    "What documents are required for KYC?",
)
CALIBRATION_OUT_OF_SCOPE_QUERIES = (
    "Who won the latest IPL season?",
    "What is the weather forecast for Delhi?",
)

_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "for",
    "how", "i", "in", "is", "it", "of", "on", "or", "the", "to", "was",
    "what", "when", "where", "which", "who", "why", "with",
}


def _terms(text: str) -> Set[str]:
    """Return normalized, non-trivial words for deterministic extractive matching."""
    normalized = set()
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        if token.endswith("ation") and len(token) > 6:
            token = token[:-5]
        elif token.endswith("ates") and len(token) > 6:
            token = token[:-4]
        elif token.endswith("ies") and len(token) > 4:
            token = token[:-3] + "y"
        elif token.endswith(("ing", "ed")) and len(token) > 5:
            token = token[:-3] if token.endswith("ing") else token[:-2]
        elif token.endswith(("ires", "sses", "xes", "ches", "shes")):
            token = token[:-2]
        elif token.endswith("s") and len(token) > 3:
            token = token[:-1]
        if token not in _STOP_WORDS and len(token) > 1:
            normalized.add(token)
    return normalized


def _complete_sentence_fragment(sentence: str) -> str:
    """Avoid returning a fixed-size chunk's dangling final clause."""
    sentence = sentence.strip()
    if not sentence or sentence[-1] in ".!?":
        return sentence
    last_comma = sentence.rfind(",")
    if last_comma < 0:
        return ""
    complete_clause = sentence[:last_comma].strip()
    return f"{complete_clause}." if complete_clause else ""


def _answer_contexts(
    hits: List[Dict[str, Any]],
) -> List[Tuple[str, str]]:
    """Reconstruct overlapping adjacent fixed-size chunks before sentence matching."""
    fixed_chunks: Dict[str, List[Tuple[int, int, str]]] = {}
    contexts: List[Tuple[int, str, str]] = []

    for rank, hit in enumerate(hits):
        metadata = hit.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        chunk_index = metadata.get("chunk_index")
        parent_document_id = hit.get("parent_document_id", "")
        chunking_strategy = hit.get("chunking_strategy", "")
        text = hit.get("text", "")

        if (
            chunking_strategy == "fixed_size"
            and isinstance(chunk_index, int)
            and not isinstance(chunk_index, bool)
            and isinstance(parent_document_id, str)
            and parent_document_id
            and isinstance(text, str)
        ):
            fixed_chunks.setdefault(parent_document_id, []).append(
                (rank, chunk_index, text)
            )
        elif isinstance(text, str) and isinstance(parent_document_id, str):
            contexts.append((rank, text, parent_document_id))

    for parent_document_id, ranked_chunks in fixed_chunks.items():
        ordered_chunks = sorted(ranked_chunks, key=lambda item: item[1])
        run: List[Tuple[int, int, str]] = []

        def append_run() -> None:
            if not run:
                return
            merged_text = run[0][2]
            for _, _, next_text in run[1:]:
                overlap = _suffix_prefix_overlap(merged_text, next_text)
                if overlap == 0:
                    contexts.extend(
                        (rank, text, parent_document_id)
                        for rank, _, text in run
                    )
                    return
                merged_text += next_text[overlap:]
            contexts.append(
                (
                    min(rank for rank, _, _ in run),
                    merged_text,
                    parent_document_id,
                )
            )

        for chunk in ordered_chunks:
            if run and chunk[1] != run[-1][1] + 1:
                append_run()
                run = []
            run.append(chunk)
        append_run()

    contexts.sort(key=lambda context: context[0])
    return [(text, parent_document_id) for _, text, parent_document_id in contexts]


def _suffix_prefix_overlap(left: str, right: str) -> int:
    """Return the longest exact suffix/prefix shared by adjacent chunks."""
    for length in range(min(len(left), len(right)), 0, -1):
        if left[-length:] == right[:length]:
            return length
    return 0


class PolicyRetriever:
    """Retrieve policy chunks from either configured Chroma collection."""

    def __init__(self, vector_store: Optional[ChromaVectorStore] = None):
        self.vector_store = vector_store or ChromaVectorStore()

    def retrieve(
        self,
        query: str,
        collection_name: str = COLLECTION_SENTENCE,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """Return ranked chunks with text, attribution, and cosine similarity."""
        if collection_name not in (COLLECTION_FIXED, COLLECTION_SENTENCE):
            raise ValueError(
                f"collection_name must be {COLLECTION_FIXED!r} or {COLLECTION_SENTENCE!r}"
            )
        return self.vector_store.query(
            query_text=query,
            collection_name=collection_name,
            top_k=top_k,
        )

    def retrieve_parent_documents(
        self,
        query: str,
        collection_name: str = COLLECTION_SENTENCE,
        top_k: int = 3,
    ) -> List[str]:
        """Return unique parent document IDs in retrieval rank order."""
        hits = self.retrieve(query=query, collection_name=collection_name, top_k=top_k)
        return list(dict.fromkeys(
            hit["parent_document_id"]
            for hit in hits
            if hit.get("parent_document_id")
        ))


class GroundedGenerator:
    """Produce deterministic answers only from sentences in retrieved chunks."""

    def __init__(
        self,
        retriever: Optional[PolicyRetriever] = None,
        threshold: Optional[float] = None,
    ):
        self.retriever = retriever or PolicyRetriever()
        self.threshold = threshold
        self._threshold_by_collection: Dict[str, Optional[float]] = {}

    def _threshold_for(self, collection_name: str) -> Optional[float]:
        if self.threshold is not None:
            return self.threshold
        if collection_name not in self._threshold_by_collection:
            report = calibrate_empirical_threshold(
                list(CALIBRATION_IN_SCOPE_QUERIES),
                list(CALIBRATION_OUT_OF_SCOPE_QUERIES),
                retriever=self.retriever,
                collection_name=collection_name,
            )
            self._threshold_by_collection[collection_name] = report["chosen_threshold"]
        return self._threshold_by_collection[collection_name]

    def generate(
        self,
        query: str,
        collection_name: str = COLLECTION_SENTENCE,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        """Return an extractive answer, or the exact fallback when unsupported."""
        hits = self.retriever.retrieve(
            query=query,
            collection_name=collection_name,
            top_k=top_k,
        )
        threshold = self._threshold_for(collection_name)
        top_similarity = hits[0]["cosine_similarity"] if hits else None
        candidate_sources = list(dict.fromkeys(
            hit["parent_document_id"] for hit in hits
            if hit.get("parent_document_id")
        ))
        measurements = [
            {
                "chunk_id": hit["chunk_id"],
                "cosine_similarity": hit["cosine_similarity"],
                "distance": hit["distance"],
            }
            for hit in hits
        ]
        result: Dict[str, Any] = {
            "query": query,
            "answer": FALLBACK_RESPONSE,
            "is_grounded": False,
            "top_1_similarity": top_similarity,
            "threshold": threshold,
            "collection_used": collection_name,
            "retrieved_chunks": hits,
            "retrieved_context": hits,
            "similarity_measurements": measurements,
            "source_documents": [],
            "retrieved_source_documents": candidate_sources,
        }

        if threshold is None:
            result["reason"] = (
                "No threshold was selected because the calibration queries did not "
                "show clear in-scope/out-of-scope separation."
            )
            return result
        if top_similarity is None or top_similarity < threshold:
            result["reason"] = (
                f"Top-1 cosine similarity ({top_similarity}) is below the "
                f"empirically calibrated threshold ({threshold:.4f})."
            )
            return result

        evidence_hits = [
            hit
            for hit in hits
            if hit["cosine_similarity"] >= threshold
        ]
        if not evidence_hits:
            result["reason"] = (
                "No retrieved chunk met the empirically calibrated similarity "
                "threshold."
            )
            return result

        query_terms = _terms(query)
        candidates: List[Tuple[int, str, str]] = []
        for context, parent_document_id in _answer_contexts(evidence_hits):
            for sentence in re.split(r"(?<=[.!?])\s+", context.strip()):
                sentence = sentence.strip()
                if not sentence:
                    continue
                overlap = len(query_terms.intersection(_terms(sentence)))
                if overlap:
                    candidates.append((overlap, sentence, parent_document_id))

        if not candidates:
            result["reason"] = (
                "Retrieved context did not contain query-specific terms to support "
                "an extractive answer."
            )
            return result

        strongest_match = max(overlap for overlap, _, _ in candidates)
        strongest_sources = list(
            dict.fromkeys(
                doc_id
                for overlap, _, doc_id in candidates
                if overlap == strongest_match and doc_id
            )
        )
        selected = []
        for overlap, sentence, doc_id in candidates:
            if overlap <= 0 or (
                strongest_sources and doc_id not in strongest_sources
            ):
                continue
            is_complete_sentence = sentence.rstrip().endswith((".", "!", "?"))
            if (
                not is_complete_sentence
                and overlap < 2
                and not re.search(r"\d", sentence)
            ):
                continue
            complete_sentence = _complete_sentence_fragment(sentence)
            if complete_sentence:
                selected.append((complete_sentence, doc_id))
        unique_sentences = list(dict.fromkeys(sentence for sentence, _ in selected))
        used_sources = list(dict.fromkeys(doc_id for _, doc_id in selected if doc_id))
        result.update(
            answer=" ".join(unique_sentences),
            is_grounded=True,
            source_documents=used_sources,
            reason=(
                "Answer is an extractive composition of sentences from retrieved "
                "knowledge-base chunks."
            ),
        )
        return result


def calibrate_empirical_threshold(
    in_scope_queries: List[str],
    out_of_scope_queries: List[str],
    retriever: Optional[PolicyRetriever] = None,
    collection_name: str = COLLECTION_SENTENCE,
) -> Dict[str, Any]:
    """Measure top-1 cosine scores and choose a midpoint only if clusters separate."""
    if not in_scope_queries or not out_of_scope_queries:
        raise ValueError("Both in_scope_queries and out_of_scope_queries must be non-empty")

    ret = retriever or PolicyRetriever()

    def measure(queries: List[str]) -> List[Dict[str, Any]]:
        measurements: List[Dict[str, Any]] = []
        for query in queries:
            hits = ret.retrieve(query, collection_name=collection_name, top_k=1)
            measurements.append({
                "query": query,
                "top_1_similarity": (
                    hits[0]["cosine_similarity"] if hits else None
                ),
                "top_doc": hits[0]["parent_document_id"] if hits else "",
            })
        return measurements

    in_results = measure(in_scope_queries)
    out_results = measure(out_of_scope_queries)
    in_scores = [
        row["top_1_similarity"] for row in in_results
        if row["top_1_similarity"] is not None
    ]
    out_scores = [
        row["top_1_similarity"] for row in out_results
        if row["top_1_similarity"] is not None
    ]
    complete = len(in_scores) == len(in_results) and len(out_scores) == len(out_results)
    separated = complete and min(in_scores) > max(out_scores)
    chosen_threshold = (
        (min(in_scores) + max(out_scores)) / 2.0 if separated else None
    )

    def cluster(scores: List[float]) -> Dict[str, Optional[float]]:
        if not scores:
            return {"min": None, "max": None, "mean": None}
        return {
            "min": min(scores),
            "max": max(scores),
            "mean": sum(scores) / len(scores),
        }

    separation_gap = min(in_scores) - max(out_scores) if complete else None
    return {
        "collection_name": collection_name,
        "in_scope_measurements": in_results,
        "out_of_scope_measurements": out_results,
        "in_scope_cluster": cluster(in_scores),
        "out_of_scope_cluster": cluster(out_scores),
        "separation_gap": separation_gap,
        "separation_found": separated,
        "chosen_threshold": chosen_threshold,
    }
