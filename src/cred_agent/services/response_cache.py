"""Process-local bounded LRU cache for safe grounded knowledge-base answers."""

import hashlib
import json
import os
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

from src.cred_agent.schemas import (
    SupportResponse,
    SupportResponseType,
    validate_support_response,
)
from src.cred_agent.services.guardrails import (
    check_response_groundedness,
    guard_user_input,
)

_WHITESPACE = re.compile(r"\s+")


def normalize_query(query: str) -> str:
    """Trim, case-fold, and collapse whitespace for stable cache identity."""
    return _WHITESPACE.sub(" ", query.strip()).casefold()


@dataclass(frozen=True)
class ResponseCacheStats:
    size: int
    max_size: int
    hits: int
    misses: int
    evictions: int
    generation: int


class InMemoryResponseCache:
    """Thread-safe, process-local LRU cache of validated KB-only responses."""

    def __init__(self, max_size: Optional[int] = None) -> None:
        configured_size = (
            max_size
            if max_size is not None
            else int(os.environ.get("CRED_RESPONSE_CACHE_MAX_SIZE", "256"))
        )
        if configured_size < 1:
            raise ValueError("response cache max_size must be positive")
        self.max_size = configured_size
        self._entries: OrderedDict[str, SupportResponse] = OrderedDict()
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._generation = 0

    @staticmethod
    def make_key(
        query: str,
        *,
        context: str = "",
        generation: int = 0,
        retrieval_profile: str = (
            "fixed-size:cosine:threshold=0.286211:top-k=3:"
            "embedding=sentence-transformers/all-MiniLM-L6-v2:"
            "generation=crewai-autogen-mock-v1"
        ),
    ) -> str:
        """Hash normalized query plus any answer-affecting runtime context."""
        identity = {
            "query": normalize_query(query),
            "context": normalize_query(context),
            "generation": generation,
            "retrieval_profile": retrieval_profile,
        }
        serialized = json.dumps(
            identity,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def key_for(
        self,
        query: str,
        *,
        context: str = "",
        retrieval_profile: str = (
            "fixed-size:cosine:threshold=0.286211:top-k=3:"
            "embedding=sentence-transformers/all-MiniLM-L6-v2:"
            "generation=crewai-autogen-mock-v1"
        ),
    ) -> str:
        with self._lock:
            generation = self._generation
        return self.make_key(
            query,
            context=context,
            generation=generation,
            retrieval_profile=retrieval_profile,
        )

    def get(self, key: str) -> Optional[SupportResponse]:
        """Return a deep, Pydantic-validated copy and update LRU counters."""
        with self._lock:
            response = self._entries.get(key)
            if response is None:
                self._misses += 1
                return None
            self._entries.move_to_end(key)
            self._hits += 1
            return validate_support_response(response.model_copy(deep=True))

    def put(self, key: str, response: SupportResponse) -> bool:
        """Insert only safe, grounded, schema-valid KB responses."""
        validated = validate_support_response(response)
        if (
            validated.response_type is not SupportResponseType.KNOWLEDGE_BASE
            or not validated.is_grounded
            or not validated.source_documents
            or not check_response_groundedness(validated)[0]
        ):
            return False
        text_values = [validated.response]
        text_values.extend(context.text for context in validated.retrieved_context)
        text_values.extend(context.source for context in validated.retrieved_context)
        if any(guard_user_input(value).pii_categories for value in text_values):
            return False
        with self._lock:
            if key in self._entries:
                self._entries.pop(key)
            self._entries[key] = validated.model_copy(deep=True)
            self._entries.move_to_end(key)
            if len(self._entries) > self.max_size:
                self._entries.popitem(last=False)
                self._evictions += 1
            return True

    def clear(self, *, reset_counters: bool = False) -> None:
        """Clear entries and invalidate outstanding keys by advancing generation."""
        with self._lock:
            self._entries.clear()
            self._generation += 1
            if reset_counters:
                self._hits = 0
                self._misses = 0
                self._evictions = 0

    def stats(self) -> ResponseCacheStats:
        """Return a consistent snapshot of entry and cache-operation counts."""
        with self._lock:
            return ResponseCacheStats(
                size=len(self._entries),
                max_size=self.max_size,
                hits=self._hits,
                misses=self._misses,
                evictions=self._evictions,
                generation=self._generation,
            )
