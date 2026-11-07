"""FastAPI boundaries for the guarded Cred support workflow."""

import json
import math
import os
import re
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, AsyncGenerator, Dict, Optional

os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.concurrency import run_in_threadpool

from src.cred_agent.config import (
    COLLECTION_FIXED_SIZE,
    COLLECTION_SENTENCE_BASED,
    DEFAULT_CHROMA_PATH,
    PROJECT_ROOT,
)
from src.cred_agent.crew import (
    IntegratedCrewResult,
    route_customer_query,
    run_customer_query,
)
from src.cred_agent.rag.chunking import chunk_documents_fixed, chunk_documents_sentence
from src.cred_agent.rag.document_loader import Document, clean_markdown_text
from src.cred_agent.rag.retrieval import PolicyRetriever
from src.cred_agent.rag.vector_store import ChromaVectorStore
from src.cred_agent.schemas import (
    LoanApplicationDetails,
    SupportResponse,
    SupportResponseType,
    safe_support_response,
    validate_support_response,
)
from src.cred_agent.services.guardrails import (
    check_response_groundedness,
    guard_user_input,
    sanitize_nested_strings,
)
from src.cred_agent.services.api_jsonl_logging import (
    APIJSONLLogger,
)
from src.cred_agent.services.runtime_budget import (
    RuntimeBudget,
    RuntimeBudgetExceeded,
)
from src.cred_agent.services.response_cache import InMemoryResponseCache
from src.cred_agent.services.session_memory import SessionConversationMemory

_DOCUMENT_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{1,63}$")
_METADATA_KEY_PATTERN = re.compile(r"^[a-zA-Z][a-zA-Z0-9_-]{0,63}$")
_RESERVED_METADATA_KEYS = {
    "parent_document_id",
    "source",
    "chunking_strategy",
    "chunk_index",
    "char_count",
}
_LOCAL_PATH_PATTERN = re.compile(
    r"(?<![\w:])(?:[A-Za-z]:[\\/]|/(?!/))"
    r"(?:[^\\/\s\"<>|]+[\\/])*[^\\/\s\"<>|]*"
)


class AskRequest(BaseModel):
    """Request body for one customer-support turn."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=4000)
    session_id: str = Field(min_length=1, max_length=128)

    @field_validator("query", "session_id")
    @classmethod
    def reject_blank_values(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value.strip()


class WebSocketMessage(BaseModel):
    """One user message received over the support WebSocket."""

    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=4000)

    @field_validator("message")
    @classmethod
    def reject_blank_message(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value.strip()


class AddDocumentRequest(BaseModel):
    """Validated request for adding one policy document to both existing indexes."""

    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(min_length=2, max_length=64)
    title: str = Field(min_length=1, max_length=200)
    content: str = Field(min_length=1, max_length=100_000)
    metadata: Dict[str, str | int | float | bool] = Field(default_factory=dict)

    @field_validator("document_id")
    @classmethod
    def validate_document_id(cls, value: str) -> str:
        normalized = value.strip().lower()
        if not _DOCUMENT_ID_PATTERN.fullmatch(normalized):
            raise ValueError(
                "document_id must contain 2-64 lowercase letters, digits, "
                "underscores, or hyphens and start with a letter or digit"
            )
        return normalized

    @field_validator("title", "content")
    @classmethod
    def reject_blank_document_fields(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value.strip()

    @field_validator("metadata")
    @classmethod
    def validate_metadata(
        cls,
        value: Dict[str, str | int | float | bool],
    ) -> Dict[str, str | int | float | bool]:
        for key, item in value.items():
            if not _METADATA_KEY_PATTERN.fullmatch(key):
                raise ValueError("metadata keys must be simple identifiers")
            if key.lower() in _RESERVED_METADATA_KEYS:
                raise ValueError(f"metadata key {key!r} is reserved")
            if isinstance(item, float) and not math.isfinite(item):
                raise ValueError("metadata numbers must be finite")
        return value


class SafeSourceAttribution(BaseModel):
    """Public document attribution without local storage paths."""

    document_id: str
    filename: Optional[str] = None


class PublicSupportResponse(BaseModel):
    """Client-safe projection of the validated internal Crew response."""

    response: str
    response_type: SupportResponseType
    is_grounded: bool
    source_documents: list[str]
    source_attributions: list[SafeSourceAttribution]
    record_id: Optional[str] = None
    loan_application: Optional[LoanApplicationDetails] = None


class AskResponse(BaseModel):
    """Structured response returned by REST and WebSocket requests."""

    trace_id: str
    session_id: str
    response_text: str
    structured_response: PublicSupportResponse
    selected_specialists: list[str]
    input_pii_categories: list[str]
    prompt_injection_detected: bool
    output_groundedness_passed: bool
    workflow_invoked: bool
    history_messages: int
    budget_outcome: str = "within_budget"
    estimated_input_tokens: int = 0
    estimated_total_tokens: int = 0
    synthetic_estimated_cost: str = "0"
    cache_used: bool = False


class APIErrorResponse(BaseModel):
    """Non-sensitive error response."""

    trace_id: str
    error: str


class AddDocumentResponse(BaseModel):
    """Indexing result for a policy document."""

    document_id: str
    title: str
    source: str
    indexed_chunks: Dict[str, int]
    collections: list[str]


class _APISession:
    """Keep LangChain history and the latest validated Crew result per session."""

    def __init__(
        self,
        retriever: Optional[PolicyRetriever] = None,
        response_cache: Optional[InMemoryResponseCache] = None,
    ) -> None:
        self.retriever = retriever
        self.response_cache = response_cache or InMemoryResponseCache()
        self.memory = SessionConversationMemory(workflow=self._run_workflow)
        self._lock = threading.RLock()
        self._latest_result: Optional[IntegratedCrewResult] = None
        self._active_input_cacheable = False

    @staticmethod
    def _response_contains_pii(response: SupportResponse) -> bool:
        text_values = [response.response]
        text_values.extend(context.text for context in response.retrieved_context)
        text_values.extend(context.source for context in response.retrieved_context)
        return any(guard_user_input(value).pii_categories for value in text_values)

    def _run_workflow(
        self,
        query: str,
        history: Any,
        runtime_budget: RuntimeBudget | None = None,
    ) -> str:
        cache_key: Optional[str] = None
        route = route_customer_query(query)
        if (
            self._active_input_cacheable
            and not history
            and not route.lookup_required
        ):
            cache_key = self.response_cache.key_for(query)
            cached = self.response_cache.get(cache_key)
            if cached is not None:
                cached_result = IntegratedCrewResult(
                    query=query,
                    route=route,
                    specialist_findings=[],
                    tool_invocations=[],
                    final_response=cached.response,
                    kickoff_count=0,
                    structured_response=cached,
                    groundedness_passed=True,
                    review_approved=True,
                    cache_used=True,
                )
                with self._lock:
                    self._latest_result = cached_result
                return cached.response

        kwargs: dict[str, Any] = {
            "retriever": self.retriever,
            "conversation_history": history,
        }
        if runtime_budget is not None:
            kwargs["runtime_budget"] = runtime_budget
        result = run_customer_query(query, **kwargs)
        if (
            cache_key is not None
            and not result.route.lookup_required
            and result.validation_error is None
            and result.groundedness_passed
            and result.review_approved is True
            and result.review_turn_count == 2
            and result.structured_response.response_type
            is SupportResponseType.KNOWLEDGE_BASE
            and result.structured_response.is_grounded
            and not self._response_contains_pii(result.structured_response)
            and check_response_groundedness(result.structured_response)[0]
        ):
            validated = validate_support_response(result.structured_response)
            self.response_cache.put(cache_key, validated)
        with self._lock:
            self._latest_result = result
        return result.final_response

    def ask(
        self,
        query: str,
        session_id: str,
        trace_id: str,
        runtime_budget: RuntimeBudget | None = None,
    ) -> AskResponse:
        runtime_budget = runtime_budget or RuntimeBudget()
        try:
            if not runtime_budget.input_registered:
                runtime_budget.register_user_input(query)
        except RuntimeBudgetExceeded:
            self._active_input_cacheable = False
            return _budget_exceeded_response(
                query,
                session_id,
                trace_id,
                runtime_budget,
                history_messages=len(self.memory.get_messages(session_id)),
            )
        guarded_input = guard_user_input(query)
        with self._lock:
            self._latest_result = None
            self._active_input_cacheable = (
                not guarded_input.pii_categories
                and not guarded_input.injection_detected
            )
            try:
                self.memory.chat(
                    session_id,
                    guarded_input.text,
                    runtime_budget=runtime_budget,
                )
            except RuntimeBudgetExceeded:
                self._active_input_cacheable = False
                return _budget_exceeded_response(
                    query,
                    session_id,
                    trace_id,
                    runtime_budget,
                    history_messages=len(self.memory.get_messages(session_id)),
                )
            result = self._latest_result
            if not isinstance(result, IntegratedCrewResult):
                internal_response = safe_support_response()
                specialists: list[str] = []
                injection_detected = guarded_input.injection_detected
                pii_categories = list(guarded_input.pii_categories)
                grounded = False
                workflow_invoked = False
            else:
                internal_response = validate_support_response(
                    result.structured_response
                )
                specialists = result.route.specialist_roles
                injection_detected = result.prompt_injection_detected
                pii_categories = list(guarded_input.pii_categories)
                grounded = result.groundedness_passed
                workflow_invoked = result.kickoff_count > 0

            structured = _public_support_response(internal_response)
            safe_text = structured.response
            body = AskResponse(
                trace_id=trace_id,
                session_id=session_id,
                response_text=safe_text,
                structured_response=structured,
                selected_specialists=specialists,
                input_pii_categories=pii_categories,
                prompt_injection_detected=injection_detected,
                output_groundedness_passed=grounded,
                workflow_invoked=workflow_invoked,
                history_messages=len(self.memory.get_messages(session_id)),
                budget_outcome=runtime_budget.outcome,
                estimated_input_tokens=runtime_budget.input_tokens,
                estimated_total_tokens=runtime_budget.total_tokens,
                synthetic_estimated_cost=format(runtime_budget.estimated_cost, "f"),
                cache_used=(
                    isinstance(result, IntegratedCrewResult)
                    and result.cache_used
                ),
            )
            self._active_input_cacheable = False
        return body


def _budget_exceeded_response(
    query: str,
    session_id: str,
    trace_id: str,
    runtime_budget: RuntimeBudget,
    history_messages: int,
) -> AskResponse:
    """Return a schema-valid safe response without adding a memory turn."""
    guarded_input = guard_user_input(query)
    safe = _public_support_response(safe_support_response())
    snapshot = runtime_budget.snapshot()
    return AskResponse(
        trace_id=trace_id,
        session_id=session_id,
        response_text=safe.response,
        structured_response=safe,
        selected_specialists=[],
        input_pii_categories=list(guarded_input.pii_categories),
        prompt_injection_detected=guarded_input.injection_detected,
        output_groundedness_passed=False,
        workflow_invoked=runtime_budget.workflow_started,
        history_messages=history_messages,
        budget_outcome=str(snapshot["outcome"]),
        estimated_input_tokens=int(snapshot["input_tokens"]),
        estimated_total_tokens=int(snapshot["total_tokens"]),
        synthetic_estimated_cost=str(snapshot["synthetic_estimated_cost"]),
    )


def _inspect_guardrails(value: Any) -> tuple[set[str], bool]:
    """Collect only guardrail categories from arbitrary request strings."""
    categories: set[str] = set()
    injection_detected = False
    if isinstance(value, str):
        if not value.strip():
            return categories, injection_detected
        guarded = guard_user_input(value)
        categories.update(guarded.pii_categories)
        injection_detected = guarded.injection_detected
    elif isinstance(value, dict):
        for item in value.values():
            item_categories, item_injection = _inspect_guardrails(item)
            categories.update(item_categories)
            injection_detected = injection_detected or item_injection
    elif isinstance(value, (list, tuple)):
        for item in value:
            item_categories, item_injection = _inspect_guardrails(item)
            categories.update(item_categories)
            injection_detected = injection_detected or item_injection
    return categories, injection_detected


def _request_outcome(http_status: int, state: Any) -> str:
    if state.outcome:
        return state.outcome
    if http_status >= 500:
        return "error"
    if http_status == 422:
        return "validation_error"
    if http_status >= 400:
        return "rejected"
    return "success"


def _get_or_create_session(app: FastAPI, session_id: str) -> _APISession:
    sessions: dict[str, _APISession] = app.state.sessions
    if session_id not in sessions:
        retriever = (
            PolicyRetriever(vector_store=app.state.vector_store)
            if app.state.vector_store is not None
            else None
        )
        sessions[session_id] = _APISession(
            retriever=retriever,
            response_cache=app.state.response_cache,
        )
    return sessions[session_id]


def _public_support_response(response: SupportResponse) -> PublicSupportResponse:
    """Project validated internal findings into a path- and chunk-safe payload."""
    attributions: list[SafeSourceAttribution] = []
    seen_sources: set[str] = set()
    contexts_by_source: dict[str, str] = {}
    for context in response.retrieved_context:
        contexts_by_source.setdefault(context.parent_document_id, context.source)

    for document_id in response.source_documents:
        if document_id in seen_sources:
            continue
        seen_sources.add(document_id)
        source = contexts_by_source.get(document_id)
        filename = None
        if source:
            normalized_source = source.replace("\\", "/")
            filename = PurePosixPath(normalized_source).name or None
        attributions.append(
            SafeSourceAttribution(
                document_id=document_id,
                filename=filename,
            )
        )

    public_text = _remove_local_paths(response.response)
    public_text = guard_user_input(public_text).text
    return PublicSupportResponse(
        response=public_text,
        response_type=response.response_type,
        is_grounded=response.is_grounded,
        source_documents=response.source_documents,
        source_attributions=attributions,
        record_id=response.record_id,
        loan_application=response.loan_application,
    )


def _remove_local_paths(text: str) -> str:
    """Replace absolute local paths in client-facing response prose with filenames."""
    return _LOCAL_PATH_PATTERN.sub(
        lambda match: PurePosixPath(match.group(0).replace("\\", "/")).name
        or "[local path omitted]",
        text,
    )


def _get_vector_store(app: FastAPI) -> ChromaVectorStore:
    if app.state.vector_store is None:
        persist_directory = (PROJECT_ROOT / DEFAULT_CHROMA_PATH).resolve()
        app.state.vector_store = ChromaVectorStore(
            persist_directory=persist_directory
        )
    return app.state.vector_store


def _document_exists(store: ChromaVectorStore, document_id: str) -> bool:
    for collection_name in (COLLECTION_FIXED_SIZE, COLLECTION_SENTENCE_BASED):
        collection = store.get_collection(collection_name)
        existing = collection.get(
            where={"parent_document_id": document_id},
            limit=1,
            include=["metadatas"],
        )
        if existing.get("ids"):
            return True
    return False


def _index_document(
    store: ChromaVectorStore,
    request: AddDocumentRequest,
) -> AddDocumentResponse:
    guarded_title = guard_user_input(request.title)
    guarded_content = guard_user_input(request.content)
    if guarded_title.injection_detected or guarded_content.injection_detected:
        raise ValueError("document contains disallowed instruction-like content")

    guarded_metadata = sanitize_nested_strings(request.metadata)
    if not isinstance(guarded_metadata, dict):
        raise ValueError("document metadata is invalid")
    if _document_exists(store, request.document_id):
        raise FileExistsError("document_id is already indexed")

    source = f"api://{request.document_id}/{request.document_id}.md"
    clean_text = clean_markdown_text(guarded_content.text)
    if not clean_text:
        raise ValueError("document content has no indexable text")
    document = Document(
        doc_id=request.document_id,
        doc_name=guarded_title.text,
        filename=f"{request.document_id}.md",
        source=source,
        text=clean_text,
        raw_content=guarded_content.text,
        metadata={"title": guarded_title.text, **guarded_metadata},
    )
    fixed_chunks = [
        replace(
            chunk,
            metadata={"title": guarded_title.text, **guarded_metadata},
        )
        for chunk in chunk_documents_fixed([document])
    ]
    sentence_chunks = [
        replace(
            chunk,
            metadata={"title": guarded_title.text, **guarded_metadata},
        )
        for chunk in chunk_documents_sentence([document])
    ]
    fixed_count = store.index_fixed_size_chunks(fixed_chunks)
    sentence_count = store.index_sentence_based_chunks(sentence_chunks)
    return AddDocumentResponse(
        document_id=request.document_id,
        title=guarded_title.text,
        source=source,
        indexed_chunks={
            COLLECTION_FIXED_SIZE: fixed_count,
            COLLECTION_SENTENCE_BASED: sentence_count,
        },
        collections=[COLLECTION_FIXED_SIZE, COLLECTION_SENTENCE_BASED],
    )


def create_app(
    *,
    vector_store: Optional[ChromaVectorStore] = None,
    log_path: Optional[Path] = None,
) -> FastAPI:
    """Create the API application, optionally with an injected existing vector store."""
    api_jsonl_logger = APIJSONLLogger(log_path)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
        try:
            yield
        finally:
            api_jsonl_logger.close()

    app = FastAPI(
        title="Cred Domain Support Agent",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.sessions = {}
    app.state.vector_store = vector_store
    app.state.api_jsonl_logger = api_jsonl_logger
    app.state.response_cache = InMemoryResponseCache()

    @app.middleware("http")
    async def log_http_request(request: Request, call_next):
        started_monotonic = time.perf_counter()
        request_started_at = datetime.now(timezone.utc).isoformat(
            timespec="milliseconds"
        )
        request.state.trace_id = str(uuid.uuid4())
        request.state.session_id = None
        request.state.pii_categories = []
        request.state.prompt_injection_detected = False
        request.state.output_groundedness_passed = None
        request.state.response_type = None
        request.state.error_category = None
        request.state.outcome = None
        request.state.budget_outcome = None
        request.state.cache_used = False
        endpoint = {
            "/ask": "/ask",
            "/add-document": "/add-document",
        }.get(request.url.path, "unmatched")
        try:
            response = await call_next(request)
        except Exception:
            app.state.api_jsonl_logger.emit(
                trace_id=request.state.trace_id,
                session_id=request.state.session_id,
                endpoint=endpoint,
                method=request.method,
                event_type="http_request",
                request_started_at=request_started_at,
                started_monotonic=started_monotonic,
                outcome="error",
                http_status=500,
                error_category=request.state.error_category or "internal_error",
                pii_categories=request.state.pii_categories,
                prompt_injection_detected=request.state.prompt_injection_detected,
                output_groundedness_passed=(
                    request.state.output_groundedness_passed
                ),
                budget_outcome=request.state.budget_outcome,
                cache_used=request.state.cache_used,
            )
            raise

        app.state.api_jsonl_logger.emit(
            trace_id=request.state.trace_id,
            session_id=request.state.session_id,
            endpoint=endpoint,
            method=request.method,
            event_type="http_request",
            request_started_at=request_started_at,
            started_monotonic=started_monotonic,
            outcome=_request_outcome(response.status_code, request.state),
            http_status=response.status_code,
            response_type=request.state.response_type,
            error_category=request.state.error_category,
            pii_categories=request.state.pii_categories,
            prompt_injection_detected=request.state.prompt_injection_detected,
            output_groundedness_passed=(
                request.state.output_groundedness_passed
            ),
            budget_outcome=request.state.budget_outcome,
            cache_used=request.state.cache_used,
        )
        return response

    @app.exception_handler(RequestValidationError)
    async def request_validation_error(
        _request: Request,
        _exc: RequestValidationError,
    ) -> JSONResponse:
        _request.state.error_category = (
            "validation_error" if _exc.errors() else "request_error"
        )
        _request.state.outcome = "validation_error"
        try:
            body = await _request.json()
        except (json.JSONDecodeError, UnicodeDecodeError, RuntimeError):
            body = None
        categories, injection_detected = _inspect_guardrails(body)
        _request.state.pii_categories = sorted(categories)
        _request.state.prompt_injection_detected = injection_detected
        return JSONResponse(
            status_code=422,
            content={
                "trace_id": _request.state.trace_id,
                "error": "Invalid request payload.",
            },
        )

    @app.post("/ask", response_model=AskResponse | APIErrorResponse)
    def ask(
        request_context: Request,
        request: AskRequest,
    ) -> AskResponse | APIErrorResponse | JSONResponse:
        trace_id = request_context.state.trace_id
        session_id = request.session_id
        request_context.state.session_id = session_id
        runtime_budget = RuntimeBudget()
        try:
            runtime_budget.register_user_input(request.query)
        except RuntimeBudgetExceeded:
            existing_session = app.state.sessions.get(session_id)
            history_messages = (
                len(existing_session.memory.get_messages(session_id))
                if existing_session is not None
                else 0
            )
            response = _budget_exceeded_response(
                request.query,
                session_id,
                trace_id,
                runtime_budget,
                history_messages,
            )
            request_context.state.pii_categories = response.input_pii_categories
            request_context.state.prompt_injection_detected = (
                response.prompt_injection_detected
            )
            request_context.state.response_type = (
                response.structured_response.response_type.value
            )
            request_context.state.output_groundedness_passed = False
            request_context.state.outcome = "rejected"
            request_context.state.error_category = "runtime_budget"
            request_context.state.budget_outcome = response.budget_outcome
            request_context.state.cache_used = response.cache_used
            return response
        try:
            response = _get_or_create_session(app, session_id).ask(
                request.query,
                session_id,
                trace_id,
                runtime_budget,
            )
            request_context.state.pii_categories = response.input_pii_categories
            request_context.state.prompt_injection_detected = (
                response.prompt_injection_detected
            )
            request_context.state.output_groundedness_passed = (
                response.output_groundedness_passed
            )
            request_context.state.response_type = (
                response.structured_response.response_type.value
            )
            request_context.state.budget_outcome = response.budget_outcome
            request_context.state.cache_used = response.cache_used
            if response.budget_outcome != "within_budget":
                request_context.state.outcome = "rejected"
                request_context.state.error_category = "runtime_budget"
            if response.prompt_injection_detected:
                request_context.state.outcome = "rejected"
                request_context.state.error_category = "prompt_injection"
            return response
        except ValueError:
            request_context.state.outcome = "rejected"
            request_context.state.error_category = "input_rejected"
            return JSONResponse(
                status_code=400,
                content=APIErrorResponse(
                    trace_id=trace_id,
                    error="The request could not be processed safely.",
                ).model_dump(),
            )
        except Exception:
            request_context.state.outcome = "error"
            request_context.state.error_category = "workflow_error"
            return JSONResponse(
                status_code=500,
                content=APIErrorResponse(
                    trace_id=trace_id,
                    error="The request could not be processed safely.",
                ).model_dump(),
            )

    @app.post(
        "/add-document",
        response_model=AddDocumentResponse | APIErrorResponse,
    )
    def add_document(
        request_context: Request,
        request: AddDocumentRequest,
    ) -> AddDocumentResponse | APIErrorResponse | JSONResponse:
        trace_id = request_context.state.trace_id
        categories, injection_detected = _inspect_guardrails(
            [request.title, request.content, request.metadata]
        )
        request_context.state.pii_categories = sorted(categories)
        request_context.state.prompt_injection_detected = injection_detected
        try:
            response = _index_document(_get_vector_store(app), request)
            app.state.response_cache.clear()
        except FileExistsError:
            request_context.state.outcome = "rejected"
            request_context.state.error_category = "duplicate_document"
            return JSONResponse(
                status_code=409,
                content={
                    "trace_id": trace_id,
                    "error": "document_id is already indexed.",
                },
            )
        except ValueError:
            request_context.state.outcome = "rejected"
            request_context.state.error_category = "document_input_rejected"
            return JSONResponse(
                status_code=400,
                content={
                    "trace_id": trace_id,
                    "error": "Document content was rejected by input validation.",
                },
            )
        except Exception:
            request_context.state.outcome = "error"
            request_context.state.error_category = "document_index_error"
            return JSONResponse(
                status_code=500,
                content={
                    "trace_id": trace_id,
                    "error": "The document could not be indexed.",
                },
            )
        return response

    @app.websocket("/ws/{session_id}")
    async def chat_websocket(websocket: WebSocket, session_id: str) -> None:
        connection_started = time.perf_counter()
        connection_started_at = datetime.now(timezone.utc).isoformat(
            timespec="milliseconds"
        )
        trace_id = str(uuid.uuid4())
        if not session_id.strip() or len(session_id) > 128:
            await websocket.accept()
            app.state.api_jsonl_logger.emit(
                trace_id=trace_id,
                session_id=session_id,
                endpoint="/ws/{session_id}",
                method="WEBSOCKET",
                event_type="websocket_connection",
                request_started_at=connection_started_at,
                started_monotonic=connection_started,
                outcome="rejected",
                http_status=None,
                error_category="invalid_session_id",
            )
            await websocket.send_json(
                {
                    "trace_id": trace_id,
                    "error": "Invalid session identifier.",
                }
            )
            await websocket.close(code=1008)
            return

        await websocket.accept()
        session = _get_or_create_session(app, session_id.strip())
        try:
            while True:
                started_monotonic = time.perf_counter()
                request_started_at = datetime.now(timezone.utc).isoformat(
                    timespec="milliseconds"
                )
                message_trace_id = str(uuid.uuid4())
                pii_categories: set[str] = set()
                injection_detected = False
                runtime_budget = RuntimeBudget()
                try:
                    payload = await websocket.receive_json()
                    pii_categories, injection_detected = _inspect_guardrails(payload)
                    message = WebSocketMessage.model_validate(payload)
                except WebSocketDisconnect:
                    app.state.api_jsonl_logger.emit(
                        trace_id=message_trace_id,
                        session_id=session_id,
                        endpoint="/ws/{session_id}",
                        method="WEBSOCKET",
                        event_type="websocket_disconnect",
                        request_started_at=request_started_at,
                        started_monotonic=started_monotonic,
                        outcome="disconnected",
                        http_status=None,
                    )
                    break
                except Exception:
                    app.state.api_jsonl_logger.emit(
                        trace_id=message_trace_id,
                        session_id=session_id,
                        endpoint="/ws/{session_id}",
                        method="WEBSOCKET",
                        event_type="websocket_message",
                        request_started_at=request_started_at,
                        started_monotonic=started_monotonic,
                        outcome="validation_error",
                        http_status=None,
                        error_category="invalid_message_payload",
                        pii_categories=pii_categories,
                        prompt_injection_detected=injection_detected,
                    )
                    await websocket.send_json(
                        {
                            "trace_id": message_trace_id,
                            "error": "Invalid message payload.",
                        }
                    )
                    continue
                try:
                    response = await run_in_threadpool(
                        session.ask,
                        message.message,
                        session_id.strip(),
                        message_trace_id,
                        runtime_budget,
                    )
                    await websocket.send_json(response.model_dump(mode="json"))
                    app.state.api_jsonl_logger.emit(
                        trace_id=message_trace_id,
                        session_id=session_id,
                        endpoint="/ws/{session_id}",
                        method="WEBSOCKET",
                        event_type="websocket_message",
                        request_started_at=request_started_at,
                        started_monotonic=started_monotonic,
                        outcome=(
                            "rejected"
                            if (
                                response.prompt_injection_detected
                                or response.budget_outcome != "within_budget"
                            )
                            else "success"
                        ),
                        http_status=None,
                        response_type=(
                            response.structured_response.response_type.value
                        ),
                        error_category=(
                            "prompt_injection"
                            if response.prompt_injection_detected
                            else (
                                "runtime_budget"
                                if response.budget_outcome != "within_budget"
                                else None
                            )
                        ),
                        pii_categories=response.input_pii_categories,
                        prompt_injection_detected=(
                            response.prompt_injection_detected
                        ),
                        output_groundedness_passed=(
                            response.output_groundedness_passed
                        ),
                        budget_outcome=response.budget_outcome,
                        cache_used=response.cache_used,
                    )
                except Exception:
                    app.state.api_jsonl_logger.emit(
                        trace_id=message_trace_id,
                        session_id=session_id,
                        endpoint="/ws/{session_id}",
                        method="WEBSOCKET",
                        event_type="websocket_message",
                        request_started_at=request_started_at,
                        started_monotonic=started_monotonic,
                        outcome="error",
                        http_status=None,
                        error_category="workflow_error",
                        pii_categories=pii_categories,
                        prompt_injection_detected=injection_detected,
                    )
                    await websocket.send_json(
                        {
                            "trace_id": message_trace_id,
                            "error": "The message could not be processed safely.",
                        }
                    )
        except WebSocketDisconnect:
            pass

    return app


app = create_app()
