"""Privacy-preserving JSON Lines request logging for the local API."""

from datetime import datetime, timezone
import hashlib
import hmac
import json
import logging
import os
from pathlib import Path
import secrets
import time
import uuid
from typing import Optional, Sequence

_SESSION_HASH_KEY = secrets.token_bytes(32)


def default_api_log_path() -> Path:
    """Choose a configurable log location outside the project by default."""
    configured_path = os.environ.get("CRED_API_LOG_PATH")
    if configured_path:
        return Path(configured_path).expanduser().resolve()

    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        base_directory = Path(local_app_data)
    else:
        base_directory = Path.home() / ".local" / "state"
    return (
        base_directory / "CredDomainSupportAgent" / "logs" / "api_requests.jsonl"
    )


def hash_session_id(session_id: Optional[str]) -> Optional[str]:
    """Return a process-local keyed identifier without writing the session ID."""
    if not session_id:
        return None
    digest = hmac.new(
        _SESSION_HASH_KEY,
        session_id.strip().encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return f"hmac-sha256:{digest[:20]}"


class APIJSONLLogger:
    """Write one sanitized, consistently shaped JSON object per event."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path or default_api_log_path()).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(
            f"src.cred_agent.api.jsonl.{uuid.uuid4().hex}"
        )
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        self._handler = logging.FileHandler(
            self.path,
            encoding="utf-8",
            delay=True,
        )
        self._handler.setFormatter(logging.Formatter("%(message)s"))
        self.logger.addHandler(self._handler)

    def emit(
        self,
        *,
        trace_id: str,
        session_id: Optional[str],
        endpoint: str,
        method: str,
        event_type: str,
        request_started_at: str,
        started_monotonic: float,
        outcome: str,
        http_status: Optional[int],
        response_type: Optional[str] = None,
        error_category: Optional[str] = None,
        pii_categories: Sequence[str] = (),
        prompt_injection_detected: bool = False,
        output_groundedness_passed: Optional[bool] = None,
        budget_outcome: Optional[str] = None,
        cache_used: bool = False,
    ) -> None:
        """Emit an event without prompts, responses, PII values, or exceptions."""
        now = datetime.now(timezone.utc)
        event = {
            "timestamp": now.isoformat(timespec="milliseconds"),
            "trace_id": trace_id,
            "session_id_hash": hash_session_id(session_id),
            "endpoint": endpoint,
            "method": method,
            "event_type": event_type,
            "request_started_at": request_started_at,
            "duration_ms": round(
                max(0.0, (time.perf_counter() - started_monotonic) * 1000),
                3,
            ),
            "outcome": outcome,
            "http_status": http_status,
            "response_type": response_type,
            "error_category": error_category,
            "pii_detected": bool(pii_categories),
            "pii_categories": sorted(set(pii_categories)),
            "prompt_injection_detected": prompt_injection_detected,
            "output_groundedness_passed": output_groundedness_passed,
            "budget_outcome": budget_outcome,
            "cache_used": cache_used,
        }
        self.logger.info(
            json.dumps(
                event,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        )

    def close(self) -> None:
        """Flush and close this app instance's dedicated file handler."""
        self.logger.removeHandler(self._handler)
        self._handler.close()
