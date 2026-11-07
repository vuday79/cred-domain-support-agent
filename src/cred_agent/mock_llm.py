"""Local deterministic CrewAI MOCK_LLM with schema-driven native tool calls."""

import hashlib
import json
import os
import re
from typing import Any, Dict, List, Mapping, Optional, Tuple

# CrewAI's telemetry reads these flags at initialization. Force them before
# importing CrewAI so use of this demonstration LLM cannot enable telemetry.
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["CREWAI_DISABLE_TRACKING"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"

from pydantic import BaseModel, PrivateAttr, ValidationError
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.llms.base_llm import BaseLLM
from crewai.task import Task
from crewai.tools import BaseTool
from crewai.utilities.types import LLMMessage

from src.cred_agent.schemas import (
    SupportResponse,
    safe_support_response,
)
from src.cred_agent.services.response_composition import (
    compose_specialist_findings,
    compose_structured_specialist_findings,
)
from src.cred_agent.services.runtime_budget import RuntimeBudget
from src.cred_agent.services.tool_permissions import is_tool_authorized

MOCK_UNSUPPORTED_RESPONSE = (
    "MOCK_LLM cannot fulfill this request with the declared tools or context."
)
_RECORD_ID_PATTERN = re.compile(r"\bAPP-\d+\b", re.IGNORECASE)
_CUSTOMER_QUESTION_PATTERN = re.compile(
    r"Customer question:\s*([^\r\n]+)",
    re.IGNORECASE,
)
_SUPPORTED_INPUT_FIELDS = {"record_id", "query", "question"}


class CredMockLLM(BaseLLM):
    """Deterministic, local-only BaseLLM for Cred project demonstrations.

    CrewAI 1.15's native tool workflow expects a list of OpenAI-shaped tool-call
    dictionaries from ``call``. This model chooses a tool by its declared
    argument schema, returns validated arguments to CrewAI, and lets CrewAI
    execute the tool and append a structured ``role="tool"`` message. On the
    subsequent call, this model responds from those tool-role messages without
    inspecting prompt text for an "Observation:" marker.
    """

    llm_type: str = "mock"
    model: str = "cred-mock-v1"
    provider: str = "mock"
    temperature: float = 0.0
    seed: int = 0
    api_key: Optional[str] = None
    _runtime_budget: Optional[RuntimeBudget] = PrivateAttr(default=None)

    def __init__(
        self,
        runtime_budget: Optional[RuntimeBudget] = None,
        **data: Any,
    ) -> None:
        data.setdefault("model", "cred-mock-v1")
        data.setdefault("provider", "mock")
        super().__init__(**data)
        self._runtime_budget = runtime_budget

    def supports_function_calling(self) -> bool:
        """Tell CrewAI to use its native structured tool-call execution path."""
        return True

    def supports_stop_words(self) -> bool:
        """Support configured stop sequences via BaseLLM's local helper."""
        return True

    def call(
        self,
        messages: str | List[LLMMessage],
        tools: Optional[List[Dict[str, BaseTool]]] = None,
        callbacks: Optional[List[Any]] = None,
        available_functions: Optional[Dict[str, Any]] = None,
        from_task: Optional[Task] = None,
        from_agent: Optional[BaseAgent] = None,
        response_model: Optional[type[BaseModel]] = None,
    ) -> str | Any:
        """Return deterministic text or CrewAI-native calls for declared tools.

        ``available_functions`` is intentionally not invoked here: CrewAI's
        native executor owns tool execution, lifecycle events, and validation.
        """
        del callbacks, available_functions, from_task
        conversation = self._normalize_messages(messages)
        latest_user_index = self._latest_message_index(conversation, "user")
        query = (
            self._message_text(conversation[latest_user_index].get("content"))
            if latest_user_index is not None
            else ""
        )

        if response_model is not None:
            if not issubclass(response_model, SupportResponse):
                response = self._apply_stop_words(MOCK_UNSUPPORTED_RESPONSE)
                return self._charge_budget(conversation, response, tools)
            try:
                response = compose_structured_specialist_findings(query)
            except (KeyError, TypeError, ValueError, ValidationError):
                response = safe_support_response()
            return self._charge_budget(conversation, response, tools)

        tool_results = self._tool_results_after(conversation, latest_user_index)
        if tool_results:
            response = self._format_tool_results(tool_results)
            return self._charge_budget(
                conversation,
                self._apply_stop_words(response),
                tools,
            )

        if not query.strip():
            return self._charge_budget(
                conversation,
                self._apply_stop_words(MOCK_UNSUPPORTED_RESPONSE),
                tools,
            )

        if not tools:
            composed = compose_specialist_findings(query)
            if composed is not None:
                response = self._apply_stop_words(composed)
            else:
                response = self._apply_stop_words(MOCK_UNSUPPORTED_RESPONSE)
            return self._charge_budget(conversation, response, tools)

        selected = self._select_tool_call(
            query,
            tools or [],
            agent_role=getattr(from_agent, "role", ""),
        )
        if selected is None:
            response = self._apply_stop_words(MOCK_UNSUPPORTED_RESPONSE)
        else:
            response = [selected]
        return self._charge_budget(conversation, response, tools)

    def _charge_budget(
        self,
        conversation: List[Mapping[str, Any]],
        response: str | Any,
        tools: Optional[List[Dict[str, BaseTool]]] = None,
    ) -> str | Any:
        if self._runtime_budget is None:
            return response
        prompt = json.dumps(
            {"messages": conversation, "tools": tools or []},
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        completion = (
            response
            if isinstance(response, str)
            else json.dumps(response, ensure_ascii=False, sort_keys=True, default=str)
        )
        self._runtime_budget.charge_model_call(prompt, completion)
        return response

    @staticmethod
    def _normalize_messages(
        messages: str | List[LLMMessage],
    ) -> List[Mapping[str, Any]]:
        if isinstance(messages, str):
            return [{"role": "user", "content": messages}]
        if not isinstance(messages, list):
            raise ValueError("messages must be a string or a list of message objects")
        normalized: List[Mapping[str, Any]] = []
        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                raise ValueError(f"messages[{index}] must be a message object")
            role = message.get("role")
            if role not in {"system", "user", "assistant", "tool"}:
                raise ValueError(f"messages[{index}] has unsupported role {role!r}")
            normalized.append(message)
        return normalized

    @staticmethod
    def _latest_message_index(
        messages: List[Mapping[str, Any]],
        role: str,
    ) -> Optional[int]:
        for index in range(len(messages) - 1, -1, -1):
            if messages[index].get("role") == role:
                return index
        return None

    @staticmethod
    def _message_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            fragments = [
                block["text"]
                for block in content
                if isinstance(block, dict) and isinstance(block.get("text"), str)
            ]
            return "\n".join(fragments)
        return ""

    def _tool_results_after(
        self,
        messages: List[Mapping[str, Any]],
        latest_user_index: Optional[int],
    ) -> List[str]:
        if latest_user_index is None:
            return []
        return [
            self._message_text(message.get("content"))
            for message in messages[latest_user_index + 1 :]
            if message.get("role") == "tool"
            and self._message_text(message.get("content")).strip()
        ]

    @staticmethod
    def _format_tool_results(results: List[str]) -> str:
        if len(results) == 1:
            return f"Tool result: {results[0]}"
        return "Tool results:\n" + "\n".join(
            f"{index}. {result}" for index, result in enumerate(results, start=1)
        )

    def _select_tool_call(
        self,
        query: str,
        tools: List[Dict[str, Any]],
        agent_role: str = "",
    ) -> Optional[Dict[str, Any]]:
        record_ids = _RECORD_ID_PATTERN.findall(query)
        candidates = []
        for tool in tools:
            tool_name, tool_schema = self._function_schema(tool)
            if not is_tool_authorized(agent_role, tool_name):
                raise PermissionError(
                    f"Agent role {agent_role!r} cannot invoke tool {tool_name!r}."
                )
            expected_argument = (
                "record_id"
                if tool_name == "cred_application_record"
                else "question"
            )
            if (
                set(tool_schema.get("properties", {})) != {expected_argument}
                or tool_schema.get("required") != [expected_argument]
            ):
                raise PermissionError(
                    f"Tool {tool_name!r} has an unauthorized argument schema."
                )
            if record_ids and self._has_schema_field(tool, "record_id"):
                candidates.append(
                    self._candidate_for_schema(
                        tool,
                        "record_id",
                        record_ids[-1],
                    )
                )
            for field in ("query", "question"):
                if self._has_schema_field(tool, field):
                    candidates.append(
                        self._candidate_for_schema(
                            tool,
                            field,
                            self._extract_customer_question(query),
                        )
                    )
        candidates = [candidate for candidate in candidates if candidate is not None]
        if len(candidates) != 1:
            return None

        tool_name, arguments, schema = candidates[0]
        self._validate_arguments(arguments, schema, tool_name)
        call_id = hashlib.sha256(
            json.dumps(
                {"tool": tool_name, "arguments": arguments},
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()[:16]
        return {
            "id": f"mock_{call_id}",
            "type": "function",
            "function": {
                "name": tool_name,
                "arguments": json.dumps(arguments, sort_keys=True),
            },
        }

    @staticmethod
    def _extract_customer_question(prompt: str) -> str:
        """Prefer the explicit customer question over surrounding task guidance."""
        match = _CUSTOMER_QUESTION_PATTERN.search(prompt)
        if match:
            return match.group(1).strip()
        return prompt.strip()

    @staticmethod
    def _function_schema(tool: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            raise ValueError("Each tool must use the declared function schema format")
        function = tool.get("function")
        if not isinstance(function, dict):
            raise ValueError("Tool schema must include a function object")
        name = function.get("name")
        parameters = function.get("parameters")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Tool function schema must declare a non-empty name")
        if not isinstance(parameters, dict):
            raise ValueError(f"Tool {name!r} must declare an object parameters schema")
        if parameters.get("type") != "object":
            raise ValueError(f"Tool {name!r} parameters schema must have type 'object'")
        properties = parameters.get("properties", {})
        required = parameters.get("required", [])
        if not isinstance(properties, dict) or not isinstance(required, list):
            raise ValueError(f"Tool {name!r} has malformed argument properties")
        if any(not isinstance(key, str) for key in required):
            raise ValueError(f"Tool {name!r} has malformed required argument names")
        if any(key not in properties for key in required):
            raise ValueError(f"Tool {name!r} requires an undeclared argument")
        return name, parameters

    @classmethod
    def _has_schema_field(cls, tool: Dict[str, Any], field: str) -> bool:
        _, parameters = cls._function_schema(tool)
        properties = parameters["properties"]
        property_schema = properties.get(field)
        if property_schema is None:
            return False
        if not isinstance(property_schema, dict):
            raise ValueError(f"Tool argument {field!r} must declare a property schema")
        if property_schema.get("type") != "string":
            raise ValueError(
                f"Tool argument {field!r} must be declared with type 'string'"
            )
        return True

    @classmethod
    def _candidate_for_schema(
        cls,
        tool: Dict[str, Any],
        field: str,
        value: str,
    ) -> Optional[Tuple[str, Dict[str, str], Dict[str, Any]]]:
        tool_name, parameters = cls._function_schema(tool)
        properties = parameters["properties"]
        if field not in properties:
            return None
        if not cls._has_schema_field(tool, field):
            return None
        required = parameters.get("required", [])
        if any(
            argument not in _SUPPORTED_INPUT_FIELDS
            for argument in required
        ):
            return None
        arguments = {field: value}
        return tool_name, arguments, parameters

    @staticmethod
    def _validate_arguments(
        arguments: Dict[str, Any],
        schema: Dict[str, Any],
        tool_name: str,
    ) -> None:
        properties = schema["properties"]
        required = schema.get("required", [])
        missing = [key for key in required if key not in arguments]
        if missing:
            raise ValueError(
                f"Tool {tool_name!r} requires unsupported argument(s): "
                f"{', '.join(missing)}"
            )
        if schema.get("additionalProperties") is False:
            extra = set(arguments) - set(properties)
            if extra:
                raise ValueError(
                    f"Tool {tool_name!r} does not accept argument(s): "
                    f"{', '.join(sorted(extra))}"
                )
        for name, value in arguments.items():
            property_schema = properties.get(name)
            if not isinstance(property_schema, dict):
                raise ValueError(
                    f"Tool {tool_name!r} has no valid schema for argument {name!r}"
                )
            expected_type = property_schema.get("type")
            if expected_type != "string" or not isinstance(value, str):
                raise ValueError(
                    f"Tool argument {name!r} must be a string per the declared schema"
                )
            if not value.strip():
                raise ValueError(f"Tool argument {name!r} must not be empty")
            if "minLength" in property_schema and len(value) < property_schema["minLength"]:
                raise ValueError(
                    f"Tool argument {name!r} is shorter than its declared minLength"
                )
            if "maxLength" in property_schema and len(value) > property_schema["maxLength"]:
                raise ValueError(
                    f"Tool argument {name!r} exceeds its declared maxLength"
                )
            if "pattern" in property_schema and re.search(
                property_schema["pattern"], value
            ) is None:
                raise ValueError(
                    f"Tool argument {name!r} does not match its declared pattern"
                )
            if "enum" in property_schema and value not in property_schema["enum"]:
                raise ValueError(
                    f"Tool argument {name!r} is not one of its declared enum values"
                )
