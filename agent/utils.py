from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping
import json
import os
from transformers import AutoTokenizer

model = os.getenv("MODEL_NAME")
model_id = model
if model and model.startswith("glm"):
    model_id = "zai-org/" + model



USER_HOME = Path.home()
WORKSPACE = Path(__file__).parent.parent

def safe_path(p: str) -> Path:
    path = (WORKSPACE / p).resolve()
    if not path.is_relative_to(WORKSPACE):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


class TokenCounter:
    def __init__(self, model_id: str):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
        )

    def count_text(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    # zai-org/GLM-5
    def count_chat(self, messages, tools) -> int:
        ids = self.tokenizer.apply_chat_template(
            messages,
            tools = tools,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors=None,
        )
        return len(ids)


def _json_safe(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    elif hasattr(value, "dict") and callable(value.dict):
        value = value.dict()

    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _normalize_message(message: Any) -> dict[str, Any]:
    normalized = _json_safe(message)
    if not isinstance(normalized, dict):
        raise ValueError("Each message must serialize to a JSON object")
    return normalized


def _normalize_messages(messages: Iterable[Any]) -> list[dict[str, Any]]:
    return [_normalize_message(message) for message in messages]


def _normalize_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    normalized = _json_safe(payload)
    if not isinstance(normalized, dict):
        raise ValueError("Conversation history payload must serialize to a JSON object")

    messages = normalized.get("messages", [])
    if not isinstance(messages, list):
        raise ValueError("Conversation history payload must contain a 'messages' list")

    normalized["messages"] = _normalize_messages(messages)

    tools = normalized.get("tools")
    if tools is not None and not isinstance(tools, list):
        raise ValueError("Conversation history payload field 'tools' must be a list")

    normalized["message_count"] = len(normalized["messages"])
    return normalized


class ConversationHistoryStore:
    def __init__(self, default_path: str = ".agent/history/conversation.json"):
        self.default_path = default_path

    def save(
        self,
        messages: Iterable[Any],
        path: str | None = None,
        *,
        metadata: Mapping[str, Any] | None = None,
        tools: list[Any] | None = None,
    ) -> Path:
        return self.save_payload(
            {
                "messages": _normalize_messages(messages),
                "tools": _json_safe(tools) if tools is not None else None,
                "metadata": _json_safe(metadata) if metadata is not None else None,
            },
            path,
        )

    def save_payload(
        self,
        payload: Mapping[str, Any],
        path: str | None = None,
    ) -> Path:
        target = safe_path(path or self.default_path)
        target.parent.mkdir(parents=True, exist_ok=True)

        normalized_payload = _normalize_payload(payload)
        stored_payload = {
            "version": 1,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "workspace": str(WORKSPACE),
            **normalized_payload,
        }

        target.write_text(
            json.dumps(stored_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return target

    def load_payload(self, path: str | None = None) -> dict[str, Any]:
        source = safe_path(path or self.default_path)
        payload = json.loads(source.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return {
                "version": 0,
                "message_count": len(payload),
                "messages": _normalize_messages(payload),
            }
        if not isinstance(payload, dict):
            raise ValueError("Conversation history file must contain a JSON object or list")
        return _normalize_payload(payload)

    def load_messages(self, path: str | None = None) -> list[dict[str, Any]]:
        payload = self.load_payload(path)
        return payload["messages"]


conversation_history_store = ConversationHistoryStore()


def save_conversation_history(
    messages: Iterable[Any],
    path: str | None = None,
    *,
    metadata: Mapping[str, Any] | None = None,
    tools: list[Any] | None = None,
) -> Path:
    return conversation_history_store.save(
        messages,
        path,
        metadata=metadata,
        tools=tools,
    )


def save_conversation_history_payload(
    payload: Mapping[str, Any],
    path: str | None = None,
) -> Path:
    return conversation_history_store.save_payload(payload, path)


def load_conversation_history(path: str | None = None) -> list[dict[str, Any]]:
    return conversation_history_store.load_messages(path)


def load_conversation_history_payload(path: str | None = None) -> dict[str, Any]:
    return conversation_history_store.load_payload(path)
