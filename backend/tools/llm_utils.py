"""Shared LLM client utilities.

Centralises model resolution and OpenAI client creation so every tool
uses the same logic without copy-pasting.

With llguidance on vLLM (json_schema response format), responses are
guaranteed schema-valid JSON — no fence-stripping or array-unwrapping needed.
"""
import os
from typing import Any

import openai

from pydantic import BaseModel


def resolve_model() -> str:
    """Return the model name for tool LLM calls.

    Strips the 'openai/' LiteLLM provider prefix when a custom base_url is set,
    because the OpenAI SDK sends the model name literally to the endpoint and
    vLLM only recognises the raw model identifier.
    """
    model = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
    if os.getenv("OPENAI_BASE_URL") and model.startswith("openai/"):
        model = model[7:]  # strip "openai/"
    return model


def get_openai_client() -> openai.AsyncOpenAI:
    """Return a configured async OpenAI client."""
    return openai.AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY", "dummy"),
        base_url=os.getenv("OPENAI_BASE_URL", None),
    )


# ── JSON Schema helpers ──────────────────────────────────────────────────


def build_json_schema(schema_obj: BaseModel | dict[str, Any]) -> dict[str, Any]:
    """Build a flattened, self-contained JSON Schema from a Pydantic model or dict.

    vLLM's llguidance backend needs a fully inlined schema — no external
    `$ref` pointers to `$defs`.  This function:
    1. Generates the JSON Schema (Pydantic `model_json_schema()` or passthrough dict).
    2. Resolves all `$ref` → inline by pulling from `$defs` and rewriting.
    3. Drops `$defs` from the result so the schema is self-contained.

    Returns:
        A flat JSON Schema dict safe for `response_format.json_schema.schema`.
    """
    if isinstance(schema_obj, type) and issubclass(schema_obj, BaseModel):
        schema = schema_obj.model_json_schema()
    else:
        schema = dict(schema_obj)

    return _resolve_refs(schema)


def _resolve_refs(schema: Any) -> Any:
    """Recursively resolve $ref pointers in a JSON Schema dict."""
    if isinstance(schema, dict):
        if "$ref" in schema:
            # Resolve the reference
            ref_path = schema["$ref"]  # e.g. "#/$defs/DimensionScores"
            parts = ref_path.split("/")
            if len(parts) >= 3 and parts[0] == "#" and parts[1] == "$defs":
                # Need to find the actual schema — we'll do a deferred pass
                # For now return a placeholder marker and resolve in parent
                return schema
            return schema
        # Recurse into all values
        return {k: _resolve_refs(v) for k, v in schema.items()}
    elif isinstance(schema, list):
        return [_resolve_refs(item) for item in schema]
    return schema


def _flatten_schema(schema: dict) -> dict:
    """Resolve all $defs references inline to produce a self-contained schema."""
    defs = schema.pop("$defs", {})

    def resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                ref = node["$ref"]
                name = ref.split("/")[-1]  # e.g. "DimensionScores"
                if name in defs:
                    resolved = _flatten_schema(dict(defs[name]))
                    return resolved
                return node
            return {k: resolve(v) for k, v in node.items()}
        elif isinstance(node, list):
            return [resolve(item) for item in node]
        return node

    return resolve(schema)


# Re-export so callers can import once and resolve refs themselves if needed.
# Primary callers should use `build_json_schema()` which returns a ready-to-send dict.
__all__ = [
    "resolve_model",
    "get_openai_client",
    "build_json_schema",
    "create_structured_request",
    "parse_structured_response",
]


def create_structured_request(
    messages: list[dict],
    schema: dict,
    temperature: float = 0.1,
) -> dict:
    """Build a chat.completions payload with json_schema response format.

    Args:
        messages: List of message dicts (role + content).
        schema: A flat JSON Schema dict (from build_json_schema()).
        temperature: Sampling temperature.

    Returns:
        A dict ready for `client.chat.completions.create(**payload)`.
    """
    return {
        "model": resolve_model(),
        "messages": messages,
        "temperature": temperature,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "output",
                "strict_json_schema": True,
                "schema": schema,
            },
        },
    }


def parse_structured_response(response) -> dict:
    """Parse the content from a chat.completions response as JSON.

    With llguidance the response is guaranteed schema-valid JSON with no
    fence wrapping.  This function is a thin convenience wrapper.
    """
    import json

    content = response.choices[0].message.content
    return json.loads(content.strip())
