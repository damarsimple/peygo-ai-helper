"""Shared LLM client utilities.

Centralises model resolution and LLM client creation so every tool
uses the same logic without copy-pasting.

Uses LiteLLM for consistency across all services (worker and api).
With llguidance on vLLM (json_schema response format), responses are
guaranteed schema-valid JSON — no fence-stripping or array-unwrapping needed.
"""
import os
from typing import Any

import litellm

from pydantic import BaseModel


def resolve_model() -> str:
    """Return the model name for tool LLM calls.

    When using LiteLLM with custom base_url, use the raw model identifier
    without the 'openai/' prefix (LiteLLM adds provider prefix for routing).
    """
    model = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
    if os.getenv("OPENAI_BASE_URL") and model.startswith("openai/"):
        model = model[7:]  # strip "openai/"
    return model


def get_llm_client():
    """Return a configured LiteLLM client wrapper.

    LiteLLM auto-appends /v1 to base_url, so no manual appending needed.
    Returns a callable that mimics OpenAI SDK interface for compatibility.
    """
    api_key = os.getenv("OPENAI_API_KEY", "dummy")
    base_url = os.getenv("OPENAI_BASE_URL", None)

    class LiteLLMClient:
        def __init__(self, api_key, base_url):
            self.api_key = api_key
            self.base_url = base_url

        @property
        def chat(self):
            return self

        @property
        def completions(self):
            return self

        async def create(self, **kwargs):
            """Mimics OpenAI SDK's client.chat.completions.create() interface."""
            # LiteLLM uses OpenAI SDK under the hood, which doesn't auto-append /v1
            # So we need to ensure api_base ends with /v1 for vLLM compatibility
            api_base = self.base_url
            if api_base and not api_base.endswith("/v1"):
                api_base = api_base.rstrip("/") + "/v1"
            # Set 21k output tokens (leaving 189k for input in 210k context)
            if "max_tokens" not in kwargs:
                kwargs["max_tokens"] = 190000
            return await litellm.acompletion(
                api_key=self.api_key,
                api_base=api_base,
                custom_llm_provider="openai",
                **kwargs
            )

    return LiteLLMClient(api_key, base_url)


# Backward compatibility alias
get_openai_client = get_llm_client


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

    return _flatten_schema(schema)



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
    temperature: float = 0.6,
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
            "type": "json_object",
            "schema": schema,
            
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
    if not content or not content.strip():
        raise ValueError("LLM returned empty content")
    return json.loads(content.strip())
