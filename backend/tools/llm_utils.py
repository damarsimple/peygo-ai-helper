"""Shared LLM client utilities.

Centralises model resolution and OpenAI client creation so every tool
uses the same logic without copy-pasting.
"""
import os
import openai


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


def extract_json_from_response(raw: str):
    """Strip markdown code fences from an LLM response, return clean string."""
    import re
    cleaned = raw.strip()
    # Try to extract JSON from code fences (handles ```json, ```, nested, etc.)
    match = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', cleaned, re.DOTALL)
    if match:
        return match.group(1).strip()
    return cleaned


def unwrap_json_array(parsed) -> list:
    """If the parsed JSON is a dict wrapping a single array value, unwrap it.

    Many models return {"items": [...]} when asked for an array with
    response_format=json_object. This extracts the array transparently.
    """
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        # Find the first value that is a list
        for v in parsed.values():
            if isinstance(v, list):
                return v
        # No list found — return dict values as fallback
        return list(parsed.values())
    return []
