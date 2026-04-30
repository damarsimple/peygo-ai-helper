import hashlib
import httpx
import time
from bs4 import BeautifulSoup
from diskcache import Cache
from pydantic import ValidationError
from google.adk.tools.tool_context import ToolContext
import json
import structlog
from backend.tools.llm_utils import (
    resolve_model, get_openai_client,
    build_json_schema, create_structured_request, parse_structured_response,
)

log = structlog.get_logger()

_cache = Cache("/tmp/pelgo_cache")

MAX_RETRIES = 2


async def extract_jd_requirements(
    job_url_or_text: str,
    tool_context: ToolContext = None,
) -> dict:
    """Extract structured job requirements from a job description URL or raw text."""
    start_time = time.time()
    is_url = job_url_or_text.startswith("http")

    if is_url:
        url_hash = hashlib.sha256(job_url_or_text.encode()).hexdigest()
        cache_key = f"jd:url:{url_hash}"
    else:
        text_hash = hashlib.sha256(job_url_or_text.encode()).hexdigest()
        cache_key = f"jd:text:{text_hash}"

    cached = _cache.get(cache_key)
    if cached is not None:
        log.debug("jd_cache_hit", cache_key=cache_key)
        _record_latency(tool_context, start_time)
        return cached

    if is_url:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(job_url_or_text)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            text = soup.get_text(separator="\n")
    else:
        text = job_url_or_text

    from backend.schemas import JDRequirements

    # Build the schema once — used for all retry attempts
    schema = build_json_schema(JDRequirements)

    # Try multiple prompts if validation fails
    prompts = [
        # Standard prompt
        f"""Extract structured job requirements from the following job description.
Return ONLY a JSON object with these fields:
{{
  "required_skills": ["skill1", "skill2"],
  "nice_to_have_skills": ["skill3"],
  "seniority_level": "junior|mid|senior|lead",
  "domain": "domain_name",
  "responsibilities": ["responsibility1"]
}}

Job description:
{text[:8000]}
""",
        # Stricter prompt — enumerate types
        f"""Extract job requirements. Return a JSON object:
{{
  "required_skills": ["skill1", "skill2"],
  "nice_to_have_skills": ["skill3"],
  "seniority_level": "mid",
  "domain": "software_engineering",
  "responsibilities": ["design", "implement"]
}}

Job description:
{text[:8000]}
""",
        # Fallback — minimal prompt
        f"""From this text, list the required skills, optional skills, seniority level,
domain, and responsibilities as JSON:

{text[:4000]}

JSON keys: required_skills (array), nice_to_have_skills (array),
seniority_level (string), domain (string), responsibilities (array)
""",
    ]

    for attempt, prompt in enumerate(prompts, 1):
        try:
            client = get_openai_client()
            payload = create_structured_request(
                messages=[{"role": "user", "content": prompt}],
                schema=schema,
                temperature=0.1,
            )
            response = await client.chat.completions.create(**payload)
            extracted = parse_structured_response(response)

            validated = JDRequirements.model_validate(extracted)
            result = validated.model_dump()

            _cache.set(cache_key, result, expire=86400)
            log.info(
                "jd_extract_success",
                cache_key=cache_key,
                attempt=attempt,
                required_skills=len(result.get("required_skills", [])),
            )
            _record_latency(tool_context, start_time)
            return result

        except (ValidationError, json.JSONDecodeError) as e:
            log.warning(
                "jd_extract_validation_failed",
                attempt=attempt,
                max_retries=MAX_RETRIES + 1,
                error=str(e)[:200],
            )
            if attempt >= MAX_RETRIES + 1:
                break  # All prompts exhausted

    # All retries exhausted — return a safe fallback
    fallback = {
        "required_skills": [],
        "nice_to_have_skills": [],
        "seniority_level": "mid",
        "domain": "unknown",
        "responsibilities": [],
        "error": "Schema validation failed after retries; returned fallback",
        "fallback": True,
    }
    log.error("jd_extract_fallback", cache_key=cache_key)
    _record_latency(tool_context, start_time)
    return fallback


def _record_latency(tool_context: ToolContext, start_time: float):
    """Record tool execution latency in tool_context state."""
    if tool_context is None:
        return
    latency_ms = int((time.time() - start_time) * 1000)
    tool_context.state["temp:extract_jd_requirements_latency_ms"] = latency_ms
