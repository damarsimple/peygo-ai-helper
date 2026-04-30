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
        # Attempt 1 — Full structured extraction with examples
        f"""You are a technical recruiter parsing a job description.
    Extract structured requirements and return ONLY a valid JSON object.

    EXTRACTION RULES:
    - required_skills: Hard requirements only. Normalise aliases
    (e.g. "C/C++" → ["C", "C++"], "Node" → "Node.js").
    Include languages, frameworks, tools, platforms, and methodologies.
    - nice_to_have_skills: Explicitly optional or "bonus" skills only.
    Do NOT include skills from required_skills here.
    - seniority_level: Map experience years to one of: junior (<2yr), mid (2–5yr),
    senior (5–9yr), lead (9+yr). If not stated, infer from tone and responsibilities.
    - domain: Single slug, e.g. "backend", "frontend", "fullstack", "embedded",
    "data_engineering", "devops", "mobile", "ml_ai", "security".
    - responsibilities: Max 6 items. Use imperative verb phrases ("Design APIs",
    not "The candidate will design APIs").

    Return ONLY this JSON — no markdown, no explanation:
    {{
    "required_skills": ["skill1", "skill2"],
    "nice_to_have_skills": ["skill3"],
    "seniority_level": "junior|mid|senior|lead",
    "domain": "domain_slug",
    "responsibilities": ["Verb phrase 1", "Verb phrase 2"]
    }}

    JOB DESCRIPTION:
    {text[:8000]}
    """,

        # Attempt 2 — Relaxed: flatten everything into required, skip nice-to-have
        # Used when attempt 1 fails validation (e.g. model put skills in wrong bucket)
        f"""Parse this job posting. If you cannot separate required vs optional skills,
    put ALL skills in required_skills and leave nice_to_have_skills empty.

    Allowed seniority_level values: "junior", "mid", "senior", "lead" — pick the closest.
    Allowed domain values: any single lowercase slug describing the engineering domain.

    Return ONLY:
    {{
    "required_skills": ["skill1", "skill2"],
    "nice_to_have_skills": [],
    "seniority_level": "mid",
    "domain": "backend",
    "responsibilities": ["Responsibility 1"]
    }}

    JOB DESCRIPTION:
    {text[:8000]}
    """,

        # Attempt 3 — Ultra-minimal: short text window, no schema pressure
        # Last resort when the text is noisy (scraping artifacts, login walls, etc.)
        f"""Extract only the most important technical skills and job level from this text.
    The text may be noisy or incomplete — do your best.

    Return ONLY valid JSON with these exact keys. Use empty arrays if unsure:
    {{
    "required_skills": [],
    "nice_to_have_skills": [],
    "seniority_level": "mid",
    "domain": "unknown",
    "responsibilities": []
    }}

    TEXT (first 5000 chars):
    {text[:5000]}
    """,
    ]

    for attempt, prompt in enumerate(prompts, 1):
        try:
            client = get_openai_client()
            payload = create_structured_request(
                messages=[{"role": "user", "content": prompt}],
                schema=schema,
                temperature=1
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
