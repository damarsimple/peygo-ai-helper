import hashlib
import structlog
import time
from diskcache import Cache
import json
import os
from pydantic import BaseModel, Field
from google.adk.tools.tool_context import ToolContext
from backend.tools.llm_utils import (
    resolve_model, get_openai_client,
    build_json_schema, create_structured_request, parse_structured_response,
)

log = structlog.get_logger()

_resource_cache = Cache("/tmp/pelgo_cache")

# Tavily API key loaded from environment (.env file)
# Falls back gracefully to DuckDuckGo + LLM if not set


# ── LLM output schemas ──────────────────────────────────────────────────


class HourEstimateOutput(BaseModel):
    """Schema for LLM hours estimation."""
    hours: list[int]


class ResourceOutput(BaseModel):
    """Schema for LLM-generated placeholder resources."""
    resources: list[dict]  # dict keeps it flexible for the downstream parser


async def research_skill_resources(
    skill_name: str,
    seniority_context: str,
    tool_context: ToolContext = None,
) -> list[dict]:
    """Search the web for high-quality learning resources for a given skill.

    Queries Tavily for courses/tutorials matching the skill and seniority level.
    Falls back to DuckDuckGo if Tavily fails, then LLM-generated placeholder list.
    Results are cached by (skill_name, seniority_context) — 24-hour TTL.

    Args:
        skill_name: The skill to find resources for (e.g., "Kubernetes").
        seniority_context: Career level for tailoring results
            (e.g., "junior", "mid", "senior", "lead").

    Returns:
        list[dict]: Each dict has "title", "url", "estimated_hours",
            "type" (course/project/cert/doc), "relevance_score".
    """
    start_time = time.time()

    cache_key = f"skill:{hashlib.md5(f'{skill_name}:{seniority_context}'.encode()).hexdigest()}"
    cached = _resource_cache.get(cache_key)
    if cached is not None:
        log.debug("research_cache_hit", skill=skill_name, cache_key=cache_key)
        # Still record latency for cache hits
        _record_latency(tool_context, start_time)
        return cached

    resources = []

    # Primary: Tavily search
    try:
        from tavily import TavilyClient
        tavily = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))
        response = tavily.search(
            query=f"learn {skill_name} {seniority_context} programming course tutorial",
            max_results=5,
            search_depth="basic"
        )
        for i, r in enumerate(response.get("results", [])[:3]):
            resources.append({
                "title": r.get("title", f"{skill_name} resource {i+1}"),
                "url": r.get("url", ""),
                "estimated_hours": 0,  # Will be estimated by LLM
                "type": _classify_resource_type(r.get("title", "")),
                "relevance_score": round(1.0 - i * 0.2, 2),
            })
        log.info("research_tavily_success", skill=skill_name, results_found=len(resources))
    except Exception as e:
        log.warning("research_tavily_failed", skill=skill_name, error=str(e)[:200])

        # Fallback: DuckDuckGo search
        try:
            from duckduckgo_search import DDGS
            with DDGS() as d:
                results = list(d.text(
                    f"learn {skill_name} {seniority_context} programming course tutorial",
                    max_results=5
                ))
            for i, r in enumerate(results[:3]):
                resources.append({
                    "title": r.get("title", f"{skill_name} resource {i+1}"),
                    "url": r.get("href", ""),
                    "estimated_hours": 0,  # Will be estimated by LLM
                    "type": _classify_resource_type(r.get("title", "")),
                    "relevance_score": round(1.0 - i * 0.2, 2),
                })
            log.info("research_ddg_success", skill=skill_name, results_found=len(resources))
        except Exception as e2:
            log.warning("research_ddg_failed", skill=skill_name, error=str(e2)[:200])

    if not resources:
        log.info("research_llm_fallback", skill=skill_name)
        resources = await _generate_placeholder_resources(skill_name, seniority_context)
    else:
        # Estimate hours via LLM for Tavily/DDG results
        resources = await _llm_estimate_hours(resources, skill_name, seniority_context)

    _resource_cache.set(cache_key, resources, expire=86400)

    # Record latency in tool_context for trace collection
    _record_latency(tool_context, start_time)

    log.info("research_complete", skill=skill_name, resource_count=len(resources))
    return resources


def _record_latency(tool_context: ToolContext, start_time: float):
    """Record tool execution latency in tool_context state."""
    if tool_context is None:
        return
    latency_ms = int((time.time() - start_time) * 1000)
    tool_context.state["temp:research_skill_resources_latency_ms"] = latency_ms
    log.debug("research_latency_recorded", latency_ms=latency_ms)


async def _llm_estimate_hours(resources: list[dict], skill: str, seniority: str) -> list[dict]:
    """Use LLM to estimate hours for a batch of resources based on skill, seniority, and resource type."""
    if not resources:
        return resources

    # Build resource summary list outside the f-string to avoid f-string brace-escaping pitfalls
    res_summary = [{"title": r["title"], "type": r["type"]} for r in resources]
    res_json = json.dumps(res_summary, indent=2)

    prompt = f"""Estimate realistic self-study hours for a {seniority} engineer to reach
    job-ready proficiency in each resource below. Assume ~2 hours/day study pace.

    Skill being learned: {skill}
    Learner level: {seniority}

    Resources:
    {res_json}

    ESTIMATION GUIDELINES:
    - "doc": official docs or short tutorials → 2–10 hours
    - "course": structured online course → 8–40 hours  
    - "cert": certification prep → 20–80 hours
    - "project": build-from-scratch project → 10–50 hours
    - Adjust DOWN for senior/lead (faster ramp-up, skip basics).
    - Adjust UP for complex skills (distributed systems, ML, security).
    - Adjust DOWN for simple tooling (linters, formatters, CLI tools).

    Return ONLY — no markdown, no explanation:
    {{"hours": [<int>, <int>, ...]}}

    Return exactly {len(resources)} integers in the same order as the input list.
    """

    try:
        client = get_openai_client()
        schema = {
            "type": "object",
            "properties": {
                "hours": {
                    "type": "array",
                    "items": {"type": "integer", "minimum": 1, "maximum": 100},
                }
            },
            "required": ["hours"],
            "additionalProperties": False,
        }
        payload = create_structured_request(
            messages=[{"role": "user", "content": prompt}],
            schema=schema,
            temperature=0.6,
        )
        response = await client.chat.completions.create(**payload)
        estimates = parse_structured_response(response)
        hours_list = estimates.get("hours", [])

        for i, r in enumerate(resources):
            if i < len(hours_list):
                try:
                    r["estimated_hours"] = max(1, min(100, int(hours_list[i])))
                except (ValueError, TypeError):
                    pass
    except Exception as e:
        log.warning("llm_hours_estimation_failed", skill=skill, error=str(e)[:100])

    return resources


def _classify_resource_type(title: str) -> str:
    t = title.lower()
    if "course" in t or "tutorial" in t:
        return "course"
    elif "cert" in t or "certificate" in t:
        return "cert"
    elif "project" in t or "github" in t:
        return "project"
    return "doc"


async def _generate_placeholder_resources(skill_name: str, seniority_context: str) -> list[dict]:
    """Suggests curated resources from known high-quality platforms as a fallback."""
    prompt = f"""Suggest 3 high-quality learning resources for: {skill_name} (Level: {seniority_context}).

    IMPORTANT — URL RULES:
    - Only use URLs from these known-good domains:
    coursera.org, udemy.com, edx.org, pluralsight.com, linkedin.com/learning,
    developer.mozilla.org, docs.python.org, kubernetes.io/docs, docs.docker.com,
    learn.microsoft.com, cloud.google.com/learn, aws.amazon.com/training,
    roadmap.sh, freecodecamp.org, missing.csail.mit.edu
    - If you are not confident in the exact URL path, use the domain root
    (e.g. "https://coursera.org") rather than inventing a path.
    - Do NOT fabricate course slugs, IDs, or deep paths.

    RESOURCE SELECTION RULES:
    - Prefer free or widely accessible resources over paywalled ones.
    - Prefer official documentation for infrastructure/platform skills
    (Kubernetes, Docker, AWS, GCP).
    - Prefer structured courses for language or framework skills
    (Python, React, SQL).
    - At least 1 of the 3 resources should be free.

    Return ONLY this JSON — no markdown:
    {{
    "resources": [
        {{
        "title": "Descriptive course or doc title",
        "url": "https://known-domain.com/path-if-confident",
        "estimated_hours": 12,
        "type": "course|project|cert|doc",
        "relevance_score": 0.95
        }}
    ]
    }}
    """
    client = get_openai_client()
    schema = {
        "type": "object",
        "properties": {
            "resources": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "url": {"type": "string"},
                        "estimated_hours": {"type": "integer", "minimum": 1, "maximum": 100},
                        "type": {"type": "string", "enum": ["course", "project", "cert", "doc"]},
                        "relevance_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    },
                    "required": ["title", "url", "type"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["resources"],
        "additionalProperties": False,
    }
    payload = create_structured_request(
        messages=[{"role": "user", "content": prompt}],
        schema=schema,
        temperature=0.3,
    )
    response = await client.chat.completions.create(**payload)
    parsed = parse_structured_response(response)

    resources = []
    for item in parsed.get("resources", [])[:3]:
        resources.append({
            "title": item.get("title", f"{skill_name} guide"),
            "url": item.get("url", ""),
            "estimated_hours": int(item.get("estimated_hours", 0)),
            "type": item.get("type", "doc") if item.get("type") in ("course", "project", "cert", "doc") else "doc",
            "relevance_score": float(item.get("relevance_score", 0.7)),
        })

    # Use LLM to estimate hours for resources where not provided
    resources_needing_hours = [r for r in resources if r["estimated_hours"] == 0]
    if resources_needing_hours:
        resources_needing_hours = await _llm_estimate_hours(resources_needing_hours, skill_name, seniority_context)
        # Merge back
        idx = 0
        for r in resources:
            if r["estimated_hours"] == 0 and idx < len(resources_needing_hours):
                r["estimated_hours"] = resources_needing_hours[idx]["estimated_hours"]
                idx += 1

    return resources
