import hashlib
import structlog
import time
from diskcache import Cache
import json
import os
from google.adk.tools.tool_context import ToolContext
from backend.tools.llm_utils import resolve_model, get_openai_client, extract_json_from_response, unwrap_json_array

log = structlog.get_logger()

_resource_cache = Cache("/tmp/pelgo_cache")

# Tavily API key loaded from environment (.env file)
# Falls back gracefully to DuckDuckGo + LLM if not set


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
    
    prompt = f"""Estimate the time needed (in hours) to complete each learning resource for a {seniority}-level learner studying {skill}.

Resources:
{json.dumps([{"title": r["title"], "type": r["type"]} for r in resources], indent=2)}

Return ONLY a JSON object with an "hours" key containing estimated hours (integers between 1 and 100) in the same order:
{{"hours": [12, 25, 8]}}

Guidelines:
- "doc" (documentation/tutorial): 2-8 hours
- "course" (online course): 10-40 hours  
- "cert" (certification): 20-80 hours
- "project" (hands-on project): 15-50 hours
- Adjust for seniority: junior needs more time, lead needs less
"""
    try:
        client = get_openai_client()
        response = await client.chat.completions.create(
            model=resolve_model(),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        content = extract_json_from_response(response.choices[0].message.content)
        estimates = json.loads(content)
        # Unwrap object wrapper if present (json_object mode returns {"hours": [...]})
        if isinstance(estimates, dict):
            for v in estimates.values():
                if isinstance(v, list):
                    estimates = v
                    break
        if isinstance(estimates, list):
            for i, r in enumerate(resources):
                if i < len(estimates):
                    try:
                        r["estimated_hours"] = max(1, min(100, int(estimates[i])))
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
    prompt = f"""Suggest 3 high-quality, REAL-WORLD learning resources for: {skill_name} (Level: {seniority_context}).
Focus on platforms like Coursera, Udemy, edX, Pluralsight, or official documentation.

Return ONLY a JSON object with a "resources" key:
{{
  "resources": [
    {{
      "title": "Clear Course or Doc Title",
      "url": "https://platform.com/specific-course-path",
      "estimated_hours": 12,
      "type": "course",
      "relevance_score": 0.95
    }}
  ]
}}
Types: course, project, cert, doc.
"""
    client = get_openai_client()
    response = await client.chat.completions.create(
        model=resolve_model(),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        response_format={"type": "json_object"},
    )
    raw = extract_json_from_response(response.choices[0].message.content)
    try:
        parsed = json.loads(raw)
        items = unwrap_json_array(parsed)
        resources = [
            {
                "title": item.get("title", f"{skill_name} guide"),
                "url": item.get("url", ""),
                "estimated_hours": int(item.get("estimated_hours", 0)),  # Default 0, will be estimated by LLM
                "type": item.get("type", "doc") if item.get("type") in ("course","project","cert","doc") else "doc",
                "relevance_score": float(item.get("relevance_score", 0.7)),
            }
            for item in items[:3]
        ]
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
    except Exception:
        # Fallback: create basic resource and estimate hours via LLM
        fallback = [{
            "title": f"Learn {skill_name}",
            "url": f"https://www.google.com/search?q=learn+{skill_name.replace(' ', '+')}",
            "estimated_hours": 0,
            "type": "doc",
            "relevance_score": 0.5,
        }]
        return await _llm_estimate_hours(fallback, skill_name, seniority_context)
