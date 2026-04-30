import json
import structlog
import time
from google.adk.tools.tool_context import ToolContext
from pydantic import BaseModel, Field
from backend.schemas import PrioritizedGap
from backend.tools.llm_utils import (
    resolve_model, get_openai_client,
    build_json_schema, create_structured_request, parse_structured_response,
)

log = structlog.get_logger()

MAX_RETRIES = 2


# ── LLM output schema ────────────────────────────────────────────────────


class PriorityOutput(BaseModel):
    """Schema for prioritized skill gaps."""
    priorities: list[PrioritizedGap]


async def prioritise_skill_gaps(
    gap_skills: list[str],
    seniority_context: str,
    tool_context: ToolContext = None,
) -> list[dict]:
    """Rank skill gaps by their estimated impact on getting a role."""
    start_time = time.time()
    if not gap_skills:
        log.info("prioritise_skill_gaps_empty", seniority=seniority_context)
        _record_latency(tool_context, start_time)
        return []

    schema = build_json_schema(PriorityOutput)

    # Try multiple prompts if validation fails
    prompts = [
        f"""You are a career advisor. Rank these skill gaps by their impact on getting a {seniority_context} role.
Gap skills: {json.dumps(gap_skills)}

For each skill, estimate the match gain percentage if the candidate learns it.
Return ONLY a JSON object with a "priorities" key containing the ranked array (highest impact first):
{{"priorities": [
  {{"skill": "SkillName", "priority_rank": 1, "estimated_match_gain_pct": 15, "rationale": "Why this skill is most important"}}
]}}
""",
        # Simpler retry prompt
        f"""Rank these skills by importance for a {seniority_context} role: {json.dumps(gap_skills)}
Return JSON: {{"priorities": [{{"skill": "name", "priority_rank": 1, "estimated_match_gain_pct": 10, "rationale": "reason"}}]}}
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
            parsed = parse_structured_response(response)

            ranked = parsed.get("priorities", [])
            validated = [PrioritizedGap.model_validate(g) for g in ranked]

            log.info(
                "skill_gaps_prioritised",
                gap_count=len(gap_skills),
                ranked_count=len(validated),
                top_skill=validated[0].skill if validated else None,
                attempt=attempt,
            )

            if tool_context:
                tool_context.state["prioritized_gaps"] = [g.model_dump() for g in validated]

            _record_latency(tool_context, start_time)
            return [g.model_dump() for g in validated]

        except Exception as e:
            log.warning(
                "prioritise_skill_gaps_failed",
                attempt=attempt,
                error=str(e)[:200],
            )
            if attempt >= MAX_RETRIES:
                break

    # All retries exhausted — deterministic fallback ranking
    log.error("prioritise_skill_gaps_fallback", gap_count=len(gap_skills))
    gain_per_skill = max(5, 100 // max(len(gap_skills), 1))
    fallback = [
        {
            "skill": skill,
            "priority_rank": i + 1,
            "estimated_match_gain_pct": max(5, gain_per_skill - i * 2),
            "rationale": f"{skill} is a required skill gap for this {seniority_context} role.",
        }
        for i, skill in enumerate(gap_skills)
    ]

    if tool_context:
        tool_context.state["prioritized_gaps"] = fallback

    _record_latency(tool_context, start_time)
    return fallback


def _record_latency(tool_context: ToolContext, start_time: float):
    """Record tool execution latency in tool_context state."""
    if tool_context is None:
        return
    latency_ms = int((time.time() - start_time) * 1000)
    tool_context.state["temp:prioritise_skill_gaps_latency_ms"] = latency_ms
