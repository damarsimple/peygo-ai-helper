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
        # Attempt 1 — Full context with concrete scoring criteria
        f"""You are a senior technical recruiter ranking skill gaps by their impact on
    candidate success in a {seniority_context} role.

    SKILL GAPS TO RANK:
    {json.dumps(gap_skills)}

    RANKING CRITERIA (in priority order):
    1. Is the skill explicitly required (not just nice-to-have)?
    2. How frequently is this skill tested in interviews for {seniority_context} roles?
    3. How long does it typically take to reach job-ready proficiency?
    (Skills with shorter ramp-up times → higher priority, faster score improvement)
    4. Is the skill foundational (blocks learning other gaps) or standalone?

    ESTIMATED MATCH GAIN RULES:
    - estimated_match_gain_pct represents how much the overall match score would improve
    if the candidate closed this single gap, assuming all other gaps remain open.
    - Values should be realistic: typically 5–25% per skill.
    - The sum of all estimated_match_gain_pct values MUST NOT exceed 100.
    - Assign higher gains to foundational or high-weight required skills.

    Return ONLY a JSON object — no markdown, no explanation:
    {{
    "priorities": [
        {{
        "skill": "SkillName",
        "priority_rank": 1,
        "estimated_match_gain_pct": 18,
        "rationale": "Required for [specific responsibility]. Most {seniority_context} interviews test this directly. Ramp-up is ~4 weeks."
        }}
    ]
    }}

    Rank ALL {len(gap_skills)} skills. Highest impact first (priority_rank 1 = most important).
    """,

        # Attempt 2 — Simplified, avoids sum-constraint which may have caused JSON failure
        f"""Rank these skill gaps by importance for a {seniority_context} engineering role.
    Prioritise: (1) explicitly required skills, (2) frequently interviewed topics,
    (3) foundational skills that unblock other gaps.

    Gaps: {json.dumps(gap_skills)}

    For estimated_match_gain_pct: assign 5–20 per skill. Total across all skills ≤ 100.

    Return ONLY:
    {{
    "priorities": [
        {{
        "skill": "Name",
        "priority_rank": 1,
        "estimated_match_gain_pct": 15,
        "rationale": "One sentence: why this gap matters most for this role level."
        }}
    ]
    }}
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
