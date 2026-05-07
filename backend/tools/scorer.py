import hashlib
import json
import structlog
import time
from diskcache import Cache
from google.adk.tools.tool_context import ToolContext
from pydantic import BaseModel, Field
from backend.schemas import CandidateProfile
from backend.tools.llm_utils import (
    resolve_model, get_openai_client,
    build_json_schema, create_structured_request, parse_structured_response,
)

log = structlog.get_logger()

_cache = Cache("/tmp/pelgo_cache")


# ── LLM output schema ────────────────────────────────────────────────────


class ScoreOutput(BaseModel):
    """Schema for the LLM scoring response."""
    matched_skills: list[str]
    gap_skills: list[str]
    required_match_ratio: float = Field(ge=0.0, le=1.0)
    nice_to_have_match_ratio: float = Field(ge=0.0, le=1.0)
    reasoning: str = ""


async def score_candidate_against_requirements(
    candidate_profile: dict,
    requirements: dict,
    raw_resume_text: str = "",
    tool_context: ToolContext = None,
) -> dict:
    """Score a candidate's profile against extracted job-description requirements."""
    start_time = time.time()

    # ── Caching ────────────────────────────────────────────────────────
    input_str = f"{json.dumps(candidate_profile, sort_keys=True)}:{json.dumps(requirements, sort_keys=True)}:{raw_resume_text}"
    cache_key = f"score:{hashlib.sha256(input_str.encode()).hexdigest()}"

    cached = _cache.get(cache_key)
    if cached is not None:
        log.debug("score_cache_hit", cache_key=cache_key)
        _record_latency(tool_context, start_time)
        # Still need to populate session state
        if tool_context:
            tool_context.state["score"] = cached
            tool_context.state["gap_skills"] = cached.get("gap_skills", [])
        return cached
    # ── Semantic Skill Matching (LLM-powered, single call) ──────────
    required = requirements.get("required_skills", [])
    nice_to_have = requirements.get("nice_to_have_skills", [])

    # Build comprehensive prompt with full context
    prompt = f"""You are a technical recruiter scoring a candidate's fit against job requirements.
    Use SEMANTIC and CONTEXTUAL matching — not just exact string comparison.

    ━━━ CANDIDATE ━━━
    Profile:
    {json.dumps(candidate_profile, indent=2)}

    Resume text (authoritative — use this to find skills not listed in the profile):
    {raw_resume_text[:3000] if raw_resume_text else "(not provided)"}

    ━━━ JOB REQUIREMENTS ━━━
    {json.dumps(requirements, indent=2)}

    ━━━ MATCHING INSTRUCTIONS ━━━
    1. For each skill in required_skills, determine if the candidate demonstrates it
    in either their profile OR resume text. Consider:
    - Direct mentions (exact or aliased, e.g. "Postgres" matches "PostgreSQL")
    - Implied proficiency (e.g. "built REST APIs in Django" implies "REST" and "Python")
    - Related but weaker signals (e.g. "Arduino" partially matches "Embedded Systems")
    Only list a skill in matched_skills if you are confident the candidate has it.
    List everything else in gap_skills.

    2. required_match_ratio = matched required skills / total required skills (float 0–1).
    Count only skills from required_skills — do not mix in nice_to_have_skills.

    3. nice_to_have_match_ratio = matched nice-to-have skills / total nice-to-have skills.
    If nice_to_have_skills is empty, return 0.0.

    4. reasoning: 2 sentences max. Sentence 1 — what the candidate matches well and why.
    Sentence 2 — the most critical gap and its impact on the role. Be specific, not generic.

    Return ONLY this JSON — no markdown, no explanation:
    {{
    "matched_skills": ["skill1", "skill2"],
    "gap_skills": ["missing1", "missing2"],
    "required_match_ratio": 0.75,
    "nice_to_have_match_ratio": 0.5,
    "reasoning": "Candidate demonstrates X and Y which are core to this role. The primary gap is Z, which is required for [specific responsibility]."
    }}
    """

    try:
        client = get_openai_client()
        schema = build_json_schema(ScoreOutput)
        payload = create_structured_request(
            messages=[{"role": "user", "content": prompt}],
            schema=schema,
            temperature=0
        )
        response = await client.chat.completions.create(**payload)
        llm_result = parse_structured_response(response)

        matched_required = llm_result.get("matched_skills", [])
        gap_skills = llm_result.get("gap_skills", [])
        required_ratio = llm_result.get("required_match_ratio", 0)
        nice_ratio = llm_result.get("nice_to_have_match_ratio", 0)

    except Exception as e:
        log.error("semantic_scoring_failed", error=str(e))
        # Fallback: simple case-insensitive substring matching
        # Checks both profile skills and raw text
        candidate_skills = [s.lower() for s in candidate_profile.get("skills", [])]
        raw_lower = raw_resume_text.lower()

        matched_required = []
        gap_skills = []
        for req in required:
            req_l = req.lower()
            if any(req_l in s for s in candidate_skills) or req_l in raw_lower:
                matched_required.append(req)
            else:
                gap_skills.append(req)

        required_ratio = len(matched_required) / max(len(required), 1)
        nice_ratio = 0

    candidate_exp = candidate_profile.get("years_experience", 0)
    all_matched = list(set(matched_required))  # Deduplicated required matches

    # Weighted: required skills count 80%, nice-to-have 20%
    skill_score = required_ratio * 0.8 + nice_ratio * 0.2

    # ── JD completeness ────────────────────────────────────────────────
    jd_fields = [
        requirements.get("required_skills"),
        requirements.get("nice_to_have_skills"),
        requirements.get("seniority_level"),
        requirements.get("domain"),
        requirements.get("responsibilities"),
    ]
    jd_completeness = sum(1 for f in jd_fields if f and len(f) > 0) / 5

    # ── Experience distance ────────────────────────────────────────────
    seniority_map = {"junior": 1, "mid": 3, "senior": 5, "lead": 8}
    target_years = seniority_map.get(requirements.get("seniority_level", "mid"), 3)
    exp_distance = 1 - min(abs(candidate_exp - target_years) / max(target_years, 1), 1)

    # ── Seniority fit (candidate seniority vs JD seniority) ────────────
    candidate_seniority = candidate_profile.get("seniority", "mid")
    jd_seniority = requirements.get("seniority_level", "mid")
    seniority_order = {"junior": 0, "mid": 1, "senior": 2, "lead": 3}
    candidate_level = seniority_order.get(candidate_seniority, 1)
    jd_level = seniority_order.get(jd_seniority, 1)
    seniority_diff = abs(candidate_level - jd_level)
    # Perfect match = 1.0, one level off = 0.7, two = 0.3, three = 0.0
    seniority_fit = max(0.0, 1.0 - seniority_diff * 0.3)

    # ── Domain match ───────────────────────────────────────────────────
    domain_match = 1.0 if (
        candidate_profile.get("domain", "").lower() ==
        requirements.get("domain", "").lower()
    ) else 0.5 if requirements.get("domain") else 0.3

    # ── Overall score ──────────────────────────────────────────────────
    overall_score = min(100, max(0, int(
        skill_score * 40 +
        jd_completeness * 5 +
        exp_distance * 20 +
        seniority_fit * 20 +
        domain_match * 15
    )))

    # ── Confidence heuristic ───────────────────────────────────────────
    # Derived from measurable signals, not asserted arbitrarily
    # High: strong skill match + complete JD
    # Medium: moderate skill match + reasonably complete JD
    # Low: everything else
    if required_ratio >= 0.7 and jd_completeness >= 0.6:
        confidence = "high"
    elif required_ratio >= 0.4 and jd_completeness >= 0.4:
        confidence = "medium"
    else:
        confidence = "low"

    result = {
        "overall_score": overall_score,
        "dimension_scores": {
            "skills": int(skill_score * 100),
            "experience": int(exp_distance * 100),
            "seniority_fit": int(seniority_fit * 100),
        },
        "matched_skills": all_matched,
        "gap_skills": gap_skills,
        "confidence": confidence,
    }

    log.info(
        "score_computed",
        overall_score=overall_score,
        confidence=confidence,
        required_match_ratio=round(required_ratio, 2),
        nice_to_have_match_ratio=round(nice_ratio, 2),
        seniority_fit=round(seniority_fit, 2),
        jd_completeness=jd_completeness,
        gap_count=len(gap_skills),
    )

    if tool_context:
        tool_context.state["score"] = result
        tool_context.state["gap_skills"] = gap_skills

    _cache.set(cache_key, result, expire=86400)
    _record_latency(tool_context, start_time)
    return result


def _record_latency(tool_context: ToolContext, start_time: float):
    """Record tool execution latency in tool_context state."""
    if tool_context is None:
        return
    latency_ms = int((time.time() - start_time) * 1000)
    tool_context.state["temp:score_candidate_against_requirements_latency_ms"] = latency_ms
