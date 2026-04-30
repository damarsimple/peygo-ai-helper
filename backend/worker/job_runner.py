import json
import structlog
from google.genai import types
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService
from backend.schemas import (
    AgentOutput, AgentTrace, ToolCallTrace, AgentOutputForLLM,
    CandidateProfile, DimensionScores, LearningPlanItem,
)
from backend.worker.trace_collector import AgentTraceCollector
from backend.worker import callbacks as cb

log = structlog.get_logger()


class JobRunner:
    """Orchestrates the full agent pipeline for one match job.

    Enforces the low-confidence guard: if the agent returns confidence='low',
    the orchestrator ensures additional research is performed or a conservative
    score is applied before persisting the result.
    """

    def __init__(self, runner: Runner, session_service: DatabaseSessionService, db_pool):
        self.runner = runner
        self.session_service = session_service
        self.db_pool = db_pool

    async def run_job(self, job_id: str, candidate_id: str, jd_input: str) -> AgentOutput:
        """Execute the full agent pipeline for one match job.

        Uses a single ADK Agent with tools (replaces SequentialAgent):
        1. Agent calls tools to analyze candidate vs job description.
        2. Agent produces structured JSON output in the final response.
        3. Low-confidence guard: if score is low, apply conservative adjustment.

        Path 1: Parse JSON from the agent's final response text.
        Path 2: Fallback reconstruction from intermediate tool state.
        """
        candidate = await self._get_candidate(candidate_id)

        # Ensure IDs are strings, not UUID objects
        job_id_str = str(job_id)
        candidate_id_str = str(candidate_id)

        log.info("job_runner_start", job_id=job_id_str, candidate_id=candidate_id_str)

        # Typed per-job callback context (replaces bare dict).
        cb._cb_state = cb.JobCallbackContext(
            job_data={
                "candidate_profile": candidate.model_dump(),
                "raw_resume_text": candidate.raw_text if hasattr(candidate, 'raw_text') else "",
                "job_description": jd_input,
                "job_id": job_id_str,
            }
        )

        # Wire the before_agent_callback to see this job's data.
        agent_before_cb = self.runner.agent.before_agent_callback
        self.runner.agent.before_agent_callback = cb.before_agent_callback

        try:
            trace_collector = AgentTraceCollector(callback_state=cb._cb_state.latency)

            # Collect final text from the event stream.
            # With a single Agent (no SequentialAgent), we parse the final
            # response directly from the last text event.
            final_texts = []
            event_debug = []

            try:
                # Pass ALL data directly in the message so the agent can see it.
                # The {key} templating in instruction may not work with Agent class.
                import json as json_mod
                candidate_dict = candidate.model_dump()
                message = f"""Here is the complete data for analysis:

CANDIDATE PROFILE (JSON):
{json_mod.dumps(candidate_dict, indent=2)}

RAW RESUME TEXT:
{candidate.raw_text if candidate.raw_text else "Not provided"}

JOB DESCRIPTION:
{jd_input}

Please analyze the candidate against this job description using the tools available.
"""
                
                async for event in self.runner.run_async(
                    user_id=candidate_id_str,
                    session_id=job_id_str,
                    new_message=types.Content(
                        role="user",
                        parts=[types.Part(text=message)]
                    ),
                ):
                    trace_collector.collect(event)
                    # Debug: log every event
                    event_debug.append({
                        "author": getattr(event, "author", None),
                        "is_final": event.is_final_response(),
                        "has_content": event.content is not None,
                        "parts_count": len(event.content.parts) if event.content and event.content.parts else 0,
                    })
                    # Collect text from ALL parts (not just final), for debugging
                    if event.content is not None and event.content.parts:
                        for part in event.content.parts:
                            if hasattr(part, 'text') and part.text:
                                final_texts.append(part.text)
                log.info(
                    "job_runner_events", job_id=job_id_str,
                    total_events=len(event_debug), collected_texts=len(final_texts),
                    events=event_debug,
                    last_text_head=final_texts[-1][:200] if final_texts else "none",
                )
            except Exception as e:
                log.warning("agent_pipeline_exception", job_id=job_id_str, error=str(e)[:200])

            # ── Parse structured output from agent's final text ──────
            parsed = None
            for i, text in enumerate(final_texts):
                try:
                    candidate = self._parse_text_as_json(text)
                    if candidate is not None:
                        parsed = candidate
                        log.info("parse_ok", job_id=job_id_str, text_index=i, text_len=len(text))
                        break
                except Exception as e:
                    log.warning("parse_exception", job_id=job_id_str, text_index=i, error=str(e))

            if parsed is None:
                log.warning(
                    "parse_all_failed", job_id=job_id_str,
                    text_count=len(final_texts),
                    last_text_head=final_texts[-1][:500] if final_texts else "none",
                )

            if parsed is not None:
                output = AgentOutput(**parsed.model_dump())
                output.job_id = job_id_str

                # ── Low-confidence guard ──────────────────────────────
                if output.confidence == "low":
                    output = self._apply_low_confidence_guard(
                        output, {},  # session_state no longer needed
                        trace_collector,
                    )

                output.agent_trace = AgentTrace(
                    tool_calls=[ToolCallTrace(**tc) for tc in trace_collector.tool_calls],
                    total_llm_calls=trace_collector.total_llm_calls,
                    fallbacks_triggered=trace_collector.fallbacks_triggered,
                    total_input_tokens=trace_collector.total_input_tokens,
                    total_output_tokens=trace_collector.total_output_tokens,
                    elapsed_ms=trace_collector.elapsed_ms,
                )
                log.info(
                    "job_runner_complete",
                    job_id=job_id_str,
                    score=output.overall_score,
                    confidence=output.confidence,
                    llm_calls=trace_collector.total_llm_calls,
                    tokens_in=trace_collector.total_input_tokens,
                    tokens_out=trace_collector.total_output_tokens,
                )
                return output

            # ── Fallback: reconstruct from intermediate tool state ──
            # Read tool responses from temp: state keys
            session = await self.session_service.get_session(
                app_name="pelgo",
                user_id=candidate_id_str,
                session_id=job_id_str,
            )
            raw_state = session.state if session else None
            # session.state can be a string (JSONB from asyncpg) or a dict
            if isinstance(raw_state, str):
                try:
                    state = json.loads(raw_state)
                except (json.JSONDecodeError, TypeError):
                    state = {}
            elif isinstance(raw_state, dict):
                state = raw_state
            else:
                state = {}

            # Extract intermediate results from temp: state keys
            score = state.get("temp:score_candidate_against_requirements_response", {})
            if not score:
                score = state.get("score", {})
            gap_data = state.get("temp:prioritise_skill_gaps_response", {})
            gap_skills = gap_data.get("prioritized_gaps", []) if isinstance(gap_data, dict) else state.get("gap_skills", [])

            score_dict = score if isinstance(score, dict) else {}

            output_data = {
                "job_id": job_id_str,
                "overall_score": score_dict.get("overall_score", 0),
                "confidence": score_dict.get("confidence", "low"),
                "dimension_scores": score_dict.get("dimension_scores", {
                    "skills": 0, "experience": 0, "seniority_fit": 0,
                }),
                "matched_skills": score_dict.get("matched_skills", []),
                "gap_skills": gap_skills if isinstance(gap_skills, list) else [],
                "reasoning": (
                    f"Candidate scored {score_dict.get('overall_score', 0)} "
                    f"with {score_dict.get('confidence', 'low')} confidence. "
                    f"Agent used fallback reconstruction — results may be conservative."
                ),
                "learning_plan": await self._build_fallback_learning_plan(
                    gap_skills if isinstance(gap_skills, list) else [{"skill": "General"}],
                    [g.get("skill", g) if isinstance(g, dict) else g for g in (gap_skills if isinstance(gap_skills, list) else [])[:5]],
                ),
            }

            validated_output = AgentOutput(**output_data)

            # Low-confidence guard on fallback too
            if validated_output.confidence == "low":
                validated_output.overall_score = max(0, validated_output.overall_score - 15)
                validated_output.reasoning += (
                    " [LOW CONFIDENCE: Score reduced by 15 points. "
                    "Insufficient signal to assess candidate confidently — "
                    "recommend manual review.]"
                )

            validated_output.agent_trace = AgentTrace(
                tool_calls=[ToolCallTrace(**tc) for tc in trace_collector.tool_calls],
                total_llm_calls=trace_collector.total_llm_calls,
                fallbacks_triggered=trace_collector.fallbacks_triggered,
                total_input_tokens=trace_collector.total_input_tokens,
                total_output_tokens=trace_collector.total_output_tokens,
                elapsed_ms=trace_collector.elapsed_ms,
            )
            log.info(
                "job_runner_complete_fallback",
                job_id=job_id_str,
                score=validated_output.overall_score,
                confidence=validated_output.confidence,
                fallback=True,
            )
            return validated_output

        finally:
            self.runner.agent.before_agent_callback = agent_before_cb

    @staticmethod
    def _apply_low_confidence_guard(
        output: AgentOutput, session_state: dict,
        trace_collector: AgentTraceCollector,
    ) -> AgentOutput:
        """Orchestrator-enforced low-confidence guard.

        When confidence is low, reduce the score to penalise uncertainty
        and flag the result. This prevents silently returning misleading scores.

        Decision: Reduce score by 15 points and append a note to reasoning.
        Rationale: Low confidence means insufficient signal. Penalising the
        score encourages reviewers to treat the match cautiously.
        """
        log.warning(
            "low_confidence_guard_triggered",
            job_id=output.job_id,
            original_score=output.overall_score,
        )
        output.overall_score = max(0, output.overall_score - 15)
        output.reasoning += (
            " [LOW CONFIDENCE: Score reduced by 15 points. "
            "Insufficient signal to assess candidate confidently — "
            "recommend manual review.]"
        )
        # Count the guard as a fallback
        trace_collector.fallbacks_triggered += 1
        return output

    @staticmethod
    def _parse_text_as_json(text: str) -> AgentOutputForLLM | None:
        """Parse raw text response as structured JSON."""
        if isinstance(text, dict):
            try:
                return AgentOutputForLLM.model_validate(text)
            except Exception:
                return None

        cleaned = text.strip()
        for tag in ("\x1e", "\x1f"):
            cleaned = cleaned.replace(tag, "")

        brace_start = cleaned.find('{')
        brace_end = cleaned.rfind('}')
        if brace_start == -1 or brace_end <= brace_start:
            return None

        try:
            data = json.loads(cleaned[brace_start:brace_end + 1])
            return AgentOutputForLLM.model_validate(data)
        except Exception:
            return None

    async def _build_fallback_learning_plan(
        self, source: list[dict], gap_skills: list[str],
    ) -> list[dict]:
        learning_plan = []
        for i, gap in enumerate(source[:5]):
            skill = gap.get("skill", "")
            if not skill:
                continue
            learning_plan.append({
                "skill": skill,
                "priority_rank": gap.get("priority_rank", i + 1),
                "estimated_match_gain_pct": gap.get("estimated_match_gain_pct", 10),
                "resources": [{
                    "title": f"Learn {skill}",
                    "url": f"https://www.google.com/search?q=learn+{skill.replace(' ', '+')}",
                    "estimated_hours": 10,  # Will be updated by LLM below
                    "type": "doc",
                    "relevance_score": 0.8,
                }],
                "rationale": gap.get("rationale", f"{skill} is important for this role."),
            })

        if not learning_plan:
            default_skill = gap_skills[0] if gap_skills else "General Skill"
            learning_plan = [{
                "skill": default_skill,
                "priority_rank": 1,
                "estimated_match_gain_pct": 10,
                "resources": [{
                    "title": f"Learn {default_skill}",
                    "url": f"https://www.google.com/search?q=learn+{default_skill.replace(' ', '+')}",
                    "estimated_hours": 10,
                    "type": "doc",
                    "relevance_score": 0.8,
                }],
                "rationale": f"{default_skill} is important for career growth.",
            }]

        # Use LLM to estimate hours for fallback resources
        from backend.tools.researcher import _llm_estimate_hours
        for item in learning_plan:
            resources = item.get("resources", [])
            if resources:
                item["resources"] = await _llm_estimate_hours(
                    resources, item["skill"], "mid"
                )

        return learning_plan

    async def _get_candidate(self, candidate_id: str) -> CandidateProfile:
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT structured_profile, raw_text FROM candidates WHERE id = $1", candidate_id
            )
            if not row:
                raise ValueError(f"Candidate {candidate_id} not found")
            profile_data = self._parse_jsonb(row["structured_profile"])
            candidate = CandidateProfile.model_validate(profile_data)
            candidate.raw_text = row["raw_text"] or ""
            return candidate

    @staticmethod
    def _parse_jsonb(value):
        """Parse JSONB value from asyncpg — may be str or already dict."""
        if isinstance(value, str):
            return json.loads(value)
        return value
