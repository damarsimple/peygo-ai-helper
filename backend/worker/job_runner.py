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

        Uses ADK SequentialAgent pattern:
        1. tool_agent runs tools and populates session state.
        2. formatter_agent reads session state and produces schema-compliant JSON.
        3. Low-confidence guard: if score is low, augment with research.

        Path 1: Parse final_output from session state (set by formatter output_key).
        Path 2: Fallback reconstruction if parser fails.
        """
        candidate = await self._get_candidate(candidate_id)

        # Ensure IDs are strings, not UUID objects
        job_id_str = str(job_id)
        candidate_id_str = str(candidate_id)

        log.info("job_runner_start", job_id=job_id_str, candidate_id=candidate_id_str)

        # ── Consistency Cache ───────────────────────────────────────────
        # If an identical job was already completed for this candidate,
        # reuse the result to ensure 100% consistency and save tokens.
        async with self.db_pool.acquire() as conn:
            existing = await conn.fetchrow(
                """SELECT result, agent_trace FROM match_jobs
                   WHERE candidate_id = $1 AND jd_input = $2
                     AND status = 'completed'
                     AND id != $3
                   ORDER BY updated_at DESC LIMIT 1""",
                candidate_id_str, jd_input, job_id_str
            )
            if existing:
                log.info("job_runner_cache_hit", job_id=job_id_str, reused_from=candidate_id_str)
                return AgentOutput(
                    job_id=job_id_str,
                    overall_score=existing["result"]["overall_score"],
                    confidence=existing["result"]["confidence"],
                    reasoning=existing["result"]["reasoning"] + " (Reused from previous identical run)",
                    matched_skills=existing["result"]["matched_skills"],
                    gap_skills=existing["result"]["gap_skills"],
                    dimension_scores=DimensionScores(**existing["result"]["dimension_scores"]),
                    learning_plan=[LearningPlanItem(**lp) for lp in existing["result"]["learning_plan"]],
                    agent_trace=AgentTrace(**existing["agent_trace"])
                )

        # Typed per-job callback context (replaces bare dict).
        cb._cb_state = cb.JobCallbackContext(
            job_data={
                "candidate_profile": candidate.model_dump(),
                "raw_resume_text": candidate.raw_text if hasattr(candidate, 'raw_text') else "",
                "job_id": job_id_str,
            }
        )

        # Wire the before_agent_callback to see this job's data.
        agent_before_cb = self.runner.agent.before_agent_callback
        self.runner.agent.before_agent_callback = cb.before_agent_callback

        try:
            trace_collector = AgentTraceCollector(callback_state=cb._cb_state.latency)

            # Run the SequentialAgent pipeline
            try:
                async for event in self.runner.run_async(
                    user_id=candidate_id_str,
                    session_id=job_id_str,
                    new_message=types.Content(
                        role="user",
                        parts=[types.Part(text=f"Analyze this job for the candidate:\n\n{jd_input}")]
                    ),
                ):
                    trace_collector.collect(event)
            except Exception as e:
                log.warning("agent_pipeline_exception", job_id=job_id_str, error=str(e)[:200])
                # Formatter schema validation failed (e.g., high temp).
                # tool_agent already ran and saved intermediate state; fall back to Path 2.

            # ── Path 1: Read structured output from session state ──────
            session = await self.session_service.get_session(
                app_name="pelgo",
                user_id=candidate_id_str,
                session_id=job_id_str,
            )
            final_text = session.state.get("final_output") if session else None

            parsed = None
            if final_text is not None:
                parsed = self._parse_text_as_json(final_text)

            if parsed is not None:
                output = AgentOutput(**parsed.model_dump())
                output.job_id = job_id_str

                # ── Low-confidence guard ──────────────────────────────
                if output.confidence == "low":
                    output = self._apply_low_confidence_guard(
                        output, session.state if session else {},
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
                    "job_runner_complete_path1",
                    job_id=job_id_str,
                    score=output.overall_score,
                    confidence=output.confidence,
                    llm_calls=trace_collector.total_llm_calls,
                    tokens_in=trace_collector.total_input_tokens,
                    tokens_out=trace_collector.total_output_tokens,
                )
                return output

            # ── Path 2: Fallback reconstruction from intermediate state ──
            state = session.state if session else {}
            score = state.get("score", {})
            gap_skills = state.get("gap_skills", [])
            score_dict = score if isinstance(score, dict) else {}

            output_data = {
                "job_id": job_id_str,
                "overall_score": score_dict.get("overall_score", 0),
                "confidence": score_dict.get("confidence", "low"),
                "dimension_scores": score_dict.get("dimension_scores", {
                    "skills": 0, "experience": 0, "seniority_fit": 0,
                }),
                "matched_skills": score_dict.get("matched_skills", []),
                "gap_skills": gap_skills,
                "reasoning": (
                    f"Candidate scored {score_dict.get('overall_score', 0)} "
                    f"with {score_dict.get('confidence', 'low')} confidence. "
                    f"Agent used fallback reconstruction — results may be conservative."
                ),
                "learning_plan": await self._build_fallback_learning_plan(
                    state.get("prioritized_gaps", [{"skill": s} for s in gap_skills[:5]]),
                    gap_skills,
                ),
            }

            validated_output = AgentOutput(**output_data)

            # Low-confidence guard on fallback too
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
                "job_runner_complete_path2",
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
            profile_data = row["structured_profile"]
            if isinstance(profile_data, str):
                profile_data = json.loads(profile_data)
            candidate = CandidateProfile.model_validate(profile_data)
            # Attach raw_text to candidate object for access in job_runner
            candidate.raw_text = row["raw_text"] or ""
            return candidate
