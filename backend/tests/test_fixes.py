"""Unit tests validating ADK compliance, callbacks, trace collection, and low-confidence guard.

Tests cover:
1. callbacks.before_agent_callback — populates session state from shared dict
2. callbacks.before_tool / after_tool — record actual tool latency
3. callbacks.on_tool_error — returns fallback dict
4. trace_collector — reads latency from shared callback state (not wall-clock)
5. job_runner — no state_delta on run_async, proper callback wiring
6. job_runner — low-confidence guard reduces score and flags result
"""
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── Fix 1: before_agent_callback ──────────────────────────────────────────


class TestBeforeAgentCallback:
    def setup_method(self):
        from backend.worker import callbacks
        callbacks._cb_state = callbacks.JobCallbackContext()

    def teardown_method(self):
        from backend.worker import callbacks
        callbacks._cb_state = callbacks.JobCallbackContext()

    def test_populates_candidate_profile(self):
        from backend.worker import callbacks

        callbacks._cb_state = callbacks.JobCallbackContext(
            job_data={
                "candidate_profile": {"skills": ["Python"], "years_experience": 3},
                "job_id": "abc-123",
            }
        )

        mock_context = MagicMock()
        mock_context.state = {}

        callbacks.before_agent_callback(mock_context)

        assert mock_context.state["candidate_profile"] == {
            "skills": ["Python"],
            "years_experience": 3,
        }
        assert mock_context.state["job_id"] == "abc-123"

    def test_returns_none_proceeds(self):
        from backend.worker import callbacks
        callbacks._cb_state = callbacks.JobCallbackContext()
        mock_context = MagicMock()
        mock_context.state = {}
        result = callbacks.before_agent_callback(mock_context)
        assert result is None

    def test_missing_job_data_is_safe(self):
        from backend.worker import callbacks
        callbacks._cb_state = callbacks.JobCallbackContext()
        mock_context = MagicMock()
        mock_context.state = {}
        callbacks.before_agent_callback(mock_context)  # No error, no state set


# ── Fix 2: before_tool / after_tool latency ──────────────────────────────


class TestToolCallbacks:
    def setup_method(self):
        from backend.worker import callbacks
        callbacks._cb_state = callbacks.JobCallbackContext()

    def teardown_method(self):
        from backend.worker import callbacks
        callbacks._cb_state = callbacks.JobCallbackContext()

    def test_before_tool_records_start(self):
        from backend.worker import callbacks
        mock_tool = MagicMock()
        mock_tool.name = "extract_jd_requirements"
        mock_args = {"job_url_or_text": "https://example.com"}
        mock_context = MagicMock()
        callbacks.before_tool(mock_tool, mock_args, mock_context)
        assert any(k for k in callbacks._shared_state() if k.startswith(f"{mock_tool.name}:") and k.endswith("_start"))

    def test_after_tool_records_latency(self):
        from backend.worker import callbacks
        mock_tool = MagicMock()
        mock_tool.name = "score_candidate"
        mock_args = {}
        # Use a real dict for state so .get() returns None (not MagicMock)
        mock_context = MagicMock()
        mock_context.state = {}
        mock_response = {"overall_score": 75}
        callbacks.before_tool(mock_tool, mock_args, mock_context)
        time.sleep(0.01)
        callbacks.after_tool(mock_tool, mock_args, mock_context, mock_response)
        latency_found = any(k for k in callbacks._shared_state() if k.startswith(f"{mock_tool.name}:") and k.endswith("_latency_ms"))
        assert latency_found
        latency_key = next(k for k in callbacks._shared_state() if k.startswith(f"{mock_tool.name}:") and k.endswith("_latency_ms"))
        assert callbacks._shared_state()[latency_key] >= 0

    def test_after_tool_no_start_is_safe(self):
        from backend.worker import callbacks
        mock_tool = MagicMock()
        mock_tool.name = "unknown_tool"
        mock_args = {}
        mock_context = MagicMock()
        mock_response = {}
        callbacks.after_tool(mock_tool, mock_args, mock_context, mock_response)  # No crash

    def test_on_tool_error_returns_fallback(self):
        from backend.worker import callbacks
        mock_tool = MagicMock()
        mock_tool.name = "failing_tool"
        mock_args = {}
        mock_context = MagicMock()
        error = ValueError("network timeout")
        result = callbacks.on_tool_error(mock_tool, mock_args, mock_context, error)
        assert result["error"] is not None
        assert result["fallback"] is True


# ── Fix 3: trace_collector reads real latency ────────────────────────────


class TestTraceCollectorLatency:
    def test_reads_latency_from_callback_state(self):
        from backend.worker.trace_collector import AgentTraceCollector
        # The trace collector reads latency from callback_state on function responses
        mock_fc = MagicMock()
        mock_fc.name = "score_candidate"
        mock_fc.args = {}
        # Build the callback_state key the same way the collector does
        import json, hashlib
        args_str = f"score_candidate:{json.dumps({}, sort_keys=True, default=str)}"
        call_id = hashlib.md5(args_str.encode()).hexdigest()[:8]
        callback_state = {f"score_candidate:{call_id}_latency_ms": 1234}
        collector = AgentTraceCollector(callback_state=callback_state)
        # Event 1: function call (records start time)
        mock_event_call = MagicMock()
        mock_event_call.get_function_calls.return_value = [mock_fc]
        mock_event_call.get_function_responses.return_value = []
        mock_event_call.is_final_response.return_value = False
        collector.collect(mock_event_call)
        # Event 2: function response (creates the tool_call entry)
        mock_fr = MagicMock()
        mock_fr.name = "score_candidate"
        mock_fr.args = {}
        mock_fr.response = {"overall_score": 75}
        mock_event_resp = MagicMock()
        mock_event_resp.get_function_calls.return_value = []
        mock_event_resp.get_function_responses.return_value = [mock_fr]
        mock_event_resp.is_final_response.return_value = False
        collector.collect(mock_event_resp)
        assert collector.tool_calls[0]["latency_ms"] == 1234

    def test_fallback_latency_zero(self):
        from backend.worker.trace_collector import AgentTraceCollector
        collector = AgentTraceCollector(callback_state={})
        mock_fc = MagicMock()
        mock_fc.name = "score_candidate"
        mock_fc.args = {}
        # Event 1: function call
        mock_event_call = MagicMock()
        mock_event_call.get_function_calls.return_value = [mock_fc]
        mock_event_call.get_function_responses.return_value = []
        mock_event_call.is_final_response.return_value = False
        collector.collect(mock_event_call)
        # Event 2: function response
        mock_fr = MagicMock()
        mock_fr.name = "score_candidate"
        mock_fr.args = {}
        mock_fr.response = {"overall_score": 75}
        mock_event_resp = MagicMock()
        mock_event_resp.get_function_calls.return_value = []
        mock_event_resp.get_function_responses.return_value = [mock_fr]
        mock_event_resp.is_final_response.return_value = False
        collector.collect(mock_event_resp)
        # Latency should be computed from wall-clock (small but >= 0)
        assert collector.tool_calls[0]["latency_ms"] >= 0

    def test_no_callback_state_is_safe(self):
        from backend.worker.trace_collector import AgentTraceCollector
        collector = AgentTraceCollector(callback_state=None)
        mock_fc = MagicMock()
        mock_fc.name = "score_candidate"
        mock_fc.args = {}
        # Event 1: function call
        mock_event_call = MagicMock()
        mock_event_call.get_function_calls.return_value = [mock_fc]
        mock_event_call.get_function_responses.return_value = []
        mock_event_call.is_final_response.return_value = False
        collector.collect(mock_event_call)
        # Event 2: function response
        mock_fr = MagicMock()
        mock_fr.name = "score_candidate"
        mock_fr.args = {}
        mock_fr.response = {"overall_score": 75}
        mock_event_resp = MagicMock()
        mock_event_resp.get_function_calls.return_value = []
        mock_event_resp.get_function_responses.return_value = [mock_fr]
        mock_event_resp.is_final_response.return_value = False
        collector.collect(mock_event_resp)
        assert collector.tool_calls[0]["latency_ms"] >= 0

    def test_final_response_counts_llm_call(self):
        from backend.worker.trace_collector import AgentTraceCollector
        collector = AgentTraceCollector(callback_state={})
        mock_event = MagicMock()
        mock_event.get_function_calls.return_value = []
        mock_event.get_function_responses.return_value = []
        mock_event.is_final_response.return_value = True
        collector.collect(mock_event)
        assert collector.total_llm_calls == 1

    def test_fallback_response_error_counts(self):
        from backend.worker.trace_collector import AgentTraceCollector
        collector = AgentTraceCollector(callback_state={})
        mock_fr = MagicMock()
        mock_fr.response = {"error": "tool failed"}
        mock_event = MagicMock()
        mock_event.get_function_calls.return_value = []
        mock_event.get_function_responses.return_value = [mock_fr]
        mock_event.is_final_response.return_value = False
        collector.collect(mock_event)
        assert collector.fallbacks_triggered == 1

    def test_elapsed_ms_property(self):
        from backend.worker.trace_collector import AgentTraceCollector
        collector = AgentTraceCollector(callback_state={})
        ms = collector.elapsed_ms
        assert ms >= 0

    def test_summary_includes_token_tracking(self):
        """Verify the summary method returns token tracking fields."""
        from backend.worker.trace_collector import AgentTraceCollector
        collector = AgentTraceCollector(callback_state={})
        summary = collector.summary()
        assert "total_input_tokens" in summary
        assert "total_output_tokens" in summary
        assert "elapsed_ms" in summary
        assert summary["total_input_tokens"] == 0
        assert summary["total_output_tokens"] == 0
        assert summary["elapsed_ms"] >= 0


# ── Fix 4: JobRunner — no state_delta on run_async ───────────────────────


class TestJobRunnerNoStateDelta:
    def test_run_async_signature(self):
        from backend.worker.job_runner import JobRunner
        import inspect
        sig = inspect.signature(JobRunner.run_job)
        params = list(sig.parameters.keys())
        assert "state_delta" not in params


class TestJobRunnerCallbackWiring:
    @pytest.mark.asyncio
    async def test_wires_callback_and_restores(self):
        """Verify the before_agent_callback is swapped in and restored in finally."""
        from backend.worker import callbacks as cb_mod
        writes: list = []

        class AgentLike:
            _callback = None

            @property
            def before_agent_callback(self):
                return self._callback

            @before_agent_callback.setter
            def before_agent_callback(self, value):
                writes.append(value)
                self._callback = value

        mock_agent = AgentLike()
        mock_runner = MagicMock()
        mock_runner.agent = mock_agent
        mock_session_service = MagicMock()
        mock_db_pool = MagicMock()
        # Ensure no consistency cache hit
        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value=None)
        mock_db_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_db_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        from backend.worker.job_runner import JobRunner

        job_runner = JobRunner(mock_runner, mock_session_service, mock_db_pool)

        # Mock candidate
        mock_candidate = MagicMock()
        mock_candidate.model_dump.return_value = {
            "skills": ["Python"], "years_experience": 3, "seniority": "mid",
        }
        job_runner._get_candidate = AsyncMock(return_value=mock_candidate)

        # Mock session for fallback path
        mock_session = MagicMock()
        mock_session.state = {
            "score": {"overall_score": 60, "confidence": "medium"},
            "gap_skills": ["Docker"],
            "prioritized_gaps": [{"skill": "Docker", "priority_rank": 1}],
        }
        mock_session_service.get_session = AsyncMock(return_value=mock_session)

        async def empty_async_iter():
            if False:
                yield
        mock_runner.run_async = MagicMock(side_effect=lambda *a, **k: empty_async_iter())

        await job_runner.run_job("job-1", "user-1", "JD text")

        assert len(writes) >= 2
        assert cb_mod.before_agent_callback in writes
        assert writes[-1] is None  # Restored in finally


# ── Fix 5: Shared state module-level access ──────────────────────────────


class TestSharedState:
    def test_shared_state_returns_dict(self):
        from backend.worker.callbacks import _shared_state
        result = _shared_state()
        assert isinstance(result, dict)

    def test_shared_state_mutable_across_calls(self):
        from backend.worker.callbacks import _shared_state
        d1 = _shared_state()
        d1["key"] = "value"
        d2 = _shared_state()
        assert d2["key"] == "value"


# ── Fix 6: Low-confidence guard ──────────────────────────────────────────


class TestLowConfidenceGuard:
    def test_guard_reduces_score(self):
        """Verify the low-confidence guard reduces score by 15 and flags reasoning."""
        from backend.worker.job_runner import JobRunner
        from backend.worker.trace_collector import AgentTraceCollector
        from backend.schemas import AgentOutput, DimensionScores, AgentTrace

        output = AgentOutput(
            job_id="test-job",
            overall_score=60,
            confidence="low",
            dimension_scores=DimensionScores(skills=50, experience=60, seniority_fit=50),
            matched_skills=["Python"],
            gap_skills=["Docker", "Kubernetes"],
            reasoning="Test reasoning",
            learning_plan=[],
            agent_trace=AgentTrace(),
        )

        trace = AgentTraceCollector(callback_state={})
        guarded = JobRunner._apply_low_confidence_guard(output, {}, trace)

        assert guarded.overall_score == 45  # 60 - 15
        assert "LOW CONFIDENCE" in guarded.reasoning
        assert "Score reduced by 15" in guarded.reasoning
        assert trace.fallbacks_triggered == 1

    def test_guard_does_not_go_negative(self):
        """Score reduced below 0 should clamp to 0."""
        from backend.worker.job_runner import JobRunner
        from backend.worker.trace_collector import AgentTraceCollector
        from backend.schemas import AgentOutput, DimensionScores, AgentTrace

        output = AgentOutput(
            job_id="test-job",
            overall_score=10,
            confidence="low",
            dimension_scores=DimensionScores(skills=10, experience=10, seniority_fit=10),
            matched_skills=[],
            gap_skills=["A", "B"],
            reasoning="Low score test reasoning that meets the minimum length requirement",
            learning_plan=[],
            agent_trace=AgentTrace(),
        )

        trace = AgentTraceCollector(callback_state={})
        guarded = JobRunner._apply_low_confidence_guard(output, {}, trace)

        assert guarded.overall_score == 0  # max(0, 10 - 15) = 0


# ── Fix 7: DimensionScores validation ───────────────────────────────────


class TestDimensionScores:
    def test_valid_scores(self):
        from backend.schemas import DimensionScores
        ds = DimensionScores(skills=80, experience=65, seniority_fit=50)
        assert ds.skills == 80
        assert ds.experience == 65
        assert ds.seniority_fit == 50

    def test_scores_cannot_exceed_100(self):
        from backend.schemas import DimensionScores
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            DimensionScores(skills=101, experience=65, seniority_fit=50)

    def test_scores_cannot_be_negative(self):
        from backend.schemas import DimensionScores
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            DimensionScores(skills=-1, experience=65, seniority_fit=50)
