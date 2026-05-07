"""ADK callback handlers for tool and agent lifecycle.

State keys use the 'temp:' prefix per ADK docs:
- 'temp:' keys are scoped to the current invocation and discarded afterward,
  preventing stale data from persisting across jobs in the same session.

Tool callbacks receive (tool, args, tool_context).
Agent callbacks receive (callback_context).

Per-Invocation Latency Tracking
-------------------------------
The `JobCallbackContext` carries job data and per-tool latency measurements
between the job runner, agent/tool callbacks, and the trace collector.
It is a typed, per-invocation object: the orchestrator (JobRunner) creates
a fresh instance for every job and assigns it, ensuring data from one
invocation never leaks into the next.
"""
import time
import json
import hashlib
from dataclasses import dataclass, field
from typing import Any

import structlog

from google.adk.agents.callback_context import CallbackContext
from google.adk.tools.tool_context import ToolContext
from google.adk.tools.base_tool import BaseTool
from backend.schemas import CandidateProfile

log = structlog.get_logger()


def _get_call_id(tool_name: str, args: dict[str, Any]) -> str:
    """Generate a unique ID for a tool call based on tool name and args."""
    args_str = f"{tool_name}:{json.dumps(args, sort_keys=True, default=str)}"
    return hashlib.md5(args_str.encode()).hexdigest()[:8]


# ── Typed callback context (replaces bare dict) ─────────────────────────
# Created fresh by JobRunner.run_job() for each invocation.
@dataclass
class JobData:
    """Typed job context passed from JobRunner into the ADK callbacks.

    Mirrors the fields populated by JobRunner.run_job() so the callback
    layer can access them without blind dict lookups.
    """
    candidate_profile: CandidateProfile | None = None
    raw_resume_text: str = ""
    job_description: str = ""
    job_id: str = ""


@dataclass
class ToolLatencyEntry:
    """Single tool-call timing record.

    Stores the start timestamp, optional pre-recorded latency (when the
    tool measures its own time), and the tool arguments for debugging.
    """
    start_time: float = 0.0
    latency_ms: int = 0
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class JobCallbackContext:
    """Typed per-job callback state managed by the orchestrator.

    The orchestrator creates a fresh instance before each agent run and
    assigns it to `_cb_state`.  Callbacks and the trace collector read
    from the same instance during the run.
    """
    job_data: JobData = field(default_factory=JobData)
    latency: dict[str, ToolLatencyEntry] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        """Dict-like access for backward compatibility with trace collector."""
        if key == "__job_data__":
            return self.job_data
        entry = self.latency.get(key)
        return entry.latency_ms if entry is not None else default

    def __setitem__(self, key: str, value: Any) -> None:
        if key == "__job_data__":
            self.job_data = value
        elif isinstance(value, ToolLatencyEntry):
            self.latency[key] = value
        elif isinstance(value, (float, int)):
            # Accept raw timestamps / latency values for backward compat
            entry = self.latency.get(key)
            if entry is None:
                self.latency[key] = ToolLatencyEntry(start_time=float(value))
            else:
                entry.latency_ms = int(value)
        else:
            # Store as args dict
            entry = self.latency.get(key)
            if entry is None:
                self.latency[key] = ToolLatencyEntry(args=value if isinstance(value, dict) else {})
            else:
                entry.args = value if isinstance(value, dict) else {}

    def __getitem__(self, key: str) -> Any:
        if key == "__job_data__":
            return self.job_data
        return self.latency[key].latency_ms

    def __contains__(self, key: str) -> bool:
        if key == "__job_data__":
            return bool(self.job_data.job_id)
        return key in self.latency

    def keys(self) -> list[str]:
        return list(self.latency.keys())


_cb_state: JobCallbackContext | None = None


def _shared_state() -> dict[str, ToolLatencyEntry]:
    """Return the shared callback latency dict, or a fallback empty dict.

    Returns the latency sub-dict for backward compatibility with tests
    and trace collector code that iterates keys.
    """
    global _cb_state
    if _cb_state is None:
        _cb_state = JobCallbackContext()
    return _cb_state.latency


# ── Agent callbacks ──────────────────────────────────────────────────────


def before_agent_callback(callback_context: CallbackContext) -> None:
    """Set up session state before the agent begins execution.

    Populates session.state with job data so tools can reference them.
    The agent receives data via the message (not session.state) for better visibility.

    Reads from the typed JobCallbackContext that was populated by
    JobRunner.run_job() before this run.
    """
    global _cb_state
    ctx = _cb_state or JobCallbackContext()
    jd_ctx = ctx.job_data

    if jd_ctx.candidate_profile is not None:
        callback_context.state["candidate_profile"] = jd_ctx.candidate_profile.model_dump()
    callback_context.state["raw_resume_text"] = jd_ctx.raw_resume_text
    callback_context.state["job_description"] = jd_ctx.job_description
    if jd_ctx.job_id:
        callback_context.state["job_id"] = jd_ctx.job_id

    log.info(
        "agent_started",
        job_id=jd_ctx.job_id,
        candidate_skills=jd_ctx.candidate_profile.skills if jd_ctx.candidate_profile else [],
        has_raw_text=bool(jd_ctx.raw_resume_text),
    )
    return None  # Proceed with normal agent execution


# ── Tool callbacks ───────────────────────────────────────────────────────


def before_tool(
    tool: BaseTool,
    args: dict[str, Any],
    tool_context: ToolContext,
) -> None:
    """Record tool start time in shared state for latency tracking.

    Uses a unique call ID (based on tool name + args) to differentiate
    multiple calls to the same tool (e.g., multiple research_skill_resources).
    """
    shared = _shared_state()
    tool_name = tool.name if hasattr(tool, "name") else str(tool)
    call_id = _get_call_id(tool_name, args)
    key = f"{tool_name}:{call_id}"
    if key not in shared:
        shared[key] = ToolLatencyEntry(start_time=time.time(), args=args)

    log.debug("tool_call_started", tool=tool_name, call_id=call_id, job_id=(_cb_state.job_data.job_id if _cb_state else None))


def after_tool(
    tool: BaseTool,
    args: dict[str, Any],
    tool_context: ToolContext,
    tool_response: dict,
) -> None:
    """Compute actual tool latency and store in shared state.
    
    Uses unique call ID (based on tool name + args) to differentiate
    multiple calls to the same tool (e.g., multiple research_skill_resources).
    Also writes the tool response into tool_context.state with a 'temp:' prefix
    so it is available in the session after the run completes (useful for
    debugging and fallback paths when output_schema is not honoured).
    """
    tool_name = tool.name if hasattr(tool, "name") else str(tool)
    call_id = _get_call_id(tool_name, args)
    key = f"{tool_name}:{call_id}"

    # Look up the tracking entry created by before_tool
    entry = _shared_state().get(key)

    # Check if tool already recorded its own latency (per-tool measurement)
    latency_ms = 0
    if tool_context is not None:
        pre_recorded = tool_context.state.get(f"temp:{tool_name}_latency_ms")
        if pre_recorded is not None:
            latency_ms = pre_recorded
            if entry is not None:
                entry.latency_ms = latency_ms
            log.debug("using_prerecorded_latency", tool=tool_name, latency_ms=latency_ms)

    # Fallback: compute from start time if no pre-recorded latency
    if latency_ms == 0 and entry is not None and entry.start_time > 0:
        latency_ms = int((time.time() - entry.start_time) * 1000)
        entry.latency_ms = latency_ms
        if tool_context is not None:
            tool_context.state[f"temp:{key}_latency_ms"] = latency_ms

    # Log structured tool completion
    log.info(
        "tool_call_completed",
        tool=tool_name,
        latency_ms=latency_ms,
        job_id=(_cb_state.job_data.job_id if _cb_state else None),
    )

    # Persist tool response in session state for fallback path
    if tool_context is not None:
        tool_context.state[f"temp:{tool_name}_response"] = tool_response
    return None  # Pass original response through unchanged


def on_tool_error(
    tool: BaseTool,
    args: dict[str, Any],
    tool_context: ToolContext,
    error: Exception,
) -> dict:
    """Return fallback dict — prevents crashing the run.

    Logs the error with full context for observability.
    """
    tool_name = tool.name if hasattr(tool, "name") else str(tool)
    log.error(
        "tool_call_failed",
        tool=tool_name,
        error=str(error)[:200],
        job_id=(_cb_state.job_data.job_id if _cb_state else None),
    )
    return {"error": f"Tool {tool_name} failed: {error}", "fallback": True}
