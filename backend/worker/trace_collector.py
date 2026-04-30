"""Collects agent trace information from the ADK event stream.

The trace collector receives the shared callback state dict (from
callbacks._shared_state()) so it can read actual per-tool latency
values recorded by the after_tool callback instead of computing
inaccurate wall-clock times.
"""
import time
import json
import hashlib
from typing import Any

import structlog

from google.adk.events.event import Event

log = structlog.get_logger()


class AgentTraceCollector:
    """Collects real tool-call traces from the ADK event stream.

    Populates tool_calls, total_llm_calls, fallbacks_triggered, and
    cumulative token usage (input_tokens + output_tokens) from the event
    stream metadata.
    """

    def __init__(self, callback_state: dict[str, Any] | None = None):
        """
        Args:
            callback_state: The shared dict populated by before_tool / after_tool
                                callbacks. If None, latency data will not be available
                                and will default to 0.
        """
        self._callback_state = callback_state or {}
        self.tool_calls: list[dict] = []
        self.events: list[Event] = []
        self.total_llm_calls: int = 0
        self.fallbacks_triggered: int = 0
        self.start_time: float = time.time()
        # Token tracking
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        # Track MULTIPLE start times per tool (for multiple calls to same tool)
        self._tool_starts: dict[str, list[float]] = {}

    def _get_call_id(self, tool_name: str, args: dict) -> str:
        """Generate call ID matching the one in callbacks.py."""
        args_str = f"{tool_name}:{json.dumps(args, sort_keys=True, default=str)}"
        return hashlib.md5(args_str.encode()).hexdigest()[:8]

    def collect(self, event: Event):
        """Process one event from the agent execution stream."""
        self.events.append(event)

        # Track token usage from event metadata if available
        usage = event.usage if hasattr(event, "usage") else None
        if usage is not None:
            if hasattr(usage, "prompt_tokens"):
                self.total_input_tokens += usage.prompt_tokens or 0
            if hasattr(usage, "completion_tokens"):
                self.total_output_tokens += usage.completion_tokens or 0
            if hasattr(usage, "total_tokens"):
                pass  # Derived from prompt + completion

        # When a function is called, record start time
        for fc in event.get_function_calls():
            self.total_llm_calls += 1
            # Try to get args from the function call to create unique ID
            args = fc.args if hasattr(fc, "args") else {}
            call_id = self._get_call_id(fc.name, args)
            start_key = f"{fc.name}:{call_id}_start"
            # Read start time from callback_state (set by before_tool)
            start_time = self._callback_state.get(start_key)
            if start_time is None:
                start_time = time.time()
            # Store with tool_name as key (for simple lookup)
            if fc.name not in self._tool_starts:
                self._tool_starts[fc.name] = []
            self._tool_starts[fc.name].append(start_time)

        # When a function response is received, finalize trace entry
        if event.get_function_responses():
            for fr in event.get_function_responses():
                # Try to get args to compute call_id
                args = fr.args if hasattr(fr, "args") else {}
                call_id = self._get_call_id(fr.name, args)
                
                # Try callback_state first (latency computed by after_tool)
                latency_ms = self._callback_state.get(f"{fr.name}:{call_id}_latency_ms")
                
                # Fallback: compute from start time (use queue-based approach)
                if latency_ms is None and fr.name in self._tool_starts and self._tool_starts[fr.name]:
                    # Pop the OLDEST start time (FIFO queue)
                    start_time = self._tool_starts[fr.name].pop(0)
                    latency_ms = int((time.time() - start_time) * 1000)
                
                self.tool_calls.append({
                    "tool": fr.name,
                    "status": "success" if not (isinstance(fr.response, dict) and fr.response.get("error")) else "failed",
                    "latency_ms": latency_ms or 0,
                })

                if isinstance(fr.response, dict) and fr.response.get("error"):
                    self.fallbacks_triggered += 1

        if event.is_final_response() and not event.get_function_calls():
            self.total_llm_calls += 1
            log.debug("trace_final_response", total_llm_calls=self.total_llm_calls)

    @property
    def elapsed_ms(self) -> int:
        """Total wall-clock time from agent start to final response."""
        return int((time.time() - self.start_time) * 1000)

    def summary(self) -> dict:
        """Return a full trace summary including token usage."""
        return {
            "tool_calls": self.tool_calls,
            "total_llm_calls": self.total_llm_calls,
            "fallbacks_triggered": self.fallbacks_triggered,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "elapsed_ms": self.elapsed_ms,
        }
