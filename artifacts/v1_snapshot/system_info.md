# Pelgo Career Intelligence — System Snapshot

## 🤖 LLM Model & Infrastructure
- **Model**: `sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP`
- **Provider**: OpenAI-Compatible (vLLM)
- **Deployment**: Local/Remote RTX 5090 instance
- **Context Handling**: 
    - Conditional `enable_thinking` injection (active for Qwen models).
    - `response_format={"type": "json_object"}` with robust array unwrapping.

## 🧠 Agent Architecture
The system uses a **Sequential Agent Pipeline** (`google-adk`) for maximum reliability and traceability:

1. **Career Tool Agent**: 
   - **Extract**: Parses JDs into structured requirements.
   - **Score**: Deterministic 5-dimension scoring.
   - **Prioritize**: Ranks skill gaps by match-gain impact.
   - **Research**: Finds learning resources via DuckDuckGo + LLM Fallback.
2. **Career Formatter**: 
   - Aggregates all tool outputs into a polished, candidate-facing `AgentOutput` schema.
   - Generates professional reasoning and actionable next steps.

## 📊 Scoring Formula (v1.0)
`overall_score = (skills * 0.40) + (experience * 0.20) + (seniority_fit * 0.20) + (domain_match * 0.15) + (completeness * 0.05)`

- **Skill Score**: 80% weight on required skills, 20% on nice-to-haves.
- **Seniority Fit**: Ordinal distance (0.0 to 1.0) between candidate and JD levels.

## 🔍 Traceability
The system captures a complete execution trace for every job, stored in `agent_trace` JSON:
- **Tool Calls**: Every tool execution with timing (latency), status, and parameters.
- **LLM Stats**: Total LLM calls and token usage (if provided by endpoint).
- **Fallbacks**: Tracking of any tool-level failures or fallback triggers.

---
*Snapshot generated on 2026-04-30*
