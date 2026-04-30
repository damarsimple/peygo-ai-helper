# Pelgo AI Lead Assignment — Career Intelligence Agent

## Quick Start

```bash
cp .env.example .env          # Add your OPENAI_API_KEY or configure OpenAI-compatible endpoint
docker compose up --build     # postgres, api (:8000), worker x2, auto-seed
# Seed runs automatically at container boot
# Visit http://localhost:8000 for the frontend
# Visit http://localhost:8000/docs for Swagger API docs
```

## Framework Choice

Google ADK — chosen for native session-scoped state management, plain-function tool
registration, and real-time event streaming via `run_async()` that lets the orchestrator
collect a verified `agent_trace` without relying on the LLM to self-report.

The agent uses ADK's **`SequentialAgent` pipeline pattern**. Structured output is
enforced at the vLLM layer via **llguidance** (`response_format: json_schema`), which
guillotine-trims the model's token space so the response is always schema-valid JSON —
no fence-stripping or manual repair needed. Configured via LiteLLM to support any
OpenAI-compatible endpoint (OpenAI, vLLM, Ollama, etc.).

## Agent Pipeline

The career coach is a `SequentialAgent` with two sub-agents:

1. **`tool_agent`** — Executes the multi-step tool workflow, writing intermediate results
   (`score`, `gap_skills`, `prioritized_gaps`) to `session.state` via `tool_context.state`.
   Uses `response_format: json_schema` (llguidance) for guaranteed schema-valid JSON
   from every tool call.
2. **`formatter_agent`** — Reads intermediate state via instruction templating
   (`{score}`, `{gap_skills}`, `{job_id}`) and produces the final schema-compliant JSON.
   Has `output_schema=AgentOutputForLLM` and `output_key="final_output"` with llguidance
   enforcement for strict output validation.

### Runtime Tool-Call Sequencing

The agent decides its tool-call sequence at runtime based on data it discovers:

1. **`extract_jd_requirements`** — Always first. Parses JD into structured requirements.
   Uses `json_schema` response format (llguidance) — output is guaranteed schema-valid.
2. **`score_candidate_against_requirements`** — Always second. Uses LLM-powered SEMANTIC
   matching to compute match score + gap skills (falls back to exact matching if LLM fails).
   Schema enforced via llguidance — no manual JSON repair needed.
3. **`prioritise_skill_gaps`** — Ranks gaps by estimated impact on getting the role.
4. **Focused research** — Researches top 1–2 prioritized gaps. If confidence is "low",
   may research 1 additional gap (max 4 research calls total).

The LLM decides when to skip steps (e.g., if confidence is high, research is skipped).
This is enforced in the system prompt AND guarded by the `JobRunner` orchestrator.

## Confidence Heuristic

Computed from measurable signals using LLM-powered semantic matching:

| Level | skill_match_ratio | jd_completeness |
|-------|-------------------|-----------------|
| High | ≥ 0.7 | ≥ 0.6 |
| Medium | ≥ 0.4 | ≥ 0.4 |
| Low | Below medium thresholds | Below medium thresholds |

Formula: `overall_score = skill_score×40 + jd_completeness×5 + exp_distance×20 + seniority_fit×20 + domain_match×15`

Where `skill_score = required_ratio×0.8 + nice_to_have_ratio×0.2` and `seniority_fit`
is computed from ordinal distance between candidate and JD seniority levels
(perfect match = 1.0, one level off = 0.7, two = 0.3, three = 0.0).

Confidence is derived from detectable signals (JD completeness, ratio of required skills
matched, domain distance) — not asserted arbitrarily. Low confidence triggers the
orchestrator guard which reduces the score by 15 points (Path 1) or 10 points (Path 2 fallback)
and flags the result.

## Termination Condition

The pipeline terminates when:
1. `tool_agent` completes all tool calls and produces a final response.
2. `formatter_agent` reads session state, generates schema-compliant JSON,
   and ADK's `output_schema` validates it (or raises `ValidationError` at high temps).
3. `JobRunner` reads `session.state["final_output"]` as the primary path, falling back to
   manual state reconstruction if validation fails.
4. The low-confidence guard is applied if confidence is "low" — score is reduced and flagged.

## Failure Modes

### Tool Timeout
**Strategy**: `asyncio.wait_for(300s)` → if exceeded, partial trace stored.
On first 2 attempts: job re-queued to `pending` for retry.
On 3rd attempt: job moves to `failed` with error detail and partial agent_trace.
**Why**: 300s allows for LLM tool calls + web searches + semantic matching. After 3 retries, the job is
dead-lettered to prevent infinite loops.

### Invalid Tool Output
**Strategy**: `extract_jd_requirements` uses `json_schema` response format (llguidance)
which guarantees schema-valid JSON. If Pydantic validation still fails (schema mismatch),
retries up to 3 times with progressively stricter prompts before returning a fallback dict
with empty skill arrays.
`score_candidate_against_requirements` handles empty/missing requirement fields gracefully
(returns low score with low confidence).
**Decision**: Retry with different prompt > proceed with partial data > abort.
**Why**: Schema validation failures are rare with llguidance (model output is constrained),
but may occur if the schema definition and prompt disagree. A different prompt format
often fixes them.

### Low Confidence Score
**Strategy**: Orchestrator-enforced guard in `JobRunner._apply_low_confidence_guard()`.
When confidence is "low":
- Score reduced by 15 points to penalise uncertainty
- Reasoning appended with "[LOW CONFIDENCE]" flag
- Fallback counter incremented in agent_trace
- Agent also instructed to research more data if confidence is low
**Decision**: Never silently return low-confidence scores. Always flag and penalise.
**Why**: Low confidence means insufficient signal. Penalising encourages reviewers to
treat the match cautiously. The prompt-directed research step tries to gather more signal
before the orchestrator guard is applied.

### Formatter Schema Rejection
**Strategy**: `JobRunner` catches `ValidationError` and reconstructs `AgentOutput` from
intermediate `session.state` keys (score, gap_skills, prioritized_gaps). With llguidance,
this path is rarely hit — schema violations are constrained by the vLLM layer.
**Why**: Edge cases at high temperatures. The tool_agent already collected valid data,
so reconstruction preserves the result.

## Tool Suite

| Tool | Description | External Calls | Caching |
|------|------------|----------------|---------|
| `extract_jd_requirements` | Parses JD text/URL → structured requirements | LLM (json_schema via llguidance) | SHA-256 cache, 24h TTL |
| `score_candidate_against_requirements` | LLM-powered semantic scoring + gap detection | LLM (json_schema via llguidance) | N/A |
| `research_skill_resources` | Finds learning resources via Tavily (primary) or DuckDuckGo (fallback) | Tavily API + LLM (json_schema) | Per-skill cache, 24h TTL |
| `prioritise_skill_gaps` | LLM-ranks gaps by impact | LLM (json_schema via llguidance) | N/A (per-job) |

**Note on `prioritise_skill_gaps` parameter naming**: The spec references `job_market_context`;
we use `seniority_context` because seniority level is the most actionable job-market signal
available from the extracted JD. This is a deliberate narrowing — a production system would
include salary band, location, and demand trend data in a broader context object.

### Google ADK Integration
All 4 tools use `response_format: json_schema` via vLLM's llguidance backend, which
constrains token generation to produce only schema-valid JSON. Configured with
LiteLLM's `LiteLlm` wrapper to support any OpenAI-compatible endpoint.
The agent uses ADK's:
- `SequentialAgent` for pipeline orchestration
- `DatabaseSessionService` for persistent session state
- `run_async()` event streaming for real trace collection
- `output_schema` with flattened schema (llguidance) on the formatter agent
- `before_agent_callback` for session state injection

ADK provided session-scoped state management that LangGraph does not have built-in,
plus real-time event streaming that lets us collect traces from the orchestrator rather
than fabricating them.

## Key Architecture Decisions

1. **`tool_agent` (tools + llguidance schema) →
   `formatter_agent` (output_schema + llguidance schema)**. Every LLM call uses
   `response_format: json_schema` via vLLM's llguidance backend, which constrains
   token generation to produce only schema-valid JSON. This eliminates the need
   for manual fence-stripping, array-unwrapping, or Pydantic repair.

2. **Session state via `before_agent_callback`**: `run_async()` does **not** accept a
   `state_delta` parameter (it is not a documented ADK API). Instead, `JobRunner` populates
   a shared dict with candidate profile and job metadata; the agent's
   `before_agent_callback` reads this dict and writes to
   `context.state["candidate_profile"]` before the LLM runs. Intermediate results
   (`score`, `gap_skills`, `prioritized_gaps`) are set in `tool_context.state` by the
   scorer and prioritiser tools, and read back by the fallback path.

3. **Separate ADK database**: ADK's `DatabaseSessionService` creates its own tables in a
   dedicated `pelgo_adk` database. Business data (candidates, match_jobs) lives in `pelgo`.
   `init.sql` creates `pelgo_adk` at container boot.

4. **Content-hash cache key for text JDs**: JD text is hashed with SHA-256 so two
   different text JDs never collide in the extraction cache. Separate namespaces
   (`jd:url:` vs `jd:text:`) prevent URL and text JDs from colliding.

5. **DuckDuckGo with LLM fallback**: Free, no API key required. When it returns nothing,
   `_generate_placeholder_resources()` uses the configured LLM to suggest resources.
   Swappable for Serper/Tavily in production.

## Trade-Offs

### Framework: ADK vs LangGraph vs CrewAI

**Chose ADK because:**
- **Session state**: ADK provides `session.state` as a first-class concept, eliminating
  the need for a custom state management layer. LangGraph requires manual `StateGraph`
  definition with explicit reducer logic for each state transition.
- **Event streaming**: `run_async()` returns a stream of events we can collect traces from.
  LangGraph requires custom checkpointers for equivalent functionality.
- **Tool registration**: Plain Python functions decorated with type hints. CrewAI requires
  wrapping tools in `BaseTool` classes with verbose boilerplate.
- **ADK stretch bonus**: Using ADK for all 4 tools qualifies for the +10 bonus points.

**Trade-off accepted**: ADK's `SequentialAgent` enforces a fixed sub-agent order. We cannot
implement a fully free-form graph where edges are decided at runtime (as LangGraph `ConditionalEdge`
would allow). Our compromise: the LLM decides the tool-call sequence *within* the `tool_agent`,
guided by conditional instructions in the system prompt. The orchestrator (`JobRunner`) provides
additional enforcement via the low-confidence guard.

### Semantic Scoring vs Pure Deterministic Scoring

**Chose LLM-powered semantic scoring** for `score_candidate_against_requirements`.
The score formula (`skill_score×40 + jd_completeness×5 + exp_distance×20 + seniority_fit×20 + domain_match×15`)
uses LLM-based semantic matching to compare skills (e.g., "C/C++" matches "C++",
"Arduino" matches "Hardware Interfaces"). Falls back to exact string matching if LLM fails.
`skill_score` includes partial credit for nice-to-have skills (20% weight vs 80% for required).
`seniority_fit` is computed from ordinal distance between candidate and JD seniority levels.

**Trade-off**: Pure deterministic scoring would be simpler and faster, but LLM semantic
matching captures nuanced skill relationships (e.g., "Docker" ≈ "Containerization").
We use LLM for scoring because accurate skill matching is the core value proposition.
The formula remains deterministic and reproducible — only the skill matching step uses LLM.

### Two-Agent Pipeline vs Single ReAct Agent

**Chose `SequentialAgent` pipeline** (tool_agent → formatter_agent) over a single ReAct agent.

**Trade-off**: A single ReAct agent would naturally alternate between tool calls and reasoning,
potentially producing a better learning_plan. However, ADK's `output_schema` on agents with
tools requires model-level JSON schema enforcement. Our pipeline pattern ensures this:
the tool_agent collects data with llguidance-enforced schemas, the formatter_agent
produces validated JSON with a flattened schema (no `$ref` pointers). Every LLM call
uses `json_schema` response format for guaranteed structured output.

### Caching Strategy

- **JD extraction**: SHA-256 content-hash caching (24h TTL). Separate namespaces for URLs
  (`jd:url:`) and text (`jd:text:`) prevent collisions.
- **Research results**: MD5 hash of `(skill_name, seniority_context)` (24h TTL).

**Trade-off**: Two candidates submitting the same JD URL share a cached extraction. This is
intentional — the same JD should always produce the same extraction. For very large-scale
deployments, Redis would replace `diskcache`, but the cache key strategy remains the same.

### Tavily (Primary) + DuckDuckGo (Fallback) vs Pure Free Search

**Chose Tavily as primary** for structured, high-quality search results with relevance scores.
DuckDuckGo as fallback for zero-cost operation when Tavily fails. LLM generates placeholder
resources if both fail.

**Trade-off**: Tavily requires an API key (free tier available). DuckDuckGo results are less
consistent and may vary by region. For production, Tavily (or Serper) provides the best
balance of quality and cost.

### PDF Resume Parsing

**Implemented PDF upload** via `pdfplumber` + LLM extraction. Extracts complete profile
including skills (with semantic matching for hardware/C++ variants), experience, seniority,
domain, education, and certifications.

**Trade-off**: LLM-based PDF parsing is more flexible than rule-based regex extraction
(because resume formats vary widely), but depends on the LLM's accuracy. The LLM prompt
is strict about JSON schema and includes instructions for extracting ALL skills (including
"C/C++", "Arduino", "I2C", etc.). Edge cases (tables, multi-column layouts) may produce
incomplete extractions. For production, an OCR-enhanced pipeline would be more robust.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/candidate` | Register a candidate (JSON) |
| POST | `/api/v1/candidate/pdf` | Register a candidate (PDF upload) |
| POST | `/api/v1/matches` | Submit job descriptions (≤10) |
| GET | `/api/v1/matches` | List matches (paginated, filterable by `status` and `candidate_id`) |
| GET | `/api/v1/matches/{id}` | Get match details |
| POST | `/api/v1/admin/requeue/{id}` | Requeue failed job |
| GET | `/health` | Health check |

## Testing

```bash
# Unit tests (no running server required)
pytest backend/tests/test_fixes.py -v

# LLM schema enforcement tests (requires vLLM + llguidance endpoint)
pytest backend/tests/test_llguidance.py -v

# Integration tests (requires running server + database)
pytest backend/tests/test_integration.py -v
```

## System Prompts

### Tool Agent
```
You are Pelgo CareerCoach — an autonomous agent that evaluates candidates against
job descriptions and produces learning plans.

## CANDIDATE TO EVALUATE
{candidate_profile}

## RAW RESUME TEXT (use for semantic skill matching)
{raw_resume_text}

## JOB ID
{job_id}

## YOUR DECISION-DRIVEN WORKFLOW
You decide the tool-call sequence at runtime based on the data you discover.
Follow these decision rules:

1. ALWAYS START: Call extract_jd_requirements(job_url_or_text=jd_input)
2. ALWAYS SCORE: Call score_candidate_against_requirements() with semantic matching
3. ALWAYS PRIORITIZE: Call prioritise_skill_gaps() to rank gaps
4. FOCUSED RESEARCH: Research top 1-2 prioritized gaps (max 4 if low confidence)

## SEMANTIC MATCHING INSTRUCTIONS
When scoring, use SEMANTIC matching: "Arduino"/"I2C" → "Hardware Interfaces",
"C/C++" → matches "C++", "React"/"Vue" → "Frontend Development"

Save all intermediate results to session state via tool_context.state.
```

### Formatter Agent
```
You are a JSON Formatter for Pelgo CareerCoach.
Construct the final structured output JSON using session state:

Score data: {score}
Gap skills: {gap_skills}
Job ID: {job_id}

Build the learning_plan from prioritized_gaps and research results.
Each learning_plan item MUST have: skill, priority_rank, estimated_match_gain_pct,
resources (with title, url, estimated_hours, type, relevance_score), rationale.

Return ONLY the exact JSON structure with ALL required fields.
```

## Logging

Structured logging via `structlog` with JSON renderer (configurable with `PRETTY_LOGS`).
Per-worker identity via `WORKER_ID` environment variable.

Key logged events:
- **Worker**: `worker_started`, `worker_shutdown_requested`, `worker_shutting_down`, `worker_stopped`
- **Jobs**: `job_claimed`, `job_completed`, `job_failed`, `job_retry`, `job_failed_timeout`, `job_retry_timeout`
- **Agent**: `job_runner_start`, `job_runner_complete_path1`, `job_runner_complete_path2`, `agent_pipeline_exception`
- **Tools**: `score_computed`, `semantic_scoring_failed`, `jd_cache_hit`, `jd_extract_success`, `jd_extract_fallback`,
  `research_tavily_success`, `research_ddg_success`, `research_llm_fallback`, `research_complete`,
  `skill_gaps_prioritised`, `prioritise_skill_gaps_empty`
- **Callbacks**: `tool_call_started`, `using_prerecorded_latency`, `tool_call_completed`, `tool_error`
- **Guards**: `low_confidence_guard_triggered`
- Plus token usage, per-tool latency, and trace collection events.

## Time Spent

<!-- Update this with your actual hours before submitting -->
Approximately **X hours** over Y days.
