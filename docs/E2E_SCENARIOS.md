# DAY27 End-to-End Scenarios

DAY27 validates existing integration wiring with deterministic LLM responses.
The pytest scenarios do not call Gemini and use a temporary SQLite database.

## E2E-01: Current State and Historical Analysis

Input: `Robot-01` current state, similar incidents, probable cause, and
maintenance direction.

Expected sequence:

```text
POST /agent/chat
    -> X-API-Key authentication
    -> get_equipment_status
    -> IndustrialDataService / AI4I sample CSV
    -> search_incidents
    -> RRF historical incident retrieval
    -> get_incident
    -> final LLM synthesis
```

Evidence includes the HTTP request ID, Agent trace ID, completed status, three
successful Agent steps, current telemetry, two seeded Robot incidents, and the
final grounded answer. No write tool or approval is expected.

## E2E-02 and E2E-03: Human Approval

The current `/agent/chat` route runs the least-privileged Incident Analysis
Agent, which intentionally cannot create maintenance requests. The existing
`/tools/chat` API is therefore used to request the safe dummy action without
changing production routing.

```text
POST /tools/chat
    -> create_maintenance_request selected
    -> Guardrail high-risk policy
    -> PENDING (tool not executed)
    -> approve: APPROVED -> EXECUTED, exactly one normal API execution
    -> duplicate approve: rejected
```

The reject branch is `PENDING -> REJECTED`; later approval is blocked and the
tool remains unexecuted. This is process-local duplicate prevention for one
approval ID, not distributed exactly-once delivery.

## E2E-04: Session Memory

The first request stores the user message and final answer under
`e2e-memory-robot`. The second request says only "that equipment". The test
asserts that `ContextBuilder` includes the earlier Robot-01 conversation before
the follow-up reaches the LLM. A stateless comparison verifies no prior context
is injected when `session_id` is absent.

## E2E-05 and E2E-06: Security and Resilience

- Missing and incorrect API keys return a sanitized 401 envelope with an HTTP
  request ID.
- The Incident Analysis Agent cannot request `create_maintenance_request`; the
  allowlist blocks it before guardrail evaluation or execution.
- One transient LLM timeout emits `retry_scheduled`, then recovers and completes.
- An unsupported tool fails permanently with no retry loop.

## Execution

Deterministic local tests:

```powershell
$env:PYTHONPATH="C:\Projects\FactoryOpsAI"
.\.venv\Scripts\python.exe -m pytest tests/e2e -q
```

Manual API demo using the configured Gemini service:

```powershell
$env:FACTORY_AGENT_API_KEY="<FACTORY_AGENT_API_KEY>"
.\scripts\demo_e2e.ps1 -ApproveMaintenanceRequest
```

The manual script requires a running API and a configured `GEMINI_API_KEY` for
Agent calls. It never prints either secret.

## Known Limitations

Passing these scenarios does not prove real Gemini answer quality, Cloud Run
deployment, wall-clock timeout enforcement, distributed approval, distributed
memory, multi-instance consistency, or SQLite persistence. The Orchestrator and
MCP client remain separate from the `/agent/chat` runtime path and are covered
by their regression tests rather than forced into this flow.
