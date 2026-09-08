# Factory Agent Architecture

Factory Agent keeps orchestration in `AgentService` and moves cross-cutting
execution details behind focused components.

```text
FastAPI API
    -> AgentService
        -> AgentRuntimeContext / AgentExecutionConfig
        -> ResilientExecutionService
            -> LlmService
            -> ToolCallingService
                -> Domain Service / Repository / Adapter
```

Cross-cutting responsibilities remain separate:

- Memory: builds request context and stores final user/assistant messages.
- Observability: `ObservabilityRecorder` creates structured trace events.
- Resilience: retry, backoff, timeout metadata, and the LLM circuit breaker.
- Guardrail: decides whether an allowed tool may run automatically.
- Approval: stores the state of high-risk action requests.

## Production API Boundary

```text
HTTP Request
    -> RequestContextMiddleware (request ID, size limit, safe request log)
        -> X-API-Key authentication
            -> FastAPI route
                -> Agent / Tool / Domain service
```

`/health` is a public liveness check. `/ready` is a public readiness check for
validated settings and a low-cost database connection; production readiness
also requires both runtime secrets to be configured. Every business route is
protected by `X-API-Key`. HTTP `request_id` and Agent `trace_id` remain separate:
the first identifies one API request and the second identifies one Agent run.
Errors use a sanitized common envelope and request logs exclude bodies, prompts,
tool results, authorization headers, and secrets.

`ToolCallingService.execute_tool()` is the raw application execution boundary.
`ResilientExecutionService.execute_tool()` adds retry behavior only for
read-only tools. Approval-required write tools are never replayed
automatically.

Dependency direction is one-way: `AgentService` depends on the runtime,
recorder, and resilient executor. Those components do not import
`AgentService`.

## Docker Deployment Boundary

```text
Host
    -> Docker Container (single process, single Uvicorn worker)
        -> FastAPI
            -> AgentService / Agent Runtime
```

The container emits logs to stdout and stderr. Secrets are injected at runtime,
not copied into the image. SQLite and the in-memory memory, approval, and circuit
breaker state are intentionally ephemeral in this development configuration.

## Configuration Flow

```text
Environment / local .env
    -> AppSettings (validated and cached)
        -> AgentExecutionConfig / RetryPolicy / CircuitBreaker
        -> LLM / Database / Retrieval services
```

`GEMINI_API_KEY` remains optional during process startup so health and non-LLM
paths stay available. `LlmService` validates the key immediately before an LLM
call. Settings logs must never include secret values or complete database URLs.

## Google Cloud Deployment

```text
Developer
    -> Cloud Build
        -> Artifact Registry
            -> Cloud Run (one instance, concurrency one)
                -> FastAPI / Agent Runtime

Secret Manager
    -> GEMINI_API_KEY / FACTORY_AGENT_API_KEY environment references
        -> AppSettings
            -> LlmService / API security dependency
```

The Cloud Run service is private by default and uses a dedicated runtime service
account with secret-level accessor permission. SQLite and process-local memory,
approval, and circuit state remain ephemeral. The one-instance demo limit reduces
state divergence but is not a persistence mechanism.
