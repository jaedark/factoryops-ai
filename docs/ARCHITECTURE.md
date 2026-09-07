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
