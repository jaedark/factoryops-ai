# FactoryOps AI

**Agentic Manufacturing Operations Platform**

FactoryOps AI는 제조 현장의 장애 이력과 운영 데이터를 기반으로
**검색 → 분석 → Tool 실행 → Agent 판단 → 외부 시스템 연계**까지 확장하는
AI 기반 제조 운영 지원 플랫폼 프로젝트입니다.

단순한 RAG 챗봇을 만드는 것이 아니라,
각 기술을 단계적으로 구현하고 **Evaluation(평가)** 으로 효과를 검증한 뒤
실제 제조 장애 대응 흐름을 수행하는 **Agentic AI(에이전트형 AI) PoC**를 완성하는 것을 목표로 합니다.

---

## 프로젝트 목표

최종적으로 사용자가 제조 장애 상황을 자연어로 요청하면 AI가:

1. 장애 상황을 이해하고
2. 과거 Incident와 기술 정보를 검색하고
3. 원인과 조치 방법을 분석하고
4. 필요한 Tool을 선택해 실제 기능을 실행하고
5. 필요하면 여러 단계를 반복 수행하고
6. 외부 시스템과 연계해 대응/보고까지 수행하는 구조를 구현합니다.

```text
User
  ↓
FactoryOps AI API
  ↓
AI Orchestrator
  ↓
Incident Analysis / Knowledge Search
Maintenance Recommendation / Report Agent
  ↓
Tools / MCP
  ↓
Incident DB / External Systems
  ↓
Vector Search / RAG
  ↓
LLM
```

### 핵심 설계 원칙

- 처음부터 Agent Framework에 의존하지 않습니다.
- 기존 방식의 한계를 확인한 뒤 다음 기술을 추가합니다.
- 새로운 기술은 구현 자체가 아니라 **평가 결과를 기준으로 채택**합니다.
- Tool은 기존 Business Logic을 재사용하는 얇은 Wrapper로 유지합니다.
- LLM이 Tool을 선택하더라도 실제 실행 권한과 검증은 Application이 가집니다.
- Retrieval 계층과 Agent 계층의 책임을 분리합니다.

---

## 현재 진행 상태

### DAY1 — Backend Foundation ✅

- FastAPI Backend
- SQLite + SQLAlchemy
- Incident CRUD / Persistence
- Repository / Service Architecture

### DAY2 — Seed & Keyword Search ✅

- 제조 장애 Sample Data
- Idempotent Seed API
- Keyword Search
- Keyword 기반 검색의 의미 검색 한계 확인

### DAY3 — Vector Search ✅

- Multilingual Embedding
- Cosine Similarity
- Semantic Search
- 자연어 표현이 달라도 의미 기반으로 장애 검색

### DAY4 — RAG Incident Analysis ✅

- Retrieval → Context → Prompt → Generation Pipeline
- Gemini 기반 Generation
- 장애 분석 결과와 Source 반환
- Similarity Threshold 적용

### DAY5 — Hybrid Search & Retrieval Evaluation ✅

- Keyword + Vector Hybrid Search
- RRF(Reciprocal Rank Fusion)
- Retrieval Evaluation Dataset
- Hit@1 / Hit@3 / MRR 측정

### DAY6 — Reranking & Hard Retrieval Evaluation ✅

Cross Encoder Reranker를 실제 Hard Query에 적용하고 효과를 검증했습니다.

| Strategy | Hit@1 | Hit@3 | MRR |
| --- | ---: | ---: | ---: |
| Vector Search | 60% | 100% | 0.8000 |
| **RRF** | **70%** | **100%** | **0.8500** |
| Vector + Rerank | 70% | 90% | 0.8000 |
| RRF + Rerank | 70% | 90% | 0.8000 |

Reranker가 일부 실패 Query는 개선했지만 새로운 Regression을 만들고 Hit@3/MRR이 악화되어
**기본 Retrieval Pipeline에는 채택하지 않았습니다.**

현재 기본 검색 전략:

```text
Keyword Search
     +
Vector Search
     ↓
RRF
     ↓
Top-K
     ↓
RAG
```

### DAY7 — Tool Calling ✅

RAG가 정해진 검색 흐름만 수행하던 구조에서,
**LLM이 사용자 요청에 맞는 Tool을 선택하는 구조**로 확장했습니다.

현재 Tool:

- `search_incidents`
  - 자연어 증상/상황 기반 유사 Incident 검색
  - 기존 RRF Retrieval 재사용
- `get_incident`
  - Incident ID 기반 상세 조회
- `get_equipment_incidents`
  - 장비별 장애 이력 조회

Tool Calling Flow:

```text
POST /tools/chat
    ↓
ToolCallingService
    ↓
Gemini + Tool Schemas
    ↓
Function Call
    ↓
Tool Name / Arguments Validation
    ↓
Tool Registry
    ↓
Existing Service 실행
    ↓
Tool Result
    ↓
Gemini Final Answer
```

### Tool Calling 안전장치

- 허용된 Tool만 `_TOOL_REGISTRY`에서 실행
- Pydantic으로 Tool Arguments 검증
- 미등록 Tool 실행 차단
- 잘못된 Arguments 차단
- 자동 테스트에서는 Gemini API를 Mock 처리
- 현재는 요청당 **Single-Step Tool Calling**만 지원

실제 Gemini 수동 테스트에서도 다음 시나리오를 확인했습니다.

```text
"Robot-01 장애 이력 알려줘"
        ↓
get_equipment_incidents
        ↓
DB 조회
        ↓
Gemini 최종 한국어 답변
```

현재 테스트 상태:

```text
33 passed, 2 warnings
```

---

## 현재 Architecture

```text
User
  ↓
FastAPI
  ↓
ToolCallingService / RAG Service
  ↓
┌──────────────────────────────┐
│ Tool Layer                   │
│ - search_incidents           │
│ - get_incident               │
│ - get_equipment_incidents    │
└──────────────────────────────┘
  ↓
Existing Services
  ├─ IncidentService
  ├─ RrfSearchService
  ├─ VectorSearchService
  └─ RAG Services
  ↓
Repository / SQLite
  ↓
Gemini LLM
```

---

## 주요 API

### System

- `GET /health`

### Incident

- `POST /incidents`
- `GET /incidents`
- `GET /incidents/{incident_id}`
- `GET /incidents/search`
- `GET /incidents/vector-search`

### Admin

- `POST /admin/seed`

### RAG

- `POST /rag/analyze`

### Tool Calling

- `POST /tools/chat`

예시 요청:

```json
{
  "message": "Robot-01 장애 이력 알려줘"
}
```

예시 응답 구조:

```json
{
  "answer": "...",
  "tool_called": "get_equipment_incidents",
  "tool_arguments": {
    "equipment_name": "Robot-01"
  },
  "tool_result": []
}
```

---

## 기술 스택

### Backend

- Python
- FastAPI
- Pydantic
- SQLAlchemy
- SQLite

### AI / Search

- Gemini API
- Embedding Model
- Vector Search
- Keyword Search
- RRF
- RAG
- Cross Encoder Reranker (experimental)
- Gemini Function / Tool Calling

### Quality

- pytest
- Mock-based LLM Test
- Retrieval Evaluation
- Hit@1 / Hit@3 / MRR

---

## Roadmap

### Completed

- [x] Backend Foundation
- [x] Sample Data / Seed
- [x] Keyword Search
- [x] Vector Search
- [x] RAG
- [x] Hybrid Search / RRF
- [x] Retrieval Evaluation
- [x] Reranking Experiment
- [x] Tool Layer
- [x] Gemini Tool Calling

### Next

- [ ] **Single Agent / Agent Loop**
- [ ] Multi-Agent Basics
- [ ] Agent Orchestration
- [ ] MCP Server
- [ ] MCP Client / Tool Discovery
- [ ] External System Integration
- [ ] Agentic RAG
- [ ] Memory / Context
- [ ] Guardrails / Human Approval
- [ ] Agent Evaluation
- [ ] Observability
- [ ] Retry / Timeout / Fallback
- [ ] Architecture Refactoring
- [ ] Docker
- [ ] Configuration / Secrets
- [ ] Cloud Deployment
- [ ] Production API
- [ ] End-to-End Scenario
- [ ] End-to-End Evaluation
- [ ] Demo / Portfolio / Final Presentation
- [ ] Optional: Knowledge Graph / Ontology

---

## 다음 단계 — DAY8

DAY7은 한 요청에 대해 최대 한 번 Tool을 호출합니다.

```text
User
  ↓
LLM
  ↓
Tool
  ↓
Final Answer
```

DAY8부터는 Tool 결과를 **Observation(관찰 결과)** 으로 사용해
LLM이 다음 행동을 다시 판단하는 **Single Agent(단일 에이전트)** 구조로 확장합니다.

```text
User Request
  ↓
LLM Decision
  ↓
Tool Execution
  ↓
Observation
  ↓
LLM Re-Decision
  ↓
필요하면 추가 Tool 실행
  ↓
Final Answer
```

이 구조를 기반으로 이후 Multi-Agent, Orchestration, MCP와 외부 시스템 연계까지 확장합니다.

---

## 최종 목표

이 프로젝트의 최종 결과는 단순한 AI Chatbot이 아닙니다.

**제조 장애 상황을 이해하고, 필요한 데이터를 검색하고, 적절한 Tool을 선택해 실행하며,
여러 단계의 판단을 거쳐 장애 대응과 보고 업무를 지원하는 Agentic Manufacturing Operations Platform**을 구현하는 것이 목표입니다.

---

## Docker 실행

Docker image는 고정된 Python 3.14.3 runtime, 애플리케이션 의존성, FastAPI 소스를 함께 패키징합니다.

```powershell
docker build -t factory-agent:day22 .
docker run --rm -p 8000:8000 -e GEMINI_API_KEY=$env:GEMINI_API_KEY factory-agent:day22
```

Compose로 실행하려면 다음 명령을 사용합니다.

```powershell
docker compose up --build -d
docker compose ps
docker compose down
```

기본 health endpoint는 `http://localhost:8000/health`입니다. `GEMINI_API_KEY`와 `GEMINI_MODEL`은 image에 포함하지 않고 실행 환경에서 주입합니다. 포트 충돌이 있으면 `FACTORY_AGENT_PORT`로 host port를 변경할 수 있습니다.

현재 session memory, approval store, circuit breaker는 process memory 기반입니다. 따라서 기본 container는 단일 process와 Uvicorn worker 1개로 실행합니다. SQLite `factoryops.db`도 container filesystem에 생성되므로 container를 삭제하면 데이터가 유지되지 않습니다.

## Configuration

로컬 실행에서는 repository root의 `.env`를 읽고, Docker와 Cloud에서는 같은 이름의 환경변수를 주입합니다. 시작용 파일은 `.env.example`이며 실제 secret은 commit하지 않습니다.

| Variable | Required | Default | Description | Secret |
| --- | --- | --- | --- | --- |
| `FACTORY_AGENT_API_KEY` | 보호 API 사용 시 | 없음 | `X-API-Key` 인증 key | Yes |
| `API_MAX_REQUEST_BYTES` | No | `1048576` | HTTP request body 상한 | No |
| `GEMINI_API_KEY` | LLM 사용 시 | 없음 | Gemini API 인증 | Yes |
| `GEMINI_MODEL` | No | `gemini-2.5-flash` | Gemini model | No |
| `DATABASE_URL` | No | `sqlite:///./factoryops.db` | SQLAlchemy connection URL | Depends |
| `FACTORY_AGENT_HOST` | No | `0.0.0.0` | API bind host | No |
| `FACTORY_AGENT_PORT` | No | `8000` | API bind port | No |
| `AGENT_MAX_STEPS` | No | `5` | Agent loop 최대 step | No |
| `LLM_TIMEOUT_SECONDS` | No | `15` | LLM timeout 정책값 | No |
| `TOOL_TIMEOUT_SECONDS` | No | `5` | Tool timeout 정책값 | No |
| `RETRY_MAX_ATTEMPTS` | No | `3` | transient 오류 최대 시도 수 | No |
| `CIRCUIT_FAILURE_THRESHOLD` | No | `3` | Circuit open 연속 실패 기준 | No |
| `RERANKER_MODEL` | No | multilingual CrossEncoder | Reranker model | No |

```powershell
Copy-Item .env.example .env
# .env에 로컬 secret을 입력한 뒤 실행
.\.venv\Scripts\python.exe -m uvicorn backend.main:app

docker run --rm -p 8000:8000 --env-file .env factory-agent:day22
```

`GEMINI_API_KEY`와 `FACTORY_AGENT_API_KEY`가 없어도 프로세스와 `/health`는 시작됩니다. 보호 API는 `FACTORY_AGENT_API_KEY`가 설정되지 않으면 503, header가 없거나 틀리면 401을 반환합니다. 실제 LLM 호출에는 Gemini key가 필요합니다. 설정값이나 예외에는 API key 및 credential이 포함된 database URL을 출력하지 않습니다.

## Production API Contract

`GET /health`, `GET /ready`, `/docs`, `/openapi.json`은 애플리케이션 API key 없이 사용할 수 있습니다. 나머지 `/admin`, `/incidents`, `/rag`, `/tools`, `/agent` API는 모두 `X-API-Key` header가 필요합니다. Cloud Run IAM 인증과 애플리케이션 API key 인증은 별도 경계이므로 private Cloud Run의 보호 route 호출에는 둘 다 필요합니다.

```powershell
$headers = @{ "X-API-Key" = $env:FACTORY_AGENT_API_KEY }
Invoke-RestMethod http://localhost:8000/incidents -Headers $headers
```

`/health`는 process liveness만 확인하며 DB, Gemini, model을 호출하지 않습니다. `/ready`는 저비용 DB `SELECT 1`을 수행하고, `APP_ENV=production`에서는 Gemini key와 Factory Agent API key가 모두 설정됐는지도 확인합니다. 오류 응답은 다음 공통 envelope를 사용하며 `X-Request-ID` response header와 body의 `request_id`가 일치합니다.

```json
{
  "error": {
    "code": "INVALID_REQUEST",
    "message": "Request is invalid",
    "request_id": "req-..."
  }
}
```

Agent message는 최대 8,000자, session ID와 approval ID는 최대 128자입니다. HTTP body는 기본 1 MiB로 제한되고 incident list/search는 한 요청당 최대 100건입니다. Request log에는 method, path, status, request ID, latency만 기록하며 body와 API key는 기록하지 않습니다. CORS와 process-local rate limiter는 추가하지 않았습니다.

## Google Cloud Run Deployment

DAY24 배포 경로는 Cloud Build가 기존 Dockerfile을 build하고 Artifact Registry에 Git commit 기반 tag로 push한 다음 Cloud Run에 배포하는 구조입니다. 기본 배포는 인증이 필요한 private service입니다.

### Prerequisites

- Google Cloud CLI 설치 및 `gcloud auth login` 완료
- 과금이 연결된 기존 Google Cloud project 선택
- Cloud Build 실행 계정에 Cloud Build, Artifact Registry write, Cloud Run deploy 권한 부여
- Cloud Build 실행 계정에 runtime service account를 사용할 `iam.serviceAccounts.actAs` 권한 부여

```powershell
gcloud config set project <PROJECT_ID>
gcloud auth list
gcloud config get-value project

# 로컬 secret은 Secret Manager secret이 아직 없을 때만 사용됩니다.
$env:GEMINI_API_KEY="<LOCAL_SECRET>"
$env:FACTORY_AGENT_API_KEY="<LOCAL_SECRET>"
.\scripts\deploy_cloud_run.ps1
```

Script는 필요한 API를 활성화하고 `asia-northeast3`의 `factory-agent` Artifact Registry repository, `factory-agent-runtime` runtime service account, `factory-agent-gemini-api-key`와 `factory-agent-api-key` secret을 확인합니다. 없는 secret은 대응하는 로컬 환경변수가 있을 때만 생성하며 값은 source나 build config에 기록하지 않습니다. Runtime service account에는 두 secret의 `roles/secretmanager.secretAccessor`만 부여합니다.

Cloud Build를 직접 다시 실행하려면 활성화된 secret version과 Git SHA를 substitution으로 전달합니다.

```powershell
$tag = (git rev-parse --short=12 HEAD).Trim()
gcloud builds submit . --config=cloudbuild.yaml --region=asia-northeast3 `
  --substitutions="_IMAGE_TAG=$tag,_SECRET_VERSION=<GEMINI_VERSION>,_API_SECRET_VERSION=<API_KEY_VERSION>"
```

기본 Cloud Run 설정은 `2 CPU`, `4Gi memory`, `min-instances=0`, `max-instances=1`, `concurrency=1`, Uvicorn worker 1개입니다. `PORT`는 Cloud Run이 주입하며 애플리케이션이 `FACTORY_AGENT_PORT`보다 우선 사용합니다. Public access가 필요하면 인증과 권한 설계를 먼저 추가한 후 명시적으로 변경해야 합니다.

인증된 smoke test와 log 조회:

```powershell
$url = gcloud run services describe factory-agent --region=asia-northeast3 --format="value(status.url)"
$token = gcloud auth print-identity-token
Invoke-RestMethod "$url/health" -Headers @{ Authorization = "Bearer $token" }
Invoke-RestMethod "$url/ready" -Headers @{ Authorization = "Bearer $token" }
Invoke-RestMethod "$url/incidents" -Headers @{
  Authorization = "Bearer $token"
  "X-API-Key" = $env:FACTORY_AGENT_API_KEY
}
gcloud run services logs read factory-agent --region=asia-northeast3 --limit=50
```

Cloud Run의 SQLite filesystem, session memory, approval store, circuit breaker는 모두 ephemeral입니다. `max-instances=1`과 `concurrency=1`은 demo 중 state 분산을 줄일 뿐 restart나 revision 교체 시 persistence를 보장하지 않습니다. Torch와 sentence-transformers 때문에 image와 cold start가 크며, 현재 embedding model은 process import 시 load됩니다.

Cloud Run, Artifact Registry, Cloud Build, Secret Manager는 비용이 발생할 수 있습니다. Demo 확인 후 필요하지 않은 resource는 다음 명령으로 정리합니다.

```powershell
gcloud run services delete factory-agent --region=asia-northeast3
gcloud artifacts repositories delete factory-agent --location=asia-northeast3
gcloud secrets delete factory-agent-gemini-api-key
gcloud secrets delete factory-agent-api-key
gcloud iam service-accounts delete factory-agent-runtime@<PROJECT_ID>.iam.gserviceaccount.com
```

## End-to-End Demo Scenario

DAY27은 새 Agent workflow를 추가하지 않고 기존 API, Agent loop, Industrial
Data, Incident RAG, memory, guardrail, approval, observability, resilience 연결을
deterministic API test로 검증합니다.

대표 분석 흐름:

```text
POST /agent/chat
  -> X-API-Key
  -> get_equipment_status (current industrial data)
  -> search_incidents (historical RRF retrieval)
  -> get_incident (selected incident detail)
  -> final synthesis
```

대표 승인 흐름은 기존 least-privilege 경계를 유지합니다. `/agent/chat`의
Incident Analysis Agent에는 write tool 권한이 없으므로 안전한 dummy 정비
요청은 `/tools/chat`에서 생성하고, 반환된 `approval_id`를 approval API로
승인합니다. 승인 전에는 tool이 실행되지 않으며 동일 ID 재승인은 차단됩니다.

Mock LLM과 임시 SQLite를 사용하는 비용 없는 E2E test:

```powershell
$env:PYTHONPATH="C:\Projects\FactoryOpsAI"
.\.venv\Scripts\python.exe -m pytest tests/e2e -q
```

실행 중인 API와 실제 Gemini 설정을 사용하는 수동 demo:

```powershell
$env:FACTORY_AGENT_API_KEY="<FACTORY_AGENT_API_KEY>"
.\scripts\demo_e2e.ps1

# Pending dummy maintenance action까지 명시적으로 승인
.\scripts\demo_e2e.ps1 -ApproveMaintenanceRequest
```

Script는 `/health`, `/ready`, 분석 요청, 정비 요청 및 선택적 승인을 순서대로
호출하며 secret을 출력하지 않습니다. 상세 expected evidence와 알려진 한계는
[`docs/E2E_SCENARIOS.md`](docs/E2E_SCENARIOS.md)에 정리되어 있습니다.
