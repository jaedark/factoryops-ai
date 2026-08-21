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
