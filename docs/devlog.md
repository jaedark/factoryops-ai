# FactoryOps AI Development Log

## Day 1 - 2026-08-11

### 오늘 목표

FactoryOps AI 프로젝트의 기본 Backend 구조를 만들고
제조 장애 데이터를 저장하고 조회할 수 있는 API를 구축한다.

### 구현 내용

- Python 가상환경 구성
- FastAPI 프로젝트 생성
- GET /health API
- POST /incidents API
- GET /incidents API
- GET /incidents/{incident_id} API
- Pydantic Schema 구성
- SQLAlchemy ORM 구성
- SQLite DB 연결
- 서버 재시작 후 데이터 유지 확인
- Repository 계층 분리
- Service 계층 분리
- Git 프로젝트 초기화

### 오늘 학습한 내용

- Python Virtual Environment
- FastAPI Router
- REST API
- Pydantic Validation
- Dependency Injection
- SQLAlchemy ORM
- Repository Pattern
- Service Layer

### 발생한 문제

Uvicorn 실행 시 프로젝트 가상환경이 아닌
Hermes Python 환경이 사용되어 SQLAlchemy import 오류가 발생했다.

### 해결 방법

프로젝트 .venv를 활성화하고

python -m uvicorn backend.main:app --reload

형태로 실행하도록 변경했다.

### 현재 Architecture

Client
→ FastAPI Router
→ Service
→ Repository
→ SQLAlchemy
→ SQLite

### Day 2 목표

- 샘플 제조 장애 데이터 10건 작성
- Seed 기능
- Keyword Search
- pytest 기반 API 테스트