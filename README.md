# FactoryOps AI

제조 현장의 장애 이력과 기술 문서를 활용하여
장애 검색, 원인 분석, 점검 지원 및 보고서 생성을 지원하는
AI 애플리케이션 프로젝트입니다.

## 목표

기존 제조 운영 경험을 기반으로
Python, FastAPI, RAG, Vector Search 및 AI Agent 기술을 학습하고
실제 동작하는 포트폴리오 서비스를 구축합니다.

## 현재 구현 기능

- FastAPI 서버
- Health Check API
- Incident 등록
- Incident 전체 조회
- Incident 단건 조회
- Pydantic 요청/응답 검증
- SQLAlchemy ORM
- SQLite 데이터 저장
- Repository / Service 구조

## API

- GET /health
- POST /incidents
- GET /incidents
- GET /incidents/{incident_id}

## 기술 스택

- Python
- FastAPI
- Pydantic
- SQLAlchemy
- SQLite

## Architecture

API
→ Service
→ Repository
→ ORM
→ Database

## 향후 계획

- 제조 장애 샘플 데이터 구축
- Keyword Search
- Vector Search
- Hybrid Search
- RAG
- AI Workflow
- Docker
- 자동 테스트