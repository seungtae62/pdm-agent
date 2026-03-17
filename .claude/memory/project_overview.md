---
name: project-overview
description: PdM Agent 프로젝트 개요 - 베어링 예지보전 AI 에이전트 시스템, 현재 핵심 기능 개발 중
type: project
---

## PdM Agent (Predictive Maintenance Agent)

제조업 베어링 설비의 예지보전을 위한 AI 에이전트 시스템.

### 아키텍처
- **Edge**: 진동 센서 데이터 → 특징 추출 → 이상감지 (Rule + Statistical + Autoencoder) → Health Index → 이벤트 페이로드 생성
- **Cloud**: PdM Agent (LangGraph, GPT-4o) → 이상 이벤트 해석 → 결함 진단 → RUL 평가 → 정비 권고
- **API**: FastAPI REST + WebSocket 스트리밍
- **UI**: Streamlit 대시보드

### 핵심 기술 스택
Python 3.12 | LangGraph (ReAct) | GPT-4o | MCP Tools | Qdrant (3 Collections) | PostgreSQL | FastAPI | Streamlit

### StateGraph 노드 (7개)
load_memory → reasoning → tool_executor → parse_diagnosis → generate_report → generate_work_order → save_memory

### MCP Tools (5개)
search_maintenance_history, search_equipment_manual, search_analysis_history, notify_maintenance_staff, search_web

### 주요 설계 원칙
1. 단일 에이전트 (멀티 에이전트 금지)
2. 수치 계산 금지 (Edge 결과 해석만)
3. 도메인 지식 기반 해석
4. 자율적 추론 깊이 결정
5. 최소 Function Call

### 현재 상태 (2026-03-17 기준)
- 핵심 기능 개발 완료 단계, UI 고도화 및 부가 기능 추가 중
- Edge 파이프라인, Agent 코어, RAG, API, UI 기본 구조 완성
- Skill 시스템 완성: Knowledge Skills (SKILL_REGISTRY) + Action Skills (RAG/Notification/Web Search) + User Skills (CRUD)
- Chat UI에서 Skill 관리 도구 호출 지원 (create/list/delete user skill)
- `/skills` 슬래시 커맨드 구현 완료 (LLM 호출 없이 즉시 응답)
- 슬래시 커맨드 프레임워크 구축 (`src/api/services/slash_commands/`)

**Why:** 프로젝트 전체 맥락을 빠르게 파악하기 위한 기본 정보.
**How to apply:** 코드 작성/수정 시 설계 원칙 준수, 아키텍처 방향에 맞는 제안 제공.
