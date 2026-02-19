# PdM Agent - Claude Code 지침서

제조업 베어링 설비의 예지보전(Predictive Maintenance)을 위한 AI 에이전트 시스템.
Edge에서 진동 센서 기반 이상감지를 수행하고, Cloud의 PdM Agent가 이상 이벤트를 해석하여 결함 진단, RUL 평가, 정비 권고를 생성한다.

---

## 기술 스택

- **언어**: Python 3.12
- **LLM**: GPT-4o (OpenAI API)
- **에이전트**: LangGraph (ReAct 패턴)
- **Tool 호출**: MCP (Model Context Protocol)
- **Embedding**: OpenAI text-embedding-3-small (1536차원)
- **Vector DB**: Qdrant (Docker, 3개 Collection)
- **Memory**: PostgreSQL (langgraph-checkpoint-postgres)
- **PDF 파싱**: PyMuPDF
- **데모 UI**: Streamlit

---

## 아키텍처 원칙

코드 작성 시 반드시 따를 제약 사항:

1. **단일 에이전트** — 역할별 멀티 에이전트 분리 금지. PdM Agent 하나로 구성한다.
2. **수치 계산 금지** — 에이전트는 Edge가 산출한 정량값을 해석하고 의미를 부여한다. 직접 계산하지 않는다.
3. **도메인 지식 기반 해석** — 시스템 프롬프트에 내장된 도메인 지식으로 전문가 수준의 해석을 수행한다.
4. **자율적 추론 깊이 결정** — 상황에 따라 추론의 깊이와 경로를 에이전트가 스스로 결정한다.
5. **최소 Function Call** — Tool 호출을 최소화하고 에이전트의 추론 능력을 극대화한다.

---

## 개발 환경 설정

```bash
# Python 가상환경
python3.12 -m venv .venv
source .venv/bin/activate

# 의존성 설치
pip install -r requirements.txt

# 환경변수
cp .env.example .env
# .env 파일에 실제 값 입력 (OPENAI_API_KEY, POSTGRES_PASSWORD 등)

# Docker 서비스 (Qdrant, PostgreSQL)
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant
docker run -d --name postgres -p 5432:5432 \
  -e POSTGRES_DB=pdm_agent \
  -e POSTGRES_USER=pdm_agent \
  -e POSTGRES_PASSWORD=<password> \
  postgres:16
```

---

## 코딩 컨벤션

- **네이밍**: snake_case (변수, 함수, 모듈), PascalCase (클래스)
- **포매팅**: Black (기본 설정, line-length 88)
- **import 정렬**: isort (Black 호환 프로파일)
- **타입 힌트**: 함수 시그니처에 타입 힌트 권장. `from __future__ import annotations` 사용
- **독스트링**: 모듈과 public 함수에 간결한 독스트링 작성 (Google 스타일)
- **환경변수**: `python-dotenv`로 `.env`에서 로드. 코드에 시크릿 하드코딩 금지

---

## 커밋 컨벤션

Conventional Commits 형식:

```
feat: 새 기능 추가
fix: 버그 수정
docs: 문서 변경
refactor: 리팩터링
test: 테스트 추가/수정
chore: 빌드, 설정 등 기타
```

---

## 테스트

```bash
# 전체 테스트 실행
pytest

# 특정 모듈 테스트
pytest tests/test_<module>.py -v
```

---

## 핵심 파일 경로

| 경로 | 설명 |
|------|------|
| `agent/graph.py` | LangGraph StateGraph 정의 |
| `agent/state.py` | State 스키마 |
| `agent/nodes/` | 노드 구현 (reasoning, tool_executor 등) |
| `agent/prompts/system_prompt.py` | 시스템 프롬프트 |
| `agent/memory/` | PostgreSQL Memory 관리 |
| `mcp_servers/rag_server.py` | RAG 검색 MCP Server |
| `mcp_servers/notification_server.py` | 알림 MCP Server |
| `scripts/` | 데이터셋 다운로드 등 유틸리티 스크립트 |
| `ui/app.py` | Streamlit 데모 UI |

---

## 설계 문서

- **상세 설계 (구현 순서, StateGraph, RAG, Memory, 페이로드, 시나리오, KPI)**: [`docs/design.md`](docs/design.md)
- **시스템 프롬프트 전문**: [`docs/system-prompt.md`](docs/system-prompt.md)
- **산출물 설계**: [`docs/deliverables/`](docs/deliverables/)
