---
name: 슬래시 커맨드 구현 현황
description: /skills 슬래시 커맨드 구현 완료 및 프레임워크 구축 현황, 향후 확장 방향
type: project
---

## 완료된 작업 (2026-03-17)

### `/skills` 슬래시 커맨드
- LLM 호출 없이 Knowledge / Action / User Skills를 분류하여 마크다운으로 즉시 응답
- `src/api/services/slash_commands/` 디렉토리 구조:
  - `base.py` — `SlashCommandResult` 데이터 클래스
  - `cmd_skills.py` — `/skills` 핸들러
  - `__init__.py` — `COMMAND_REGISTRY` 기반 라우터
- `llm_chat_runner.py` run() 상단에 슬래시 커맨드 인터셉트 로직 추가
- `tests/test_slash_commands.py` — 6개 테스트 통과

### 프레임워크 확장 방법
- 새 커맨드: `cmd_<name>.py` 파일 생성 → `COMMAND_REGISTRY`에 등록

## 남은 작업

1. **Streamlit UI 통합 테스트** — UI에서 `/skills` 입력 시 정상 렌더링 확인 필요
2. **추가 슬래시 커맨드 후보** — 아직 구체적 요구사항 없음, 필요 시 확장
3. **미해결 버그** — 스트리밍 완료 후 thought step 렌더링 누락 (별도 메모리 참조)

**Why:** 슬래시 커맨드 관련 작업 맥락을 유지하여 후속 대화에서 빠르게 이어가기 위함.
**How to apply:** 슬래시 커맨드 관련 요청 시 이 메모리를 참조하여 구현 위치와 패턴을 확인.
