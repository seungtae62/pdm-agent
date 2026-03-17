---
name: Thought 렌더링 버그 (해결됨)
description: 스트리밍 완료 후 thought step이 1개만 저장되던 버그 - 해결 완료
type: project
---

**해결됨 (2026-03-15)**

reasoning 노드 외 LLM 호출(generate_report, generate_work_order)의 토큰이 ReasoningTokenEvent로 잘못 emit되어 팬텀 thought step이 생성되고, 완료 시 `in_reasoning=False`라 저장되지 않던 버그.

**수정 내용:**
1. Backend (`agent_runner.py`): `current_node` 추적 → reasoning 노드에서만 `ReasoningTokenEvent` emit
2. Frontend (`app.py`): `in_reasoning` 가드 추가 (방어적 처리)
