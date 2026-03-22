# 05. 테스트 및 품질 검증

> PdM Agent — 예지보전 AI 에이전트

## 주요 문제 해결 및 기술 리서치 (테스트 단계)

| **이슈** | **문제** | **해결** |
| --- | --- | --- |
| **품질/환각** | 이벤트 페이로드에 없는 특징량 값을 생성하거나 존재하지 않는 결함 패턴을 추론 | Core Skills 내 명확한 해석 규칙 명시 + 시스템 프롬프트에 "수치 계산 금지", "데이터 부족 시 불확실성 고지" 원칙 강제 |
| **속도/지연** | 기존 MCP stdio transport Skills 호출당 2~3초 오버헤드 | Action Skills(in-process 직접 호출)로 전환하여 MCP 프로토콜 오버헤드 제거 + asyncio 비동기 처리 + 실시간 토큰 스트리밍 |
| **보안/가드레일** | 대화형 상호작용에서 시스템 프롬프트 내 도메인 지식 노출 위험 + LLM 수치 계산 시도 | 내부 지침 질문 거부 규칙 + 수치 계산 요청 시 "Edge 영역" 안내 반환 |

## LLM 답변 품질 평가 및 개선

**KPI 1: 이상 감지 → 정비 권고 처리시간 단축률:**

| **항목** | **내용** |
| --- | --- |
| 평가 방식 | 이벤트 수신 ~ 작업지시서 생성까지 소요 시간 측정. 기존 수동 프로세스(설비 확인 + 원인 분석 + 이력 조회 + 보고서 작성 + 작업지시서 작성) 대비 에이전트 처리 시간 비교 |
| KPI 목표 | 90% 이상 단축 |
| 결과 | 기존 수동 수 시간 → 에이전트 수 분 이내 완료. 목표 달성 |
| 비고 | Warning/Critical 이벤트에서 리포트 + 작업지시서 자동 생성까지 포함 |

**KPI 2: Skills 도입 토큰 효율성 개선율:**

| **항목** | **내용** |
| --- | --- |
| 평가 방식 | Skills 도입 전(시스템 프롬프트에 전체 도메인 지식 상주: ~15K 토큰) 대비 도입 후(Core Skills 조건부 로딩) 토큰 사용량 비교 |
| KPI 목표 | 정상 이벤트 60% 절감 |
| 결과 | 정상: ~6K (도입 전 ~15K 대비 60% 절감), 이상: ~12K (필요 Skills만 로딩). 목표 달성 |
| 비고 | 정상 이벤트에서는 Skills 미로드로 토큰 절감. 이상 이벤트에서도 전체 상주 대비 20% 절감 |

**KPI 3: Deep Search 다관점 분석 품질:**

| **항목** | **내용** |
| --- | --- |
| 평가 방식 | 3개 전문가(Maintenance Engineer, Senior Analyst, Equipment Specialist) 중 Critic Pass 비율, confidence score 평균, 병렬 실행 응답 시간(순차 대비) |
| KPI 목표 | Critic Pass Rate 80%+, Confidence Score 평균 0.7+ |
| 결과 | Critic Pass Rate: 초회 Pass 비율 양호, Review-Revise 후 최종 Pass Rate 목표 충족. Confidence Score: KB 데이터 기반 검색에서 0.7+ 달성. 병렬 실행으로 순차 대비 응답 시간 단축 |
| 비고 | Equipment Specialist의 search_web 결과는 confidence score가 상대적으로 낮아 가중치 투표에서 보완적 역할 |

**결함 진단 정확도:**

| **항목** | **내용** |
| --- | --- |
| 평가 방식 | IMS 3개 Dataset 기반 진단 vs 실제 고장 유형 대조 (내륜/외륜/전동체 결함) |
| 결과 | 주파수 특징 명확 구간에서 높은 정확도. 전이 초기 구간은 "의심" 판정으로 보수적 처리 |
| 개선 | fault-diagnosis Skill에 전이 구간 판정 기준 세분화 (고조파, 사이드밴드 패턴 추가) |

**추론 근거 설명 품질:**

| **항목** | **내용** |
| --- | --- |
| 평가 방식 | LLM-as-a-Judge (GPT-4o). 결함 근거 구체성, P-F 판정 논리, 불확실성 명시, 권고 실행 가능성, 도메인 용어 정확성 |
| 결과 | 평균 4.2/5. Warning/Critical에서 높은 점수. Normal은 조기 종료 설계 의도와 부합 |
| 개선 | response-normal Skill에 Thought 1 교차 확인 요약 포함 |

## 성능 및 비용 최적화

| **항목** | **결과** |
| --- | --- |
| **Skills 호출 효율성** | 정상: 0회 / 이상: 1~2회 / 급속열화: 2~3회. "최소 Skills 호출" 원칙 달성 |
| **토큰 사용량** | 정상: ~6K (Core Skills 미로드) / 이상: ~12K (필요 Core Skills만 조건부 로딩). 전체 상시 로드(~15K) 대비 정상에서 60% 절감. MCP 기반 Tool 호출에서 Action Skills(in-process 직접 호출)로 전환하여 프로토콜 오버헤드 제거 |
| **Deep Search 응답 시간** | `asyncio.gather()` 병렬 실행으로 순차 대비 응답 시간 단축. 3개 Research Agent 동시 검색 |

## Deep Search Engine 테스트

| **항목** | **검증 내용** | **결과** |
| --- | --- | --- |
| **STORM 구조 동작** | Decompose → Research(3개 병렬) → Review → Synthesize 전체 파이프라인 정상 동작 | 정상 동작 확인 |
| **병렬 실행** | 3개 Research Agent가 `asyncio.gather()`로 동시 실행되는지 확인 | 병렬 실행 확인, 순차 대비 응답 시간 단축 |
| **Critic Review-Revise** | Revise 판정 시 해당 perspective만 재검색, 최대 3회 제한 | 정상 동작. 3회 초과 시 현재 결과로 진행 확인 |
| **Confidence Score** | 검색 결과 양/품질 기반 휴리스틱 산출 정상 여부 | 0.0~1.0 범위 정상 산출. KB 데이터 기반 검색에서 0.7+ 달성 |
| **가중치 투표** | confidence score에 비례한 합성 기여도 결정 정상 여부 | 정상 동작. 높은 confidence perspective의 결과가 합성에 우세 반영 |
| **트리거 조건** | 사용자 명시적 요청, Critical + 유사 사례 부족 시 정상 발동 | 정상 발동 확인 |

## 예외 처리 및 가드레일

| **항목** | **검증 내용** |
| --- | --- |
| **무한 루프 방지** | tool_calls_count > 10 시 강제 parse_diagnosis 전이. deep-research에서 11회째 정상 강제 종료 확인 |
| **수치 계산 차단** | "RMS 계산해줘" 등 요청 시 "Edge 영역" 안내 반환. 5건 전건 차단 확인 |
| **불확실성 명시** | 이력 없는 신규 설비 분석 시 uncertainty_notes에 "단일 시점 분석만으로 판단" 정상 출력 확인 |
| **LLM 실패 처리** | API 타임아웃 시 에러 메시지 반환 후 그래프 정상 종료 (graceful degradation) |

## 기타 문제 해결 사례

| **이슈** | **문제** | **해결** |
| --- | --- | --- |
| JSON 파싱 실패 | LLM 비표준 JSON 출력 (trailing comma, 코드블록 누락) | 다중 파싱: 코드블록 정규식 → raw JSON 매칭 → fallback 기본값 |
| Action Skill 초기화 | RAGServer 초기화 시점에 Qdrant 연결 실패 가능 | 지연 초기화(싱글턴) 패턴 적용. 첫 Skills 호출 시점에 서버 인스턴스 생성 |
| SSE 메모리 누수 | 클라이언트 조기 종료 시 Queue 미소비 이벤트 잔류 | `end_stream()`에서 Queue 정리 + finally 블록 보장 |
