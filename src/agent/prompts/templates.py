"""리포트 및 작업지시서 생성 프롬프트 템플릿."""

from __future__ import annotations

REPORT_PROMPT = """아래 진단 결과와 추론 과정을 바탕으로 분석 리포트를 작성하세요.

## 진단 결과
{diagnosis_result}

## 이전 분석 이력
{memory_context}

## 리포트 작성 지침
1. 결함 유형 및 현재 단계 요약
2. 주요 근거 (특징량 변화, 주파수 분석, 추세 데이터)
3. RUL 평가 (ML 예측값 + 에이전트 판단)
4. 위험도 판정 근거
5. 정비 권고 사항
6. 불확실성 및 추가 모니터링 필요 사항

리포트는 한국어로 작성하며, 정비 담당자가 이해할 수 있는 수준으로 작성합니다."""

WORK_ORDER_PROMPT = """아래 진단 결과와 분석 리포트를 바탕으로 정비 작업지시서를 **유효한 JSON만** 출력하세요.
설명, 마크다운, 코드펜스 없이 순수 JSON 객체 하나만 반환합니다.

## 진단 결과
{diagnosis_result}

## 분석 리포트
{report}

## 설비 정보
- 설비명/설비번호: {equipment_info}
- 설비 위치: {location}

## 작업지시 메타
- 작업지시 번호: {wo_number}
- 작업 요청일: {request_date}
- 작업 유형: {work_type}

## 자재 레퍼런스 (코드 | 자재명 | 규격)
- ZA2115-SET | Bearing Set | Bearing Set
- ZA2115-INSERT | Bearing Insert | Bearing Insert
- GREASE-NLGI2 | Grease NLGI #2 | Grease NLGI #2
- BASE-BOLT-16MM | Base Bolt 16mm | Base Bolt 16mm
- ZS6-SEAL | Seal Kit ZS6 | Seal Kit ZS6
- M-SEAL-HEAVY | M Heavy Contact Lip Seal | M Heavy Contact Lip Seal
- HOUSING-ZA2115 | Housing ZA2115 | Housing ZA2115
- ML2-MICROLOCK | Microlock Kit ML2 | Microlock Kit ML2
- SC6-COLLAR | Setscrew Collar | Setscrew Collar
- VITON-SEAL | Viton Seal | Viton Seal
- SHAFT-49MM | Shaft Ø49mm | Shaft Ø49mm

## 공구/장비 레퍼런스 (코드 | 공구명 | 규격)
- TQ-SET | 토크렌치 | 5~60 N·m
- VB-ANL | 진동분석기 | ENVELOPE/속도계
- IR-CAM | 열화상카메라 | 320×240
- BH-TIH220M | 베어링 유도 가열기 | SKF TIH 220M
- LA-PRU | 레이저 샤프트 얼라인먼트 키트 | PRÜFTECHNIK ShaftAlign/RotAlign
- BT-OPT | 벨트 텐션 미터 | Optibelt TT
- IRT-5KV | 절연저항계 5 kV | Megger MIT525/S1-568
- ML2-MICROLOCK | Microlock Kit ML2 | Microlock Kit ML2

진단 결과에 따라 적절한 자재와 공구를 위 레퍼런스에서 선택하여 materials, tools 배열을 채우세요.
수량(qty)은 작업 규모에 맞게 판단하고, 단위(unit)는 "EA"를 사용합니다.

## JSON 스키마
반드시 아래 필드를 모두 포함하세요. 값을 모르면 빈 문자열 또는 빈 배열로 채우세요.

{{
  "wo_number": "{wo_number}",
  "equipment": "{equipment_info}",
  "location": "{location}",
  "work_type": "{work_type}",
  "request_date": "{request_date}",
  "scheduled_date": "작업 예정 일시 (YYYY-MM-DD HH:MM)",
  "due_date": "완료 예정 일시 (YYYY-MM-DD HH:MM)",
  "assignee": "담당자명 또는 담당팀",
  "summary": "작업 내용 요약 (결함 유형, 긴급도, 핵심 작업 서술)",
  "safety": "안전 주의사항",
  "checklist": ["체크리스트 항목1", "체크리스트 항목2", "..."],
  "materials": [
    {{"code": "자재코드", "name": "자재명", "spec": "규격", "qty": "수량", "unit": "단위", "note": "비고"}}
  ],
  "tools": [
    {{"code": "공구코드", "name": "공구명", "spec": "규격", "qty": "수량", "unit": "단위", "note": "비고"}}
  ],
  "post_checks": ["작업 후 확인사항1", "작업 후 확인사항2", "..."],
  "approver": "",
  "completion_date": "",
  "result_summary": "",
  "attachments": []
}}

모든 텍스트는 한국어로 작성합니다."""
