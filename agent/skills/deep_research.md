<!-- Skill: deep-research -->
<!-- Trigger: deep_research_activated = true -->
<!-- Description: 분석적 심층 조사 절차 -->

### Deep Research (분석적 심층 조사)

대화형 상호작용에서 사용자가 분석적 질문을 했을 때 Deep Research를 수행합니다.
이는 일반적인 Tool 호출(정보 조회)과 다른 분석적 조사입니다.
이벤트 자동 분석에서는 Deep Research를 수행하지 않습니다.

**탐색 전략:**
1. 가설 수립: 현재 관찰된 패턴에서 가능한 원인/시나리오를 도출합니다
2. 내부 RAG 탐색: search_maintenance_history, search_equipment_manual, search_analysis_history로 내부 근거를 확보합니다
3. 외부 웹 검색: 내부 지식만으로 불충분하면 search_web으로 외부 기술 문헌을 검색합니다. 결과는 "외부 참고 (검증 필요)"로 표기합니다
4. 결과 해석 → 후속 질문: 발견 내용을 해석한 후, 추가 확인이 필요하면 후속 쿼리를 생성하여 재검색합니다
5. 종합 분석: 내부 + 외부 결과를 교차 검증하여 결론을 도출합니다

내부 RAG 결과가 충분하면 외부 검색을 하지 않습니다. 불필요한 검색 반복은 하지 않습니다.
