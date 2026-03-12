"""Agent Skills 레지스트리 테스트."""

from agent.skills.registry import load_matching_skills


def test_load_skills_normal():
    """정상 상태에서는 결함 진단 Skill이 로드되지 않고 응답 양식만 로드된다."""
    state = {
        "event_payload": {
            "anomaly_detection_result": {
                "anomaly_detected": False,
                "health_state": "normal",
            }
        }
    }
    result = load_matching_skills(state)
    assert "fault-diagnosis" not in result
    assert "response-normal" in result


def test_load_skills_anomaly():
    """이상 감지 시 결함 진단 및 특징량 해석 Skill이 로드된다."""
    state = {
        "event_payload": {
            "anomaly_detection_result": {
                "anomaly_detected": True,
                "health_state": "warning",
            }
        }
    }
    result = load_matching_skills(state)
    assert "fault-diagnosis" in result
    assert "feature-interpret" in result
    assert "response-alert" in result
    assert "response-normal" not in result


def test_load_skills_deep_research():
    """Deep Research 활성화 시 해당 Skill이 로드된다."""
    state = {
        "event_payload": {
            "anomaly_detection_result": {
                "anomaly_detected": True,
                "health_state": "critical",
            }
        },
        "deep_research_activated": True,
    }
    result = load_matching_skills(state)
    assert "deep-research" in result
    assert "fault-diagnosis" in result
    assert "response-alert" in result


def test_load_skills_empty_state():
    """빈 state에서도 response-normal이 로드된다."""
    state: dict = {}
    result = load_matching_skills(state)
    assert "response-normal" in result


def test_skills_priority_order():
    """Skills가 priority 순으로 정렬되어 로드된다."""
    state = {
        "event_payload": {
            "anomaly_detection_result": {
                "anomaly_detected": True,
                "health_state": "warning",
            }
        }
    }
    result = load_matching_skills(state)
    fd_pos = result.index("fault-diagnosis")
    fi_pos = result.index("feature-interpret")
    ra_pos = result.index("response-alert")
    assert fd_pos < fi_pos < ra_pos
