from __future__ import annotations

from fastapi.testclient import TestClient

from emergency_response_allocation.server.app import app
from emergency_response_allocation.server.baselines import evaluate_baselines
from emergency_response_allocation.server.evaluation import list_tasks


client = TestClient(app)


def _first_valid_action(mask: list[bool]) -> int:
    for index, is_valid in enumerate(mask):
        if is_valid:
            return index
    raise AssertionError("Expected at least one valid action")


def test_health_and_schema_endpoints() -> None:
    health_response = client.get("/health")
    assert health_response.status_code == 200
    assert health_response.json()["status"] == "healthy"

    schema_response = client.get("/schema")
    assert schema_response.status_code == 200
    schema = schema_response.json()
    assert "action_index" in schema["action"]["properties"]
    assert "observation_vector" in schema["observation"]["properties"]
    assert "current_sim_time" in schema["state"]["properties"]

    tasks_response = client.get("/tasks")
    assert tasks_response.status_code == 200
    tasks = tasks_response.json()
    assert len(tasks) >= 3
    assert tasks[0]["task_id"]


def test_http_reset_step_and_state_endpoints() -> None:
    reset_response = client.post("/reset", json={"seed": 7})
    assert reset_response.status_code == 200
    reset_payload = reset_response.json()
    reset_observation = reset_payload["observation"]
    assert len(reset_observation["observation_vector"]) == 111
    assert len(reset_observation["valid_action_mask"]) == 55
    assert any(reset_observation["valid_action_mask"])

    step_response = client.post("/step", json={"action": {"action_index": 0}})
    assert step_response.status_code == 200
    step_payload = step_response.json()
    assert isinstance(step_payload["reward"], float)
    assert len(step_payload["observation"]["observation_vector"]) == 111

    state_response = client.get("/state")
    assert state_response.status_code == 200
    state_payload = state_response.json()
    assert len(state_payload["valid_action_mask"]) == 55
    assert state_payload["current_sim_time"] >= 0.0
    assert "info" in state_payload


def test_websocket_session_flow() -> None:
    with client.websocket_connect("/ws") as websocket:
        websocket.send_json({"type": "reset", "data": {"seed": 11}})
        reset_message = websocket.receive_json()
        assert reset_message["type"] == "observation"
        reset_data = reset_message["data"]
        action_index = _first_valid_action(reset_data["observation"]["valid_action_mask"])

        websocket.send_json({"type": "step", "data": {"action_index": action_index}})
        step_message = websocket.receive_json()
        assert step_message["type"] == "observation"
        assert "reward" in step_message["data"]

        websocket.send_json({"type": "state"})
        state_message = websocket.receive_json()
        assert state_message["type"] == "state"
        assert state_message["data"]["step_count"] == 1
        assert state_message["data"]["episode_done"] is False


def test_baseline_evaluation_helper() -> None:
    results = evaluate_baselines(num_episodes=2)
    assert set(results.keys()) == {"random", "nearest", "severity_first"}
    for summary in results.values():
        assert summary.avg_response_time >= 0.0
        assert 0.0 <= summary.coverage_rate <= 1.0


def test_grade_endpoint_returns_unit_interval_scores() -> None:
    task = list_tasks()[0]
    payload = {
        "avg_response_time": 6.0,
        "p95_response_time": 10.0,
        "severity_weighted_response_score": 180.0,
        "coverage_rate": 0.9,
        "ambulance_utilization": [0.62, 0.65, 0.71, 0.69, 0.67],
        "missed_critical": 0,
        "episode_reward": 18.5,
        "step_count": 20,
    }
    response = client.post(f"/grade/{task.task_id}", json=payload)
    assert response.status_code == 200
    graded = response.json()
    assert 0.0 <= graded["score"] <= 1.0
    assert 0.0 <= graded["reward"] <= 1.0
