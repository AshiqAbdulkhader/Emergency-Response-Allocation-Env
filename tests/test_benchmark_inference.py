from __future__ import annotations

import sys
from pathlib import Path

import pytest

from emergency_response_allocation.server.baselines import nearest_ambulance_policy
from emergency_response_allocation.server.evaluation import (
    GradeRequest,
    grade_request_for_task,
    list_tasks,
    run_task,
)
from emergency_response_allocation.server.simulator import ERASSimulator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference import (
    fallback_action_index,
    action_string,
    end_line,
    parse_action_index,
    start_line,
    step_line,
)


def test_task_registry_and_graders_stay_in_unit_interval() -> None:
    tasks = list_tasks()
    assert len(tasks) >= 3

    for task in tasks:
        result = run_task(task, nearest_ambulance_policy)
        assert 0.0 <= result.score <= 1.0
        assert 0.0 <= result.reward <= 1.0

        request = GradeRequest(
            avg_response_time=result.info.avg_response_time,
            p95_response_time=result.info.p95_response_time,
            severity_weighted_response_score=result.info.severity_weighted_response_score,
            coverage_rate=result.info.coverage_rate,
            ambulance_utilization=result.info.ambulance_utilization,
            missed_critical=result.info.missed_critical,
            episode_reward=result.episode_reward,
            step_count=result.step_count,
        )
        graded = grade_request_for_task(task.task_id, request)
        assert graded.score == pytest.approx(result.score)
        assert graded.reward == pytest.approx(result.reward)


def test_inference_helpers_choose_valid_actions_and_logs() -> None:
    simulator = ERASSimulator()
    observation = simulator.reset(seed=13)
    fallback_index = fallback_action_index(observation)
    assert observation.valid_action_mask[fallback_index] is True
    assert action_string(fallback_index).startswith(("assign(", "hold("))

    assert parse_action_index('{"action_index": 12, "reason": "dispatch"}') == 12
    assert parse_action_index("action_index: 7") == 7

    start_record = start_line(
        task="night_shift_balance",
        env="emergency_response_allocation",
        model="Qwen/Qwen2.5-72B-Instruct",
    )
    step_record = step_line(
        step=1,
        action="assign(ambulance=0,incident_slot=0)",
        reward=-1.25,
        done=False,
        error=None,
    )
    end_record = end_line(
        success=True,
        steps=14,
        score=0.75,
        rewards=[-1.25, 0.5, 1.0],
    )

    assert start_record == (
        "[START] task=night_shift_balance "
        "env=emergency_response_allocation model=Qwen/Qwen2.5-72B-Instruct"
    )
    assert step_record == (
        "[STEP] step=1 action=assign(ambulance=0,incident_slot=0) "
        "reward=-1.25 done=false error=null"
    )
    assert end_record == (
        "[END] success=true steps=14 score=0.750 rewards=-1.25,0.50,1.00"
    )
