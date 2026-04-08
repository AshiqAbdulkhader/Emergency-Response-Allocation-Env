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
    format_end_log,
    format_start_log,
    format_step_log,
    parse_action_index,
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

    assert parse_action_index('{"action_index": 12, "reason": "dispatch"}') == 12
    assert parse_action_index("action_index: 7") == 7

    task = list_tasks()[0]
    start_line = format_start_log(task, 1, 3)
    step_line = format_step_log(
        task_id=task.task_id,
        step_index=1,
        current_sim_time=12.5,
        event_type="assignment",
        action_index=fallback_index,
        step_reward=-1.25,
        done=False,
        used_fallback=True,
    )
    end_line = format_end_log(
        task_id=task.task_id,
        score=0.75,
        reward=0.8,
        episode_reward=11.25,
        step_count=14,
        status="success",
    )

    assert start_line.startswith("[START] ")
    assert "task_id=" in start_line
    assert step_line.startswith("[STEP] ")
    assert "fallback=1" in step_line
    assert end_line.startswith("[END] ")
    assert "score=0.750000" in end_line
