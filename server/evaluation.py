# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Benchmark tasks and normalized graders for ERAS."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace
from typing import Callable

from pydantic import BaseModel, Field

try:
    from ..models import ERASInfo
    from .config import SimulationConfig
    from .simulator import ERASSimulator
except ImportError:
    from models import ERASInfo
    from server.config import SimulationConfig
    from server.simulator import ERASSimulator


PolicyFn = Callable[[ERASSimulator], int]


@dataclass(frozen=True)
class ERASTask:
    """Single benchmark task definition."""

    task_id: str
    title: str
    difficulty: str
    description: str
    seed: int
    grader_name: str
    max_decision_steps: int = 96
    config_overrides: dict[str, object] = field(default_factory=dict)

    def build_config(self) -> SimulationConfig:
        return replace(SimulationConfig(), **self.config_overrides)


class TaskSpec(BaseModel):
    """Serializable task specification for APIs and tooling."""

    task_id: str
    title: str
    difficulty: str
    description: str
    seed: int
    grader_name: str
    max_decision_steps: int


class GradeRequest(BaseModel):
    """Payload for grading externally-run task episodes."""

    avg_response_time: float = 0.0
    p95_response_time: float = 0.0
    severity_weighted_response_score: float = 0.0
    coverage_rate: float = 0.0
    ambulance_utilization: list[float] = Field(default_factory=list)
    missed_critical: int = 0
    episode_reward: float = 0.0
    step_count: int = 0


class GradeResult(BaseModel):
    """Normalized task grading output."""

    task_id: str
    grader_name: str
    score: float
    reward: float
    metrics: dict[str, float]


@dataclass(frozen=True)
class TaskRunResult:
    """Result of executing a full task episode with a policy."""

    task: ERASTask
    score: float
    reward: float
    episode_reward: float
    step_count: int
    done: bool
    info: ERASInfo


BENCHMARK_TASKS: tuple[ERASTask, ...] = (
    ERASTask(
        task_id="night_shift_balance",
        title="Night Shift Balance",
        difficulty="easy",
        description=(
            "Lower-demand night dispatch. Reward stable coverage while avoiding "
            "unnecessary idle ambulances."
        ),
        seed=7,
        grader_name="coverage_balance",
    ),
    ERASTask(
        task_id="rush_hour_triage",
        title="Rush Hour Triage",
        difficulty="medium",
        description=(
            "Peak-hour traffic and heavier arrival pressure. Reward fast "
            "severity-aware response under congestion."
        ),
        seed=23,
        grader_name="critical_response",
        config_overrides={
            "arrival_rates_per_hour": (0.4, 3.4, 1.2, 3.4, 0.4),
            "traffic_peak_multiplier": 2.0,
        },
    ),
    ERASTask(
        task_id="citywide_surge",
        title="Citywide Surge",
        difficulty="hard",
        description=(
            "High-demand surge with tighter critical thresholds. Reward coverage "
            "and resilience when the city is saturated."
        ),
        seed=89,
        grader_name="surge_resilience",
        config_overrides={
            "arrival_rates_per_hour": (0.6, 3.8, 1.5, 3.8, 0.8),
            "traffic_peak_multiplier": 2.1,
            "critical_response_threshold_minutes": 7.0,
            "catastrophic_overdue_critical_limit": 2,
        },
    ),
)


def list_tasks() -> list[ERASTask]:
    """Return the benchmark task registry."""

    return list(BENCHMARK_TASKS)


def get_task(task_id: str) -> ERASTask:
    """Return a task by id."""

    for task in BENCHMARK_TASKS:
        if task.task_id == task_id:
            return task
    raise KeyError(f"Unknown task id: {task_id}")


def list_task_specs() -> list[TaskSpec]:
    """Return benchmark tasks in API-friendly form."""

    return [
        TaskSpec(
            task_id=task.task_id,
            title=task.title,
            difficulty=task.difficulty,
            description=task.description,
            seed=task.seed,
            grader_name=task.grader_name,
            max_decision_steps=task.max_decision_steps,
        )
        for task in BENCHMARK_TASKS
    ]


def grade_request_for_task(task_id: str, request: GradeRequest) -> GradeResult:
    """Grade externally provided metrics for a specific task."""

    info = ERASInfo(
        avg_response_time=request.avg_response_time,
        p95_response_time=request.p95_response_time,
        severity_weighted_response_score=request.severity_weighted_response_score,
        coverage_rate=request.coverage_rate,
        ambulance_utilization=list(request.ambulance_utilization),
        missed_critical=request.missed_critical,
    )
    return grade_task_result(
        get_task(task_id),
        info=info,
        episode_reward=request.episode_reward,
        step_count=request.step_count,
    )


def grade_task_result(
    task: ERASTask,
    *,
    info: ERASInfo,
    episode_reward: float,
    step_count: int,
) -> GradeResult:
    """Return bounded grader outputs for a completed task episode."""

    mean_utilization = (
        sum(info.ambulance_utilization) / len(info.ambulance_utilization)
        if info.ambulance_utilization
        else 0.0
    )
    metrics = {
        "coverage": _clip01(info.coverage_rate),
        "avg_response": _inverse_time_score(info.avg_response_time, target=8.0),
        "p95_response": _inverse_time_score(info.p95_response_time, target=12.0),
        "severity_weighted": _inverse_time_score(
            info.severity_weighted_response_score, target=250.0
        ),
        "critical_failures": 1.0 / (1.0 + max(0.0, float(info.missed_critical))),
        "utilization": _utilization_score(mean_utilization),
        "steps": _inverse_time_score(float(step_count), target=float(task.max_decision_steps)),
        "episode_reward": _sigmoid_scale(episode_reward, scale=80.0),
    }
    score = _task_score(task.grader_name, metrics)
    reward = _clip01((0.7 * score) + (0.3 * metrics["episode_reward"]))
    return GradeResult(
        task_id=task.task_id,
        grader_name=task.grader_name,
        score=score,
        reward=reward,
        metrics=metrics,
    )


def run_task(task: ERASTask, policy_fn: PolicyFn) -> TaskRunResult:
    """Execute a benchmark task episode and return its graded result."""

    simulator = ERASSimulator(config=task.build_config())
    observation = simulator.reset(seed=task.seed, episode_id=task.task_id)
    episode_reward = float(observation.reward or 0.0)

    while not simulator.done and simulator.step_count < task.max_decision_steps:
        action_index = policy_fn(simulator)
        observation = simulator.step(action_index)
        episode_reward += float(observation.reward or 0.0)

    grade = grade_task_result(
        task,
        info=simulator.state.info,
        episode_reward=episode_reward,
        step_count=simulator.step_count,
    )
    return TaskRunResult(
        task=task,
        score=grade.score,
        reward=grade.reward,
        episode_reward=episode_reward,
        step_count=simulator.step_count,
        done=simulator.done,
        info=simulator.state.info,
    )


def _task_score(grader_name: str, metrics: dict[str, float]) -> float:
    if grader_name == "coverage_balance":
        return _weighted_average(
            metrics,
            coverage=0.40,
            utilization=0.20,
            avg_response=0.20,
            critical_failures=0.15,
            steps=0.05,
        )
    if grader_name == "critical_response":
        return _weighted_average(
            metrics,
            avg_response=0.30,
            p95_response=0.20,
            critical_failures=0.25,
            coverage=0.15,
            severity_weighted=0.10,
        )
    if grader_name == "surge_resilience":
        return _weighted_average(
            metrics,
            severity_weighted=0.30,
            coverage=0.25,
            critical_failures=0.20,
            utilization=0.15,
            p95_response=0.10,
        )
    raise KeyError(f"Unknown grader name: {grader_name}")


def _weighted_average(metrics: dict[str, float], **weights: float) -> float:
    total = 0.0
    for key, weight in weights.items():
        total += metrics[key] * weight
    return _clip01(total)


def _utilization_score(value: float) -> float:
    target = 0.70
    tolerance = 0.30
    distance = abs(value - target)
    return _clip01(1.0 - (distance / tolerance))


def _inverse_time_score(value: float, *, target: float) -> float:
    positive_value = max(0.0, value)
    return _clip01(1.0 / (1.0 + (positive_value / max(target, 1e-6))))


def _sigmoid_scale(value: float, *, scale: float) -> float:
    scaled = max(-30.0, min(30.0, value / max(scale, 1e-6)))
    return 1.0 / (1.0 + math.exp(-scaled))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def task_to_dict(task: ERASTask) -> dict[str, object]:
    """Return a JSON-serializable task dictionary."""

    payload = asdict(task)
    payload.pop("config_overrides", None)
    return payload
