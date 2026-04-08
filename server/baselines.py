# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Baseline dispatch heuristics and evaluation helpers for ERAS."""

from __future__ import annotations

import random
from dataclasses import dataclass

try:
    from ..models import HOLD_ACTION_OFFSET
    from .config import SimulationConfig
    from .entities import Incident
    from .simulator import ERASSimulator
except ImportError:
    from models import HOLD_ACTION_OFFSET
    from server.config import SimulationConfig
    from server.entities import Incident
    from server.simulator import ERASSimulator


@dataclass
class EvaluationSummary:
    avg_response_time: float = 0.0
    p95_response_time: float = 0.0
    severity_weighted_response_score: float = 0.0
    coverage_rate: float = 0.0
    utilization: float = 0.0
    missed_critical_cases: float = 0.0


def random_policy(simulator: ERASSimulator, rng: random.Random | None = None) -> int:
    chooser = rng or random.Random()
    valid_actions = [
        action_index
        for action_index, is_valid in enumerate(simulator.build_action_mask())
        if is_valid
    ]
    if not valid_actions:
        raise RuntimeError("No valid actions available for random policy")
    return chooser.choice(valid_actions)


def nearest_ambulance_policy(simulator: ERASSimulator) -> int:
    visible_incidents = simulator.get_observable_incidents()
    free_ambulances = simulator.get_free_ambulances()
    if not visible_incidents or not free_ambulances:
        raise RuntimeError("No dispatch action available for nearest heuristic")

    best_choice: tuple[float, int, int, int] | None = None
    for slot_index, incident in enumerate(visible_incidents):
        for ambulance in free_ambulances:
            travel_time = simulator._travel_time(  # noqa: SLF001
                ambulance.location, incident.location, simulator.current_time
            )
            severity_rank = -simulator._severity_code(incident.severity.value)  # noqa: SLF001
            candidate = (
                travel_time,
                severity_rank,
                slot_index,
                ambulance.ambulance_id,
            )
            if best_choice is None or candidate < best_choice:
                best_choice = candidate

    assert best_choice is not None
    _travel_time, _severity_rank, incident_slot, ambulance_id = best_choice
    return simulator.encode_assign_action(ambulance_id, incident_slot)


def severity_first_policy(simulator: ERASSimulator) -> int:
    visible_incidents = simulator.get_observable_incidents()
    free_ambulances = simulator.get_free_ambulances()
    if not visible_incidents or not free_ambulances:
        raise RuntimeError("No dispatch action available for severity-first heuristic")

    incident = visible_incidents[0]
    incident_slot = 0
    nearest_ambulance = min(
        free_ambulances,
        key=lambda ambulance: simulator._travel_time(  # noqa: SLF001
            ambulance.location, incident.location, simulator.current_time
        ),
    )
    return simulator.encode_assign_action(nearest_ambulance.ambulance_id, incident_slot)


def evaluate_policy(
    policy_name: str,
    policy_fn,
    seeds: list[int],
    config: SimulationConfig | None = None,
) -> EvaluationSummary:
    summaries: list[EvaluationSummary] = []
    randomizer = random.Random(0)

    for seed in seeds:
        simulator = ERASSimulator(config=config)
        simulator.reset(seed=seed)

        while not simulator.done:
            if policy_name == "random":
                action_index = policy_fn(simulator, randomizer)
            else:
                action_index = policy_fn(simulator)
            simulator.step(action_index)

        info = simulator.state.info
        utilization = (
            sum(info.ambulance_utilization) / len(info.ambulance_utilization)
            if info.ambulance_utilization
            else 0.0
        )
        summaries.append(
            EvaluationSummary(
                avg_response_time=info.avg_response_time,
                p95_response_time=info.p95_response_time,
                severity_weighted_response_score=info.severity_weighted_response_score,
                coverage_rate=info.coverage_rate,
                utilization=utilization,
                missed_critical_cases=float(info.missed_critical),
            )
        )

    return _mean_summary(summaries)


def evaluate_baselines(
    num_episodes: int = 100, config: SimulationConfig | None = None
) -> dict[str, EvaluationSummary]:
    seeds = list(range(num_episodes))
    return {
        "random": evaluate_policy("random", random_policy, seeds, config=config),
        "nearest": evaluate_policy(
            "nearest", nearest_ambulance_policy, seeds, config=config
        ),
        "severity_first": evaluate_policy(
            "severity_first", severity_first_policy, seeds, config=config
        ),
    }


def format_comparison_table(
    baseline_results: dict[str, EvaluationSummary],
    rl_agent: EvaluationSummary | None = None,
) -> str:
    rl_summary = rl_agent or EvaluationSummary()
    rows = [
        (
            "Avg Response Time",
            baseline_results["random"].avg_response_time,
            baseline_results["nearest"].avg_response_time,
            baseline_results["severity_first"].avg_response_time,
            rl_summary.avg_response_time,
        ),
        (
            "95th Percentile",
            baseline_results["random"].p95_response_time,
            baseline_results["nearest"].p95_response_time,
            baseline_results["severity_first"].p95_response_time,
            rl_summary.p95_response_time,
        ),
        (
            "Severity-Weighted Score",
            baseline_results["random"].severity_weighted_response_score,
            baseline_results["nearest"].severity_weighted_response_score,
            baseline_results["severity_first"].severity_weighted_response_score,
            rl_summary.severity_weighted_response_score,
        ),
        (
            "Coverage Rate",
            baseline_results["random"].coverage_rate,
            baseline_results["nearest"].coverage_rate,
            baseline_results["severity_first"].coverage_rate,
            rl_summary.coverage_rate,
        ),
        (
            "Utilization",
            baseline_results["random"].utilization,
            baseline_results["nearest"].utilization,
            baseline_results["severity_first"].utilization,
            rl_summary.utilization,
        ),
        (
            "Missed Critical Cases",
            baseline_results["random"].missed_critical_cases,
            baseline_results["nearest"].missed_critical_cases,
            baseline_results["severity_first"].missed_critical_cases,
            rl_summary.missed_critical_cases,
        ),
    ]

    lines = [
        "| Metric | Random | Nearest | Severity-First | RL Agent |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric, random_value, nearest_value, severity_value, rl_value in rows:
        lines.append(
            f"| {metric} | {random_value:.3f} | {nearest_value:.3f} | "
            f"{severity_value:.3f} | {rl_value:.3f} |"
        )
    return "\n".join(lines)


def _mean_summary(summaries: list[EvaluationSummary]) -> EvaluationSummary:
    if not summaries:
        return EvaluationSummary()
    count = float(len(summaries))
    return EvaluationSummary(
        avg_response_time=sum(item.avg_response_time for item in summaries) / count,
        p95_response_time=sum(item.p95_response_time for item in summaries) / count,
        severity_weighted_response_score=(
            sum(item.severity_weighted_response_score for item in summaries) / count
        ),
        coverage_rate=sum(item.coverage_rate for item in summaries) / count,
        utilization=sum(item.utilization for item in summaries) / count,
        missed_critical_cases=(
            sum(item.missed_critical_cases for item in summaries) / count
        ),
    )
