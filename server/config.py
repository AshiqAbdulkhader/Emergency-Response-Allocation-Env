# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Configuration for the Emergency Response Allocation simulator."""

from __future__ import annotations

from dataclasses import dataclass, field

try:
    from ..models import MAX_OBSERVABLE_INCIDENTS, NUM_AMBULANCES
except ImportError:
    from models import MAX_OBSERVABLE_INCIDENTS, NUM_AMBULANCES


GridPoint = tuple[int, int]


@dataclass(frozen=True)
class RewardConfig:
    """Reward shaping parameters."""

    response_time_penalty_weight: float = 1.0
    waiting_time_penalty_weight: float = 0.8
    idle_penalty_weight: float = 0.5
    successful_rescue_weight: float = 1.2
    catastrophic_failure_penalty: float = -100.0


@dataclass(frozen=True)
class SimulationConfig:
    """Static layout and stochastic process settings."""

    grid_size: int = 20
    num_ambulances: int = NUM_AMBULANCES
    max_observable_incidents: int = MAX_OBSERVABLE_INCIDENTS
    depot_locations: tuple[GridPoint, ...] = ((2, 2), (10, 10), (17, 17))
    hospital_locations: tuple[GridPoint, ...] = ((4, 4), (15, 15))
    ambulance_depot_ids: tuple[int, ...] = (0, 0, 1, 2, 2)
    episode_duration_minutes: float = 24.0 * 60.0
    quiet_window_minutes: float = 30.0
    critical_response_threshold_minutes: float = 8.0
    catastrophic_overdue_critical_limit: int = 3
    arrival_phase_boundaries_minutes: tuple[float, ...] = (
        0.0,
        6.0 * 60.0,
        10.0 * 60.0,
        16.0 * 60.0,
        20.0 * 60.0,
        24.0 * 60.0,
    )
    arrival_rates_per_hour: tuple[float, ...] = (0.4, 3.0, 1.0, 3.0, 0.4)
    commercial_bounds: tuple[int, int, int, int] = (6, 6, 13, 13)
    traffic_peak_windows_hours: tuple[tuple[float, float], ...] = (
        (8.0, 10.0),
        (17.0, 19.0),
    )
    traffic_night_windows_hours: tuple[tuple[float, float], ...] = (
        (0.0, 6.0),
        (22.0, 24.0),
    )
    traffic_peak_multiplier: float = 1.8
    traffic_offpeak_multiplier: float = 1.0
    traffic_night_multiplier: float = 0.7
    severity_probabilities: tuple[tuple[str, float], ...] = (
        ("critical", 0.2),
        ("moderate", 0.5),
        ("low", 0.3),
    )
    severity_weights: dict[str, float] = field(
        default_factory=lambda: {"critical": 3.0, "moderate": 1.5, "low": 0.5}
    )
    scene_time_ranges_minutes: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "critical": (15.0, 25.0),
            "moderate": (8.0, 15.0),
            "low": (5.0, 10.0),
        }
    )
    reward: RewardConfig = field(default_factory=RewardConfig)
