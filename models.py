# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Public data models for the Emergency Response Allocation environment."""

from __future__ import annotations

from pydantic import BaseModel, Field

from openenv.core.env_server.types import Action, Observation, State

NUM_AMBULANCES = 5
MAX_OBSERVABLE_INCIDENTS = 10
AMBULANCE_FEATURES = 4
INCIDENT_FEATURES = 4
TRAVEL_TIME_FEATURES = NUM_AMBULANCES * MAX_OBSERVABLE_INCIDENTS
OBSERVATION_VECTOR_LENGTH = (
    NUM_AMBULANCES * AMBULANCE_FEATURES
    + MAX_OBSERVABLE_INCIDENTS * INCIDENT_FEATURES
    + TRAVEL_TIME_FEATURES
    + 1
)
HOLD_ACTION_OFFSET = NUM_AMBULANCES * MAX_OBSERVABLE_INCIDENTS
ACTION_DIM = HOLD_ACTION_OFFSET + NUM_AMBULANCES


def _zero_observation_vector() -> list[float]:
    return [0.0] * OBSERVATION_VECTOR_LENGTH


def _default_action_mask() -> list[bool]:
    return [False] * ACTION_DIM


def _default_utilization() -> list[float]:
    return [0.0] * NUM_AMBULANCES


def _default_incident_ids() -> list[int]:
    return [-1] * MAX_OBSERVABLE_INCIDENTS


def _default_travel_matrix() -> list[list[float]]:
    return [[0.0] * MAX_OBSERVABLE_INCIDENTS for _ in range(NUM_AMBULANCES)]


class ERASInfo(BaseModel):
    """Metrics and diagnostic information returned with every observation."""

    avg_response_time: float = 0.0
    incidents_served: int = 0
    incidents_total: int = 0
    missed_critical: int = 0
    ambulance_utilization: list[float] = Field(default_factory=_default_utilization)
    current_sim_time: float = 0.0
    p95_response_time: float = 0.0
    severity_weighted_response_score: float = 0.0
    coverage_rate: float = 0.0


class AmbulanceSnapshot(BaseModel):
    """Structured ambulance state for debugging and visualization."""

    ambulance_id: int
    x: int
    y: int
    status: str
    eta_free: float = 0.0
    depot_id: int


class IncidentSnapshot(BaseModel):
    """Structured incident state for debugging and visualization."""

    incident_id: int = -1
    x: int = 0
    y: int = 0
    severity: str = "none"
    severity_code: int = 0
    time_since_reported: float = 0.0


class EmergencyResponseAllocationAction(Action):
    """Discrete dispatch action encoded as a single integer index."""

    action_index: int = Field(
        ...,
        ge=0,
        lt=ACTION_DIM,
        description=(
            "Discrete action index. 0-49 encode ambulance-to-incident assignments "
            "(ambulance_id * 10 + incident_slot). 50-54 encode hold actions for "
            "ambulances 0-4."
        ),
    )


class EmergencyResponseAllocationObservation(Observation):
    """Fixed-size observation plus structured debug fields."""

    observation_vector: list[float] = Field(
        default_factory=_zero_observation_vector,
        min_length=OBSERVATION_VECTOR_LENGTH,
        max_length=OBSERVATION_VECTOR_LENGTH,
        description="Flattened fixed-size observation vector for policy networks.",
    )
    valid_action_mask: list[bool] = Field(
        default_factory=_default_action_mask,
        min_length=ACTION_DIM,
        max_length=ACTION_DIM,
        description="Boolean mask indicating which discrete actions are valid.",
    )
    ambulances: list[AmbulanceSnapshot] = Field(
        default_factory=list,
        description="Structured ambulance state for all 5 ambulances.",
    )
    incidents: list[IncidentSnapshot] = Field(
        default_factory=list,
        description=(
            "Visible pending incidents, padded to 10 slots and sorted by severity "
            "then time since reported."
        ),
    )
    travel_times: list[list[float]] = Field(
        default_factory=_default_travel_matrix,
        description="5x10 ambulance-to-incident travel-time matrix.",
    )
    visible_incident_ids: list[int] = Field(
        default_factory=_default_incident_ids,
        min_length=MAX_OBSERVABLE_INCIDENTS,
        max_length=MAX_OBSERVABLE_INCIDENTS,
        description="Incident ids aligned with the 10 visible incident slots.",
    )
    time_of_day: float = Field(
        default=0.0,
        ge=0.0,
        le=24.0,
        description="Current simulation time represented in hours [0, 24].",
    )
    event_type: str = Field(
        default="dispatch_decision",
        description="Reason the simulator paused for the current decision.",
    )
    info: ERASInfo = Field(
        default_factory=ERASInfo,
        description="Step metrics and debugging information.",
    )


class EmergencyResponseAllocationState(State):
    """Internal environment state exposed for inspection and debugging."""

    current_sim_time: float = 0.0
    time_of_day: float = 0.0
    ambulances: list[AmbulanceSnapshot] = Field(default_factory=list)
    incidents: list[IncidentSnapshot] = Field(default_factory=list)
    valid_action_mask: list[bool] = Field(
        default_factory=_default_action_mask,
        min_length=ACTION_DIM,
        max_length=ACTION_DIM,
    )
    observation_vector: list[float] = Field(
        default_factory=_zero_observation_vector,
        min_length=OBSERVATION_VECTOR_LENGTH,
        max_length=OBSERVATION_VECTOR_LENGTH,
    )
    info: ERASInfo = Field(default_factory=ERASInfo)
    event_queue_size: int = 0
    pending_incident_count: int = 0
    inflight_incident_count: int = 0
    episode_done: bool = False
    last_event_type: str = "reset"
