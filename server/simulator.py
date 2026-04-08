# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Event-driven simulator for the Emergency Response Allocation environment."""

from __future__ import annotations

import heapq
import math
import random
from uuid import uuid4

try:
    from ..models import (
        ACTION_DIM,
        HOLD_ACTION_OFFSET,
        MAX_OBSERVABLE_INCIDENTS,
        NUM_AMBULANCES,
        AmbulanceSnapshot,
        EmergencyResponseAllocationObservation,
        EmergencyResponseAllocationState,
        ERASInfo,
        IncidentSnapshot,
    )
    from .config import GridPoint, SimulationConfig
    from .entities import (
        Ambulance,
        AmbulanceStatus,
        Event,
        EventType,
        EVENT_PRIORITIES,
        Incident,
        IncidentStatus,
        SeverityLevel,
        ZoneType,
    )
except ImportError:
    from models import (
        ACTION_DIM,
        HOLD_ACTION_OFFSET,
        MAX_OBSERVABLE_INCIDENTS,
        NUM_AMBULANCES,
        AmbulanceSnapshot,
        EmergencyResponseAllocationObservation,
        EmergencyResponseAllocationState,
        ERASInfo,
        IncidentSnapshot,
    )
    from server.config import GridPoint, SimulationConfig
    from server.entities import (
        Ambulance,
        AmbulanceStatus,
        Event,
        EventType,
        EVENT_PRIORITIES,
        Incident,
        IncidentStatus,
        SeverityLevel,
        ZoneType,
    )


class ERASSimulator:
    """Core simulator that powers the OpenEnv-facing environment wrapper."""

    def __init__(
        self, config: SimulationConfig | None = None, auto_reset: bool = True
    ) -> None:
        self.config = config or SimulationConfig()
        self._rng = random.Random()
        self._event_sequence = 0
        self._all_dispatchable_locations = self._build_dispatchable_locations()
        self._next_incident_time: float | None = None
        self._reset_internal_state()
        if auto_reset:
            self.reset()

    def reset(
        self, seed: int | None = None, episode_id: str | None = None
    ) -> EmergencyResponseAllocationObservation:
        if seed is not None:
            self._rng = random.Random(seed)
        self._reset_internal_state()
        self.episode_id = episode_id or str(uuid4())
        self.last_event_type = "reset"
        self._initialize_ambulances()
        self._schedule_next_incident(self.current_time)
        self._advance_until_actionable_event()
        return self._build_observation(reward=None)

    def step(self, action_index: int) -> EmergencyResponseAllocationObservation:
        if self.done:
            return self._build_observation(reward=0.0)

        mask = self.build_action_mask()
        if action_index < 0 or action_index >= ACTION_DIM:
            raise ValueError(f"Action index out of range: {action_index}")
        if not mask[action_index]:
            raise ValueError(f"Invalid action for current state: {action_index}")

        self.step_count += 1
        reward = 0.0

        if action_index < HOLD_ACTION_OFFSET:
            ambulance_id, incident_slot = divmod(
                action_index, self.config.max_observable_incidents
            )
            incident = self.get_observable_incidents()[incident_slot]
            reward += self._dispatch_incident(ambulance_id, incident.incident_id)
            self.last_event_type = "assignment"
            if self._has_actionable_assignment():
                return self._build_observation(reward=reward)
        else:
            self.last_event_type = "hold"
            reward += self._idle_penalty()

        reward += self._advance_until_actionable_event()
        return self._build_observation(reward=reward)

    @property
    def state(self) -> EmergencyResponseAllocationState:
        info = self._build_info()
        return EmergencyResponseAllocationState(
            episode_id=self.episode_id,
            step_count=self.step_count,
            current_sim_time=self.current_time,
            time_of_day=self.time_of_day,
            ambulances=self._build_ambulance_snapshots(),
            incidents=self._build_incident_snapshots(),
            valid_action_mask=self.build_action_mask(),
            observation_vector=self.build_observation_vector(),
            info=info,
            event_queue_size=len(self._event_queue),
            pending_incident_count=len(self.get_pending_incidents()),
            inflight_incident_count=self._count_inflight_incidents(),
            episode_done=self.done,
            last_event_type=self.last_event_type,
        )

    @property
    def time_of_day(self) -> float:
        return (self.current_time / 60.0) % 24.0

    def get_observable_incidents(self) -> list[Incident]:
        pending = self.get_pending_incidents()
        pending.sort(
            key=lambda incident: (
                -self.config.severity_weights[incident.severity.value],
                -(self.current_time - incident.reported_at),
                incident.incident_id,
            )
        )
        return pending[: self.config.max_observable_incidents]

    def get_pending_incidents(self) -> list[Incident]:
        return [
            incident
            for incident in self._incidents.values()
            if incident.status == IncidentStatus.PENDING
        ]

    def get_free_ambulances(self) -> list[Ambulance]:
        return [
            ambulance
            for ambulance in self._ambulances
            if ambulance.status == AmbulanceStatus.FREE
        ]

    def build_action_mask(self) -> list[bool]:
        mask = [False] * ACTION_DIM
        visible_incidents = self.get_observable_incidents()
        if not visible_incidents:
            return mask

        for ambulance in self._ambulances:
            if ambulance.status != AmbulanceStatus.FREE:
                continue
            for slot_index, _incident in enumerate(visible_incidents):
                action_index = (
                    ambulance.ambulance_id * self.config.max_observable_incidents
                    + slot_index
                )
                mask[action_index] = True
            mask[HOLD_ACTION_OFFSET + ambulance.ambulance_id] = True
        return mask

    def build_observation_vector(self) -> list[float]:
        vector: list[float] = []
        visible_incidents = self.get_observable_incidents()

        for ambulance in self._ambulances:
            vector.extend(
                [
                    float(ambulance.location[0]),
                    float(ambulance.location[1]),
                    1.0 if ambulance.status == AmbulanceStatus.FREE else 0.0,
                    max(0.0, ambulance.eta_free_at - self.current_time),
                ]
            )

        for slot_index in range(self.config.max_observable_incidents):
            if slot_index < len(visible_incidents):
                incident = visible_incidents[slot_index]
                vector.extend(
                    [
                        float(incident.location[0]),
                        float(incident.location[1]),
                        float(self._severity_code(incident.severity.value)),
                        max(0.0, self.current_time - incident.reported_at),
                    ]
                )
            else:
                vector.extend([0.0, 0.0, 0.0, 0.0])

        travel_times = self._build_travel_time_matrix(visible_incidents)
        for row in travel_times:
            vector.extend(row)

        vector.append(self.time_of_day)
        return vector

    def encode_assign_action(self, ambulance_id: int, incident_slot: int) -> int:
        return ambulance_id * self.config.max_observable_incidents + incident_slot

    def encode_hold_action(self, ambulance_id: int) -> int:
        return HOLD_ACTION_OFFSET + ambulance_id

    def _reset_internal_state(self) -> None:
        self.current_time = 0.0
        self.done = False
        self.step_count = 0
        self.episode_id = str(uuid4())
        self.last_event_type = "reset"
        self._event_queue: list[Event] = []
        self._ambulances: list[Ambulance] = []
        self._incidents: dict[int, Incident] = {}
        self._incident_counter = 0
        self._response_times: list[float] = []
        self._severity_weighted_response_score = 0.0
        self._incidents_served = 0
        self._missed_critical = 0
        self._next_incident_time = None

    def _initialize_ambulances(self) -> None:
        self._ambulances = []
        for ambulance_id, depot_id in enumerate(self.config.ambulance_depot_ids):
            home_location = self.config.depot_locations[depot_id]
            self._ambulances.append(
                Ambulance(
                    ambulance_id=ambulance_id,
                    depot_id=depot_id,
                    home_location=home_location,
                    location=home_location,
                    status=AmbulanceStatus.FREE,
                    eta_free_at=0.0,
                )
            )

    def _build_dispatchable_locations(self) -> list[GridPoint]:
        locations: list[GridPoint] = []
        for x in range(self.config.grid_size):
            for y in range(self.config.grid_size):
                zone = self._zone_type((x, y))
                if zone not in (ZoneType.HOSPITAL, ZoneType.DEPOT):
                    locations.append((x, y))
        return locations

    def _zone_type(self, location: GridPoint) -> ZoneType:
        if location in self.config.depot_locations:
            return ZoneType.DEPOT
        if location in self.config.hospital_locations:
            return ZoneType.HOSPITAL

        min_x, min_y, max_x, max_y = self.config.commercial_bounds
        x, y = location
        if min_x <= x <= max_x and min_y <= y <= max_y:
            return ZoneType.COMMERCIAL
        return ZoneType.RESIDENTIAL

    def _build_ambulance_snapshots(self) -> list[AmbulanceSnapshot]:
        snapshots: list[AmbulanceSnapshot] = []
        for ambulance in self._ambulances:
            snapshots.append(
                AmbulanceSnapshot(
                    ambulance_id=ambulance.ambulance_id,
                    x=ambulance.location[0],
                    y=ambulance.location[1],
                    status=ambulance.status.value,
                    eta_free=max(0.0, ambulance.eta_free_at - self.current_time),
                    depot_id=ambulance.depot_id,
                )
            )
        return snapshots

    def _build_incident_snapshots(self) -> list[IncidentSnapshot]:
        snapshots: list[IncidentSnapshot] = []
        visible_incidents = self.get_observable_incidents()
        for slot_index in range(self.config.max_observable_incidents):
            if slot_index < len(visible_incidents):
                incident = visible_incidents[slot_index]
                snapshots.append(
                    IncidentSnapshot(
                        incident_id=incident.incident_id,
                        x=incident.location[0],
                        y=incident.location[1],
                        severity=incident.severity.value,
                        severity_code=self._severity_code(incident.severity.value),
                        time_since_reported=max(
                            0.0, self.current_time - incident.reported_at
                        ),
                    )
                )
            else:
                snapshots.append(IncidentSnapshot())
        return snapshots

    def _build_travel_time_matrix(
        self, visible_incidents: list[Incident] | None = None
    ) -> list[list[float]]:
        incidents = visible_incidents if visible_incidents is not None else self.get_observable_incidents()
        matrix: list[list[float]] = []
        for ambulance in self._ambulances:
            row = [0.0] * self.config.max_observable_incidents
            for slot_index, incident in enumerate(incidents):
                if ambulance.status == AmbulanceStatus.FREE:
                    row[slot_index] = self._travel_time(
                        ambulance.location, incident.location, self.current_time
                    )
            matrix.append(row)
        return matrix

    def _build_info(self) -> ERASInfo:
        avg_response_time = (
            sum(self._response_times) / len(self._response_times)
            if self._response_times
            else 0.0
        )
        p95_response_time = self._percentile(self._response_times, 95.0)
        incidents_total = len(self._incidents)
        coverage_rate = (
            self._incidents_served / incidents_total if incidents_total else 0.0
        )
        elapsed = max(self.current_time, 1e-6)
        utilization = [
            min(1.0, self._current_busy_time(ambulance) / elapsed)
            for ambulance in self._ambulances
        ]
        return ERASInfo(
            avg_response_time=avg_response_time,
            incidents_served=self._incidents_served,
            incidents_total=incidents_total,
            missed_critical=self._missed_critical,
            ambulance_utilization=utilization,
            current_sim_time=self.current_time,
            p95_response_time=p95_response_time,
            severity_weighted_response_score=self._severity_weighted_response_score,
            coverage_rate=coverage_rate,
        )

    def _build_observation(
        self, reward: float | None
    ) -> EmergencyResponseAllocationObservation:
        visible_incidents = self.get_observable_incidents()
        visible_ids = [-1] * self.config.max_observable_incidents
        for slot_index, incident in enumerate(visible_incidents):
            visible_ids[slot_index] = incident.incident_id

        return EmergencyResponseAllocationObservation(
            done=self.done,
            reward=reward,
            observation_vector=self.build_observation_vector(),
            valid_action_mask=self.build_action_mask(),
            ambulances=self._build_ambulance_snapshots(),
            incidents=self._build_incident_snapshots(),
            travel_times=self._build_travel_time_matrix(visible_incidents),
            visible_incident_ids=visible_ids,
            time_of_day=self.time_of_day,
            event_type=self.last_event_type,
            info=self._build_info(),
        )

    def _push_event(self, scheduled_time: float, event_type: EventType, **payload: object) -> None:
        if scheduled_time > self.config.episode_duration_minutes:
            return
        event = Event(
            scheduled_time=scheduled_time,
            priority=EVENT_PRIORITIES[event_type],
            sequence=self._event_sequence,
            event_type=event_type,
            payload=dict(payload),
        )
        self._event_sequence += 1
        heapq.heappush(self._event_queue, event)

    def _schedule_next_incident(self, from_time: float) -> None:
        next_time = self._sample_next_incident_time(from_time)
        self._next_incident_time = next_time
        if next_time is not None:
            self._push_event(next_time, EventType.INCIDENT_ARRIVAL)

    def _sample_next_incident_time(self, from_time: float) -> float | None:
        time_cursor = from_time
        boundaries = self.config.arrival_phase_boundaries_minutes
        rates = self.config.arrival_rates_per_hour

        while time_cursor < self.config.episode_duration_minutes:
            phase_index = self._phase_index(time_cursor)
            phase_end = boundaries[phase_index + 1]
            rate_per_hour = rates[phase_index]
            if rate_per_hour <= 0.0:
                time_cursor = phase_end
                continue

            delta_minutes = self._rng.expovariate(rate_per_hour) * 60.0
            candidate = time_cursor + delta_minutes
            if candidate < phase_end:
                return candidate
            time_cursor = phase_end
        return None

    def _phase_index(self, sim_minutes: float) -> int:
        boundaries = self.config.arrival_phase_boundaries_minutes
        for index in range(len(boundaries) - 1):
            if boundaries[index] <= sim_minutes < boundaries[index + 1]:
                return index
        return len(boundaries) - 2

    def _sample_incident_location(self) -> GridPoint:
        weights: list[float] = []
        for location in self._all_dispatchable_locations:
            zone = self._zone_type(location)
            weight = 1.0
            if zone == ZoneType.COMMERCIAL and self._is_peak_demand_period():
                weight = 3.0
            elif zone == ZoneType.RESIDENTIAL and self._is_night_demand_period():
                weight = 2.0
            weights.append(weight)
        return self._rng.choices(self._all_dispatchable_locations, weights=weights, k=1)[0]

    def _sample_severity(self) -> SeverityLevel:
        threshold = self._rng.random()
        cumulative = 0.0
        for severity_name, probability in self.config.severity_probabilities:
            cumulative += probability
            if threshold <= cumulative:
                return SeverityLevel(severity_name)
        return SeverityLevel.LOW

    def _sample_service_time(self, severity: SeverityLevel) -> float:
        low, high = self.config.scene_time_ranges_minutes[severity.value]
        return self._rng.uniform(low, high)

    def _travel_time(
        self, origin: GridPoint, destination: GridPoint, sim_time_minutes: float
    ) -> float:
        distance = math.dist(origin, destination)
        return distance * self._traffic_multiplier(sim_time_minutes)

    def _traffic_multiplier(self, sim_time_minutes: float) -> float:
        hour = (sim_time_minutes / 60.0) % 24.0
        for start, end in self.config.traffic_peak_windows_hours:
            if start <= hour < end:
                return self.config.traffic_peak_multiplier
        for start, end in self.config.traffic_night_windows_hours:
            if start <= hour < end:
                return self.config.traffic_night_multiplier
        return self.config.traffic_offpeak_multiplier

    def _is_peak_demand_period(self) -> bool:
        hour = self.time_of_day
        return (6.0 <= hour < 10.0) or (16.0 <= hour < 20.0)

    def _is_night_demand_period(self) -> bool:
        hour = self.time_of_day
        return hour < 6.0 or hour >= 20.0

    def _dispatch_incident(self, ambulance_id: int, incident_id: int) -> float:
        ambulance = self._ambulances[ambulance_id]
        incident = self._incidents[incident_id]
        if ambulance.status != AmbulanceStatus.FREE:
            raise ValueError(f"Ambulance {ambulance_id} is not free")
        if incident.status != IncidentStatus.PENDING:
            raise ValueError(f"Incident {incident_id} is not pending")

        travel_to_scene = self._travel_time(
            ambulance.location, incident.location, self.current_time
        )
        hospital = self._nearest_hospital(incident.location)
        travel_to_hospital = self._travel_time(
            incident.location, hospital, self.current_time + travel_to_scene
        )
        travel_to_depot = self._travel_time(
            hospital,
            ambulance.home_location,
            self.current_time + travel_to_scene + incident.service_time + travel_to_hospital,
        )

        ambulance.status = AmbulanceStatus.BUSY
        ambulance.assigned_incident_id = incident_id
        ambulance.busy_start = self.current_time
        ambulance.eta_free_at = (
            self.current_time
            + travel_to_scene
            + incident.service_time
            + travel_to_hospital
            + travel_to_depot
        )

        incident.status = IncidentStatus.DISPATCHED
        incident.assigned_ambulance_id = ambulance_id
        incident.dispatch_at = self.current_time
        incident.estimated_travel_time = travel_to_scene
        incident.hospital_location = hospital

        self._push_event(
            self.current_time + travel_to_scene,
            EventType.ARRIVE_SCENE,
            ambulance_id=ambulance_id,
            incident_id=incident_id,
        )

        severity_weight = self.config.severity_weights[incident.severity.value]
        reward_cfg = self.config.reward
        return -reward_cfg.response_time_penalty_weight * severity_weight * travel_to_scene

    def _idle_penalty(self) -> float:
        pending_incidents = self.get_pending_incidents()
        if not pending_incidents:
            return 0.0
        idle_free_ambulances = len(self.get_free_ambulances())
        return -self.config.reward.idle_penalty_weight * idle_free_ambulances

    def _advance_until_actionable_event(self) -> float:
        reward = 0.0

        while not self.done:
            if self._should_end_naturally():
                self.done = True
                self.last_event_type = "natural_end"
                break

            if not self._event_queue:
                self.done = True
                self.last_event_type = "queue_exhausted"
                break

            next_event_time = self._event_queue[0].scheduled_time
            if next_event_time >= self.config.episode_duration_minutes:
                self.current_time = self.config.episode_duration_minutes
                self.done = True
                self.last_event_type = "hard_cutoff"
                break

            self.current_time = next_event_time
            while (
                self._event_queue
                and abs(self._event_queue[0].scheduled_time - next_event_time) < 1e-9
            ):
                event = heapq.heappop(self._event_queue)
                reward += self._process_event(event)

            if self._catastrophic_failure():
                reward += self.config.reward.catastrophic_failure_penalty
                self.done = True
                self.last_event_type = "catastrophic_failure"
                break

            if self._has_actionable_assignment():
                self.last_event_type = "dispatch_decision"
                break

        return reward

    def _process_event(self, event: Event) -> float:
        if event.event_type == EventType.INCIDENT_ARRIVAL:
            self._next_incident_time = None
            self._create_incident()
            self._schedule_next_incident(self.current_time)
            return 0.0

        if event.event_type == EventType.ARRIVE_SCENE:
            incident_id = int(event.payload["incident_id"])
            ambulance_id = int(event.payload["ambulance_id"])
            incident = self._incidents[incident_id]
            ambulance = self._ambulances[ambulance_id]
            incident.status = IncidentStatus.ON_SCENE
            incident.arrival_at_scene = self.current_time
            ambulance.location = incident.location
            response_time = self.current_time - incident.reported_at
            self._response_times.append(response_time)
            self._severity_weighted_response_score += (
                self.config.severity_weights[incident.severity.value] * response_time
            )
            self._incidents_served += 1
            if (
                incident.severity == SeverityLevel.CRITICAL
                and response_time > self.config.critical_response_threshold_minutes
            ):
                self._missed_critical += 1

            self._push_event(
                self.current_time + incident.service_time,
                EventType.SERVICE_COMPLETE,
                incident_id=incident_id,
                ambulance_id=ambulance_id,
            )
            return self._arrival_reward(incident, response_time)

        if event.event_type == EventType.SERVICE_COMPLETE:
            incident_id = int(event.payload["incident_id"])
            ambulance_id = int(event.payload["ambulance_id"])
            incident = self._incidents[incident_id]
            ambulance = self._ambulances[ambulance_id]
            incident.status = IncidentStatus.TRANSPORTING
            hospital_location = incident.hospital_location or self._nearest_hospital(
                incident.location
            )
            incident.hospital_location = hospital_location
            travel_to_hospital = self._travel_time(
                incident.location, hospital_location, self.current_time
            )
            ambulance.location = incident.location
            self._push_event(
                self.current_time + travel_to_hospital,
                EventType.ARRIVE_HOSPITAL,
                incident_id=incident_id,
                ambulance_id=ambulance_id,
            )
            return 0.0

        if event.event_type == EventType.ARRIVE_HOSPITAL:
            incident_id = int(event.payload["incident_id"])
            ambulance_id = int(event.payload["ambulance_id"])
            incident = self._incidents[incident_id]
            ambulance = self._ambulances[ambulance_id]
            hospital_location = incident.hospital_location or self._nearest_hospital(
                incident.location
            )
            ambulance.location = hospital_location
            incident.status = IncidentStatus.RESOLVED
            incident.resolved_at = self.current_time
            travel_to_depot = self._travel_time(
                hospital_location, ambulance.home_location, self.current_time
            )
            self._push_event(
                self.current_time + travel_to_depot,
                EventType.AMBULANCE_FREE,
                ambulance_id=ambulance_id,
            )
            return 0.0

        if event.event_type == EventType.AMBULANCE_FREE:
            ambulance_id = int(event.payload["ambulance_id"])
            ambulance = self._ambulances[ambulance_id]
            ambulance.location = ambulance.home_location
            ambulance.status = AmbulanceStatus.FREE
            ambulance.assigned_incident_id = None
            ambulance.completed_jobs += 1
            if ambulance.busy_start is not None:
                ambulance.busy_time += self.current_time - ambulance.busy_start
            ambulance.busy_start = None
            ambulance.eta_free_at = self.current_time
            return 0.0

        return 0.0

    def _create_incident(self) -> None:
        severity = self._sample_severity()
        incident = Incident(
            incident_id=self._incident_counter,
            location=self._sample_incident_location(),
            severity=severity,
            reported_at=self.current_time,
            service_time=self._sample_service_time(severity),
        )
        self._incidents[incident.incident_id] = incident
        self._incident_counter += 1

    def _nearest_hospital(self, location: GridPoint) -> GridPoint:
        return min(
            self.config.hospital_locations,
            key=lambda hospital: math.dist(location, hospital),
        )

    def _arrival_reward(self, incident: Incident, response_time: float) -> float:
        severity_weight = self.config.severity_weights[incident.severity.value]
        reward_cfg = self.config.reward
        safe_response_time = max(response_time, 1.0)
        return (
            reward_cfg.successful_rescue_weight * severity_weight * (1.0 / safe_response_time)
            - reward_cfg.waiting_time_penalty_weight * severity_weight * response_time
        )

    def _has_actionable_assignment(self) -> bool:
        return bool(self.get_pending_incidents()) and bool(self.get_free_ambulances())

    def _should_end_naturally(self) -> bool:
        if not self._incidents:
            return False
        if (
            self.current_time
            < self.config.episode_duration_minutes - self.config.quiet_window_minutes
        ):
            return False
        if self.get_pending_incidents():
            return False
        if self._count_inflight_incidents() > 0:
            return False
        if self._next_incident_time is None:
            return True
        return (self._next_incident_time - self.current_time) > self.config.quiet_window_minutes

    def _count_inflight_incidents(self) -> int:
        return sum(
            1
            for incident in self._incidents.values()
            if incident.status
            in (
                IncidentStatus.DISPATCHED,
                IncidentStatus.ON_SCENE,
                IncidentStatus.TRANSPORTING,
            )
        )

    def _catastrophic_failure(self) -> bool:
        overdue_critical = 0
        for incident in self._incidents.values():
            if incident.severity != SeverityLevel.CRITICAL:
                continue
            if incident.status in (IncidentStatus.ON_SCENE, IncidentStatus.TRANSPORTING, IncidentStatus.RESOLVED):
                continue
            if (self.current_time - incident.reported_at) > self.config.critical_response_threshold_minutes:
                overdue_critical += 1
        return (
            overdue_critical > self.config.catastrophic_overdue_critical_limit
        )

    def _current_busy_time(self, ambulance: Ambulance) -> float:
        busy_time = ambulance.busy_time
        if ambulance.status == AmbulanceStatus.BUSY and ambulance.busy_start is not None:
            busy_time += self.current_time - ambulance.busy_start
        return busy_time

    def _severity_code(self, severity_name: str) -> int:
        return {"low": 1, "moderate": 2, "critical": 3}.get(severity_name, 0)

    def _percentile(self, values: list[float], percentile: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        if len(ordered) == 1:
            return ordered[0]
        index = (percentile / 100.0) * (len(ordered) - 1)
        lower = math.floor(index)
        upper = math.ceil(index)
        if lower == upper:
            return ordered[lower]
        blend = index - lower
        return ordered[lower] * (1.0 - blend) + ordered[upper] * blend
