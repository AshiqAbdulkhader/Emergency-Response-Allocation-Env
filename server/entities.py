# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Internal simulator entities for the Emergency Response Allocation environment."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .config import GridPoint


class AmbulanceStatus(str, Enum):
    FREE = "free"
    BUSY = "busy"


class IncidentStatus(str, Enum):
    PENDING = "pending"
    DISPATCHED = "dispatched"
    ON_SCENE = "on_scene"
    TRANSPORTING = "transporting"
    RESOLVED = "resolved"


class SeverityLevel(str, Enum):
    CRITICAL = "critical"
    MODERATE = "moderate"
    LOW = "low"


class ZoneType(str, Enum):
    RESIDENTIAL = "residential"
    COMMERCIAL = "commercial"
    HOSPITAL = "hospital"
    DEPOT = "depot"


class EventType(str, Enum):
    INCIDENT_ARRIVAL = "incident_arrival"
    ARRIVE_SCENE = "arrive_scene"
    SERVICE_COMPLETE = "service_complete"
    ARRIVE_HOSPITAL = "arrive_hospital"
    AMBULANCE_FREE = "ambulance_free"


EVENT_PRIORITIES: dict[EventType, int] = {
    EventType.AMBULANCE_FREE: 0,
    EventType.INCIDENT_ARRIVAL: 1,
    EventType.ARRIVE_SCENE: 2,
    EventType.SERVICE_COMPLETE: 2,
    EventType.ARRIVE_HOSPITAL: 2,
}


@dataclass(order=True)
class Event:
    scheduled_time: float
    priority: int
    sequence: int
    event_type: EventType = field(compare=False)
    payload: dict[str, Any] = field(default_factory=dict, compare=False)


@dataclass
class Ambulance:
    ambulance_id: int
    depot_id: int
    home_location: GridPoint
    location: GridPoint
    status: AmbulanceStatus = AmbulanceStatus.FREE
    eta_free_at: float = 0.0
    busy_time: float = 0.0
    busy_start: float | None = None
    assigned_incident_id: int | None = None
    completed_jobs: int = 0


@dataclass
class Incident:
    incident_id: int
    location: GridPoint
    severity: SeverityLevel
    reported_at: float
    service_time: float
    status: IncidentStatus = IncidentStatus.PENDING
    assigned_ambulance_id: int | None = None
    dispatch_at: float | None = None
    arrival_at_scene: float | None = None
    resolved_at: float | None = None
    hospital_location: GridPoint | None = None
    estimated_travel_time: float | None = None
