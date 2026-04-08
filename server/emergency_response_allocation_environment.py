# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""OpenEnv environment wrapper for the Emergency Response Allocation simulator."""

from __future__ import annotations

from openenv.core.env_server.interfaces import Environment

try:
    from ..models import (
        EmergencyResponseAllocationAction,
        EmergencyResponseAllocationObservation,
        EmergencyResponseAllocationState,
    )
    from .config import SimulationConfig
    from .simulator import ERASSimulator
except ImportError:
    from models import (
        EmergencyResponseAllocationAction,
        EmergencyResponseAllocationObservation,
        EmergencyResponseAllocationState,
    )
    from server.config import SimulationConfig
    from server.simulator import ERASSimulator


class EmergencyResponseAllocationEnvironment(
    Environment[
        EmergencyResponseAllocationAction,
        EmergencyResponseAllocationObservation,
        EmergencyResponseAllocationState,
    ]
):
    """Event-driven ambulance dispatch environment."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, config: SimulationConfig | None = None):
        super().__init__()
        self._simulator = ERASSimulator(config=config, auto_reset=True)

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        **kwargs,
    ) -> EmergencyResponseAllocationObservation:
        return self._simulator.reset(seed=seed, episode_id=episode_id)

    def step(
        self,
        action: EmergencyResponseAllocationAction,
        timeout_s: float | None = None,
        **kwargs,
    ) -> EmergencyResponseAllocationObservation:
        return self._simulator.step(action.action_index)

    @property
    def state(self) -> EmergencyResponseAllocationState:
        return self._simulator.state
