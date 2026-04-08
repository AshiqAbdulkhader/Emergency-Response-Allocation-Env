# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Emergency Response Allocation Environment client."""

from __future__ import annotations

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

try:
    from .models import (
        EmergencyResponseAllocationAction,
        EmergencyResponseAllocationObservation,
        EmergencyResponseAllocationState,
    )
except ImportError:
    from models import (
        EmergencyResponseAllocationAction,
        EmergencyResponseAllocationObservation,
        EmergencyResponseAllocationState,
    )


class EmergencyResponseAllocationEnv(
    EnvClient[
        EmergencyResponseAllocationAction,
        EmergencyResponseAllocationObservation,
        EmergencyResponseAllocationState,
    ]
):
    """Client for the Emergency Response Allocation OpenEnv server."""

    def _step_payload(self, action: EmergencyResponseAllocationAction) -> Dict:
        return action.model_dump(exclude_none=True)

    def _parse_result(
        self, payload: Dict
    ) -> StepResult[EmergencyResponseAllocationObservation]:
        obs_data = payload.get("observation", {})
        observation = EmergencyResponseAllocationObservation.model_validate(
            {
                **obs_data,
                "reward": payload.get("reward"),
                "done": payload.get("done", False),
            }
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> EmergencyResponseAllocationState:
        return EmergencyResponseAllocationState.model_validate(payload)
