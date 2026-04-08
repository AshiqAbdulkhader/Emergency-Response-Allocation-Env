# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FastAPI app for the Emergency Response Allocation OpenEnv environment."""

from fastapi import FastAPI

try:
    from openenv.core.env_server.http_server import create_app
    from openenv.core.env_server.types import SchemaResponse
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import (
        EmergencyResponseAllocationAction,
        EmergencyResponseAllocationObservation,
        EmergencyResponseAllocationState,
    )
    from ..server.evaluation import GradeRequest, GradeResult, TaskSpec, grade_request_for_task, list_task_specs
    from ..server.emergency_response_allocation_environment import (
        EmergencyResponseAllocationEnvironment,
    )
except ImportError:
    from models import (
        EmergencyResponseAllocationAction,
        EmergencyResponseAllocationObservation,
        EmergencyResponseAllocationState,
    )
    from server.evaluation import (
        GradeRequest,
        GradeResult,
        TaskSpec,
        grade_request_for_task,
        list_task_specs,
    )
    from server.emergency_response_allocation_environment import (
        EmergencyResponseAllocationEnvironment,
    )


app = create_app(
    EmergencyResponseAllocationEnvironment,
    EmergencyResponseAllocationAction,
    EmergencyResponseAllocationObservation,
    env_name="emergency_response_allocation",
    max_concurrent_envs=4,
)


def _remove_route(app_instance: FastAPI, path: str, method: str) -> None:
    app_instance.router.routes = [
        route
        for route in app_instance.router.routes
        if not (
            getattr(route, "path", None) == path
            and method.upper() in getattr(route, "methods", set())
        )
    ]


_remove_route(app, "/state", "GET")
_remove_route(app, "/schema", "GET")


@app.get("/state", response_model=EmergencyResponseAllocationState, tags=["State Management"])
async def state() -> EmergencyResponseAllocationState:
    env = EmergencyResponseAllocationEnvironment()
    try:
        return env.state
    finally:
        env.close()


@app.get("/schema", response_model=SchemaResponse, tags=["Schema"])
async def schema() -> SchemaResponse:
    return SchemaResponse(
        action=EmergencyResponseAllocationAction.model_json_schema(),
        observation=EmergencyResponseAllocationObservation.model_json_schema(),
        state=EmergencyResponseAllocationState.model_json_schema(),
    )


@app.get("/tasks", response_model=list[TaskSpec], tags=["Evaluation"])
async def tasks() -> list[TaskSpec]:
    """Enumerate ERAS benchmark tasks."""

    return list_task_specs()


@app.post("/grade/{task_id}", response_model=GradeResult, tags=["Evaluation"])
async def grade(task_id: str, request: GradeRequest) -> GradeResult:
    """Grade externally supplied task metrics into [0, 1] scores."""

    return grade_request_for_task(task_id, request)


def main(host: str = "0.0.0.0", port: int = 8000):
    """Run the environment server with uvicorn."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
