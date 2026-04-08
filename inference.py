# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Submission-style inference runner for ERAS."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import textwrap
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

from openai import OpenAI

try:
    from .client import EmergencyResponseAllocationEnv
    from .models import EmergencyResponseAllocationAction, EmergencyResponseAllocationObservation
    from .server.evaluation import ERASTask, get_task, grade_task_result, list_tasks
except ImportError:
    from client import EmergencyResponseAllocationEnv
    from models import EmergencyResponseAllocationAction, EmergencyResponseAllocationObservation
    from server.evaluation import ERASTask, get_task, grade_task_result, list_tasks


LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME") or os.getenv("IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
OPENENV_URL = os.getenv("OPENENV_URL") or "http://127.0.0.1:8000"
BENCHMARK = os.getenv("ERAS_BENCHMARK", "emergency_response_allocation")
MAX_STEPS = int(os.getenv("MAX_STEPS", "96"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "220"))
SUCCESS_SCORE_THRESHOLD = float(os.getenv("SUCCESS_SCORE_THRESHOLD", "0.1"))

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are interacting with an event-driven ambulance dispatch environment.
    Choose exactly one valid discrete action index for the current state.
    Prefer rapid service for critical incidents, then lower travel time and long waits.
    Reply with exactly one JSON object with this schema:
    {"action_index": 12, "reason": "short explanation"}
    """
).strip()


@dataclass(frozen=True)
class ActionDecision:
    action_index: int
    action_str: str
    raw_output: str


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run ERAS benchmark tasks with strict benchmark stdout formatting."
    )
    parser.add_argument(
        "--task-id",
        action="append",
        default=None,
        help="Repeatable ERAS task id selector. Defaults to all benchmark tasks.",
    )
    parser.add_argument(
        "--openenv-url",
        default=OPENENV_URL,
        help="ERAS server URL used when LOCAL_IMAGE_NAME is not set.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_argument_parser().parse_args(argv)


def validate_environment() -> None:
    if not API_KEY:
        raise RuntimeError("HF_TOKEN must be defined for OpenAI client authentication.")


def resolve_tasks(requested_ids: Sequence[str] | None) -> list[ERASTask]:
    if not requested_ids:
        return list_tasks()
    return [get_task(task_id) for task_id in requested_ids]


def build_openai_client() -> OpenAI:
    return OpenAI(base_url=API_BASE_URL, api_key=API_KEY)


def start_line(task: str, env: str, model: str) -> str:
    return f"[START] task={task} env={env} model={model}"


def step_line(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str],
) -> str:
    error_value = error if error else "null"
    return (
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_value}"
    )


def end_line(success: bool, steps: int, score: float, rewards: List[float]) -> str:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    return (
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}"
    )


def log_start(task: str, env: str, model: str) -> None:
    print(start_line(task=task, env=env, model=model), flush=True)


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    print(
        step_line(step=step, action=action, reward=reward, done=done, error=error),
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    print(end_line(success=success, steps=steps, score=score, rewards=rewards), flush=True)


def completion_to_text(completion: Any) -> str:
    try:
        content = completion.choices[0].message.content
    except (AttributeError, IndexError, KeyError, TypeError):
        return str(completion).strip()

    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts).strip()
    return str(content).strip()


def build_user_prompt(
    task: ERASTask,
    step: int,
    observation: EmergencyResponseAllocationObservation,
    history: Sequence[str],
) -> str:
    history_block = "\n".join(history[-4:]) if history else "None"
    ambulance_lines = [
        (
            f"id={ambulance.ambulance_id} loc=({ambulance.x},{ambulance.y}) "
            f"status={ambulance.status} eta={ambulance.eta_free:.2f}"
        )
        for ambulance in observation.ambulances
    ]
    incident_lines = []
    for slot, incident in enumerate(observation.incidents):
        if incident.incident_id < 0:
            continue
        incident_lines.append(
            f"slot={slot} incident_id={incident.incident_id} "
            f"severity={incident.severity} loc=({incident.x},{incident.y}) "
            f"wait={incident.time_since_reported:.2f}"
        )
    valid_actions = [
        action_string(action_index)
        for action_index, is_valid in enumerate(observation.valid_action_mask)
        if is_valid
    ]
    return textwrap.dedent(
        f"""
        Task: {task.task_id}
        Description: {task.description}
        Step: {step}
        TimeOfDayHours: {observation.time_of_day:.2f}
        EventType: {observation.event_type}
        Ambulances:
        {chr(10).join(ambulance_lines) or 'None'}
        VisibleIncidents:
        {chr(10).join(incident_lines) or 'None'}
        ValidActions:
        {chr(10).join(valid_actions) or 'None'}
        PreviousSteps:
        {history_block}
        Choose the best next action.
        """
    ).strip()


def get_model_action(
    client: OpenAI,
    task: ERASTask,
    step: int,
    observation: EmergencyResponseAllocationObservation,
    history: Sequence[str],
) -> ActionDecision:
    user_prompt = build_user_prompt(task, step, observation, history)
    raw_output = ""
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        raw_output = completion_to_text(completion)
        parsed_index = parse_action_index(raw_output)
        if parsed_index is not None and is_valid_action(parsed_index, observation):
            return ActionDecision(
                action_index=parsed_index,
                action_str=action_string(parsed_index),
                raw_output=raw_output,
            )
    except Exception:
        raw_output = ""

    fallback_index = fallback_action_index(observation)
    return ActionDecision(
        action_index=fallback_index,
        action_str=action_string(fallback_index),
        raw_output=raw_output,
    )


def extract_json_object(text: str) -> Optional[dict[str, Any]]:
    stripped = text.strip()
    if not stripped:
        return None

    fence_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", stripped, re.DOTALL)
    if fence_match:
        stripped = fence_match.group(1).strip()

    candidates: list[str] = [stripped]
    for start in (match.start() for match in re.finditer(r"\{", stripped)):
        depth = 0
        for index in range(start, len(stripped)):
            char = stripped[index]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(stripped[start : index + 1])
                    break

    for candidate in sorted(candidates, key=len, reverse=True):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def parse_action_index(text: str) -> Optional[int]:
    payload = extract_json_object(text)
    if payload is not None:
        raw_action = payload.get("action_index")
        if isinstance(raw_action, bool):
            return None
        try:
            return int(raw_action)
        except (TypeError, ValueError):
            pass

    match = re.search(r'"action_index"\s*:\s*(-?\d+)', text)
    if match:
        return int(match.group(1))

    match = re.search(r"\b(\d+)\b", text)
    if match:
        return int(match.group(1))
    return None


def is_valid_action(
    action_index: int,
    observation: EmergencyResponseAllocationObservation,
) -> bool:
    return (
        0 <= action_index < len(observation.valid_action_mask)
        and observation.valid_action_mask[action_index]
    )


def action_string(action_index: int) -> str:
    if action_index < 50:
        ambulance_id, incident_slot = divmod(action_index, 10)
        return f"assign(ambulance={ambulance_id},incident_slot={incident_slot})"
    return f"hold(ambulance={action_index - 50})"


def fallback_action_index(observation: EmergencyResponseAllocationObservation) -> int:
    best_choice: tuple[float, int] | None = None
    for action_index, is_valid in enumerate(observation.valid_action_mask):
        if not is_valid:
            continue
        if action_index >= 50:
            candidate = (-1e6, action_index)
        else:
            ambulance_id, incident_slot = divmod(action_index, 10)
            incident = observation.incidents[incident_slot]
            travel_time = observation.travel_times[ambulance_id][incident_slot]
            utility = (
                float(incident.severity_code) * 10.0
                + float(incident.time_since_reported) * 0.25
                - float(travel_time)
            )
            candidate = (utility, action_index)
        if best_choice is None or candidate > best_choice:
            best_choice = candidate

    if best_choice is None:
        raise RuntimeError("No valid action available.")
    return best_choice[1]


async def create_env(openenv_url: str) -> EmergencyResponseAllocationEnv:
    if LOCAL_IMAGE_NAME:
        return await EmergencyResponseAllocationEnv.from_docker_image(LOCAL_IMAGE_NAME)

    env = EmergencyResponseAllocationEnv(base_url=openenv_url)
    await env.connect()
    return env


def clamp_score(score: float) -> float:
    return min(max(float(score), 0.0), 1.0)


async def run_task_episode(
    client: OpenAI,
    task: ERASTask,
    openenv_url: str,
) -> None:
    env: EmergencyResponseAllocationEnv | None = None
    rewards: List[float] = []
    history: List[str] = []
    steps_taken = 0
    score = 0.0
    success = False
    max_steps = min(MAX_STEPS, task.max_decision_steps)

    log_start(task=task.task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        env = await create_env(openenv_url)
        result = await env.reset(seed=task.seed, episode_id=task.task_id)

        for step in range(1, max_steps + 1):
            if result.done:
                break

            decision = get_model_action(client, task, step, result.observation, history)
            result = await env.step(
                EmergencyResponseAllocationAction(action_index=decision.action_index)
            )

            reward = float(result.reward or 0.0)
            done = bool(result.done)
            error = None

            rewards.append(reward)
            steps_taken = step
            history.append(
                f"step={step} action={decision.action_str} reward={reward:.2f}"
            )

            log_step(
                step=step,
                action=decision.action_str,
                reward=reward,
                done=done,
                error=error,
            )

            if done:
                break

        if env is not None:
            state = await env.state()
            graded = grade_task_result(
                task,
                info=state.info,
                episode_reward=sum(rewards),
                step_count=state.step_count,
            )
            score = clamp_score(graded.score)
            success = score >= SUCCESS_SCORE_THRESHOLD
            if not (0.0 <= score <= 1.0):
                raise ValueError(f"Task {task.task_id} returned score outside [0, 1].")
    finally:
        if env is not None:
            try:
                await env.close()
            except Exception:
                pass
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


async def main(argv: Sequence[str] | None = None) -> None:
    validate_environment()
    args = parse_args(argv)
    tasks = resolve_tasks(args.task_id)
    client = build_openai_client()

    for task in tasks:
        await run_task_episode(client=client, task=task, openenv_url=args.openenv_url)


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))
