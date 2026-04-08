# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Benchmark inference runner for ERAS using an OpenAI-compatible LLM client."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Sequence

try:
    from .client import EmergencyResponseAllocationEnv
    from .models import EmergencyResponseAllocationAction, EmergencyResponseAllocationObservation
    from .server.evaluation import ERASTask, grade_task_result, get_task, list_tasks
except ImportError:
    from client import EmergencyResponseAllocationEnv
    from models import EmergencyResponseAllocationAction, EmergencyResponseAllocationObservation
    from server.evaluation import ERASTask, grade_task_result, get_task, list_tasks


DEFAULT_OPENENV_URL = "http://127.0.0.1:8000"
VALID_ENV_VARS = ("API_BASE_URL", "MODEL_NAME", "HF_TOKEN")
SYSTEM_PROMPT = """
You are an ambulance dispatch planner for an event-driven emergency response simulator.
Choose exactly one valid discrete action index for the current decision state.
Prefer fast responses for critical incidents, then minimize travel time and waiting time.
Output only a single JSON object with this schema:
{"action_index": 12, "reason": "short explanation"}
""".strip()


@dataclass(frozen=True)
class InferenceConfig:
    api_base_url: str
    model_name: str
    hf_token: str
    openenv_url: str
    max_completion_tokens: int
    temperature: float


@dataclass(frozen=True)
class ActionDecision:
    action_index: int
    reason: str
    raw_output: str
    used_fallback: bool


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run ERAS benchmark tasks with an OpenAI-compatible LLM client."
    )
    parser.add_argument(
        "--task-id",
        action="append",
        default=None,
        help="Repeatable benchmark task id selector. Defaults to all tasks.",
    )
    parser.add_argument(
        "--openenv-url",
        default=os.environ.get("OPENENV_URL", DEFAULT_OPENENV_URL),
        help="Base URL for the ERAS OpenEnv server.",
    )
    parser.add_argument("--max-completion-tokens", type=int, default=220)
    parser.add_argument("--temperature", type=float, default=0.1)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_argument_parser().parse_args(argv)


def load_runtime_config(args: argparse.Namespace) -> InferenceConfig:
    values = {name: os.environ.get(name, "").strip() for name in VALID_ENV_VARS}
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise RuntimeError(
            "Missing required environment variables: " + ", ".join(missing)
        )
    return InferenceConfig(
        api_base_url=values["API_BASE_URL"],
        model_name=values["MODEL_NAME"],
        hf_token=values["HF_TOKEN"],
        openenv_url=args.openenv_url,
        max_completion_tokens=args.max_completion_tokens,
        temperature=args.temperature,
    )


def build_openai_client(config: InferenceConfig):
    from openai import OpenAI

    return OpenAI(base_url=config.api_base_url, api_key=config.hf_token)


def resolve_tasks(requested_ids: Sequence[str] | None) -> list[ERASTask]:
    if not requested_ids:
        return list_tasks()
    return [get_task(task_id) for task_id in requested_ids]


class LLMDispatchAgent:
    """Select ERAS actions using an OpenAI-compatible chat model."""

    def __init__(self, client: Any, config: InferenceConfig):
        self._client = client
        self._config = config

    def choose_action(
        self,
        task: ERASTask,
        observation: EmergencyResponseAllocationObservation,
    ) -> ActionDecision:
        prompt = build_prompt(task, observation)
        raw_output = ""
        try:
            response = self._client.chat.completions.create(
                model=self._config.model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=self._config.temperature,
                max_tokens=self._config.max_completion_tokens,
            )
            raw_output = completion_to_text(response)
            parsed_index = parse_action_index(raw_output)
            if parsed_index is not None and _is_valid_action(parsed_index, observation):
                return ActionDecision(
                    action_index=parsed_index,
                    reason=extract_reason(raw_output),
                    raw_output=raw_output,
                    used_fallback=False,
                )
        except Exception as exc:  # pragma: no cover - exercised in live inference
            raw_output = f"LLM_ERROR: {exc}"

        fallback_index = fallback_action_index(observation)
        return ActionDecision(
            action_index=fallback_index,
            reason="fallback_valid_action",
            raw_output=raw_output,
            used_fallback=True,
        )


def build_prompt(task: ERASTask, observation: EmergencyResponseAllocationObservation) -> str:
    lines = [
        f"Task: {task.task_id}",
        f"Objective: {task.description}",
        f"Difficulty: {task.difficulty}",
        (
            f"TimeOfDayHours: {observation.time_of_day:.2f} | "
            f"EventType: {observation.event_type}"
        ),
        "Ambulances:",
    ]
    for ambulance in observation.ambulances:
        lines.append(
            "  "
            f"id={ambulance.ambulance_id} loc=({ambulance.x},{ambulance.y}) "
            f"status={ambulance.status} eta_free={ambulance.eta_free:.2f}"
        )

    lines.append("VisibleIncidents:")
    for slot_index, incident in enumerate(observation.incidents):
        if incident.incident_id < 0:
            continue
        lines.append(
            "  "
            f"slot={slot_index} incident_id={incident.incident_id} "
            f"loc=({incident.x},{incident.y}) severity={incident.severity} "
            f"wait={incident.time_since_reported:.2f}"
        )

    lines.append("ValidActions:")
    for action_index, is_valid in enumerate(observation.valid_action_mask):
        if not is_valid:
            continue
        lines.append("  " + describe_action(action_index, observation))

    lines.append(
        "Return exactly one JSON object with keys action_index and reason. "
        "Do not include markdown or extra text."
    )
    return "\n".join(lines)


def describe_action(
    action_index: int,
    observation: EmergencyResponseAllocationObservation,
) -> str:
    if action_index < 50:
        ambulance_id, incident_slot = divmod(action_index, 10)
        incident = observation.incidents[incident_slot]
        travel_time = observation.travel_times[ambulance_id][incident_slot]
        incident_id = observation.visible_incident_ids[incident_slot]
        return (
            f"{action_index}=assign ambulance={ambulance_id} "
            f"slot={incident_slot} incident_id={incident_id} "
            f"severity={incident.severity} wait={incident.time_since_reported:.2f} "
            f"travel={travel_time:.2f}"
        )
    ambulance_id = action_index - 50
    return f"{action_index}=hold ambulance={ambulance_id}"


def completion_to_text(response: Any) -> str:
    try:
        choice = response.choices[0]
    except (AttributeError, IndexError, KeyError, TypeError):
        return str(response)
    message = getattr(choice, "message", None)
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        fragments: list[str] = []
        for item in content:
            if isinstance(item, str):
                fragments.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                fragments.append(item["text"])
            else:
                text = getattr(item, "text", None)
                if isinstance(text, str):
                    fragments.append(text)
        return "".join(fragments).strip()
    return str(content).strip()


def extract_json_object(text: str) -> dict[str, Any] | None:
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
    candidates.sort(key=len, reverse=True)

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def parse_action_index(text: str) -> int | None:
    payload = extract_json_object(text)
    if payload is not None:
        raw_value = payload.get("action_index")
        if isinstance(raw_value, bool):
            return None
        try:
            return int(raw_value)
        except (TypeError, ValueError):
            pass

    match = re.search(r'"action_index"\s*:\s*(-?\d+)', text)
    if match:
        return int(match.group(1))

    integer_match = re.search(r"\b(\d+)\b", text)
    if integer_match:
        return int(integer_match.group(1))
    return None


def extract_reason(text: str) -> str:
    payload = extract_json_object(text)
    if payload is not None and isinstance(payload.get("reason"), str):
        reason = payload["reason"].strip()
        if reason:
            return reason
    return "llm_selected_action"


def fallback_action_index(observation: EmergencyResponseAllocationObservation) -> int:
    best_action: tuple[float, int] | None = None
    for action_index, is_valid in enumerate(observation.valid_action_mask):
        if not is_valid:
            continue
        if action_index >= 50:
            candidate = (-1e6, action_index)
        else:
            ambulance_id, incident_slot = divmod(action_index, 10)
            incident = observation.incidents[incident_slot]
            travel_time = observation.travel_times[ambulance_id][incident_slot]
            severity_bonus = float(incident.severity_code) * 10.0
            wait_bonus = float(incident.time_since_reported) * 0.25
            utility = severity_bonus + wait_bonus - travel_time
            candidate = (utility, action_index)
        if best_action is None or candidate > best_action:
            best_action = candidate

    if best_action is None:
        raise RuntimeError("No valid action available for fallback policy")
    return best_action[1]


def _is_valid_action(
    action_index: int, observation: EmergencyResponseAllocationObservation
) -> bool:
    return 0 <= action_index < len(observation.valid_action_mask) and observation.valid_action_mask[action_index]


def format_start_log(task: ERASTask, index: int, total: int) -> str:
    return (
        "[START] "
        f"task_id={task.task_id} "
        f"task_index={index} "
        f"task_total={total} "
        f"difficulty={task.difficulty} "
        f"seed={task.seed} "
        f"grader={task.grader_name}"
    )


def format_step_log(
    *,
    task_id: str,
    step_index: int,
    current_sim_time: float,
    event_type: str,
    action_index: int,
    step_reward: float,
    done: bool,
    used_fallback: bool,
) -> str:
    return (
        "[STEP] "
        f"task_id={task_id} "
        f"step_index={step_index} "
        f"current_sim_time={current_sim_time:.2f} "
        f"event_type={event_type} "
        f"action_index={action_index} "
        f"step_reward={step_reward:.6f} "
        f"done={int(done)} "
        f"fallback={int(used_fallback)}"
    )


def format_end_log(
    *,
    task_id: str,
    score: float,
    reward: float,
    episode_reward: float,
    step_count: int,
    status: str,
) -> str:
    return (
        "[END] "
        f"task_id={task_id} "
        f"score={score:.6f} "
        f"reward={reward:.6f} "
        f"episode_reward={episode_reward:.6f} "
        f"step_count={step_count} "
        f"status={status}"
    )


def emit_log(line: str) -> None:
    print(line, flush=True)


def run_benchmark(
    *,
    tasks: Sequence[ERASTask],
    config: InferenceConfig,
    agent: LLMDispatchAgent,
) -> list[dict[str, float | str]]:
    summaries: list[dict[str, float | str]] = []

    for task_index, task in enumerate(tasks, start=1):
        emit_log(format_start_log(task, task_index, len(tasks)))
        with EmergencyResponseAllocationEnv(base_url=config.openenv_url).sync() as env:
            result = env.reset(seed=task.seed, episode_id=task.task_id)
            episode_reward = float(result.reward or 0.0)
            decision_count = 0

            while not result.done and decision_count < task.max_decision_steps:
                decision = agent.choose_action(task, result.observation)
                result = env.step(
                    EmergencyResponseAllocationAction(action_index=decision.action_index)
                )
                decision_count += 1
                step_reward = float(result.reward or 0.0)
                episode_reward += step_reward
                emit_log(
                    format_step_log(
                        task_id=task.task_id,
                        step_index=decision_count,
                        current_sim_time=result.observation.info.current_sim_time,
                        event_type=result.observation.event_type,
                        action_index=decision.action_index,
                        step_reward=step_reward,
                        done=result.done,
                        used_fallback=decision.used_fallback,
                    )
                )

            state = env.state()

        grade = grade_task_result(
            task,
            info=state.info,
            episode_reward=episode_reward,
            step_count=state.step_count,
        )
        verify_unit_interval(grade.score, "score", task.task_id)
        verify_unit_interval(grade.reward, "reward", task.task_id)
        status = "success" if state.episode_done else "step_limit"
        emit_log(
            format_end_log(
                task_id=task.task_id,
                score=grade.score,
                reward=grade.reward,
                episode_reward=episode_reward,
                step_count=state.step_count,
                status=status,
            )
        )
        summaries.append(
            {
                "task_id": task.task_id,
                "score": grade.score,
                "reward": grade.reward,
                "episode_reward": episode_reward,
            }
        )

    return summaries


def verify_unit_interval(value: float, label: str, task_id: str) -> None:
    if not (0.0 <= value <= 1.0):
        raise ValueError(
            f"{label} for task {task_id} must be in [0.0, 1.0], received {value}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_runtime_config(args)
    client = build_openai_client(config)
    agent = LLMDispatchAgent(client=client, config=config)
    tasks = resolve_tasks(args.task_id)
    run_benchmark(tasks=tasks, config=config, agent=agent)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
