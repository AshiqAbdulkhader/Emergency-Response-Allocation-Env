"""Train a local dispatch policy model with TRL GRPO and OpenEnv rewards."""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from client import EmergencyResponseAllocationEnv
from models import (
    ACTION_DIM,
    HOLD_ACTION_OFFSET,
    MAX_OBSERVABLE_INCIDENTS,
    NUM_AMBULANCES,
    EmergencyResponseAllocationAction,
    EmergencyResponseAllocationObservation,
)
from server.baselines import nearest_ambulance_policy, random_policy, severity_first_policy
from server.config import SimulationConfig
from server.simulator import ERASSimulator

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_OUTPUT_DIR = "training/grpo-output"
DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_COMPLETION_TOKEN_BUDGET = 192
INVALID_ACTION_PENALTY = -3.0
ENVIRONMENT_ERROR_PENALTY = -6.0

SYSTEM_PROMPT = """You are an ambulance dispatch planner operating the Emergency Response Allocation System (ERAS).

Your job is to choose one valid action at each dispatch event.

Primary objective:
- minimize severity-weighted response time

Secondary objectives:
- maximize incident coverage
- keep ambulance utilization healthy
- avoid missed critical cases

Policy rules:
- Critical incidents are most urgent, then moderate, then low.
- Prefer shorter travel times when priorities are similar.
- Only choose actions listed as valid.
- Use hold only when deliberately keeping a free ambulance in reserve is better than dispatching now.

Return ONLY one JSON object and no extra prose.
"""

ACTION_TYPE_ALIASES = {
    "assign": "assign",
    "dispatch": "assign",
    "allocate": "assign",
    "send": "assign",
    "respond": "assign",
    "hold": "hold",
    "wait": "hold",
    "idle": "hold",
    "noop": "hold",
    "no_op": "hold",
    "no-op": "hold",
}

SEVERITY_WEIGHTS = SimulationConfig().severity_weights


@dataclass
class ParsedDispatchAction:
    action_index: int | None = None
    action_type: str | None = None
    ambulance_id: int | None = None
    incident_slot: int | None = None
    incident_id: int | None = None
    justification: str | None = None
    confidence: float = 0.5


def compact_preview(value: Any, max_chars: int = 160) -> str:
    try:
        text = json.dumps(value, ensure_ascii=True, sort_keys=True)
    except TypeError:
        text = str(value)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _edit_distance(a: str, b: str) -> int:
    if len(a) < len(b):
        return _edit_distance(b, a)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (ca != cb)))
        prev = curr
    return prev[-1]


def get_payload_value(payload: Dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in payload:
            return payload[name]

    lowered = {str(key).lower(): value for key, value in payload.items()}
    for name in names:
        if name.lower() in lowered:
            return lowered[name.lower()]

    for key, value in lowered.items():
        for name in names:
            threshold = max(2, len(name) // 3)
            if _edit_distance(key, name.lower()) <= threshold:
                return value
    return None


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a GRPO dispatch policy against the ERAS OpenEnv environment."
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset-episodes", type=int, default=12)
    parser.add_argument("--rollout-steps", type=int, default=8)
    parser.add_argument(
        "--collection-policy",
        choices=["random", "nearest", "severity_first"],
        default="nearest",
        help="Policy used to build prompt states for GRPO training.",
    )
    parser.add_argument(
        "--reward-backend",
        choices=["local", "remote"],
        default="local",
        help="Use in-process simulator rewards or a live OpenEnv server.",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="Base URL for the OpenEnv server when reward-backend=remote.",
    )
    parser.add_argument("--num-generations", type=int, default=2)
    parser.add_argument(
        "--max-completion-length",
        type=int,
        default=DEFAULT_COMPLETION_TOKEN_BUDGET,
    )
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument(
        "--plot-metric-key",
        default=None,
        help="Optional extra metric key from trainer log history to plot.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--load-model-only",
        action="store_true",
        help="Download and load the selected model and tokenizer, then exit.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to model/tokenizer loading.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the prompt dataset and smoke-test the reward function without training.",
    )
    parser.add_argument(
        "--push-to-hub",
        type=str,
        default=None,
        help="Hugging Face repo id to push the trained model to.",
    )
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    return build_argument_parser().parse_args(argv)


def make_training_args(**overrides: Any) -> argparse.Namespace:
    parser = build_argument_parser()
    defaults = vars(parser.parse_args([]))
    unknown = sorted(set(overrides) - set(defaults))
    if unknown:
        raise ValueError(f"Unknown training args: {', '.join(unknown)}")
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def is_free_ambulance(status: str) -> bool:
    return status.lower() == "free"


def describe_action_index(
    action_index: int,
    obs: EmergencyResponseAllocationObservation,
) -> Dict[str, Any]:
    if action_index >= HOLD_ACTION_OFFSET:
        ambulance_id = action_index - HOLD_ACTION_OFFSET
        return {
            "action_index": action_index,
            "action_type": "hold",
            "ambulance_id": ambulance_id,
            "incident_slot": None,
            "incident_id": None,
            "description": f"hold ambulance {ambulance_id}",
        }

    ambulance_id, incident_slot = divmod(action_index, MAX_OBSERVABLE_INCIDENTS)
    incident = obs.incidents[incident_slot]
    incident_id = obs.visible_incident_ids[incident_slot]
    travel_time = obs.travel_times[ambulance_id][incident_slot]
    return {
        "action_index": action_index,
        "action_type": "assign",
        "ambulance_id": ambulance_id,
        "incident_slot": incident_slot,
        "incident_id": incident_id if incident_id != -1 else None,
        "severity": incident.severity,
        "travel_time": travel_time,
        "description": (
            f"assign ambulance {ambulance_id} -> slot {incident_slot} "
            f"(incident {incident_id}, {incident.severity}, {travel_time:.1f}m)"
        ),
    }


def valid_action_descriptions(
    obs: EmergencyResponseAllocationObservation,
) -> List[Dict[str, Any]]:
    descriptions: List[Dict[str, Any]] = []
    for action_index, is_valid in enumerate(obs.valid_action_mask):
        if is_valid:
            descriptions.append(describe_action_index(action_index, obs))
    return descriptions


def format_observation(obs: EmergencyResponseAllocationObservation) -> str:
    parts = [
        f"Event: {obs.event_type}",
        (
            f"Time of day: {obs.time_of_day:.2f}h | Sim time: {obs.info.current_sim_time:.1f}m "
            f"| Served: {obs.info.incidents_served}/{obs.info.incidents_total}"
        ),
        (
            f"Avg response: {obs.info.avg_response_time:.2f}m | "
            f"P95: {obs.info.p95_response_time:.2f}m | "
            f"Missed critical: {obs.info.missed_critical}"
        ),
        (
            "Objective hint: prefer severity-aware short-travel dispatches and avoid "
            "unnecessary holds."
        ),
    ]

    free_ambulances = [ambulance for ambulance in obs.ambulances if is_free_ambulance(ambulance.status)]
    busy_ambulances = [ambulance for ambulance in obs.ambulances if not is_free_ambulance(ambulance.status)]

    parts.append("Free ambulances:")
    if free_ambulances:
        for ambulance in free_ambulances:
            parts.append(
                f"- A{ambulance.ambulance_id} at ({ambulance.x}, {ambulance.y}) depot={ambulance.depot_id}"
            )
    else:
        parts.append("- none")

    parts.append("Busy ambulances:")
    if busy_ambulances:
        for ambulance in busy_ambulances:
            parts.append(
                f"- A{ambulance.ambulance_id} eta_free={ambulance.eta_free:.1f}m at ({ambulance.x}, {ambulance.y})"
            )
    else:
        parts.append("- none")

    parts.append("Visible incident slots (slots can reorder after every step):")
    visible_any = False
    for slot_index, incident in enumerate(obs.incidents):
        if incident.incident_id == -1:
            continue
        visible_any = True
        weight = SEVERITY_WEIGHTS.get(incident.severity, 0.0)
        parts.append(
            f"- slot {slot_index} -> incident {incident.incident_id} | "
            f"{incident.severity} (w={weight:.1f}) | pos=({incident.x}, {incident.y}) "
            f"| waiting={incident.time_since_reported:.1f}m"
        )
    if not visible_any:
        parts.append("- none")

    parts.append("Travel times from free ambulances to visible slots:")
    if visible_any and free_ambulances:
        for ambulance in free_ambulances:
            entries: List[str] = []
            for slot_index, incident in enumerate(obs.incidents):
                if incident.incident_id == -1:
                    continue
                travel_time = obs.travel_times[ambulance.ambulance_id][slot_index]
                entries.append(f"slot {slot_index}: {travel_time:.1f}m")
            joined = ", ".join(entries) if entries else "no visible incidents"
            parts.append(f"- A{ambulance.ambulance_id}: {joined}")
    else:
        parts.append("- none")

    parts.append("Valid actions:")
    for action in valid_action_descriptions(obs):
        parts.append(f"- {action['action_index']}: {action['description']}")

    parts.append(
        'Output ONLY a single JSON object with these exact keys and no extra text:\n'
        '{"action_index": 0, "action_type": "assign", "ambulance_id": 0, '
        '"incident_slot": 0, "incident_id": 0, "justification": "why this dispatch is best", '
        '"confidence": 0.8}'
    )
    return "\n".join(parts)


def build_training_prompt(obs: EmergencyResponseAllocationObservation) -> str:
    return f"{SYSTEM_PROMPT}\n\n{format_observation(obs)}"


def policy_action_index(
    policy: str,
    simulator: ERASSimulator,
    rng: random.Random,
) -> int:
    if policy == "random":
        return random_policy(simulator, rng)
    if policy == "severity_first":
        return severity_first_policy(simulator)
    return nearest_ambulance_policy(simulator)


def action_completion_json(
    action_index: int,
    obs: EmergencyResponseAllocationObservation,
) -> str:
    action = describe_action_index(action_index, obs)
    justification: str
    if action["action_type"] == "hold":
        justification = (
            f"Hold ambulance {action['ambulance_id']} in reserve because that is the preferred valid action."
        )
    else:
        justification = (
            f"Dispatch ambulance {action['ambulance_id']} to incident {action['incident_id']} "
            f"with {action['severity']} severity and {action['travel_time']:.1f}m travel time."
        )
    payload = {
        "action_index": action["action_index"],
        "action_type": action["action_type"],
        "ambulance_id": action["ambulance_id"],
        "incident_slot": action["incident_slot"],
        "incident_id": action["incident_id"],
        "justification": justification,
        "confidence": 0.8,
    }
    return json.dumps(payload)


def build_prompt_examples(
    *,
    dataset_episodes: int,
    rollout_steps: int,
    collection_policy: str,
    seed: int,
) -> List[Dict[str, str]]:
    examples: List[Dict[str, str]] = []
    rng = random.Random(seed)

    for episode_idx in range(dataset_episodes):
        episode_seed = seed + episode_idx
        simulator = ERASSimulator()
        obs = simulator.reset(seed=episode_seed)
        history_actions: List[int] = []

        for _step_idx in range(rollout_steps):
            if obs.done:
                break
            if not any(obs.valid_action_mask):
                break

            next_action_index = policy_action_index(collection_policy, simulator, rng)
            examples.append(
                {
                    "prompt": build_training_prompt(obs),
                    "episode_seed": str(episode_seed),
                    "history_actions": json.dumps(history_actions),
                    "reference_action": action_completion_json(next_action_index, obs),
                    "event_type": obs.event_type,
                    "current_sim_time": f"{obs.info.current_sim_time:.4f}",
                }
            )
            history_actions.append(next_action_index)
            obs = simulator.step(next_action_index)

    return examples


def content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: List[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                if isinstance(part.get("text"), str):
                    parts.append(part["text"])
                elif isinstance(part.get("content"), str):
                    parts.append(part["content"])
        return "".join(parts).strip()
    return str(content).strip()


def completion_to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion.strip()
    if isinstance(completion, dict):
        return content_to_text(completion.get("content", ""))
    if isinstance(completion, list):
        for item in reversed(completion):
            if isinstance(item, dict) and "content" in item:
                text = content_to_text(item["content"])
                if text:
                    return text
            if isinstance(item, str) and item.strip():
                return item.strip()
    return str(completion).strip()


def _strip_js_comments(text: str) -> str:
    text = re.sub(r"//[^\n]*", "", text)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    return text


def _normalize_jsonish_text(text: str) -> str:
    text = _strip_js_comments(text)
    text = re.sub(r"(?<=:\s)\bNone\b", "null", text)
    text = re.sub(r"(?<=:\s)\bTrue\b", "true", text)
    text = re.sub(r"(?<=:\s)\bFalse\b", "false", text)
    text = re.sub(r'"([^"\n]+?):"\s*,', r'"\1": "",', text)
    return text


def _repair_truncated_json(text: str) -> Optional[str]:
    s = text.strip()
    if not s.startswith("{"):
        return None

    s = re.sub(r',\s*"[^"\n]*$', "", s)
    s = re.sub(r',\s*"[^"\n]*"\s*:\s*$', "", s)

    in_string = False
    escape = False
    for ch in s:
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string

    if in_string:
        s += '"'

    open_braces = s.count("{") - s.count("}")
    open_brackets = s.count("[") - s.count("]")
    s += "]" * max(0, open_brackets)
    s += "}" * max(0, open_braces)

    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return s
    except json.JSONDecodeError:
        pass

    s = re.sub(r",\s*([}\]])", r"\1", s)
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return s
    except json.JSONDecodeError:
        pass
    return None


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    stripped = _normalize_jsonish_text(text).strip()
    fence_prefix = "```"
    if stripped.startswith(fence_prefix) and stripped.endswith(fence_prefix):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            stripped = "\n".join(lines[1:-1]).strip()

    candidates: List[str] = [stripped]
    start = stripped.find("{")
    while start != -1:
        depth = 0
        for idx in range(start, len(stripped)):
            char = stripped[idx]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(stripped[start : idx + 1])
                    break
        start = stripped.find("{", start + 1)

    first_brace = stripped.find("{")
    if first_brace != -1:
        repaired = _repair_truncated_json(stripped[first_brace:])
        if repaired is not None:
            candidates.append(repaired)

    candidates.sort(key=len, reverse=True)
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def coerce_optional_int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return int(float(text))
        except ValueError:
            return None
    return None


def normalize_optional_string(value: Any) -> Optional[str]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        value = value.strip()
        return value or None
    if isinstance(value, (int, float)):
        return str(value)
    return compact_preview(value, 80)


def normalize_action_type(raw_action_type: Any) -> Optional[str]:
    if not isinstance(raw_action_type, str):
        return None

    candidate = raw_action_type.strip().lower()
    if candidate in ACTION_TYPE_ALIASES:
        return ACTION_TYPE_ALIASES[candidate]

    candidate = re.sub(r"[^a-z0-9]+", "_", candidate).strip("_")
    if candidate in ACTION_TYPE_ALIASES:
        return ACTION_TYPE_ALIASES[candidate]

    heuristics = [
        (("assign",), "assign"),
        (("dispatch",), "assign"),
        (("alloc",), "assign"),
        (("send",), "assign"),
        (("hold",), "hold"),
        (("wait",), "hold"),
        (("reserve",), "hold"),
        (("noop",), "hold"),
    ]
    for fragments, normalized in heuristics:
        if all(fragment in candidate for fragment in fragments):
            return normalized
    return None


def parse_action_completion(text: str) -> Optional[ParsedDispatchAction]:
    payload = extract_json_object(text)
    if payload is not None:
        action_index = coerce_optional_int(
            get_payload_value(payload, "action_index", "index", "action")
        )
        action_type = normalize_action_type(
            get_payload_value(payload, "action_type", "decision_type", "type")
        )
        ambulance_id = coerce_optional_int(
            get_payload_value(payload, "ambulance_id", "ambulance", "unit_id", "unit")
        )
        incident_slot = coerce_optional_int(
            get_payload_value(payload, "incident_slot", "slot", "slot_index", "incident_index")
        )
        incident_id = coerce_optional_int(
            get_payload_value(payload, "incident_id", "call_id", "job_id")
        )
        confidence = get_payload_value(payload, "confidence")
        if confidence is None:
            confidence = 0.5
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.5

        justification = normalize_optional_string(
            get_payload_value(payload, "justification", "reasoning", "reason", "rationale")
        )
        return ParsedDispatchAction(
            action_index=action_index,
            action_type=action_type,
            ambulance_id=ambulance_id,
            incident_slot=incident_slot,
            incident_id=incident_id,
            justification=justification,
            confidence=min(1.0, max(0.0, float(confidence))),
        )

    action_index_match = re.search(
        r'["\']?action_index["\']?\s*:\s*([0-9]+)',
        text,
        re.IGNORECASE,
    )
    ambulance_match = re.search(
        r'["\']?ambulance_id["\']?\s*:\s*([0-9]+)',
        text,
        re.IGNORECASE,
    )
    slot_match = re.search(
        r'["\']?(?:incident_slot|slot|slot_index)["\']?\s*:\s*([0-9]+)',
        text,
        re.IGNORECASE,
    )
    incident_id_match = re.search(
        r'["\']?incident_id["\']?\s*:\s*([0-9]+)',
        text,
        re.IGNORECASE,
    )
    action_type_match = re.search(
        r'["\']?(?:action_type|decision_type|type)["\']?\s*:\s*["\']([^"\']+)["\']',
        text,
        re.IGNORECASE,
    )

    if not any(
        (
            action_index_match,
            ambulance_match,
            slot_match,
            incident_id_match,
            action_type_match,
        )
    ):
        return None

    return ParsedDispatchAction(
        action_index=int(action_index_match.group(1)) if action_index_match else None,
        action_type=normalize_action_type(action_type_match.group(1)) if action_type_match else None,
        ambulance_id=int(ambulance_match.group(1)) if ambulance_match else None,
        incident_slot=int(slot_match.group(1)) if slot_match else None,
        incident_id=int(incident_id_match.group(1)) if incident_id_match else None,
    )


def resolve_dispatch_action(
    parsed: ParsedDispatchAction,
    obs: EmergencyResponseAllocationObservation,
) -> Optional[EmergencyResponseAllocationAction]:
    mask = obs.valid_action_mask

    if parsed.action_index is not None:
        if 0 <= parsed.action_index < ACTION_DIM and mask[parsed.action_index]:
            return EmergencyResponseAllocationAction(action_index=parsed.action_index)

    incident_slot = parsed.incident_slot
    if incident_slot is None and parsed.incident_id is not None:
        for slot_index, incident_id in enumerate(obs.visible_incident_ids):
            if incident_id == parsed.incident_id:
                incident_slot = slot_index
                break

    if parsed.action_type == "hold":
        ambulance_id = parsed.ambulance_id
        if ambulance_id is None:
            hold_candidates = [
                idx - HOLD_ACTION_OFFSET
                for idx in range(HOLD_ACTION_OFFSET, ACTION_DIM)
                if mask[idx]
            ]
            if len(hold_candidates) == 1:
                ambulance_id = hold_candidates[0]
        if ambulance_id is None:
            return None
        hold_index = HOLD_ACTION_OFFSET + ambulance_id
        if 0 <= ambulance_id < NUM_AMBULANCES and mask[hold_index]:
            return EmergencyResponseAllocationAction(action_index=hold_index)
        return None

    if parsed.ambulance_id is not None and incident_slot is not None:
        if 0 <= parsed.ambulance_id < NUM_AMBULANCES and 0 <= incident_slot < MAX_OBSERVABLE_INCIDENTS:
            action_index = parsed.ambulance_id * MAX_OBSERVABLE_INCIDENTS + incident_slot
            if mask[action_index]:
                return EmergencyResponseAllocationAction(action_index=action_index)

    if parsed.ambulance_id is not None and incident_slot is None:
        candidates = [
            idx
            for idx in range(HOLD_ACTION_OFFSET)
            if mask[idx] and idx // MAX_OBSERVABLE_INCIDENTS == parsed.ambulance_id
        ]
        if len(candidates) == 1:
            return EmergencyResponseAllocationAction(action_index=candidates[0])

    if parsed.ambulance_id is None and incident_slot is not None:
        candidates = [
            idx
            for idx in range(HOLD_ACTION_OFFSET)
            if mask[idx] and idx % MAX_OBSERVABLE_INCIDENTS == incident_slot
        ]
        if len(candidates) == 1:
            return EmergencyResponseAllocationAction(action_index=candidates[0])

    assign_candidates = [idx for idx in range(HOLD_ACTION_OFFSET) if mask[idx]]
    if len(assign_candidates) == 1:
        return EmergencyResponseAllocationAction(action_index=assign_candidates[0])

    return None


def decode_history_actions(history_actions: Optional[str]) -> List[int]:
    if not history_actions:
        return []
    raw_actions = json.loads(history_actions)
    return [
        int(action_index)
        for action_index in raw_actions
        if isinstance(action_index, (int, float, str))
    ]


def normalise_column(values: Any, length: int) -> List[Any]:
    if values is None:
        return [None] * length
    if isinstance(values, list):
        if len(values) == length:
            return values
        if len(values) == 1:
            return values * length
        return values[:length] + [None] * max(0, length - len(values))
    return [values] * length


class OpenEnvReward:
    """Reward function compatible with TRL GRPOTrainer."""

    def __init__(
        self,
        *,
        reward_backend: str,
        base_url: str,
        invalid_action_penalty: float = INVALID_ACTION_PENALTY,
        environment_error_penalty: float = ENVIRONMENT_ERROR_PENALTY,
    ) -> None:
        self.__name__ = "openenv_reward"
        self.reward_backend = reward_backend
        self.base_url = base_url
        self.invalid_action_penalty = invalid_action_penalty
        self.environment_error_penalty = environment_error_penalty

    def __call__(
        self,
        completions: List[Any],
        episode_seed: Optional[List[str]] = None,
        history_actions: Optional[List[str]] = None,
        **_: Any,
    ) -> List[float]:
        seeds = normalise_column(episode_seed, len(completions))
        histories = normalise_column(history_actions, len(completions))
        rewards: List[float] = []

        for completion, current_seed, current_history in zip(
            completions,
            seeds,
            histories,
        ):
            parsed = parse_action_completion(completion_to_text(completion))
            if parsed is None:
                rewards.append(self.invalid_action_penalty)
                continue

            try:
                if self.reward_backend == "remote":
                    reward = self._score_remote(parsed, current_seed, current_history)
                else:
                    reward = self._score_local(parsed, current_seed, current_history)
            except Exception:
                reward = self.environment_error_penalty
            rewards.append(float(reward))

        return rewards

    def _score_local(
        self,
        parsed: ParsedDispatchAction,
        episode_seed: Optional[str],
        history_actions: Optional[str],
    ) -> float:
        simulator = ERASSimulator()
        seed = int(episode_seed) if episode_seed is not None else None
        obs = simulator.reset(seed=seed)
        for previous_action_index in decode_history_actions(history_actions):
            obs = simulator.step(previous_action_index)
            if obs.done:
                return float(obs.reward or 0.0)

        action = resolve_dispatch_action(parsed, obs)
        if action is None:
            return self.invalid_action_penalty

        obs = simulator.step(action.action_index)
        return float(obs.reward or 0.0)

    def _score_remote(
        self,
        parsed: ParsedDispatchAction,
        episode_seed: Optional[str],
        history_actions: Optional[str],
    ) -> float:
        with EmergencyResponseAllocationEnv(base_url=self.base_url).sync() as env:
            result = env.reset(seed=int(episode_seed) if episode_seed is not None else None)
            obs = result.observation
            for previous_action_index in decode_history_actions(history_actions):
                result = env.step(EmergencyResponseAllocationAction(action_index=previous_action_index))
                obs = result.observation
                if result.done:
                    return float(result.reward or 0.0)

            action = resolve_dispatch_action(parsed, obs)
            if action is None:
                return self.invalid_action_penalty

            result = env.step(action)
            if result.reward is not None:
                return float(result.reward)
            return float(result.observation.reward or 0.0)


def is_numeric_log_value(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def available_numeric_log_keys(log_history: Sequence[Dict[str, Any]]) -> List[str]:
    keys = {
        key
        for entry in log_history
        if isinstance(entry, dict)
        for key, value in entry.items()
        if key != "step" and is_numeric_log_value(value)
    }
    return sorted(keys)


def extract_log_series(
    log_history: Sequence[Dict[str, Any]],
    key: Optional[str],
) -> List[Tuple[float, float]]:
    if not key:
        return []

    series: List[Tuple[float, float]] = []
    synthetic_step = 0
    for entry in log_history:
        if not isinstance(entry, dict) or key not in entry:
            continue
        value = entry.get(key)
        if not is_numeric_log_value(value):
            continue

        raw_step = entry.get("step")
        if is_numeric_log_value(raw_step):
            step = float(raw_step)
        else:
            synthetic_step += 1
            step = float(synthetic_step)
        series.append((step, float(value)))

    return series


def select_reward_key(log_history: Sequence[Dict[str, Any]]) -> Optional[str]:
    numeric_keys = available_numeric_log_keys(log_history)
    reward_keys = [key for key in numeric_keys if "reward" in key.lower()]
    if not reward_keys:
        return None

    preferred = [
        "reward",
        "mean_reward",
        "reward_mean",
        "rewards/openenv_reward",
        "rewards/openenv_reward_reward",
    ]
    lowered = {key.lower(): key for key in reward_keys}
    for key in preferred:
        if key in lowered:
            return lowered[key]

    reward_keys.sort(key=lambda key: ("/" in key, len(key), key))
    return reward_keys[0]


def select_metric_key(
    log_history: Sequence[Dict[str, Any]],
    *,
    reward_key: Optional[str],
    requested_key: Optional[str] = None,
) -> Optional[str]:
    numeric_keys = available_numeric_log_keys(log_history)
    if requested_key:
        if requested_key not in numeric_keys:
            available = ", ".join(numeric_keys) or "none"
            raise ValueError(
                f"Requested plot metric '{requested_key}' was not logged. "
                f"Available numeric keys: {available}"
            )
        return requested_key

    excluded = {
        "epoch",
        "loss",
        "learning_rate",
        "step",
        "total_flos",
        "train_loss",
        "train_runtime",
        "train_samples_per_second",
        "train_steps_per_second",
    }
    if reward_key:
        excluded.add(reward_key)

    preferred = [
        "kl",
        "objective/kl",
        "completion_length",
        "mean_completion_length",
        "grad_norm",
        "entropy",
        "accuracy",
        "learning_rate",
        "epoch",
    ]
    numeric_set = set(numeric_keys)
    for key in preferred:
        if key in numeric_set and key not in excluded:
            return key

    candidates = [
        key for key in numeric_keys if key not in excluded and "reward" not in key.lower()
    ]
    if candidates:
        return candidates[0]

    for fallback in ("learning_rate", "epoch"):
        if fallback in numeric_set:
            return fallback
    return None


def save_plot(
    path: Path,
    *,
    series: Sequence[Tuple[float, float]],
    title: str,
    ylabel: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    if series:
        x_values, y_values = zip(*series)
        ax.plot(x_values, y_values, marker="o", linewidth=1.8)
    else:
        ax.text(
            0.5,
            0.5,
            "No logged data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_training_plots(
    log_history: Sequence[Dict[str, Any]],
    output_dir: str | Path,
    metric_key: Optional[str] = None,
) -> Dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    reward_key = select_reward_key(log_history)
    selected_metric_key = select_metric_key(
        log_history,
        reward_key=reward_key,
        requested_key=metric_key,
    )

    loss_series = extract_log_series(log_history, "loss")
    reward_series = extract_log_series(log_history, reward_key)
    metric_series = extract_log_series(log_history, selected_metric_key)

    loss_path = output_path / "training_loss.png"
    reward_path = output_path / "training_reward.png"
    metric_path = output_path / "training_metric.png"
    dashboard_path = output_path / "training_dashboard.png"
    manifest_path = output_path / "training_plot_manifest.json"

    save_plot(loss_path, series=loss_series, title="Training Loss", ylabel="Loss")
    save_plot(
        reward_path,
        series=reward_series,
        title=f"Training Reward ({reward_key or 'not logged'})",
        ylabel="Reward",
    )
    save_plot(
        metric_path,
        series=metric_series,
        title=f"Training Metric ({selected_metric_key or 'not logged'})",
        ylabel=selected_metric_key or "Metric",
    )

    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    plot_specs = [
        (axes[0], loss_series, "Training Loss", "Loss"),
        (axes[1], reward_series, f"Training Reward ({reward_key or 'not logged'})", "Reward"),
        (
            axes[2],
            metric_series,
            f"Training Metric ({selected_metric_key or 'not logged'})",
            selected_metric_key or "Metric",
        ),
    ]
    for axis, series, title, ylabel in plot_specs:
        if series:
            x_values, y_values = zip(*series)
            axis.plot(x_values, y_values, marker="o", linewidth=1.8)
        else:
            axis.text(
                0.5,
                0.5,
                "No logged data available",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
        axis.set_title(title)
        axis.set_xlabel("Step")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(dashboard_path, dpi=150)
    plt.close(fig)

    manifest = {
        "available_numeric_keys": available_numeric_log_keys(log_history),
        "reward_key": reward_key,
        "metric_key": selected_metric_key,
        "plots": {
            "loss": str(loss_path),
            "reward": str(reward_path),
            "metric": str(metric_path),
            "dashboard": str(dashboard_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest["plots"]


def run_dry_run_preview(
    examples: Sequence[Dict[str, str]],
    reward_fn: OpenEnvReward,
    output_dir: str,
) -> None:
    if not examples:
        raise ValueError("No training prompts were generated for the dry run.")

    sample = examples[0]
    sample_reward = reward_fn(
        completions=[[{"role": "assistant", "content": sample["reference_action"]}]],
        episode_seed=[sample["episode_seed"]],
        history_actions=[sample["history_actions"]],
    )[0]

    print(f"Built {len(examples)} prompt states.")
    print(f"Output directory: {Path(output_dir)}")
    print(f"Sample episode seed: {sample['episode_seed']}")
    print(f"Sample reward for reference action: {sample_reward:+.3f}")
    print("\nSample prompt:\n")
    print(sample["prompt"])


def resolve_torch_runtime() -> Dict[str, Any]:
    import torch

    use_cuda = torch.cuda.is_available()
    bf16 = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)()) if use_cuda else False
    dtype = torch.bfloat16 if bf16 else (torch.float16 if use_cuda else torch.float32)
    return {
        "use_cuda": use_cuda,
        "device": "cuda:0" if use_cuda else "cpu",
        "dtype": dtype,
        "bf16": bf16,
        "fp16": use_cuda and not bf16,
        "device_name": torch.cuda.get_device_name(0) if use_cuda else "cpu",
    }


def _guard_invalid_torchao_version() -> None:
    import functools
    import importlib.metadata as importlib_metadata
    import sys
    from packaging.version import InvalidVersion, Version

    if getattr(importlib_metadata, "_openenv_torchao_guard_installed", False):
        metadata_guard_installed = True
    else:
        original_version = importlib_metadata.version

        def guarded_version(distribution_name: str) -> str:
            version = original_version(distribution_name)
            if distribution_name.lower() == "torchao":
                try:
                    Version(version)
                except InvalidVersion as exc:
                    raise importlib_metadata.PackageNotFoundError(
                        f"Malformed torchao version metadata: {version!r}"
                    ) from exc
            return version

        importlib_metadata.version = guarded_version
        importlib_metadata._openenv_torchao_guard_installed = True
        metadata_guard_installed = False

    import_utils = sys.modules.get("transformers.utils.import_utils")
    if import_utils is not None and not getattr(import_utils, "_openenv_torchao_guard_installed", False):
        original_is_package_available = import_utils._is_package_available

        def guarded_is_package_available(pkg_name: str, return_version: bool = False):
            if pkg_name != "torchao":
                return original_is_package_available(pkg_name, return_version=return_version)
            is_available, package_version = original_is_package_available(
                pkg_name, return_version=True
            )
            if not is_available:
                return (False, package_version) if return_version else (False, None)
            try:
                Version(package_version)
            except InvalidVersion:
                return (False, "0") if return_version else (False, None)
            return (True, package_version) if return_version else (True, None)

        min_version = getattr(import_utils, "TORCHAO_MIN_VERSION", "0")

        @functools.lru_cache
        def guarded_is_torchao_available(min_version_override: str = min_version) -> bool:
            is_available, package_version = guarded_is_package_available(
                "torchao",
                return_version=True,
            )
            if not is_available:
                return False
            try:
                return Version(package_version) >= Version(min_version_override)
            except InvalidVersion:
                return False

        if hasattr(import_utils.is_torchao_available, "cache_clear"):
            import_utils.is_torchao_available.cache_clear()
        import_utils._is_package_available = guarded_is_package_available
        import_utils.is_torchao_available = guarded_is_torchao_available
        import_utils._openenv_torchao_guard_installed = True

        transformers_utils = sys.modules.get("transformers.utils")
        if transformers_utils is not None:
            transformers_utils.is_torchao_available = guarded_is_torchao_available

    if metadata_guard_installed and import_utils is None:
        return


def _guard_partial_vllm_install() -> None:
    import functools
    import importlib

    try:
        import trl.import_utils as trl_import_utils
    except Exception:
        return

    if getattr(trl_import_utils, "_openenv_vllm_guard_installed", False):
        return

    def _has_usable_vllm() -> bool:
        try:
            importlib.import_module("vllm")
            importlib.import_module("vllm.distributed.device_communicators.pynccl")
            importlib.import_module("vllm.distributed.utils")
        except Exception:
            return False
        return True

    @functools.lru_cache
    def guarded_is_vllm_available(*args: Any, **kwargs: Any) -> bool:
        return _has_usable_vllm()

    if hasattr(trl_import_utils.is_vllm_available, "cache_clear"):
        trl_import_utils.is_vllm_available.cache_clear()
    trl_import_utils.is_vllm_available = guarded_is_vllm_available
    trl_import_utils._openenv_vllm_guard_installed = True


def load_model_artifacts(
    model_id: str,
    *,
    trust_remote_code: bool,
):
    _guard_invalid_torchao_version()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    runtime = resolve_torch_runtime()
    print(f"Loading tokenizer for {model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model for {model_id} ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
        torch_dtype=runtime["dtype"],
    )
    model = model.to(runtime["device"] if runtime["use_cuda"] else "cpu")
    return tokenizer, model


def build_openenv_reward(args: argparse.Namespace) -> OpenEnvReward:
    return OpenEnvReward(
        reward_backend=args.reward_backend,
        base_url=args.base_url,
    )


def prepare_prompt_examples(args: argparse.Namespace) -> Dict[str, Any]:
    examples = build_prompt_examples(
        dataset_episodes=args.dataset_episodes,
        rollout_steps=args.rollout_steps,
        collection_policy=args.collection_policy,
        seed=args.seed,
    )
    return {"examples": examples}


def build_grpo_config(
    args: argparse.Namespace,
    runtime: Dict[str, Any],
):
    import inspect

    _guard_invalid_torchao_version()
    _guard_partial_vllm_install()
    from trl import GRPOConfig

    config_kwargs = {
        "output_dir": args.output_dir,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_generations": args.num_generations,
        "max_completion_length": args.max_completion_length,
        "max_prompt_length": args.max_prompt_length,
        "num_train_epochs": args.num_train_epochs,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "bf16": runtime["bf16"],
        "fp16": runtime["fp16"],
        "report_to": "none",
        "remove_unused_columns": False,
    }
    supported_params = set(inspect.signature(GRPOConfig.__init__).parameters)

    if (
        "max_length" in supported_params
        and "max_prompt_length" not in supported_params
        and "max_completion_length" not in supported_params
    ):
        config_kwargs["max_length"] = args.max_prompt_length + args.max_completion_length

    filtered_kwargs = {
        key: value for key, value in config_kwargs.items() if key in supported_params
    }
    skipped = sorted(set(config_kwargs) - set(filtered_kwargs))
    if skipped:
        print(
            "GRPOConfig compatibility: skipping unsupported fields "
            f"{', '.join(skipped)}"
        )
    return GRPOConfig(**filtered_kwargs)


def build_grpo_trainer(
    *,
    model: Any,
    tokenizer: Any,
    reward_func: Any,
    train_dataset: Any,
    args: argparse.Namespace,
    runtime: Dict[str, Any],
):
    import inspect

    _guard_invalid_torchao_version()
    _guard_partial_vllm_install()
    from trl import GRPOTrainer

    config = build_grpo_config(args, runtime)
    trainer_signature = inspect.signature(GRPOTrainer.__init__).parameters
    trainer_kwargs = {
        "model": model,
        "args": config,
        "train_dataset": train_dataset,
    }
    if "reward_funcs" in trainer_signature:
        trainer_kwargs["reward_funcs"] = reward_func
    elif "reward_function" in trainer_signature:
        trainer_kwargs["reward_function"] = reward_func
    else:
        raise RuntimeError("Unsupported GRPOTrainer signature: no reward function parameter found.")

    if "processing_class" in trainer_signature:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_signature:
        trainer_kwargs["tokenizer"] = tokenizer

    return GRPOTrainer(**trainer_kwargs)


def generate_action_with_model(
    model: Any,
    tokenizer: Any,
    prompt_or_observation: str | EmergencyResponseAllocationObservation,
    *,
    max_new_tokens: int = DEFAULT_COMPLETION_TOKEN_BUDGET,
    temperature: float = 0.2,
    top_p: float = 0.9,
    do_sample: bool = True,
) -> Dict[str, Any]:
    import torch

    if isinstance(prompt_or_observation, EmergencyResponseAllocationObservation):
        prompt = build_training_prompt(prompt_or_observation)
    else:
        prompt = str(prompt_or_observation)

    model_device = getattr(model, "device", None)
    if model_device is None:
        model_device = resolve_torch_runtime()["device"]

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(model_device) for key, value in inputs.items()}
    prompt_tokens = inputs["input_ids"].shape[1]

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "temperature": temperature,
        "top_p": top_p,
        "pad_token_id": tokenizer.pad_token_id,
    }
    with torch.no_grad():
        output_ids = model.generate(**inputs, **generation_kwargs)

    new_tokens = output_ids[0][prompt_tokens:]
    response_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    parsed = parse_action_completion(response_text)
    return {
        "prompt": prompt,
        "response_text": response_text,
        "parsed_action": parsed,
    }


def run_training(args: argparse.Namespace) -> Dict[str, Any]:
    random.seed(args.seed)

    if args.load_model_only:
        runtime = resolve_torch_runtime()
        tokenizer, model = load_model_artifacts(
            args.model_id,
            trust_remote_code=args.trust_remote_code,
        )
        device = getattr(model, "device", "unknown")
        print(f"Model ready: {args.model_id}")
        print(f"Tokenizer vocab size: {len(tokenizer)}")
        print(f"Model device: {device}")
        print(f"Runtime device name: {runtime['device_name']}")
        return {
            "args": args,
            "runtime": runtime,
            "tokenizer": tokenizer,
            "model": model,
        }

    prompt_data = prepare_prompt_examples(args)
    examples = prompt_data["examples"]
    reward_fn = build_openenv_reward(args)

    if args.dry_run:
        run_dry_run_preview(examples, reward_fn, args.output_dir)
        return {
            "args": args,
            "runtime": None,
            "examples": examples,
            "reward_fn": reward_fn,
        }

    runtime = resolve_torch_runtime()

    from datasets import Dataset

    train_dataset = Dataset.from_list(examples)
    tokenizer, model = load_model_artifacts(
        args.model_id,
        trust_remote_code=args.trust_remote_code,
    )

    print(
        f"Training runtime: device={runtime['device']} "
        f"name={runtime['device_name']} dtype={runtime['dtype']}"
    )
    print(
        "OpenEnv reward: "
        f"backend={args.reward_backend} examples={len(examples)} "
        f"collection_policy={args.collection_policy}"
    )

    trainer = build_grpo_trainer(
        model=model,
        tokenizer=tokenizer,
        reward_func=reward_fn,
        train_dataset=train_dataset,
        args=args,
        runtime=runtime,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if args.push_to_hub:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(repo_id=args.push_to_hub, repo_type="model", exist_ok=True)
        print(f"Pushing model to Hugging Face Hub: {args.push_to_hub}")
        api.upload_folder(
            folder_path=args.output_dir,
            repo_id=args.push_to_hub,
            repo_type="model",
            create_pr=False,
        )
        print(f"Model pushed to https://huggingface.co/{args.push_to_hub}")

    plot_paths = save_training_plots(
        trainer.state.log_history,
        args.output_dir,
        metric_key=args.plot_metric_key,
    )
    print("Saved training plots:")
    for plot_name, plot_path in plot_paths.items():
        print(f"  - {plot_name}: {plot_path}")

    return {
        "args": args,
        "runtime": runtime,
        "examples": examples,
        "reward_fn": reward_fn,
        "train_dataset": train_dataset,
        "tokenizer": tokenizer,
        "model": model,
        "trainer": trainer,
        "plot_paths": plot_paths,
    }


def main() -> None:
    run_training(parse_args())


if __name__ == "__main__":
    main()
