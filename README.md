---
title: Emergency Response Allocation Environment Server
emoji: 🚑
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - reinforcement-learning
  - emergency-response
---

# Emergency Response Allocation Environment

Emergency Response Allocation System (ERAS) is an OpenEnv-compatible reinforcement
learning environment for ambulance dispatch. It simulates a 20x20 urban grid,
stochastic emergency demand, time-varying traffic, ambulance availability, and
severity-aware reward shaping.

The simulator is event-driven: the agent is asked to act only when there is a
meaningful dispatch decision, namely when at least one pending incident and at
least one free ambulance are both available.

## Implemented Scenario

- Map: 20x20 abstract city grid
- Ambulances: 5
- Depots: 3 at northwest, center, and southeast anchors
- Hospitals: 2 at northwest and southeast quadrants
- Visible incidents: top 10 pending incidents, sorted by severity then wait time
- Episode horizon: 24 simulated hours
- Demand model: piecewise Poisson arrivals with morning/evening peaks
- Traffic model:
  - Peak: 8-10am and 5-7pm -> 1.8x travel
  - Off-peak -> 1.0x travel
  - Night: 10pm-6am -> 0.7x travel

## Observation Space

Each observation contains a fixed-size `observation_vector` of length `111`:

- Ambulances: `5 x 4`
  - `x`
  - `y`
  - `is_free`
  - `eta_free`
- Incidents: `10 x 4`
  - `x`
  - `y`
  - `severity_code`
  - `time_since_reported`
- Travel times: `5 x 10`
  - ambulance-to-incident travel matrix
- Time of day: `1`

The observation also includes:

- `valid_action_mask`: boolean mask over the discrete action space
- `ambulances`: structured debug view of ambulance states
- `incidents`: structured debug view of visible incidents
- `travel_times`: structured 5x10 matrix
- `info`: metrics such as average response time, coverage, utilization, and
  missed critical cases

## Action Space

The action space is discrete with `55` actions:

- `0-49`: assign `ambulance_i` to visible `incident_slot_j`
- `50-54`: hold action for ambulance `0-4`

Assignment indexing is:

```text
action_index = ambulance_id * 10 + incident_slot
```

Hold indexing is:

```text
action_index = 50 + ambulance_id
```

Only valid actions are enabled in `valid_action_mask`.

## Reward

The environment uses shaped rewards:

- Immediate dispatch signal:
  - `-w1 * severity_weight * estimated_travel_time`
- Delayed arrival signal:
  - `+w4 * severity_weight * (1 / response_time)`
  - `-w2 * severity_weight * response_time`
- Hold penalty:
  - `-w3 * number_of_idle_free_ambulances` when pending incidents exist

Default weights:

- `w1 = 1.0`
- `w2 = 0.8`
- `w3 = 0.5`
- `w4 = 1.2`

## API Endpoints

The OpenEnv app exposes:

- `GET /health`
- `GET /schema`
- `GET /state`
- `POST /reset`
- `POST /step`
- `WS /ws`

`/state` and `/schema` are overridden to expose the ERAS-specific state model
rather than the generic base OpenEnv state. Persistent episode interaction is
handled through `WS /ws` or the `EmergencyResponseAllocationEnv` client.

## Quick Start

### Run the server with uv

```bash
uv run python -m emergency_response_allocation.server.app --port 8000
```

### Use the synchronous client

```python
from emergency_response_allocation import (
    EmergencyResponseAllocationAction,
    EmergencyResponseAllocationEnv,
)

env = EmergencyResponseAllocationEnv(base_url="http://127.0.0.1:8000").sync()

with env:
    result = env.reset(seed=7)
    action_index = next(
        i for i, ok in enumerate(result.observation.valid_action_mask) if ok
    )
    result = env.step(EmergencyResponseAllocationAction(action_index=action_index))
    state = env.state()
    print(result.reward)
    print(state.current_sim_time)
```

## Baselines

Implemented baselines live in `server/baselines.py`:

- Random allocation
- Nearest ambulance heuristic
- Severity-first greedy heuristic

Example:

```python
from emergency_response_allocation.server.baselines import (
    evaluate_baselines,
    format_comparison_table,
)

results = evaluate_baselines(num_episodes=100)
print(format_comparison_table(results))
```

## Training

The repo now includes a root training harness at `train.py` for local GRPO
experiments against ERAS.

### Install training dependencies

```bash
uv sync --extra train
```

Note: the `train` extra installs a standard PyTorch build. For GPU training,
you may want to install a CUDA-specific PyTorch build following the official
Transformers and PyTorch installation guidance, then re-run `uv sync --extra train`.

### Smoke-test the harness

This path only builds prompt states and checks reward replay, so it is the
fastest way to confirm the environment and script are wired correctly:

```bash
uv run --extra train python train.py --dry-run --dataset-episodes 1 --rollout-steps 2
```

### Run local reward training

This uses the in-process simulator for rewards and does not require a server:

```bash
uv run --extra train python train.py \
  --model-id Qwen/Qwen2.5-1.5B-Instruct \
  --dataset-episodes 12 \
  --rollout-steps 8 \
  --collection-policy nearest \
  --reward-backend local \
  --output-dir training/grpo-output
```

### Run remote reward training through OpenEnv

Start the server in one terminal:

```bash
uv run python -m emergency_response_allocation.server.app --port 8000
```

Then run training in another terminal:

```bash
uv run --extra train python train.py \
  --reward-backend remote \
  --base-url http://127.0.0.1:8000
```

### Helpful flags

- `--load-model-only`: verify model/tokenizer loading without training
- `--dry-run`: build prompts and test the reward function only
- `--push-to-hub <repo-id>`: upload saved artifacts after training
- `--plot-metric-key <metric>`: choose an extra logged metric to plot
- `--collection-policy random|nearest|severity_first`: choose how prompt
  states are generated before GRPO

Training outputs and plots are written under `training/grpo-output` by default.

## Verification

The repo includes endpoint tests that run in the project `uv` environment:

```bash
uv run --extra dev pytest -q
```
