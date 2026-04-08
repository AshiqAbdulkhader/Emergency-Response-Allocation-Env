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

## Mathematical View

ERAS can be viewed as an event-driven dispatch process over a city grid

```text
G = {0, ..., 19} x {0, ..., 19}
```

with internal simulator state

```text
x_t = (B_t, U_t, Q_t, t)
```

where:

- `B_t` is the full ambulance state set
- `U_t` is the full incident set, including pending, dispatched, and resolved cases
- `Q_t` is the future event queue
- `t` is simulated time in minutes

The agent does not consume `x_t` directly. Instead, it receives a fixed-size
observation

```text
o_t = [A_t ; I_t ; vec(T_t) ; h_t] in R^111
```

where:

- `A_t in R^(5x4)` is the ambulance feature block
- `I_t in R^(10x4)` is the visible incident block, padded with zeros
- `T_t in R^(5x10)` is the ambulance-to-incident travel-time matrix
- `h_t in R` is the current time of day in hours

Travel times are computed as

```text
tau((x1, y1), (x2, y2), t)
  = ||(x1, y1) - (x2, y2)||_2 * m(t)
```

where `m(t)` is the traffic multiplier:

- `1.8` during peak traffic windows
- `1.0` off-peak
- `0.7` at night

Because only the top 10 pending incidents are exposed, the agent-facing view is
a priority-truncated representation of the full simulator state.

## Observation Space

Each observation contains a fixed-size `observation_vector` of length `111`.
The layout is:

```text
o_t = [A_t ; I_t ; vec(T_t) ; h_t]
```

with:

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

More explicitly:

```text
A_t[i] = (x_i, y_i, free_i, eta_i)
I_t[j] = (x_j, y_j, s_j, w_j)
T_t[i, j] = tau(ambulance_i, incident_j, t)
```

where:

- `x`, `y` are raw grid coordinates in `[0, 19]`
- `free_i in {0, 1}`
- `eta_i` is remaining time until ambulance `i` is free, in simulated minutes
- `s_j in {0, 1, 2, 3}` with `0=padding`, `1=low`, `2=moderate`, `3=critical`
- `w_j` is time since incident `j` was reported, in simulated minutes

The flattened vector uses this ordering:

- indices `0-19`: ambulance block
- indices `20-59`: incident block
- indices `60-109`: flattened travel-time matrix
- index `110`: time of day in hours `[0, 24)`

Visible incidents are sorted by priority before encoding:

```text
sort key = (-severity_weight, -time_since_reported, incident_id)
```

If fewer than 10 pending incidents exist, the remaining incident slots and
their travel-time entries are padded with zeros.

The observation also includes:

- `valid_action_mask`: boolean mask over the discrete action space
- `ambulances`: structured debug view of ambulance states
- `incidents`: structured debug view of visible incidents
- `travel_times`: structured 5x10 matrix
- `visible_incident_ids`: stable incident ids aligned to the 10 incident slots
- `info`: metrics such as average response time, coverage, utilization, and
  missed critical cases

## Action Space

The action space is discrete with `55` actions:

```text
A = {0, 1, ..., 54}
```

It is partitioned into assignment actions and hold actions:

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

Equivalently:

```text
assign(i, j) = 10i + j
hold(i) = 50 + i
```

with `i in {0, ..., 4}` and `j in {0, ..., 9}`.

The valid action set at decision time `t` is:

```text
V_t =
  {assign(i, j) : ambulance i is free and incident slot j is occupied}
  union
  {hold(i) : ambulance i is free and at least one visible incident exists}
```

`valid_action_mask` is the binary encoding of `V_t` in `{0, 1}^55`.

One important representation detail is that assignment actions target incident
slots, not globally stable incident positions. Slot `j` always refers to the
`j`th incident in the current priority-sorted visible list, and the
`visible_incident_ids` field tells you which underlying incident ids those slots
map to.

After an action is applied, the simulator advances to the next actionable event
rather than the next fixed time tick. In other words, one RL step corresponds
to one dispatch decision followed by event-queue progression.

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
- `GET /tasks`
- `POST /reset`
- `POST /step`
- `POST /grade/{task_id}`
- `WS /ws`

`/state` and `/schema` are overridden to expose the ERAS-specific state model
rather than the generic base OpenEnv state. Persistent episode interaction is
handled through `WS /ws` or the `EmergencyResponseAllocationEnv` client.

`/tasks` enumerates the benchmark tasks used by the root `inference.py`
submission runner. `/grade/{task_id}` converts completed episode metrics into a
normalized task `score` and `reward`, both guaranteed to lie in `[0.0, 1.0]`.

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

## Benchmark Tasks And Graders

The repo now includes three benchmark-ready tasks with deterministic graders:

- `night_shift_balance`:
  low-demand night scenario graded on coverage, utilization, response time, and
  missed critical cases
- `rush_hour_triage`:
  peak-traffic scenario graded on critical response quality, tail latency, and
  coverage
- `citywide_surge`:
  harder surge scenario graded on severity-weighted performance, resilience, and
  coverage under saturation

Task definitions and graders live in `server/evaluation.py`.

Each grader returns:

- `score` in `[0.0, 1.0]`
- `reward` in `[0.0, 1.0]`

The environment's internal shaped episode reward is still unbounded. The grader
maps final episode metrics plus episode reward into a normalized evaluation
reward for benchmark submission.

You can enumerate tasks and grade externally computed metrics through the API:

```bash
curl http://127.0.0.1:8000/tasks
```

```bash
curl -X POST http://127.0.0.1:8000/grade/night_shift_balance \
  -H "Content-Type: application/json" \
  -d '{
    "avg_response_time": 6.5,
    "p95_response_time": 10.2,
    "severity_weighted_response_score": 180.0,
    "coverage_rate": 0.92,
    "ambulance_utilization": [0.61, 0.68, 0.74, 0.66, 0.63],
    "missed_critical": 0,
    "episode_reward": 24.0,
    "step_count": 21
  }'
```

## Inference Submission

The required submission runner is [inference.py](/Users/ashiq/Workspace/Emergency-Response-Allocation-Env/inference.py)
at the repo root. It uses the OpenAI Python client for all LLM calls.

Before submitting, ensure these environment variables are configured:

- `API_BASE_URL`: OpenAI-compatible LLM endpoint
- `MODEL_NAME`: model identifier to use for inference
- `HF_TOKEN`: Hugging Face token or compatible API key
- `LOCAL_IMAGE_NAME`: local Docker image name, only when running with
  `from_docker_image(...)`

Defaults are intentionally provided only for:

- `API_BASE_URL`
- `MODEL_NAME`

The script supports two execution modes:

- If `LOCAL_IMAGE_NAME` is set, it launches the environment with
  `EmergencyResponseAllocationEnv.from_docker_image(...)`
- Otherwise it connects to `OPENENV_URL` and uses an already-running ERAS server

Additional optional variable:

- `OPENENV_URL`: ERAS environment server URL when not using Docker

A starter env file is provided in [.env.example](/Users/ashiq/Workspace/Emergency-Response-Allocation-Env/.env.example).

Example:

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="your-token"
export LOCAL_IMAGE_NAME="emergency-response-allocation:latest"
export OPENENV_URL="http://127.0.0.1:8000"
```

Run the inference harness with:

```bash
uv run python inference.py
```

By default it:

- enumerates all benchmark tasks
- runs one episode per task
- grades each task
- verifies that every task score stays in `[0.0, 1.0]`

The script emits exactly three stdout line types:

- `[START]`
- `[STEP]`
- `[END]`

Each line is a single structured record with fixed field ordering:

```text
[START] task=<task_name> env=<benchmark> model=<model_name>
[STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
[END] success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
```

Formatting rules implemented by the script:

- exactly one `[START]` line at episode begin
- exactly one `[STEP]` line per successful `env.step(...)`
- exactly one `[END]` line after `env.close()`, even if the episode fails
- `reward` and `rewards` use two decimal places
- `done` and `success` are lowercase booleans
- `error` is `null` when there is no step error
- `score` is normalized and clamped to `[0.0, 1.0]`

Example:

```text
[START] task=night_shift_balance env=emergency_response_allocation model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=assign(ambulance=0,incident_slot=0) reward=-2.84 done=false error=null
[END] success=true steps=19 score=0.812 rewards=-2.84,0.00,1.23
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
