---
title: EmailEnv
emoji: 📧
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: apache-2.0
tags:
  - openenv
  - reinforcement-learning
  - email
  - spam-detection
  - reply-generation
  - python
---

# EmailEnv

An **OpenEnv**-compatible RL environment where an AI agent triages incoming
emails and performs three tasks in sequence: **spam detection**, **priority
assignment**, and **reply drafting**. Graders are **deterministic** and based on
hand‑written rubrics, giving **partial-credit reward signals** in the range

## Motivation

Email triage and customer support are high‑value, real‑world workflows that
human agents perform every day. Training RL agents in this setting teaches them
to:

- Filter spam and scams
- Draft polite, on‑policy customer support replies

Skills learned here transfer directly to real customer support tooling and
agentic assistants.

---

## Action & Observation Spaces

### Action Space

Actions are represented by the `Action` model in `models.py`:

| Field        | Type                                           | Description |
|--------------|------------------------------------------------|-------------|
| `type`       | `"classify_spam" \| "set_priority" \| "generate_reply" \| "skip"` | What the agent wants to do for the current email |
| `is_spam`    | `bool \| None`                                 | Required when `type == "classify_spam"` |
| `priority`   | `"low" \| "medium" \| "high" \| None`          | Required when `type == "set_priority"` |
| `reply_text` | `str \| None`                                  | Used when `type == "generate_reply"`; if omitted, the environment may auto‑generate a reply |

### Observation Space

Observations are represented by the `Observation` model in `models.py`:

| Field             | Type   | Description |
|-------------------|--------|-------------|
| `email`           | `Email \| None` | The current email object, or `null` when the inbox is exhausted |
| `task`            | `"spam_classification" \| "email_prioritization" \| "reply_generation"` | Which sub‑task the agent is performing |
| `step_index`      | `int`  | Index of the current step within the episode |
| `total_steps`     | `int`  | Total number of emails in the episode |
| `remaining_emails`| `int`  | How many emails are left to process |

The `Email` model includes fields like `id`, `subject`, `body`, `sender`,
`timestamp`, and `metadata` (which holds private ground‑truth labels used by the
graders).

### State Space

Environment state is tracked by the `EnvState` model:

| Field           | Type      | Description |
|-----------------|-----------|-------------|
| `current_task`  | `EmailTask \| None` | The email task currently being solved |
| `task_number`   | `int`     | Index of the active task in the episode (0–2) |
| `total_reward`  | `float`   | Cumulative reward so far |
| `is_done`       | `bool`    | Whether the episode has ended |
| `history`       | `list[StepResult]` | All past graded steps in the episode |

---

## Tasks

| ID                         | Difficulty | Domain            | What the agent must do |
|----------------------------|-----------|-------------------|-------------------------|
| `spam_classification`      | 🟢 Easy   | Spam Detection    | Decide whether an email is **spam** or **not spam** |
| `email_prioritization`     | 🟡 Medium | Prioritisation    | Assign priority: `low`, `medium`, or `high` |
| `reply_generation`         | 🔴 Hard   | Reply Drafting    | Draft a professional, on‑policy support reply |

Internally, each episode walks through three `EmailTask` scenarios (spam →
priority → reply) and maintains a running total reward.

---

## Reward System

Each step is graded by deterministic rubric‑based graders in `graders.py`.
Rewards are always in `[0.0, 1.0]` and support partial credit.

In simplified form:

```text
Reward = average(criterion_scores)
criterion_scores ∈ {0.0, 1.0}
```

### Spam Grader (Easy)

- `label_correct` – 1.0 if the predicted spam / not‑spam label matches ground truth

### Priority Grader (Medium)

- `exact_priority_match` – 1.0 if the predicted priority equals ground truth
- `off_by_one` – partial credit when the prediction is close (e.g. `medium` vs `high`)

### Reply Grader (Hard)

The reply grader combines multiple binary criteria:

- `relevance` – reply addresses the actual issue
- `tone` – polite, professional, on‑brand
- `completeness` – all key points covered
- `helpfulness` – offers concrete next steps or solutions

The episode ends when all three tasks are completed. Total reward is the sum of
per‑task rewards.

---

## Baseline Scores

Example scores obtained with a simple `gpt-5-nano`‑based agent (deterministic
prompts, temperature 0.0). These are indicative only – your results may differ.

| Task                    | Heuristic Baseline | gpt-5-nano Baseline |
|-------------------------|:------------------:|:-------------------:|
| `spam_classification`   | 1.00               | **1.00**             |
| `email_prioritization`  | 1.00               | **1.00**             |
| `reply_generation`      | 0.70               | **0.70**             |
| **Average**             | 0.90               | **0.90**             |

---

## Setup & Usage

### Install

For local development, install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Run the Server

Start the FastAPI server on port 7860:

```bash
uvicorn server:app --host 0.0.0.0 --port 7860
```

### HTTP API

```bash
# Reset (no body required)
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" -d "{}"

# Reset with explicit task hint (currently ignored but logged)
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task": "spam_classification"}'

# Step – classify spam
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"type": "classify_spam", "is_spam": true}'

# State
curl http://localhost:7860/state

# Health
curl http://localhost:7860/health
```

### Run Inference Agent

`inference.py` contains a simple OpenAI‑powered loop that interacts with the
environment.

```bash
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-5-nano
export OPENAI_API_KEY=sk-...
python inference.py
```

On Windows PowerShell:

```powershell
$Env:API_BASE_URL = "https://api.openai.com/v1"
$Env:MODEL_NAME = "gpt-5-nano"
$Env:OPENAI_API_KEY = "sk-..."
python inference.py
```

### Docker / Hugging Face Spaces

The repository includes a `Dockerfile` suitable for Spaces:

```bash
docker build -t emailenv .
docker run -p 7860:7860 \
  -e API_BASE_URL=https://api.openai.com/v1 \
  -e MODEL_NAME=gpt-5-nano \
  -e OPENAI_API_KEY=sk-... \
  emailenv
```

On Hugging Face, `openenv.yaml` points Spaces at this Dockerfile and exposes
port 7860.

---

## Logging Format

The FastAPI server logs structured events for easier analysis:

```text
API_BASE_URL : https://api.openai.com/v1
MODEL_NAME   : gpt-5-nano
============================================================
[START] task=spam_classification env=emailenv model=gpt-5-nano
[STEP] step=spam_001 action=type=classify_spam reward=1.00 done=false error=null
...
```

These logs make it easy to replay episodes, compute statistics, and compare
agent variants.

---

## Project Structure

```text
EmailEnv/
├── models.py          # Pydantic Action, Observation, State, Reward, EmailTask, EnvState
├── tasks.py           # Task definitions for spam / priority / reply
├── graders.py         # Deterministic rubric-based graders
├── emailenv_class.py  # Core EmailEnv class (reset / step / state)
├── server.py          # FastAPI app: /web /reset /step /state /health
├── client.py          # Optional HTTP client helpers
├── inference.py       # OpenAI-powered agent loop
├── openenv.yaml       # OpenEnv manifest (spec_version: 1)
├── pyproject.toml     # Project metadata (not required for Docker build)
├── requirements.txt   # Runtime dependencies
└── Dockerfile         # Container build (used by HF Spaces)
```

---

## License

Apache-2.0
