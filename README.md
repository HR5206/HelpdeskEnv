---
title: HelpdeskEnv
emoji: 🎫
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
license: apache-2.0
tags:
  - openenv
  - reinforcement-learning
  - multi-agent
  - helpdesk
  - self-improving
  - knowledge-base
  - email
  - python
---

# 🎫 HelpdeskEnv

A **multi-agent IT Helpdesk** OpenEnv environment where specialized AI agents
collaborate to resolve IT support tickets through **tiered escalation**,
**Knowledge Base self-improvement**, and **SLA-driven planning**.

> Built for the [OpenEnv Hackathon](https://openenv.org) — Round 2 evolution of EmailEnv.

---

## 🏆 Hackathon Themes Addressed

| Theme | How It's Implemented |
|-------|---------------------|
| **Multi-Agent Interaction** | 4 agents (Triage, L1, L2, L3) with role-based actions and handoffs |
| **Long-Horizon Planning** | SLA step budgets force efficient multi-step resolution strategies |
| **World Modeling (Knowledge Base)** | Persistent KB with keyword search — agents query before acting |
| **Self-Improving Systems** | L3 writes KB articles → future episodes have more knowledge → scores improve |

---

## 🏗️ Architecture

```text
                    ┌─────────────┐
     Ticket ───────►│   TRIAGE    │ classify: category, priority, tier
                    └──────┬──────┘
                           │ route
              ┌────────────┼────────────┐
              ▼            ▼            ▼
         ┌────────┐   ┌────────┐   ┌────────┐
         │   L1   │   │   L2   │   │   L3   │
         │ Simple │   │ Medium │   │ Expert │
         └───┬────┘   └───┬────┘   └───┬────┘
             │            │            │
        search_kb    apply_fix    apply_complex_fix
        apply_sol    respond      write_kb_entry ◄── SELF-IMPROVEMENT
        respond      escalate↗   respond
        escalate↗
              │            │            │
              └────────────┴────────────┘
                           │
                    ┌──────▼──────┐
                    │ Knowledge   │  Persists across episodes
                    │    Base     │  2 seed + agent-created articles
                    └─────────────┘
```

### Agent Workflow Per Ticket

1. **Triage Agent** classifies the ticket → routes to L1/L2/L3
2. **Support Agent** searches KB → applies fix → responds to customer
3. **L3 only**: writes KB articles for novel issues
4. `respond_to_customer` resolves the ticket → moves to next

### Self-Improvement Loop

```text
Episode 1: KB=2 seed entries → L3 solves novel tickets from scratch → writes 3 KB articles
Episode 2: KB=5 entries → L1/L2 find solutions in KB → resolve faster → higher scores
Episode 5: KB=7+ entries → most tickets have KB matches → near-optimal performance
```

---

## 📋 Tasks (6 Total)

### Round 1 — EmailEnv (preserved)

| Task | Difficulty | Grader | Scoring |
|------|-----------|--------|---------|
| Spam Classification | 🟢 Easy | `grade_spam` | Binary: correct/incorrect |
| Email Prioritization | 🟡 Medium | `grade_priority` | Distance-based: 4 levels (low/medium/high/critical) |
| Reply Generation | 🔴 Hard | `grade_reply` | Weighted: politeness 40% + length 30% + relevance 30% |

### Round 2 — HelpdeskEnv (new)

| Task | Difficulty | Grader | Scoring |
|------|-----------|--------|---------|
| Ticket Triage | 🟡 Medium | `grade_triage` | Category 40% + priority 30% + tier 30% |
| Ticket Resolution | 🔴 Hard | `grade_efficiency` | SLA compliance 60% + escalation efficiency 40% |
| KB Contribution | 🔴 Hard | `grade_kb_contribution` | Relevance 35% + length 30% + specificity 35% |

---

## 🚀 Quick Start

### Install

```bash
pip install -r requirements.txt
```

### Run Integration Tests

```bash
python test_integration.py
```

### Run the Helpdesk Inference (no API key needed)

```bash
python inference.py
```

This runs:
- 3 EmailEnv episodes (spam, priority, reply)
- 3 HelpdeskEnv multi-agent episodes with self-improvement demo

### Start the Server

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

---

## 🔌 API Endpoints

### EmailEnv (Round 1)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST | Reset email environment |
| `/step` | POST | Submit email action |
| `/state` | GET | Get email env state |
| `/health` | GET | Health check |
| `/metadata` | GET | Environment metadata |
| `/schema` | GET | Action/observation schemas |
| `/tasks` | GET | List all 6 tasks |

### HelpdeskEnv (Round 2)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/helpdesk/reset` | POST | Start new episode `{"seed": 42, "num_tickets": 3}` |
| `/helpdesk/step` | POST | Submit agent action |
| `/helpdesk/state` | GET | Get current helpdesk state |
| `/helpdesk/kb` | GET | KB statistics and entries |
| `/helpdesk/kb/search?q=password` | GET | Search the Knowledge Base |

### Example: Helpdesk API Flow

```bash
# 1. Reset
curl -X POST http://localhost:7860/helpdesk/reset \
  -H "Content-Type: application/json" \
  -d '{"seed": 42, "num_tickets": 2}'

# 2. Triage
curl -X POST http://localhost:7860/helpdesk/step \
  -H "Content-Type: application/json" \
  -d '{"ticket_id":"ticket_001","agent_role":"TRIAGE","action_type":"triage","action_value":"{\"category\":\"password_reset\",\"priority\":\"medium\",\"tier\":\"L1\"}"}'

# 3. Check state
curl http://localhost:7860/helpdesk/state

# 4. Search KB
curl "http://localhost:7860/helpdesk/kb/search?q=password+expired"
```

---

## 📁 Project Structure

```text
HelpdeskEnv/
├── models.py              # All Pydantic models (Email + Helpdesk)
├── tasks.py               # 9 email scenarios + 5 ticket scenarios
├── graders.py             # 6 graders (spam, priority, reply, triage, efficiency, KB)
├── knowledge_base.py      # KBEntry model + KnowledgeBase class (persistent)
├── helpdeskenv_class.py   # HelpdeskEnv: reset/step/state with multi-agent routing
├── emailenv_class.py      # EmailEnv: original Round 1 environment
├── heuristics.py          # Keyword-based fallback agents (no API key needed)
├── inference.py           # LLM + heuristic inference loops
├── test_integration.py    # 10 end-to-end integration tests
├── agents/
│   ├── __init__.py        # Agent prompt exports
│   ├── triage.py          # Triage Agent system prompt + builder
│   ├── l1_agent.py        # L1 Agent prompt (KB-assisted)
│   ├── l2_agent.py        # L2 Agent prompt (independent diagnosis)
│   └── l3_agent.py        # L3 Agent prompt (KB article writing)
├── server/
│   ├── __init__.py
│   └── app.py             # FastAPI server with Email + Helpdesk endpoints
├── openenv.yaml           # OpenEnv manifest (v2.0.0, 6 tasks)
├── requirements.txt       # Runtime dependencies
├── Dockerfile             # Container build for HF Spaces
└── README.md              # This file
```

---

## 🧪 Reward System

All rewards are in `[0.0, 1.0]` with partial credit.

### Per-Ticket Combined Reward

When a ticket is resolved, the combined reward is:

```text
ticket_reward = resolution_quality × 0.30
              + response_quality   × 0.20
              + efficiency         × 0.20
              + triage_accuracy    × 0.15
              + kb_contribution    × 0.15
```

### Self-Improvement Metric

KB growth across episodes is tracked via `kb.stats()`:

```json
{
  "total_entries": 5,
  "seed_entries": 2,
  "agent_created_entries": 3,
  "total_usage": 12,
  "categories_covered": ["password_reset", "software_install", "network_issue"]
}
```

---

## 🐳 Docker / Hugging Face Spaces

```bash
docker build -t helpdeskenv .
docker run -p 7860:7860 helpdeskenv
```

On Hugging Face, `openenv.yaml` points Spaces at the Dockerfile and exposes port 7860.

---

## 📄 License

Apache-2.0