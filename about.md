# About HelpdeskEnv

## 📌 Project Overview
**HelpdeskEnv** is a cutting-edge **multi-agent IT Helpdesk environment** designed as an OpenEnv simulation. It serves as an evolution of the previous "EmailEnv" project (created for Round 1 of the OpenEnv Hackathon) and is specifically built for Round 2.

The primary goal of this environment is to provide a robust platform for evaluating Language Models (LLMs) and Reinforcement Learning (RL) agents in complex, long-horizon tasks. Specialized AI agents collaborate to resolve IT support tickets through **tiered escalation**, **SLA-driven planning**, and an innovative **Self-Improving Knowledge Base**.

## 🏗️ How It Works (The Architecture)

The system operates by simulating an IT Helpdesk ticket queue where different AI agents take on distinct roles to resolve user issues. 

The core flow of a ticket involves multiple agents:
1. **Triage Agent:** Every new ticket first arrives at Triage. This agent reads the ticket, categorizes the issue, determines its priority, and routes it to the appropriate support tier (L1, L2, or L3) based on complexity.
2. **Support Agents (L1, L2, L3):** Once routed, the assigned Support Agent takes over.
   - **L1 (Simple Issues):** Handles basic requests (e.g., password resets) primarily using the Knowledge Base.
   - **L2 (Medium Issues):** Handles more complex troubleshooting.
   - **L3 (Expert Issues):** Handles novel or highly technical problems.
3. **Action Loop:** Support agents can perform several actions:
   - `search_kb`: Query the Knowledge Base for similar past issues.
   - `apply_fix` / `apply_complex_fix`: Propose a technical solution to the problem.
   - `respond_to_customer`: Draft a reply to the user and resolve the ticket.
   - `escalate`: If an L1 or L2 agent cannot solve the issue, they can escalate it up the chain.
   - `write_kb_entry` (**L3 Only**): If L3 encounters a novel issue, it writes a new Knowledge Base article to document the solution for future agents.

## 🧠 The Self-Improvement Loop
One of the standout features of HelpdeskEnv is its persistent **Knowledge Base (KB)**. 
Unlike typical environments that reset completely between episodes, the KB in HelpdeskEnv survives across episodes. 

Here is how the system self-improves:
- **Episode 1:** The KB starts with just 2 seed entries. The L3 agent has to solve novel tickets from scratch. Once solved, L3 writes new KB articles documenting the fixes.
- **Episode 2:** The KB now has 5 entries. When similar tickets appear, L1 and L2 agents can find the solutions via `search_kb`. They resolve tickets faster without needing to escalate to L3.
- **Episode 5+:** Most common issues are now documented in the KB. Agents achieve near-optimal performance, resolving tickets quickly and efficiently, leading to much higher reward scores.

## 📋 Evaluation & Tasks
The environment evaluates agents across 3 tasks. Each action is graded by specialized automated graders.

**Tasks:**
- **Ticket Triage** (Medium): Scored on correctly identifying the category, priority, and routing tier.
- **Ticket Resolution** (Hard): Scored heavily on SLA compliance (resolving quickly) and escalation efficiency.
- **KB Contribution** (Hard): Scored on the relevance, length, and specificity of articles written by L3.

**Combined Reward Metric:**
When a ticket is fully resolved, the agent receives a combined reward calculated as follows:
- Resolution Quality: 30%
- Response Quality: 20%
- Efficiency (SLA): 20%
- Triage Accuracy: 15%
- KB Contribution: 15%

## 📂 Key Components
- `helpdeskenv_class.py`: The core state machine that orchestrates ticket routing, action validation, and episode lifecycles.
- `knowledge_base.py`: Implements the persistent vector-like KB allowing agents to search and add entries.
- `graders.py`: The automated evaluation logic for tasks.
- `tasks.py`: Contains the definition of the ticket scenarios.
- `models.py`: Pydantic models enforcing rigid schemas for the REST API and internal state.
- `server/app.py`: A FastAPI server that exposes the OpenEnv standard HTTP endpoints (`/reset`, `/step`, `/state`).
- `inference.py`: Scripts to run baseline agents or LLMs against the environment.

## 🚀 Getting Started
The environment can be run completely locally:
1. **Install dependencies:** `pip install -r requirements.txt`
2. **Start the API Server:** `uvicorn server.app:app --host 0.0.0.0 --port 7860`
3. **Run Inference/Evaluation:** `python inference.py`

This project encapsulates a highly dynamic, stateful AI environment that pushes the boundaries of multi-agent collaboration, structured planning, and continuous learning.
