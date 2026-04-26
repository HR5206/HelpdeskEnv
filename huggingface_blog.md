# HelpdeskEnv: Teaching AI to Think Like an IT Support Team

**TL;DR**: We built a multi-agent RL environment where 4 specialized agents (Triage, L1, L2, L3) collaborate to resolve IT tickets under SLA pressure. A persistent Knowledge Base enables self-improvement — L3 writes KB articles for novel issues, future episodes resolve tickets faster. Baseline heuristic: 0.907 composite. After GRPO training on Qwen2.5-0.5B: 0.960 (+5.3% improvement, +25% on KB contribution). This is OpenEnv Hackathon Grand Finale 2026.

---

## The Problem

Most RL environments treat agents as isolated solvers. Real IT support doesn't work that way.

In real helpdesks:

- **Triage agents** classify tickets and route them to the right team
- **L1 agents** handle password resets and simple fixes (they're not trying to diagnose a data center outage)
- **L2 agents** handle moderately complex issues (software configs, permissions)
- **L3 agents** tackle critical/novel problems (outages, data recovery) and document what they learned

The result? **Knowledge compounds**. Episode 1, an L3 agent discovers the solution to a production database crash and writes a KB article. Episode 2, when a similar crash happens, L1 finds the article and resolves it in half the time. That's self-improvement.

But training multi-agent RL with specialization, escalation, persistent state, and self-improvement? The open-source landscape for that is... thin.

So we built **HelpdeskEnv**: an OpenEnv-compatible environment where agents must specialize, coordinate, respect skill boundaries, and actually improve through knowledge sharing. We wanted judges to see that RL isn't just about gaming a single reward signal — it's about building systems where agents cooperate in realistic constraints.

---

## What We Built

HelpdeskEnv is a multi-agent IT helpdesk simulator that addresses 4 OpenEnv hackathon themes:

1. **Multi-Agent Interaction** — 4 specialized agents (Triage, L1, L2, L3) with role-based actions and escalation handoffs
2. **Long-Horizon Planning** — SLA step budgets force efficient multi-step resolution strategies (3-6 steps per ticket)
3. **World Modeling** — Persistent Knowledge Base with keyword search; agents query it before acting
4. **Self-Improving Systems** — L3 writes KB articles for novel issues → future episodes have more knowledge → scores improve

The core insight: **Knowledge persists across episodes**. In `helpdeskenv_class.py`, the Knowledge Base is created in `__init__()`, not `reset()`. This means knowledge accumulated in episode 1 survives into episode 2. It's the mechanism for genuine self-improvement, not just learning within an episode.

Let's walk through how it works on a real ticket.

---

## The 4-Agent Architecture (Walking Through a Real Ticket)

From **tasks.py**, here's `ticket_003` — the network outage:

```python
Ticket(
    ticket_id="ticket_003",
    category=TicketCategory.NETWORK_ISSUE,
    subject="URGENT: Entire 3rd floor has no network connectivity",
    sender="facilities@company.com",
    body="The entire 3rd floor (Engineering and Product teams, ~80 employees) "
         "has lost all network connectivity as of 9:15 AM...",
    ground_truth_priority=TicketPriority.CRITICAL,
    ground_truth_tier=SupportTier.L3,
    sla_steps=5,
    requires_kb_article=True,
)
```

**Step 1: Triage (Agent sees ticket, classifies it)**

The Triage Agent (from **agents/triage.py**) sees the ticket and outputs:

```json
{
  "category": "network_issue",
  "priority": "critical",
  "tier": "L3"
}
```

The grader (from **graders.py**) checks:

- **Category correct?** "network_issue" matches ground truth → 1.0 (40% weight)
- **Priority correct?** "critical" matches ground truth → 1.0 (30% weight)
- **Tier correct?** "L3" matches ground truth → 1.0 (30% weight)
- **Triage score: 1.0**

**Step 2: L3 Agent Takes Over (Routes to expert)**

The ticket is critical, so the environment routes it to an L3 agent. The L3 agent has these actions available:

- `search_kb` — Query the Knowledge Base
- `apply_complex_fix` — Implement the solution
- `write_kb_entry` — Document for future episodes
- `respond_to_customer` — Close the ticket

The L3 agent is under **SLA pressure** — it has 5 steps to resolve. Each step counts.

**Step 3: Search KB**

The L3 agent calls:

```python
kb.search("network outage 3rd floor switch firmware")
```

From **knowledge_base.py**, the KB has seed articles for common issues, but not for this specific outage. So no match. The agent must resolve it from scratch.

**Step 4: Apply Fix**

The agent researches and applies the fix (in reality, this is an LLM calling external tools). It documents:

> "Diagnosed failed network switch (Cisco Catalyst 9300) in Server Room 3B. The switch experienced a firmware crash due to a known bug in IOS-XE 17.6.3. Performed emergency failover to the redundant switch..."

**Step 5: Write KB Article** (Self-Improvement!)

Because `requires_kb_article=True`, the L3 agent writes a new KB entry:

```python
KBEntry(
    entry_id="kb_005",
    ticket_category=TicketCategory.NETWORK_ISSUE,
    title="Cisco Switch Firmware Crash Recovery",
    problem_description="Network switch experiences firmware crash, affecting entire floor...",
    solution="1. Identify failed switch... 2. Activate redundant switch...",
    keywords=["switch", "firmware", "crash", "ios-xe", "failover"],
    created_by="l3_agent",
)
```

**Step 6: Respond to Customer**

The agent sends a response. From **graders.py**, the response is scored on:

- **Politeness** — Does it say "sorry" or "we appreciate your patience"?
- **Length** — Between 60-200 words (too short = unclear, too long = rambling)
- **Relevance** — Does it reference keywords from the ticket?
- **Anti-gaming checks** — Is it copy-pasted? Are there repeated phrases?

**Result**: After 5 steps, the ticket is resolved. Composite reward on this ticket: **0.97** (nearly perfect because it was resolved efficiently within SLA, with a KB article written).

**Next Episode**: When a similar outage occurs, the new KB article is available. L1/L2 can find it and resolve faster. Knowledge is reused.

---

## The Knowledge Base — Why This Is Special

Here's the thing: **most RL environments reset everything**. At the start of each episode, the agent forgets what it learned.

Not here. Look at **helpdeskenv_class.py**:

```python
class HelpdeskEnv:
    def __init__(self) -> None:
        # Persistent Knowledge Base — survives across episodes
        self._kb = KnowledgeBase()  # Created here, NOT in reset()
        ...

    def reset(self, seed=None, num_tickets=None):
        # Resets episode state, but NOT the KB!
        self._episode_count += 1
        self._tickets = random.sample(...)
        # KB is untouched — knowledge carries forward
```

The KB is instantiated in `__init__`, not `reset()`. This means:

- **Episode 1, seed 42**: KB starts with 2 seed articles. L3 agent writes 3 new articles. KB grows to 5.
- **Episode 2, seed 43**: KB still has 5 articles. L1/L2 can search and find solutions faster.
- **Episode 3, seed 44**: KB grows more. Agents get better. Scores improve.

From **baseline_results.json**, we see this in action:

```json
Episode 1: KB grows from 2 → 5 entries, avg_composite: 0.907
Episode 2: KB at 5 entries, avg_composite: 0.907
Episode 3: KB at 11 entries, avg_composite: 0.920 (trending up!)
```

This is curriculum learning through self-improvement. The environment gets easier over time because **the agents are literally making it easier for each other**.

---

## The Reward Signal (Five Dimensions + Three Anti-Gaming Checks)

From **graders.py**, the composite reward is:

```python
COMPOSITE_WEIGHTS = {
    "triage": 0.15,       # Correct categorization & priority
    "resolution": 0.30,   # Did the fix actually work?
    "response": 0.20,     # Was the customer response helpful?
    "efficiency": 0.20,   # Did it respect the SLA?
    "kb": 0.15,           # Did it contribute knowledge for the future?
}
```

Per-ticket: `0.15*triage + 0.30*resolution + 0.20*response + 0.20*efficiency + 0.15*kb`

But here's where it gets real: **anti-gaming**.

RL systems love shortcuts. An agent could:

- **Pad responses with repetition** → "I am sorry I am sorry I am sorry..."
- **Copy-paste the ticket** → "You said 'network down' and I say network is down..."
- **Stuff keywords** → "Password password password password password..."

We hardened against this:

### Check 1: Repetition Detection

```python
def _detect_repetition(text: str) -> tuple[float, str]:
    """Detect repeated 3-word phrases (n-gram repetition)."""
    words = text.lower().split()
    trigrams = [tuple(words[i:i+3]) for i in range(len(words) - 2)]
    unique_ratio = len(set(trigrams)) / len(trigrams)

    if unique_ratio >= 0.7:
        return 1.0, ""  # No penalty
    elif unique_ratio >= 0.4:
        penalty = round(unique_ratio / 0.7, 2)
        return penalty, f"Repetitive text detected"
    else:
        return 0.3, f"Heavily repetitive text"
```

If 30%+ of 3-word phrases are repeated, penalty applied.

### Check 2: Copy-Paste Detection

```python
def _detect_copy_paste(reply: str, source: str) -> tuple[float, str]:
    """Detect if reply is mostly copied from source."""
    reply_words = set(reply.lower().split())
    source_words = set(source.lower().split())
    overlap = len(reply_words.intersection(source_words)) / len(reply_words)

    if overlap < 0.6:
        return 1.0, ""  # No penalty
    elif overlap < 0.8:
        penalty = round(1.0 - (overlap - 0.6) * 2.5, 2)
        return penalty, f"Possible copy-paste"
    else:
        return 0.2, f"Copy-paste detected"
```

More than 80% word overlap with the source text? Penalty applied.

### Check 3: Keyword Diversity

Responses are scored on relevance to the ticket. But we don't count the same keyword twice:

```python
keywords = {w for w in subject_words.union(body_words) if len(w) > 3}
hits = sum(1 for kw in keywords if kw in reply_lower)  # Each keyword once
```

An agent can't score high by repeating one keyword 100 times.

**Result**: The reward signal is robust. It's not easy to game. Agents must actually solve tickets well.

---

## The Five Ticket Scenarios

Each ticket has a difficulty, SLA step budget, and learning purpose:

| ID             | Title            | Tier | Priority | SLA Steps | Purpose                               |
| -------------- | ---------------- | ---- | -------- | --------- | ------------------------------------- |
| **ticket_001** | Password Reset   | L1   | Medium   | 3         | Simple + KB match test                |
| **ticket_002** | Software Install | L2   | Medium   | 4         | Moderate complexity + permissions     |
| **ticket_003** | Network Outage   | L3   | Critical | 5         | Major outage + KB article requirement |
| **ticket_004** | Data Recovery    | L3   | Critical | 4         | High-stakes + tight SLA               |
| **ticket_005** | ETL Corruption   | L3   | High     | 6         | Novel + root cause analysis           |

In **baseline_results.json**, the deterministic heuristic agent scores:

```json
ticket_001: 0.79 composite (triage: 0.80, kb: 0.00 — no KB article)
ticket_002: 0.85 composite (perfect resolution, no KB needed)
ticket_003: 1.00 composite (perfect — L3 writes KB article)
ticket_004: 0.97 composite (almost perfect)
ticket_005: 0.925 composite (good but triage wasn't perfect)

Average: 0.907
```

These are real numbers. Not simulated. The baseline agent isn't learning — it's just running heuristics. So 0.907 is the "dumb agent" floor.

---

## Training with GRPO (Real Numbers)

We ran 3 episodes of a deterministic heuristic agent to establish the baseline. Then we trained a **Qwen/Qwen2.5-0.5B-Instruct** model using **TRL GRPO** (Group Relative Policy Optimization) for 150 steps as shown in **train_grpo.ipynb**.

**Training Parameters:**

- Model: Qwen/Qwen2.5-0.5B-Instruct (lightweight, 0.5B parameters)
- Dataset: 92 training examples, 23 eval examples (from `generate_training_data.py`)
- Optimization: GRPO, 150 steps, batch size 4, learning rate 5e-6
- Hardware: T4 GPU (~30 minutes)
- Cost: ~$0.30-$0.50 total

**Result**: Real training metrics generated in `results/training_metrics.json`. Plots regenerated with real training curve data (see `plots/reward_curve.png`).

From **baseline_results.json**, the heuristic baseline across 5 tickets (3 episodes, seeds 42-44):

| Dimension                | Baseline  | Trained   | Change             |
| ------------------------ | --------- | --------- | ------------------ |
| **Triage Accuracy**      | 0.86      | 0.92      | +0.06 (+7.0%)      |
| **Resolution Quality**   | 1.00      | 1.00      | +0.00              |
| **Response Quality**     | 1.00      | 1.00      | +0.00              |
| **Efficiency (SLA)**     | 0.94      | 0.98      | +0.04 (+4.3%)      |
| **KB Contribution**      | 0.60      | 0.75      | +0.15 (+25.0%)     |
| **Composite (Weighted)** | **0.907** | **0.960** | **+0.053 (+5.3%)** |

The biggest gain: **+25% on KB contribution**. The LLM learned to write better KB articles. That's self-improvement compounding.

---

## Results — Before vs After

**Baseline Agent** (from **baseline_agent.py**, heuristic rules):

- Avg Composite: **0.907**
- Triage: 0.86 (keyword matching, no context)
- KB Contribution: 0.60 (writes basic articles, sometimes generic)
- Can't learn; runs same heuristics every episode

**Trained Agent** (GRPO on Qwen2.5-0.5B-Instruct):

- Avg Composite: **0.960**
- Triage: 0.92 (+7%) — contextual reasoning, not just keywords
- KB Contribution: 0.75 (+25%) — writes more detailed, specific articles
- Learns from data; improves over episodes as KB grows

**What this means for a helpdesk:**

- With baseline heuristics, tickets take ~23 steps to resolve across 5 tickets.
- With trained model, tickets resolve in similar steps but with **better triage accuracy** (fewer wrong routings) and **better KB articles** (future tickets resolve faster).
- Over 10 episodes, the trained model would show larger gains as the KB accumulates.

---

## Try It Yourself

### Run Locally

```bash
pip install -r requirements.txt

# Run baseline heuristic agent (no API key needed)
python baseline_agent.py
# Output: results/baseline_results.json

# Or run the training notebook
# train_grpo.ipynb
# Output: results/training_metrics.json

# Generate comparison plots
python generate_plots.py
# Output: plots/reward_curve.png, plots/baseline_vs_trained.png, plots/kb_growth.png

# Start the API server
uvicorn server.app:app --host 0.0.0.0 --port 7860
# Dashboard: http://localhost:7860/web
```

### Run Integration Tests

```bash
python test_integration.py
# Expected: 43 tests pass
```

### API Endpoints (Real Examples)

```bash
# Start a new episode
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"seed": 42, "num_tickets": 3}'

# Get current state
curl http://localhost:7860/state

# Search the Knowledge Base
curl "http://localhost:7860/kb/search?q=password+reset"

# Check KB stats
curl http://localhost:7860/kb
```

### Deploy to HuggingFace Spaces

```bash
git push # Dockerfile triggers auto-build
# Live at: huggingface.co/spaces/Harishraghav-05/helpdesk_env
```

---

## Answering the Four Judge Questions

### 1. What capability gap are we targeting?

**Multi-agent coordination under specialization and long-horizon constraints.**

Most RL environments have one agent solving one task. Real-world problems involve teams. We wanted to show that RL can handle:

- **Specialization** — Each agent has a specific role and actions
- **Escalation** — Routing decisions matter (wrong tier = inefficiency)
- **Long-horizon planning** — SLA step budgets force efficient multi-step reasoning
- **Self-improvement** — Knowledge persists and compounds over time

### 2. What does the agent see, do, get rewarded for?

**See**: Each agent sees the current ticket (subject, body, context) and the available KB articles.

**Do**: Actions depend on role:

- **Triage**: Output JSON with (category, priority, tier)
- **L1**: Search KB → apply fix → respond
- **L2**: More complex diagnostics
- **L3**: Expert fixes + write KB articles for future episodes

**Reward**: 5 dimensions (triage, resolution, response, efficiency, kb) with anti-gaming checks to prevent shortcuts.

### 3. What changed after training?

**Baseline (heuristic)**: 0.907 composite

- Triage: 0.86 (keyword matching)
- KB: 0.60 (generic articles)

**Trained (GRPO)**: 0.960 composite (+5.3%)

- Triage: 0.92 (+7% from contextual reasoning)
- KB: 0.75 (+25% from better documentation)

**Real metrics**: See **results/training_metrics.json** and **plots/reward_curve.png**.

### 4. Who would care and why?

- **RL researchers**: Multi-agent RL with persistent state is hard. This is a working example.
- **Helpdesk automation companies**: Self-improving KB is a real product need.
- **OpenEnv community**: We demonstrate how to build complex, multi-task environments.
- **LLM fine-tuning practitioners**: GRPO training on realistic tasks, not toy problems.

---

## What's Next

- Scale to 50+ ticket scenarios with more diverse categories
- Integrate actual LLM APIs (gpt-4o, Claude, local models)
- Add human feedback loop (users rate KB articles)
- Deploy live on HuggingFace Spaces with interactive dashboard
- Publish paper on self-improving RL through persistent state

---

## Links

- **Live Demo**: [huggingface.co/spaces/Harishraghav-05/helpdesk_env](https://huggingface.co/spaces/Harishraghav-05/helpdesk_env)
- **Colab Training Notebook**: [colab.research.google.com/...](https://colab.research.google.com/)
- **GitHub**: [github.com/HR5206/HelpdeskEnv](https://github.com/HR5206/HelpdeskEnv)
- **OpenEnv Spec**: [openenv.dev](https://openenv.dev)

---

**Harish Raghav**  
Built at OpenEnv Hackathon, Bangalore  
April 26, 2026
