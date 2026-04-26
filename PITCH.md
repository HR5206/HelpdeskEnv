## Question 1: What capability gap does this target?
**LLMs cannot coordinate across specialized roles to solve multi-step problems under time pressure.**
Today's LLMs can answer questions and generate text, but real IT support requires:
- A **triage agent** that classifies issues accurately under ambiguity
- **Tiered specialists** (L1/L2/L3) that know their skill boundaries and escalate appropriately
- **Time pressure** (SLA budgets) that punish wasteful actions
- **Institutional memory** (Knowledge Base) that grows over time so the same mistake isn't debugged twice
No existing RL environment tests all four of these capabilities together. HelpdeskEnv does.
---
## Question 2: What does the agent see, do, and get rewarded for?
**Observation:** A structured IT support ticket with subject, body, sender, and context.
**Actions (role-dependent):**
- Triage Agent → classify category, priority, tier (JSON output)
- L1/L2 → search KB, apply fix, respond to customer, or escalate
- L3 → all of the above + write KB articles for novel issues
**Reward signal (per ticket, 5 dimensions):**
ticket_reward = triage_accuracy × 0.15 + resolution_quality × 0.30 + response_quality × 0.20 + efficiency (SLA) × 0.20 + kb_contribution × 0.15

The reward is **dense** (feedback after every action), **multi-dimensional** (not just correct/incorrect), and **hard to game** (keyword diversity checks, copy-paste detection, minimum length thresholds).
---
## Question 3: What changed after training?
| Metric | Baseline (heuristic) | Trained (GRPO) | Delta |
|---|---|---|---|
| Avg Episode Reward | [TBD Part 3] | [TBD Part 5] | [TBD] |
| Triage Accuracy | [TBD] | [TBD] | [TBD] |
| SLA Compliance | [TBD] | [TBD] | [TBD] |
| KB Articles Written | [TBD] | [TBD] | [TBD] |
| KB Size After 5 Episodes | [TBD] | [TBD] | [TBD] |
*(These numbers will be filled in after Parts 3, 5, and 6.)*
**The self-improvement narrative:** Episode 1 KB has 2 seed articles. By episode 5, the trained agent has written 5+ new KB articles. L1/L2 agents in later episodes resolve tickets faster because they find solutions in the KB. This is measurable and plotted.
---
## Question 4: Who would care and why?
**Enterprise IT leaders** — IT helpdesks handle 15-20 million tickets per year in large organizations. Training AI agents that can triage, resolve, and build institutional knowledge would reduce resolution times and costs.
**AI safety researchers** — Multi-agent coordination with escalation boundaries is a testbed for studying role-based authority, delegation, and the risks of agents operating beyond their skill level.
**The RL community** — Self-improving environments where the world state (KB) changes across episodes are rare. Most RL environments are stateless between episodes. HelpdeskEnv's persistent KB creates a novel curriculum learning dynamic.