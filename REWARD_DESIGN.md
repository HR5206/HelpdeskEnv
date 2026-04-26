# Reward Design Philosophy — HelpdeskEnv
## The 5-Dimensional Composite Reward
When a ticket is fully resolved, the composite reward is:
ticket_reward = triage_accuracy x 0.15 + resolution_quality x 0.30 + response_quality x 0.20 + efficiency (SLA) x 0.20 + kb_contribution x 0.15

### Why these weights?
| Dimension | Weight | Rationale |
|---|---|---|
| **Resolution Quality** | 30% | The primary goal is fixing the problem. This gets the highest weight because an agent that doesn't solve the issue has failed regardless of everything else. |
| **Response Quality** | 20% | IT support isn't just fixing — it's communicating. A correct fix with a rude response is a bad outcome in real helpdesks. |
| **Efficiency (SLA)** | 20% | Real helpdesks have SLAs. An agent that takes 10 steps to do what could be done in 3 wastes resources. This weight creates time pressure without dominating the score. |
| **Triage Accuracy** | 15% | Misrouting costs time but is recoverable (the ticket still gets resolved, just slower). Hence lower weight than resolution. |
| **KB Contribution** | 15% | Writing KB articles is valuable but optional (only L3 tickets require it). The weight rewards self-improvement without penalizing L1/L2 tickets that don't need articles. |
### Why these weights sum to 1.0
The composite is always in [0.0, 1.0]. This makes it directly comparable across episodes and compatible with standard RL training loops that expect normalized rewards.
## Anti-Gaming Mechanisms
### Problem: Keyword stuffing
**Attack:** Agent repeats "sorry thank you please" to max politeness score.
**Defense:** `_score_politeness()` already counts unique phrases (each word checked once via `if word in`). Additionally, `_score_length()` applies `_detect_repetition()` which penalizes text with >30% repeated 3-grams.
### Problem: Copy-paste exploitation
**Attack:** Agent copies the ticket body verbatim as its "response" to score high on relevance.
**Defense:** `_score_relevance()` applies `_detect_copy_paste()` which measures word-level overlap between reply and source. >60% overlap triggers a penalty multiplier (down to 0.2x at >80%).
### Problem: Padding for length
**Attack:** Agent generates filler text to reach the ideal 60-200 word count.
**Defense:** `_score_length()` applies `_detect_repetition()` which checks trigram uniqueness. Padding with "step 1 step 1 step 1" gets caught.
### Problem: Gaming specificity keywords
**Attack:** Agent writes "resolved fixed verified confirmed diagnosed" without actual content.
**Defense:** The specificity check counts DISTINCT keywords (each counted once). Combined with the length and relevance requirements, an agent needs genuine content to score well across all three dimensions simultaneously.
### What CAN'T be gamed
- `grade_triage()` compares against exact ground truth — no shortcut exists.
- `grade_efficiency()` uses environment-tracked step/escalation counts — the agent cannot modify these.
- The composite reward requires scoring well on ALL 5 dimensions simultaneously. Gaming one dimension while neglecting others yields a low composite.