## Criterion 1: Environment Innovation (40% of final score)
**Current Score: 7/10**
### What we HAVE:
- Multi-agent architecture with 4 distinct roles (Triage, L1, L2, L3)
- Tiered escalation with role-based action constraints
- Persistent Knowledge Base that survives across episodes (self-improvement)
- SLA step budgets creating time pressure (long-horizon planning)
- 5 realistic IT ticket scenarios with varying difficulty
- 3 separate graders with multi-dimensional scoring (not just binary)
- Keyword-based KB search with usage tracking
### What's MISSING to reach 10/10:
- [ ] Anti-gaming checks in graders — an agent can score high by repeating polite phrases without solving the problem (Part 2)
- [ ] A `grade_composite()` function that formally combines all reward dimensions per ticket (Part 2)
- [ ] A `REWARD_DESIGN.md` document explaining why each reward weight exists — judges want to see intentional design, not arbitrary numbers (Part 2)
- [ ] More ticket scenarios (currently only 5 — could feel thin) — consider adding 2-3 more in a future iteration
- [ ] The `openenv.yaml` still has legacy EmailEnv tags like "email", "spam-detection" (this Part fixes that)
---
## Criterion 2: Storytelling & Presentation (30% of final score)
**Current Score: 3/10**
### What we HAVE:
- A working HF Spaces deployment
- A basic `/web` landing page with task descriptions
- A README with architecture diagram and API docs
### What's MISSING to reach 10/10:
- [ ] A clear 2-sentence pitch that a non-technical judge can understand in 10 seconds (this Part — PITCH.md)
- [ ] A live dashboard that shows the multi-agent workflow visually — current `/web` is a static info page (Part 7)
- [ ] Embedded reward curve plots in the README (Parts 5, 6)
- [ ] A before/after comparison table showing improvement (Parts 3, 6)
- [ ] A video or blog post linked from README (Part 10)
- [ ] The README structure doesn't match what judges scan for — needs complete rewrite (Part 9)
---
## Criterion 3: Showing Improvement in Rewards (20% of final score)
**Current Score: 1/10**
### What we HAVE:
- The KB persistence mechanism exists (L3 writes articles → future episodes benefit)
- `inference.py` runs multiple episodes and prints a summary table
- The heuristic agents work and produce non-zero rewards
### What's MISSING to reach 10/10:
- [ ] A deterministic baseline agent with saved results in `results/baseline_results.json` — this is the "before" (Part 3)
- [ ] A TRL GRPO training notebook that actually trains a model (Part 5)
- [ ] `reward_curve.png` — a real plot from a real training run, with labeled axes (Part 6)
- [ ] `baseline_vs_trained.png` — a grouped bar chart comparing baseline vs trained agent (Part 6)
- [ ] `kb_growth.png` — a line chart showing KB size increasing across episodes (Part 6)
- [ ] All plots committed as `.png` files in `plots/` directory and embedded in README (Parts 6, 9)
---
## Criterion 4: Reward & Training Pipeline (10% of final score)
**Current Score: 2/10**
### What we HAVE:
- Three working graders: `grade_triage`, `grade_efficiency`, `grade_kb_contribution`
- Helper scoring functions: `_score_politeness`, `_score_length`, `_score_relevance`
- A combined reward formula documented in the README (resolution 30% + response 20% + efficiency 20% + triage 15% + KB 15%)
- The heuristic agents can play full episodes end-to-end
### What's MISSING to reach 10/10:
- [ ] The combined reward formula exists only as prose in the README — it's not implemented as a callable function (Part 2 — `grade_composite()`)
- [ ] No training dataset in the required format for TRL GRPO (Part 4)
- [ ] No Colab notebook (Part 5)
- [ ] No integration between the training loop and the live environment API (Part 5)
- [ ] `graders.py` still has unused email-era section headers ("Task 3: Reply Generation Grader") despite the functions being used by helpdesk graders (Part 2 cleanup)
---
## Overall Assessment
| Criterion | Weight | Current Score | Target | Gap |
|---|---|---|---|---|
| Environment Innovation | 40% | 7/10 | 9/10 | Reward hardening, anti-gaming |
| Storytelling & Presentation | 30% | 3/10 | 9/10 | Dashboard, plots, README, video |
| Showing Improvement | 20% | 1/10 | 8/10 | Baseline, training, real curves |
| Training Pipeline | 10% | 2/10 | 8/10 | Composite reward, notebook, dataset |
**Weighted current score: 0.40×7 + 0.30×3 + 0.20×1 + 0.10×2 = 4.1 / 10**
**Weighted target score: 0.40×9 + 0.30×9 + 0.20×8 + 0.10×8 = 8.7 / 10**
**The biggest point-per-effort opportunities:**
1. Parts 5–6 (Training + Plots) → moves Criterion 3 from 1 to 8 (+1.4 weighted points)
2. Parts 7, 9 (Dashboard + README) → moves Criterion 2 from 3 to 9 (+1.8 weighted points)
3. Part 2 (Reward Hardening) → moves Criterion 1 from 7 to 9 (+0.8 weighted points)