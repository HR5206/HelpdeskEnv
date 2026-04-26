# baseline_agent.py
"""Deterministic baseline agent for HelpdeskEnv.
Runs one full episode using keyword-based heuristic agents (no LLM, no
randomness). Tracks per-ticket component rewards and computes composite
scores using grade_composite().
This script produces the "before" numbers that the trained GRPO model
will be compared against (Judging Criterion 3: Showing Improvement, 20%).
Usage:
    python baseline_agent.py
Output:
    - Prints per-ticket score breakdown to console
    - Saves structured results to results/baseline_results.json
"""
import json
import os
import sys
from typing import Dict, List, Any
from helpdeskenv_class import HelpdeskEnv
from models import HelpdeskAction, AgentRole
from heuristics import heuristic_triage, heuristic_l1, heuristic_l2, heuristic_l3
from graders import grade_composite
# Fixed seed for reproducibility — every run produces identical results
SEED = 42
MAX_STEPS_PER_TICKET = 10
def run_baseline_episode(env: HelpdeskEnv, seed: int = SEED) -> Dict[str, Any]:
    """Run one deterministic episode and collect per-ticket component scores.
    Returns a dict with:
        - per_ticket: list of {ticket_id, components, composite}
        - episode_summary: {avg_composite, total_steps, kb_stats, ...}
    """
    response = env.reset(seed=seed)
    total_tickets = response.total_tickets
    print(f"  Episode {env.episode_count}: {total_tickets} tickets, KB size: {response.kb_size}")
    per_ticket_results: List[Dict[str, Any]] = []
    total_steps = 0
    done = False
    while not done and total_steps < total_tickets * MAX_STEPS_PER_TICKET:
        state = env.state()
        if state.is_done:
            break
        ticket = state.current_ticket
        agent = state.current_agent
        if ticket is None or agent is None:
            break
        # Track which ticket we're on and its component scores
        current_ticket_id = ticket.ticket_id
        support_step = state.steps_on_current_ticket
        # Select heuristic action based on current agent role
        if agent == AgentRole.TRIAGE:
            action_data = heuristic_triage(ticket)
        elif agent == AgentRole.L1_SUPPORT:
            l_step = max(0, support_step - 1)
            action_data = heuristic_l1(ticket, env.kb(), step=l_step)
        elif agent == AgentRole.L2_SUPPORT:
            l_step = max(0, support_step - 1)
            action_data = heuristic_l2(ticket, env.kb(), step=l_step)
        elif agent == AgentRole.L3_SUPPORT:
            l_step = max(0, support_step - 1)
            action_data = heuristic_l3(ticket, env.kb(), step=l_step)
        else:
            break
        # Submit the action
        helpdesk_action = HelpdeskAction(
            ticket_id=ticket.ticket_id,
            agent_role=agent,
            action_type=action_data["action_type"],
            action_value=action_data["action_value"],
        )
        try:
            result = env.step(helpdesk_action)
            total_steps += 1
            done = result.done
            # When a ticket is resolved (respond_to_customer triggers this),
            # extract the component scores from the env's internal tracking
            if action_data["action_type"] == "respond_to_customer":
                components = _extract_ticket_components(env, current_ticket_id)
                composite, _ = grade_composite(**components)
                per_ticket_results.append({
                    "ticket_id": current_ticket_id,
                    "subject": ticket.subject[:60],
                    "category": ticket.category.value,
                    "tier": ticket.ground_truth_tier.value,
                    "components": components,
                    "composite": round(composite, 4),
                })
        except Exception as exc:
            print(f"  [ERROR] Step failed: {exc}", file=sys.stderr)
            total_steps += 1
            done = True
    # Episode summary
    composites = [r["composite"] for r in per_ticket_results]
    avg_composite = round(sum(composites) / len(composites), 4) if composites else 0.0
    kb_stats = env.kb().stats()
    return {
        "seed": seed,
        "episode": env.episode_count,
        "total_steps": total_steps,
        "total_tickets": total_tickets,
        "avg_composite": avg_composite,
        "per_ticket": per_ticket_results,
        "kb_size": kb_stats["total_entries"],
        "kb_agent_entries": kb_stats["agent_created_entries"],
    }
def _extract_ticket_components(env: HelpdeskEnv, ticket_id: str) -> Dict[str, float]:
    """Extract per-component reward scores from the env's internal state.
    Reads from env._ticket_rewards (set during step processing) and
    env._history (for triage scores). Falls back to defaults if missing.
    """
    rewards = env._ticket_rewards.get(ticket_id, [])
    # Resolution score (from apply_solution/fix/complex_fix step)
    resolution = next((r for t, r in rewards if t == "resolution"), 0.5)
    # Response score (from respond_to_customer step)
    response = next((r for t, r in rewards if t == "response"), 0.5)
    # KB contribution (from write_kb_entry step, 0.0 if none written)
    kb = next((r for t, r in rewards if t == "kb_contribution"), 0.0)
    # Triage score (find from history by checking feedback text)
    triage = 0.5
    for h in env._history:
        if h.task_id == ticket_id and h.feedback and "Triage Score" in h.feedback:
            triage = h.reward
            break
    # Efficiency (recompute from env state — the env already computed this
    # during _resolve_ticket, but we recalculate for transparency)
    from graders import grade_efficiency
    eff_result = grade_efficiency(
        env._steps_on_ticket, 
        env._current_ticket.sla_steps if env._current_ticket else 3,
        env._escalation_count
    )
    efficiency = eff_result.reward
    return {
        "triage": round(triage, 4),
        "resolution": round(resolution, 4),
        "response": round(response, 4),
        "efficiency": round(efficiency, 4),
        "kb": round(kb, 4),
    }
def main():
    print("=" * 60)
    print("HelpdeskEnv -- Deterministic Baseline Agent")
    print("=" * 60)
    env = HelpdeskEnv()
    # Run 3 episodes to show KB self-improvement across episodes
    all_results: List[Dict[str, Any]] = []
    for ep in range(3):
        seed = SEED + ep
        print(f"\n{'_' * 60}")
        print(f"Episode {ep + 1}/3 (seed={seed})")
        print(f"{'_' * 60}")
        result = run_baseline_episode(env, seed=seed)
        all_results.append(result)
        # Print per-ticket table for this episode
        print(f"\n  {'Ticket':<14} {'Triage':>7} {'Resol':>7} {'Resp':>7} {'Effic':>7} {'KB':>7} {'Composite':>10}")
        print(f"  {'_' * 62}")
        for t in result["per_ticket"]:
            c = t["components"]
            print(
                f"  {t['ticket_id']:<14} "
                f"{c['triage']:>7.2f} "
                f"{c['resolution']:>7.2f} "
                f"{c['response']:>7.2f} "
                f"{c['efficiency']:>7.2f} "
                f"{c['kb']:>7.2f} "
                f"{t['composite']:>10.4f}"
            )
        print(f"  {'_' * 62}")
        print(f"  Avg Composite: {result['avg_composite']:.4f}  |  KB size: {result['kb_size']}")
    # Cross-episode summary
    print(f"\n{'=' * 60}")
    print("Cross-Episode Summary (Self-Improvement)")
    print(f"{'=' * 60}")
    print(f"  {'Ep':>3} {'Avg Composite':>14} {'Steps':>6} {'KB Size':>8} {'KB New':>7}")
    print(f"  {'_' * 42}")
    for r in all_results:
        print(
            f"  {r['episode']:3d} "
            f"{r['avg_composite']:14.4f} "
            f"{r['total_steps']:6d} "
            f"{r['kb_size']:8d} "
            f"{r['kb_agent_entries']:7d}"
        )
    # Save results
    os.makedirs("results", exist_ok=True)
    output_path = os.path.join("results", "baseline_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=True)
    print(f"\nResults saved to: {output_path}")
    print(f"{'=' * 60}")
if __name__ == "__main__":
    main()
# ============================================================================
# Expected output (from running with seed=42, 43, 44):
# The exact numbers will be filled in after running the script.
# These become the "BEFORE" column in the judging comparison table.
# ============================================================================