# baseline_agent.py
"""Deterministic baseline agent for HelpdeskEnv.
Runs episodes using keyword-based heuristic agents (no LLM, no randomness).
Tracks per-ticket component rewards and computes composite scores.
This produces the "before" numbers for Judging Criterion 3 (20%).
Usage:  python baseline_agent.py
Output: Console table + results/baseline_results.json
"""
import json
import os
import sys
from typing import Dict, List, Any, Optional
from helpdeskenv_class import HelpdeskEnv
from models import HelpdeskAction, AgentRole, Ticket
from heuristics import heuristic_triage, heuristic_l1, heuristic_l2, heuristic_l3
from graders import grade_composite, grade_efficiency
SEED = 42
MAX_STEPS_PER_TICKET = 10
def run_baseline_episode(env: HelpdeskEnv, seed: int = SEED) -> Dict[str, Any]:
    """Run one deterministic episode and collect per-ticket component scores."""
    response = env.reset(seed=seed)
    total_tickets = response.total_tickets
    print(f"  Episode {env.episode_count}: {total_tickets} tickets, KB size: {response.kb_size}")
    per_ticket_results: List[Dict[str, Any]] = []
    total_steps = 0
    done = False
    # Track scores per ticket as they come in from step results
    # Key: ticket_id, Value: dict of component scores + metadata
    tracker: Dict[str, Dict[str, Any]] = {}
    while not done and total_steps < total_tickets * MAX_STEPS_PER_TICKET:
        state = env.state()
        if state.is_done:
            break
        ticket = state.current_ticket
        agent = state.current_agent
        if ticket is None or agent is None:
            break
        tid = ticket.ticket_id
        # Initialize tracker for this ticket on first encounter
        if tid not in tracker:
            tracker[tid] = {
                "subject": ticket.subject[:60],
                "category": ticket.category.value,
                "tier": ticket.ground_truth_tier.value,
                "sla_steps": ticket.sla_steps,
                "triage": 0.5,       # default
                "resolution": 0.5,   # default
                "response": 0.5,     # default
                "kb": 0.0,           # default (no article)
                "steps_taken": 0,
                "escalations": 0,
            }
        support_step = state.steps_on_current_ticket
        # Select heuristic action
        if agent == AgentRole.TRIAGE:
            action_data = heuristic_triage(ticket)
        elif agent == AgentRole.L1_SUPPORT:
            action_data = heuristic_l1(ticket, env.kb(), step=max(0, support_step - 1))
        elif agent == AgentRole.L2_SUPPORT:
            action_data = heuristic_l2(ticket, env.kb(), step=max(0, support_step - 1))
        elif agent == AgentRole.L3_SUPPORT:
            action_data = heuristic_l3(ticket, env.kb(), step=max(0, support_step - 1))
        else:
            break
        action_type = action_data["action_type"]
        helpdesk_action = HelpdeskAction(
            ticket_id=tid,
            agent_role=agent,
            action_type=action_type,
            action_value=action_data["action_value"],
        )
        try:
            result = env.step(helpdesk_action)
            total_steps += 1
            tracker[tid]["steps_taken"] += 1
            done = result.done
            # Record component scores based on action type
            if action_type == "triage":
                tracker[tid]["triage"] = result.reward
            elif action_type in ("apply_solution", "apply_fix", "apply_complex_fix"):
                tracker[tid]["resolution"] = result.reward
            elif action_type == "write_kb_entry":
                tracker[tid]["kb"] = result.reward
            elif action_type == "escalate":
                tracker[tid]["escalations"] += 1
            elif action_type == "respond_to_customer":
                tracker[tid]["response"] = result.reward
                # Ticket is now resolved — compute efficiency and composite
                t = tracker[tid]
                eff = grade_efficiency(
                    t["steps_taken"], t["sla_steps"], t["escalations"]
                )
                t["efficiency"] = eff.reward
                components = {
                    "triage": round(t["triage"], 4),
                    "resolution": round(t["resolution"], 4),
                    "response": round(t["response"], 4),
                    "efficiency": round(t["efficiency"], 4),
                    "kb": round(t["kb"], 4),
                }
                composite, _ = grade_composite(**components)
                per_ticket_results.append({
                    "ticket_id": tid,
                    "subject": t["subject"],
                    "category": t["category"],
                    "tier": t["tier"],
                    "components": components,
                    "composite": round(composite, 4),
                })
        except Exception as exc:
            print(f"  [ERROR] Step failed: {exc}", file=sys.stderr)
            total_steps += 1
            done = True
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
def main():
    print("=" * 60)
    print("HelpdeskEnv -- Deterministic Baseline Agent")
    print("=" * 60)
    env = HelpdeskEnv()
    all_results: List[Dict[str, Any]] = []
    for ep in range(3):
        seed = SEED + ep
        print(f"\n{'_' * 60}")
        print(f"Episode {ep + 1}/3 (seed={seed})")
        print(f"{'_' * 60}")
        result = run_baseline_episode(env, seed=seed)
        all_results.append(result)
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
    os.makedirs("results", exist_ok=True)
    output_path = os.path.join("results", "baseline_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=True)
    print(f"\nResults saved to: {output_path}")
    print(f"{'=' * 60}")
if __name__ == "__main__":
    main()