# generate_training_data.py
"""Generate training data for TRL GRPO from HelpdeskEnv heuristic rollouts.
Produces JSONL files where each line is:
  {"prompt": "<system + ticket observation>", "completion": "<optimal action JSON>"}
The prompt format matches what the GRPO trainer will feed to the model.
The completions are the heuristic agent's "expert" actions — not perfect,
but a strong baseline for the model to learn from and improve on.
Usage:  python generate_training_data.py
Output: training_data/helpdesk_train.jsonl (80%)
        training_data/helpdesk_eval.jsonl  (20%)
"""
import json
import os
import random
from typing import Dict, List, Any
from helpdeskenv_class import HelpdeskEnv
from models import HelpdeskAction, AgentRole, Ticket
from heuristics import heuristic_triage, heuristic_l1, heuristic_l2, heuristic_l3
# Reproducible
SEED = 123
NUM_EPISODES = 5
TRAIN_SPLIT = 0.8
# System prompts per agent role
SYSTEM_PROMPTS = {
    AgentRole.TRIAGE: (
        "You are the Triage Agent in an IT Helpdesk. "
        "Classify the ticket by category, priority, and support tier. "
        "Respond with a JSON object: {\"category\": \"...\", \"priority\": \"...\", \"tier\": \"...\"}. "
        "Categories: password_reset, software_install, network_issue, hardware_failure, data_recovery, other. "
        "Priorities: low, medium, high, critical. Tiers: L1, L2, L3."
    ),
    AgentRole.L1_SUPPORT: (
        "You are an L1 Support Agent. Handle simple, well-documented issues. "
        "Available actions: search_kb, apply_solution, respond_to_customer, escalate. "
        "Respond with a JSON object: {\"action_type\": \"...\", \"action_value\": \"...\"}."
    ),
    AgentRole.L2_SUPPORT: (
        "You are an L2 Support Agent. Handle moderately complex issues. "
        "Available actions: search_kb, apply_fix, respond_to_customer, escalate. "
        "Respond with a JSON object: {\"action_type\": \"...\", \"action_value\": \"...\"}."
    ),
    AgentRole.L3_SUPPORT: (
        "You are an L3 Support Agent (Expert). Handle critical and novel issues. "
        "Available actions: search_kb, apply_complex_fix, respond_to_customer, escalate, write_kb_entry. "
        "Respond with a JSON object: {\"action_type\": \"...\", \"action_value\": \"...\"}."
    ),
}
def format_ticket_observation(ticket: Ticket, agent: AgentRole, step: int) -> str:
    """Format the ticket as a user prompt the model will see during training."""
    obs = (
        f"Ticket ID: {ticket.ticket_id}\n"
        f"Subject: {ticket.subject}\n"
        f"From: {ticket.sender}\n"
        f"Body:\n{ticket.body}\n"
    )
    if ticket.context:
        obs += f"\nContext: {ticket.context}\n"
    obs += f"\nYour role: {agent.value} | Step: {step}"
    return obs
def format_action_completion(action_data: Dict[str, str], agent: AgentRole) -> str:
    """Format the heuristic action as the completion the model should learn."""
    return json.dumps({
        "action_type": action_data["action_type"],
        "action_value": action_data["action_value"],
    }, ensure_ascii=True)
def generate_examples() -> List[Dict[str, str]]:
    """Run multiple episodes and collect (prompt, completion) pairs."""
    env = HelpdeskEnv()
    examples: List[Dict[str, str]] = []
    for ep in range(NUM_EPISODES):
        seed = SEED + ep
        response = env.reset(seed=seed)
        done = False
        total_steps = 0
        max_steps = response.total_tickets * 10
        while not done and total_steps < max_steps:
            state = env.state()
            if state.is_done:
                break
            ticket = state.current_ticket
            agent = state.current_agent
            if ticket is None or agent is None:
                break
            support_step = state.steps_on_current_ticket
            # Get heuristic action
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
            # Build training example
            system = SYSTEM_PROMPTS[agent]
            user = format_ticket_observation(ticket, agent, support_step)
            prompt = f"<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n"
            completion = format_action_completion(action_data, agent)
            examples.append({
                "prompt": prompt,
                "completion": completion,
            })
            # Step the env to advance state
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
            except Exception as exc:
                print(f"  [WARN] Step error in ep {ep}: {exc}")
                total_steps += 1
                done = True
    return examples
def main():
    print("=" * 60)
    print("HelpdeskEnv -- Training Data Generator")
    print("=" * 60)
    random.seed(SEED)
    examples = generate_examples()
    # Shuffle and split
    random.shuffle(examples)
    split_idx = int(len(examples) * TRAIN_SPLIT)
    train = examples[:split_idx]
    eval_set = examples[split_idx:]
    # Save
    os.makedirs("training_data", exist_ok=True)
    train_path = os.path.join("training_data", "helpdesk_train.jsonl")
    eval_path = os.path.join("training_data", "helpdesk_eval.jsonl")
    for path, data in [(train_path, train), (eval_path, eval_set)]:
        with open(path, "w", encoding="utf-8") as f:
            for ex in data:
                f.write(json.dumps(ex, ensure_ascii=True) + "\n")
    # Statistics
    total_tokens_est = sum(
        len(ex["prompt"].split()) + len(ex["completion"].split())
        for ex in examples
    )
    # Difficulty breakdown by role
    role_counts: Dict[str, int] = {}
    for ex in examples:
        for role in ["Triage", "L1", "L2", "L3"]:
            if role.lower() in ex["prompt"].lower():
                role_counts[role] = role_counts.get(role, 0) + 1
                break
    print(f"\n  Total examples:     {len(examples)}")
    print(f"  Train split:        {len(train)} ({TRAIN_SPLIT:.0%})")
    print(f"  Eval split:         {len(eval_set)} ({1-TRAIN_SPLIT:.0%})")
    print(f"  Est. total tokens:  ~{total_tokens_est:,}")
    print(f"\n  Role breakdown:")
    for role, count in sorted(role_counts.items()):
        print(f"    {role}: {count} examples")
    # Cost estimate for T4-small
    steps_200 = 200
    t4_hourly = 0.40  # $/hr for T4-small
    est_hours = 0.5   # ~30 min for 200 steps on Qwen2.5-1.5B
    est_cost = est_hours * t4_hourly
    print(f"\n  Training cost estimate (T4-small, 200 steps):")
    print(f"    ~{est_hours:.1f} hrs x ${t4_hourly}/hr = ~${est_cost:.2f}")
    print(f"\n  Saved to:")
    print(f"    {train_path}")
    print(f"    {eval_path}")
    print(f"{'=' * 60}")
if __name__ == "__main__":
    main()