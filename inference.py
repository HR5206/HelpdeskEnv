"""HelpdeskEnv Inference Script"""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Optional

from openai import OpenAI

from helpdeskenv_class import HelpdeskEnv
from models import HelpdeskAction, AgentRole
from heuristics import heuristic_triage, heuristic_l1, heuristic_l2, heuristic_l3

API_BASE_URL: str = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-4o-mini")
API_KEY: Optional[str] = (
    os.getenv("HF_TOKEN")
    or os.getenv("API_KEY")
    or os.getenv("OPENAI_API_KEY")
)
BENCHMARK: str = "helpdeskenv"
MAX_STEPS: int = 10
TEMPERATURE: float = 0.0
MAX_TOKENS: int = 256
SUCCESS_SCORE_THRESHOLD: float = 0.5


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    action_clean = action.replace("\n", " ")
    print(
        f"[STEP] step={step} action={action_clean} reward={reward:.2f} "
        f"done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

def _get_client() -> Optional[OpenAI]:
    if not API_KEY:
        return None
    return OpenAI(base_url=API_BASE_URL, api_key=API_KEY)


def _call_openai(client: OpenAI, system_prompt: str, user_prompt: str) -> str:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        stream=False,
    )
    return (response.choices[0].message.content or "").strip()


def _parse_json(content: str) -> Dict:
    try:
        return json.loads(content)
    except Exception:
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(content[start:end + 1])
            except Exception:
                return {}
        return {}


# ---------------------------------------------------------------------------
# Single task episode runner
# ---------------------------------------------------------------------------

def run_helpdesk_episode(
    env: HelpdeskEnv,
    seed: int = 42,
    num_tickets: Optional[int] = None,
    use_llm: bool = False,
) -> Dict:
    """Run one full helpdesk episode with the multi-agent system.
    Orchestrates: reset → triage → support agent loop → resolution.
    Uses heuristic fallbacks by default (no API key needed).
    Args:
        env: The HelpdeskEnv instance (KB persists across calls).
        seed: Random seed for ticket selection.
        num_tickets: Number of tickets (None = all).
        use_llm: If True, uses OpenAI API; if False, uses heuristics.
    Returns:
        Dict with episode results (scores, steps, KB stats).
    """
    task_name = "helpdesk_episode"
    log_start(task=task_name, env="helpdeskenv", model=MODEL_NAME if use_llm else "heuristic")
    response = env.reset(seed=seed, num_tickets=num_tickets)
    print(f"  [INFO] Episode {env.episode_count}: {response.total_tickets} tickets, KB size: {response.kb_size}")
    rewards: List[float] = []
    steps_taken = 0
    done = False
    max_steps = response.total_tickets * 10  # Safety limit
    try:
        while not done and steps_taken < max_steps:
            state = env.state()
            if state.is_done:
                done = True
                break
            ticket = state.current_ticket
            agent = state.current_agent
            if ticket is None or agent is None:
                break
            # Get the action from the appropriate agent
            support_step = state.steps_on_current_ticket  # steps already taken on this ticket
            if agent == AgentRole.TRIAGE:
                action_data = heuristic_triage(ticket)
            elif agent == AgentRole.L1_SUPPORT:
                # L1 step count: subtract 1 for the triage step
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
            # Build the HelpdeskAction
            helpdesk_action = HelpdeskAction(
                ticket_id=ticket.ticket_id,
                agent_role=agent,
                action_type=action_data["action_type"],
                action_value=action_data["action_value"],
            )
            # Step the environment
            try:
                result = env.step(helpdesk_action)
                reward_value = float(result.reward or 0.0)
                rewards.append(reward_value)
                steps_taken += 1
                done = result.done
                action_log = f"{agent.value}:{action_data['action_type']}"
                log_step(
                    step=steps_taken,
                    action=action_log,
                    reward=reward_value,
                    done=done,
                    error=None,
                )
            except Exception as exc:
                steps_taken += 1
                log_step(step=steps_taken, action="error", reward=0.0, done=True, error=str(exc))
                done = True
        # Calculate final score
        score = sum(rewards) / len(rewards) if rewards else 0.0
        score = max(0.0, min(1.0, score))
        success = score >= SUCCESS_SCORE_THRESHOLD
    except Exception as exc:
        print(f"  [ERROR] Episode failed: {exc}", file=sys.stderr, flush=True)
        score = 0.0
        success = False
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    # Collect results
    kb_stats = env.kb().stats()
    return {
        "episode": env.episode_count,
        "score": round(score, 4),
        "steps": steps_taken,
        "rewards": rewards,
        "success": success,
        "kb_size": kb_stats["total_entries"],
        "kb_agent_entries": kb_stats["agent_created_entries"],
    }
def run_helpdesk_multi_episode(num_episodes: int = 3, num_tickets: Optional[int] = None) -> None:
    """Run multiple helpdesk episodes to demonstrate self-improvement.
    The KB persists across episodes, so performance should improve
    as agents write more KB articles.
    Args:
        num_episodes: Number of episodes to run.
        num_tickets: Tickets per episode (None = all).
    """
    print("=" * 60)
    print("HelpdeskEnv — Multi-Episode Self-Improvement Demo")
    print("=" * 60)
    env = HelpdeskEnv()
    results = []
    for i in range(num_episodes):
        seed = 42 + i  # Different seed each episode for variety
        print(f"\n{'─' * 60}")
        print(f"Episode {i + 1}/{num_episodes} (seed={seed})")
        print(f"{'─' * 60}")
        result = run_helpdesk_episode(env, seed=seed, num_tickets=num_tickets)
        results.append(result)
    # Summary
    print(f"\n{'=' * 60}")
    print("Multi-Episode Summary")
    print(f"{'=' * 60}")
    print(f"{'Ep':>3} {'Score':>7} {'Steps':>6} {'KB Size':>8} {'KB New':>7} {'Success':>8}")
    print(f"{'─' * 43}")
    for r in results:
        print(
            f"{r['episode']:3d} "
            f"{r['score']:7.4f} "
            f"{r['steps']:6d} "
            f"{r['kb_size']:8d} "
            f"{r['kb_agent_entries']:7d} "
            f"{'PASS' if r['success'] else 'FAIL':>8}"
        )
    # Show improvement
    if len(results) >= 2:
        first = results[0]["score"]
        last = results[-1]["score"]
        improvement = last - first
        if improvement > 0:
            print(f"\n[IMPROVED] Score improved by {improvement:.4f} across {num_episodes} episodes!")
        else:
            print(f"\nScore delta: {improvement:.4f} across {num_episodes} episodes")
    print(f"\nFinal KB: {results[-1]['kb_size']} entries "
          f"({results[-1]['kb_agent_entries']} agent-created)")

# ---------------------------------------------------------------------------
# Main — run 3 separate episodes
# ---------------------------------------------------------------------------


def main() -> None:
    """Run all inference tasks for HelpdeskEnv."""
    client = _get_client()
    run_helpdesk_multi_episode(num_episodes=3)
if __name__ == "__main__":
    main()