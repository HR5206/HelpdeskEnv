"""EmailEnv Inference Script"""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Optional

from openai import OpenAI

from emailenv_class import EmailEnv
from models import Action, TaskType, AgentAction, ErrorResponse

API_BASE_URL: str = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-4o-mini")
API_KEY: Optional[str] = (
    os.getenv("HF_TOKEN")
    or os.getenv("API_KEY")
    or os.getenv("OPENAI_API_KEY")
)
BENCHMARK: str = "emailenv"
MAX_STEPS: int = 10
TEMPERATURE: float = 0.0
MAX_TOKENS: int = 256
SUCCESS_SCORE_THRESHOLD: float = 0.5

TASK_SEQUENCE = [
    "spam_classification",
    "email_prioritization",
    "reply_generation",
]


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
# Heuristic fallbacks
# ---------------------------------------------------------------------------

def _heuristic_is_spam(subject: str, body: str) -> bool:
    text = (subject + " " + body).lower()
    return any(k in text for k in ["discount", "offer", "congratulations", "winner", "prize", "lottery", "free"])


def _heuristic_priority(subject: str, body: str) -> str:
    text = (subject + " " + body).lower()
    if any(k in text for k in ["urgent", "critical", "unable", "failed", "error", "stuck", "cannot"]):
        return "high"
    if any(k in text for k in ["invoice", "reminder", "upgrade", "question", "request", "feature"]):
        return "medium"
    return "low"


def _heuristic_reply(subject: str, body: str) -> str:
    return (
        f"Thank you for reaching out regarding '{subject}'. "
        "We appreciate your message and sincerely apologize for any inconvenience. "
        "Our team has received your request and will look into this promptly. "
        "Please feel free to reply if you have additional questions. "
        "Best regards, Support Team"
    )


# ---------------------------------------------------------------------------
# Action builders
# ---------------------------------------------------------------------------

def _build_spam_action(client: Optional[OpenAI], subject: str, body: str, sender: str) -> Action:
    if client is None:
        return Action(type="classify_spam", is_spam=_heuristic_is_spam(subject, body))
    system_prompt = (
        "You classify emails as spam or not spam. "
        'Respond only with JSON: {"type":"classify_spam","is_spam":true|false}.'
    )
    try:
        data = _parse_json(_call_openai(client, system_prompt, f"Subject: {subject}\nSender: {sender}\nBody: {body}"))
        return Action(type="classify_spam", is_spam=bool(data.get("is_spam", False)))
    except Exception:
        return Action(type="classify_spam", is_spam=_heuristic_is_spam(subject, body))


def _build_priority_action(client: Optional[OpenAI], subject: str, body: str, sender: str) -> Action:
    if client is None:
        return Action(type="set_priority", priority=_heuristic_priority(subject, body))
    system_prompt = (
        "You assign priority to customer emails. "
        'Respond only with JSON: {"type":"set_priority","priority":"low"|"medium"|"high"}.'
    )
    try:
        data = _parse_json(_call_openai(client, system_prompt, f"Subject: {subject}\nSender: {sender}\nBody: {body}"))
        priority = str(data.get("priority", "medium")).lower()
        if priority not in {"low", "medium", "high"}:
            priority = "medium"
        return Action(type="set_priority", priority=priority)
    except Exception:
        return Action(type="set_priority", priority=_heuristic_priority(subject, body))


def _build_reply_action(client: Optional[OpenAI], subject: str, body: str, sender: str) -> Action:
    if client is None:
        return Action(type="generate_reply", reply_text=_heuristic_reply(subject, body))
    system_prompt = (
        "You write professional, polite customer support replies. "
        'Respond only with JSON: {"type":"generate_reply","reply_text":"..."}.'
    )
    try:
        data = _parse_json(_call_openai(client, system_prompt, f"Subject: {subject}\nSender: {sender}\nBody: {body}"))
        reply = str(data.get("reply_text", "")).strip() or _heuristic_reply(subject, body)
        return Action(type="generate_reply", reply_text=reply)
    except Exception:
        return Action(type="generate_reply", reply_text=_heuristic_reply(subject, body))


def _format_action_log(action: Action) -> str:
    if action.type == "classify_spam":
        return f"classify_spam(is_spam={str(bool(action.is_spam)).lower()})"
    if action.type == "set_priority":
        return f"set_priority(priority={action.priority})"
    if action.type == "generate_reply":
        snippet = (action.reply_text or "").replace("\n", " ")
        if len(snippet) > 80:
            snippet = snippet[:77] + "..."
        return f"generate_reply(reply_text={snippet})"
    return "skip"


# ---------------------------------------------------------------------------
# Single task episode runner
# ---------------------------------------------------------------------------

def run_episode(task_name: str, client: Optional[OpenAI]) -> None:
    """Run one episode logged under the given task_name."""

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    env = EmailEnv()
    env.reset(seed=42)

    rewards: List[float] = []
    steps_taken = 0
    done = False
    score = 0.0
    success = False

    try:
        while not done and steps_taken < MAX_STEPS:
            state = env.state()
            current_task = state.current_task

            if current_task is None:
                break

            subject = current_task.subject
            body = current_task.body
            sender = current_task.sender

            if current_task.task_type == TaskType.SPAM:
                action = _build_spam_action(client, subject, body, sender)
            elif current_task.task_type == TaskType.PRIORITY:
                action = _build_priority_action(client, subject, body, sender)
            elif current_task.task_type == TaskType.REPLY:
                action = _build_reply_action(client, subject, body, sender)
            else:
                action = Action(type="skip")

            action_str = _format_action_log(action)

            if action.type == "classify_spam" and action.is_spam is not None:
                action_value = "spam" if action.is_spam else "not_spam"
            elif action.type == "set_priority" and action.priority is not None:
                action_value = action.priority
            elif action.type == "generate_reply" and action.reply_text is not None:
                action_value = action.reply_text
            else:
                action_value = "skip"

            agent_action = AgentAction(task_id=current_task.task_id, action_value=action_value)

            try:
                result = env.step(agent_action)
                if isinstance(result, ErrorResponse):
                    steps_taken += 1
                    log_step(step=steps_taken, action=action_str, reward=0.0, done=True, error=result.error)
                    done = True
                    break

                reward_value = float(result.reward or 0.0)
                rewards.append(reward_value)
                steps_taken += 1
                done = result.done
                log_step(step=steps_taken, action=action_str, reward=reward_value, done=done, error=None)

            except Exception as exc:
                steps_taken += 1
                log_step(step=steps_taken, action=action_str, reward=0.0, done=True, error=str(exc))
                done = True

        score = sum(rewards) / len(rewards) if rewards else 0.0
        score = max(0.0, min(1.0, score))
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Episode failed: {exc}", file=sys.stderr, flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ---------------------------------------------------------------------------
# Main — run 3 separate episodes, one per task
# ---------------------------------------------------------------------------

def main() -> None:
    client = _get_client()
    for task_name in TASK_SEQUENCE:
        run_episode(task_name, client)


if __name__ == "__main__":
    main()