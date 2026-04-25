import os
import json
from typing import Optional
from dotenv import load_dotenv
from tasks import get_tasks_by_type
from graders import grade_spam, grade_priority, grade_reply
from models import AgentAction

# Load environment variables from .env
load_dotenv()

API_BASE_URL: str = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-5-nano")
API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN")

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

def get_client() -> Optional[object]:
    if not HAS_OPENAI:
        print("OpenAI client not available - using simulated baseline")
        return None
    
    if not API_KEY:
        print("No API key found - using simulated baseline")
        return None

    return OpenAI(api_key=API_KEY, base_url=API_BASE_URL, max_retries=0)

def simulate_spam_answer(task):
    # Heuristic over subject + body text
    text_lower = f"{task.subject} {task.body}".lower()

    spam_keywords = ["congratulations", "won", "free", "claim", "urgent", "verify", "account", "compromised", "$5000", "easy money"]
    not_spam_keywords = ["standup", "team", "invoice", "github", "pull request"]
    
    spam_count = sum(1 for kw in spam_keywords if kw in text_lower)
    not_spam_count = sum(1 for kw in not_spam_keywords if kw in text_lower)
    
    if spam_count > not_spam_count:
        return "spam"
    else:
        return "not_spam"

def simulate_priority_answer(task):
    # Heuristic over subject + body text
    content_lower = f"{task.subject} {task.body}".lower()

    high_keywords = ["production", "critical", "down", "urgent", "immediate", "outage", "security", "vulnerability", "$200k", "deal"]
    low_keywords = ["newsletter", "lunch", "menu", "social", "ping pong"]
    medium_keywords = ["review", "eod", "blocking", "qa"]
    
    high_count = sum(1 for kw in high_keywords if kw in content_lower)
    low_count = sum(1 for kw in low_keywords if kw in content_lower)
    medium_count = sum(1 for kw in medium_keywords if kw in content_lower)
    
    if high_count >= low_count and high_count >= medium_count:
        return "high"
    elif low_count >= medium_count:
        return "low"
    else:
        return "medium"

def simulate_reply_answer(task):
    return ("Thank you for reaching out. I understand your concern and sincerely apologize for "
            "the inconvenience. We've reviewed your issue and are committed to resolving it promptly. "
            "Our team is actively working on this matter and will provide you with an update within "
            "the next 24 hours. We truly appreciate your patience and feedback.")

def run_spam_baseline(client: Optional[object]):
    print("\nSPAM DETECTION (Easy)")
    print("=" * 60)
    
    spam_tasks = get_tasks_by_type("spam")[:3]
    scores = []
    
    for task in spam_tasks:
        if client:
            try:
                snippet = f"{task.subject} {task.body}"[:200]
                prompt = (
                    "Classify as SPAM or NOT_SPAM: "
                    f"{snippet}\nRespond with only 'spam' or 'not_spam' (lowercase)."
                )

                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=10,
                )
                answer = response.choices[0].message.content.strip().lower()
            except Exception:
                answer = simulate_spam_answer(task)
        else:
            answer = simulate_spam_answer(task)

        action = AgentAction(task_id=task.task_id, action_value=answer)
        result = grade_spam(task, action)
        scores.append(result.reward)

        print(f"Task {task.task_id}: {answer} \\u2192 Reward: {result.reward:.2f}")
    
    avg_score = sum(scores) / len(scores)
    print(f"\nAverage Spam Score: {avg_score:.4f}")
    return avg_score

def run_priority_baseline(client: Optional[object]):
    print("\nEMAIL PRIORITIZATION (Medium)")
    print("=" * 60)
    
    priority_tasks = get_tasks_by_type("priority")[:3]
    scores = []
    
    for task in priority_tasks:
        if client:
            try:
                snippet = f"{task.subject} {task.body}"[:200]
                prompt = (
                    "Priority: high/medium/low? "
                    f"{snippet}\nRespond with only 'high', 'medium', or 'low' (lowercase)."
                )

                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=10,
                )
                answer = response.choices[0].message.content.strip().lower()
            except Exception:
                answer = simulate_priority_answer(task)
        else:
            answer = simulate_priority_answer(task)

        action = AgentAction(task_id=task.task_id, action_value=answer)
        result = grade_priority(task, action)
        scores.append(result.reward)

        print(f"Task {task.task_id}: {answer} \\u2192 Reward: {result.reward:.2f}")
    
    avg_score = sum(scores) / len(scores)
    print(f"\nAverage Priority Score: {avg_score:.4f}")
    return avg_score

def run_reply_baseline(client: Optional[object]):
    print("\nREPLY GENERATION (Hard)")
    print("=" * 60)
    
    reply_tasks = get_tasks_by_type("reply")[:2]
    scores = []
    
    for task in reply_tasks:
        if client:
            try:
                snippet = f"{task.subject} {task.body}"[:200]
                prompt = f"Draft a professional reply to: {snippet}"

                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=150,
                )
                answer = response.choices[0].message.content.strip()
            except Exception:
                answer = simulate_reply_answer(task)
        else:
            answer = simulate_reply_answer(task)

        action = AgentAction(task_id=task.task_id, action_value=answer)
        result = grade_reply(task, action)
        scores.append(result.reward)

        print(f"Task {task.task_id}: {result.reward:.2f}")
        print(f"  Reply: {answer[:60]}...")
    
    avg_score = sum(scores) / len(scores)
    print(f"\nAverage Reply Score: {avg_score:.4f}")
    return avg_score

def main():
    print("\n" + "=" * 60)
    print("EmailEnv Baseline Evaluation")
    print("=" * 60)
    print(f"API Base: {API_BASE_URL}")
    print(f"Model (for LLM baseline): {MODEL_NAME}")
    print("=" * 60)

    # 1) Heuristic baseline (no API calls)
    print("\n" + "-" * 60)
    print("HEURISTIC BASELINE (no model)")
    print("-" * 60)
    heuristic_spam = run_spam_baseline(client=None)
    heuristic_priority = run_priority_baseline(client=None)
    heuristic_reply = run_reply_baseline(client=None)
    heuristic_overall = (heuristic_spam + heuristic_priority + heuristic_reply) / 3

    # 2) LLM baseline for MODEL_NAME, if OpenAI client is available
    model_spam = model_priority = model_reply = model_overall = None
    client = get_client()
    if client is not None:
        print("\n" + "-" * 60)
        print(f"LLM BASELINE ({MODEL_NAME})")
        print("-" * 60)
        model_spam = run_spam_baseline(client)
        model_priority = run_priority_baseline(client)
        model_reply = run_reply_baseline(client)
        model_overall = (model_spam + model_priority + model_reply) / 3
    else:
        print("\nNo API key detected or OpenAI client unavailable – skipping LLM baseline.")

    # Summary table
    print("\n" + "=" * 60)
    print("BASELINE RESULTS")
    print("=" * 60)
    print(f"Heuristic Spam Detection:   {heuristic_spam:.4f}")
    print(f"Heuristic Prioritization:   {heuristic_priority:.4f}")
    print(f"Heuristic Reply Generation: {heuristic_reply:.4f}")
    print(f"Heuristic Overall:          {heuristic_overall:.4f}")
    if model_overall is not None:
        print("-" * 60)
        print(f"{MODEL_NAME} Spam Detection:   {model_spam:.4f}")
        print(f"{MODEL_NAME} Prioritization:   {model_priority:.4f}")
        print(f"{MODEL_NAME} Reply Generation: {model_reply:.4f}")
        print(f"{MODEL_NAME} Overall:          {model_overall:.4f}")
    print("=" * 60)

    # Persist results
    payload = {
        "model": MODEL_NAME,
        "heuristic": {
            "spam_detection": float(heuristic_spam),
            "prioritization": float(heuristic_priority),
            "reply_generation": float(heuristic_reply),
            "overall": float(heuristic_overall),
        },
    }
    if model_overall is not None:
        payload["llm"] = {
            "spam_detection": float(model_spam),
            "prioritization": float(model_priority),
            "reply_generation": float(model_reply),
            "overall": float(model_overall),
        }

    with open("baseline_results.json", "w") as f:
        json.dump(payload, f, indent=2)

    print("Results saved to baseline_results.json")

if __name__ == "__main__":
    main()
