# graders.py
"""Consolidated graders for all email tasks and helpdesk ticket actions."""
from models import EmailTask, AgentAction, StepResult
# ============================================================================
# Task 1: Spam Classification Grader
# ============================================================================
VALID_LABELS = {"spam", "not_spam"}
def grade_spam(task: EmailTask, action: AgentAction) -> StepResult:
    """
    Grades Task 1: Spam Classification.
    Scoring:
    - 1.0 -> correct label
    - 0.0 -> wrong label or invalid input
    """
    submitted = action.action_value.strip().lower()
    correct = task.ground_truth.strip().lower()
    if submitted not in VALID_LABELS:
        return StepResult(
            task_id = task.task_id,
            reward = 0.0,
            done = False,
            feedback = (
                f"Invalid action '{submitted}'. "
                f"Must be one of: {VALID_LABELS}."
            ),
            correct_answer = correct
        )
    if submitted == correct:
        reward = 1.0
        feedback = f"Correct! This email is '{correct}'."
    else:
        reward = 0.0
        feedback = (
            f"Incorrect. You said '{submitted}' "
            f"but the correct answer is '{correct}'."
        )
    return StepResult(
        task_id = task.task_id,
        reward = reward,
        done = False,
        feedback = feedback,
        correct_answer = correct
    )
# ============================================================================
# Task 2: Email Prioritization Grader (Updated for 4-level scale)
# ============================================================================
# MODIFIED: Added "critical" as the 4th priority level.
# The numeric scale maps to ordinal positions so that distance-based
# partial credit works: |predicted - actual| / max_distance gives a
# normalized error that converts to a reward.
VALID_PRIORITIES = {"low", "medium", "high", "critical"}
PRIORITY_SCALE = {
    "low": 0,
    "medium": 1,
    "high": 2,
    "critical": 3,   # NEW: highest priority level
}
# Maximum possible distance between any two priority levels.
# Used to normalize the distance into a 0.0–1.0 reward.
_MAX_PRIORITY_DISTANCE = max(PRIORITY_SCALE.values()) - min(PRIORITY_SCALE.values())
def grade_priority(task: EmailTask, action: AgentAction) -> StepResult:
    """
    Grades Task 2: Email Prioritization.
    Scoring (4-level scale with distance-based partial credit):
    - distance 0 -> 1.0  (exact match)
    - distance 1 -> 0.67 (one level off)
    - distance 2 -> 0.33 (two levels off)
    - distance 3 -> 0.0  (completely wrong)
    The formula: reward = round(1.0 - (distance / max_distance), 2)
    This generalizes cleanly to any number of priority levels.
    """
    submitted = action.action_value.strip().lower()
    correct = task.ground_truth.strip().lower()
    if submitted not in VALID_PRIORITIES:
        return StepResult(
            task_id = task.task_id,
            reward = 0.0,
            done = False,
            feedback = (
                f"Invalid action '{submitted}'."
                f"Must be one of: {VALID_PRIORITIES}."
            ),
            correct_answer = correct
        )
    
    distance = abs(PRIORITY_SCALE[submitted] - PRIORITY_SCALE[correct])
    # Normalized reward: 0 distance = 1.0, max distance = 0.0
    reward = round(1.0 - (distance / _MAX_PRIORITY_DISTANCE), 2)
    if distance == 0:
        feedback = f"Correct! Priority is '{correct}'."
    elif distance == 1:
        feedback = (
            f"Partially correct. You said '{submitted}' "
            f"but the correct priority is '{correct}'. "
            f"You were one level off."
        )
    elif distance == 2:
        feedback = (
            f"Incorrect. You said '{submitted}' "
            f"but the correct priority is '{correct}'. "
            f"That's two levels off."
        )
    else:
        feedback = (
            f"Incorrect. You said '{submitted}' "
            f"but the correct priority is '{correct}'. "
            f"That's three levels off — completely wrong."
        )
    return StepResult(
        task_id = task.task_id,
        reward = reward,
        done = False,
        feedback = feedback,
        correct_answer = correct
    )
# ============================================================================
# Task 3: Reply Generation Grader
# ============================================================================
POLITE_WORDS = [
    "sorry", "apologies", "apologize", "thank you", "thanks",
    "appreciate", "understand", "please", "kindly", "sincerly",
    "best regards", "warm regards", "happy to help", "feel free"
]
NEGATIVE_WORDS = [
    "unfortunately can't help", "not my problem", "you should have",
    "that's wrong", "i don't care", "whatever", "figure it out"
]
MIN_WORDS = 30
IDEAL_MIN_WORDS = 60
IDEAL_MAX_WORDS = 200
def _score_politeness(reply: str) -> tuple[float, str]:
    """
    Checks how many polite phrases appear in the reply.
    Returns a score (0.0 - 1.0) and feedback string.
    """
    reply_lower = reply.lower()
    for word in NEGATIVE_WORDS:
        if word in reply_lower:
            return 0.0, f"Reply contains inappropriate phrase: '{word}'"
    hits = sum(1 for word in POLITE_WORDS if word in reply_lower)
    if hits >= 3:
        return 1.0, "Reply is very polite and professional."
    elif hits == 2:
        return 0.75, "Reply is fairly polite (2 polite signals found)."
    elif hits == 1:
        return 0.5, "Reply has minimal politeness (1 polite signal found)."
    else:
        return 0.25, "Reply lacks polite language."
def _score_length(reply: str) -> tuple[float, str]:
    """
    Checks whether the reply is a reasonable length.
    Returns a score (0.0 - 1.0) and feedback string.
    """
    word_count = len(reply.split())
    if word_count < MIN_WORDS:
        return 0.25, f"Reply too short ({word_count} words). Minimum is {MIN_WORDS}."
    elif IDEAL_MIN_WORDS <= word_count <= IDEAL_MAX_WORDS:
        return 1.0, f"Reply length is ideal ({word_count} words)."
    elif word_count > IDEAL_MAX_WORDS:
        return 0.75, f"Reply is slightly long ({word_count} words.) Aim for under {IDEAL_MAX_WORDS}."
    else:
        return 0.75, f"Reply is acceptable but brief ({word_count} words)."
def _score_relevance(reply: str, task: EmailTask) -> tuple[float, str]:
    """
    Checks whether the reply references key words from the email.
    A relevant reply should mention something from the subject or body.
    Returns a score (0.0 - 1.0) and feedback string.
    """
    reply_lower = reply.lower()
    subject_words = set(task.subject.lower().split())
    body_words = set(task.body[:100].lower().split())
    keywords = subject_words.union(body_words)
    stopwords = {
        "the", "a", "an", "is", "it", "in", "on", "at", "to",
        "for", "of", "and", "or", "but", "i", "you", "we", "my",
        "your", "this", "that", "with", "have", "has", "be", "are"
    }
    keywords = {w for w in keywords if w not in stopwords and len(w) > 3}
    hits = sum(1 for kw in keywords if kw in reply_lower)
    if hits >= 4:
        return 1., f"Reply is clearly relevant ({hits} keyword matches)."
    elif hits >= 2:
        return 0.75, f"Reply is somewhat relevant ({hits} keyword matches)."
    elif hits == 1:
        return 0.5, f"Reply barely references the email ({hits} keyword match)."
    else:
        return 0.25, "Reply seems unrelated to the email."
def grade_reply(task: EmailTask, action: AgentAction) -> StepResult:
    """
    Grades Task 3: Drafting Polite Replies.
    Final score is a weighted average of three heuristics:
    - Politeness: 40%
    - Length: 30%;
    - Relevance: 30%
    """
    reply = action.action_value.strip()
    if not reply:
        return StepResult(
            task_id = task.task_id,
            reward = 0.0,
            done = False,
            feedback = "No reply was submitted.",
            correct_answer = task.ground_truth
        )
    
    politeness_score, politeness_fb = _score_politeness(reply)
    length_score, length_fb = _score_length(reply)
    relevance_score, relevance_fb = _score_relevance(reply, task)
    
    final_reward = round(
        (politeness_score * 0.4) +
        (length_score * 0.3) +
        (relevance_score * 0.3),
        2
    )
    feedback = (
        f"Reply Score Breakdown:\n"
        f" Politeness (40%): {politeness_score:.2f} - {politeness_fb}\n"
        f" Length (30%): {length_score:.2f} - {length_fb}\n"
        f" Relevance (30%): {relevance_score:.2f} - {relevance_fb}\n"
        f"_______________________________________________\n"
        f" Final Score: {final_reward:.2f}"
    )
    return StepResult(
        task_id = task.task_id,
        reward = final_reward,
        done = False,
        feedback = feedback,
        correct_answer = task.ground_truth
    )
# ============================================================================
# Quick Validation (run with: python graders.py)
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Priority Grader — 4-Level Scale Validation")
    print("=" * 60)
    # Test all distances for "critical" as ground truth
    test_cases = [
        ("critical", "critical", 1.0),
        ("high",     "critical", 0.67),
        ("medium",   "critical", 0.33),
        ("low",      "critical", 0.0),
        # Test with "low" as ground truth (reverse direction)
        ("low",      "low",      1.0),
        ("medium",   "low",      0.67),
        ("high",     "low",      0.33),
        ("critical", "low",      0.0),
        # Test middle values
        ("medium",   "high",     0.67),
        ("high",     "medium",   0.67),
    ]
    all_passed = True
    for submitted, correct, expected in test_cases:
        task = EmailTask(
            task_id="test",
            task_type="priority",
            subject="Test",
            sender="test@test.com",
            body="Test body",
            ground_truth=correct,
        )
        action = AgentAction(task_id="test", action_value=submitted)
        result = grade_priority(task, action)
        status = "✅" if result.reward == expected else "❌"
        if result.reward != expected:
            all_passed = False
        print(
            f"  {status} predicted={submitted:8s} actual={correct:8s} "
            f"→ reward={result.reward:.2f} (expected {expected:.2f})"
        )
    print()
    if all_passed:
        print("All tests passed! 4-level priority grader is working correctly.")
    else:
        print("SOME TESTS FAILED — check the output above.")