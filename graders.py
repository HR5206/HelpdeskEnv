"""Consolidated graders for all three email tasks."""

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
# Task 2: Email Prioritization Grader
# ============================================================================

VALID_PRIORITIES = {"high", "medium", "low"}

PRIORITY_SCALE = {
    "high": 2,
    "medium": 1, 
    "low": 0
}

def grade_priority(task: EmailTask, action: AgentAction) -> StepResult:

    """
    Grades Task 2: Email Prioritization.

    Scoring:

    - 1.0 -> exact match
    - 0.5 -> one level off
    - 0.0 -> two levels off
    - 0.0 -> invalid input

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

    if distance == 0:
        reward = 1.0
        feedback = f"Correct! Priority is '{correct}'."
    elif distance == 1:
        reward = 0.5
        feedback = (
            f"Partially correct. You said '{submitted}' "
            f"but the correct priority is '{correct}'. "
            f"You were one level off."
        )
    else:
        reward = 0.0
        feedback = (
            f"Incorrect. You said '{submitted}' "
            f"but the correct priority is '{correct}'."
            f"That's two levels off."
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
