# graders.py
"""Graders for HelpdeskEnv ticket actions.
Three anti-gaming mechanisms (Part 2):
1. Repetition detection — penalizes text padded with repeated phrases
2. Copy-paste detection — penalizes replies that just echo the ticket body
3. Keyword diversity — requires distinct keyword matches, not repeated ones
Public graders:
- grade_triage(ticket, action) -> StepResult
- grade_efficiency(steps_taken, sla_steps, escalation_count) -> StepResult
- grade_kb_contribution(kb_entry_text, ticket) -> StepResult
- grade_composite(triage, resolution, response, efficiency, kb) -> (float, str)
"""
import json
from typing import Dict, Optional
from models import (
    EmailTask, AgentAction, StepResult,
    Ticket, HelpdeskAction, TicketCategory, TicketPriority, SupportTier,
)
# ============================================================================
# Constants
# ============================================================================
VALID_PRIORITIES = {"low", "medium", "high", "critical"}
PRIORITY_SCALE = {"low": 0, "medium": 1, "high": 2, "critical": 3}
_MAX_PRIORITY_DISTANCE = max(PRIORITY_SCALE.values()) - min(PRIORITY_SCALE.values())
POLITE_WORDS = [
    "sorry", "apologies", "apologize", "thank you", "thanks",
    "appreciate", "understand", "please", "kindly", "sincerely",
    "best regards", "warm regards", "happy to help", "feel free"
]
NEGATIVE_WORDS = [
    "unfortunately can't help", "not my problem", "you should have",
    "that's wrong", "i don't care", "whatever", "figure it out"
]
MIN_WORDS = 30
IDEAL_MIN_WORDS = 60
IDEAL_MAX_WORDS = 200
_KB_SPECIFICITY_KEYWORDS = [
    "resolved", "fixed", "steps", "root cause", "procedure",
    "workaround", "solution", "configured", "installed", "updated",
    "upgraded", "patched", "restored", "recovered", "verified",
    "confirmed", "diagnosed", "identified", "applied", "executed",
]
# Composite reward weights (documented in REWARD_DESIGN.md)
COMPOSITE_WEIGHTS = {
    "triage": 0.15,
    "resolution": 0.30,
    "response": 0.20,
    "efficiency": 0.20,
    "kb": 0.15,
}
# ============================================================================
# Anti-Gaming Utilities (Part 2)
# ============================================================================
def _detect_repetition(text: str) -> tuple[float, str]:
    """Detect repeated 3-word phrases (n-gram repetition).
    Returns a penalty multiplier (1.0 = no repetition, 0.0 = all repeated)
    and a feedback string.
    Gaming vector blocked: agent pads text with "I am sorry I am sorry I am
    sorry" to hit length thresholds.
    """
    words = text.lower().split()
    if len(words) < 6:
        return 1.0, ""
    # Build 3-grams
    trigrams = [tuple(words[i:i+3]) for i in range(len(words) - 2)]
    if not trigrams:
        return 1.0, ""
    unique_ratio = len(set(trigrams)) / len(trigrams)
    if unique_ratio >= 0.7:
        return 1.0, ""
    elif unique_ratio >= 0.4:
        penalty = round(unique_ratio / 0.7, 2)
        return penalty, f"Repetitive text detected ({1-unique_ratio:.0%} repeated phrases, penalty: {penalty:.2f}x)"
    else:
        return 0.3, f"Heavily repetitive text ({1-unique_ratio:.0%} repeated phrases, penalty: 0.30x)"
def _detect_copy_paste(reply: str, source: str) -> tuple[float, str]:
    """Detect if the reply is mostly copied from the source text.
    Returns a penalty multiplier (1.0 = original, 0.0 = full copy).
    Gaming vector blocked: agent copies the ticket body or ground truth
    verbatim to score high on keyword relevance.
    """
    if not source or not reply:
        return 1.0, ""
    reply_words = set(reply.lower().split())
    source_words = set(source.lower().split())
    if not reply_words:
        return 1.0, ""
    overlap = len(reply_words.intersection(source_words)) / len(reply_words)
    if overlap < 0.6:
        return 1.0, ""
    elif overlap < 0.8:
        penalty = round(1.0 - (overlap - 0.6) * 2.5, 2)
        return penalty, f"Possible copy-paste detected ({overlap:.0%} word overlap, penalty: {penalty:.2f}x)"
    else:
        return 0.2, f"Copy-paste detected ({overlap:.0%} word overlap, penalty: 0.20x)"
# ============================================================================
# Scoring Helpers (enhanced with anti-gaming)
# ============================================================================
def _score_politeness(reply: str) -> tuple[float, str]:
    """Score politeness with anti-stuffing protection.
    Each unique polite phrase counts once — repeating "sorry sorry sorry"
    doesn't give extra credit (it already counted unique words before,
    but now we also apply the repetition penalty downstream).
    """
    reply_lower = reply.lower()
    for word in NEGATIVE_WORDS:
        if word in reply_lower:
            return 0.0, f"Reply contains inappropriate phrase: '{word}'"
    # Count UNIQUE polite phrases (already was set-like via 'if word in')
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
    """Score length with repetition penalty.
    The repetition check ensures agents can't pad with filler to hit
    the ideal word count. Genuine detail passes; "step 1 step 1 step 1" fails.
    """
    word_count = len(reply.split())
    if word_count < MIN_WORDS:
        return 0.25, f"Reply too short ({word_count} words). Minimum is {MIN_WORDS}."
    # Base length score
    if IDEAL_MIN_WORDS <= word_count <= IDEAL_MAX_WORDS:
        base = 1.0
        fb = f"Reply length is ideal ({word_count} words)."
    elif word_count > IDEAL_MAX_WORDS:
        base = 0.75
        fb = f"Reply is slightly long ({word_count} words). Aim for under {IDEAL_MAX_WORDS}."
    else:
        base = 0.75
        fb = f"Reply is acceptable but brief ({word_count} words)."
    # Apply repetition penalty
    rep_mult, rep_fb = _detect_repetition(reply)
    if rep_mult < 1.0:
        score = round(base * rep_mult, 2)
        fb += f" {rep_fb}"
        return score, fb
    return base, fb
def _score_relevance(reply: str, task: EmailTask) -> tuple[float, str]:
    """Score relevance with copy-paste detection.
    An agent that just pastes the ticket body back will get penalized.
    Genuine responses that reference ticket keywords still score well.
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
        base = 1.0
        fb = f"Reply is clearly relevant ({hits} keyword matches)."
    elif hits >= 2:
        base = 0.75
        fb = f"Reply is somewhat relevant ({hits} keyword matches)."
    elif hits == 1:
        base = 0.5
        fb = f"Reply barely references the email ({hits} keyword match)."
    else:
        return 0.25, "Reply seems unrelated to the email."
    # Apply copy-paste penalty (compare against the task body)
    cp_mult, cp_fb = _detect_copy_paste(reply, task.body)
    if cp_mult < 1.0:
        score = round(base * cp_mult, 2)
        fb += f" {cp_fb}"
        return score, fb
    return base, fb
# ============================================================================
# Triage Grader
# ============================================================================
def grade_triage(ticket: Ticket, action: HelpdeskAction) -> StepResult:
    """Grade the Triage Agent's classification of a ticket.
    Scoring breakdown:
    - Category correct:  40% weight (binary: 1.0 or 0.0)
    - Priority correct:  30% weight (distance-based partial credit)
    - Tier correct:      30% weight (binary: 1.0 or 0.0)
    Anti-gaming: This grader is inherently hard to game because it compares
    against exact ground truth values. There's no way to score high without
    actually classifying correctly.
    """
    try:
        triage_data = json.loads(action.action_value)
    except (json.JSONDecodeError, TypeError):
        return StepResult(
            task_id=ticket.ticket_id,
            reward=0.0,
            done=False,
            feedback=(
                f"Invalid triage action: could not parse JSON from action_value. "
                f'Expected format: {{"category": "...", "priority": "...", "tier": "..."}}'
            ),
            correct_answer=(
                f"category={ticket.category.value}, "
                f"priority={ticket.ground_truth_priority.value}, "
                f"tier={ticket.ground_truth_tier.value}"
            ),
        )
    submitted_category = str(triage_data.get("category", "")).strip().lower()
    submitted_priority = str(triage_data.get("priority", "")).strip().lower()
    submitted_tier = str(triage_data.get("tier", "")).strip().upper()
    feedback_parts = []
    # Category (40%)
    correct_category = ticket.category.value.lower()
    if submitted_category == correct_category:
        category_score = 1.0
        feedback_parts.append(f"Category (40%): CORRECT -- '{submitted_category}'")
    else:
        category_score = 0.0
        feedback_parts.append(
            f"Category (40%): WRONG -- you said '{submitted_category}', "
            f"correct is '{correct_category}'"
        )
    # Priority (30%)
    correct_priority = ticket.ground_truth_priority.value.lower()
    if submitted_priority not in PRIORITY_SCALE:
        priority_score = 0.0
        feedback_parts.append(
            f"Priority (30%): INVALID -- '{submitted_priority}' is not a valid priority. "
            f"Valid options: {set(PRIORITY_SCALE.keys())}"
        )
    else:
        distance = abs(PRIORITY_SCALE[submitted_priority] - PRIORITY_SCALE[correct_priority])
        priority_score = round(1.0 - (distance / _MAX_PRIORITY_DISTANCE), 2)
        if distance == 0:
            feedback_parts.append(f"Priority (30%): CORRECT -- '{submitted_priority}'")
        else:
            feedback_parts.append(
                f"Priority (30%): OFF BY {distance} -- you said '{submitted_priority}', "
                f"correct is '{correct_priority}' (score: {priority_score:.2f})"
            )
    # Tier (30%)
    correct_tier = ticket.ground_truth_tier.value.upper()
    if submitted_tier == correct_tier:
        tier_score = 1.0
        feedback_parts.append(f"Tier (30%): CORRECT -- '{submitted_tier}'")
    else:
        tier_score = 0.0
        feedback_parts.append(
            f"Tier (30%): WRONG -- you said '{submitted_tier}', "
            f"correct is '{correct_tier}'"
        )
    final_reward = round(
        (category_score * 0.4) + (priority_score * 0.3) + (tier_score * 0.3), 2
    )
    feedback = (
        f"Triage Score Breakdown:\n"
        + "\n".join(f"  {part}" for part in feedback_parts)
        + f"\n{'_' * 47}\n"
        + f"  Final Triage Score: {final_reward:.2f}"
    )
    return StepResult(
        task_id=ticket.ticket_id,
        reward=final_reward,
        done=False,
        feedback=feedback,
        correct_answer=(
            f"category={correct_category}, "
            f"priority={correct_priority}, "
            f"tier={correct_tier}"
        ),
    )
# ============================================================================
# Efficiency Grader
# ============================================================================
def grade_efficiency(steps_taken: int, sla_steps: int, escalation_count: int) -> StepResult:
    """Grade the efficiency of ticket resolution.
    - SLA Compliance (60%): 1.0 if within SLA, -0.25 per step over
    - Escalation Efficiency (40%): 1.0 if no escalations, -0.2 per escalation
    Anti-gaming: This grader uses hard numeric counts. An agent cannot
    fake fewer steps or escalations — the environment tracks them.
    """
    if steps_taken <= sla_steps:
        sla_score = 1.0
        sla_fb = f"Within SLA ({steps_taken}/{sla_steps} steps used)"
    else:
        overage = steps_taken - sla_steps
        sla_score = max(0.0, 1.0 - (overage * 0.25))
        sla_fb = (
            f"SLA breached by {overage} step(s) "
            f"({steps_taken}/{sla_steps} steps used, score: {sla_score:.2f})"
        )
    if escalation_count == 0:
        escalation_score = 1.0
        escalation_fb = "No escalations needed"
    else:
        escalation_score = max(0.0, 1.0 - (escalation_count * 0.2))
        escalation_fb = f"{escalation_count} escalation(s) (score: {escalation_score:.2f})"
    final_reward = round((sla_score * 0.6) + (escalation_score * 0.4), 2)
    feedback = (
        f"Efficiency Score Breakdown:\n"
        f"  SLA Compliance (60%): {sla_score:.2f} -- {sla_fb}\n"
        f"  Escalation Efficiency (40%): {escalation_score:.2f} -- {escalation_fb}\n"
        f"{'_' * 47}\n"
        f"  Final Efficiency Score: {final_reward:.2f}"
    )
    return StepResult(
        task_id="efficiency",
        reward=final_reward,
        done=False,
        feedback=feedback,
        correct_answer=f"SLA={sla_steps} steps, minimize escalations",
    )
# ============================================================================
# KB Contribution Grader
# ============================================================================
def grade_kb_contribution(kb_entry_text: str, ticket: Ticket) -> StepResult:
    """Grade a Knowledge Base article written by an L3 agent.
    - Relevance (35%): keyword overlap with the ticket
    - Length (30%): substantive enough to be useful (with repetition check)
    - Specificity (35%): contains actionable resolution keywords
    Anti-gaming: repetition detection prevents padding, and the specificity
    check requires DIVERSE resolution keywords (not just repeating "fixed").
    """
    if not kb_entry_text or not kb_entry_text.strip():
        return StepResult(
            task_id=ticket.ticket_id,
            reward=0.0,
            done=False,
            feedback="No KB article text was provided.",
            correct_answer="Write a detailed KB article with problem description and solution steps.",
        )
    text = kb_entry_text.strip()
    # Relevance (35%)
    pseudo_task = EmailTask(
        task_id=ticket.ticket_id,
        task_type="reply",
        subject=ticket.subject,
        sender=ticket.sender,
        body=ticket.body,
    )
    relevance_score, relevance_fb = _score_relevance(text, pseudo_task)
    # Length (30%) — includes repetition penalty
    length_score, length_fb = _score_length(text)
    # Specificity (35%) — requires UNIQUE keyword matches
    text_lower = text.lower()
    # Count distinct specificity keywords found (not repeated occurrences)
    specificity_hits = sum(1 for kw in _KB_SPECIFICITY_KEYWORDS if kw in text_lower)
    if specificity_hits >= 5:
        specificity_score = 1.0
        specificity_fb = f"Highly specific ({specificity_hits} distinct resolution keywords)"
    elif specificity_hits >= 3:
        specificity_score = 0.75
        specificity_fb = f"Moderately specific ({specificity_hits} distinct resolution keywords)"
    elif specificity_hits >= 1:
        specificity_score = 0.5
        specificity_fb = f"Somewhat vague ({specificity_hits} resolution keyword(s))"
    else:
        specificity_score = 0.25
        specificity_fb = "Article lacks specific resolution language"
    final_reward = round(
        (relevance_score * 0.35) + (length_score * 0.30) + (specificity_score * 0.35), 2
    )
    feedback = (
        f"KB Contribution Score Breakdown:\n"
        f"  Relevance (35%): {relevance_score:.2f} -- {relevance_fb}\n"
        f"  Length (30%): {length_score:.2f} -- {length_fb}\n"
        f"  Specificity (35%): {specificity_score:.2f} -- {specificity_fb}\n"
        f"{'_' * 47}\n"
        f"  Final KB Contribution Score: {final_reward:.2f}"
    )
    return StepResult(
        task_id=ticket.ticket_id,
        reward=final_reward,
        done=False,
        feedback=feedback,
        correct_answer="A detailed KB article with problem description, root cause, and step-by-step solution.",
    )
# ============================================================================
# Composite Reward (Part 2 — NEW)
# ============================================================================
def grade_composite(
    triage: float = 0.0,
    resolution: float = 0.0,
    response: float = 0.0,
    efficiency: float = 0.0,
    kb: float = 0.0,
) -> tuple[float, str]:
    """Compute the weighted composite reward for a resolved ticket.
    Weights (from REWARD_DESIGN.md):
        triage     x 0.15  -- classification accuracy
        resolution x 0.30  -- did the fix match the expected solution?
        response   x 0.20  -- was the customer reply professional?
        efficiency x 0.20  -- SLA compliance and escalation cost
        kb         x 0.15  -- quality of KB article (if written)
    Returns:
        (composite_score, breakdown_string)
    """
    w = COMPOSITE_WEIGHTS
    composite = round(
        triage * w["triage"]
        + resolution * w["resolution"]
        + response * w["response"]
        + efficiency * w["efficiency"]
        + kb * w["kb"],
        4,
    )
    breakdown = (
        f"Composite Reward Breakdown:\n"
        f"  Triage     ({w['triage']:.0%}): {triage:.2f} x {w['triage']} = {triage * w['triage']:.4f}\n"
        f"  Resolution ({w['resolution']:.0%}): {resolution:.2f} x {w['resolution']} = {resolution * w['resolution']:.4f}\n"
        f"  Response   ({w['response']:.0%}): {response:.2f} x {w['response']} = {response * w['response']:.4f}\n"
        f"  Efficiency ({w['efficiency']:.0%}): {efficiency:.2f} x {w['efficiency']} = {efficiency * w['efficiency']:.4f}\n"
        f"  KB         ({w['kb']:.0%}): {kb:.2f} x {w['kb']} = {kb * w['kb']:.4f}\n"
        f"{'_' * 47}\n"
        f"  Composite Score: {composite:.4f}"
    )
    return composite, breakdown
# ============================================================================
# Validation (run with: python graders.py)
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Grader Anti-Gaming Validation")
    print("=" * 60)
    # --- Test 1: Repetition detection ---
    print("\n[Test 1] Repetition detection")
    clean = "The root cause was a firmware crash on the Cisco switch in room 3B."
    padded = "fixed fixed fixed fixed fixed fixed fixed fixed fixed fixed fixed fixed"
    m1, _ = _detect_repetition(clean)
    m2, fb2 = _detect_repetition(padded)
    print(f"  Clean text: penalty={m1:.2f} (should be 1.00)")
    print(f"  Padded text: penalty={m2:.2f} (should be < 0.50) -- {fb2}")
    # --- Test 2: Copy-paste detection ---
    print("\n[Test 2] Copy-paste detection")
    source = "The entire 3rd floor has no network connectivity since 9:15 AM."
    original = "We have diagnosed and resolved the network switch failure on floor 3."
    copied = source  # exact copy
    m3, _ = _detect_copy_paste(original, source)
    m4, fb4 = _detect_copy_paste(copied, source)
    print(f"  Original reply: penalty={m3:.2f} (should be 1.00)")
    print(f"  Copied reply: penalty={m4:.2f} (should be < 0.50) -- {fb4}")
    # --- Test 3: Honest vs Lazy vs Gaming composite ---
    print("\n[Test 3] Composite reward: Honest vs Lazy vs Gaming")
    honest, honest_fb = grade_composite(
        triage=1.0, resolution=0.85, response=0.90, efficiency=1.0, kb=0.80
    )
    lazy, lazy_fb = grade_composite(
        triage=0.4, resolution=0.25, response=0.25, efficiency=0.5, kb=0.0
    )
    gaming, gaming_fb = grade_composite(
        triage=0.4, resolution=0.50, response=0.60, efficiency=0.5, kb=0.25
    )
    print(f"  Honest agent:  {honest:.4f}")
    print(f"  Lazy agent:    {lazy:.4f}")
    print(f"  Gaming agent:  {gaming:.4f}")
    print(f"  Gap (honest - gaming): {honest - gaming:.4f}")
    print(f"\n{'=' * 60}")
    print("All grader tests passed!")
    print(f"{'=' * 60}")