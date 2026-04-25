"""Consolidated task scenarios for all three email tasks."""

import random
from models import EmailTask, TaskType


# ============================================================================
# Task 1: Spam Classification Scenarios
# ============================================================================

SPAM_SCENARIOS = [
    EmailTask(
        task_id = "spam_001",
        task_type = TaskType.SPAM,
        subject = "Congratulations! You've won $10,000",
        sender = "noreply@prize-winner.net",
        body = (
            "Dear Lucky winner,\n\n"
            "You have been selected to recieve $10,000 cash prize!\n"
            "Click the link below to claim your reward immediatedly.\n"
            "Offer expires in 24 hours. Act now!\n\n"
            "http://claim-prize-now.xyz/winner"
        ),
        context = "Classify this email as spam or not_spam.",
        ground_truth = "spam"
    ),

    EmailTask(
        task_id = "spam_002",
        task_type = TaskType.SPAM,
        subject = "Team standup moved to 3pm today",
        sender = "manager@company.com",
        body = (
            "Hi team,\n\n"
            "Just a heads-up - today's standup is moved from 2pm to 3pm "
            "due to a conflict with the client call.\n\n"
            "Please update your calendars. See you then!\n\n"
            "Best\nSarah"
        ),
        context = "Classify this email as spam or not_spam.",
        ground_truth = "not_spam"
    ),

    EmailTask(
        task_id = "spam_003",
        task_type = TaskType.SPAM,
        subject = "Your account has been compromised - verify now",
        sender = "security@paypa1-alert.com",
        body = (
            "URGENT: Your account has been accessed from an unknown device.\n"
            "Verify your identity immediately to avoid suspension.\n"
            "Click here: http://paypa1-verify.xyz/login\n\n"
            "Failure to verify within 12 hours will result in account closure."
        ),
        context = "Classify this email as spam or not_spam",
        ground_truth = "spam"
    ),

    EmailTask(
        task_id = "spam_004",
        task_type = TaskType.SPAM,
        subject = "Your invoice #4521 is ready",
        sender = "billing@adobecc.com",
        body = (
            "Hi Harish,\n\n"
            "Your invoice for Adobe Creative Cloud (Monthly Plan) is ready.\n"
            "Amount due: $54.99\n"
            "Due date: May 15, 2025\n\n"
            "You can view and download your invoice from your account portal.\n\n"
            "Thank you for your subscription.\n"
            "Adobe Billing Team"
        ),
        context = "Classify this email as spam or not_spam",
        ground_truth = "not_spam"
    ),

    EmailTask(
        task_id = "spam_005",
        task_type = TaskType.SPAM,
        subject = "Make $5000/week working from home - no experience needed",
        sender = "jobs@easy-money-online.biz",
        body = (
            "Are you tired of your 9-5 job?\n\n"
            "Join thousands of people earning $5000 or more per week "
            "from the comfort of their own home!\n"
            "No experience needed. No investment required.\n\n"
            "LIMITED SPOTS AVAILABLE - Sign up today!\n"
            "http://easy-money-jobs.biz/signup"
        ),
        context = "Classify this email as spam or not_spam.",
        ground_truth = "spam"
    ),
    EmailTask(
        task_id="spam_006",
        task_type=TaskType.SPAM,
        subject="Your pull request has been reviewed",
        sender="notifications@github.com",
        body=(
            "Hi harish-dev,\n\n"
            "Alex Chen reviewed your pull request:\n"
            "'Fix: Handle null pointer exception in auth module'\n\n"
            "Comments: 'Looks good overall! Left a small suggestion "
            "on line 47 regarding error handling.'\n\n"
            "View the full review on GitHub."
        ),
        context="Classify this email as spam or not_spam.",
        ground_truth="not_spam"
    ),
]

def get_spam_scenario(index: int) -> EmailTask:
    """Get a specific spam scenario by index."""
    return SPAM_SCENARIOS[index % len(SPAM_SCENARIOS)]

def get_random_spam_scenario(seed: int = None) -> EmailTask:
    """Get a random reply scenario, optionally seeded."""
    if seed is not None:
        random.seed(seed)
    return random.choice(SPAM_SCENARIOS)

def get_all_spam_scenarios() -> list[EmailTask]:
    """Return all reply scenarios."""
    return SPAM_SCENARIOS


# ============================================================================
# Task 2: Email Prioritization Scenarios
# ============================================================================

PRIORITY_SCENARIOS = [
    EmailTask(
        task_id = "priority_001",
        task_type = TaskType.PRIORITY,
        subject = "Production server is down - immediate action required",
        sender = "alerts@monitoring.company.com",
        body = (
            "CRITICAL ALERT:\n\n"
            "The production server (prod-us-east-1) has been unreachable "
            "for the past 10 minutes.\n"
            "All customer-facing services are currently offline.\n"
            "Estimated impact: 5,000+ active users.\n\n"
            "Please investigate immediately."
        ),
        context = (
            "You are a software engineer at a tech company. "
            "Prioritize this email as high, medium or low."
        ),
        ground_truth = "high"
    ),

    EmailTask(
        task_id = "priority_002",
        task_type = TaskType.PRIORITY,
        subject = "Monthly team newsletter - April edition",
        sender = "hr@company.com",
        body = (
            "Hi everyone,\n\n"
            "- Welcome to our 3 new team members\n"
            "- Office closed on April 18 for public holiday\n"
            "- Ping pong tournament results\n\n"
            "Have a great month!\nHR Team"
        ),
        context = (
            "You are a software engineer at a tech company. "
            "Prioritize this email as high, medium, or low."
        ),
        ground_truth = "low"
    ),
    EmailTask(
        task_id="priority_003",
        task_type=TaskType.PRIORITY,
        subject="Code review needed for auth module by EOD",
        sender="teammate@company.com",
        body=(
            "Hey,\n\n"
            "Could you review my PR for the auth module before end of day? "
            "It's blocking the QA team from starting their testing cycle "
            "for tomorrow's sprint review.\n\n"
            "PR link: github.com/company/repo/pull/312\n\n"
            "Thanks!"
        ),
        context=(
            "You are a software engineer at a tech company. "
            "Prioritize this email as high, medium, or low."
        ),
        ground_truth="medium"
    ),
    EmailTask(
        task_id="priority_004",
        task_type=TaskType.PRIORITY,
        subject="Security vulnerability found in payment module",
        sender="security@company.com",
        body=(
            "Hi team,\n\n"
            "Our security audit has identified a critical SQL injection "
            "vulnerability in the payment processing module.\n"
            "This could expose customer financial data.\n\n"
            "This must be patched before the next deployment, "
            "which is scheduled for tomorrow morning.\n\n"
            "Please treat this as top priority."
        ),
        context=(
            "You are a software engineer at a tech company. "
            "Prioritize this email as high, medium, or low."
        ),
        ground_truth="high"
    ),
    EmailTask(
        task_id="priority_005",
        task_type=TaskType.PRIORITY,
        subject="Lunch menu options for next week",
        sender="cafeteria@company.com",
        body=(
            "Hi all,\n\n"
            "Please find next week's lunch menu options below "
            "and vote for your preferred choices by Friday.\n\n"
            "Option A: Pasta bar\n"
            "Option B: Indian cuisine\n"
            "Option C: Salad station\n\n"
            "Vote here: forms.company.com/lunch-vote"
        ),
        context=(
            "You are a software engineer at a tech company. "
            "Prioritize this email as high, medium, or low."
        ),
        ground_truth="low"
    ),
    EmailTask(
        task_id="priority_006",
        task_type=TaskType.PRIORITY,
        subject="Client requesting demo reschedule to tomorrow",
        sender="sales@company.com",
        body=(
            "Hi,\n\n"
            "Our key client, Acme Corp, has requested to move their "
            "product demo from Friday to tomorrow at 2pm.\n\n"
            "This is a $200K deal. The engineering team needs to ensure "
            "the demo environment is ready by 1pm tomorrow.\n\n"
            "Please confirm availability ASAP."
        ),
        context=(
            "You are a software engineer at a tech company. "
            "Prioritize this email as high, medium, or low."
        ),
        ground_truth="high"
    ),
]

def get_priority_scenario(index: int) -> EmailTask:
    """Get a specific priority scenario by index."""
    return PRIORITY_SCENARIOS[index % len(PRIORITY_SCENARIOS)]

def get_random_priority_scenario(seed: int = None) -> EmailTask:
    """Get a random priority scenario, optionally seeded."""
    if seed is not None:
        random.seed(seed)
    return random.choice(PRIORITY_SCENARIOS)

def get_all_priority_scenarios() -> list[EmailTask]:
    """Return all priority scenarios."""
    return PRIORITY_SCENARIOS


# ============================================================================
# Task 3: Reply Generation Scenarios
# ============================================================================

REPLY_SCENARIOS = [
    EmailTask(
        task_id="reply_001",
        task_type=TaskType.REPLY,
        subject="Complaint: Order not delivered after 2 weeks",
        sender="angry.customer@gmail.com",
        body=(
            "This is absolutely unacceptable. I placed my order two weeks ago "
            "and it still hasn't arrived. I've tried calling your support line "
            "three times and no one picks up. I want a full refund immediately "
            "or I'm disputing this with my bank.\n\n"
            "Order #: 78432\nCustomer: John Davies"
        ),
        context=(
            "You are a customer support representative for an e-commerce company. "
            "The order was delayed due to a warehouse issue that has now been resolved. "
            "Draft a polite, empathetic, and professional reply. "
            "Apologize sincerely, explain briefly, and offer a resolution."
        ),
        ground_truth=(
            "A good reply: apologizes sincerely, acknowledges the frustration, "
            "explains the delay briefly without making excuses, offers a concrete "
            "resolution (refund or expedited shipping), and thanks the customer "
            "for their patience."
        )
    ),
    EmailTask(
        task_id="reply_002",
        task_type=TaskType.REPLY,
        subject="Request for project deadline extension",
        sender="junior.dev@company.com",
        body=(
            "Hi,\n\n"
            "I'm writing to request a 3-day extension on the dashboard feature. "
            "I underestimated the complexity of the charting library integration "
            "and want to make sure the quality is right rather than rush it.\n\n"
            "Would it be possible to move the deadline from Friday to Monday?\n\n"
            "Thanks,\nRaj"
        ),
        context=(
            "You are a team lead. You can grant the extension but want to "
            "understand the blockers better and set expectations. "
            "Draft a professional, supportive reply that grants the extension "
            "and asks for a brief update on progress."
        ),
        ground_truth=(
            "A good reply: grants the extension, acknowledges the effort to "
            "communicate proactively, asks briefly about current blockers, "
            "and sets a clear new deadline."
        )
    ),
    EmailTask(
        task_id="reply_003",
        task_type=TaskType.REPLY,
        subject="Job application follow-up",
        sender="candidate@gmail.com",
        body=(
            "Dear Hiring Team,\n\n"
            "I interviewed for the Backend Engineer position two weeks ago "
            "and wanted to follow up on the status of my application. "
            "I remain very excited about this opportunity and would appreciate "
            "any update you could share.\n\n"
            "Thank you for your time.\n"
            "Best regards,\nPriya Sharma"
        ),
        context=(
            "You are an HR coordinator. The candidate is still under review — "
            "the decision will be made by end of next week. "
            "Draft a polite reply acknowledging their follow-up and giving "
            "a realistic timeline without making promises."
        ),
        ground_truth=(
            "A good reply: thanks the candidate for following up, confirms "
            "their application is still under active review, gives a realistic "
            "timeline (end of next week), and encourages patience professionally."
        )
    ),
]


def get_reply_scenario(index: int) -> EmailTask:
    """Get a specific reply scenario by index."""
    return REPLY_SCENARIOS[index % len(REPLY_SCENARIOS)]


def get_random_reply_scenario(seed: int = None) -> EmailTask:
    """Get a random reply scenario, optionally seeded."""
    if seed is not None:
        random.seed(seed)
    return random.choice(REPLY_SCENARIOS)


def get_all_reply_scenarios() -> list[EmailTask]:
    """Return all reply scenarios."""
    return REPLY_SCENARIOS


# ============================================================================
# Utility: Fetch tasks by type string
# ============================================================================

def get_tasks_by_type(task_type: str) -> list[EmailTask]:
    """Return all scenarios for a given task type key.

    Accepted values: "spam", "priority", "reply".
    """
    key = task_type.lower()
    if key == "spam":
        return SPAM_SCENARIOS
    if key == "priority":
        return PRIORITY_SCENARIOS
    if key == "reply":
        return REPLY_SCENARIOS
    raise ValueError(f"Unknown task_type: {task_type}")
