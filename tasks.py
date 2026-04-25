# tasks.py
"""Consolidated task scenarios for all three email tasks and helpdesk tickets."""
import random
from typing import Optional
from models import EmailTask, TaskType, Ticket, TicketCategory, TicketPriority, SupportTier
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
# ============================================================================
# HelpdeskEnv: IT Support Ticket Scenarios
# ============================================================================
TICKET_SCENARIOS: list[Ticket] = [
    # ------------------------------------------------------------------
    # ticket_001: Password Reset — L1, medium priority, sla_steps=3
    # ------------------------------------------------------------------
    # WHY this is easy:
    # - Password resets are the most common IT ticket type globally
    # - L1 agents should find a matching KB article (seeded in Part 6)
    # - 3-step SLA is tight but achievable: search_kb → apply_solution → respond
    # - No KB article needed (the solution is already well-documented)
    Ticket(
        ticket_id="ticket_001",
        category=TicketCategory.PASSWORD_RESET,
        subject="Cannot log in to my account — password expired",
        sender="jsmith@company.com",
        body=(
            "Hi IT Support,\n\n"
            "I'm unable to log in to my workstation this morning. The system "
            "says my password has expired and I need to reset it, but I can't "
            "access the self-service portal because it requires my old password "
            "to authenticate.\n\n"
            "I have a client presentation at 10 AM and urgently need access. "
            "My employee ID is EMP-4521 and my username is jsmith.\n\n"
            "Please help ASAP.\n\n"
            "Thanks,\nJohn Smith\nSales Department"
        ),
        context=(
            "Employee is locked out due to password expiration. "
            "The self-service reset portal requires the current (expired) password. "
            "Standard procedure: verify identity via employee ID, perform "
            "admin reset in Active Directory, and provide a temporary password."
        ),
        ground_truth_priority=TicketPriority.MEDIUM,
        ground_truth_tier=SupportTier.L1,
        ground_truth_resolution=(
            "Verified employee identity using employee ID EMP-4521. "
            "Performed administrative password reset in Active Directory. "
            "Set temporary password and instructed user to change it on first login. "
            "Confirmed user can access workstation successfully."
        ),
        sla_steps=3,
        requires_kb_article=False,
    ),
    # ------------------------------------------------------------------
    # ticket_002: Software Install — L2, medium priority, sla_steps=4
    # ------------------------------------------------------------------
    # WHY this is a step up from ticket_001:
    # - Software installs require checking compatibility and dependencies
    # - L2 agents need deeper technical skills than L1
    # - 4-step SLA gives room for: search_kb → diagnose → apply_fix → respond
    # - The Triage Agent must correctly route to L2 (not L1)
    # - Still no KB article needed (software installs are routine)
    Ticket(
        ticket_id="ticket_002",
        category=TicketCategory.SOFTWARE_INSTALL,
        subject="Need Visual Studio Code installed with Python extensions",
        sender="anewton@company.com",
        body=(
            "Hello,\n\n"
            "I recently joined the Data Science team and need Visual Studio Code "
            "installed on my workstation (Windows 11, Asset Tag WS-8834). "
            "I also need the following extensions:\n"
            "- Python (Microsoft)\n"
            "- Jupyter\n"
            "- Pylance\n\n"
            "My manager (Dr. Sarah Chen) has already approved this software "
            "request via ServiceNow ticket REQ-20240315.\n\n"
            "Additionally, I need Python 3.11 installed and added to the "
            "system PATH. I tried installing it myself but got a permissions "
            "error — I don't have admin rights.\n\n"
            "Thanks,\nAlex Newton\nData Science Team"
        ),
        context=(
            "New employee needs development tools installed. Manager approval "
            "is confirmed via ServiceNow. The workstation runs Windows 11 "
            "with standard user permissions. IT has admin access to install "
            "software via SCCM or remote desktop."
        ),
        ground_truth_priority=TicketPriority.MEDIUM,
        ground_truth_tier=SupportTier.L2,
        ground_truth_resolution=(
            "Verified manager approval via ServiceNow REQ-20240315. "
            "Connected to workstation WS-8834 via remote desktop. "
            "Installed Visual Studio Code v1.87 from the approved software catalog. "
            "Installed Python 3.11.8 and added to system PATH. "
            "Installed requested VS Code extensions: Python, Jupyter, Pylance. "
            "Verified Python interpreter is detected by VS Code. "
            "Confirmed with user that all tools are working correctly."
        ),
        sla_steps=4,
        requires_kb_article=False,
    ),
    # ------------------------------------------------------------------
    # ticket_003: Network Outage — L3, critical, sla_steps=5
    # ------------------------------------------------------------------
    # WHY this is hard:
    # - Affects entire departments — high blast radius
    # - Requires root cause analysis (switch failure, not just "restart it")
    # - Critical priority = tightest SLA pressure
    # - Must write a KB article so future outages resolve faster
    Ticket(
        ticket_id="ticket_003",
        category=TicketCategory.NETWORK_ISSUE,
        subject="URGENT: Entire 3rd floor has no network connectivity",
        sender="facilities@company.com",
        body=(
            "CRITICAL ISSUE:\n\n"
            "The entire 3rd floor (Engineering and Product teams, ~80 employees) "
            "has lost all network connectivity as of 9:15 AM. Both wired and "
            "wireless connections are down.\n\n"
            "Symptoms:\n"
            "- No internet access on any device\n"
            "- Internal services (Slack, Jira, Git) unreachable\n"
            "- VoIP phones are dead\n"
            "- The network switch in Server Room 3B is showing amber lights\n\n"
            "This is blocking all engineering work. Two teams have sprint "
            "demos scheduled for 2 PM today.\n\n"
            "Please treat as highest priority.\n"
            "- Mike Torres, Facilities Manager"
        ),
        context=(
            "Major network outage affecting an entire floor. The network "
            "switch in Server Room 3B appears to be the point of failure. "
            "This requires L3 networking expertise to diagnose — could be "
            "a switch hardware failure, firmware crash, or upstream trunk "
            "port issue. Standard L1/L2 cannot resolve this."
        ),
        ground_truth_priority=TicketPriority.CRITICAL,
        ground_truth_tier=SupportTier.L3,
        ground_truth_resolution=(
            "Diagnosed failed network switch (Cisco Catalyst 9300) in Server Room 3B. "
            "The switch experienced a firmware crash due to a known bug in IOS-XE 17.6.3. "
            "Performed emergency failover to the redundant switch. "
            "Restored connectivity for all 3rd floor devices within 45 minutes. "
            "Scheduled firmware upgrade to IOS-XE 17.9.4 during maintenance window. "
            "Created KB article documenting the failure mode and recovery procedure."
        ),
        sla_steps=5,
        requires_kb_article=True,
    ),
    # ------------------------------------------------------------------
    # ticket_004: Data Recovery — L3, critical, sla_steps=4
    # ------------------------------------------------------------------
    # WHY this is hard:
    # - Data loss is high-stakes — wrong action makes it worse
    # - Requires specialized backup/recovery knowledge
    # - Tight SLA (4 steps) despite complexity — tests efficiency
    # - Must document recovery procedure for future incidents
    Ticket(
        ticket_id="ticket_004",
        category=TicketCategory.DATA_RECOVERY,
        subject="CRITICAL: Accidentally deleted production database table",
        sender="dbadmin@company.com",
        body=(
            "EMERGENCY:\n\n"
            "During a routine maintenance script, I accidentally ran a DROP TABLE "
            "command on the 'customer_orders' table in the production database "
            "(PostgreSQL, server: db-prod-01).\n\n"
            "Details:\n"
            "- Table: customer_orders (~2.3M rows)\n"
            "- Time of deletion: 8:47 AM today\n"
            "- Last known backup: 2:00 AM daily snapshot\n"
            "- The application is currently returning 500 errors for all "
            "order-related pages\n\n"
            "I have NOT run any VACUUM or other commands since the deletion. "
            "The WAL logs should still be intact.\n\n"
            "Please help recover this data IMMEDIATELY.\n"
            "- Raj Patel, Database Administrator"
        ),
        context=(
            "Production database table was accidentally dropped. The PostgreSQL "
            "WAL (Write-Ahead Log) files are still available and the last "
            "daily backup was ~7 hours ago. Point-in-time recovery (PITR) "
            "using WAL replay is the best approach. This requires L3 DBA "
            "expertise — incorrect recovery attempts could overwrite WAL files."
        ),
        ground_truth_priority=TicketPriority.CRITICAL,
        ground_truth_tier=SupportTier.L3,
        ground_truth_resolution=(
            "Immediately preserved WAL log files to prevent overwrite. "
            "Restored 'customer_orders' table from 2:00 AM daily backup using pg_restore. "
            "Applied WAL replay (Point-in-Time Recovery) to roll forward to 8:46 AM, "
            "recovering all transactions up to one minute before deletion. "
            "Verified row count matches expected ~2.3M records. "
            "Restarted application services — 500 errors resolved. "
            "Total data loss: ~1 minute of transactions (8:46-8:47 AM). "
            "Created KB article documenting PITR recovery procedure for PostgreSQL."
        ),
        sla_steps=4,
        requires_kb_article=True,
    ),
    # ------------------------------------------------------------------
    # ticket_005: Novel/Unseen Issue — L3, high, sla_steps=6
    # ------------------------------------------------------------------
    # WHY this is the hardest:
    # - "other" category = doesn't fit standard playbooks
    # - No existing KB article covers this (tests agent improvisation)
    # - Longer SLA (6 steps) because diagnosis is open-ended
    # - The KB article written here becomes valuable for future episodes
    # - Tests the Self-Improving Systems theme most directly
    Ticket(
        ticket_id="ticket_005",
        category=TicketCategory.OTHER,
        subject="Mysterious data corruption in nightly ETL pipeline",
        sender="dataeng@company.com",
        body=(
            "Hi Team,\n\n"
            "We've discovered that our nightly ETL pipeline has been silently "
            "corrupting data for the past 3 days. The issue was only caught "
            "when the analytics team noticed impossible values in their dashboards.\n\n"
            "Symptoms:\n"
            "- Revenue figures in the data warehouse are ~3x higher than actual\n"
            "- The ETL job logs show SUCCESS with no errors\n"
            "- The source database (MySQL) has correct values\n"
            "- The transformation step uses a Python script (transform_orders.py)\n"
            "- The issue started exactly 3 days ago after a routine deploy\n\n"
            "We've paused the pipeline to prevent further corruption, but the "
            "analytics team needs accurate data for the board meeting on Friday.\n\n"
            "This doesn't match any known issue we've seen before.\n\n"
            "- Lisa Wang, Data Engineering Lead"
        ),
        context=(
            "Novel issue: ETL pipeline silently corrupting data. The issue "
            "correlates with a deploy 3 days ago. Root cause is likely in "
            "the Python transformation script — possibly a currency conversion "
            "bug, duplicate row processing, or a JOIN that creates cartesian "
            "products. This is an 'other' category because it doesn't fit "
            "standard IT support playbooks. No existing KB article exists."
        ),
        ground_truth_priority=TicketPriority.HIGH,
        ground_truth_tier=SupportTier.L3,
        ground_truth_resolution=(
            "Investigated the deploy from 3 days ago — found a code change in "
            "transform_orders.py that modified the JOIN condition between orders "
            "and order_items tables. The new JOIN was missing the order_id filter, "
            "creating a partial cartesian product that inflated revenue figures. "
            "Fixed the JOIN condition and re-ran the ETL for the affected 3-day window. "
            "Verified data warehouse figures now match source database. "
            "Added data validation checks to the pipeline (row count assertions, "
            "revenue sanity bounds). Created KB article documenting the diagnosis "
            "approach for silent ETL data corruption."
        ),
        sla_steps=6,
        requires_kb_article=True,
    ),
]
def get_ticket_scenario(ticket_id: str) -> Optional[Ticket]:
    """Get a specific ticket scenario by its ticket_id.
    Args:
        ticket_id: The unique ticket identifier (e.g. 'ticket_001').
    Returns:
        The matching Ticket, or None if not found.
    """
    for ticket in TICKET_SCENARIOS:
        if ticket.ticket_id == ticket_id:
            return ticket
    return None
def get_random_ticket_scenario(seed: Optional[int] = None) -> Ticket:
    """Get a random ticket scenario, optionally seeded for reproducibility.
    Args:
        seed: Optional random seed for deterministic selection.
    Returns:
        A randomly selected Ticket from TICKET_SCENARIOS.
    """
    if seed is not None:
        random.seed(seed)
    return random.choice(TICKET_SCENARIOS)
def get_all_ticket_scenarios() -> list[Ticket]:
    """Return all ticket scenarios.
    Returns:
        The complete list of Ticket scenarios.
    """
    return TICKET_SCENARIOS