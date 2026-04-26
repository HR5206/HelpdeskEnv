# tasks.py
"""Consolidated task scenarios for helpdesk tickets."""
import random
from typing import Optional
from models import EmailTask, TaskType, Ticket, TicketCategory, TicketPriority, SupportTier

# ============================================================================
# Utility: Fetch tasks by type string (Kept for compatibility)
# ============================================================================
def get_tasks_by_type(task_type: str) -> list[EmailTask]:
    """Return all scenarios for a given task type key.
    """
    raise ValueError(f"Legacy EmailEnv tasks are removed.")

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
        # ------------------------------------------------------------------
    # ticket_006: Hardware Failure — L2, medium, sla_steps=5
    # ------------------------------------------------------------------
    # WHY this is moderate difficulty:
    # - Printer/scanner hardware issues are common but need diagnostics
    # - L2 must troubleshoot connectivity and driver issues
    # - Affects department workflow (not individual)
    # - 5-step SLA: search_kb → diagnose → apply_fix → respond
    # - KB article useful for future printer issues
    Ticket(
        ticket_id="ticket_006",
        category=TicketCategory.HARDWARE_FAILURE,
        subject="Office printer not responding to print jobs from all workstations",
        sender="operations@company.com",
        body=(
            "Hello IT,\n\n"
            "The HP LaserJet Enterprise M555 printer in Conference Room B is not "
            "accepting print jobs from any workstation. When users try to print, "
            "the job sits in the queue indefinitely.\n\n"
            "Details:\n"
            "- Printer IP: 192.168.1.45\n"
            "- Print jobs are queued but never processed\n"
            "- The printer's web interface (192.168.1.45:80) shows 'Ready' status\n"
            "- Error code on display: 'Unexpected error. Service required.'\n"
            "- Last updated: 3 days ago (firmware update?)\n"
            "- Approx 15 people unable to print — blocking document prep work\n\n"
            "Can this be fixed urgently?\n\n"
            "Thanks,\nSarah Mitchell\nOperations Manager"
        ),
        context=(
            "Enterprise printer malfunction affecting a department. The printer "
            "shows 'Ready' on its web interface but displays an error. This could be "
            "a firmware issue, a jammed tray sensor, or a network communication problem. "
            "L2 should search for printer error code solutions, power-cycle the device, "
            "and check network connectivity. May require remote troubleshooting via "
            "printer's web interface or escalation to L3 if hardware failure confirmed."
        ),
        ground_truth_priority=TicketPriority.MEDIUM,
        ground_truth_tier=SupportTier.L2,
        ground_truth_resolution=(
            "Connected to printer's web interface and reviewed logs — error pointed to "
            "firmware crash during update 3 days ago. Rebooted printer via web interface. "
            "Cleared print queue. Reinstalled printer driver on test workstation. "
            "Sent test print job — successful. Cleared print queue on all workstations "
            "and verified printer accepts jobs from multiple users. "
            "All jobs printing normally. Created KB article documenting HP M555 firmware "
            "recovery procedure."
        ),
        sla_steps=5,
        requires_kb_article=True,
    ),
        # ------------------------------------------------------------------
    # ticket_007: License Compliance — L3, high, sla_steps=6
    # ------------------------------------------------------------------
    # WHY this is challenging:
    # - Involves legal/compliance risk (not just technical)
    # - Requires knowledge of license types and audit processes
    # - "Other" category — doesn't fit standard IT playbooks
    # - High priority due to audit implications
    # - Longer SLA because diagnosis involves multiple stakeholders
    # - KB article important for future license audits
    Ticket(
        ticket_id="ticket_007",
        category=TicketCategory.OTHER,
        subject="Compliance Alert: Unlicensed software detected on enterprise machines",
        sender="compliance@company.com",
        body=(
            "URGENT COMPLIANCE MATTER:\n\n"
            "Our quarterly software audit tool has flagged 47 machines running "
            "unlicensed copies of Adobe Creative Suite (Photoshop, After Effects, etc.). "
            "This creates legal exposure and violates our Microsoft Enterprise Agreement.\n\n"
            "Details:\n"
            "- Affected machines: Primarily in Design and Marketing departments\n"
            "- Software: Adobe Creative Suite versions 2023 and 2024\n"
            "- License status: Unlicensed (not in our volume license agreement)\n"
            "- Risk: Potential audit penalties + legal liability\n"
            "- Audit deadline: Friday (3 days)\n\n"
            "We need to either:\n"
            "1. Obtain proper licenses immediately, OR\n"
            "2. Uninstall the software before the audit\n\n"
            "The design teams are currently using this software for active projects. "
            "We need a solution that doesn't halt production.\n\n"
            "- Legal/Compliance Team"
        ),
        context=(
            "Software license compliance issue with legal/business implications. "
            "This is complex because it requires coordination between IT, procurement, "
            "and legal. The L3 agent must: (1) document current license status, "
            "(2) identify compliant alternatives or procurement options, "
            "(3) communicate timeline to stakeholders, (4) implement solution. "
            "This tests agent reasoning beyond pure technical skills."
        ),
        ground_truth_priority=TicketPriority.HIGH,
        ground_truth_tier=SupportTier.L3,
        ground_truth_resolution=(
            "Contacted procurement to request emergency Adobe Creative Cloud Enterprise "
            "license quotes for 47 users. Coordinated with Legal to document current "
            "installation timestamps and compliance status. Implemented staged approach: "
            "(1) Provisioned 20 Adobe Creative Cloud licenses via subscription (Friday), "
            "(2) Transitioned 20 Design team members to licensed version by EOB Thursday, "
            "(3) Identified free/open-source alternatives (GIMP, DaVinci Resolve) for "
            "remaining 27 machines pending budget approval. Created KB article documenting "
            "compliance audit response procedures and software license management workflow. "
            "Scheduled recurring quarterly license audits."
        ),
        sla_steps=6,
        requires_kb_article=True,
    ),
        # ------------------------------------------------------------------
    # ticket_008: VPN Access Issue — L2, medium, sla_steps=4
    # ------------------------------------------------------------------
    # WHY this is good difficulty:
    # - VPN connectivity is common but requires network knowledge
    # - L2 should understand VPN client setup, certificates, and DNS
    # - Affects remote worker productivity (timely resolution important)
    # - 4-step SLA: search_kb → diagnose → apply_fix → respond
    # - No KB article required (standard troubleshooting)
    Ticket(
        ticket_id="ticket_008",
        category=TicketCategory.NETWORK_ISSUE,
        subject="VPN client won't connect after laptop OS update to Windows 11 Pro",
        sender="jchen@company.com",
        body=(
            "Hi IT Support,\n\n"
            "My VPN client stopped working after I updated my laptop from Windows 10 "
            "to Windows 11 Pro yesterday. I'm working remotely today and can't access "
            "the company network.\n\n"
            "Details:\n"
            "- VPN Client: Cisco AnyConnect v4.10.07046\n"
            "- Error message: 'Certificate verification failed'\n"
            "- OS: Windows 11 Pro (Build 22621)\n"
            "- No VPN access for ~8 hours already\n"
            "- Other remote workers are reporting same issue after OS upgrades\n\n"
            "I've tried:\n"
            "- Restarting the VPN client\n"
            "- Rebooting the laptop\n"
            "- Uninstalling and reinstalling AnyConnect (same error)\n\n"
            "I have deadline deliverables that require company network access. "
            "Please help ASAP.\n\n"
            "Thanks,\nJason Chen\nEngineering"
        ),
        context=(
            "VPN client incompatibility with Windows 11 after OS upgrade. "
            "Certificate verification failure suggests certificate store issue "
            "or TLS/SSL compatibility problem with new Windows 11 build. "
            "Root cause: Cisco AnyConnect v4.10.x is not fully compatible with "
            "Windows 11 Pro (Build 22621). L2 should check KB for Windows 11 "
            "VPN compatibility, update AnyConnect to v4.12+ (supports Windows 11), "
            "and verify certificate chain. May require updating device certificates."
        ),
        ground_truth_priority=TicketPriority.MEDIUM,
        ground_truth_tier=SupportTier.L2,
        ground_truth_resolution=(
            "Identified AnyConnect v4.10 incompatibility with Windows 11 Build 22621. "
            "Advised user to update to Cisco AnyConnect v4.12.04 which includes Windows 11 "
            "support. Provided download link and installation instructions. "
            "User successfully installed update. Cleared VPN cache files "
            "(C:\\Users\\[user]\\AppData\\Local\\Cisco\\Cisco AnyConnect). "
            "Re-imported device certificate using certificate management tool. "
            "Tested VPN connection — successful. User regained network access. "
            "Escalated to procurement to push AnyConnect v4.12+ as standard build "
            "for Windows 11 rollout."
        ),
        sla_steps=4,
        requires_kb_article=False,
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