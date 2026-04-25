# agents/triage.py
"""Triage Agent prompt templates.
The Triage Agent is the first agent to see every ticket. Its job:
1. Read the ticket subject, body, and context
2. Classify the ticket category (what type of issue)
3. Assess the priority level (how urgent)
4. Assign the support tier (which team handles it)
Output: JSON string with category, priority, and tier fields.
"""
from models import Ticket
# ============================================================================
# System Prompt — defines the agent's identity and rules
# ============================================================================
TRIAGE_SYSTEM_PROMPT = """You are a Triage Agent in an IT Helpdesk system.
Your role is to classify incoming support tickets by analyzing the subject, body, and context.
You must determine THREE things for each ticket:
1. CATEGORY — What type of issue is this?
   Valid categories:
   - "password_reset" — Account lockouts, expired passwords, authentication issues
   - "software_install" — Software installation, updates, licensing, configuration
   - "network_issue" — Connectivity problems, outages, VPN, DNS, firewall issues
   - "hardware_failure" — Physical device failures, printer issues, monitor problems
   - "data_recovery" — Lost files, database issues, backup restoration
   - "other" — Issues that don't fit the above categories
2. PRIORITY — How urgent is this ticket?
   Valid priorities:
   - "low" — Informational, no immediate impact (newsletters, lunch menus)
   - "medium" — Affects one user, workaround exists (password reset, software request)
   - "high" — Affects multiple users or has deadline pressure (security issue, demo prep)
   - "critical" — Major outage, data loss, or company-wide impact (server down, data breach)
3. TIER — Which support team should handle this?
   Valid tiers:
   - "L1" — Simple, well-documented issues (password resets, basic troubleshooting)
   - "L2" — Moderately complex issues requiring technical skills (software config, permissions)
   - "L3" — Critical/novel issues requiring expert knowledge (outages, data recovery, new bugs)
RULES:
- You MUST respond with ONLY a valid JSON object
- Do NOT include any explanation or text outside the JSON
- Use lowercase for category and priority values
- Use uppercase for tier values (L1, L2, L3)
OUTPUT FORMAT:
{"category": "<category>", "priority": "<priority>", "tier": "<tier>"}
"""
# ============================================================================
# User Prompt Builder — formats ticket details for the LLM
# ============================================================================
def build_triage_prompt(ticket: Ticket) -> str:
    """Build the user prompt for the Triage Agent.
    Takes a Ticket object and formats it into a clear, structured
    prompt that the LLM can classify. The prompt includes all
    observable fields (subject, sender, body, context) but NOT
    the ground truth fields.
    Args:
        ticket: The Ticket to classify.
    Returns:
        A formatted string to send as the user message to the LLM.
    """
    prompt = f"""Classify the following IT support ticket.
TICKET ID: {ticket.ticket_id}
SUBJECT: {ticket.subject}
FROM: {ticket.sender}
BODY:
{ticket.body}
"""
    if ticket.context:
        prompt += f"""
ADDITIONAL CONTEXT:
{ticket.context}
"""
    prompt += """
Respond with a JSON object containing:
- "category": the ticket category
- "priority": the urgency level
- "tier": the support tier (L1, L2, or L3)
JSON Response:"""
    return prompt
# ============================================================================
# Quick Validation (run with: python -m agents.triage)
# ============================================================================
if __name__ == "__main__":
    from tasks import get_all_ticket_scenarios
    print("=" * 60)
    print("Triage Agent — Prompt Template Validation")
    print("=" * 60)
    tickets = get_all_ticket_scenarios()
    for ticket in tickets[:2]:
        print(f"\n{'─' * 60}")
        print(f"Ticket: {ticket.ticket_id}")
        print(f"{'─' * 60}")
        print("\n[SYSTEM PROMPT]")
        print(TRIAGE_SYSTEM_PROMPT[:200] + "...")
        print("\n[USER PROMPT]")
        print(build_triage_prompt(ticket))
        print(f"\n[EXPECTED ANSWER]")
        print(
            f'{{"category": "{ticket.category.value}", '
            f'"priority": "{ticket.ground_truth_priority.value}", '
            f'"tier": "{ticket.ground_truth_tier.value}"}}'
        )
    print(f"\n{'=' * 60}")
    print("Triage prompt templates validated!")
    print(f"{'=' * 60}")