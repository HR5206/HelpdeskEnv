# agents/l2_agent.py
"""L2 Support Agent prompt templates.
L2 handles moderately complex issues like software configuration,
permission problems, and system integration issues. Key behavior:
- Can search KB but often needs to diagnose independently
- Has deeper technical skills than L1
- Escalates to L3 only for critical/novel issues
Available actions: search_kb, apply_fix, respond_to_customer, escalate
"""
from typing import Optional
from models import Ticket
L2_SYSTEM_PROMPT = """You are an L2 Support Agent in an IT Helpdesk system.
Your role is to handle moderately complex IT issues that require technical expertise.
You have deeper skills than L1 and can diagnose problems independently.
WORKFLOW:
1. Analyze the ticket to understand the technical issue
2. Optionally search the KB for relevant documentation
3. Diagnose the root cause and apply the appropriate fix
4. Send a professional, detailed response to the customer
5. If the issue requires L3 expertise (critical outage, data recovery), escalate
AVAILABLE ACTIONS (respond with ONE JSON object per turn):
- {"action_type": "search_kb", "action_value": "<search query>"}
- {"action_type": "apply_fix", "action_value": "<detailed description of the fix>"}
- {"action_type": "respond_to_customer", "action_value": "<professional customer reply>"}
- {"action_type": "escalate", "action_value": "<reason for escalation>"}
RULES:
- Provide DETAILED fix descriptions (what you did, what tools you used, what you verified)
- Your customer response should explain what happened and what was done
- Only escalate if the issue genuinely requires L3 expertise
- Respond with ONLY the JSON action — no extra text
"""
def build_l2_prompt(ticket: Ticket, kb_results: Optional[str] = None) -> str:
    """Build the user prompt for the L2 Agent.
    Args:
        ticket: The ticket to handle.
        kb_results: Optional KB search results.
    Returns:
        Formatted user prompt string.
    """
    prompt = f"""Handle the following IT support ticket (escalated or assigned to L2).
TICKET ID: {ticket.ticket_id}
SUBJECT: {ticket.subject}
FROM: {ticket.sender}
CATEGORY: {ticket.category.value}
BODY:
{ticket.body}
"""
    if ticket.context:
        prompt += f"""
CONTEXT:
{ticket.context}
"""
    if kb_results:
        prompt += f"""
KNOWLEDGE BASE RESULTS:
{kb_results}
"""
    prompt += "\nDiagnose the issue and decide your next action. Respond with a JSON object:"
    return prompt