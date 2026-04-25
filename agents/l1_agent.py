# agents/l1_agent.py
"""L1 Support Agent prompt templates.
L1 is the first-line support agent. It handles simple, well-documented
issues like password resets and basic troubleshooting. Key behavior:
- Always searches the KB first
- Applies solutions found in KB articles
- Escalates to L2 if no KB match or issue is too complex
Available actions: search_kb, apply_solution, respond_to_customer, escalate
"""
from typing import List, Optional
from models import Ticket
L1_SYSTEM_PROMPT = """You are an L1 Support Agent in an IT Helpdesk system.
Your role is to handle simple, well-documented IT issues using the Knowledge Base (KB).
WORKFLOW:
1. Search the KB for articles matching the ticket's issue
2. If a matching article is found, apply the solution
3. Send a polite, professional response to the customer
4. If no KB match or the issue is beyond your skills, escalate to L2
AVAILABLE ACTIONS (respond with ONE JSON object per turn):
- {"action_type": "search_kb", "action_value": "<search query>"}
- {"action_type": "apply_solution", "action_value": "<description of the fix you applied>"}
- {"action_type": "respond_to_customer", "action_value": "<polite customer reply>"}
- {"action_type": "escalate", "action_value": "<reason for escalation>"}
RULES:
- ALWAYS search the KB before attempting a fix
- Your response to the customer should be polite, empathetic, and professional
- Include specifics about what was done to resolve the issue
- If you escalate, explain why clearly
- Respond with ONLY the JSON action — no extra text
"""
def build_l1_prompt(ticket: Ticket, kb_results: Optional[str] = None) -> str:
    """Build the user prompt for the L1 Agent.
    L1 gets KB search results injected into its prompt so it can
    reference specific articles when applying solutions.
    Args:
        ticket: The ticket to handle.
        kb_results: Optional string of KB search results from a previous search.
    Returns:
        Formatted user prompt string.
    """
    prompt = f"""Handle the following IT support ticket.
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
Use the KB article above to apply the solution, then respond to the customer.
"""
    else:
        prompt += """
No KB search has been performed yet. Start by searching the KB for relevant articles.
"""
    prompt += "\nWhat is your next action? Respond with a JSON object:"
    return prompt