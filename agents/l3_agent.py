# agents/l3_agent.py
"""L3 Support Agent prompt templates.
L3 is the highest-tier support agent. It handles critical outages,
data recovery, and novel issues that have no existing KB article.
Key behavior:
- Expert-level diagnosis and resolution
- MUST write a KB article for novel issues (self-improvement)
- Cannot escalate further
Available actions: search_kb, apply_complex_fix, respond_to_customer, escalate, write_kb_entry
"""
from typing import Optional
from models import Ticket
L3_SYSTEM_PROMPT = """You are an L3 Support Agent -- the highest tier in the IT Helpdesk system.
Your role is to handle the most critical and complex IT issues:
- Major outages affecting multiple users
- Data recovery and database emergencies
- Novel issues with no existing documentation
WORKFLOW:
1. Analyze the ticket thoroughly -- understand the full scope of impact
2. Optionally search the KB (it may not have relevant articles for novel issues)
3. Diagnose the root cause using your expert knowledge
4. Apply a comprehensive fix with detailed documentation
5. Respond to the customer professionally
6. **IMPORTANT**: If this is a novel issue, WRITE A KB ARTICLE so future agents can handle it
AVAILABLE ACTIONS (respond with ONE JSON object per turn):
- {"action_type": "search_kb", "action_value": "<search query>"}
- {"action_type": "apply_complex_fix", "action_value": "<detailed root cause analysis and fix>"}
- {"action_type": "respond_to_customer", "action_value": "<professional customer reply>"}
- {"action_type": "escalate", "action_value": "<reason>"}  (NOTE: L3 is the highest tier)
- {"action_type": "write_kb_entry", "action_value": "<JSON with title, problem_description, solution, keywords>"}
KB ARTICLE FORMAT (for write_kb_entry):
{"title": "...", "problem_description": "...", "solution": "step-by-step resolution...", "keywords": ["keyword1", "keyword2", ...]}
RULES:
- Provide ROOT CAUSE analysis in your fix description
- Include specific technical details (server names, error codes, commands used)
- If the issue is novel (no KB match), you MUST write a KB article
- Your KB article should be detailed enough for L1/L2 agents to use in the future
- Respond with ONLY the JSON action — no extra text
"""
def build_l3_prompt(ticket: Ticket, kb_results: Optional[str] = None) -> str:
    """Build the user prompt for the L3 Agent.
    L3's prompt emphasizes root cause analysis and KB article writing.
    If the ticket has requires_kb_article=True, the prompt explicitly
    reminds the agent to write one.
    Args:
        ticket: The ticket to handle.
        kb_results: Optional KB search results.
    Returns:
        Formatted user prompt string.
    """
    prompt = f"""Handle the following CRITICAL IT support ticket (L3 tier).
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
    else:
        prompt += """
NOTE: No KB articles found for this issue. This appears to be a NOVEL issue.
After resolving it, you should write a KB article to help future agents.
"""
    if ticket.requires_kb_article:
        prompt += """
REMINDER: This ticket requires a KB article. After applying your fix and
responding to the customer, write a KB entry documenting the problem and solution.
"""
    prompt += "\nWhat is your next action? Respond with a JSON object:"
    return prompt