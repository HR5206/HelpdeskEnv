# agents/__init__.py
"""Agent prompt templates for the HelpdeskEnv multi-agent system.
Each agent module contains:
- A system prompt defining the agent's role and capabilities
- A build_*_prompt() function that formats the user prompt with ticket details
- Documentation of the expected output format
Agents:
- triage: Classifies tickets by category, priority, and tier
- l1_agent: Handles simple issues using the Knowledge Base
- l2_agent: Handles moderately complex issues
- l3_agent: Handles critical/novel issues, writes KB articles
"""
from agents.triage import TRIAGE_SYSTEM_PROMPT, build_triage_prompt
__all__ = [
    "TRIAGE_SYSTEM_PROMPT",
    "build_triage_prompt",
]