"""
This file exists primarily to satisfy the OpenEnv CLI validation checks
(which strictly looks for a client.py alongside openenv.yaml).

For most usages within this project, you can import and use the underlying
primitives directly from `helpdeskenv_class.py` and `models.py`.
"""

from helpdeskenv_class import HelpdeskEnv
from models import (
    Ticket, HelpdeskAction, HelpdeskEnvState, StepResult,
    AgentRole, TicketCategory, TicketPriority, SupportTier
)

__all__ = [
    "HelpdeskEnv",
    "Ticket",
    "HelpdeskAction",
    "HelpdeskEnvState",
    "StepResult",
    "AgentRole",
    "TicketCategory",
    "TicketPriority",
    "SupportTier"
]
