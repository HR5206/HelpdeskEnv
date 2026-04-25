# models.py
"""Consolidated models for the EmailEnv / HelpdeskEnv environment.
Contains all Pydantic models used across the application:
- EmailEnv OpenEnv models (Email, Observation, Action, State, Reward)
- Task and grading models (EmailTask, AgentAction, StepResult, EnvState, etc.)
- HelpdeskEnv models (TicketCategory, AgentRole, TicketPriority, SupportTier, Ticket)
- HelpdeskEnv action/state models (HelpdeskAction, HelpdeskEnvState, HelpdeskResetResponse)
"""
from datetime import datetime
from typing import Any, Dict, List, Optional, Literal
from enum import Enum
from pydantic import BaseModel, Field
# ============================================================================
# OpenEnv Core Models (with fallback classes)
# ============================================================================
try:
    from openenv_core import Observation as OpenEnvObservation        
    from openenv_core import Action as OpenEnvAction
    from openenv_core import State as OpenEnvState
    from openenv_core import Reward as OpenEnvReward
except Exception:  # type: ignore
    class OpenEnvObservation(BaseModel):
        pass
    class OpenEnvAction(BaseModel):
        pass
    class OpenEnvState(BaseModel):
        pass
    class OpenEnvReward(BaseModel):
        pass
# ============================================================================
# Email and OpenEnv Models
# ============================================================================
class Email(BaseModel):
    """Represents an email message."""
    id: str
    subject: str
    body: str
    sender: str
    timestamp: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)
class Observation(OpenEnvObservation):
    """Observation from the EmailEnv environment."""
    email: Optional[Email] = None
    task: Literal["spam_classification", "email_prioritization", "reply_generation"]
    step_index: int
    total_steps: int
    remaining_emails: int
class Action(OpenEnvAction):
    """Action submitted to the EmailEnv environment."""
    type: Literal["classify_spam", "set_priority", "generate_reply", "skip"]
    is_spam: Optional[bool] = None
    priority: Optional[Literal["low", "medium", "high"]] = None
    reply_text: Optional[str] = None
class State(OpenEnvState):
    """State snapshot of the EmailEnv environment."""
    current_email_index: int
    total_emails: int
    completed: bool
    task: Literal["spam_classification", "email_prioritization", "reply_generation"]
class Reward(OpenEnvReward):
    """Reward signal from the EmailEnv environment."""
    value: float
    task_scores: Dict[str, float] = Field(default_factory=dict)
    done: bool = False
# ============================================================================
# Task and Grading Models
# ============================================================================
class TaskType(str, Enum):
    """
    Defines the three task types in EmailEnv.
    Using str + Enum means the values are plain strings
    (e.g. "spam", "priority", "reply") - easy to use in JSON.
    """
    SPAM = "spam"
    PRIORITY = "priority"
    REPLY = "reply"
class EmailTask(BaseModel):
    """
    Represents one email scenario presented to the agent.
    This is the 'observation' - what the agent sees.
    """
    task_id: str = Field(..., description="Unique identifier for this task")
    task_type: TaskType = Field(..., description="Which of the 3 tasks is this")
    subject: str = Field(..., description="Email subject line")
    sender: str = Field(..., description="Email sender address")
    body: str = Field(..., description="Full email body text")
    context: Optional[str] = Field(
        None,
        description="Extra content for the agent, e.g. 'You are a customer support rep'"
    )
    ground_truth: Optional[str] = Field(
        None,
        description="The correct answer (hidden from agent, used by grader)"
    )
class AgentAction(BaseModel):
    """
    Represents the action submitted by the agent.
    For Task 1 (spam): action_value = "spam" or "not_spam"
    For Task 2 (priority): action_value = "high", "medium" or "low"
    For Task 3 (reply): action_value = the full drafted reply text
    """
    task_id: str = Field(..., description="Must match the current task's task_id")
    action_value: str = Field(..., description="The agent's answer or response")
class StepResult(BaseModel):
    """
    Returned by step() after the agent submits an action.
    Contains the reward, whether the episode is done,
    and optional feedback for the agent.
    """
    task_id: str = Field(..., description="The task that was just graded")
    reward: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Score between 0.0 (wrong) and 1.0 (perfect)"
    )
    done: bool = Field(..., description="True if the episode is complete")
    feedback: Optional[str] = Field(
        None,
        description="Human-readable explanation of the score"
    )
    correct_answer: Optional[str] = Field(
        None,
        description="Revealed after grading so agent can learn"
    )
class EnvState(BaseModel):
    """
    A snapshot of the environment at any point in time.
    Returned by state() endpoint.
    """
    current_task: Optional[EmailTask] = Field(
        None,
        description="The email task that is currently active"
    )
    task_number: int = Field(
        default=0,
        description="Which task index we are on (0, 1, or 2)"
    )
    total_reward: float = Field(
        default=0.0,
        description="Cumulative reward across all tasks so far"
    )
    is_done: bool = Field(
        default=False,
        description="True when all 3 tasks are complete"
    )
    history: list[StepResult] = Field(
        default_factory=list,
        description="List of all past StepResults in this episode"
    )
class ResetResponse(BaseModel):
    """
    Returned when the agent calls reset() to start a new episode.
    Gives the agent its first task.
    """
    observation: EmailTask = Field(
        ...,
        description="The first email task for the agent to solve"
    )
    available_tasks: list[str] = Field(
        default=["spam_classification", "email_prioritization", "reply_generation"],
        description="List of available task types"
    )
class ErrorResponse(BaseModel):
    """
    Returned when something goes wrong (wrong task_id, bad input, etc.)
    """
    error: str = Field(..., description="Description of what went wrong")
    detail: Optional[str] = Field(None, description="Extra data if available")
# ============================================================================
# HelpdeskEnv Enums
# ============================================================================
class TicketCategory(str, Enum):
    """
    Categories of IT helpdesk tickets.
    These map to real-world IT support domains. The Triage Agent must
    classify incoming tickets into one of these categories to route
    them to the correct support tier. Six categories give enough variety
    to make classification non-trivial without being overwhelming.
    """
    PASSWORD_RESET = "password_reset"
    SOFTWARE_INSTALL = "software_install"
    NETWORK_ISSUE = "network_issue"
    HARDWARE_FAILURE = "hardware_failure"
    DATA_RECOVERY = "data_recovery"
    OTHER = "other"
class AgentRole(str, Enum):
    """
    Roles in the multi-agent helpdesk system.
    The four roles form an escalation hierarchy:
    - TRIAGE: First contact — classifies and routes tickets
    - L1_SUPPORT: Handles simple, well-documented issues (password resets, etc.)
    - L2_SUPPORT: Handles moderately complex issues (software, config problems)
    - L3_SUPPORT: Handles critical/novel issues (outages, data recovery, new bugs)
    This hierarchy is central to the Multi-Agent Interactions theme.
    """
    TRIAGE = "triage"
    L1_SUPPORT = "l1_support"
    L2_SUPPORT = "l2_support"
    L3_SUPPORT = "l3_support"
class TicketPriority(str, Enum):
    """
    Four-level priority scale for helpdesk tickets.
    Extends the original 3-level EmailEnv scale (low/medium/high) with
    a 'critical' level. This is important because:
    - It increases the difficulty of the classification task
    - Critical tickets have tighter SLAs, adding time pressure
    - The distance-based grading in grade_priority() naturally extends
      to 4 levels (distance 0=perfect, 1=close, 2=far, 3=completely wrong)
    """
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
class SupportTier(str, Enum):
    """
    Support tiers defining the escalation chain.
    L1 → L2 → L3 is the standard IT support model:
    - L1: First-line support, uses Knowledge Base to resolve common issues
    - L2: Second-line, has deeper technical skills for software/config issues
    - L3: Third-line, handles the hardest problems and creates new KB articles
    The tier determines which agent handles the ticket and what actions
    are available. Unnecessary escalations incur a penalty (efficiency grader).
    """
    L1 = "L1"
    L2 = "L2"
    L3 = "L3"
# ============================================================================
# HelpdeskEnv Ticket Model
# ============================================================================
class Ticket(BaseModel):
    """
    Represents a single IT helpdesk ticket — the core data object that
    flows through the entire multi-agent system.
    Design notes:
    - ticket_id: Unique identifier for tracking across agent handoffs
    - category: The correct category (used by triage grader)
    - subject/sender/body/context: The observable information agents see
    - ground_truth_priority: Correct priority (grading the triage agent)
    - ground_truth_tier: Correct support tier (grading routing decisions)
    - ground_truth_resolution: The expected fix (grading resolution quality)
    - sla_steps: Maximum steps allowed before SLA breach — this drives
      the Long-Horizon Planning theme by creating time pressure
    - requires_kb_article: If True, the L3 agent should write a KB entry
      after resolving — this drives the Self-Improving Systems theme
    """
    ticket_id: str = Field(
        ...,
        description="Unique identifier for this ticket (e.g. 'ticket_001')"
    )
    category: TicketCategory = Field(
        ...,
        description="The correct category for this ticket"
    )
    subject: str = Field(
        ...,
        description="Short description / subject line of the ticket"
    )
    sender: str = Field(
        ...,
        description="The user who submitted the ticket (email or username)"
    )
    body: str = Field(
        ...,
        description="Full description of the issue from the user"
    )
    context: Optional[str] = Field(
        None,
        description="Additional context for the agent (system info, history, etc.)"
    )
    ground_truth_priority: TicketPriority = Field(
        ...,
        description="The correct priority level (hidden from agent, used by grader)"
    )
    ground_truth_tier: SupportTier = Field(
        ...,
        description="The correct support tier (hidden from agent, used by grader)"
    )
    ground_truth_resolution: str = Field(
        ...,
        description="The expected resolution description (hidden from agent, used by grader)"
    )
    sla_steps: int = Field(
        ...,
        ge=1,
        description="Maximum number of steps allowed to resolve within SLA"
    )
    requires_kb_article: bool = Field(
        default=False,
        description="Whether this ticket should result in a new KB article (self-improvement)"
    )
# ============================================================================
# HelpdeskEnv Action Model
# ============================================================================
class HelpdeskAction(BaseModel):
    """
    Action submitted by an agent in the HelpdeskEnv.
    This is the multi-agent equivalent of AgentAction. The key difference
    is that it carries agent_role (WHO is acting) and action_type (WHAT
    kind of action), enabling the environment to validate that the right
    agent is performing the right action at the right time.
    Action types and their expected action_value payloads:
    - "triage": JSON string {"category": "...", "priority": "...", "tier": "..."}
    - "search_kb": A search query string to look up the Knowledge Base
    - "apply_solution": Text describing the fix applied (L1 simple fixes)
    - "apply_fix": Text describing the fix applied (L2 intermediate fixes)
    - "apply_complex_fix": Text describing the fix applied (L3 complex fixes)
    - "respond_to_customer": Full customer-facing reply text
    - "escalate": Reason for escalation (text)
    - "write_kb_entry": JSON string with KB article content
    """
    ticket_id: str = Field(
        ...,
        description="The ticket being acted on — must match the current ticket"
    )
    agent_role: AgentRole = Field(
        ...,
        description="Which agent is performing this action"
    )
    action_type: str = Field(
        ...,
        description=(
            "Type of action: 'triage', 'search_kb', 'apply_solution', "
            "'apply_fix', 'apply_complex_fix', 'respond_to_customer', "
            "'escalate', 'write_kb_entry'"
        )
    )
    action_value: str = Field(
        ...,
        description="The action payload — format depends on action_type"
    )
# Valid action types per agent role — used by the environment to enforce
# that agents only perform actions they're authorized for.
# This is defined here (not in the env class) so it can be used in
# /schema and /metadata endpoints for documentation.
VALID_ACTION_TYPES: Dict[str, List[str]] = {
    "triage": ["triage"],
    "l1_support": ["search_kb", "apply_solution", "respond_to_customer", "escalate"],
    "l2_support": ["search_kb", "apply_fix", "respond_to_customer", "escalate"],
    "l3_support": [
        "search_kb", "apply_complex_fix", "respond_to_customer",
        "escalate", "write_kb_entry"
    ],
}
# ============================================================================
# HelpdeskEnv State Model
# ============================================================================
class HelpdeskEnvState(BaseModel):
    """
    Full observable state of the HelpdeskEnv at any point in time.
    This is what gets returned by the /state endpoint. It gives agents
    and external observers a complete picture of:
    - WHERE we are: current ticket, current agent, ticket progress
    - HOW we're doing: total reward, step count, SLA status
    - WHAT happened: history of all past step results
    - SELF-IMPROVEMENT metrics: KB entries added, escalation count
    The escalation_count and kb_entries_added fields are particularly
    important for the hackathon: they make the self-improvement and
    multi-agent collaboration directly visible in the state.
    """
    current_ticket: Optional[Ticket] = Field(
        None,
        description="The ticket currently being processed (None if episode is done)"
    )
    current_agent: Optional[AgentRole] = Field(
        None,
        description="Which agent is currently handling the ticket"
    )
    ticket_number: int = Field(
        default=0,
        description="Index of the current ticket (0-based) in this episode"
    )
    total_tickets: int = Field(
        default=0,
        description="Total number of tickets in this episode"
    )
    total_reward: float = Field(
        default=0.0,
        description="Cumulative reward across all tickets so far"
    )
    steps_on_current_ticket: int = Field(
        default=0,
        description="How many steps have been taken on the current ticket"
    )
    is_done: bool = Field(
        default=False,
        description="True when all tickets in the episode have been resolved"
    )
    history: List[StepResult] = Field(
        default_factory=list,
        description="Chronological list of all StepResults in this episode"
    )
    kb_entries_added: int = Field(
        default=0,
        description="Number of KB articles written during this episode (self-improvement metric)"
    )
    escalation_count: int = Field(
        default=0,
        description="Total escalations across all tickets (efficiency metric)"
    )
# ============================================================================
# HelpdeskEnv Reset Response
# ============================================================================
class HelpdeskResetResponse(BaseModel):
    """
    Returned by /reset to start a new helpdesk episode.
    Gives the agent:
    - The first ticket to process (observation)
    - Metadata about the episode (total tickets, available actions)
    - KB size — this is crucial for the self-improvement narrative.
      In episode 1, kb_size might be 2 (just seed articles). By episode 5,
      it could be 8+, showing that the system has learned from past tickets.
    """
    observation: Ticket = Field(
        ...,
        description="The first ticket for the Triage Agent to classify"
    )
    total_tickets: int = Field(
        ...,
        description="How many tickets the agent must resolve in this episode"
    )
    available_actions: Dict[str, List[str]] = Field(
        default_factory=lambda: VALID_ACTION_TYPES.copy(),
        description="Map of agent_role -> list of valid action_types"
    )
    kb_size: int = Field(
        default=0,
        description="Current number of entries in the Knowledge Base (shows growth over episodes)"
    )
# ============================================================================
# Export All Models
# ============================================================================
__all__ = [
    # OpenEnv models
    "Email",
    "Observation",
    "Action",
    "State",
    "Reward",
    # Task models
    "TaskType",
    "EmailTask",
    "AgentAction",
    "StepResult",
    "EnvState",
    "ResetResponse",
    "ErrorResponse",
    # HelpdeskEnv enums
    "TicketCategory",
    "AgentRole",
    "TicketPriority",
    "SupportTier",
    # HelpdeskEnv models
    "Ticket",
    "HelpdeskAction",
    "HelpdeskEnvState",
    "HelpdeskResetResponse",
    # HelpdeskEnv constants
    "VALID_ACTION_TYPES",
]