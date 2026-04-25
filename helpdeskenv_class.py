# helpdeskenv_class.py
"""HelpdeskEnv — Multi-Agent IT Helpdesk OpenEnv Environment.
This is the core environment class that orchestrates the multi-agent
helpdesk simulation. It manages:
- Ticket queue and agent routing
- Knowledge Base (persists across episodes for self-improvement)
- Step-by-step grading of agent actions
- Episode lifecycle (reset → step → done)
The environment follows the OpenEnv API pattern:
- reset() → start a new episode, return first observation
- step(action) → process an agent action, return result
- state() → return current environment state
"""
import random
from typing import Optional, Dict, Any, List
import json
from models import (
    Ticket, HelpdeskAction, HelpdeskEnvState, HelpdeskResetResponse,
    StepResult, ErrorResponse, AgentRole, SupportTier, VALID_ACTION_TYPES,
)
from knowledge_base import KnowledgeBase
from tasks import get_all_ticket_scenarios
from graders import grade_triage
class HelpdeskEnv:
    """Multi-Agent IT Helpdesk Environment.
    Architecture:
    - Each episode presents a queue of tickets to resolve
    - Each ticket flows through: Triage → L1/L2/L3 Agent(s) → Resolution
    - The Knowledge Base persists across episodes (self-improvement)
    - Rewards combine: resolution quality, response quality, efficiency,
      SLA compliance, and KB contribution
    Usage:
        env = HelpdeskEnv()
        # Episode 1
        obs = env.reset(seed=42, num_tickets=3)
        while not env.state().is_done:
            result = env.step(action)
        # Episode 2 — KB retains knowledge from episode 1!
        obs = env.reset(seed=43, num_tickets=3)
        ...
    """
    def __init__(self) -> None:
        """Initialize the HelpdeskEnv.
        Creates the Knowledge Base (persists across episodes) and
        sets up empty state. The environment is not ready to use
        until reset() is called.
        Key: The KB is created here, NOT in reset(). This means
        knowledge accumulated in episode N is available in episode N+1.
        This is the self-improvement mechanism.
        """
        # Persistent Knowledge Base — survives across episodes
        self._kb = KnowledgeBase()
        # Episode state — reset each episode
        self._tickets: List[Ticket] = []
        self._current_ticket_idx: int = 0
        self._current_agent: Optional[AgentRole] = None
        self._steps_on_ticket: int = 0
        self._escalation_count: int = 0
        self._total_reward: float = 0.0
        self._history: List[StepResult] = []
        self._is_done: bool = True  # True until reset() is called
        self._kb_entries_added: int = 0
        self._episode_count: int = 0
        # Per-ticket tracking for the combined reward calculation
        self._ticket_rewards: Dict[str, List[StepResult]] = {}
    def reset(
        self,
        seed: Optional[int] = None,
        num_tickets: Optional[int] = None,
    ) -> HelpdeskResetResponse:
        """Start a new episode.
        Selects tickets from the scenario pool, resets episode state,
        and returns the first ticket for the Triage Agent.
        IMPORTANT: The Knowledge Base is NOT reset. This is intentional —
        it's the self-improvement mechanism. Knowledge from previous
        episodes carries forward.
        Args:
            seed: Optional random seed for reproducible ticket selection.
            num_tickets: How many tickets to include (default: all scenarios).
                         If less than available, randomly samples from pool.
        Returns:
            HelpdeskResetResponse with the first ticket and episode metadata.
        """
        self._episode_count += 1
        # Select tickets for this episode
        all_scenarios = get_all_ticket_scenarios()
        if seed is not None:
            random.seed(seed)
        if num_tickets is None or num_tickets >= len(all_scenarios):
            # Use all tickets, optionally shuffled
            self._tickets = list(all_scenarios)
            if seed is not None:
                random.shuffle(self._tickets)
        else:
            # Random sample of tickets
            self._tickets = random.sample(all_scenarios, min(num_tickets, len(all_scenarios)))
        # Reset episode state (but NOT the KB!)
        self._current_ticket_idx = 0
        self._current_agent = AgentRole.TRIAGE  # Always start with triage
        self._steps_on_ticket = 0
        self._escalation_count = 0
        self._total_reward = 0.0
        self._history = []
        self._is_done = False
        self._kb_entries_added = 0
        self._ticket_rewards = {}
        # Return the first ticket for the Triage Agent
        first_ticket = self._tickets[0]
        return HelpdeskResetResponse(
            observation=first_ticket,
            total_tickets=len(self._tickets),
            available_actions=VALID_ACTION_TYPES,
            kb_size=self._kb.size(),
        )
    def state(self) -> HelpdeskEnvState:
        """Return the current environment state.
        This is called by the /state endpoint and provides a complete
        snapshot of the environment for external observers and agents.
        Returns:
            HelpdeskEnvState with all current metrics and ticket info.
        """
        current_ticket = None
        if not self._is_done and self._current_ticket_idx < len(self._tickets):
            current_ticket = self._tickets[self._current_ticket_idx]
        return HelpdeskEnvState(
            current_ticket=current_ticket,
            current_agent=self._current_agent,
            ticket_number=self._current_ticket_idx,
            total_tickets=len(self._tickets),
            total_reward=round(self._total_reward, 4),
            steps_on_current_ticket=self._steps_on_ticket,
            is_done=self._is_done,
            history=self._history,
            kb_entries_added=self._kb_entries_added,
            escalation_count=self._escalation_count,
        )
    def kb(self) -> KnowledgeBase:
        """Return the Knowledge Base instance.
        Used by the /kb endpoint and by agents during search_kb actions.
        Returns:
            The persistent KnowledgeBase instance.
        """
        return self._kb
    @property
    def episode_count(self) -> int:
        """Return the number of episodes completed (including current)."""
        return self._episode_count
    def _current_ticket(self) -> Optional[Ticket]:
        """Get the current ticket being processed.
        Returns:
            The current Ticket, or None if the episode is done.
        """
        if self._is_done or self._current_ticket_idx >= len(self._tickets):
            return None
        return self._tickets[self._current_ticket_idx]
    def _advance_to_next_ticket(self) -> bool:
        """Move to the next ticket in the queue.
        Resets per-ticket state (step counter, agent role back to triage).
        Returns:
            True if there is a next ticket, False if all tickets are done.
        """
        self._current_ticket_idx += 1
        if self._current_ticket_idx >= len(self._tickets):
            self._is_done = True
            self._current_agent = None
            return False
        # Reset per-ticket state for the new ticket
        self._current_agent = AgentRole.TRIAGE
        self._steps_on_ticket = 0
        self._escalation_count = 0
        return True

    def _validate_action(self, action: HelpdeskAction) -> Optional[StepResult]:
        """Validate an incoming action before processing.

        Checks:
        1. Episode is not already done
        2. ticket_id matches current ticket
        3. agent_role matches current agent
        4. action_type is valid for this agent role

        Args:
            action: The HelpdeskAction to validate.

        Returns:
            None if valid, or a StepResult with error feedback if invalid.
        """
        ticket = self._current_ticket()

        # Check episode is active
        if self._is_done or ticket is None:
            return StepResult(
                task_id=action.ticket_id,
                reward=0.0,
                done=True,
                feedback="Episode is already complete. Call reset() to start a new episode.",
            )

        # Check ticket_id matches
        if action.ticket_id != ticket.ticket_id:
            return StepResult(
                task_id=action.ticket_id,
                reward=0.0,
                done=False,
                feedback=(
                    f"Wrong ticket_id: you sent '{action.ticket_id}' "
                    f"but current ticket is '{ticket.ticket_id}'."
                ),
            )

        # Check agent_role matches
        if action.agent_role != self._current_agent:
            return StepResult(
                task_id=ticket.ticket_id,
                reward=0.0,
                done=False,
                feedback=(
                    f"Wrong agent_role: you sent '{action.agent_role.value}' "
                    f"but current agent is '{self._current_agent.value}'."
                ),
            )

        # Check action_type is valid for this role
        valid_types = VALID_ACTION_TYPES.get(action.agent_role.value, [])
        if action.action_type not in valid_types:
            return StepResult(
                task_id=ticket.ticket_id,
                reward=0.0,
                done=False,
                feedback=(
                    f"Invalid action_type '{action.action_type}' for agent "
                    f"'{action.agent_role.value}'. Valid types: {valid_types}"
                ),
            )

        return None  # All checks passed

    def step(self, action: HelpdeskAction) -> StepResult:
        """Process one agent action.

        This is the core game loop. The flow depends on which agent is active:

        TRIAGE phase (current_agent == TRIAGE):
        1. Validate the action
        2. Grade the triage classification (category, priority, tier)
        3. Route to the assigned support tier agent
        4. Return the grading result

        SUPPORT phase (current_agent == L1/L2/L3):
        → Handled in Part 12

        Args:
            action: The HelpdeskAction submitted by the current agent.

        Returns:
            StepResult with reward, feedback, and done status.
        """
        # Validate the action
        validation_error = self._validate_action(action)
        if validation_error is not None:
            return validation_error

        ticket = self._current_ticket()
        self._steps_on_ticket += 1

        # ── TRIAGE PHASE ──────────────────────────────────────────
        if self._current_agent == AgentRole.TRIAGE:
            return self._handle_triage(ticket, action)

        # ── SUPPORT PHASE (L1/L2/L3) ─────────────────────────────
        # Placeholder — will be implemented in Part 12
        return StepResult(
            task_id=ticket.ticket_id,
            reward=0.0,
            done=False,
            feedback=f"Support agent actions not yet implemented (agent: {self._current_agent.value}).",
        )

    def _handle_triage(self, ticket: Ticket, action: HelpdeskAction) -> StepResult:
        """Handle a triage action: classify and route the ticket.

        Grades the triage classification, then routes the ticket to
        the support tier that the triage agent specified. Uses the
        ground truth tier for routing (so even if the agent gets the
        tier wrong, the ticket still goes to the correct agent).

        Args:
            ticket: The current ticket being triaged.
            action: The triage action with JSON classification.

        Returns:
            StepResult with triage score and routing feedback.
        """
        # Grade the triage classification
        result = grade_triage(ticket, action)

        # Track the reward
        self._total_reward += result.reward
        self._history.append(result)

        # Parse the submitted tier to determine routing
        # Use ground truth tier for routing (ensures correct workflow)
        # even if the agent misclassified — the penalty is in the score
        try:
            triage_data = json.loads(action.action_value)
            submitted_tier = str(triage_data.get("tier", "")).strip().upper()
        except (json.JSONDecodeError, TypeError):
            submitted_tier = ""

        # Route to the correct support tier (ground truth)
        tier = ticket.ground_truth_tier
        if tier == SupportTier.L1:
            self._current_agent = AgentRole.L1_SUPPORT
        elif tier == SupportTier.L2:
            self._current_agent = AgentRole.L2_SUPPORT
        else:
            self._current_agent = AgentRole.L3_SUPPORT

        # Append routing info to the feedback
        result.feedback += (
            f"\n\n→ Ticket routed to {self._current_agent.value} "
            f"(ground truth tier: {tier.value})"
        )

        return result

# ============================================================================
# Quick Validation (run with: python helpdeskenv_class.py)
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("HelpdeskEnv — Constructor & Reset Validation")
    print("=" * 60)
    env = HelpdeskEnv()
    print(f"\n1. KB initialized with {env.kb().size()} seed entries")
    assert env.kb().size() == 2
    # Test reset with all tickets
    response = env.reset(seed=42)
    print(f"\n2. Reset with all tickets:")
    print(f"   Total tickets: {response.total_tickets}")
    print(f"   First ticket: {response.observation.ticket_id} — {response.observation.subject}")
    print(f"   KB size: {response.kb_size}")
    assert response.total_tickets >= 2
    assert response.kb_size == 2
    # Test state after reset
    state = env.state()
    print(f"\n3. State after reset:")
    print(f"   Current agent: {state.current_agent}")
    print(f"   Ticket number: {state.ticket_number}/{state.total_tickets}")
    print(f"   Is done: {state.is_done}")
    assert state.current_agent == AgentRole.TRIAGE
    assert state.is_done is False
    # Test reset with limited tickets
    response2 = env.reset(seed=99, num_tickets=2)
    print(f"\n4. Reset with num_tickets=2:")
    print(f"   Total tickets: {response2.total_tickets}")
    assert response2.total_tickets == 2
    # Test episode counter
    print(f"\n5. Episode count: {env.episode_count}")
    assert env.episode_count == 2
    # Test KB persistence across resets
    print(f"\n6. KB still has {env.kb().size()} entries after reset (not cleared!)")
    assert env.kb().size() == 2
    print("\n" + "=" * 60)
    print("All HelpdeskEnv constructor & reset tests passed!")
    print("=" * 60)