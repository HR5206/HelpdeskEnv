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
    StepResult, ErrorResponse, AgentRole, SupportTier, TicketCategory,
    VALID_ACTION_TYPES, EmailTask, AgentAction,
)
from knowledge_base import KnowledgeBase, KBEntry
from tasks import get_all_ticket_scenarios
from graders import (
    grade_triage, grade_efficiency, grade_kb_contribution,
    _score_relevance, _score_politeness, _score_length,
)
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
        return self._handle_support(ticket, action)

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
    
    def _handle_support(self, ticket: Ticket, action: HelpdeskAction) -> StepResult:
        """Handle a support agent action (L1/L2/L3).
        Dispatches to the appropriate handler based on action_type:
        - search_kb: Query the Knowledge Base
        - apply_solution/apply_fix/apply_complex_fix: Attempt resolution
        - respond_to_customer: Send reply and resolve ticket
        - escalate: Move to next tier
        - write_kb_entry: Add article to Knowledge Base
        Args:
            ticket: The current ticket.
            action: The support action.
        Returns:
            StepResult with action-specific feedback and reward.
        """
        action_type = action.action_type
        # ── SEARCH KB ─────────────────────────────────────────────
        if action_type == "search_kb":
            query = action.action_value.strip()
            results = self._kb.search(query, top_k=3)
            if results:
                kb_text = "\n\n".join(
                    f"📄 [{r.entry_id}] {r.title}\n"
                    f"   Problem: {r.problem_description[:100]}...\n"
                    f"   Solution: {r.solution[:150]}..."
                    for r in results
                )
                feedback = f"KB Search Results for '{query}':\n{kb_text}"
            else:
                feedback = f"No KB articles found for query: '{query}'"
            result = StepResult(
                task_id=ticket.ticket_id,
                reward=0.0,  # search is free — no reward or penalty
                done=False,
                feedback=feedback,
            )
            self._history.append(result)
            return result
        # ── APPLY SOLUTION / FIX / COMPLEX FIX ───────────────────
        if action_type in ("apply_solution", "apply_fix", "apply_complex_fix"):
            submitted_fix = action.action_value.strip()
            ground_truth = ticket.ground_truth_resolution
            # Score using keyword relevance between submitted fix and ground truth
            pseudo_task = EmailTask(
                task_id=ticket.ticket_id,
                task_type="reply",
                subject=ticket.subject,
                sender=ticket.sender,
                body=ground_truth,  # Use ground truth as the "email body" for keyword matching
            )
            relevance_score, relevance_fb = _score_relevance(submitted_fix, pseudo_task)
            length_score, length_fb = _score_length(submitted_fix)
            resolution_reward = round(relevance_score * 0.6 + length_score * 0.4, 2)
            feedback = (
                f"Resolution Score Breakdown:\n"
                f"  Relevance to expected fix (60%): {relevance_score:.2f} — {relevance_fb}\n"
                f"  Detail/length (40%): {length_score:.2f} — {length_fb}\n"
                f"  Resolution Score: {resolution_reward:.2f}"
            )
            result = StepResult(
                task_id=ticket.ticket_id,
                reward=resolution_reward,
                done=False,
                feedback=feedback,
                correct_answer=ground_truth[:200] + "..." if len(ground_truth) > 200 else ground_truth,
            )
            self._total_reward += result.reward
            self._history.append(result)
            # Store as resolution reward for this ticket
            if ticket.ticket_id not in self._ticket_rewards:
                self._ticket_rewards[ticket.ticket_id] = []
            self._ticket_rewards[ticket.ticket_id].append(
                ("resolution", resolution_reward)
            )
            return result
        # ── RESPOND TO CUSTOMER ───────────────────────────────────
        if action_type == "respond_to_customer":
            reply_text = action.action_value.strip()
            # Grade the customer response using existing reply scoring
            politeness_score, pol_fb = _score_politeness(reply_text)
            length_score, len_fb = _score_length(reply_text)
            pseudo_task = EmailTask(
                task_id=ticket.ticket_id,
                task_type="reply",
                subject=ticket.subject,
                sender=ticket.sender,
                body=ticket.body,
            )
            relevance_score, rel_fb = _score_relevance(reply_text, pseudo_task)
            response_reward = round(
                politeness_score * 0.4 + length_score * 0.3 + relevance_score * 0.3, 2
            )
            feedback = (
                f"Customer Response Score:\n"
                f"  Politeness (40%): {politeness_score:.2f} — {pol_fb}\n"
                f"  Length (30%): {length_score:.2f} — {len_fb}\n"
                f"  Relevance (30%): {relevance_score:.2f} — {rel_fb}\n"
                f"  Response Score: {response_reward:.2f}"
            )
            # Store response reward
            if ticket.ticket_id not in self._ticket_rewards:
                self._ticket_rewards[ticket.ticket_id] = []
            self._ticket_rewards[ticket.ticket_id].append(
                ("response", response_reward)
            )
            # Responding to customer RESOLVES the ticket
            combined = self._resolve_ticket(ticket, response_reward)
            is_last = not self._advance_to_next_ticket()
            feedback += f"\n\n✅ Ticket resolved! Combined ticket reward: {combined:.2f}"
            if is_last:
                feedback += "\n🏁 All tickets resolved — episode complete!"
            result = StepResult(
                task_id=ticket.ticket_id,
                reward=response_reward,
                done=is_last,
                feedback=feedback,
            )
            self._total_reward += response_reward
            self._history.append(result)
            return result
        # ── ESCALATE ──────────────────────────────────────────────
        if action_type == "escalate":
            self._escalation_count += 1
            reason = action.action_value.strip() or "No reason provided"
            # Move to next tier
            if self._current_agent == AgentRole.L1_SUPPORT:
                self._current_agent = AgentRole.L2_SUPPORT
                new_tier = "L2"
            elif self._current_agent == AgentRole.L2_SUPPORT:
                self._current_agent = AgentRole.L3_SUPPORT
                new_tier = "L3"
            else:
                # L3 can't escalate further — stay at L3
                result = StepResult(
                    task_id=ticket.ticket_id,
                    reward=0.0,
                    done=False,
                    feedback="⚠️ L3 is the highest tier — cannot escalate further.",
                )
                self._history.append(result)
                return result
            result = StepResult(
                task_id=ticket.ticket_id,
                reward=0.0,  # No reward for escalation (efficiency penalty later)
                done=False,
                feedback=(
                    f"↗️ Escalated to {new_tier}: {reason}\n"
                    f"Total escalations on this ticket: {self._escalation_count}"
                ),
            )
            self._history.append(result)
            return result
        # ── WRITE KB ENTRY ────────────────────────────────────────
        if action_type == "write_kb_entry":
            try:
                kb_data = json.loads(action.action_value)
            except (json.JSONDecodeError, TypeError):
                # Treat as plain text if not JSON
                kb_data = {
                    "title": f"KB Article for {ticket.ticket_id}",
                    "problem_description": ticket.body[:200],
                    "solution": action.action_value,
                    "keywords": ticket.subject.lower().split(),
                }
            # Grade the KB contribution
            combined_text = (
                kb_data.get("problem_description", "") + " " +
                kb_data.get("solution", "")
            )
            kb_result = grade_kb_contribution(combined_text, ticket)
            # Add to the Knowledge Base if quality is sufficient
            if kb_result.reward >= 0.3:
                new_entry = KBEntry(
                    entry_id="",  # Auto-generated
                    ticket_category=ticket.category,
                    title=kb_data.get("title", f"Article for {ticket.ticket_id}"),
                    problem_description=kb_data.get("problem_description", ""),
                    solution=kb_data.get("solution", ""),
                    keywords=kb_data.get("keywords", []),
                    created_by="l3_agent",
                )
                added = self._kb.add(new_entry)
                self._kb_entries_added += 1
                kb_result.feedback += f"\n\n📝 KB article added as '{added.entry_id}' (KB size: {self._kb.size()})"
            else:
                kb_result.feedback += "\n\n❌ KB article quality too low — not added (threshold: 0.30)"
            # Store KB reward
            if ticket.ticket_id not in self._ticket_rewards:
                self._ticket_rewards[ticket.ticket_id] = []
            self._ticket_rewards[ticket.ticket_id].append(
                ("kb_contribution", kb_result.reward)
            )
            self._total_reward += kb_result.reward
            self._history.append(kb_result)
            return kb_result
        # ── UNKNOWN ACTION (should not reach here due to validation) ─
        return StepResult(
            task_id=ticket.ticket_id,
            reward=0.0,
            done=False,
            feedback=f"Unknown action_type: {action_type}",
        )
    def _resolve_ticket(self, ticket: Ticket, response_reward: float) -> float:
        """Calculate the combined reward for a fully resolved ticket.
        Combines all grading dimensions with the specified weights:
        - Resolution quality:  30%
        - Response quality:    20%
        - Efficiency:          20%
        - Triage accuracy:     15%
        - KB contribution:     15%
        Args:
            ticket: The ticket that was just resolved.
            response_reward: The reward from the customer response.
        Returns:
            The combined weighted reward for this ticket.
        """
        rewards = self._ticket_rewards.get(ticket.ticket_id, [])
        # Gather component scores (use defaults if not found)
        resolution_score = next(
            (r for t, r in rewards if t == "resolution"), 0.5
        )
        response_score = response_reward
        kb_score = next(
            (r for t, r in rewards if t == "kb_contribution"), 0.0
        )
        # Efficiency score (calculated now with final step count)
        efficiency_result = grade_efficiency(
            self._steps_on_ticket, ticket.sla_steps, self._escalation_count
        )
        efficiency_score = efficiency_result.reward
        # Triage score (find from history)
        triage_score = 0.5  # default
        for h in self._history:
            if h.task_id == ticket.ticket_id and h.feedback and "Triage Score" in h.feedback:
                triage_score = h.reward
                break
        # Combined weighted reward
        combined = round(
            resolution_score * 0.30 +
            response_score * 0.20 +
            efficiency_score * 0.20 +
            triage_score * 0.15 +
            kb_score * 0.15,
            4
        )
        return combined
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