# test_integration.py
"""Integration tests for the HelpdeskEnv multi-agent system.
Validates the entire system end-to-end:
- Models, tasks, graders, knowledge base
- HelpdeskEnv reset/step/state lifecycle
- Multi-agent triage -> support -> resolution flow
- KB persistence across episodes (self-improvement)
- Heuristic agents produce valid actions
Run with: python test_integration.py
"""
import json
import sys
from typing import Dict
from models import (
    Ticket, HelpdeskAction, AgentRole, TicketCategory,
    TicketPriority, SupportTier, StepResult, EmailTask, AgentAction,
)
from knowledge_base import KnowledgeBase, KBEntry
from tasks import get_all_ticket_scenarios, get_ticket_scenario
from graders import (
    grade_triage, grade_efficiency,
    grade_kb_contribution
)
from helpdeskenv_class import HelpdeskEnv
from heuristics import heuristic_triage, heuristic_l1, heuristic_l2, heuristic_l3
# ============================================================================
# Test helpers
# ============================================================================
_pass_count = 0
_fail_count = 0
def check(name: str, condition: bool, detail: str = "") -> None:
    """Assert a condition, tracking pass/fail counts."""
    global _pass_count, _fail_count
    if condition:
        _pass_count += 1
        print(f"  [PASS] {name}")
    else:
        _fail_count += 1
        msg = f"  [FAIL] {name}"
        if detail:
            msg += f" — {detail}"
        print(msg)
# ============================================================================
# Test 1: Knowledge Base initialization and search
# ============================================================================
def test_knowledge_base():
    print("\n[Test 1] Knowledge Base — init, search, add")
    kb = KnowledgeBase()
    check("KB has 2 seed entries", kb.size() == 2)
    # Search password
    results = kb.search("password expired locked out")
    check("Password search returns results", len(results) >= 1)
    check("Top result is password-related", "password" in results[0].title.lower() if results else False)
    # Search software
    results = kb.search("install visual studio python IDE")
    check("Software search returns results", len(results) >= 1)
    # No match
    results = kb.search("quantum computing dark matter")
    check("Unrelated search returns empty", len(results) == 0)
    # Add entry
    entry = KBEntry(
        entry_id="",
        ticket_category=TicketCategory.NETWORK_ISSUE,
        title="Test Network Article",
        problem_description="Switch crash",
        solution="Reboot switch",
        keywords=["network", "switch", "crash"],
        created_by="l3_agent",
    )
    added = kb.add(entry)
    check("New entry added", kb.size() == 3)
    check("Entry got an ID", len(added.entry_id) > 0)
    # Stats
    stats = kb.stats()
    check("Stats show 3 entries", stats["total_entries"] == 3)
    check("Stats show 1 agent entry", stats["agent_created_entries"] == 1)
# ============================================================================
# Test 2: Triage grader
# ============================================================================
def test_triage_grader():
    print("\n[Test 2] Triage Grader — correct and incorrect classifications")
    ticket = get_ticket_scenario("ticket_001")
    # Perfect triage
    perfect_action = HelpdeskAction(
        ticket_id=ticket.ticket_id,
        agent_role=AgentRole.TRIAGE,
        action_type="triage",
        action_value=json.dumps({
            "category": ticket.category.value,
            "priority": ticket.ground_truth_priority.value,
            "tier": ticket.ground_truth_tier.value,
        }),
    )
    result = grade_triage(ticket, perfect_action)
    check("Perfect triage scores 1.0", result.reward == 1.0, f"got {result.reward}")
    # Wrong everything
    wrong_action = HelpdeskAction(
        ticket_id=ticket.ticket_id,
        agent_role=AgentRole.TRIAGE,
        action_type="triage",
        action_value=json.dumps({
            "category": "network_issue",
            "priority": "critical",
            "tier": "L3",
        }),
    )
    result = grade_triage(ticket, wrong_action)
    check("Wrong triage scores < 0.5", result.reward < 0.5, f"got {result.reward}")
    # Invalid JSON
    bad_action = HelpdeskAction(
        ticket_id=ticket.ticket_id,
        agent_role=AgentRole.TRIAGE,
        action_type="triage",
        action_value="not json at all",
    )
    result = grade_triage(ticket, bad_action)
    check("Invalid JSON scores 0.0", result.reward == 0.0)
# ============================================================================
# Test 3: Full L1 ticket workflow (triage -> search -> apply -> respond)
# ============================================================================
def test_l1_workflow():
    print("\n[Test 3] Full L1 Workflow -- triage -> search -> apply -> respond")
    env = HelpdeskEnv()
    response = env.reset(seed=42, num_tickets=1)
    ticket = response.observation
    # Triage
    triage_data = heuristic_triage(ticket)
    triage_action = HelpdeskAction(
        ticket_id=ticket.ticket_id,
        agent_role=AgentRole.TRIAGE,
        action_type=triage_data["action_type"],
        action_value=triage_data["action_value"],
    )
    result = env.step(triage_action)
    check("Triage step succeeds", result.reward >= 0.0)
    state = env.state()
    agent = state.current_agent
    check("Agent changed from TRIAGE", agent != AgentRole.TRIAGE)
    # Support steps
    for step_num in range(3):
        state = env.state()
        if state.is_done:
            break
        agent = state.current_agent
        if agent == AgentRole.L1_SUPPORT:
            action_data = heuristic_l1(ticket, env.kb(), step=step_num)
        elif agent == AgentRole.L2_SUPPORT:
            action_data = heuristic_l2(ticket, env.kb(), step=step_num)
        else:
            action_data = heuristic_l3(ticket, env.kb(), step=step_num)
        action = HelpdeskAction(
            ticket_id=ticket.ticket_id,
            agent_role=agent,
            action_type=action_data["action_type"],
            action_value=action_data["action_value"],
        )
        result = env.step(action)
    check("Episode completed", env.state().is_done)
    check("Total reward > 0", env.state().total_reward > 0)
# ============================================================================
# Test 4: L3 with KB article writing
# ============================================================================
def test_l3_kb_article():
    print("\n[Test 4] L3 Agent — KB article writing for novel issues")
    env = HelpdeskEnv()
    initial_kb_size = env.kb().size()
    # Use a hard ticket that requires KB article
    ticket = get_ticket_scenario("ticket_003")
    env.reset(seed=100, num_tickets=5)
    # Simulate L3 writing a KB article
    kb_entry_json = json.dumps({
        "title": "Network Switch Firmware Crash Recovery",
        "problem_description": "Cisco switch firmware crash causing floor-wide outage",
        "solution": (
            "Root cause: firmware crash on core switch.\n"
            "Steps to resolve:\n"
            "1. Diagnosed root cause via switch logs\n"
            "2. Applied failover to redundant switch\n"
            "3. Updated firmware on crashed switch\n"
            "4. Verified restored connectivity\n"
            "5. Confirmed with affected users"
        ),
        "keywords": ["network", "switch", "firmware", "crash", "outage", "cisco"],
    })
    kb_result = grade_kb_contribution(kb_entry_json, ticket)
    check("KB article scores > 0.3", kb_result.reward >= 0.3, f"got {kb_result.reward}")
    check("KB article has feedback", len(kb_result.feedback) > 0)
# ============================================================================
# Test 5: Escalation mechanics
# ============================================================================
def test_escalation():
    print("\n[Test 5] Escalation -- L1 -> L2 -> L3")
    env = HelpdeskEnv()
    env.reset(seed=42, num_tickets=1)
    ticket = env.state().current_ticket
    # Triage first
    triage = heuristic_triage(ticket)
    env.step(HelpdeskAction(
        ticket_id=ticket.ticket_id,
        agent_role=AgentRole.TRIAGE,
        action_type=triage["action_type"],
        action_value=triage["action_value"],
    ))
    state = env.state()
    initial_agent = state.current_agent
    # Try escalating if at L1 or L2
    if initial_agent in (AgentRole.L1_SUPPORT, AgentRole.L2_SUPPORT):
        result = env.step(HelpdeskAction(
            ticket_id=ticket.ticket_id,
            agent_role=initial_agent,
            action_type="escalate",
            action_value="Issue too complex for this tier",
        ))
        new_agent = env.state().current_agent
        check("Escalation changes agent", new_agent != initial_agent)
        check("Escalation reward is 0", result.reward == 0.0)
    else:
        check("Ticket assigned to L3 (skip escalation test)", True)
# ============================================================================
# Test 6: Multi-ticket episode
# ============================================================================
def test_multi_ticket():
    print("\n[Test 6] Multi-ticket episode — all 5 tickets")
    env = HelpdeskEnv()
    response = env.reset(seed=42, num_tickets=5)
    check("5 tickets loaded", response.total_tickets == 5)
    steps = 0
    max_steps = 50
    while not env.state().is_done and steps < max_steps:
        state = env.state()
        ticket = state.current_ticket
        agent = state.current_agent
        if ticket is None or agent is None:
            break
        support_step = max(0, state.steps_on_current_ticket - 1)
        if agent == AgentRole.TRIAGE:
            action_data = heuristic_triage(ticket)
        elif agent == AgentRole.L1_SUPPORT:
            action_data = heuristic_l1(ticket, env.kb(), step=support_step)
        elif agent == AgentRole.L2_SUPPORT:
            action_data = heuristic_l2(ticket, env.kb(), step=support_step)
        else:
            action_data = heuristic_l3(ticket, env.kb(), step=support_step)
        env.step(HelpdeskAction(
            ticket_id=ticket.ticket_id,
            agent_role=agent,
            action_type=action_data["action_type"],
            action_value=action_data["action_value"],
        ))
        steps += 1
    check("Episode completed", env.state().is_done)
    check("Total reward > 0", env.state().total_reward > 0, f"got {env.state().total_reward}")
    check(f"Completed in {steps} steps", steps <= max_steps)
# ============================================================================
# Test 7: KB persistence across episodes (self-improvement)
# ============================================================================
def test_kb_persistence():
    print("\n[Test 7] KB Persistence — self-improvement across episodes")
    env = HelpdeskEnv()
    initial_kb = env.kb().size()
    check(f"Initial KB size: {initial_kb}", initial_kb == 2)
    # Run episode 1
    env.reset(seed=42, num_tickets=5)
    steps = 0
    while not env.state().is_done and steps < 50:
        state = env.state()
        ticket = state.current_ticket
        agent = state.current_agent
        if ticket is None or agent is None:
            break
        s = max(0, state.steps_on_current_ticket - 1)
        if agent == AgentRole.TRIAGE:
            ad = heuristic_triage(ticket)
        elif agent == AgentRole.L1_SUPPORT:
            ad = heuristic_l1(ticket, env.kb(), s)
        elif agent == AgentRole.L2_SUPPORT:
            ad = heuristic_l2(ticket, env.kb(), s)
        else:
            ad = heuristic_l3(ticket, env.kb(), s)
        env.step(HelpdeskAction(
            ticket_id=ticket.ticket_id, agent_role=agent,
            action_type=ad["action_type"], action_value=ad["action_value"],
        ))
        steps += 1
    kb_after_ep1 = env.kb().size()
    check(f"KB grew after episode 1: {initial_kb} -> {kb_after_ep1}", kb_after_ep1 > initial_kb)
    # Run episode 2 -- KB should persist!
    env.reset(seed=43, num_tickets=5)
    kb_after_reset = env.kb().size()
    check(f"KB persisted after reset: {kb_after_reset}", kb_after_reset == kb_after_ep1)
# ============================================================================
# Test 8: Heuristic agents produce valid actions
# ============================================================================
def test_heuristics():
    print("\n[Test 8] Heuristic Agents -- produce valid action formats")
    tickets = get_all_ticket_scenarios()
    kb = KnowledgeBase()
    for ticket in tickets[:3]:
        # Triage
        triage = heuristic_triage(ticket)
        check(f"Triage {ticket.ticket_id}: valid type", triage["action_type"] == "triage")
        parsed = json.loads(triage["action_value"])
        check(f"Triage {ticket.ticket_id}: has category", "category" in parsed)
        check(f"Triage {ticket.ticket_id}: has priority", "priority" in parsed)
        check(f"Triage {ticket.ticket_id}: has tier", "tier" in parsed)
# ============================================================================
# Test 9: Full inference loop (smoke test)
# ============================================================================
def test_inference_loop():
    print("\n[Test 9] Full Inference Loop — smoke test")
    try:
        from inference import run_helpdesk_episode
        env = HelpdeskEnv()
        result = run_helpdesk_episode(env, seed=42, num_tickets=2)
        check("Inference returns result", isinstance(result, dict))
        check("Result has score", "score" in result)
        check("Result has kb_size", "kb_size" in result)
        check(f"Score: {result['score']:.4f}", result["score"] >= 0.0)
    except Exception as e:
        check(f"Inference loop ran", False, str(e))
# ============================================================================
# Main — run all tests
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("HelpdeskEnv — Integration Tests")
    print("=" * 60)
    test_knowledge_base()
    test_triage_grader()
    test_l1_workflow()
    test_l3_kb_article()
    test_escalation()
    test_multi_ticket()
    test_kb_persistence()
    test_heuristics()
    test_inference_loop()
    print(f"\n{'=' * 60}")
    print(f"Results: {_pass_count} passed, {_fail_count} failed")
    print(f"{'=' * 60}")
    if _fail_count > 0:
        print("\n[WARNING] SOME TESTS FAILED — review output above")
        sys.exit(1)
    else:
        print("\n[SUCCESS] ALL TESTS PASSED!")
        sys.exit(0)