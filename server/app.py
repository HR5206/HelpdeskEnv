import os
import logging
from pathlib import Path
from fastapi import FastAPI, Body
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import Any, Dict, List, Optional

from helpdeskenv_class import HelpdeskEnv
from models import (
    HelpdeskAction, HelpdeskEnvState, HelpdeskResetResponse,
    AgentRole, Ticket, TicketCategory, TicketPriority, SupportTier
)

# Configure structured logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Environment variables configuration
API_BASE_URL: str = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-5-nano")

app = FastAPI(title="HelpdeskEnv", version="2.0.0")
_helpdesk_env = HelpdeskEnv()

class ResetRequest(BaseModel):
    seed: Optional[int] = None
    num_tickets: Optional[int] = None

    class Config:
        json_schema_extra = {
            "example": {"seed": 42, "num_tickets": 3}
        }

class CustomTicketRequest(BaseModel):
    subject: str
    sender: str
    body: str
    category: str
    priority: str
    tier: str
    resolution: str
    sla_steps: int = 5
    requires_kb_article: bool = False

    class Config:
        json_schema_extra = {
            "example": {
                "subject": "Printer on fire",
                "sender": "boss@company.com",
                "body": "The 3rd floor printer is literally smoking.",
                "category": "hardware_failure",
                "priority": "critical",
                "tier": "L3",
                "resolution": "Extinguished fire and ordered replacement.",
                "sla_steps": 2,
                "requires_kb_article": True
            }
        }

class StepRequest(BaseModel):
    ticket_id: str
    agent_role: str
    action_type: str
    action_value: str

    class Config:
        json_schema_extra = {
            "example": {
                "ticket_id": "ticket_001",
                "agent_role": "TRIAGE",
                "action_type": "triage",
                "action_value": '{"category": "password_reset", "priority": "medium", "tier": "L1"}',
            }
        }

class StepResponse(BaseModel):
    task_id: str
    reward: float
    done: bool
    feedback: Optional[str]
    correct_answer: Optional[str] = None

# ============================================================================
# Required OpenEnv validator endpoints
# ============================================================================

@app.get("/health")
async def health() -> Dict[str, str]:
    """Health check endpoint â€” must return 'healthy' for OpenEnv validator."""
    return {"status": "healthy"}

@app.get("/metadata")
async def metadata() -> Dict[str, Any]:
    """Environment metadata â€” required by OpenEnv validator."""
    return {
        "name": "HelpdeskEnv",
        "description": (
            "A multi-agent IT Helpdesk OpenEnv environment. "
            "Agents collaborate across triage, L1/L2/L3 support tiers to resolve IT tickets. "
            "Features: multi-agent routing, long-horizon SLA planning, persistent Knowledge Base "
            "for self-improvement, and tiered escalation workflows."
        ),
        "version": "2.1.0",
        "author": "HelpdeskEnv Team",
        "themes": [
            "multi_agent_interaction",
            "long_horizon_planning",
            "world_modeling",
            "self_improving_systems",
        ],
    }

@app.get("/schema")
async def schema() -> Dict[str, Any]:
    """Action/observation/state schemas â€” required by OpenEnv validator."""
    return {
        "action": {
            "type": "object",
            "properties": {
                "ticket_id": {"type": "string"},
                "agent_role": {"type": "string"},
                "action_type": {"type": "string"},
                "action_value": {"type": "string"},
            },
            "required": ["ticket_id", "agent_role", "action_type", "action_value"],
        },
        "observation": {
            "type": "object",
            "properties": {
                "ticket_id": {"type": "string"},
                "category": {"type": "string"},
                "subject": {"type": "string"},
                "sender": {"type": "string"},
                "body": {"type": "string"},
                "context": {"type": "string", "nullable": True},
            },
        },
        "state": {
            "type": "object",
            "properties": {
                "current_ticket": {"type": "object", "nullable": True},
                "current_agent": {"type": "string", "nullable": True},
                "ticket_number": {"type": "integer"},
                "total_tickets": {"type": "integer"},
                "total_reward": {"type": "number"},
                "steps_on_current_ticket": {"type": "integer"},
                "is_done": {"type": "boolean"},
            },
        },
    }

@app.get("/tasks")
async def tasks() -> List[Dict[str, Any]]:
    """List all tasks with grader info â€” required by OpenEnv validator."""
    return [
        {
            "id": "helpdesk_triage",
            "name": "Helpdesk Ticket Triage",
            "description": "Classify a ticket by category, priority, and support tier.",
            "difficulty": "medium",
            "grader": {"type": "python", "path": "graders.py", "function": "grade_triage"},
        },
        {
            "id": "helpdesk_resolution",
            "name": "Helpdesk Ticket Resolution",
            "description": "Resolve IT tickets through multi-agent collaboration.",
            "difficulty": "hard",
            "grader": {"type": "python", "path": "graders.py", "function": "grade_efficiency"},
        },
        {
            "id": "helpdesk_kb_contribution",
            "name": "Knowledge Base Contribution",
            "description": "Write KB articles for novel issues to enable self-improvement.",
            "difficulty": "hard",
            "grader": {"type": "python", "path": "graders.py", "function": "grade_kb_contribution"},
        },
    ]

@app.post("/mcp")
async def mcp(body: Dict[str, Any] = Body(default={})) -> Dict[str, Any]:
    """MCP JSON-RPC endpoint â€” required by OpenEnv validator."""
    return {
        "jsonrpc": "2.0",
        "id": body.get("id", 1),
        "result": {
            "name": "HelpdeskEnv",
            "description": "Multi-agent IT Helpdesk environment",
        },
    }

# ============================================================================
# API endpoints
# ============================================================================

@app.get("/web", response_class=HTMLResponse)
async def home():
    """Serve the live dashboard."""
    dashboard_path = Path(__file__).parent / "live-dashboard.html"
    if dashboard_path.exists():
        return HTMLResponse(content=dashboard_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>Dashboard not found</h1>", status_code=404)

@app.post("/heuristic_step")
async def heuristic_step() -> Dict[str, Any]:
    """Run one heuristic agent step. Used by the dashboard Auto Play."""
    try:
        from heuristics import heuristic_triage, heuristic_l1, heuristic_l2, heuristic_l3
        env_state = _helpdesk_env.state()
        if env_state.is_done or env_state.current_ticket is None:
            return {"error": "Episode is done or not started. Call /reset first.", "done": True}
        ticket = env_state.current_ticket
        agent = env_state.current_agent
        step_num = env_state.steps_on_current_ticket
        if agent == AgentRole.TRIAGE:
            ad = heuristic_triage(ticket)
        elif agent == AgentRole.L1_SUPPORT:
            ad = heuristic_l1(ticket, _helpdesk_env.kb(), step_num)
        elif agent == AgentRole.L2_SUPPORT:
            ad = heuristic_l2(ticket, _helpdesk_env.kb(), step_num)
        else:
            ad = heuristic_l3(ticket, _helpdesk_env.kb(), step_num)
        action = HelpdeskAction(
            ticket_id=ticket.ticket_id,
            agent_role=agent,
            action_type=ad["action_type"],
            action_value=ad["action_value"],
        )
        result = _helpdesk_env.step(action)
        return {
            "task_id": result.task_id,
            "agent_role": agent.value,
            "action_type": ad["action_type"],
            "reward": result.reward,
            "done": result.done,
            "feedback": result.feedback,
        }
    except Exception as e:
        logger.error(f"[HEURISTIC_STEP ERROR] {str(e)}")
        return {"error": str(e), "done": True}


@app.post("/reset")
async def reset(body: Optional[ResetRequest] = Body(None)) -> Dict[str, Any]:
    """Reset the helpdesk environment and start a new episode."""
    try:
        seed = body.seed if body else None
        num_tickets = body.num_tickets if body else None
        response = _helpdesk_env.reset(seed=seed, num_tickets=num_tickets)
        return {
            "observation": response.observation.model_dump(),
            "total_tickets": response.total_tickets,
            "available_actions": response.available_actions,
            "kb_size": response.kb_size,
            "episode": _helpdesk_env.episode_count,
        }
    except Exception as e:
        logger.error(f"[RESET ERROR] {str(e)}")
        raise

@app.post("/reset/custom")
async def reset_custom(body: CustomTicketRequest = Body(...)) -> Dict[str, Any]:
    """Reset the environment with a user-provided custom ticket."""
    try:
        custom_ticket = Ticket(
            ticket_id=f"custom_{_helpdesk_env.episode_count + 1}",
            category=TicketCategory(body.category.lower()),
            subject=body.subject,
            sender=body.sender,
            body=body.body,
            ground_truth_priority=TicketPriority(body.priority.lower()),
            ground_truth_tier=SupportTier(body.tier.upper()),
            ground_truth_resolution=body.resolution,
            sla_steps=body.sla_steps,
            requires_kb_article=body.requires_kb_article
        )
        response = _helpdesk_env.reset(custom_tickets=[custom_ticket])
        return {
            "observation": response.observation.model_dump(),
            "total_tickets": response.total_tickets,
            "available_actions": response.available_actions,
            "kb_size": response.kb_size,
            "episode": _helpdesk_env.episode_count,
        }
    except Exception as e:
        logger.error(f"[CUSTOM RESET ERROR] {str(e)}")
        raise

@app.post("/step", response_model=StepResponse)
async def step(body: StepRequest = Body(...)):
    """Submit an agent action to the helpdesk environment."""
    try:
        action = HelpdeskAction(
            ticket_id=body.ticket_id,
            agent_role=AgentRole(body.agent_role.lower()),
            action_type=body.action_type,
            action_value=body.action_value,
        )
        result = _helpdesk_env.step(action)
        return StepResponse(
            task_id=result.task_id,
            reward=result.reward,
            done=result.done,
            feedback=result.feedback,
            correct_answer=result.correct_answer,
        )
    except Exception as e:
        logger.error(f"[STEP ERROR] {str(e)}")
        raise

@app.get("/state")
async def state() -> Dict[str, Any]:
    """Return the current helpdesk environment state."""
    state = _helpdesk_env.state()
    return {
        "current_ticket": state.current_ticket.model_dump() if state.current_ticket else None,
        "current_agent": state.current_agent.value if state.current_agent else None,
        "ticket_number": state.ticket_number,
        "total_tickets": state.total_tickets,
        "total_reward": state.total_reward,
        "steps_on_current_ticket": state.steps_on_current_ticket,
        "is_done": state.is_done,
        "kb_entries_added": state.kb_entries_added,
        "escalation_count": state.escalation_count,
        "history_length": len(state.history),
    }

@app.get("/kb")
async def kb() -> Dict[str, Any]:
    """Return Knowledge Base statistics and entries."""
    return _helpdesk_env.kb().stats()

@app.get("/kb/search")
async def kb_search(q: str = "", top_k: int = 3) -> Dict[str, Any]:
    """Search the Knowledge Base."""
    results = _helpdesk_env.kb().search(q, top_k=top_k)
    return {
        "query": q,
        "results": [
            {
                "entry_id": r.entry_id,
                "title": r.title,
                "category": r.ticket_category.value,
                "problem_description": r.problem_description[:200],
                "solution": r.solution[:300],
                "keywords": r.keywords,
                "times_used": r.times_used,
            }
            for r in results
        ],
        "total_results": len(results),
    }

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
