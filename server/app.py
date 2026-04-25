import os
import logging
from fastapi import FastAPI, Body
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Any, Dict, List, Optional

from emailenv_class import EmailEnv
from models import (
	Action,
	Observation,
	State,
	Reward,
	ResetResponse,
	EmailTask,
	EnvState,
	AgentAction,
)

from helpdeskenv_class import HelpdeskEnv
from models import (
    HelpdeskAction, HelpdeskEnvState, HelpdeskResetResponse,
    AgentRole, Ticket,
)

# Configure structured logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Environment variables configuration
API_BASE_URL: str = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-5-nano")
LOCAL_IMAGE_NAME: str = os.getenv("LOCAL_IMAGE_NAME", "")

app = FastAPI(title="EmailEnv", version="1.0.0")
_env = EmailEnv()
_helpdesk_env = HelpdeskEnv()

class ResetRequest(BaseModel):
	task: Optional[str] = None

	class Config:
		json_schema_extra = {
			"example": {"task": "spam_classification"}
		}


class StepRequest(BaseModel):
	action: Action

	class Config:
		json_schema_extra = {
			"example": {
				"action": {
					"type": "classify_spam",
					"is_spam": True,
					"priority": None,
					"reply_text": None,
				}
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
	"""Health check endpoint — must return 'healthy' for OpenEnv validator."""
	return {"status": "healthy"}


@app.get("/metadata")
async def metadata() -> Dict[str, Any]:
    """Environment metadata — required by OpenEnv validator."""
    return {
        "name": "HelpdeskEnv",
        "description": (
            "A multi-agent IT Helpdesk OpenEnv environment. "
            "Agents collaborate across triage, L1/L2/L3 support tiers to resolve IT tickets. "
            "Features: multi-agent routing, long-horizon SLA planning, persistent Knowledge Base "
            "for self-improvement, and tiered escalation workflows. "
            "Also includes EmailEnv tasks for spam detection, prioritization, and reply generation."
        ),
        "version": "2.0.0",
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
	"""Action/observation/state schemas — required by OpenEnv validator."""
	return {
		"action": {
			"type": "object",
			"properties": {
				"type": {"type": "string", "enum": ["classify_spam", "set_priority", "generate_reply", "skip"]},
				"is_spam": {"type": "boolean", "nullable": True},
				"priority": {"type": "string", "enum": ["low", "medium", "high"], "nullable": True},
				"reply_text": {"type": "string", "nullable": True},
			},
			"required": ["type"],
		},
		"observation": {
			"type": "object",
			"properties": {
				"task_id": {"type": "string"},
				"task_type": {"type": "string"},
				"subject": {"type": "string"},
				"sender": {"type": "string"},
				"body": {"type": "string"},
				"context": {"type": "string", "nullable": True},
			},
		},
		"state": {
			"type": "object",
			"properties": {
				"current_task": {"type": "object", "nullable": True},
				"task_number": {"type": "integer"},
				"total_reward": {"type": "number"},
				"is_done": {"type": "boolean"},
			},
		},
	}


@app.get("/tasks")
async def tasks() -> List[Dict[str, Any]]:
    """List all tasks with grader info — required by OpenEnv validator."""
    return [
        # EmailEnv tasks (Round 1)
        {
            "id": "spam_classification",
            "name": "Spam Classification",
            "description": "Classify an incoming email as spam or not_spam.",
            "difficulty": "easy",
            "grader": {"type": "python", "path": "graders.py", "function": "grade_spam"},
        },
        {
            "id": "email_prioritization",
            "name": "Email Prioritization",
            "description": "Assign a priority level (low/medium/high/critical) to an email.",
            "difficulty": "medium",
            "grader": {"type": "python", "path": "graders.py", "function": "grade_priority"},
        },
        {
            "id": "reply_generation",
            "name": "Reply Generation",
            "description": "Draft a polite, relevant reply to an email.",
            "difficulty": "hard",
            "grader": {"type": "python", "path": "graders.py", "function": "grade_reply"},
        },
        # HelpdeskEnv tasks (Round 2)
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
	"""MCP JSON-RPC endpoint — required by OpenEnv validator."""
	return {
		"jsonrpc": "2.0",
		"id": body.get("id", 1),
		"result": {
			"name": "EmailEnv",
			"description": "Email triage environment",
		},
	}


# ============================================================================
# Original API endpoints (unchanged)
# ============================================================================

@app.get("/web", response_class=HTMLResponse)
async def home():
	"""Serve the homepage."""
	return """<!DOCTYPE html>
<html lang=\"en\">
<head>
	<meta charset=\"UTF-8\">
	<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
	<title>HelpdeskEnv</title>
	<style>
		* { margin: 0; padding: 0; box-sizing: border-box; }
		body { 
			font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
			background: linear-gradient(135deg, #4338ca 0%, #7e22ce 100%);
			min-height: 100vh;
			display: flex;
			align-items: center;
			justify-content: center;
			padding: 20px;
		}
		.container {
			background: white;
			border-radius: 12px;
			box-shadow: 0 20px 50px rgba(0,0,0,0.3);
			padding: 40px;
			max-width: 650px;
			text-align: center;
		}
		h1 { color: #1e1b4b; margin-bottom: 10px; font-size: 2.8em; }
		.emoji { font-size: 3.5em; display: block; margin: 15px 0; }
		p.subtitle { color: #4f46e5; font-weight: bold; font-size: 1.1em; margin-bottom: 15px; }
		p.desc { color: #475569; line-height: 1.6; margin-bottom: 25px; }
		.tasks { background: #f8fafc; border-radius: 10px; padding: 25px; margin: 25px 0; text-align: left; border: 1px solid #e2e8f0; }
		.tasks h2 { color: #334155; margin-bottom: 15px; font-size: 1.3em; border-bottom: 2px solid #e2e8f0; padding-bottom: 10px; }
		.task-item { padding: 12px 0; border-bottom: 1px solid #e2e8f0; }
		.task-item:last-child { border-bottom: none; }
		.task-item strong { color: #4338ca; }
		.links { margin-top: 30px; display: flex; gap: 15px; justify-content: center; flex-wrap: wrap; }
		a { display: inline-block; padding: 12px 24px; background: #4f46e5; color: white; text-decoration: none; border-radius: 8px; font-weight: 600; transition: all 0.2s ease; box-shadow: 0 4px 6px rgba(79, 70, 229, 0.2); }
		a:hover { background: #4338ca; transform: translateY(-2px); box-shadow: 0 6px 12px rgba(79, 70, 229, 0.3); }
	</style>
</head>
<body>
	<div class=\"container\">
		<span class=\"emoji\">🎫</span>
		<h1>HelpdeskEnv</h1>
		<p class=\"subtitle\">Multi-agent IT Helpdesk Environment (v2.0.0)</p>
		<p class=\"desc\">An OpenEnv-compatible framework featuring tiered support agents (Triage → L1 → L2 → L3), a self-improving Knowledge Base, and SLA-driven resolution planning.</p>
		<div class=\"tasks\">
			<h2>Key Tasks</h2>
			<div class=\"task-item\"><strong>Ticket Triage</strong> - Classify incoming issues by category, priority, and support tier</div>
			<div class=\"task-item\"><strong>Multi-Agent Resolution</strong> - Collaborate across L1/L2/L3 agents within SLA constraints</div>
			<div class=\"task-item\"><strong>KB Contribution</strong> - Write articles for novel issues to enable self-improvement</div>
			<div class=\"task-item\" style=\"color: #64748b; font-size: 0.9em; margin-top: 5px;\"><em>+ Includes 3 legacy EmailEnv tasks (Spam, Prioritization, Replies)</em></div>
		</div>
		<div class=\"links\">
			<a href=\"/docs\">API Docs</a>
			<a href=\"https://github.com/HR5206/HelpdeskEnv\">🔗 GitHub</a>
		</div>
	</div>
</body>
</html>"""


@app.post("/reset", response_model=ResetResponse)
async def reset(body: Optional[ResetRequest] = Body(None)):
	"""Reset the environment."""
	try:
		logger.info(f"[START] task=multi env=emailenv model={MODEL_NAME}")
		result = _env.reset(seed=None)
		return ResetResponse(
			observation=result.observation,
			available_tasks=[
				"spam_classification",
				"email_prioritization",
				"reply_generation",
			],
		)
	except Exception as e:
		logger.error(f"[RESET ERROR] {str(e)}")
		raise


@app.post("/step", response_model=StepResponse)
async def step(body: Action | StepRequest = Body(...)):
	"""Take a step in the environment."""
	try:
		if isinstance(body, StepRequest):
			action = body.action
		else:
			action = body

		current_state = _env.state()
		current_task = current_state.current_task

		if not current_task:
			raise ValueError("No current task. Call /reset first.")

		action_value = None
		if action.type == "classify_spam" and action.is_spam is not None:
			action_value = "spam" if action.is_spam else "not_spam"
		elif action.type == "set_priority" and action.priority is not None:
			action_value = action.priority
		elif action.type == "generate_reply" and action.reply_text is not None:
			action_value = action.reply_text
		elif action.type == "skip":
			action_value = "skip"
		else:
			raise ValueError(f"Invalid action: {action.type}")

		agent_action = AgentAction(
			task_id=current_task.task_id,
			action_value=action_value,
		)

		action_str = f"type={action.type}"
		result = _env.step(agent_action)

		logger.info(
			f"[STEP] step={result.task_id} action={action_str} "
			f"reward={result.reward:.2f} done={str(result.done).lower()} error=null"
		)

		return StepResponse(
			task_id=result.task_id,
			reward=result.reward,
			done=result.done,
			feedback=result.feedback,
			correct_answer=result.correct_answer,
		)
	except Exception as e:
		logger.info(
			f"[STEP] step=unknown action=unknown reward=0.00 done=false error={str(e)}"
		)
		raise


@app.get("/state", response_model=EnvState)
async def state() -> EnvState:
	"""Return the current environment state."""
	return _env.state()

class HelpdeskResetRequest(BaseModel):
    seed: Optional[int] = None
    num_tickets: Optional[int] = None
    class Config:
        json_schema_extra = {
            "example": {"seed": 42, "num_tickets": 3}
        }
class HelpdeskStepRequest(BaseModel):
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
@app.post("/helpdesk/reset")
async def helpdesk_reset(body: Optional[HelpdeskResetRequest] = Body(None)) -> Dict[str, Any]:
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
        logger.error(f"[HELPDESK RESET ERROR] {str(e)}")
        raise
@app.post("/helpdesk/step")
async def helpdesk_step(body: HelpdeskStepRequest = Body(...)) -> Dict[str, Any]:
    """Submit an agent action to the helpdesk environment."""
    try:
        action = HelpdeskAction(
            ticket_id=body.ticket_id,
            agent_role=AgentRole(body.agent_role.lower()),
            action_type=body.action_type,
            action_value=body.action_value,
        )
        result = _helpdesk_env.step(action)
        return {
            "task_id": result.task_id,
            "reward": result.reward,
            "done": result.done,
            "feedback": result.feedback,
            "correct_answer": result.correct_answer,
        }
    except Exception as e:
        logger.error(f"[HELPDESK STEP ERROR] {str(e)}")
        raise
@app.get("/helpdesk/state")
async def helpdesk_state() -> Dict[str, Any]:
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
@app.get("/helpdesk/kb")
async def helpdesk_kb() -> Dict[str, Any]:
    """Return Knowledge Base statistics and entries."""
    return _helpdesk_env.kb().stats()
@app.get("/helpdesk/kb/search")
async def helpdesk_kb_search(q: str = "", top_k: int = 3) -> Dict[str, Any]:
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