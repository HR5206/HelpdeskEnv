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

# Configure structured logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Environment variables configuration
API_BASE_URL: str = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-5-nano")
LOCAL_IMAGE_NAME: str = os.getenv("LOCAL_IMAGE_NAME", "")

app = FastAPI(title="EmailEnv", version="1.0.0")
_env = EmailEnv()


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
		"name": "EmailEnv",
		"description": "An OpenEnv-compliant environment for professional email triage and customer support, including spam detection, prioritization, and reply generation.",
		"version": "1.0.0",
		"author": "EmailEnv Maintainers",
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
		{
			"id": "spam_classification",
			"name": "Spam Classification",
			"description": "Classify an incoming email as spam or not_spam.",
			"difficulty": "easy",
			"grader": {
				"type": "python",
				"path": "graders.py",
				"function": "grade_spam",
			},
		},
		{
			"id": "email_prioritization",
			"name": "Email Prioritization",
			"description": "Assign a priority level (high, medium, low) to an incoming email.",
			"difficulty": "medium",
			"grader": {
				"type": "python",
				"path": "graders.py",
				"function": "grade_priority",
			},
		},
		{
			"id": "reply_generation",
			"name": "Reply Generation",
			"description": "Draft a polite, relevant, and appropriately-lengthed reply to an email.",
			"difficulty": "hard",
			"grader": {
				"type": "python",
				"path": "graders.py",
				"function": "grade_reply",
			},
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
	<title>EmailEnv</title>
	<style>
		* { margin: 0; padding: 0; box-sizing: border-box; }
		body { 
			font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
			background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
			min-height: 100vh;
			display: flex;
			align-items: center;
			justify-content: center;
			padding: 20px;
		}
		.container {
			background: white;
			border-radius: 10px;
			box-shadow: 0 10px 40px rgba(0,0,0,0.2);
			padding: 40px;
			max-width: 600px;
			text-align: center;
		}
		h1 { color: #333; margin-bottom: 10px; font-size: 2.5em; }
		.emoji { font-size: 3em; display: block; margin: 20px 0; }
		p { color: #666; line-height: 1.8; margin: 15px 0; }
		.tasks { background: #f5f5f5; border-radius: 8px; padding: 20px; margin: 25px 0; text-align: left; }
		.tasks h2 { color: #333; margin-bottom: 15px; font-size: 1.2em; }
		.task-item { padding: 10px 0; border-bottom: 1px solid #ddd; }
		.task-item:last-child { border-bottom: none; }
		.task-item strong { color: #667eea; }
		.links { margin-top: 30px; display: flex; gap: 10px; justify-content: center; flex-wrap: wrap; }
		a { display: inline-block; padding: 10px 20px; background: #667eea; color: white; text-decoration: none; border-radius: 5px; transition: background 0.3s; }
		a:hover { background: #764ba2; }
	</style>
</head>
<body>
	<div class=\"container\">
		<span class=\"emoji\">📧</span>
		<h1>EmailEnv</h1>
		<p><strong>OpenEnv environment for email triage and customer support</strong></p>
		<div class=\"tasks\">
			<h2>Tasks</h2>
			<div class=\"task-item\"><strong>Spam Detection</strong> - Classify emails as spam or legitimate</div>
			<div class=\"task-item\"><strong>Prioritization</strong> - Assign priority levels (low/medium/high)</div>
			<div class=\"task-item\"><strong>Reply Generation</strong> - Generate customer support replies</div>
		</div>
		<div class=\"links\">
			<a href=\"/docs\">API Docs</a>
			<a href=\"https://github.com/HR5206/emailenv\">🔗 GitHub</a>
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


def main():
	import uvicorn
	uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
	main()