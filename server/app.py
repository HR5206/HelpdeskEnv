import os
import logging
from fastapi import FastAPI, Body
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Any, Dict, List, Optional

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

app = FastAPI(title="HelpdeskEnv", version="2.0.0")
_helpdesk_env = HelpdeskEnv()

class ResetRequest(BaseModel):
    seed: Optional[int] = None
    num_tickets: Optional[int] = None

    class Config:
        json_schema_extra = {
            "example": {"seed": 42, "num_tickets": 3}
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
            "for self-improvement, and tiered escalation workflows."
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
    """List all tasks with grader info — required by OpenEnv validator."""
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
    """MCP JSON-RPC endpoint — required by OpenEnv validator."""
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
    """Serve the homepage."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HelpdeskEnv Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        :root {
            --bg: #0f172a; --surface: #1e293b; --surface2: #334155;
            --text: #f1f5f9; --muted: #94a3b8; --accent: #6366f1;
            --accent2: #818cf8; --green: #10b981; --amber: #f59e0b;
            --red: #ef4444; --border: rgba(148,163,184,0.15);
        }
        body { font-family: 'Inter', sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; }
        .header { padding: 40px 0 20px; text-align: center; }
        .header h1 { font-size: 2.2em; font-weight: 800; background: linear-gradient(135deg, var(--accent), var(--green)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .header p { color: var(--muted); margin-top: 8px; font-size: 1.05em; }
        .container { max-width: 1100px; margin: 0 auto; padding: 0 24px 60px; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 24px; }
        .card { background: var(--surface); border: 1px solid var(--border); border-radius: 16px; padding: 24px; transition: transform 0.2s, box-shadow 0.2s; }
        .card:hover { transform: translateY(-2px); box-shadow: 0 8px 32px rgba(0,0,0,0.3); }
        .card h2 { font-size: 1.1em; font-weight: 700; margin-bottom: 16px; color: var(--accent2); }
        .card.full { grid-column: 1 / -1; }
        /* Pipeline */
        .pipeline { display: flex; align-items: center; justify-content: center; gap: 8px; flex-wrap: wrap; }
        .agent-node { padding: 14px 20px; border-radius: 12px; font-weight: 600; font-size: 0.9em; background: var(--surface2); border: 2px solid var(--border); transition: all 0.4s; position: relative; }
        .agent-node.active { border-color: var(--accent); background: rgba(99,102,241,0.15); box-shadow: 0 0 20px rgba(99,102,241,0.3); animation: pulse 1.5s infinite; }
        .agent-node.done { border-color: var(--green); background: rgba(16,185,129,0.1); }
        .arrow { color: var(--muted); font-size: 1.2em; }
        @keyframes pulse { 0%,100% { box-shadow: 0 0 20px rgba(99,102,241,0.2); } 50% { box-shadow: 0 0 30px rgba(99,102,241,0.5); } }
        /* Reward bars */
        .reward-row { display: flex; align-items: center; margin: 8px 0; gap: 12px; }
        .reward-label { width: 90px; font-size: 0.85em; color: var(--muted); font-weight: 500; }
        .reward-bar-bg { flex: 1; height: 10px; background: var(--surface2); border-radius: 5px; overflow: hidden; }
        .reward-bar { height: 100%; border-radius: 5px; transition: width 1s ease; }
        .reward-val { width: 45px; text-align: right; font-size: 0.85em; font-weight: 600; }
        /* Metrics */
        .metric { text-align: center; padding: 16px; }
        .metric-val { font-size: 2em; font-weight: 800; }
        .metric-label { font-size: 0.8em; color: var(--muted); margin-top: 4px; }
        .metrics-row { display: flex; justify-content: space-around; }
        /* Log */
        .log { background: var(--bg); border: 1px solid var(--border); border-radius: 10px; padding: 16px; max-height: 220px; overflow-y: auto; font-family: 'Courier New', monospace; font-size: 0.8em; line-height: 1.6; }
        .log-entry { padding: 2px 0; }
        .log-entry .ts { color: var(--muted); }
        .log-entry .ag { color: var(--accent2); font-weight: 600; }
        .log-entry .rw { color: var(--green); }
        /* Buttons */
        .btn { padding: 12px 28px; border: none; border-radius: 10px; font-weight: 700; font-size: 0.95em; cursor: pointer; transition: all 0.2s; }
        .btn-primary { background: linear-gradient(135deg, var(--accent), #7c3aed); color: white; }
        .btn-primary:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(99,102,241,0.4); }
        .btn-primary:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }
        .btn-group { display: flex; gap: 12px; justify-content: center; margin: 20px 0; }
        /* KB */
        .kb-stat { display: flex; align-items: center; gap: 16px; padding: 12px; background: var(--bg); border-radius: 10px; margin: 6px 0; }
        .kb-icon { width: 40px; height: 40px; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 1.2em; }
        .kb-info { flex: 1; }
        .kb-info strong { font-size: 0.95em; }
        .kb-info small { color: var(--muted); font-size: 0.8em; }
        /* Responsive */
        @media (max-width: 700px) { .grid { grid-template-columns: 1fr; } .pipeline { flex-direction: column; } .arrow { transform: rotate(90deg); } }
        /* Tags */
        .tag { display: inline-block; padding: 4px 10px; border-radius: 6px; font-size: 0.75em; font-weight: 600; margin: 2px; }
        .tag-blue { background: rgba(99,102,241,0.15); color: var(--accent2); }
        .tag-green { background: rgba(16,185,129,0.15); color: var(--green); }
        .tag-amber { background: rgba(245,158,11,0.15); color: var(--amber); }
        .links-row { display: flex; gap: 12px; justify-content: center; margin-top: 20px; }
        .links-row a { color: var(--accent2); text-decoration: none; font-weight: 600; font-size: 0.9em; padding: 8px 16px; border: 1px solid var(--border); border-radius: 8px; transition: all 0.2s; }
        .links-row a:hover { background: var(--surface2); }
    </style>
</head>
<body>
<div class="header">
    <h1>HelpdeskEnv</h1>
    <p>Multi-Agent IT Helpdesk -- OpenEnv Environment v2.1.0</p>
    <div style="margin-top:12px">
        <span class="tag tag-blue">Multi-Agent</span>
        <span class="tag tag-green">Self-Improving KB</span>
        <span class="tag tag-amber">SLA Pressure</span>
        <span class="tag tag-blue">TRL GRPO</span>
    </div>
</div>
<div class="container">
    <!-- Pipeline -->
    <div class="card full">
        <h2>Agent Pipeline</h2>
        <div class="pipeline" id="pipeline">
            <div class="agent-node" id="node-triage">Triage</div>
            <span class="arrow">&#8594;</span>
            <div class="agent-node" id="node-l1">L1 Support</div>
            <span class="arrow">&#8594;</span>
            <div class="agent-node" id="node-l2">L2 Support</div>
            <span class="arrow">&#8594;</span>
            <div class="agent-node" id="node-l3">L3 Expert</div>
        </div>
        <div class="btn-group">
            <button class="btn btn-primary" id="btn-demo" onclick="runDemo()">Run Live Demo</button>
        </div>
        <div class="log" id="log"><div class="log-entry"><span class="ts">[ready]</span> Click "Run Live Demo" to simulate a ticket...</div></div>
    </div>
    <div class="grid">
        <!-- Reward Breakdown -->
        <div class="card">
            <h2>Reward Breakdown</h2>
            <div class="reward-row"><span class="reward-label">Triage</span><div class="reward-bar-bg"><div class="reward-bar" id="bar-triage" style="width:0%;background:var(--accent)"></div></div><span class="reward-val" id="val-triage">--</span></div>
            <div class="reward-row"><span class="reward-label">Resolution</span><div class="reward-bar-bg"><div class="reward-bar" id="bar-resolution" style="width:0%;background:var(--green)"></div></div><span class="reward-val" id="val-resolution">--</span></div>
            <div class="reward-row"><span class="reward-label">Response</span><div class="reward-bar-bg"><div class="reward-bar" id="bar-response" style="width:0%;background:var(--accent2)"></div></div><span class="reward-val" id="val-response">--</span></div>
            <div class="reward-row"><span class="reward-label">Efficiency</span><div class="reward-bar-bg"><div class="reward-bar" id="bar-efficiency" style="width:0%;background:var(--amber)"></div></div><span class="reward-val" id="val-efficiency">--</span></div>
            <div class="reward-row"><span class="reward-label">KB</span><div class="reward-bar-bg"><div class="reward-bar" id="bar-kb" style="width:0%;background:#8b5cf6"></div></div><span class="reward-val" id="val-kb">--</span></div>
            <div style="margin-top:16px;padding-top:12px;border-top:1px solid var(--border);display:flex;justify-content:space-between;align-items:center">
                <span style="font-weight:700;color:var(--muted)">Composite</span>
                <span style="font-size:1.4em;font-weight:800" id="val-composite">--</span>
            </div>
        </div>
        <!-- Baseline Metrics -->
        <div class="card">
            <h2>Baseline Performance</h2>
            <div class="metrics-row">
                <div class="metric"><div class="metric-val" style="color:var(--green)">0.907</div><div class="metric-label">Avg Composite</div></div>
                <div class="metric"><div class="metric-val" style="color:var(--accent2)">23</div><div class="metric-label">Steps/Episode</div></div>
                <div class="metric"><div class="metric-val" style="color:var(--amber)">5</div><div class="metric-label">Tickets</div></div>
            </div>
            <div style="margin-top:16px;font-size:0.85em;color:var(--muted);text-align:center">
                Heuristic agent | 3 episodes | Seeds 42-44<br>
                KB grew from 2 &rarr; 11 entries across episodes
            </div>
        </div>
        <!-- KB Status -->
        <div class="card">
            <h2>Knowledge Base</h2>
            <div class="kb-stat"><div class="kb-icon" style="background:rgba(99,102,241,0.15)">&#128218;</div><div class="kb-info"><strong>Seed Articles</strong><br><small>Password Reset, Software Install</small></div><span style="font-weight:700">2</span></div>
            <div class="kb-stat"><div class="kb-icon" style="background:rgba(16,185,129,0.15)">&#129302;</div><div class="kb-info"><strong id="kb-agent-label">Agent-Created</strong><br><small>Written by L3 during episodes</small></div><span style="font-weight:700" id="kb-agent-count">0</span></div>
            <div class="kb-stat"><div class="kb-icon" style="background:rgba(245,158,11,0.15)">&#128200;</div><div class="kb-info"><strong>Total Size</strong><br><small>Persists across episodes</small></div><span style="font-weight:700" id="kb-total">2</span></div>
        </div>
        <!-- Architecture -->
        <div class="card">
            <h2>How It Works</h2>
            <div style="font-size:0.88em;line-height:1.7;color:var(--muted)">
                <p><strong style="color:var(--text)">1. Ticket arrives</strong> &mdash; employee reports an IT issue</p>
                <p><strong style="color:var(--text)">2. Triage classifies</strong> &mdash; category, priority, support tier</p>
                <p><strong style="color:var(--text)">3. Support agent resolves</strong> &mdash; search KB, apply fix, respond</p>
                <p><strong style="color:var(--text)">4. L3 writes KB article</strong> &mdash; novel solutions documented</p>
                <p><strong style="color:var(--text)">5. Next episode benefits</strong> &mdash; KB grows, agents improve</p>
            </div>
            <div style="margin-top:16px;padding:12px;background:var(--bg);border-radius:8px;font-size:0.8em;color:var(--green);text-align:center">
                Self-improvement: KB persists across episodes
            </div>
        </div>
    </div>
    <div class="links-row">
        <a href="/docs">API Docs</a>
        <a href="/health">Health</a>
        <a href="/kb">KB Stats</a>
        <a href="/metadata">Metadata</a>
        <a href="https://github.com/HR5206/emailenv">GitHub</a>
    </div>
</div>
<script>
const DEMO_STEPS = [
    {agent:"triage",node:"node-triage",msg:"Triage Agent classifying ticket...",scores:{triage:1.0},delay:1200},
    {agent:"triage",node:"node-triage",msg:"Category: password_reset | Priority: high | Tier: L1",scores:{},delay:800},
    {agent:"l1",node:"node-l1",msg:"L1 Agent searching Knowledge Base...",scores:{},delay:1000},
    {agent:"l1",node:"node-l1",msg:"Found KB article: 'Password Reset via Active Directory'",scores:{},delay:900},
    {agent:"l1",node:"node-l1",msg:"Applying solution from KB article...",scores:{resolution:1.0},delay:1100},
    {agent:"l1",node:"node-l1",msg:"Sending professional response to customer...",scores:{response:1.0,efficiency:0.85},delay:1000},
    {agent:"done",node:"",msg:"Ticket RESOLVED! Computing composite reward...",scores:{kb:0.0},delay:600},
];
let demoRunning = false;
function addLog(text, agent) {
    const log = document.getElementById("log");
    const ts = new Date().toLocaleTimeString();
    const agentTag = agent ? '<span class="ag">[' + agent.toUpperCase() + ']</span> ' : '';
    log.innerHTML += '<div class="log-entry"><span class="ts">[' + ts + ']</span> ' + agentTag + text + '</div>';
    log.scrollTop = log.scrollHeight;
}
function setBar(name, val) {
    document.getElementById("bar-"+name).style.width = (val*100)+"%";
    document.getElementById("val-"+name).textContent = val.toFixed(2);
}
function clearNodes() {
    document.querySelectorAll(".agent-node").forEach(n=>{n.classList.remove("active","done");});
}
async function runDemo() {
    if (demoRunning) return;
    demoRunning = true;
    document.getElementById("btn-demo").disabled = true;
    document.getElementById("log").innerHTML = "";
    clearNodes();
    ["triage","resolution","response","efficiency","kb"].forEach(n=>{ document.getElementById("bar-"+n).style.width="0%"; document.getElementById("val-"+n).textContent="--"; });
    document.getElementById("val-composite").textContent = "--";
    addLog("New ticket: 'Cannot log in -- password expired' from jsmith@company.com", "");
    let scores = {};
    for (const step of DEMO_STEPS) {
        await new Promise(r=>setTimeout(r, step.delay));
        clearNodes();
        if (step.node) {
            document.getElementById(step.node).classList.add("active");
        }
        addLog(step.msg, step.agent === "done" ? "" : step.agent);
        Object.assign(scores, step.scores);
        for (const [k,v] of Object.entries(scores)) { setBar(k, v); }
    }
    await new Promise(r=>setTimeout(r, 800));
    clearNodes();
    document.querySelectorAll(".agent-node").forEach(n=>n.classList.add("done"));
    const composite = (scores.triage||0)*0.15 + (scores.resolution||0)*0.3 + (scores.response||0)*0.2 + (scores.efficiency||0)*0.2 + (scores.kb||0)*0.15;
    document.getElementById("val-composite").textContent = composite.toFixed(4);
    document.getElementById("val-composite").style.color = composite > 0.8 ? "var(--green)" : composite > 0.5 ? "var(--amber)" : "var(--red)";
    addLog('<span class="rw">Composite reward: ' + composite.toFixed(4) + '</span>', "");
    // Simulate KB growth
    let agentCount = parseInt(document.getElementById("kb-agent-count").textContent);
    document.getElementById("kb-agent-count").textContent = agentCount + 1;
    document.getElementById("kb-total").textContent = 2 + agentCount + 1;
    demoRunning = false;
    document.getElementById("btn-demo").disabled = false;
}
</script>
</body>
</html>"""

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