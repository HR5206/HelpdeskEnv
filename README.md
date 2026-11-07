# HelpdeskEnv

I use this hackathon project to explore ticket routing through triage and three support tiers. The environment scores actions and keeps accepted knowledge-base articles for later episodes in the same process. It is a simulated IT support workflow with synthetic ticket fixtures.

## Run without model credentials

Use Python 3.12. From the repository root:

```sh
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell instead: .venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
python test_integration.py
python -m unittest test_regressions -v
# Optional dashboard escaping check (Node.js required):
node test_dashboard.mjs
python -m uvicorn server.app:app --host 127.0.0.1 --port 7860
```

Open `http://127.0.0.1:7860/web` for the dashboard or `/docs` for the API. Leave `HF_TOKEN`, `API_KEY`, and `OPENAI_API_KEY` unset to use the heuristic path. Supplying credentials can enable model calls and associated costs in the inference runner.

## What I am checking

A ticket is triaged, sent to L1/L2/L3, and resolved or escalated. Graders score triage, resolution, response quality, step-based efficiency, and knowledge-base contributions. SLA is measured in action steps, not elapsed service time. Accepted L3 articles remain searchable across resets until the process restarts.

The integration script exercises 43 heuristic checks. Additional regression tests cover invalid episode sizes, deterministic sampling without modifying global randomness, API input validation, and missing training measurements. The dashboard escapes ticket, KB, and log text before inserting it into HTML. These tests validate the fixture workflow, not trained-model performance.

## Evidence and limitations

I do not claim a measured trained-model uplift in this version. The checked-in training log ends at step 50; the older submission described 150 steps and +5.3%. That comparison needs a saved evaluation run, matching baseline, model revision, dataset split, and raw results before I can use it as evidence.

`generate_plots.py` now plots recorded rewards and KB sizes only. It rejects missing rewards, adds no synthetic confidence bands, and does not invent trained scores. Historical submission text is retained with an explicit evidence notice; old comparison images are removed.

The API owns one shared in-memory environment. It has no per-user sessions, authentication, durable KB storage, or production ticket integration. Fixtures expose expected answers, and heuristic keyword graders can be gamed. Repeated support actions can accumulate rewards; this is not a robust adversarial RL benchmark. The OpenEnv-style endpoints have not been certified against the current OpenEnv validator. Do not describe the `/mcp` metadata stub as a complete MCP server.

## Deployment

The repository links a [Hugging Face Space](https://huggingface.co/spaces/Harishraghav-05/helpdesk_env); its page was reachable during review, but that does not verify the running application. Check `/health`, `/web`, then reset and complete a fixture episode yourself.

For a recruiter, a short recording of one episode is useful and reproducible. Keep hosted demos restricted to synthetic fixtures. Use the existing Dockerfile on port 7860, one worker, and no model keys for the heuristic demo. Review isolation, quotas, and authentication before allowing untrusted public writes; there is no need to buy GPU hosting for this demonstration.

## Start reading

`helpdeskenv_class.py` owns transitions, `graders.py` scores actions, `knowledge_base.py` searches accepted articles, and `heuristics.py` supplies deterministic actions. `server/app.py` exposes the dashboard/API. The training notebook is an experiment, separate from the verified local path.
