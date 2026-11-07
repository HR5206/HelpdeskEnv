import random
import unittest
from helpdeskenv_class import HelpdeskEnv
from generate_plots import reward_points
from server.app import ResetRequest, StepRequest
from pydantic import ValidationError

class Regressions(unittest.TestCase):
    def test_empty_episodes_rejected(self):
        for count in (0, -1, True):
            with self.assertRaises(ValueError): HelpdeskEnv().reset(num_tickets=count)
        with self.assertRaises(ValueError): HelpdeskEnv().reset(custom_tickets=[])
    def test_random_seed_is_local(self):
        random.seed(123); state = random.getstate()
        first = HelpdeskEnv().reset(seed=42, num_tickets=2)
        self.assertEqual(state, random.getstate())
        self.assertEqual(first.observation, HelpdeskEnv().reset(seed=42, num_tickets=2).observation)
    def test_api_rejects_bad_inputs(self):
        with self.assertRaises(ValidationError): ResetRequest(num_tickets=0)
        with self.assertRaises(ValidationError): StepRequest(ticket_id="x", agent_role="invalid", action_type="x", action_value="x")
    def test_missing_metrics_never_become_scores(self):
        with self.assertRaises(ValueError): reward_points({})
        self.assertEqual(reward_points({"log_history":[{"step":1,"reward":0.25}]}), [(1,0.25)])

class ApiRegressions(unittest.IsolatedAsyncioTestCase):
    async def test_http_validation_and_dashboard(self):
        import httpx
        from server.app import app
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            self.assertEqual((await client.get("/health")).status_code, 200)
            self.assertEqual((await client.get("/web")).status_code, 200)
            self.assertEqual((await client.post("/reset", json={"num_tickets":0})).status_code, 422)
            self.assertEqual((await client.post("/reset", json={"seed":42,"num_tickets":1})).status_code, 200)
            self.assertEqual((await client.post("/step", json={"ticket_id":"x","agent_role":"bad","action_type":"x","action_value":"x"})).status_code, 422)

if __name__ == "__main__": unittest.main()
