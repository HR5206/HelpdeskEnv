"""Client utilities for interacting with the EmailEnv environment.

This lightweight module is provided to match the expected OpenEnv
environment structure (which looks for a `client.py` alongside
`openenv.yaml`).  For most usages within this project, you can import
and use the underlying primitives directly from the `emailenv` package.

Example:
    from emailenv import EmailEnv, Action

    env = EmailEnv()
    obs = env.reset(task="spam_classification")
    ...
"""

from emailenv import EmailEnv, Email, Observation, Action, State, Reward

__all__ = [
    "EmailEnv",
    "Email",
    "Observation",
    "Action",
    "State",
    "Reward",
]
