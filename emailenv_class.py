"""EmailEnv - An OpenEnv-compliant email workflow environment.

Simulates 3 sequential email tasks per episode:
    1. Spam Classification (easy)
    2. Email Prioritization (medium)
    3. Reply Drafting (hard)

Core methods:
    reset() -> starts a new episode
    step() -> submits an action and receives a reward
    state() -> returns the current environment state
"""

import random
from models import (
    EmailTask,
    AgentAction,
    StepResult,
    EnvState,
    ResetResponse,
    ErrorResponse,
    TaskType
)

from tasks import (
    get_random_spam_scenario,
    get_random_priority_scenario,
    get_random_reply_scenario
)

from graders import (
    grade_spam,
    grade_priority,
    grade_reply
)


class EmailEnv:
    """
    EmailEnv - An OpenEnv-compliant workflow environment.

    Simulates 3 sequential email tasks per episode:
        1. Spam Classification (easy)
        2. Email Prioritization (medium)
        3. Reply Drafting (hard)

    Core methods:
        reset() -> starts a new episode
        step() -> submits an action and receives a reward
        state() -> returns the current environment state
    """

    def __init__(self):
        self._state = EnvState()
        self._tasks: list[EmailTask] = []
        self._seed: int = None

    def reset(self, seed: int = None) -> ResetResponse:
        """
        Resets the environment and begins a new episode.

        Args:
            seed: Optional integer for reproducible task selection

        Returns:
            ResetResponse with the first task and fresh state.
        """
        self._seed = seed

        self._tasks = [
            get_random_spam_scenario(seed=seed),
            get_random_priority_scenario(seed=seed),
            get_random_reply_scenario(seed=seed)
        ]

        self._state = EnvState(
            current_task=self._tasks[0],
            task_number=0,
            total_reward=0.0,
            is_done=False,
            history=[]
        )

        return ResetResponse(
            observation=self._tasks[0]
        )

    def step(self, action: AgentAction) -> StepResult | ErrorResponse:
        """
        Processes the agent's action for the current task

        Args:
            action: AgentAction with task_id and action_value.

        Returns:
            StepResult with reward, feedback, and done flag.
            ErrorResponse if something is wrong.
        """

        if not self._tasks:
            return ErrorResponse(
                error="Environment not initialized.",
                detail="Call reset() before step()."
            )

        if self._state.is_done:
            return ErrorResponse(
                error="Episode is already complete.",
                detail="Call reset() to start a new episode."
            )

        current_task = self._state.current_task
        if action.task_id != current_task.task_id:
            return ErrorResponse(
                error="Task ID mismatch.",
                detail=(
                    f"Expected task_id='{current_task.task_id}' "
                    f"but received '{action.task_id}'"
                )
            )

        task_type = current_task.task_type

        if task_type == TaskType.SPAM:
            result = grade_spam(current_task, action)
        elif task_type == TaskType.PRIORITY:
            result = grade_priority(current_task, action)
        elif task_type == TaskType.REPLY:
            result = grade_reply(current_task, action)
        else:
            return ErrorResponse(
                error="Unknown task type.",
                detail=f"Task type '{task_type}' is not recognized."
            )

        self._state.total_reward = round(
            self._state.total_reward + result.reward, 2
        )

        self._state.history.append(result)

        next_task_number = self._state.task_number + 1

        if next_task_number >= len(self._tasks):
            self._state.is_done = True
            self._state.current_task = None
            self._state.task_number = next_task_number
            result.done = True
            result.feedback = (
                result.feedback +
                f"\n\nEpisode complete! "
                f"Total reward: {self._state.total_reward:.2f} / 3.00"
            )
        else:
            self._state.task_number = next_task_number
            self._state.current_task = self._tasks[next_task_number]
            result.done = False

        return result

    def state(self) -> EnvState:
        """
        Returns the current state of the environment.

        Returns:
            EnvState with current task, reward, history, and done flag.
        """
        return self._state

    def next_task(self) -> EmailTask | None:
        """
        Returns the current task the agent needs to solve.
        Returns None if the episode is done or not started.
        """
        return self._state.current_task