# -*- coding: utf-8 -*-
import copy
from pathlib import Path
from typing import List, Optional

import torch
from jinja2 import Environment, FileSystemLoader

from trinity.common.experience import Experience
from trinity.common.models.model import ModelWrapper
from trinity.common.workflows.envs.R3L.scienceworld import utils
from trinity.common.workflows.workflow import WORKFLOWS, Task, Workflow


@WORKFLOWS.register_module("raft_baseline_scienceworld_workflow")
class RAFTBaselineScienceWorldWorkflow(Workflow):
    """
    RAFT Baseline workflow for ScienceWorld environment.
    Performs rollouts for Reinforcement Learning from AI Feedback Training.
    """

    def __init__(
            self,
            model: ModelWrapper,
            task: Task,
            auxiliary_models: Optional[List] = None,
    ):
        super().__init__(
            model=model,
            task=task,
            auxiliary_models=auxiliary_models,
        )
        # Initialize workflow parameters
        self.temperature = getattr(task.rollout_args, "temperature", 1.0)
        self.max_env_steps = 30
        self.max_tokens = 16384
        self.task = task
        self.is_eval = task.is_eval
        self.whether_save_data = False

        # Initialize Jinja2 templates
        prompts_dir = Path(__file__).parent / "prompts"
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(prompts_dir)),
            trim_blocks=True,
            lstrip_blocks=True,
        )

        # Cache templates to avoid repeated loading
        self.sciworld_system_template = self.jinja_env.get_template("sciworld_system.j2")

        print(
            f"Initializing RAFTScienceWorldWorkflow, temperature={self.temperature}"
        )
        self.reset(task)

        # Default experience for error cases
        self.default_exp = Experience(
            tokens=torch.tensor([0, 0], dtype=torch.long),
            prompt_length=1,
            action_mask=torch.tensor([False], dtype=torch.bool),
            logprobs=torch.tensor([0.0], dtype=torch.float),
            metrics={
                "success": 0.0,
                "reward": 0.0,
            },
            reward=0.0
        )

    def reset(self, task: Task):
        """Reset the workflow with a new task"""
        self.task_desc = task.task_desc or "0"
        self.is_eval = task.is_eval
        self.task = task
        self.n = task.repeat_times

    def run(self) -> List[Experience]:
        """Run the RAFT workflow and return experiences"""

        env = utils.create_sciworld_environment(self.task_desc)

        if self.is_eval:
            try:
                trajectory, reward, done, steps, format_valid = utils.first_rollout(
                    self, env
                )
                exp = self.model.convert_messages_to_experience(trajectory[:-1])
                exp.reward = reward
                exp.metrics = {
                    "success": 1.0 if reward >= 1.0 else 0.0,
                    "steps": steps,
                    "reward": reward,
                }
                return [exp]
            except Exception:
                return [copy.deepcopy(self.default_exp)]

        # Single rollout execution
        exp_lst = []
        for i in range(self.n):
            try:
                trajectory, reward, done, steps, format_valid = utils.first_rollout(
                    self, env
                )
                exp = self.model.convert_messages_to_experience(trajectory[:-1])
                exp.reward = reward
                exp.metrics = {
                    "success": 1.0 if reward >= 1.0 else 0.0,
                    "steps": steps,
                    "reward": reward,
                }
            except Exception:
                exp = copy.deepcopy(self.default_exp)
            exp_lst.append(exp)

        return exp_lst

    def resettable(self) -> bool:
        """Indicate that this workflow can be reset to avoid re-initialization"""
        return True

    def set_repeat_times(self, repeat_times, run_id_base):
        self.repeat_times = repeat_times
        self.run_id_base = run_id_base
