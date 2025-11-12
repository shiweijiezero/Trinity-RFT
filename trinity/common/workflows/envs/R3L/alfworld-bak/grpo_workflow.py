# -*- coding: utf-8 -*-
from pathlib import Path
from typing import List, Optional

import torch
from jinja2 import Environment, FileSystemLoader

from trinity.common.experience import Experience
from trinity.common.models.model import ModelWrapper
from trinity.common.workflows.envs.R3L.alfworld import utils
from trinity.common.workflows.workflow import WORKFLOWS, Task, Workflow


@WORKFLOWS.register_module("grpo_baseline_alfworld_workflow")
class GRPOBaselineAlfworldWorkflow(Workflow):
    """
    GRPO Baseline Workflow for Alfworld environment.
    Performs simple rollouts without reflection or learning from experience.
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
        self.max_env_steps = 50
        self.max_tokens = 512
        self.max_reflect_tokens = 4096
        self.task = task
        self.is_eval = task.is_eval
        self.whether_save_data = False

        # History length for sliding window (verl-agent uses 2)
        self.history_length = 2

        # Initialize Jinja2 templates
        prompts_dir = Path(__file__).parent / "prompts"
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(prompts_dir)),
            trim_blocks=True,
            lstrip_blocks=True,
        )

        # Cache templates to avoid repeated loading
        self.alfworld_system_template = self.jinja_env.get_template("alfworld_system.j2")

        print(
            f"Initializing GRPOBaselineAlfworldWorkflow, temperature={self.temperature}, history_length={self.history_length}"
        )
        self.reset(task)

    def reset(self, task: Task):
        """Reset the workflow with a new task"""
        self.game_file_path = task.task_desc or task.raw_task.get("game_file", "")
        self.is_eval = task.is_eval
        self.task = task
        self.n = task.repeat_times
        self.temperature = getattr(task.rollout_args, "temperature", 1.0)

    def run(self) -> List[Experience]:
        """Run the GRPO baseline workflow and return experiences"""

        if self.is_eval:
            return utils.eval_alfworld(self)

        # Single rollout execution
        env = utils.create_alfworld_environment(self.game_file_path, self.max_env_steps)
        exp_lst = []
        for i in range(self.n):
            try:
                trajectory, reward, done, steps, format_valid = utils.first_rollout(
                    self, env
                )
                # print(f"trajectory: {trajectory}")
                print(f"[GRPO] First rollout - reward: {reward}, steps: {steps}")
                exp = self.model.convert_messages_to_experience(trajectory[:-1])
                exp.reward = reward
                exp.metrics = {
                    "success": 1.0 if reward >= 1.0 else 0.0,
                    "steps": steps,
                    "reward": reward,
                }
                exp_lst.append(exp)
            except Exception as e:
                print(f"[GRPO] Rollout {i} failed with exception: {e}")
                # exp = Experience(
                #     tokens=torch.tensor([0, 0], dtype=torch.long),
                #     prompt_length=1,
                #     action_mask=torch.tensor([False], dtype=torch.bool),
                #     logprobs=torch.tensor([0.0], dtype=torch.float),
                #     metrics={
                #         "success": 0.0,
                #         "reward": 0.0,
                #     }
                # )
                # exp.reward = 0.0
        return exp_lst

    def resettable(self) -> bool:
        """Indicate that this workflow can be reset to avoid re-initialization"""
        return True

    def set_repeat_times(self, repeat_times, run_id_base):
        self.repeat_times = repeat_times
        self.run_id_base = run_id_base
