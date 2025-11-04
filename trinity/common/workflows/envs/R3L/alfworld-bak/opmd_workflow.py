# -*- coding: utf-8 -*-
from pathlib import Path
from typing import List, Optional

from jinja2 import Environment, FileSystemLoader

from trinity.common.experience import Experience
from trinity.common.models.model import ModelWrapper
from trinity.common.workflows.envs.R3L.alfworld import utils
from trinity.common.workflows.workflow import WORKFLOWS, Task, Workflow


@WORKFLOWS.register_module("opmd_baseline_alfworld_workflow")
class OPMDBaselineAlfworldWorkflow(Workflow):
    """
    OPMD Baseline workflow for Alfworld environment.
    Performs rollouts for offline policy model distillation.
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

        # Initialize Jinja2 templates
        prompts_dir = Path(__file__).parent / "prompts"
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(prompts_dir)),
            trim_blocks=True,
            lstrip_blocks=True,
        )

        # Cache templates to avoid repeated loading
        self.alfworld_system_template = self.jinja_env.get_template("alfworld_system.j2")

        print(f"Initializing OPMDAlfworldWorkflow, temperature={self.temperature}")
        self.reset(task)

    def reset(self, task: Task):
        """Reset the workflow with a new task"""
        self.game_file_path = task.task_desc or task.raw_task.get("game_file", "")
        self.is_eval = task.is_eval
        self.task = task
        self.n = task.repeat_times
        self.temperature = getattr(task.rollout_args, "temperature", 1.0)

    def run(self) -> List[Experience]:
        """Run the OPMD workflow and return experiences"""

        if self.is_eval:
            return utils.eval_alfworld(self)

        # Single rollout execution
        env = utils.create_alfworld_environment(self.game_file_path, self.max_env_steps)
        exp_lst = []
        for i in range(self.n):
            try:
                trajectory, reward, done, steps, format_valid = utils.first_rollout(self, env)
                print(f"[OPMD] First rollout - reward: {reward}, steps: {steps}")
                exp = self.model.convert_messages_to_experience(trajectory[:-1])
                exp.reward = reward
                exp.metrics = {
                    "success": 1.0 if reward >= 1.0 else 0.0,
                    "steps": steps,
                    "reward": reward,
                }
                exp_lst.append(exp)
            except Exception:
                pass

        return exp_lst

    def resettable(self) -> bool:
        """Indicate that this workflow can be reset to avoid re-initialization"""
        return True

    def set_repeat_times(self, repeat_times, run_id_base):
        self.repeat_times = repeat_times
        self.run_id_base = run_id_base
