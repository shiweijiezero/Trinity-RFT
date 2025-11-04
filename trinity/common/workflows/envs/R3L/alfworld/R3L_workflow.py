# -*- coding: utf-8 -*-
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from jinja2 import Environment, FileSystemLoader

from trinity.common.experience import Experience
from trinity.common.models.model import ModelWrapper
from trinity.common.workflows.envs.R3L.alfworld import utils
from trinity.common.workflows.workflow import WORKFLOWS, Task, Workflow


@WORKFLOWS.register_module("R3L_alfworld_workflow")
class R3LAlfworldWorkflow(Workflow):
    """
    R3L workflow for alfworld
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
        self.max_env_steps = 25
        self.max_tokens = 512
        self.max_reflect_tokens = 4096
        self.task = task
        self.is_eval = task.is_eval

        self.whether_save_data = False
        # Create data directories
        self.data_dir = f"R3L_alfworld_data"
        self.eval_dir = os.path.join(self.data_dir, "eval")
        self.train_dir = os.path.join(self.data_dir, "train")

        os.makedirs(self.eval_dir, exist_ok=True)
        os.makedirs(self.train_dir, exist_ok=True)

        # Initialize Jinja2 templates
        prompts_dir = Path(__file__).parent / "prompts"
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(prompts_dir)),
            trim_blocks=True,
            lstrip_blocks=True,
        )

        # Cache templates to avoid repeated loading
        self.alfworld_system_template = self.jinja_env.get_template("alfworld_system.j2")
        self.reflection_template = self.jinja_env.get_template("reflection.j2")

        self.default_exp = Experience(
            tokens=torch.tensor([0, 0], dtype=torch.long),
            prompt_length=1,
            action_mask=torch.tensor([False], dtype=torch.bool),
            logprobs=torch.tensor([0.0], dtype=torch.float),
            metrics={
                "success": 0.0,
                "reward": 0.0,
            },
            reward=0.0,
        )

        self.default_second_exp = Experience(
            tokens=torch.tensor([0, 0], dtype=torch.long),
            prompt_length=1,
            action_mask=torch.tensor([False], dtype=torch.bool),
            logprobs=torch.tensor([0.0], dtype=torch.float),
            metrics={
                "second_success": 0.0,
                "second_reward": 0.0,
            },
            reward=0.0,
        )

        print(
            f"Initializing R3LAlfworldWorkflow with experience learning, temperature={self.temperature}"
        )
        self.reset(task)

    def reset(self, task: Task):
        """Reset the workflow with a new task"""
        self.game_file_path = task.task_desc or task.raw_task.get("game_file", "")
        self.is_eval = task.is_eval
        self.temperature = getattr(task.rollout_args, "temperature", 1.0)
        self.task = task
        self.n = task.repeat_times

    def get_reflect(
        self, trajectory: List[Dict[str, str]]
    ) -> tuple[Optional[Dict[str, Any]], Optional[str], Optional[Any]]:
        """
        Generates a comprehensive reflection report using a single, unified self-interrogation prompt.
        The model first assesses its own performance and then follows the appropriate reflection path.
        """
        # Format trajectory for LLM reading
        formatted_trajectory = utils.format_trajectory_for_reflection(trajectory)

        # Use Jinja2 template to render reflection prompt
        reflect_prompt = self.reflection_template.render()

        # Call model and parse results
        try:
            responses = self.model.chat(
                [
                    {"role": "system", "content": reflect_prompt},
                    {
                        "role": "user",
                        "content": "Here is last attempt trajectory log: \n\n"
                        + formatted_trajectory
                        + "\n\nPlease output in the specified JSON format.",
                    },
                ],
                n=1,
                temperature=self.temperature,
                max_tokens=self.max_reflect_tokens,
            )
            reflection_text = responses[0].response_text.strip()

            # Find first '{' and last '}'
            first_brace = reflection_text.find("{")
            last_brace = reflection_text.rfind("}")

            if first_brace != -1 and last_brace != -1 and first_brace < last_brace:
                json_str = reflection_text[first_brace : last_brace + 1]
            else:
                json_str = reflection_text

            reflection_data = json.loads(json_str)
            return reflection_data, reflection_text, responses[0]

        except Exception as e:
            return None, None, None

    def _adjust_action_mask_for_retry(self, experience: Experience, retry_step: int):
        """
        Adjust action_mask in-place to exclude retry prefix from training.
        Only tokens from retry_step onwards should be trained.

        Args:
            experience: The experience object with action_mask to adjust
            retry_step: The step from which training should start
        """
        if retry_step <= 0:
            return

        # Note: experience.action_mask already excludes prompt tokens
        action_mask = experience.action_mask

        # Find all assistant response regions and mark the first 'retry_step' as non-trainable
        if torch.any(action_mask == 1):
            # Find all segments where action_mask == 1 (assistant responses)
            assistant_segments = []
            in_segment = False
            segment_start = 0

            for i, mask_val in enumerate(action_mask):
                if mask_val == 1 and not in_segment:
                    # Start of a new segment
                    segment_start = i
                    in_segment = True
                elif mask_val == 0 and in_segment:
                    # End of current segment
                    assistant_segments.append((segment_start, i))
                    in_segment = False

            # Handle case where sequence ends with assistant response
            if in_segment:
                assistant_segments.append((segment_start, len(action_mask)))

            # Set the first 'retry_step' assistant segments to 0 (non-trainable)
            for i in range(min(retry_step, len(assistant_segments))):
                start, end = assistant_segments[i]
                action_mask[start:end] = 0

    def run(self) -> List[Experience]:
        """Run the R3L alfworld workflow and return experiences"""

        if self.is_eval:
            return utils.eval_alfworld(self)

        # Generate unique task ID
        task_id = f"{str(self.task.batch_id).replace('/', '_')}_{self.task.task_id}"

        env = utils.create_alfworld_environment(self.game_file_path)
        exp_lst = []
        for i in range(self.n // 2):  # Half for rollout, half for reflection + retry
            try:
                trajectory, reward, done, steps, format_valid = utils.first_rollout(self, env)
                print(f"[R3L] First rollout - reward: {reward}, steps: {steps}")
                exp = self.model.convert_messages_to_experience(trajectory[:-1])
                exp.reward = reward
                exp.metrics = {
                    "success": 1.0 if reward >= 1.0 else 0.0,
                    "steps": steps,
                    "reward": reward,
                }
                # Set eid
                exp.eid.task = str(self.task.task_id) + f"_explore"
                exp_run_id = len(exp_lst) + self.run_id_base
                exp.eid.run = exp_run_id
                exp_lst.append(exp)

                if self.whether_save_data:
                    # Save first attempt experience data
                    first_record = utils.create_experience_record(
                        task_id=task_id,
                        trajectory=trajectory,
                        reward=reward,
                        steps=steps,
                        success=reward >= 1.0,
                        attempt_type="first",
                    )
                    utils.save_experience_data(
                        task_id=f"{task_id}_attempt_{i}_first",
                        experience_data=first_record,
                        data_dir=self.train_dir,
                    )

                # Reflect on first attempt
                reflect_checklist, reflection_text, reflect_exp = self.get_reflect(trajectory)
                is_valid, is_perfect = utils.validate_reflect_report(reflect_checklist, steps)

                if not is_valid or is_perfect:
                    # If first attempt reward is 1.0 and reflection gives perfect, record reflection exp
                    if reward >= 1.0 and is_perfect and reflect_exp is not None:
                        reflect_exp.reward = 1.0
                        # Set eid
                        reflect_exp.eid.task = str(self.task.task_id) + f"_reflect_{i}"
                        reflect_exp.eid.run = len(exp_lst) + self.run_id_base
                        exp_lst.append(reflect_exp)

                    # Do another rollout to ensure the batch has enough data
                    try:
                        retry_env = utils.create_alfworld_environment(self.game_file_path)
                        (
                            retry_trajectory,
                            retry_reward,
                            retry_done,
                            retry_steps,
                            retry_format_valid,
                        ) = utils.first_rollout(self, retry_env)

                        retry_exp = self.model.convert_messages_to_experience(retry_trajectory[:-1])
                        retry_exp.reward = retry_reward
                        retry_exp.metrics = {
                            "success": 1.0 if retry_reward >= 1.0 else 0.0,
                            "steps": retry_steps,
                            "reward": retry_reward,
                        }
                        # Set eid
                        retry_exp.eid.task = str(self.task.task_id) + f"_explore"
                        retry_exp.eid.run = len(exp_lst) + self.run_id_base
                        exp_lst.append(retry_exp)

                        if self.whether_save_data:
                            # Save retry attempt experience data
                            retry_record = utils.create_experience_record(
                                task_id=task_id,
                                trajectory=retry_trajectory,
                                reward=retry_reward,
                                steps=retry_steps,
                                success=retry_reward >= 1.0,
                                attempt_type="retry_after_invalid_reflection",
                            )
                            utils.save_experience_data(
                                task_id=f"{task_id}_attempt_{i}_retry",
                                experience_data=retry_record,
                                data_dir=self.train_dir,
                            )
                    except Exception as e:
                        print(f"Retry rollout after invalid reflection failed: {e}")

                else:
                    guidance_prompt = utils.reflect_report_to_guidance_prompt(reflect_checklist)
                    # Extract retry_step from validated reflection report
                    retry_step = reflect_checklist["analysis"]["retry_strategy"]["retry_step"]

                    try:
                        second_env = utils.create_alfworld_environment(self.game_file_path)
                        (
                            distill_trajectory,
                            second_trajectory,
                            second_reward,
                            second_done,
                            second_steps,
                            second_format_valid,
                        ) = utils.second_rollout(
                            self, second_env, guidance_prompt, trajectory, retry_step
                        )
                        print(
                            f"[R3L] Second rollout - reward: {second_reward}, steps: {second_steps}, improve: {second_reward > reward}"
                        )
                        second_exp = self.model.convert_messages_to_experience(
                            distill_trajectory[:-1]
                        )

                        # Adjust action_mask to exclude retry prefix from training
                        if retry_step > 0:
                            self._adjust_action_mask_for_retry(second_exp, retry_step)
                            # Also adjust first rollout exp for fair comparison
                            for existing_exp in exp_lst:
                                if existing_exp.eid.run == exp_run_id:
                                    self._adjust_action_mask_for_retry(existing_exp, retry_step)
                                    break

                        second_exp.reward = second_reward
                        second_exp.metrics = {
                            "second_success": 1.0 if second_reward >= 1.0 else 0.0,
                            "second_steps": second_steps,
                            "second_reward": second_reward,
                            "second_improve": 1.0 if second_reward > reward else 0.0,
                            "second_reward_diff": second_reward - reward,
                        }
                        # Set eid
                        second_exp.eid.task = str(self.task.task_id) + f"_explore"
                        second_exp.eid.run = len(exp_lst) + self.run_id_base
                        exp_lst.append(second_exp)

                        if self.whether_save_data:
                            # Save second attempt experience data
                            second_record = utils.create_experience_record(
                                task_id=task_id,
                                trajectory=second_trajectory,
                                reward=second_reward,
                                steps=second_steps,
                                success=second_reward >= 1.0,
                                attempt_type="second",
                                additional_metrics={
                                    "first_reward": reward,
                                    "improvement": second_reward > reward,
                                    "reward_difference": second_reward - reward,
                                    "step_difference": second_steps - steps,
                                },
                            )
                            utils.save_experience_data(
                                task_id=f"{task_id}_attempt_{i}_second",
                                experience_data=second_record,
                                data_dir=self.train_dir,
                            )

                        # If second attempt score is higher than first, or second is perfect with fewer steps,
                        # record reflection and retry data
                        if (second_reward > reward and second_reward >= 1.0) or (
                            second_reward >= 1.0 and second_steps < steps
                        ):
                            reflect_exp.reward = 1.0
                            # Set eid
                            reflect_exp.eid.task = str(self.task.task_id) + f"_reflect_{i}"
                            reflect_exp.eid.run = len(exp_lst) + self.run_id_base
                            exp_lst.append(reflect_exp)

                            # Convert retry data to exp
                            retry_exp = self.model.convert_messages_to_experience(
                                second_trajectory[:-1]
                            )

                            # Adjust action_mask to exclude retry prefix from training
                            if retry_step > 0:
                                self._adjust_action_mask_for_retry(retry_exp, retry_step)

                            retry_exp.reward = 1.0
                            # Set eid
                            retry_exp.eid.task = str(self.task.task_id) + f"_retry_{i}"
                            retry_exp.eid.run = len(exp_lst) + self.run_id_base
                            exp_lst.append(retry_exp)

                            print("Reflection and retry led to improvement, recording both...")
                    except Exception:
                        pass
            except Exception:
                pass

        return exp_lst

    def resettable(self) -> bool:
        """Indicate that this workflow can be reset to avoid re-initialization"""
        return True

    def set_repeat_times(self, repeat_times, run_id_base):
        self.repeat_times = repeat_times
        self.run_id_base = run_id_base
