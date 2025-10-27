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
from trinity.common.workflows.envs.R3L.dapo import utils
from trinity.common.workflows.workflow import WORKFLOWS, Task, Workflow


@WORKFLOWS.register_module("R3L_dapo_workflow")
class R3LDapoWorkflow(Workflow):
    """
    R3L workflow for DAPO mathematical problem solving
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
        self.max_attempts = 3
        self.max_tokens = 4096
        self.task = task
        self.is_eval = task.is_eval

        self.whether_save_data = False
        # Create data directories
        self.data_dir = f"R3L_dapo_data"
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
        self.dapo_system_template = self.jinja_env.get_template("math_system.j2")
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
            reward=0.0
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
            reward=0.0
        )

        print(
            f"Initializing R3LDapoWorkflow with experience learning, temperature={self.temperature}"
        )
        self.reset(task)

    def reset(self, task: Task):
        """Reset the workflow with a new task"""
        self.is_eval = task.is_eval
        self.task = task
        self.n = task.repeat_times

        # Extract prompt and ground truth from task
        if hasattr(task, 'raw_task') and task.raw_task:
            raw_task = task.raw_task

            # Format 1: prompt is a list (math_dapo format)
            if "prompt" in raw_task and isinstance(raw_task["prompt"], list):
                if len(raw_task["prompt"]) > 0 and isinstance(raw_task["prompt"][0], dict):
                    self.prompt = raw_task["prompt"][0].get("content", "")
                else:
                    self.prompt = ""

                reward_model_data = raw_task.get("reward_model", {})
                if isinstance(reward_model_data, dict):
                    self.ground_truth = reward_model_data.get("ground_truth", "")
                else:
                    self.ground_truth = ""

            # Format 2: question/answer format (AIME format)
            elif "question" in raw_task and "answer" in raw_task:
                self.prompt = raw_task.get("question", "")
                self.ground_truth = raw_task.get("answer", "")

            # Fallback: simple prompt/answer
            else:
                self.prompt = raw_task.get("prompt", "")
                self.ground_truth = raw_task.get("answer", "")
        else:
            self.prompt = ""
            self.ground_truth = ""

    def get_reflect(self, trajectory: List[Dict[str, str]]) -> tuple[
        Optional[Dict[str, Any]], Optional[str], Optional[Any]]:
        """
        Generates a comprehensive reflection report using a single, unified self-interrogation prompt.
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
                    {"role": "user", "content": "Here is last attempt trajectory log: \n\n" + formatted_trajectory}
                ],
                n=1,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            reflection_text = responses[0].response_text.strip()

            # Parse JSON
            json_match = re.search(r"```json\s*(\{.*?\})\s*```", reflection_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = reflection_text

            reflection_data = json.loads(json_str)
            return reflection_data, reflection_text, responses[0]

        except Exception as e:
            return None, None, None

    def run(self) -> List[Experience]:
        """Run the R3L dapo workflow and return experiences"""

        if self.is_eval:
            return utils.eval_dapo(self)

        # Generate unique task ID
        task_id = f"{str(self.task.batch_id).replace('/', '_')}_{self.task.task_id}"

        exp_lst = []
        for i in range(self.n // 2):  # Half for rollout, half for reflection + retry
            try:
                trajectory, reward, success, predicted_answer, ground_truth, attempts = utils.first_rollout(self)
                print(f"[R3L] First rollout - reward: {reward}, attempts: {attempts}")
                exp = self.model.convert_messages_to_experience(trajectory[:-1])
                exp.reward = reward
                exp.metrics = {
                    "success": 1.0 if success else 0.0,
                    "reward": reward,
                    "attempts": attempts,
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
                        success=success,
                        predicted_answer=predicted_answer,
                        ground_truth=ground_truth,
                        attempt_type="first"
                    )
                    utils.save_experience_data(
                        task_id=f"{task_id}_attempt_{i}_first",
                        experience_data=first_record,
                        data_dir=self.train_dir
                    )

                # Reflect on first attempt
                reflect_checklist, reflection_text, reflect_exp = self.get_reflect(trajectory)
                is_valid, is_perfect = utils.validate_reflect_report(reflect_checklist, attempts)

                if not is_valid or is_perfect:
                    print(f"Skip second rollout due to invalid ({not is_valid}) or perfect ({is_perfect}) reflection.")
                    # If first attempt reward is 1.0 and reflection gives perfect, record reflection exp
                    if reward >= 1.0 and is_perfect and reflect_exp is not None:
                        reflect_exp.reward = 1.0
                        # Set eid
                        reflect_exp.eid.task = str(self.task.task_id) + f"_reflect_{i}"
                        reflect_exp.eid.run = len(exp_lst) + self.run_id_base
                        exp_lst.append(reflect_exp)

                    if not is_valid:
                        # Do another rollout to ensure the batch has enough data
                        try:
                            retry_trajectory, retry_reward, retry_success, retry_predicted_answer, retry_ground_truth, retry_attempts = utils.first_rollout(self)

                            retry_exp = self.model.convert_messages_to_experience(retry_trajectory[:-1])
                            retry_exp.reward = retry_reward
                            retry_exp.metrics = {
                                "success": 1.0 if retry_success else 0.0,
                                "reward": retry_reward,
                                "attempts": retry_attempts,
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
                                    success=retry_success,
                                    predicted_answer=retry_predicted_answer,
                                    ground_truth=retry_ground_truth,
                                    attempt_type="retry_after_invalid_reflection"
                                )
                                utils.save_experience_data(
                                    task_id=f"{task_id}_attempt_{i}_retry",
                                    experience_data=retry_record,
                                    data_dir=self.train_dir
                                )
                        except Exception as e:
                            print(f"Retry rollout after invalid reflection failed: {e}")

                else:
                    print("[R3L] Valid reflection obtained, proceeding to second rollout...")
                    guidance_prompt = utils.reflect_report_to_guidance_prompt(reflect_checklist)
                    # Extract retry_step from validated reflection report
                    retry_step = reflect_checklist["analysis"]["retry_strategy"]["retry_step"] if "retry_strategy" in reflect_checklist.get("analysis", {}) else 0

                    try:
                        (
                            distill_trajectory,
                            second_trajectory,
                            second_reward,
                            second_success,
                            second_predicted_answer,
                            second_ground_truth,
                            second_attempts,
                        ) = utils.second_rollout(
                            self, guidance_prompt, trajectory, retry_step
                        )
                        print(f"[R3L] Second rollout - reward: {second_reward}, attempts: {second_attempts}, improve: {second_reward > reward}")
                        second_exp = self.model.convert_messages_to_experience(distill_trajectory[:-1])

                        second_exp.reward = second_reward
                        second_exp.metrics = {
                            "second_success": 1.0 if second_success else 0.0,
                            "second_reward": second_reward,
                            "second_attempts": second_attempts,
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
                                success=second_success,
                                predicted_answer=second_predicted_answer,
                                ground_truth=second_ground_truth,
                                attempt_type="second",
                                additional_metrics={
                                    "first_reward": reward,
                                    "improvement": second_reward > reward,
                                    "reward_difference": second_reward - reward,
                                }
                            )
                            utils.save_experience_data(
                                task_id=f"{task_id}_attempt_{i}_second",
                                experience_data=second_record,
                                data_dir=self.train_dir
                            )

                        # If second attempt score is higher than first, record reflection and retry data
                        if second_reward > reward and second_reward >= 1.0:
                            reflect_exp.reward = 1.0
                            # Set eid
                            reflect_exp.eid.task = str(self.task.task_id) + f"_reflect_{i}"
                            reflect_exp.eid.run = len(exp_lst) + self.run_id_base
                            exp_lst.append(reflect_exp)

                            # Convert retry data to exp
                            retry_exp = self.model.convert_messages_to_experience(second_trajectory[:-1])

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
