# -*- coding: utf-8 -*-
import copy
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import ray
import torch
from jinja2 import Environment, FileSystemLoader

from trinity.common.experience import Experience
from trinity.common.models.model import ModelWrapper
from trinity.common.workflows.envs.opmd_reflect_enhanced_restart_webshop import opmd_reflect_enhanced_restart_utils
from trinity.common.workflows.envs.opmd_reflect_enhanced_restart_webshop.opmd_reflect_enhanced_restart_utils import format_observation
from trinity.common.workflows.workflow import WORKFLOWS, Task, Workflow


@WORKFLOWS.register_module("opmd_reflect_enhanced_restart_webshop_workflow")
class OPMDReflectEnhancedRestartWebshopWorkflow(Workflow):
    """
    Distill workflow for webshop using trajectory context.
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
        self.logprobs = 1  # 暂时设为1，后续可以改为5或者20
        self.top_k = getattr(task.rollout_args, "top_k", 20)
        self.top_p = getattr(task.rollout_args, "top_p", 0.95)
        self.max_env_steps = 15
        self.max_tokens = 4096
        self.task = task
        self.is_eval = task.is_eval

        # Create data directories
        self.data_dir = f"opmd_reflect_webshop_data"
        self.eval_dir = os.path.join(self.data_dir, "eval")
        self.sft_dir = os.path.join(self.data_dir, "sft_data")
        self.non_sft_dir = os.path.join(self.data_dir, "non_sft_data")

        os.makedirs(self.eval_dir, exist_ok=True)
        os.makedirs(self.sft_dir, exist_ok=True)
        os.makedirs(self.non_sft_dir, exist_ok=True)

        # Initialize WebShop environment
        try:
            import sys

            # sys.path.append("/nas/shiweijie/trinity/webshop")
            sys.path.append("/home/wshiah/code/shiweijie/weijie/trinity/webshop")
            # Try gymnasium first, fallback to gym
            import gym
            from web_agent_site.envs import WebAgentTextEnv  # noqa: F401

            # NOTE: Hosting the env require ~15GB CPU memory.
            # If you want easier env, you can set the num_products to 1000 or 100000.
            self.env = gym.make(
                "WebAgentTextEnv-v0",
                observation_mode="text_rich",
                num_products=None,
                human_goals=True,
            )
        except Exception as e:
            error_message = f"Error importing WebAgentTextEnv {str(e)}. Please make sure you have installed the web_agent_site package, following the instructions in https://github.com/princeton-nlp/WebShop"
            raise ImportError(error_message)

        # Initialize Jinja2 templates
        prompts_dir = Path(__file__).parent / "prompts"
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(prompts_dir)),
            trim_blocks=True,
            lstrip_blocks=True,
        )

        # Cache templates to avoid repeated loading
        self.webshop_system_template = self.jinja_env.get_template("webshop_system.j2")
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
            reward=-0.1
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
            reward=-0.1
        )

        print(
            f"Initializing ExpLearnWebshopWorkflow with experience learning, temperature={self.temperature}"
        )
        self.reset(task)

    def reset(self, task: Task):
        """Reset the workflow with a new task"""
        self.session_id = int(task.task_desc or "0")
        self.is_eval = task.is_eval
        self.task = task
        self.n = task.repeat_times

    def first_rollout(self, env, session_id) -> tuple[List[Dict[str, str]], float, bool, int, bool]:
        """Run a single rollout"""
        # print(f"About to reset env with session_id: {session_id}")
        env.reset(session=session_id)
        observation = env.observation
        trajectory = []
        action_history = []  # Track last 3 actions for repetition detection

        system_prompt = self.webshop_system_template.render()
        trajectory.append({"role": "system", "content": system_prompt})

        default_reward = -0.1
        reward = default_reward
        valid_format = True
        step = 0

        for step in range(self.max_env_steps):
            available_actions = env.get_available_actions()
            trajectory.append(
                {"role": "user", "content": format_observation(observation, available_actions)}
            )

            # Get model response with experience guidance
            responses = self.model.chat(
                trajectory,
                n=1,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
            )
            response_text = responses[0].response_text.strip()
            trajectory.append({"role": "assistant", "content": response_text})

            # Parse the three components for action execution
            think, action = opmd_reflect_enhanced_restart_utils.parse_response(response_text)
            if action is None:
                valid_format = False
                feedback = "Invalid response format, missing valid <think> or <action> tags, please ensure to follow the output format strictly: <think>...</think> <action>...</action>"
                trajectory.append({"role": "user", "content": f"Feedback: {feedback}"})
                # print(f"Terminating due to invalid response format: {response_text}")
                return trajectory, default_reward, False, step + 1, valid_format

            # Check for consecutive action repetition
            action_history.append(action)
            if len(action_history) > 2:
                action_history.pop(0)

            # If last 2 actions are the same, terminate with failure
            if len(action_history) >= 2 and all(
                    action == action_history[0] for action in action_history
            ) and "next" not in action.lower() and "prev" not in action.lower() and "search" not in action.lower():
                feedback = f"Repeated invalid action {action} multiple times, shopping task failed"
                trajectory.append({"role": "user", "content": f"Feedback: {feedback}"})
                # print(f"Terminating due to 5 consecutive identical actions: {action_text}")
                valid_format = False
                return trajectory, default_reward, False, step + 1, valid_format

            # Validate and execute action in environment
            action_valid, error_msg = opmd_reflect_enhanced_restart_utils.validate_action(action, available_actions)
            if action_valid:
                observation, reward, done, info = env.step(action)
            else:
                observation, reward, done = error_msg, default_reward, False

            if done:
                break

        # Generate timeout feedback
        if reward >= 1.0 and step + 1 < self.max_env_steps:
            feedback = f"Shopping task completed successfully (reward: {reward}/1.0), and satisfying the step limit ({step + 1}/{self.max_env_steps} steps)"
        elif reward >= 1.0 and step + 1 >= self.max_env_steps:
            feedback = (
                f"Shopping task completed successfully (reward: {reward}/1.0), but exceeded the step limit ({step + 1}/{self.max_env_steps} steps)"
            )
        elif reward < 1.0 and step + 1 < self.max_env_steps:
            feedback = (
                f"Shopping task not completed (reward: {reward}/1.0), but within the step limit ({step + 1}/{self.max_env_steps} steps). It may not satisfy the Attribute Matching, Option Matching, or Price Matching requirements, please you carefully check and ensure all requirements are satisfied."
            )
        else:
            feedback = (
                f"Shopping task not completed (reward: {reward}/1.0), and exceeded the step limit ({step + 1}/{self.max_env_steps} steps). It may not satisfy the Attribute Matching, Option Matching, or Price Matching requirements, please you carefully check and ensure all requirements are satisfied."
            )

        # Add timeout feedback to trajectory
        trajectory.append({"role": "user", "content": f"Feedback: {feedback}"})
        return trajectory, reward, False, step + 1, valid_format

    def get_reflect(self, trajectory: List[Dict[str, str]]) -> tuple[
        Optional[Dict[str, Any]], Optional[str], Optional[Any]]:
        """
        Generates a comprehensive reflection report using a single, unified self-interrogation prompt.
        The model first assesses its own performance and then follows the appropriate reflection path.
        """
        # print("Generating reflection report using the unified self-interrogation prompt...")

        # 格式化轨迹以供LLM阅读
        formatted_trajectory = opmd_reflect_enhanced_restart_utils.format_trajectory_for_reflection(trajectory)

        # 使用Jinja2模板渲染反思提示
        reflect_prompt = self.reflection_template.render()

        # 调用模型并解析结果
        try:
            responses = self.model.chat(
                [
                    {"role": "system", "content": reflect_prompt},
                    {"role": "user", "content": "Here is last attempt trajectory log: \n\n" + formatted_trajectory}
                ],
                n=1,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                max_tokens=self.max_tokens,
            )
            reflection_text = responses[0].response_text.strip()

            # print(f"raw reflection text: {reflection_text}")

            # 解析JSON
            json_match = re.search(r"```json\s*(\{.*?\})\s*```", reflection_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = reflection_text

            reflection_data = json.loads(json_str)
            return reflection_data, reflection_text, responses[0]

        except Exception as e:
            # print(f"Failed during unified reflection process: {e}")
            return None, None, None

    def second_rollout(
            self,
            env,
            session_id: int,
            guidance_prompt: str,
            first_trajectory: List[Dict[str, str]],
            retry_step: int = 0,
    ) -> tuple[List[Dict[str, str]], List[Dict[str, str]], float, bool, int, bool]:
        """
        Performs rollout starting from a specific retry step, reusing previous responses.

        Args:
            env: The environment instance.
            session_id: The ID for the current task session.
            guidance_prompt: The pre-generated guidance from reflection.
            first_trajectory: The full log of the initial attempt.
            retry_step: The step to start retry from (0-based, 0 means from beginning).

        Returns:
            A tuple containing (distill_trajectory, second_trajectory, reward, done status,
            step count, and format validity).
        """

        # Reset environment to start fresh
        env.reset(session=session_id)
        observation = env.observation
        trajectory = []
        distill_trajectory = []
        action_history = []  # Track last 3 actions for repetition detection

        # Prepare system prompts
        original_system_prompt = self.webshop_system_template.render()

        default_reward = -0.1
        reward = default_reward
        valid_format = True

        # Copy responses from first trajectory up to retry_step
        step = 0
        if retry_step > 0:
            # Add original system prompt only
            trajectory.append({"role": "system", "content": original_system_prompt})
            distill_trajectory.append({"role": "system", "content": original_system_prompt})

            # Replay first trajectory up to retry_step to restore environment state
            first_step = 0
            for msg in first_trajectory[1:]:  # Skip system message
                if msg["role"] == "user":
                    # This is an observation - copy it and continue
                    trajectory.append(msg)
                    distill_trajectory.append(msg)
                elif msg["role"] == "assistant":
                    if first_step < retry_step:
                        # Copy the assistant response from first trajectory
                        trajectory.append(msg)
                        distill_trajectory.append(msg)

                        # Execute the action to restore environment state
                        think, action = opmd_reflect_enhanced_restart_utils.parse_response(msg["content"])
                        if think is not None and action is not None:
                            action_valid, error_msg = opmd_reflect_enhanced_restart_utils.validate_action(action, env.get_available_actions())
                            if action_valid:
                                observation, reward, done, info = env.step(action)
                                action_history.append(action)
                                if len(action_history) > 2:
                                    action_history.pop(0)
                            else:
                                # If action becomes invalid during replay, start from beginning
                                retry_step = 0
                                break
                        first_step += 1
                        step = first_step

                        if done:
                            # If environment finished during replay, no need to continue
                            return distill_trajectory, trajectory, reward, done, step, valid_format
                    else:
                        break

            # Add guidance prompt as a separate system message before retry point
            guidance_system_msg = {"role": "system", "content": f"# Previous Attempt Analysis & Guidance\n{guidance_prompt}"}
            trajectory.append(guidance_system_msg)
            # Don't add guidance to distill_trajectory to keep it clean

        else:
            # Starting from beginning - add system prompt with guidance
            merged_system_prompt = f"{original_system_prompt}\n\n# Previous Attempt Analysis & Guidance\n{guidance_prompt}"
            trajectory.append({"role": "system", "content": merged_system_prompt})
            distill_trajectory.append({"role": "system", "content": original_system_prompt})

        for step in range(step, self.max_env_steps):
            available_actions = env.get_available_actions()
            trajectory.append(
                {"role": "user", "content": format_observation(observation, available_actions)}
            )
            distill_trajectory.append(
                {"role": "user", "content": format_observation(observation, available_actions)}
            )

            # Get model response with guidance
            responses = self.model.chat(
                trajectory,
                n=1,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
            )
            response_text = responses[0].response_text.strip()
            trajectory.append({"role": "assistant", "content": response_text})
            distill_trajectory.append({"role": "assistant", "content": response_text})

            # Parse the response
            think, action = opmd_reflect_enhanced_restart_utils.parse_response(response_text)
            if think is None or action is None:
                valid_format = False
                feedback = "Invalid response format, missing valid <think> or <action> tags, please ensure to follow the output format strictly: <think>...</think> <action>...</action>"
                trajectory.append({"role": "user", "content": f"Feedback: {feedback}"})
                distill_trajectory.append({"role": "user", "content": f"Feedback: {feedback}"})
                return distill_trajectory, trajectory, default_reward, False, step + 1, valid_format

            # Check for consecutive action repetition
            action_history.append(action)
            if len(action_history) > 2:
                action_history.pop(0)

            # If last 2 actions are the same, terminate with failure
            if len(action_history) >= 2 and all(
                    action == action_history[0] for action in action_history
            ) and "next" not in action.lower() and "prev" not in action.lower() and "search" not in action.lower():
                feedback = f"Repeated invalid action {action} multiple times, shopping task failed"
                trajectory.append({"role": "user", "content": f"Feedback: {feedback}"})
                distill_trajectory.append({"role": "user", "content": f"Feedback: {feedback}"})
                valid_format = False
                return distill_trajectory, trajectory, default_reward, False, step + 1, valid_format

            # Validate and execute action in environment
            action_valid, error_msg = opmd_reflect_enhanced_restart_utils.validate_action(action, available_actions)
            if action_valid:
                observation, reward, done, info = env.step(action)
            else:
                observation, reward, done = error_msg, default_reward, False

            if done:
                break

        # Generate feedback
        if reward >= 1.0 and step + 1 < self.max_env_steps:
            feedback = f"Shopping task completed successfully (reward: {reward}/1.0), and satisfying the step limit ({step + 1}/{self.max_env_steps} steps)"
        elif reward >= 1.0 and step + 1 >= self.max_env_steps:
            feedback = (
                f"Shopping task completed successfully (reward: {reward}/1.0), but exceeded the step limit ({step + 1}/{self.max_env_steps} steps)"
            )
        elif reward < 1.0 and step + 1 < self.max_env_steps:
            feedback = (
                f"Shopping task not completed (reward: {reward}/1.0), but within the step limit ({step + 1}/{self.max_env_steps} steps). It may not satisfy the Attribute Matching, Option Matching, or Price Matching requirements, please you carefully check and ensure all requirements are satisfied."
            )
        else:
            feedback = (
                f"Shopping task not completed (reward: {reward}/1.0), and exceeded the step limit ({step + 1}/{self.max_env_steps} steps). It may not satisfy the Attribute Matching, Option Matching, or Price Matching requirements, please you carefully check and ensure all requirements are satisfied."
            )

        # Add feedback to trajectory
        trajectory.append({"role": "user", "content": f"Feedback: {feedback}"})
        distill_trajectory.append({"role": "user", "content": f"Feedback: {feedback}"})

        # For compatibility, return the same trajectory as both distill_trajectory and second_trajectory
        # since we're starting fresh instead of resuming from a checkpoint
        return distill_trajectory, trajectory, reward, False, step + 1, valid_format

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

    def eval_webshop(self) -> List[Experience]:
        """Evaluate a single webshop trajectory"""
        try:
            trajectory, reward, done, steps, valid_format = self.first_rollout(
                self.env, self.session_id
            )
            exp = self.model.convert_messages_to_experience(trajectory[:-1])
            exp.reward = reward
            exp.metrics = {
                "success": 1.0 if reward >= 1.0 else 0.0,
                "steps": steps,
                "reward": reward,
            }

            # Save evaluation data
            eval_task_id = f"{str(self.task.batch_id).replace('/', '_')}_{self.task.task_id}"
            eval_record = opmd_reflect_enhanced_restart_utils.create_experience_record(
                task_id=eval_task_id,
                trajectory=trajectory,
                reward=reward,
                steps=steps,
                success=reward >= 1.0,
                attempt_type="evaluation"
            )
            opmd_reflect_enhanced_restart_utils.save_experience_data(
                task_id=f"{eval_task_id}_eval",
                experience_data=eval_record,
                data_dir=self.eval_dir
            )
        except Exception as e:
            # logger.warning(f"Single rollout failed during eval: {e}")
            task_id = f"{str(self.task.batch_id).replace('/', '_')}_{self.task.task_id}"
            exp = Experience(
                tokens=torch.tensor([0, 0], dtype=torch.long),
                prompt_length=1,
                action_mask=torch.tensor([False], dtype=torch.bool),
                logprobs=torch.tensor([0.0], dtype=torch.float),
                metrics={
                    "success": 0.0,
                    "reward": 0.0,
                }
            )
            exp.reward = -0.1
        return [exp]

    def run(self) -> List[Experience]:
        """Run the experience learning webshop workflow and return experiences"""

        if self.is_eval:
            # print("pass evaluation mode")
            # return [opmd_reflect_enhanced_restart_utils.generate_default_experience()]
            return self.eval_webshop()

        # Generate unique task ID using timestamp
        task_id = f"{str(self.task.batch_id).replace('/', '_')}_{self.task.task_id}"

        exp_lst = []
        for i in range(self.n // 2):  # 一半用于rollout，一半在此基础上进行反思再rollout
            try:
                trajectory, reward, done, steps, format_valid = self.first_rollout(
                    self.env, self.session_id
                )

                exp = self.model.convert_messages_to_experience(trajectory[:-1])
                exp.reward = reward
                # exp.info = {"valid": format_valid}
                # print(exp.info)
                exp.metrics = {
                    "success": 1.0 if reward >= 1.0 else 0.0,
                    "steps": steps,
                    "reward": reward,
                }
                # 设置eid
                exp.eid.task = str(self.task.task_id) + f"_explore"
                exp_run_id = len(exp_lst) + self.run_id_base
                exp.eid.run = exp_run_id
                exp_lst.append(exp)

                # Save first attempt experience data
                first_record = opmd_reflect_enhanced_restart_utils.create_experience_record(
                    task_id=task_id,
                    trajectory=trajectory,
                    reward=reward,
                    steps=steps,
                    success=reward >= 1.0,
                    attempt_type="first"
                )
                opmd_reflect_enhanced_restart_utils.save_experience_data(
                    task_id=f"{task_id}_attempt_{i}_first",
                    experience_data=first_record,
                    data_dir=self.data_dir
                )

                # 对首次尝试进行反思
                reflect_checklist, reflection_text, reflect_exp = self.get_reflect(trajectory)
                is_valid, is_perfect = opmd_reflect_enhanced_restart_utils.validate_reflect_report(reflect_checklist, steps)

                if not is_valid or is_perfect:
                    # print("Reflect report is invalid or indicates perfection, skipping second rollout")
                    # 如果第一次尝试的reward是1.0且反思给出完美，则记录反思exp
                    if reward >= 1.0 and is_perfect and reflect_exp is not None:
                        reflect_exp.reward = 1.0
                        # 设置eid
                        reflect_exp.eid.task = str(self.task.task_id) + f"_reflect_{i}"
                        reflect_exp.eid.run = len(exp_lst) + self.run_id_base
                        exp_lst.append(reflect_exp)

                    if not is_valid:
                        # 再进行一次rollout，以让整个batch有足够的数据
                        try:
                            retry_trajectory, retry_reward, retry_done, retry_steps, retry_format_valid = self.first_rollout(
                                self.env, self.session_id
                            )

                            retry_exp = self.model.convert_messages_to_experience(retry_trajectory[:-1])
                            retry_exp.reward = retry_reward
                            retry_exp.metrics = {
                                "success": 1.0 if retry_reward >= 1.0 else 0.0,
                                "steps": retry_steps,
                                "reward": retry_reward,
                            }
                            # 设置eid
                            retry_exp.eid.task = str(self.task.task_id) + f"_explore"
                            retry_exp.eid.run = len(exp_lst) + self.run_id_base
                            exp_lst.append(retry_exp)

                            # Save retry attempt experience data
                            retry_record = opmd_reflect_enhanced_restart_utils.create_experience_record(
                                task_id=task_id,
                                trajectory=retry_trajectory,
                                reward=retry_reward,
                                steps=retry_steps,
                                success=retry_reward >= 1.0,
                                attempt_type="retry_after_invalid_reflection"
                            )
                            opmd_reflect_enhanced_restart_utils.save_experience_data(
                                task_id=f"{task_id}_attempt_{i}_retry",
                                experience_data=retry_record,
                                data_dir=self.data_dir
                            )
                        except Exception as e:
                            print(f"Retry rollout after invalid reflection failed: {e}")

                else:
                    guidance_prompt = opmd_reflect_enhanced_restart_utils.reflect_report_to_guidance_prompt(reflect_checklist)
                    # Extract retry_step from validated reflection report
                    retry_step = reflect_checklist["analysis"]["retry_strategy"]["retry_step"]

                    try:
                        (
                            distill_trajectory,
                            second_trajectory,
                            second_reward,
                            second_done,
                            second_steps,
                            second_format_valid,
                        ) = self.second_rollout(
                            self.env, self.session_id, guidance_prompt, trajectory, retry_step
                        )

                        second_exp = self.model.convert_messages_to_experience(distill_trajectory[:-1])

                        # Adjust action_mask to exclude retry prefix from training
                        if retry_step > 0:
                            self._adjust_action_mask_for_retry(second_exp, retry_step)
                            # Also adjust first rollout exp for fair comparison
                            # Find and modify the exp that was already added to exp_lst
                            for existing_exp in exp_lst:
                                if existing_exp.eid.run == exp_run_id:
                                    self._adjust_action_mask_for_retry(existing_exp, retry_step)
                                    break

                        second_exp.reward = second_reward
                        # second_exp.info = {"valid": second_format_valid}
                        second_exp.metrics = {
                            "second_success": 1.0 if second_reward >= 1.0 else 0.0,
                            "second_steps": second_steps,
                            "second_reward": second_reward,
                            "second_improve": 1.0 if second_reward > reward else 0.0,
                            "second_reward_diff": second_reward - reward,
                        }
                        # 设置eid
                        second_exp.eid.task = str(self.task.task_id) + f"_explore"
                        second_exp.eid.run = len(exp_lst) + self.run_id_base
                        exp_lst.append(second_exp)

                        # Save second attempt experience data
                        second_record = opmd_reflect_enhanced_restart_utils.create_experience_record(
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
                                "step_difference": second_steps - steps
                            }
                        )
                        opmd_reflect_enhanced_restart_utils.save_experience_data(
                            task_id=f"{task_id}_attempt_{i}_second",
                            experience_data=second_record,
                            data_dir=self.data_dir
                        )

                        # 如果第二次尝试的分数高于第一次，或第二次是满分情况下步数更少，则记录反思和重试数据
                        if (second_reward > reward and second_reward >= 1.0) or (second_reward >= 1.0 and second_steps < steps):  # or (second_reward >= 1.0 and second_steps < steps) 暂时不考虑步数
                            # 将反思数据转换为exp
                            # reflect_exp.reward = second_reward - reward
                            reflect_exp.reward = 1.0
                            # 设置eid
                            reflect_exp.eid.task = str(self.task.task_id) + f"_reflect_{i}"
                            reflect_exp.eid.run = len(exp_lst) + self.run_id_base
                            exp_lst.append(reflect_exp)

                            # 将重试数据转换为exp
                            retry_exp = self.model.convert_messages_to_experience(second_trajectory[:-1])

                            # Adjust action_mask to exclude retry prefix from training
                            if retry_step > 0:
                                self._adjust_action_mask_for_retry(retry_exp, retry_step)

                            # retry_exp.reward = second_reward - reward
                            retry_exp.reward = 1.0
                            # 设置eid
                            retry_exp.eid.task = str(self.task.task_id) + f"_retry_{i}"
                            retry_exp.eid.run = len(exp_lst) + self.run_id_base
                            exp_lst.append(retry_exp)

                            # print
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
