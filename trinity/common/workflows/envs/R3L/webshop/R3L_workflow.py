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
from trinity.common.workflows.envs.R3L.webshop import utils
from trinity.common.workflows.workflow import WORKFLOWS, Task, Workflow


@WORKFLOWS.register_module("R3L_webshop_workflow")
class R3LWebshopWorkflow(Workflow):
    """
    R3L workflow for webshop
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
        self.max_env_steps = 15
        self.max_tokens = 4096
        self.task = task
        self.is_eval = task.is_eval

        self.whether_save_data = False
        # Create data directories
        self.data_dir = f"R3L_webshop_data"
        self.eval_dir = os.path.join(self.data_dir, "eval")
        self.train_dir = os.path.join(self.data_dir, "train")

        os.makedirs(self.eval_dir, exist_ok=True)
        os.makedirs(self.train_dir, exist_ok=True)

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
                "reward": -0.1,
            },
            reward=-0.1 # Default minimum reward for webshop tasks
        )

        self.default_second_exp = Experience(
            tokens=torch.tensor([0, 0], dtype=torch.long),
            prompt_length=1,
            action_mask=torch.tensor([False], dtype=torch.bool),
            logprobs=torch.tensor([0.0], dtype=torch.float),
            metrics={
                "second_success": 0.0,
                "second_reward": -0.1,
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

    def get_reflect(self, trajectory: List[Dict[str, str]]) -> tuple[
        Optional[Dict[str, Any]], Optional[str], Optional[Any]]:
        """
        Generates a comprehensive reflection report using a single, unified self-interrogation prompt.
        The model first assesses its own performance and then follows the appropriate reflection path.
        """
        # print("Generating reflection report using the unified self-interrogation prompt...")

        # 格式化轨迹以供LLM阅读
        formatted_trajectory = utils.format_trajectory_for_reflection(trajectory)

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
                trajectory, reward, done, steps, format_valid = utils.first_rollout(
                    self, self.env, self.session_id
                )
                print(f"[R3L] First rollout - reward: {reward}, steps: {steps}")

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

                if self.whether_save_data:
                    # Save first attempt experience data
                    first_record = utils.create_experience_record(
                        task_id=task_id,
                        trajectory=trajectory,
                        reward=reward,
                        steps=steps,
                        success=reward >= 1.0,
                        attempt_type="first"
                    )
                    utils.save_experience_data(
                        task_id=f"{task_id}_attempt_{i}_first",
                        experience_data=first_record,
                        data_dir=self.train_dir
                    )

                # 对首次尝试进行反思
                reflect_checklist, reflection_text, reflect_exp = self.get_reflect(trajectory)
                is_valid, is_perfect = utils.validate_reflect_report(reflect_checklist, steps)

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
                            retry_trajectory, retry_reward, retry_done, retry_steps, retry_format_valid = utils.first_rollout(
                                self, self.env, self.session_id
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

                            if self.whether_save_data:
                                # Save retry attempt experience data
                                retry_record = utils.create_experience_record(
                                    task_id=task_id,
                                    trajectory=retry_trajectory,
                                    reward=retry_reward,
                                    steps=retry_steps,
                                    success=retry_reward >= 1.0,
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
                    guidance_prompt = utils.reflect_report_to_guidance_prompt(reflect_checklist)
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
                        ) = utils.second_rollout(
                            self, self.env, self.session_id, guidance_prompt, trajectory, retry_step
                        )
                        print(f"[R3L] Second rollout - reward: {second_reward}, steps: {second_steps}, improve: {second_reward > reward}")
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
                                    "step_difference": second_steps - steps
                                }
                            )
                            utils.save_experience_data(
                                task_id=f"{task_id}_attempt_{i}_second",
                                experience_data=second_record,
                                data_dir=self.train_dir
                            )

                        # 如果第二次尝试的分数高于第一次，或第二次是满分情况下步数更少，则记录反思和重试数据
                        if (second_reward > reward and second_reward >= 1.0) or (second_reward >= 1.0 and second_steps < steps):
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
