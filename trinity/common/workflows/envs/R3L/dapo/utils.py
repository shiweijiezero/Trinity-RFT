# -*- coding: utf-8 -*-
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from jinja2 import Environment, FileSystemLoader

try:
    from math_verify import parse, verify
except ImportError:
    parse = None
    verify = None

from trinity.common.experience import Experience


def first_rollout(self) -> tuple[List[Dict[str, str]], float, bool, str, str, int]:
    """Run math problem solving with multiple attempts (max 3 attempts)"""
    trajectory = []

    # Add system prompt
    system_prompt = self.dapo_system_template.render()
    trajectory.append({"role": "system", "content": system_prompt})

    # Add user prompt (math problem)
    if self.prompt:
        trajectory.append({"role": "user", "content": self.prompt})
    else:
        trajectory.append({"role": "user", "content": "Please solve the given mathematical problem."})

    final_reward = 0.0
    final_success = False
    final_predicted_answer = ""
    attempt_count = 0

    # Try up to 3 attempts
    for attempt in range(self.max_attempts):
        attempt_count = attempt + 1
        
        # Get model response
        responses = self.model.chat(
            trajectory,
            n=1,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        response_text = responses[0].response_text.strip()
        trajectory.append({"role": "assistant", "content": response_text})

        # Parse think and answer
        think, predicted_answer = parse_response(response_text)
        
        if predicted_answer is None:
            # Invalid format
            feedback = "Invalid response format. Please ensure you provide your answer in <answer>...</answer> tags."
            trajectory.append({"role": "user", "content": f"Feedback: {feedback}"})
            continue

        # Verify answer
        is_correct = math_verify(predicted_answer, self.ground_truth)
        
        if is_correct:
            final_reward = 1.0
            final_success = True
            final_predicted_answer = predicted_answer
            feedback = f"Correct! Your answer {predicted_answer} matches the expected answer."
            trajectory.append({"role": "user", "content": f"Feedback: {feedback}"})
            break
        else:
            # Wrong answer
            if attempt < self.max_attempts - 1:
                feedback = f"Incorrect. Your answer {predicted_answer} does not match. Please try again."
                trajectory.append({"role": "user", "content": f"Feedback: {feedback}"})
            else:
                # Last attempt
                feedback = f"Incorrect. Your answer {predicted_answer} does not match the expected answer. Maximum attempts reached."
                trajectory.append({"role": "user", "content": f"Feedback: {feedback}"})
            final_predicted_answer = predicted_answer

    return trajectory, final_reward, final_success, final_predicted_answer, self.ground_truth, attempt_count


def second_rollout(
        self,
        guidance_prompt: str,
        first_trajectory: List[Dict[str, str]],
        retry_step: int = 0,
) -> tuple[List[Dict[str, str]], List[Dict[str, str]], float, bool, str, str, int]:
    """
    Performs rollout with guidance from reflection.
    For math problems, we typically start from the beginning with guidance.
    """
    trajectory = []
    distill_trajectory = []

    # Prepare system prompts
    original_system_prompt = self.dapo_system_template.render()

    # Starting from beginning with guidance
    merged_system_prompt = f"{original_system_prompt}\n\n# Previous Attempt Analysis & Guidance\n{guidance_prompt}"
    trajectory.append({"role": "system", "content": merged_system_prompt})
    distill_trajectory.append({"role": "system", "content": original_system_prompt})

    # Add user prompt (math problem)
    if self.prompt:
        trajectory.append({"role": "user", "content": self.prompt})
        distill_trajectory.append({"role": "user", "content": self.prompt})
    else:
        trajectory.append({"role": "user", "content": "Please solve the given mathematical problem."})
        distill_trajectory.append({"role": "user", "content": "Please solve the given mathematical problem."})

    final_reward = 0.0
    final_success = False
    final_predicted_answer = ""
    attempt_count = 0

    # Try up to 3 attempts
    for attempt in range(self.max_attempts):
        attempt_count = attempt + 1
        
        # Get model response with guidance
        responses = self.model.chat(
            trajectory,
            n=1,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        response_text = responses[0].response_text.strip()
        trajectory.append({"role": "assistant", "content": response_text})
        distill_trajectory.append({"role": "assistant", "content": response_text})

        # Parse think and answer
        think, predicted_answer = parse_response(response_text)
        
        if predicted_answer is None:
            # Invalid format
            feedback = "Invalid response format. Please ensure you provide your answer in <answer>...</answer> tags."
            trajectory.append({"role": "user", "content": f"Feedback: {feedback}"})
            distill_trajectory.append({"role": "user", "content": f"Feedback: {feedback}"})
            continue

        # Verify answer
        is_correct = math_verify(predicted_answer, self.ground_truth)
        
        if is_correct:
            final_reward = 1.0
            final_success = True
            final_predicted_answer = predicted_answer
            feedback = f"Correct! Your answer {predicted_answer} matches the expected answer."
            trajectory.append({"role": "user", "content": f"Feedback: {feedback}"})
            distill_trajectory.append({"role": "user", "content": f"Feedback: {feedback}"})
            break
        else:
            # Wrong answer
            if attempt < self.max_attempts - 1:
                feedback = f"Incorrect. Your answer {predicted_answer} does not match. Please try again."
                trajectory.append({"role": "user", "content": f"Feedback: {feedback}"})
                distill_trajectory.append({"role": "user", "content": f"Feedback: {feedback}"})
            else:
                # Last attempt
                feedback = f"Incorrect. Your answer {predicted_answer} does not match the expected answer. Maximum attempts reached."
                trajectory.append({"role": "user", "content": f"Feedback: {feedback}"})
                distill_trajectory.append({"role": "user", "content": f"Feedback: {feedback}"})
            final_predicted_answer = predicted_answer

    return distill_trajectory, trajectory, final_reward, final_success, final_predicted_answer, self.ground_truth, attempt_count


def eval_dapo(self) -> List[Experience]:
    """Evaluate a single math problem"""
    try:
        trajectory, reward, success, predicted_answer, ground_truth, attempts = first_rollout(self)
        exp = self.model.convert_messages_to_experience(trajectory[:-1])
        exp.reward = reward
        exp.metrics = {
            "success": 1.0 if success else 0.0,
            "reward": reward,
            "attempts": attempts,
        }

        if self.whether_save_data:
            # Save evaluation data
            eval_task_id = f"{str(self.task.batch_id).replace('/', '_')}_{self.task.task_id}"
            eval_record = create_experience_record(
                task_id=eval_task_id,
                trajectory=trajectory,
                reward=reward,
                success=success,
                predicted_answer=predicted_answer,
                ground_truth=ground_truth,
                attempt_type="evaluation"
            )
            save_experience_data(
                task_id=f"{eval_task_id}_eval",
                experience_data=eval_record,
                data_dir=self.eval_dir
            )
    except Exception as e:
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
        exp.reward = 0.0
    return [exp]


def _get_jinja_env():
    """Initialize Jinja2 environment for template loading."""
    prompts_dir = Path(__file__).parent / "prompts"
    return Environment(
        loader=FileSystemLoader(str(prompts_dir)),
        trim_blocks=True,
        lstrip_blocks=True,
    )


def parse_response(response: str) -> Tuple[Optional[str], Optional[str]]:
    """Parse think and answer from math response"""
    try:
        # Extract think section
        think_pattern = r"<think>\s*(.*?)\s*</think>"
        think_match = re.search(think_pattern, response, re.DOTALL)
        think = think_match.group(1).strip() if think_match else None

        # Extract answer from <answer> tags
        answer_pattern = r"<answer>\s*(.*?)\s*</answer>"
        answer_match = re.search(answer_pattern, response, re.DOTALL | re.IGNORECASE)
        if answer_match:
            answer = answer_match.group(1).strip()
        else:
            # Fallback: look for "Answer:" pattern
            answer_line_pattern = r"Answer:\s*(.+?)(?:\n|$)"
            answer_line_match = re.search(answer_line_pattern, response, re.IGNORECASE)
            answer = answer_line_match.group(1).strip() if answer_line_match else None

        return think, answer
    except Exception as e:
        print(f"Error parsing response: {e}")
        return None, None


def math_verify(predicted_answer: str, ground_truth: str) -> bool:
    """
    Verify if the predicted math answer matches the ground truth using math_verify library.
    """
    if not predicted_answer or not ground_truth:
        return False

    if parse is None or verify is None:
        # Fallback: simple string comparison
        pred_clean = str(predicted_answer).strip().lower()
        gt_clean = str(ground_truth).strip().lower()
        return pred_clean == gt_clean

    try:
        # Parse and verify
        gold = parse(ground_truth)
        answer = parse(predicted_answer)
        return verify(gold, answer)
    except Exception:
        # Fallback comparison
        return str(predicted_answer).strip().lower() == str(ground_truth).strip().lower()


def format_trajectory_for_reflection(trajectory: List[Dict[str, str]]) -> str:
    """
    Format trajectory for reflection analysis.
    Includes all messages including feedback.
    """
    formatted_lines = []
    step_counter = 0

    for msg in trajectory:
        role = msg["role"]
        content = msg["content"]

        if role == "system":
            formatted_lines.append(f"**System Prompt:**\n{content}\n" + "=" * 50)
        elif role == "user":
            formatted_lines.append(f"\n**Step {step_counter} - User:**")
            formatted_lines.append(f"{content}")
        elif role == "assistant":
            formatted_lines.append(f"\n**Step {step_counter} - Assistant Response:**")
            formatted_lines.append(f"{content}")
            step_counter += 1

    return "\n".join(formatted_lines)


def validate_reflect_report(report: Dict[str, Any], total_steps: int) -> Tuple[bool, bool]:
    """
    Validate the structure and content of the reflection report.

    Returns:
        tuple[bool, bool]: (is_valid, is_perfect)
    """
    if not isinstance(report, dict):
        print("Reflection report is not a dict")
        return False, False

    # Check required keys
    if "outcome_assessment" not in report or "analysis" not in report:
        print("Missing required top-level keys in reflection report")
        return False, False

    outcome = report["outcome_assessment"]
    analysis = report["analysis"]

    # Check valid outcome values
    valid_outcomes = ["OPTIMAL", "SUBOPTIMAL_SUCCESS", "PARTIAL", "INEFFECTIVE"]
    if outcome not in valid_outcomes:
        print(f"Invalid outcome_assessment: {outcome}")
        return False, False

    # If OPTIMAL, it's perfect
    is_perfect = (outcome == "OPTIMAL")

    # Check retry_strategy
    if not is_perfect and "retry_strategy" in analysis:
        retry_strategy = analysis["retry_strategy"]
        retry_step = retry_strategy.get("retry_step")

        if retry_step is not None:
            if not isinstance(retry_step, int) or retry_step < 0 or retry_step > total_steps:
                print(f"Invalid retry_step: {retry_step} (total steps: {total_steps})")
                return False, False

    return True, is_perfect


def reflect_report_to_guidance_prompt(report: Dict[str, Any]) -> str:
    """
    Convert a validated reflection report into a guidance prompt for second attempt.
    The guidance should provide directional hints without revealing the answer.
    """
    try:
        analysis = report.get("analysis", {})
        flaw_analysis = analysis.get("flaw_analysis", {})
        lessons_learned = analysis.get("lessons_learned", {})

        # Build guidance sections
        guidance_parts = []

        # Add summary
        if "summary" in analysis:
            guidance_parts.append(f"## Analysis Summary\n{analysis['summary']}")

        # Add error diagnosis (without answer)
        if "diagnosis" in flaw_analysis:
            diagnosis = flaw_analysis["diagnosis"]
            guidance_parts.append(f"\n## Error Diagnosis")
            if "category" in diagnosis and diagnosis["category"]:
                guidance_parts.append(f"Error Type: {diagnosis['category']}")
            if "root_cause" in diagnosis and diagnosis["root_cause"]:
                guidance_parts.append(f"Root Cause: {diagnosis['root_cause']}")

        # Add method hints (directional guidance)
        if "better_approach" in flaw_analysis:
            better_approach = flaw_analysis["better_approach"]
            guidance_parts.append(f"\n## Recommended Approach")
            if "key_insights" in better_approach and better_approach["key_insights"]:
                guidance_parts.append(f"Key Insights: {better_approach['key_insights']}")
            if "method_hints" in better_approach and better_approach["method_hints"]:
                guidance_parts.append(f"Method Hints: {better_approach['method_hints']}")
            if "strategy" in better_approach and better_approach["strategy"]:
                guidance_parts.append(f"Strategy: {better_approach['strategy']}")

        # Add corrective principle
        if "corrective_principle" in lessons_learned and lessons_learned["corrective_principle"]:
            guidance_parts.append(f"\n## Corrective Principle\n{lessons_learned['corrective_principle']}")

        # Add verification reminders
        if "verification_reminders" in lessons_learned and lessons_learned["verification_reminders"]:
            guidance_parts.append(f"\n## Verification Reminders\n{lessons_learned['verification_reminders']}")

        return "\n\n".join(guidance_parts)

    except Exception as e:
        print(f"Error converting reflection to guidance: {e}")
        return "Please try solving the problem again carefully."


def create_experience_record(
    task_id: str,
    trajectory: List[Dict[str, str]],
    reward: float,
    success: bool,
    predicted_answer: str = "",
    ground_truth: str = "",
    attempt_type: str = "first",
    additional_metrics: Optional[Dict] = None
) -> Dict[str, Any]:
    """Create an experience record for data saving"""
    record = {
        "task_id": task_id,
        "timestamp": datetime.now().isoformat(),
        "attempt_type": attempt_type,
        "trajectory": trajectory,
        "reward": reward,
        "success": success,
        "predicted_answer": predicted_answer,
        "ground_truth": ground_truth,
    }

    if additional_metrics:
        record.update(additional_metrics)

    return record


def save_experience_data(
    task_id: str,
    experience_data: Dict[str, Any],
    data_dir: str
):
    """Save experience data to file"""
    os.makedirs(data_dir, exist_ok=True)
    file_path = os.path.join(data_dir, f"{task_id}.json")

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(experience_data, f, ensure_ascii=False, indent=2)


def generate_default_experience() -> Experience:
    """Generate a default experience for failed cases"""
    return Experience(
        tokens=torch.tensor([0, 0], dtype=torch.long),
        prompt_length=1,
        action_mask=torch.tensor([False], dtype=torch.bool),
        logprobs=torch.tensor([0.0], dtype=torch.float),
        metrics={"success": 0.0, "reward": 0.0},
        reward=0.0
    )
