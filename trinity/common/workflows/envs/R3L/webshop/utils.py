import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from jinja2 import Environment, FileSystemLoader
import torch
from trinity.common.experience import Experience


def _get_jinja_env():
    """Initialize Jinja2 environment for template loading."""
    prompts_dir = Path(__file__).parent / "prompts"
    return Environment(
        loader=FileSystemLoader(str(prompts_dir)),
        trim_blocks=True,
        lstrip_blocks=True,
    )


def format_observation(observation: str, available_actions: dict):
    return f"Environment Observation: {observation} \n Available Actions: {available_actions}"


def parse_response(response):
    """Parse all three components from response with a single regex"""
    think, action = None, None
    try:
        # Use single regex to extract all three components at once
        pattern = r"<think>\s*(.*?)\s*</think>.*?<action>\s*(.*?)\s*</action>"
        match = re.search(pattern, response, re.DOTALL)

        if match:
            think, action = match.group(1).strip(), match.group(2).strip()
    except Exception:
        pass
    return think, action


def validate_action(action, available_actions):
    """Validate action format and availability"""
    import re

    # Parse action format: action_name[action_arg]
    pattern = re.compile(r"(.+)\[(.+)\]")
    m = re.match(pattern, action)
    if m is None:
        return (
            False,
            "Invalid action format. You should use format: action_name[action_arg], like search[query] or click[button].",
        )

    action_name, action_arg = m.groups()
    action_name = action_name.strip()
    action_arg = action_arg.strip()

    # Validate search action
    if action_name == "search":
        if not action_arg:
            return (
                False,
                "Invalid search action, please type in the query you want to search in the square brackets.",
            )
        if not available_actions["has_search_bar"]:
            return (
                False,
                "Cannot perform search action without search bar. Please click the Back to Search button first.",
            )
        return True, ""

    # Validate click action
    elif action_name == "click":
        if not action_arg:
            return (
                False,
                "Invalid click action, please specify the button name in the square brackets.",
            )
        # Convert to lowercase for comparison (as clickables are typically lowercase)
        action_arg_lower = action_arg.lower()
        if action_arg_lower not in available_actions["clickables"]:
            return (
                False,
                f"Button '{action_arg}' not found on current page. Available buttons: {available_actions['clickables']}",
            )
        return True, ""

    # Unknown action
    else:
        return (
            False,
            f"Unknown action '{action_name}'. Only 'search' and 'click' actions are supported.",
        )


def format_trajectory_for_reflection(trajectory: List[Dict[str, str]]) -> str:
    """
    Correctly formats the trajectory for reflection, including the system prompt
    and numbering the user/assistant turns.
    """
    formatted_lines = []
    # 使用一个计数器来追踪 user/assistant 的交互轮次
    turn_counter = 0  # 从 0 开始计数

    for msg in trajectory:
        role = msg["role"]
        content = msg["content"]

        if role == "system":
            # 系统提示不计入步骤，但必须作为规则展示在最前面
            formatted_lines.append(f"**System Rules & Context:**\n{content}\n" + "=" * 30)
        elif role == "user":
            # 标记一个新回合的开始
            formatted_lines.append(f"\n**Step {turn_counter}**")
            formatted_lines.append(f"  - User Observation/Feedback:\n    {content.strip()}")
        elif role == "assistant":
            # 助理的思考和行动
            formatted_lines.append(f"  - Assistant Thought & Action:\n    {content.strip()}")
            # 一轮完整的 user-assistant 对话结束后，回合数增加
            turn_counter += 1

    return "\n".join(formatted_lines)


def validate_reflect_report(report: Dict[str, Any], max_steps: int = None) -> tuple[bool, bool]:
    """
    Validates the structure and content of the reflection report
    based on the new reflection.j2 schema.

    Args:
        report: The reflection report to validate
        max_steps: Maximum number of steps in trajectory for retry_step bounds checking (optional)

    Returns:
        tuple[bool, bool]: (is_valid, is_perfect)
        - is_valid: Whether the report structure is valid
        - is_perfect: Whether the report indicates the trajectory is perfect (only meaningful if is_valid is True)
    """
    if (
            not isinstance(report, dict)
            or "outcome_assessment" not in report
            or "analysis" not in report
    ):
        print("Validation failed: Report is not a dict or missing top-level keys.")
        return False, False

    outcome = report["outcome_assessment"]
    analysis = report["analysis"]

    # Check for required top-level analysis keys
    if "summary" not in analysis:
        print("Validation failed: Missing 'summary' in analysis.")
        return False, False

    if outcome == "OPTIMAL":
        # For OPTIMAL, we only need summary and no flaw analysis
        print("OPTIMAL report validation successful.")
        return True, True

    elif outcome in ["SUBOPTIMAL_SUCCESS", "PARTIAL", "INEFFECTIVE"]:
        # For non-optimal outcomes, validate flaw_analysis structure
        flaw_analysis = analysis.get("flaw_analysis", {})

        # Validate diagnosis
        diagnosis = flaw_analysis.get("diagnosis", {})
        valid_categories = [
            "Strategy Flaw",
            "Reasoning Flaw",
            "Execution Flaw",
            "Knowledge Gap",
            "Inefficiency"
        ]
        if diagnosis.get("category") not in valid_categories and diagnosis.get("category") != "null":
            print(f"Validation failed: Invalid 'category'. Got: {diagnosis.get('category')}")
            return False, False

        # Validate better_approach
        better_approach = flaw_analysis.get("better_approach", {})
        required_better_approach_keys = ["strategy", "key_differences", "projected_benefits"]
        for key in required_better_approach_keys:
            if key not in better_approach:
                print(f"Validation failed: Missing '{key}' in better_approach. Got: {better_approach}")
                return False, False

        # Validate lessons_learned
        lessons_learned = analysis.get("lessons_learned", {})
        if not (
                "corrective_principle" in lessons_learned
                and "revised_action_plan" in lessons_learned
        ):
            print(f"Validation failed: Invalid 'lessons_learned'. Got: {lessons_learned}")
            return False, False

        # Validate retry_strategy
        retry_strategy = analysis.get("retry_strategy", {})
        if not retry_strategy:
            print("Validation failed: Missing 'retry_strategy' in analysis.")
            return False, False

        # Validate retry_step
        if "retry_step" not in retry_strategy:
            print("Validation failed: Missing 'retry_step' in retry_strategy.")
            return False, False

        retry_step = retry_strategy["retry_step"]
        if retry_step is not None:
            try:
                retry_step = int(retry_step)
            except (ValueError, TypeError):
                print(f"Validation failed: 'retry_step' must be an integer or null. Got: {retry_step}")
                return False, False
            if not isinstance(retry_step, int) or retry_step < 0:
                print(f"Validation failed: 'retry_step' must be a non-negative integer or null. Got: {retry_step}")
                return False, False

            # Check trajectory bounds if max_steps is provided
            if max_steps is not None:
                if retry_step >= max_steps:
                    print(
                        f"Validation failed: 'retry_step' ({retry_step}) exceeds trajectory bounds (0 to {max_steps - 1}).")
                    return False, False

        # Validate retry_rationale
        if "retry_rationale" not in retry_strategy:
            print("Validation failed: Missing 'retry_rationale' in retry_strategy.")
            return False, False

        print(f"{outcome} report validation successful.")
        return True, False

    else:
        print(f"Validation failed: Unknown 'outcome_assessment': {outcome}")
        return False, False


def reflect_report_to_guidance_prompt(report: Dict[str, Any]) -> str:
    """
    Converts a validated reflection report into a structured, actionable
    guidance prompt for the agent's second attempt. This prompt is framed
    as an internal directive to ensure the model's output is clean for SFT.
    """
    # Convert the report dictionary to a formatted string
    report_str = json.dumps(report, indent=2, ensure_ascii=False)

    # Load and render template
    jinja_env = _get_jinja_env()
    template = jinja_env.get_template("self_correction.j2")

    return template.render(report=report_str)


def save_experience_data(
        task_id: str,
        experience_data: Dict,
        data_dir: str = "opmd_reflect_webshop_data"
) -> str:
    """
    Save experience data including trajectory, rewards, and steps to a JSON file.

    Args:
        task_id: Unique identifier for the task
        experience_data: Dictionary containing experience information
        data_dir: Directory to save the data

    Returns:
        Path to the saved file
    """
    os.makedirs(data_dir, exist_ok=True)

    # Add timestamp for uniqueness
    filename = f"{task_id}.json"
    filepath = os.path.join(data_dir, filename)

    # Ensure experience_data is JSON serializable
    serializable_data = {}
    for key, value in experience_data.items():
        if isinstance(value, torch.Tensor):
            serializable_data[key] = value.tolist()
        elif hasattr(value, '__dict__'):
            # For complex objects, convert to dict representation
            serializable_data[key] = str(value)
        else:
            serializable_data[key] = value

    # Add metadata
    serializable_data["saved_at"] = datetime.now().isoformat()
    serializable_data["task_id"] = task_id

    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, indent=2, ensure_ascii=False)
        # print(f"Experience data saved to: {filepath}")
        return filepath
    except Exception as e:
        # print(f"Failed to save experience data: {e}")
        return ""


def create_experience_record(
        task_id: str,
        trajectory: List[Dict[str, str]],
        reward: float,
        steps: int,
        success: bool,
        attempt_type: str = "first",
        reflection_data: Optional[Dict] = None,
        additional_metrics: Optional[Dict] = None
) -> Dict:
    """
    Create a structured experience record for saving.

    Args:
        task_id: Unique identifier for the task
        trajectory: List of conversation messages
        reward: Final reward received
        steps: Number of steps taken
        success: Whether the task was completed successfully
        attempt_type: Type of attempt ("first", "second", "reflect")
        reflection_data: Optional reflection analysis data
        additional_metrics: Additional metrics to record

    Returns:
        Dictionary containing structured experience data
    """
    experience_record = {
        "task_id": task_id,
        "attempt_type": attempt_type,
        "trajectory": trajectory,
        "metrics": {
            "reward": reward,
            "steps": steps,
            "success": success,
            "trajectory_length": len(trajectory)
        },
        "created_at": datetime.now().isoformat()
    }

    if reflection_data:
        experience_record["reflection"] = reflection_data

    if additional_metrics:
        experience_record["metrics"].update(additional_metrics)

    return experience_record
