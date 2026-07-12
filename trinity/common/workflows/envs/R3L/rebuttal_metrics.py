"""Lightweight metrics used by the ARR May rebuttal experiments."""

from typing import Any


def _completion_tokens(experience: Any) -> float:
    """Count generated/action tokens without depending on a tokenizer."""
    if experience is None:
        return 0.0
    action_mask = getattr(experience, "action_mask", None)
    if action_mask is None:
        return 0.0
    try:
        return float(action_mask.sum().item())
    except (AttributeError, TypeError, ValueError):
        return 0.0


def initialize_base_metrics(base_experience: Any) -> None:
    """Initialize per-base counters before reflection and retry."""
    base_tokens = _completion_tokens(base_experience)
    base_experience.metrics.update(
        {
            "base_completion_tokens": base_tokens,
            "reflection_completion_tokens": 0.0,
            "retry_completion_tokens": 0.0,
            "total_generation_tokens": base_tokens,
            "reflection_valid_rate": 0.0,
            "retry_trigger_rate": 0.0,
            "retry_skip_success_rate": 0.0,
            "retry_skip_invalid_rate": 0.0,
            "retry_completed_rate": 0.0,
        }
    )


def record_reflection_decision(
    base_experience: Any,
    reflection_experience: Any,
    *,
    is_valid: bool,
    is_perfect: bool,
) -> None:
    """Record the deterministic retry decision for one base trajectory."""
    reflection_tokens = _completion_tokens(reflection_experience)
    triggered = bool(is_valid and not is_perfect)
    base_experience.metrics.update(
        {
            "reflection_completion_tokens": reflection_tokens,
            "reflection_valid_rate": float(is_valid),
            "retry_trigger_rate": float(triggered),
            "retry_skip_success_rate": float(is_valid and is_perfect),
            "retry_skip_invalid_rate": float(not is_valid),
            "total_generation_tokens": (
                base_experience.metrics["base_completion_tokens"] + reflection_tokens
            ),
        }
    )


def record_retry_completion(base_experience: Any, retry_experience: Any) -> None:
    """Record completion tokens once a triggered retry finishes."""
    retry_tokens = _completion_tokens(retry_experience)
    base_experience.metrics.update(
        {
            "retry_completion_tokens": retry_tokens,
            "retry_completed_rate": 1.0,
            "total_generation_tokens": (
                base_experience.metrics["base_completion_tokens"]
                + base_experience.metrics["reflection_completion_tokens"]
                + retry_tokens
            ),
        }
    )
