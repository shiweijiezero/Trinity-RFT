"""Controlled pivot perturbation evaluation for the ARR May rebuttal."""

import copy
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Dict, List

from trinity.common.experience import Experience
from trinity.common.models.model import ModelWrapper
from trinity.common.workflows.envs.R3L.alfworld import utils
from trinity.common.workflows.envs.R3L.alfworld.R3L_workflow import (
    R3LAlfworldWorkflow,
)
from trinity.common.workflows.workflow import WORKFLOWS, Task


PIVOT_OFFSETS = {
    "model": 0,
    "early_2": -2,
    "late_2": 2,
    "early_5": -5,
    "late_5": 5,
}


@WORKFLOWS.register_module("pivot_perturbation_alfworld_workflow")
class PivotPerturbationAlfworldWorkflow(R3LAlfworldWorkflow):
    """Evaluate retry outcomes while changing only the restart pivot."""

    can_repeat: bool = False

    def __init__(
        self,
        model: ModelWrapper,
        task: Task,
        auxiliary_models: List | None = None,
    ) -> None:
        super().__init__(model=model, task=task, auxiliary_models=auxiliary_models)
        args = dict(task.workflow_args or {})
        self.output_dir = Path(
            args.get(
                "output_dir",
                "rebuttal/arr_may_emnlp/results/pivot_perturbation_alfworld",
            )
        )
        self.variants = list(
            args.get(
                "variants",
                [
                    "model",
                    "early_2",
                    "late_2",
                    "early_5",
                    "late_5",
                    "start",
                ],
            )
        )
        self.save_trajectories = bool(args.get("save_trajectories", False))

    def _result_experience(self, metrics: Dict[str, float]) -> Experience:
        experience = copy.deepcopy(self.default_exp)
        experience.metrics = metrics
        return experience

    def _task_key(self) -> str:
        raw_key = f"{self.task.task_id}:{self.game_file_path}"
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:16]

    def _write_result(self, record: Dict) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        safe_task_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(self.task.task_id))
        output_path = self.output_dir / f"{safe_task_id}_{self._task_key()}.json"
        temp_path = output_path.with_suffix(f".{os.getpid()}.tmp")
        with temp_path.open("w", encoding="utf-8") as output_file:
            json.dump(record, output_file, ensure_ascii=False, indent=2)
        os.replace(temp_path, output_path)

    def _variant_pivots(self, model_pivot: int, total_steps: int) -> Dict[str, int]:
        upper_bound = max(total_steps - 1, 0)
        candidates = {
            "model": model_pivot,
            "early_2": max(0, model_pivot - 2),
            "late_2": min(upper_bound, model_pivot + 2),
            "early_5": max(0, model_pivot - 5),
            "late_5": min(upper_bound, model_pivot + 5),
            "start": 0,
        }
        unknown = [variant for variant in self.variants if variant not in candidates]
        if unknown:
            raise ValueError(f"Unknown pivot variants: {unknown}")
        return {variant: candidates[variant] for variant in self.variants}

    def run(self) -> List[Experience]:
        record: Dict = {
            "task_id": str(self.task.task_id),
            "game_file": self.game_file_path,
            "variants": {},
        }
        metrics = {
            "pivot_eval_eligible": 0.0,
            "pivot_base_success": 0.0,
            "pivot_reflection_valid": 0.0,
        }

        try:
            base_env = utils.create_alfworld_environment(self.game_file_path)
            trajectory, reward, _, steps, format_valid = utils.first_rollout(
                self, base_env
            )
            record.update(
                {
                    "base_reward": reward,
                    "base_steps": steps,
                    "base_format_valid": format_valid,
                }
            )
            metrics["pivot_base_success"] = float(reward >= 1.0)

            if self.save_trajectories:
                record["base_trajectory"] = trajectory
            if reward >= 1.0:
                record["skip_reason"] = "base_success"
                self._write_result(record)
                return [self._result_experience(metrics)]

            reflect_report, reflection_text, _ = self.get_reflect(trajectory)
            is_valid, is_perfect = utils.validate_reflect_report(reflect_report, steps)
            metrics["pivot_reflection_valid"] = float(is_valid)
            record.update(
                {
                    "reflection_valid": is_valid,
                    "reflection_perfect": is_perfect,
                    "reflection_report": reflect_report,
                }
            )
            if self.save_trajectories:
                record["reflection_text"] = reflection_text
            if not is_valid or is_perfect:
                record["skip_reason"] = (
                    "invalid_reflection"
                    if not is_valid
                    else "reflection_marked_success"
                )
                self._write_result(record)
                return [self._result_experience(metrics)]

            model_pivot = int(reflect_report["retry_from_step"])
            record["model_pivot"] = model_pivot

            # Keep semantic guidance fixed across variants. The actual restart point is
            # controlled only by the retry_step argument below.
            guidance_report = copy.deepcopy(reflect_report)
            guidance_report.pop("retry_from_step", None)
            guidance_prompt = utils.reflect_report_to_guidance_prompt(guidance_report)
            variant_pivots = self._variant_pivots(model_pivot, steps)

            result_cache: Dict[int, Dict] = {}
            for variant, tested_pivot in variant_pivots.items():
                if tested_pivot not in result_cache:
                    retry_env = utils.create_alfworld_environment(self.game_file_path)
                    (
                        _,
                        retry_trajectory,
                        retry_reward,
                        _,
                        retry_steps,
                        retry_format_valid,
                    ) = utils.second_rollout(
                        self,
                        retry_env,
                        guidance_prompt,
                        trajectory,
                        tested_pivot,
                    )
                    result_cache[tested_pivot] = {
                        "tested_pivot": tested_pivot,
                        "retry_reward": retry_reward,
                        "retry_success": retry_reward >= 1.0,
                        "reward_improvement": retry_reward - reward,
                        "retry_steps": retry_steps,
                        "retry_format_valid": retry_format_valid,
                    }
                    if self.save_trajectories:
                        result_cache[tested_pivot]["retry_trajectory"] = (
                            retry_trajectory
                        )

                variant_result = copy.deepcopy(result_cache[tested_pivot])
                requested_delta = PIVOT_OFFSETS.get(variant)
                actual_delta = tested_pivot - model_pivot
                variant_result.update(
                    {
                        "requested_delta": requested_delta,
                        "actual_delta": actual_delta,
                        "clipped": (
                            requested_delta is not None
                            and actual_delta != requested_delta
                        ),
                    }
                )
                record["variants"][variant] = variant_result
                metrics[f"pivot_{variant}_success"] = float(
                    variant_result["retry_success"]
                )
                metrics[f"pivot_{variant}_reward"] = float(
                    variant_result["retry_reward"]
                )

            metrics["pivot_eval_eligible"] = 1.0
            record["unique_retry_rollouts"] = len(result_cache)
            self._write_result(record)
            return [self._result_experience(metrics)]
        except Exception as error:
            record["error"] = f"{type(error).__name__}: {error}"
            self._write_result(record)
            raise
