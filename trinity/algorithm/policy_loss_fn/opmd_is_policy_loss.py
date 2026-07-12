"""OPMD policy loss with clipped importance sampling."""

from typing import Dict, Tuple

import torch

from trinity.algorithm.policy_loss_fn.policy_loss_fn import POLICY_LOSS_FN, PolicyLossFn
from trinity.algorithm.utils import masked_loss, masked_mean


@POLICY_LOSS_FN.register_module("opmd_clipped_is")
class OPMDClippedISPolicyLossFn(PolicyLossFn):
    """Add PPO-style ratio clipping while preserving OPMD loss scaling."""

    def __init__(
        self,
        backend: str = "verl",
        tau: float = 1.0,
        clip_range: float = 0.2,
        loss_agg_mode: str = "token-mean",
    ) -> None:
        super().__init__(backend=backend)
        self.tau = tau
        self.clip_range = clip_range
        self.loss_agg_mode = loss_agg_mode

    def __call__(  # type: ignore
        self,
        logprob: torch.Tensor,
        old_logprob: torch.Tensor,
        action_mask: torch.Tensor,
        advantages: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict]:
        log_ratio = logprob - old_logprob
        ratio = torch.exp(log_ratio)
        clipped_ratio = torch.clamp(
            ratio,
            1.0 - self.clip_range,
            1.0 + self.clip_range,
        )
        pg_losses = -advantages * ratio
        clipped_pg_losses = -advantages * clipped_ratio
        token_losses = torch.maximum(pg_losses, clipped_pg_losses)
        loss = masked_loss(
            token_losses,
            action_mask,
            loss_agg_mode=self.loss_agg_mode,
        )
        loss = loss / (1.0 + self.tau)

        clip_fraction = masked_mean(
            torch.ne(ratio, clipped_ratio).float(),
            action_mask,
        )
        ratio_mean = masked_mean(ratio, action_mask)
        approx_kl = masked_mean(-log_ratio, action_mask)
        return loss, {
            "opmd_is_loss": loss.detach().item(),
            "is_clipfrac": clip_fraction.detach().item(),
            "importance_ratio_mean": ratio_mean.detach().item(),
            "old_policy_kl": approx_kl.detach().item(),
        }

    @classmethod
    def default_args(cls) -> Dict:
        return {
            "tau": 1.0,
            "clip_range": 0.2,
            "loss_agg_mode": "token-mean",
        }
