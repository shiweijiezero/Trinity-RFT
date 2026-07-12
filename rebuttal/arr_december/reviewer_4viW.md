# Reviewer 4viW

**Scores:**
- Confidence: 3 (Pretty sure)
- Soundness: 3.5
- Excitement: 4 (Exciting)
- Overall Assessment: 3.5 (Borderline Conference)
- Reproducibility: 4
- Datasets: 1
- Software: 1

---

## Paper Summary

This paper argues that RL for LLM reasoning/agents struggles with both: exploration and exploitation. They have noticed that trajectory-level rewards penalize valid prefixes for later errors, and failure-heavy batches create "avoid this" gradients but few "do this" gradients, causing probability mass to disperse (they describe this as an entropy-collapse style issue).

In this work, authors propose R3L, where they combine the following:

Reflect-then-Retry: first run a base attempt. If it fails, do a reflection pass to diagnose the error and identify a pivot/failure point. Retry generation starting from the pivot with corrective guidance (cheaper than restarting).
Pivotal Credit Assignment: Mask the shared prefix so gradients update only the diverging suffix, preventing "good prefix punished by later mistake."
Positive Amplification: Scale positive advantages by a factor α (they use α=3) so successful signals dominate even if failures are more common, preventing unstable dispersion.
They also include auxiliary SFT on verified corrections to maintain reflection/retry skills during policy updates. Their final objective removes importance sampling and KL constraints (they argue retry trajectories are behavior-shifted anyway, and amplification stabilizes drift).

The authors evaluate their methods over various benchmarks and models, and show R3L is best/second-best across most settings; ablations show Reflect-then-Retry is the biggest contributor.

## Summary Of Strengths

Tackles a real RL pathology: penalizing valid prefixes is a big issue in long trajectories; pivot masking is a strong and intuitive fix.

Reduces wasted rollouts: Restarting from a pivot can plausibly cut exploration cost substantially.

Strong empirical evaluation: They test across both agentic and math reasoning tasks and multiple model families/sizes.

## Summary Of Weaknesses

Additional system complexity and overhead: Reflection + retry adds extra inference steps (even if cheaper than full rollouts).

(minor) Cold-start for small models: the paper itself notes small models initially can't reflect well and need warm-up.

Risk of reflection errors: Incorrect pivot identification or low-quality reflections could reinforce wrong behavior.

## Comments, Suggestions And Typos

Provide explicit compute comparisons: wall-clock, environment steps, and token budget vs baselines?

Analyze pivot quality: where do reflections fail, and how often are pivots correct?

Sensitivity analysis for α and group size, and for how many retries are allowed?
