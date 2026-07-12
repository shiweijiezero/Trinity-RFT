# Reviewer uCGY

**Scores:**
- Confidence: 4 (Quite sure)
- Soundness: 3 (Acceptable)
- Excitement: 3 (Interesting)
- Overall Assessment: 3 (Findings)
- Reproducibility: 3
- Datasets: 2
- Software: 3

---

## Paper Summary

R3L is a reinforcement-learning recipe for LLM reasoning and agentic tasks, built on GRPO, that targets three failure modes the authors identify: (1) inefficient exploration (stochastic sampling rarely finds successful trajectories on hard tasks, and repeated from-scratch rollouts are costly), (2) coarse credit assignment (trajectory-level rewards penalize valid prefixes for late errors), and (3) training instability in failure-dominated groups (negative samples overwhelm the few positives, dispersing probability mass and causing what the authors call entropy collapse). Each of three components addresses one failure mode:

Language-Guided Reflect-then-Retry: for each base trajectory, the model produces a structured reflection (outcome classification, root-cause analysis, improvement suggestion, and a pivot turn k_pivot where the issue first appears), then restarts generation from k_pivot conditioned on that guidance to synthesize a corrected suffix. A "context distillation" step then builds the training input by pairing the original prefix with the corrected suffix while REMOVING the guidance, so corrections transfer to inference where no guidance is available. Two auxiliary SFT tasks (learn-to-reflect, learn-to-retry) are maintained on verified successful corrections.
Pivotal Credit Assignment: because base and retry trajectories share a prefix up to k_pivot, the shared prefix carries no contrastive signal, so a binary mask zeroes out gradient on all pre-pivot turns and updates only the diverging suffix.
Positive Amplification: a single factor alpha > 1 (alpha = 3.0) scales up positive advantages so constructive gradients dominate; the authors additionally DROP both the KL penalty and importance-sampling clipping from GRPO, arguing amplification alone prevents entropy collapse and that importance sampling is unreliable for guidance-generated retry data.
The budget is kept at N by splitting into N/2 base and N/2 conditionally-triggered partial retries (retry only fires when reflection deems the base non-successful).

Evaluation: three backbones (Qwen2.5-1.5B-Instruct, Qwen2.5-7B-Instruct, Qwen3-4B) plus a cross-architecture check on Llama-3.2-3B-Instruct. Agentic environments ALFWorld, WebShop, ScienceWorld; mathematical reasoning trained on DAPO and evaluated on GSM8K, Math500, MinervaMath, OlympiadBench, AMC23, and the DAPO test set. Baselines RAFT, OPMD, GRPO, GSPO, Reflect-GRPO (their reimplementation of Reflect-Retry-Reward), and Critique-GRPO, all reproduced in the Trinity-RFT framework. Headline claims: R3L is best or second-best on all 27 (backbone x benchmark) settings and first on all 9 agentic settings, with 5% to 52% relative improvement over baselines, plus lower rollout time and roughly halved training tokens on multi-step tasks (Tables 9-10). Appendix A provides informal theoretical arguments (entropy-collapse decomposition, a gradient-dominance condition on alpha, a variance-reduction bound from prefix masking, and a local-convergence sketch). Code is released at an anonymized repo.

Possible misunderstanding to flag for the authors: I read the math-task "guidance" as being generated from a comparison to the ground-truth answer (Appendix K.3) rather than from an environment error message. If that is wrong, they should clarify, because it bears on fairness vs baselines (see W4).

## Summary Of Strengths

S1. Clear problem decomposition with a one-to-one mapping to solution components. The three named failure modes (exploration efficiency, credit coarseness, failure-dominated instability) are well-motivated and each maps cleanly to one mechanism. The paper is easy to follow and the design choices are individually sensible.

S2. Broad and consistent empirical coverage. Results span three backbones plus a cross-architecture check, three agentic environments and six math benchmarks, six baselines, and multiple ablation axes (component removal in Table 2, amplification factor 
 in Table 5, synchronization frequency 
 in Table 6, retry count in Table 7, group size 
 in Table 8). 
 is first on all 9 agentic settings and best-or-second on all 27 settings. The consistency across such a wide grid is a real signal, even setting aside the variance caveat in W1.

S3. Compute efficiency is a genuine, load-bearing advantage, not an afterthought. Tables 9-10 show 
 has the lowest rollout time in all four measured settings (e.g. Qwen3-4B on DAPO: 773.5 s/step vs GRPO 1148.0 and Critique-GRPO 2318.4) and roughly halves training tokens on multi-step tasks, largely from pivot-based partial retries and from dropping the reference-model forward pass. A method that is both more accurate and cheaper is more likely to be adopted.

S4. The ablations yield actual insight, not just "every component helps." The retry-count study (Table 7) shows a single retry captures the bulk of the gain and that more retries HURT under a fixed budget, with a plausible explanation (fewer distinct base trajectories, one-shot reflections do not accumulate). The "can Positive Amplification alone match full 
?" decomposition (GRPO+PA reaches 0.807 on ALFWorld vs GRPO 0.720 and full 
 0.928, so amplification is roughly 40% of the gain) is exactly the kind of attribution reviewers want.

## Summary Of Weaknesses

W1. Single-run results with no variance estimates, which directly undercuts the stability claim. Table 1 and the ablation tables appear to report single runs with no seeds, error bars, or significance tests. RL fine-tuning is high-variance (the paper says so), and stability is a HEADLINE contribution, so the absence of across-seed variance is the most serious gap. Many "wins" are within 
 to 
 (e.g. ScienceWorld-7B 
 0.403 vs Critique-GRPO 0.388; several math columns), where a single run cannot support "best on all 27 settings." At minimum, report mean 
 std over 3+ seeds for the main table, or a subset, and ideally seed-variance curves to substantiate the stability claim empirically rather than only via Figure 4 on one task.

W2. The theoretical section is loosely coupled to practice and should be framed as intuition, not guarantees. The gradient-dominance condition (Theorem 3) assumes comparable gradient norms (
); Corollary 4's claim that 
 "covers the most practical spectrum" rests on ranges (
, $|\bar{A}-| / \bar{A}+ \in [1.0, 2.0]
p_{\text{retry}} > 0
p$ and the advantage ratio during training) or soften the framing.

W3. Dropping BOTH KL regularization and importance sampling is a strong change supported by thin evidence. The justification is that Positive Amplification prevents entropy collapse and that importance sampling is unreliable for guidance-generated retry data. The empirical support is Figure 4 (KL and gradient norm stay controlled) on a SINGLE task (ALFWorld). Removing the standard off-policy safeguards is precisely where instability would appear, so a one-task stability plot plus informal theory is insufficient. Please show the KL / gradient-norm / clip-fraction curves on at least WebShop and one math setting, and ideally an ablation that re-adds KL or IS to isolate what each removal costs or saves.

W4. Potential fairness/leakage issue in the math retry guidance, and modest pivot-localization accuracy. For math tasks there is no environment error message, so the retry guidance is "generated based on comparison with the ground truth" (Appendix K.3); the authors state it does not reveal the answer but "points out the type of error." This is a mechanism the sampling baselines (GRPO, GSPO, RAFT) do not receive, so it is important to establish that 
's math gains are not partly from privileged access to ground-truth-derived hints during trajectory synthesis. Relatedly, pivot-localization oracle agreement is modest (Table 11: 43-76% depending on task/step), and for ALFWorld/WebShop the oracle itself is derived from human judgment plus DeepSeek labeling on 100 trajectories, which is a fairly loose ground truth. Please clarify (a) whether baselines get any comparable ground-truth-conditioned signal, and (b) how sensitive results are to pivot-localization error.

W5. Baseline reproduction needs a direct explanation where baselines fall below zero-shot. Some reproduced baselines look implausibly weak: e.g. in Table 13, Critique-GRPO on Qwen2.5-7B GSM8K is 0.678, below GRPO (0.846) and below the zero-shot reference (0.853); Reflect-GRPO shows similar drops. A baseline that trains BELOW its own zero-shot starting point suggests either under-tuning or a specific bad interaction, and if baselines are under-tuned the "best on 27/27" comparison is inflated. Appendix J explains cross-distribution degradation in general, but this specific below-zero-shot behavior should be explained per baseline, with the hyperparameter-tuning effort for baselines stated explicitly.

W6. A genuine definitional inconsistency in the central amplification rule (not a rendering artifact). Equation 11 (page 5, verified in the image) defines the amplified advantage as 
 when 
, and 
 when 
. But the Table 5 caption (page 14) says "All configurations assign advantage 1 to trajectories achieving maximum reward" and Appendix K.3 (page 24) says max-reward trajectories receive 
 while positive-advantage trajectories receive 
. With 
 these differ by 
 for the strongest positive signal in every group, which is not a minor discrepancy. Please reconcile: do max-reward trajectories get 
 or 
? This affects both reproducibility and the interpretation of the amplification mechanism.

W7. Slightly promotional framing and a coupled ablation. The abstract leads with "5% to 52% relative improvements"; the 52% is GSM8K on the 1.5B model over GRPO, which is also the single benchmark where a stronger baseline (Critique-GRPO 0.798) BEATS 
 (0.721). Leading with a top-end number drawn from a weak-baseline comparison on the one benchmark 
 loses is misleading; report typical gains over the STRONGEST baseline. Separately, the "w/o Reflect" ablation (Table 2) necessarily also disables Pivotal Credit (Credit depends on reflection-identified pivots), so that row confounds two removals and cannot isolate the reflection contribution; the authors note this, but it limits what the ablation shows.

W8. Impact is bounded by scope: all backbones are 7B or smaller and all tasks have verifiable rewards. The generality claim ("applicable to any domain where a preference signal exists") is untested; open-ended or subjective-reward domains are explicitly out of scope. This is acknowledged in Limitations, but it does constrain the contribution, and the method's reliance on a base model capable of useful self-reflection (the acknowledged 1.5B cold-start) suggests the recipe may not transfer cleanly to settings where reflection quality is low.

## Comments, Suggestions And Typos

Statistical reporting (ties to W1): even a small seed study on the main table would substantially strengthen the paper and is the single highest-value addition for the rebuttal.

Reconcile the 
 rule (W6) in the camera-ready regardless of the review outcome; it is a correctness/clarity issue readers will hit.

Figures a reviewer should view directly: Figure 1 (page 2, the core GRPO-vs-
 concept with color-coded steps and probability shifts), Figure 4 (page 11, the six-panel stability evidence), Figure 5 (page 20, the four trajectory types), and Figure 6 (page 20, pivot-point drift). These carry content absent from the text layer.

Consider reporting, for the retry mechanism, the fraction of base trajectories that trigger a retry (the trigger rate), since it determines the true rollout cost between 
 and 
 and would make the compute comparison in Table 9 fully concrete.

Typos: the submission checklist has "voilate" (violate) and "not human-related data" (no human-related data). Minor, but worth a pass. Consider also defining "entropy collapse" precisely at first use in the main text rather than only in Appendix A.2.1.

The "Reflect-GRPO" label for the Reflect-Retry-Reward reimplementation is reasonable but could be confused with Critique-GRPO; a one-line note at first use clarifying it is your reimplementation of Bensal et al. would help.

## Limitations And Societal Impact

Adequately discussed. The Limitations section is honest and specific (reflection adds an inference pass, though the pivot mechanism offsets rollout cost; cold-start on the 1.5B model; scope limited to verifiable-reward tasks). No societal-impact concerns for this methodological RL-training work. Reward the authors for the candor rather than penalize it.

## Ethical Concerns

There are no concerns with this submission

Needs Ethics Review: No

## Other Fields

Knowledge Of Or Educated Guess At Author Identity: No

Knowledge Of Paper: N/A, I do not know anything about the paper from outside sources

Knowledge Of Paper Source: N/A, I do not know anything about the paper from outside sources

Impact Of Knowledge Of Paper: N/A, I do not know anything about the paper from outside sources

Reviewer Certification: I certify that the review I entered accurately reflects my assessment of the work. If you used any type of automated tool to help you craft your review, I hereby certify that its use was restricted to improving grammar and style, and the substance of the review is either my own work or the work of an acknowledged secondary reviewer.

Publication Ethics Policy Compliance: I used a privacy-preserving tool exclusively for the use case(s) approved by PEC policy, such as language edits
