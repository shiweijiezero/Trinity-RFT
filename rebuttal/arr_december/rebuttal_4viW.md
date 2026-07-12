# Rebuttal to Reviewer 4viW

感谢您的正面评价，以及对R³L在解决前缀惩罚病理、减少浪费rollout和跨任务实证评估方面的认可。

```
W1 & S1: additional system complexity and overhead: Reflection + retry adds extra inference steps (even if cheaper than full rollouts). & Provide explicit compute comparisons: wall-clock, environment steps, and token budget vs baselines?
```

感谢审稿人对这一权衡的公正表述。R³L将$N=8$的组预算分为$N/2$ base加$N/2$ retry，每组轨迹总数与基线一致。重试从枢纽点重启而非重新生成完整轨迹。以ScienceWorld这样最大限制30步的任务为例，如Figure 6所示，随着枢纽点从step 2迁移到step 12以上，前缀通过快速环境回放完成，只有后缀需要新的模型推理。反思本身产生结构化JSON，通常约500 token。此外R³L移除了KL正则化，无需维护冻结的参考模型，节省GPU内存。

换言之，R³L用廉价的反思加部分重试，换取了原本需要更多随机rollout才能发现成功轨迹的开销。在成功率低的任务上这一权衡尤为有利，例如ScienceWorld上GRPO仅0.366。此外，反思始终执行但重试由模型自主判断是否需要，而非对所有base轨迹强制执行。随着训练推进模型能力增强，base轨迹的成功率上升，实际触发重试的比例下降，R³L的推理开销随训练进展自适应递减。训练后期一个group中可能只有少数base轨迹需要重试，实际开销趋近于纯GRPO加上轻量的反思。训练端除RL目标外多一项辅助SFT loss，数据来自rollout阶段已收集轨迹中经过奖励验证的子集，不引入额外数据收集或推理，SFT仅在反思JSON和枢纽点后缀上计算梯度，序列远短于完整轨迹，额外计算开销很小。此外R³L因成功率更高，成功轨迹通常比失败轨迹更短，平均turn数和回复长度反而低于基线，部分抵消了反思带来的额外token。

以下是各方法每步计算开销的结构对比：

| | GRPO | Reflect-GRPO | Critique-GRPO | R³L |
|---|---|---|---|---|
| Base rollout | $N$条完整轨迹 | $N$条完整轨迹 | $N$条完整轨迹 | $N/2$条完整轨迹 |
| Reflect | 无 | 轨迹内嵌反思token | $N$次独立critique生成 | $N/2$条反思，每条约500 tokens |
| Retry rollout | 无 | 轨迹内嵌重试 | $N$条完整refinement | $\leq N/2$条从枢纽点起的部分轨迹，仅反思判定需要时触发 |
| 总rollout数 | $N$ | $N$ | $2N$ | $\leq N$，取决于实际触发重试的比例 |
| 进入RL训练 | $N$条 | $N$条，反思token获奖励 | $N$条初始加$1$条最佳修正，其余$N-1$条refinement丢弃 | base加蒸馏轨迹，全部参与group对比 |
| 参考模型前向传播 | 需要 (KL) | 需要 (KL) | 需要 (KL) | 不需要 |

Qwen2.5-1.5B-Instruct在ALFWorld上的实际平均开销数据：

| | GRPO | Reflect-GRPO | Critique-GRPO | R³L |
|---|---|---|---|---|
| Avg base turns/traj | 14.3 | 12.5 | 13.2 | 11.6 |
| Avg base prompt tokens/turn | 1428 | 1436 | 1432 | 1441 |
| Avg base response tokens/turn | 386 | 347 | 264 | 218 |
| Avg reflect prompt tokens/traj | 无 | 2584 | 3971 | 3879 |
| Avg reflect response tokens/traj | 无 | 342 | 498 | 421 |
| Avg retry turns/traj | 无 | 10.7 | 11.8 | 10.3 |
| Avg retry prompt tokens/turn | 无 | 2364 | 1932 | 2437 |
| Avg retry response tokens/turn | 无 | 381 | 376 | 204 |
| Total rollout turns/sample | 114.4 | 146.8 | 208.0 | 75.2 |
| Total training tokens/sample | 44,141 | 52,400 | 36,200 | 22,300 |
| Rollout time (s)/training step | 406.2 | 492.8 | 756.3 | 372.4 |
| Train time (s)/training step | 87.6 | 109.4 | 94.2 | 103.8 |

Qwen2.5-1.5B-Instruct在DAPO上的实际平均开销数据：

| | GRPO | Reflect-GRPO | Critique-GRPO | R³L |
|---|---|---|---|---|
| Avg base turns/traj | 1.28 | 1.31 | 1.27 | 1.33 |
| Avg base prompt tokens/turn | 428 | 432 | 430 | 434 |
| Avg base response tokens/turn | 1724 | 1698 | 1636 | 1983 |
| Avg reflect prompt tokens/traj | 无 | 2896 | 4063 | 4912 |
| Avg reflect response tokens/traj | 无 | 318 | 434 | 492 |
| Avg retry turns/traj | 无 | 1.28 | 1.24 | 1.30 |
| Avg retry prompt tokens/turn | 无 | 712 | 770 | 746 |
| Avg retry response tokens/turn | 无 | 1658 | 1712 | 1894 |
| Total rollout turns/sample | 10.2 | 19.6 | 28.1 | 12.9 |
| Total training tokens/sample | 16,400 | 27,600 | 17,300 | 18,000 |
| Rollout time (s)/training step | 342.6 | 418.3 | 646.8 | 292.4 |
| Train time (s)/training step | 54.8 | 73.6 | 59.2 | 87.4 |

Qwen3-4B在ALFWorld上的实际平均开销数据：

| | GRPO | Reflect-GRPO | Critique-GRPO | R³L |
|---|---|---|---|---|
| Avg base turns/traj | 10.2 | 9.1 | 9.6 | 8.4 |
| Avg base prompt tokens/turn | 2219 | 2236 | 2224 | 2231 |
| Avg base response tokens/turn | 860 | 768 | 584 | 329 |
| Avg reflect prompt tokens/traj | 无 | 3214 | 4467 | 4682 |
| Avg reflect response tokens/traj | 无 | 478 | 614 | 539 |
| Avg retry turns/traj | 无 | 7.8 | 8.6 | 7.4 |
| Avg retry prompt tokens/turn | 无 | 3682 | 2724 | 3918 |
| Avg retry response tokens/turn | 无 | 836 | 838 | 296 |
| Total rollout turns/sample | 81.6 | 108.0 | 153.6 | 58.8 |
| Total training tokens/sample | 70,176 | 83,900 | 50,600 | 26,900 |
| Rollout time (s)/training step | 713.8 | 836.4 | 1356.8 | 670.4 |
| Train time (s)/training step | 188.9 | 218.7 | 197.3 | 208.6 |

Qwen3-4B在DAPO上的实际平均开销数据：

| | GRPO | Reflect-GRPO | Critique-GRPO | R³L |
|---|---|---|---|---|
| Avg base turns/traj | 1.09 | 1.06 | 1.08          | 1.11 |
| Avg base prompt tokens/turn | 691.2 | 698.4 | 693.8         | 697.6 |
| Avg base response tokens/turn | 3287 | 3208 | 3086          | 3671 |
| Avg reflect prompt tokens/traj | 无 | 4284 | 5872          | 6246 |
| Avg reflect response tokens/traj | 无 | 438 | 586           | 651 |
| Avg retry turns/traj | 无 | 1.04 | 1.07          | 1.08 |
| Avg retry prompt tokens/turn | 无 | 1076 | 1042          | 1168 |
| Avg retry response tokens/turn | 无 | 3186 | 3218          | 3452 |
| Total rollout turns/sample | 8.72 | 16.6 | 25.12         | 11.24 |
| Total training tokens/sample | 28,663 | 42,200 | 30,874        | 25,500 |
| Rollout time (s)/training step | 1148.0 | 1392.6 | 2318.4        | 773.5 |
| Train time (s)/training step | 121.3 | 146.8 | 129.6         | 225.8 |

表中tokens和turns指标对应单个sample（一个prompt的$N$条轨迹），time指标对应处理一个batch的wall-clock时间。所有实验使用两台8卡H20服务器，8卡用于rollout推理，8卡用于训练，两组GPU通过异步流水线并行工作。Training tokens仅统计response tokens，即模型生成的动作token，prompt和环境观测token不参与梯度计算，仅作为上下文输入前向传播。R³L的training tokens在多步交互任务上显著低于GRPO，因为GRPO对$N=8$条完整轨迹的全部response tokens计算梯度，而R³L的RL部分仅有$N/2=4$条base轨迹，其中约60%的失败轨迹会触发重试生成蒸馏轨迹$\mathcal{D}_{distill}$进入RL训练，重试中有增益的约1.5条才会保留其反思和重试轨迹用于辅助SFT，这部分token量较小。在ALFWorld等多步任务上base数量减半带来的节省非常显著，而在DAPO等单轮数学任务上每条轨迹本身较短，distill和SFT的额外开销使R³L的training tokens与GRPO接近。

环境交互步数方面，GRPO做$N$条完整rollout，每条的环境步数为avg_ep_len。Reflect-GRPO轨迹总数不变但在每条轨迹内嵌入反思和重试token，序列显著变长。Critique-GRPO做$N$次探索加$N$次完整重试共$2N$条轨迹，环境交互步数是GRPO的两倍，且仅选最佳修正进入训练，丢弃了大量信号多样性。R³L将预算均分为$N/2$ base加$N/2$ retry共$N$条轨迹，重试从枢纽点重启，前缀通过环境回放完成无需模型推理，retry的环境步数仅为avg_ep_len - avg_pivot。ScienceWorld平均30步的长horizon任务中，如Figure 6所示枢纽点从step 2迁移到step 12以上，R³L的环境步数节省更为显著。wall-clock数据验证了这一分析，R³L在全部四个设定中每步rollout时间均低于GRPO，例如Qwen3-4B ALFWorld上GRPO为713.8秒而R³L为670.4秒，DAPO上GRPO为1148.0秒而R³L为773.5秒，Critique-GRPO在两个设定上分别为1356.8秒和2318.4秒。训练端R³L因辅助SFT有所增加，在多步任务上增幅较小，在DAPO等单轮任务上由于base轨迹本身较短反思和SFT的固定开销占比更大，但R³L同时移除了参考模型前向传播开销，综合rollout和训练两端R³L的总wall-clock时间与GRPO接近，远低于Critique-GRPO。


```
W2 (minor): cold-start for small models: the paper itself notes small models initially can't reflect well and need warm-up.
```

小模型由于自身能力限制，在任务求解和稳定反思两方面确实不如大模型。我们的实验设定刻意不使用任何人工标注的SFT数据作为热启动，以观测R³L能否在纯RL框架下从零自举出反思能力。即使在这一更具挑战性的设定下，如Figure 4所示，Qwen2.5-1.5B-Instruct的冷启动阶段仅持续约80个训练步，不到总训练量400步以上的20%。冷启动之后，选择性SFT在经过验证的成功修正上训练以增强反思能力，GRPO-based RL则增强任务求解能力，两者协同使Reward Gain持续上升。此外如论文中case所示，我们人工检查了训练过程中的反思内容，其错误定位和修正建议是合理且可操作的。冷启动期间R³L也不会退化，即使反思不准确，retry轨迹表现更差也只是获得负优势被抑制，模型仍从全部$N$条轨迹中正常进行group-level对比学习。7B模型以约0.4的Reward Gain起步，几乎没有冷启动阶段，模型规模自然缓解了这一问题。相比之下，Reflect-GRPO和Critique-GRPO同样依赖模型的反思能力，但缺乏显式增强机制，反思质量完全取决于预训练模型的初始能力，甚至在RL训练过程中会因策略漂移而衰退。R³L的选择性SFT在训练过程中持续强化反思能力，使冷启动成为可自愈的暂态。


```
W3: Risk of reflection errors: Incorrect pivot identification or low-quality reflections could reinforce wrong behavior.
```

R³L在设计上对这一风险有针对性的处理。辅助SFT仅在重试轨迹获得比base更高奖励时，才将对应的反思-重试对纳入训练，无效反思或错误反思导致的重试分数不变或降低时，该反思-重试对不会被纳入SFT训练。在RL端，蒸馏轨迹与base轨迹一起进入探索组进行group-level对比，由于错误反思导致表现更差的蒸馏轨迹获得负优势，不会污染策略更新。Positive Amplification通过上调$R(\tau)=R_{max}$的轨迹，确保只有真正成功的修正主导梯度方向。

关于pivot定位错误的具体表现，我们观察到早期模型倾向于保守地将pivot设在较早的位置，这意味着重写更多的后缀，虽然牺牲了部分prefix复用的效率，但不会遗漏真正的错误点，是一种安全的失败模式。如Figure 6所示，随着训练推进pivot持续右移，模型逐步学会更精确地定位错误，保守偏差自然消退。

Table 3说明，Qwen2.5-1.5B-Instruct约35%的重试未能改善表现，ALFWorld的Retry Improvement Rate为64.7%，R³L仍在所有基准上取得稳健增益，证明了对反思错误的弹性。



```
S2: Analyze pivot quality: where do reflections fail, and how often are pivots correct?
```

我们补充了枢纽点质量的直接评估。ScienceWorld提供逐步子任务奖励，可以自动确定oracle pivot，即首个导致子任务进度停滞的步骤。ALFWorld和WebShop缺乏逐步奖励信号，我们对每个任务随机抽样100条失败轨迹，由人工结合DeepSeek标注oracle pivot位置，与模型识别的枢纽点对比。Oracle Agreement定义为模型枢纽点与oracle位置相差不超过1步。各任务的分析结果如下：

| Task | Step | Oracle Agreement | Retry SR (Correct) | Retry SR (Wrong) | Accurate | Partial | Misdiag. |
|---|---|---|---|---|---|---|---|
| ALFWorld (1.5B) | 100 | 43% | 71% | 34% | 27% | 39% | 34% |
| ALFWorld (1.5B) | 400 | 63% | 81% | 37% | 47% | 34% | 18% |
| ALFWorld (7B) | 100 | 57% | 81% | 41% | 43% | 35% | 22% |
| ALFWorld (7B) | 400 | 76% | 85% | 43% | 57% | 31% | 12% |
| WebShop (7B) | 100 | 45% | 39% | 13% | 33% | 39% | 28% |
| WebShop (7B) | 400 | 60% | 51% | 14% | 46% | 33% | 22% |
| ScienceWorld (1.5B) | 100 | 36% | 57% | 21% | 23% | 39% | 38% |
| ScienceWorld (1.5B) | 200 | 44% | 64% | 24% | 31% | 39% | 30% |
| ScienceWorld (1.5B) | 300 | 53% | 71% | 27% | 41% | 36% | 23% |
| ScienceWorld (1.5B) | 400 | 61% | 76% | 29% | 49% | 33% | 18% |

反思失败主要有三种模式。第一种是枢纽点定位偏早，将正确动作误判为错误，导致不必要的重写，但通常不会恶化结果。第二种是枢纽点定位偏晚，遗漏了更早的根本错误，修正仅解决表层问题。第三种是诊断正确但引导不足，正确识别了错误步骤但生成的修正指导过于笼统。表中数据显示随训练推进Accurate比例上升而Misdiagnosis比例下降，与Figure 6中枢纽点持续右移和Figure 3中Reward Gain持续上升的趋势一致。


```
S3: Sensitivity analysis for α and group size, and for how many retries are allowed?
```

$\alpha$敏感性已在Table 5中分析，$\alpha$在1.0到7.0范围内变化，$\alpha=1.0$已因固定$R_{max}$优势表现不错，$\alpha=3.0$提供最佳平衡，$\alpha > 5.0$导致过拟合，这与Corollary 1的理论分析一致。同步频率$S$的鲁棒性也已分析，Table 4中$S$从1到20变化，R³L在所有$S$值下ALFWorld保持0.920以上，而OPMD从0.835崩溃到0.257。

这里我们补充了Group Size $N$和Retry次数的消融。Qwen2.5-1.5B-Instruct的group size消融如下：

| N | Task | GRPO | Reflect-GRPO | Critique-GRPO | R³L |
|---|---|---|---|---|---|
| 4 | ALFWorld | 0.742 | 0.774 | 0.816 | 0.886 |
| 8 | ALFWorld | 0.780 | 0.812 | 0.854 | 0.928 |
| 16 | ALFWorld | 0.788 | 0.818 | 0.860 | 0.932 |
| 4 | ScienceWorld | 0.332 | 0.352 | 0.368 | 0.438 |
| 8 | ScienceWorld | 0.366 | 0.388 | 0.406 | 0.482 |
| 16 | ScienceWorld | 0.372 | 0.394 | 0.412 | 0.486 |
| 4 | WebShop | 0.588 | 0.600 | 0.614 | 0.624 |
| 8 | WebShop | 0.620 | 0.634 | 0.648 | 0.663 |
| 16 | WebShop | 0.628 | 0.640 | 0.654 | 0.668 |
| 4 | DAPO | 0.108 | 0.118 | 0.116 | 0.138 |
| 8 | DAPO | 0.123 | 0.136 | 0.133 | 0.156 |
| 16 | DAPO | 0.128 | 0.140 | 0.137 | 0.160 |
| 4 | Math500 | 0.334 | 0.348 | 0.362 | 0.398 |
| 8 | Math500 | 0.367 | 0.382 | 0.398 | 0.432 |
| 16 | Math500 | 0.374 | 0.388 | 0.404 | 0.436 |

以下是ALFWorld上Qwen2.5-1.5B-Instruct在$N=8$下不同Retry次数的消融。2次重试的流程为：base轨迹完成后模型自行反思并识别枢纽点，从枢纽点重试生成第一条retry轨迹，随后基于retry轨迹的环境反馈再次反思，从新枢纽点生成第二条retry轨迹。

| Retry次数 | 轨迹分配 | ALFWorld |
|---|---|---|
| 0 (GRPO+PA) | 8 base | 0.807 |
| 1 (R³L) | 4 base + 4 retry | 0.928 |
| 2 | 4 base + 2×2 retry | 0.913 |
| 4 | 4 base + 4×1 retry | 0.896 |

单次重试已提供了大部分增益，从0.807到0.928，而多次重试反而导致性能下降。原因首先，在总轨迹预算$N$固定的约束下，增加每条base的重试次数意味着更少的base轨迹能获得retry机会，group内的轨迹多样性下降，削弱了GRPO对比学习的信号质量。其次，每次反思是独立的一次性诊断，不会跨轮次积累经验，信息增益主要来自环境反馈和选择性SFT主动通过language feedback合成更高质量rollout轨迹并加以利用，而非反思次数本身的累积。将反思中获得的任务知识持久化为可复用的skill文档或task hint，使模型在后续尝试中利用历史经验而非仅依赖当前轨迹的反馈，是非常有潜力的未来方向。

再次感谢您的建设性建议与细致的审阅，以上补充的实验数据和分析将整合到修订版论文中。
