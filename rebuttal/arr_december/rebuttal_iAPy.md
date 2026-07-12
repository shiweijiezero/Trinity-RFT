# Rebuttal to Reviewer iAPy

感谢审稿人的细致评审，以及对Pivot-Based Credit Masking、Positive Amplification和Reflect-then-Retry的认可。以下是我们的回复和补充分析。


```
W1: While the integration is coherent, elements such as retry-based refinement, advantage reweighting, and partial masking have precedents in related literature.
```

感谢审稿人肯定整合的连贯性。R³L的核心贡献在于识别出LLM RL中三个正交的结构性瓶颈，并设计统一框架使三者协同工作。Table 2的消融实验证实每个组件提供独立且可叠加的增益，说明它们解决的确实是不同问题，且三者的优化目标兼容，不会产生梯度方向冲突或训练震荡。

与先前工作相比，R³L在每个维度上都有明确的设计区别。Critique-GRPO做$N$次探索加$N$次完整重试但只选最佳修正进入训练，丢弃了大量信号多样性，R³L将预算均分为$N/2$ base加$N/2$ retry，所有轨迹参与group对比学习。Critique-GRPO和Reflect-GRPO都没有枢纽信用分配，仍然对整条轨迹统一赋予优势值，前缀惩罚问题依然存在。在信用分配上，Process Reward Models、GiGPO、VinePPO依赖昂贵的人工标注、学习的评判器或蒙特卡洛rollout，R³L直接利用base-retry对的结构对称性获取对比信号，无需外部监督。在优势重加权上，BAPO需要多个动态裁剪参数，GSPO需要自适应裁剪范围，Positive Amplification仅用单一固定$\alpha=3.0$即在所有模型和任务上有效，且有Theorem 2的梯度主导性分析作为理论支撑。

除了核心组件层面的差异，R³L在实现细节上也有针对性的设计。反思输出采用结构化JSON格式，包含轨迹结果分类、根因分析和重试起始步三个字段，确保反思信号可解析且可操作，同时约束输出格式以减少偶发的生成退化和数据污染。重试并非对所有失败轨迹强制执行，而是由模型根据反思结果自主判断是否触发，训练后期需要重试的比例自然下降，推理开销自适应递减。反思和引导文本通过context distillation在训练前移除，蒸馏轨迹仅保留原始前缀拼接修正后缀，进入RL训练的轨迹与基线结构完全一致。辅助SFT仅在经过奖励验证的成功修正上训练，持续增强反思和重试能力，避免了Reflect-GRPO和Critique-GRPO中随策略分布漂移出现的反思能力退化。这些算法设计和工程实现共同使R³L在大多数设置中取得了一致改进。


```
W2 & S1: The framework assumes that the model can reliably identify the true error turn and generate an effective correction. However, self-reflection may mislocalize failures or produce superficial diagnoses, and the paper does not provide direct evaluation of pivot accuracy or correction validity beyond final task reward. & The paper would benefit from a more direct evaluation of the self-reflection component. In particular, reporting metrics such as pivot identification accuracy, correction success rate conditioned on detected pivots, or agreement with oracle error locations would strengthen the claim that reflection reliably localizes and fixes errors, rather than merely improving outcomes indirectly.
```

感谢这一建设性建议。我们在论文的Table 3、Figure 3和Figure 6中首先间接说明了反思的有效性。Table 3的Retry Improvement Rate衡量重试轨迹获得更高奖励的比例，Qwen2.5-7B在ALFWorld上达73.9%，WebShop上36.5%，Qwen2.5-1.5B-Instruct分别为64.7%和23.7%。重试成功的前提是反思正确定位了错误，因此该指标是反思质量的功能性度量。Figure 6显示平均枢纽点在训练中持续右移，ALFWorld从step 2到step 6以上，ScienceWorld从step 2到step 12以上，表明模型逐步学会识别更精确的失败位置。Figure 3中持续的正向Reward Gain说明反思产生的是可操作的修正而非肤浅诊断。

我们进一步补充了枢纽点定位准确性的直接评估。ScienceWorld提供逐步子任务奖励，可以自动确定oracle pivot，即首个导致子任务进度停滞的步骤。ALFWorld和WebShop缺乏逐步奖励信号，我们对每个任务随机抽样100条失败轨迹，由人工结合DeepSeek标注oracle pivot位置，与模型识别的枢纽点对比。Oracle Agreement定义为模型枢纽点与oracle位置相差不超过1步。各任务的枢纽点准确率和条件重试成功率如下：

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

枢纽点正确时的重试成功率显著高于错误时，且准确诊断比例随训练推进持续上升，与Figure 6趋势一致。此外，R³L对反思错误具有内建鲁棒性，错误反思引导的重试若表现更差，会获得负优势并被RL目标主动抑制，辅助SFT仅在经过验证的成功修正上训练，进一步过滤低质量反思。


```
W3: The reflection step requires an extra inference pass and auxiliary supervision. The paper does not provide a detailed wall-clock or cost analysis comparing R³L with strong baselines under equal compute budgets.
```

感谢您指出这一不足，论文确实缺少详细的开销分析。首先需要澄清一个事实，反思token通过context distillation在训练前被显式移除，进入RL训练的轨迹与基线结构完全一致，R³L的额外推理开销仅存在于rollout端。训练端除RL目标外多一项辅助SFT loss，数据来自rollout阶段已收集轨迹中经过奖励验证的子集，不引入额外数据收集或推理，SFT仅在反思JSON和从枢纽点起的重试后缀上计算梯度，序列远短于完整轨迹，且与RL目标共享同一模型前向传播，额外的计算开销很小。

推理端的实际开销也有多个因素抵消。重试由模型根据反思结果自主判断是否触发，训练后期随着模型能力提升需要重试的比例自然下降，推理开销自适应递减。同时R³L因成功率更高，成功轨迹通常比失败轨迹更短，平均turn数和回复长度反而低于基线，部分抵消了反思带来的额外token。

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
| Avg base turns/traj | 1.09 | 1.06 | 1.08 | 1.11 |
| Avg base prompt tokens/turn | 691.2 | 698.4 | 693.8 | 697.6 |
| Avg base response tokens/turn | 3287 | 3208 | 3086 | 3671 |
| Avg reflect prompt tokens/traj | 无 | 4284 | 5872 | 6246 |
| Avg reflect response tokens/traj | 无 | 438 | 586 | 651 |
| Avg retry turns/traj | 无 | 1.04 | 1.07 | 1.08 |
| Avg retry prompt tokens/turn | 无 | 1076 | 1042 | 1168 |
| Avg retry response tokens/turn | 无 | 3186 | 3218 | 3452 |
| Total rollout turns/sample | 8.72 | 16.6 | 25.12 | 11.24 |
| Total training tokens/sample | 28,663 | 42,200 | 30,874 | 25,500 |
| Rollout time (s)/training step | 1148.0 | 1392.6 | 2318.4 | 773.5 |
| Train time (s)/training step | 121.3 | 146.8 | 129.6 | 225.8 |

表中tokens和turns指标对应单个sample（一个prompt的$N$条轨迹），time指标对应处理一个batch的wall-clock时间。所有实验使用两台8卡H20服务器，8卡用于rollout推理，8卡用于训练，两组GPU通过异步流水线并行工作。Training tokens仅统计response tokens，即模型生成的动作token，prompt和环境观测token不参与梯度计算，仅作为上下文输入前向传播。R³L的training tokens在多步交互任务上显著低于GRPO，因为GRPO对$N=8$条完整轨迹的全部response tokens计算梯度，而R³L的RL部分仅有$N/2=4$条base轨迹，其中约60%的失败轨迹会触发重试生成蒸馏轨迹$\mathcal{D}_{distill}$进入RL训练，重试中有增益的约1.5条才会保留其反思和重试轨迹用于辅助SFT，这部分token量较小。在ALFWorld等多步任务上base数量减半带来的节省非常显著，而在DAPO等单轮数学任务上每条轨迹本身较短，distill和SFT的额外开销使R³L的training tokens与GRPO接近。

wall-clock数据直接回应了您的关切。R³L每步rollout用时在全部四个设定下均低于纯GRPO基线，Qwen2.5-1.5B-Instruct上ALFWorld为372.4秒对406.2秒，DAPO为292.4秒对342.6秒，Qwen3-4B上ALFWorld为670.4秒对713.8秒，DAPO为773.5秒对1148.0秒。$N/2$ base的探索量减半、成功轨迹更短带来的平均turn数下降、以及枢纽点重启省去前缀推理，这些因素使R³L的rollout端净开销反而低于GRPO。Critique-GRPO因$2N$条完整轨迹加$N$次critique生成，rollout时间约为GRPO的1.8到2.0倍。训练端R³L因辅助SFT有所增加，在多步任务上增幅较小，在DAPO等单轮任务上由于base轨迹本身较短反思和SFT的固定开销占比更大，但R³L同时移除了参考模型前向传播开销，综合rollout和训练两端R³L的总wall-clock时间与GRPO接近。

感谢您的细致审阅与建议，我们会将以上补充实验和分析体现在修订版中。
