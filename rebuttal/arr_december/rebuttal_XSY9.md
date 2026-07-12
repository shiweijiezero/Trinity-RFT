# Rebuttal to Reviewer XSY9

感谢您的详细评审，以及对有效前缀惩罚问题的认可和对枢纽点对比信用分配机制的肯定。以下是我们的回复和补充实验。


```
W1 & S1: The reported results for the baselines are significantly lower than that in Qwen2.5 technical report, casting doubt on the entire empirical evaluation. According to the Qwen2.5 technical report, Qwen2.5-1.5B-Instruct achieves 73.2 on GSM8K and 55.2 on MATH. However, the authors report a baseline (GRPO) of only 47.4 on GSM8K and 36.7 on MATH. ... & The authors must provide a detailed explanation for why their baseline numbers for Qwen2.5 are ~25-30 points lower (on GSM8K/MATH) than official figures.
```

感谢您指出这一容易产生困惑的问题，我们在论文中确实应该更清晰地说明。

Table 1中的数字是RL训练后的结果，不是预训练模型的zero-shot性能。以GSM8K上GRPO的0.474为例，这是从Qwen2.5-1.5B-Instruct出发在DAPO训练集上经过RL训练后、在<answer>评估协议下得到的分数。您引用的技术报告0.732则是同一模型在\boxed{}评估协议下未经RL训练的直接评估性能。两者在评估协议和训练状态两个维度上均不同，0.26的表面差距实际由格式差异和训练分布迁移两个独立因素叠加而成。

格式差异方面，Qwen2.5-Instruct在训练阶段对\boxed{}格式做了充分适配，技术报告中评估使用的system prompt为：

> Please reason step by step, and put your final answer within \boxed{}.

我们的训练和评估统一使用<think>/<answer>标签格式，这是DeepSeek-R1以来RL训练研究中广泛采用的推理格式，也是我们为统一agentic任务和数学推理任务而选择的协议，system prompt为：

> A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within \<think\> \</think\> and \<answer\> \</answer\> tags, respectively, i.e., \<think\> reasoning process here \</think\> \<answer\> answer here \</answer\>.

格式切换对benchmark分数的影响在先前工作中已有记录，[1]展示了答案提取规则在同一模型上可造成数十分的波动，[2]为规避格式解析问题将所有训练答案转为整数。

[1] Jo, Hwiyeol, et al. "Finding Answers in Thought Matters: Revisiting Evaluation on Large Language Models with Reasoning." arXiv preprint arXiv:2510.14773 (2025).
[2] Yu, Qiying, et al. "Dapo: An open-source llm reinforcement learning system at scale." arXiv preprint arXiv:2503.14476 (2025).

为严格量化我们场景下格式差异的实际影响，我们在四个基础模型上分别用两种协议进行了zero-shot评估。所有评估在完整测试集上进行，GSM8K为openai/gsm8k全部1319道，Math500为500题，MinervaMath为272题，OlympiadBench为674题，AMC23为40题，DAPO为300题。表中数字均为我们在相同评估脚本下自行评估的结果。

Qwen2.5-1.5B-Instruct：

| Benchmark | \boxed{} | <answer> | Gap |
|---|---|---|---|
| GSM8K | 0.738 | 0.665 | -0.073 |
| Math500 | 0.440 | 0.422 | -0.018 |
| MinervaMath | 0.118 | 0.132 | +0.015 |
| OlympiadBench | 0.099 | 0.083 | -0.016 |
| AMC23 | 0.300 | 0.150 | -0.150 |
| DAPO | 0.133 | 0.107 | -0.027 |

Qwen2.5-7B-Instruct：

| Benchmark | \boxed{} | <answer> | Gap |
|---|---|---|---|
| GSM8K | 0.923 | 0.853 | -0.070 |
| Math500 | 0.620 | 0.612 | -0.008 |
| MinervaMath | 0.239 | 0.235 | -0.004 |
| OlympiadBench | 0.205 | 0.186 | -0.019 |
| AMC23 | 0.575 | 0.575 | 0 |
| DAPO | 0.337 | 0.337 | 0 |

Llama-3.2-3B-Instruct：

| Benchmark | \boxed{} | <answer> | Gap |
|---|---|---|---|
| GSM8K | 0.723 | 0.736 | +0.013 |
| Math500 | 0.366 | 0.380 | +0.014 |
| MinervaMath | 0.114 | 0.099 | -0.015 |
| OlympiadBench | 0.144 | 0.122 | -0.022 |
| AMC23 | 0.200 | 0.175 | -0.025 |
| DAPO | 0.137 | 0.130 | -0.007 |

Qwen3-4B：

| Benchmark | \boxed{} | <answer> | Gap |
|---|---|---|---|
| GSM8K | 0.932 | 0.938 | +0.006 |
| Math500 | 0.648 | 0.690 | +0.042 |
| MinervaMath | 0.254 | 0.283 | +0.029 |
| OlympiadBench | 0.432 | 0.504 | +0.073 |
| AMC23 | 0.625 | 0.600 | -0.025 |
| DAPO | 0.446 | 0.443 | -0.003 |

格式差异主要集中在Qwen2.5系列的GSM8K上，1.5B和7B分别降0.073和0.070，Math500不到0.020，其余benchmark在±0.020以内。AMC23仅40题，1.5B上-0.150的波动属于小样本噪声，7B上差异为0。Llama-3.2-3B在两种格式下几乎一致，GSM8K在<answer>下反而略高0.013，说明格式敏感性是Qwen2.5在训练阶段对\boxed{}格式充分适配后的特异性现象，并非通用问题。Qwen3-4B在<answer>格式下多个benchmark更好，OlympiadBench高0.073，Math500高0.042，与Qwen3本身采用<think>/<answer>格式训练一致。

以您关注的Qwen2.5-1.5B-Instruct GSM8K为例分解这0.26的差距。我们的\boxed{}评估得0.738，与技术报告0.732一致，切换<answer>格式降至0.665，格式差异约0.073，占总差距的28%。剩余72%来自训练分布迁移：Table 1中所有方法在DAPO训练集上训练，DAPO的题目分布与GSM8K不同，GRPO训练后GSM8K从0.665降至0.474，降了0.191。7B模型上两个因素的影响都更小，R³L在<answer>下GSM8K达0.897，\boxed{}评估为0.923，差距仅0.026。

GSM8K作为预训练语料中被广泛覆盖的经典benchmark，模型对其形成了与特定格式和数据绑定的解题模式，在格式切换和训练分布变化时尤其脆弱。竞赛级benchmark依赖推理能力而非记忆模式，受两种变化的影响都很小，这与上面四个模型的实测数据一致。

为直接展示训练分布迁移的影响，我们列出Qwen2.5-1.5B-Instruct在不同训练集上训练后在全部数学benchmark上的评估结果。

DAPO训练集训练后，<answer>协议评估：

| Method | GSM8K | Math500 | MinervaMath | Olympiad | AMC23 | DAPO |
|---|---|---|---|---|---|---|
| Zero-shot | 0.665 | 0.422 | 0.132 | 0.083 | 0.150 | 0.107 |
| GRPO | 0.474 | 0.368 | 0.099 | 0.114 | 0.250 | 0.123 |
| Reflect-GRPO | 0.672 | 0.376 | 0.102 | 0.130 | 0.300 | 0.136 |
| Critique-GRPO | 0.798 | 0.404 | 0.110 | 0.124 | 0.275 | 0.133 |
| R³L | 0.721 | 0.424 | 0.125 | 0.151 | 0.325 | 0.156 |

DAPO训练集训练后，\boxed{}协议评估：

| Method | GSM8K | Math500 | MinervaMath | Olympiad | AMC23 | DAPO |
|---|---|---|---|---|-------|---|
| Zero-shot | 0.738 | 0.440 | 0.118 | 0.099 | 0.300 | 0.133 |
| GRPO | 0.576 | 0.392 | 0.092 | 0.125 | 0.400 | 0.147 |
| Reflect-GRPO | 0.742 | 0.404 | 0.096 | 0.140 | 0.375 | 0.160 |
| Critique-GRPO | 0.793 | 0.428 | 0.103 | 0.135 | 0.375 | 0.153 |
| R³L | 0.788 | 0.446 | 0.114 | 0.162 | 0.425 | 0.173 |

Qwen2.5-7B-Instruct在DAPO训练集训练后，<answer>协议评估：

| Method | GSM8K | Math500 | MinervaMath | Olympiad | AMC23 | DAPO |
|---|---|---|---|---|---|---|
| Zero-shot | 0.853 | 0.612 | 0.235 | 0.186 | 0.575 | 0.337 |
| GRPO | 0.846 | 0.572 | 0.239 | 0.277 | 0.675 | 0.393 |
| Reflect-GRPO | 0.765 | 0.532 | 0.194 | 0.250 | 0.550 | 0.396 |
| Critique-GRPO | 0.678 | 0.522 | 0.152 | 0.170 | 0.300 | 0.390 |
| R³L | 0.897 | 0.658 | 0.275 | 0.301 | 0.700 | 0.436 |

Qwen2.5-7B-Instruct在DAPO训练集训练后，\boxed{}协议评估：

| Method | GSM8K | Math500 | MinervaMath | Olympiad | AMC23 | DAPO |
|---|---|---|---|---|---|---|
| Zero-shot | 0.923 | 0.620 | 0.239 | 0.205 | 0.575 | 0.337 |
| GRPO | 0.915 | 0.584 | 0.246 | 0.285 | 0.650 | 0.383 |
| Reflect-GRPO | 0.846 | 0.544 | 0.162 | 0.195 | 0.350 | 0.392 |
| Critique-GRPO | 0.872 | 0.550 | 0.205 | 0.264 | 0.550 | 0.407 |
| R³L | 0.943 | 0.658 | 0.276 | 0.305 | 0.675 | 0.424 |

两个模型在DAPO训练后的<answer>结果呈现出一致的分布迁移模式。DAPO作为训练集的同域benchmark，所有方法都稳定提升，R³L在1.5B上DAPO从0.107涨到0.156，7B上从0.337涨到0.436。与DAPO题目分布接近的竞赛级benchmark也普遍受益，Olympiad在1.5B上从0.083提升到0.151，7B上从0.186提升到0.301，AMC23在1.5B上从0.150提升到0.325，7B上从0.575提升到0.700。而与DAPO分布较远的GSM8K和Math500则出现了不同程度的衰退，尤其是1.5B上GRPO在GSM8K从0.665降至0.474。

不同方法抵抗跨域遗忘的能力差异显著。R³L在两个模型上几乎所有benchmark都超过或持平于zero-shot，1.5B上GSM8K从zero-shot的0.665提升到0.721，7B上GSM8K从0.853提升到0.897，Math500从0.612提升到0.658，说明反思重试机制在提升目标任务的同时有效缓解了跨域衰退。相反，标准GRPO在1.5B上GSM8K降了0.191，7B上Critique-GRPO的GSM8K从0.853降至0.678，AMC23从0.575降至0.300，出现了严重的能力遗忘。

训练分布迁移可通过对照实验直接验证：在GSM8K和MATH训练集上分别进行同域RL训练，评估对应测试集。配置见匿名仓库`examples/R3L/gsm8k/`和`examples/R3L/math/`，`*_qwen_1.5B.yaml`为\boxed{}协议，`*_1.5B.yaml`为<answer>协议。

在in-domain的训练设定下，我们也补充了GSM8K训练集训练后，GSM8K测试集评估：

| Method | <answer> | \boxed{} |
|---|---|---|
| Zero-shot | 0.665 | 0.738 |
| GRPO | 0.814 | 0.798 |
| Reflect-GRPO | 0.822 | 0.830 |
| Critique-GRPO | 0.846 | 0.842 |
| R³L | 0.867 | 0.874 |

MATH训练集训练后，Math500测试集评估：

| Method | <answer> | \boxed{} |
|---|---|---|
| Zero-shot | 0.422 | 0.440 |
| GRPO | 0.481 | 0.493 |
| Reflect-GRPO | 0.505 | 0.498 |
| Critique-GRPO | 0.518 | 0.512 |
| R³L | 0.533 | 0.530 |

同域训练下所有方法在对应benchmark上均相对zero-shot取得显著提升，两种评估协议下趋势完全一致。对比DAPO训练后GSM8K从0.665降至0.474、Math500从0.422降至0.368的情况，确认了Table 1中基线偏低的主因是跨域训练分布迁移而非方法缺陷。

以上格式对比评估可通过匿名仓库中的脚本复现：

`python eval_math/eval_format_comparison.py --model_path Qwen/Qwen2.5-1.5B-Instruct`
`python eval_math/eval_format_comparison.py --model_path Qwen/Qwen2.5-7B-Instruct`
`python eval_math/eval_format_comparison.py --model_path Qwen/Qwen3-4B`
`python eval_math/eval_format_comparison.py --model_path meta-llama/Llama-3.2-3B-Instruct`

RL训练配置见匿名仓库`examples/R3L/dapo/`、`examples/R3L/gsm8k/`和`examples/R3L/math/`，其中`*_qwen_1.5B.yaml`为\boxed{}协议，`*_1.5B.yaml`为<answer>协议。所有对比方法使用完全相同的训练和评估条件，方法间的相对比较公平且有意义。我们将在revised版本中补充完整的格式对比实验、同域训练对照实验以及详细的差距分解分析，在论文中明确说明评估协议差异及其影响，并讨论跨域训练带来的分布偏移和能力衰退问题，以及不同方法在抵抗遗忘方面的差异。

```
W2 & S2: The paper focuses on Qwen2.5 and Llama 3.2, which were released in 2024. In the current landscape of 2026, these models are considered legacy. ... & Providing results on Qwen3 or Llama 4 is necessary to confirm that the "Reflect-then-Retry" mechanism isn't just fixing flaws that have already been solved by more advanced base models.
```

我们理解您对模型时效性的关注。R³L解决的问题是模型无关的。有效前缀惩罚、失败主导下的熵崩塌、低效随机探索是RL训练的结构性挑战，不会因更新的基础模型而消失，更强的模型在其难度前沿任务上仍面临相同瓶颈。跨模型一致性提供了比单一新模型更强的泛化证据。我们在Qwen2.5-1.5B、Qwen2.5-7B、Llama-3.2-3B三个模型上27个设置中均取得最优或次优表现，且在7B这一较强基础模型上也有显著改进。

我们在Qwen3-4B上补充了实验。值得说明的是，Qwen3作为reasoning模型采用了动态上下文管理机制，在多轮对话中自动裁剪历史turn的`<think>`内容仅保留最近一轮的思考过程。这一设计与标准的多轮对话RL训练流程不兼容，因为后者需要将完整的多轮交互作为一条轨迹进行梯度计算。为此我们将多轮对话拆解为多组单轮对话分别训练，并在框架中实现了step-level GRPO等适配模块，配置见匿名仓库`examples/R3L/*/step_*_4B.yaml`，对应的workflow实现在`trinity/common/workflows/envs/R3L/*/step_*_workflow.py`。结果如下：

| Task | RAFT | GRPO | OPMD | GSPO | Reflect-GRPO | Critique-GRPO | R³L |
|---|---|---|---|---|---|---|---|
| ALFWorld | 0.886 | 0.912 | 0.878 | 0.916 | 0.938 | 0.942 | 0.962 |
| WebShop | 0.659 | 0.695 | 0.678 | 0.707 | 0.724 | 0.725 | 0.746 |
| ScienceWorld | 0.350 | 0.368 | 0.342 | 0.376 | 0.378 | 0.374 | 0.398 |
| GSM8K | 0.914 | 0.934 | 0.922 | 0.928 | 0.920 | 0.926 | 0.948 |
| MATH500 | 0.638 | 0.718 | 0.663 | 0.722 | 0.654 | 0.705 | 0.753 |
| MinervaMath | 0.326 | 0.350 | 0.358 | 0.364 | 0.366 | 0.368 | 0.383 |
| Olympiad | 0.516 | 0.546 | 0.492 | 0.538 | 0.552 | 0.544 | 0.571 |
| AMC23 | 0.650 | 0.725 | 0.675 | 0.800 | 0.650 | 0.775 | 0.800 |
| DAPO | 0.505 | 0.540 | 0.531 | 0.548 | 0.547 | 0.558 | 0.583 |

R³L在更先进的模型下仍有优异表现。Qwen3-4B作为reasoning模型在数学上的基础性能远超参数量更大的Qwen2.5-7B-Instruct，GRPO基线在MATH500上达到0.718而7B仅0.572，但R³L在此更高的基准上仍取得一致改进。值得注意的是Reflect-GRPO在agentic任务上表现突出而在数学任务上反而低于GRPO，MATH500仅0.654，AMC23仅0.650，这与Qwen2.5-7B上的趋势一致。对推理能力已经很强的模型，简单地追加反思和重试可能干扰原有的推理路径，而R³L通过枢纽点定位将反思转化为结构化的信用分配信号，在所有任务类型上均保持优势。四个模型覆盖1.5B到7B参数规模、Qwen和Llama两个架构系列、标准指令微调和reasoning训练两种范式，R³L的一致改进确认其解决的是RL训练的结构性瓶颈而非特定模型缺陷。

```
W3 & S3: The method introduces several "moving parts" (reflection, retry, guidance distillation, pivotal masking, and positive amplification). ... it appears to be a collection of heuristics added atop existing ideas (like [1,2]). ... The "Positive Amplification" (α) introduces an additional hyperparameter that likely requires task-specific tuning ... & Can similar gains be achieved by further refining the "Positive Amplification" on top of standard GRPO without the expensive reflection/retry loop?
```

R³L的组件确实不少，但这些组件并非在GRPO上逐层堆叠的独立启发式，而是围绕同一目标形成的完整流水线，在增加能力的同时移除了基线中的若干组件，净复杂度反而更低。

R³L的核心设计是将GRPO的$N$条随机探索预算重新分配为$N/2$条base探索加$\leq N/2$条定向重试，在不增加总轨迹数的前提下同时获得多样性和针对性。重试从枢纽点重启而非重新生成完整轨迹，每条retry严格短于base rollout。反思由模型自主判断是否需要触发重试，随着训练推进模型能力提升，需要重试的比例自然下降，探索开销自适应递减。反思和引导文本通过context distillation在进入RL训练前被移除，训练序列与基线结构完全一致。这意味着反思在探索阶段提供了更丰富的信号，但不污染训练数据，实现了探索端和训练端的关注点分离。

枢纽信用分配利用了反思重试产生的结构性对齐：base轨迹和retry轨迹在枢纽点之前共享完全相同的前缀，仅在分歧点之后的行动不同，构成了天然的控制实验。这种步级的对比信号只有在反思重试产生结构化轨迹对之后才可能获得，是整个流水线的关键产出而非独立添加的模块。Positive Amplification解决稀疏奖励下成功轨迹被淹没的问题，同时移除了KL正则化和重要性采样裁剪，省去了参考模型的前向传播。三个机制环环相扣，Table 2消融确认了各自独立且可叠加的增益。

超参数方面，R³L移除了KL系数$\beta$和裁剪参数$\epsilon$，仅引入$\alpha=3.0$这一个超参数。GRPO需要$\beta$和$\epsilon$，GSPO需要$\beta$和自适应裁剪范围，见Table 7。Table 5确认单一固定的$\alpha=3.0$在全部模型和任务上稳健有效，无需逐任务调参，这有Corollary 4的理论支撑。

与您提到的Critique-GRPO[1]对比，两者都包含反思和重试，但设计理念和效率有本质区别。Critique-GRPO对$N$条base轨迹逐一生成critique再逐一重试，总共$2N$次rollout，训练时从$N$条refinement中只选最佳一条与原始$N$条合并，$N-1$条refinement的计算被浪费。R³L将预算控制在$\leq N$次rollout内，base和经过奖励验证的蒸馏轨迹全部进入RL group对比学习，没有计算浪费。更关键的区别在于信用分配：Critique-GRPO的critique是自然语言反馈文本，训练仍使用GRPO的轨迹级奖励信号，而R³L通过枢纽点对齐实现了步级信用分配，模型能精确学到在哪一步出错以及如何修正。VL-Rethinker[2]针对视觉语言模型设计，通过token级奖励激励模型在生成中插入反思token，应用场景和反思机制均不同。

关于单独使用Positive Amplification (PA)能否达到类似效果，Table 4中我们已在GRPO上单独加入PA进行了对比。GRPO加PA在ALFWorld上从标准GRPO的0.747提升到0.807，GSM8K上从0.474提升到0.504，PA对训练稳定性的改善是实在的。但仅凭PA无法接近R³L的0.928和0.721，差距仍然很大。Table 2消融中去掉反思重试后ALFWorld从0.928降到0.894，GSM8K从0.721降到0.562，反思重试提供了最大的单一贡献且不可替代。


```
W4 & S4: Flawed Compute-Efficiency Comparison: ... they do not seem to account for the increased sequence length (token count) incurred by generating natural language reflections and guidance. A true comparison of "Exploration Efficiency" should be conducted under a strict total token budget ... & Please include a table comparing the average tokens generated per training step for R³L vs. GRPO to ensure a fair comparison of exploration costs.
```

您提出应在严格的total token budget下对比，这一关注非常合理。R³L的rollout端将GRPO的$N$条探索预算重新分配为$N/2$条base加$\leq N/2$条重试，base数量减半直接降低了一半的探索生成量。所有$N/2$条base轨迹会进行一次反思生成，这是额外的推理开销，但反思是单次model调用生成结构化JSON，token量远小于一次完整环境交互。约60%的失败轨迹触发重试，重试从枢纽点重启而非从头生成完整轨迹，仅需生成后缀部分，前缀通过环境回放完成无需模型推理。反思和引导文本通过context distillation在进入RL训练前被显式移除，RL训练的序列结构与基线完全一致。训练端R³L的RL部分覆盖$N/2$条base轨迹和少量蒸馏轨迹$\mathcal{D}_{distill}$，此外多一项辅助SFT loss，数据来自rollout阶段已收集的轨迹中经过奖励验证的成功修正子集，SFT仅在反思JSON和从枢纽点起的重试后缀上计算梯度，序列远短于完整轨迹。R³L同时移除了KL正则化，省去了参考模型的前向传播开销。

各方法每步计算开销的结构对比如下：

| | GRPO | Reflect-GRPO | Critique-GRPO | R³L |
|---|---|---|---|---|
| Base rollout | $N$条完整轨迹 | $N$条完整轨迹 | $N$条完整轨迹 | $N/2$条完整轨迹 |
| Reflect | 无 | 轨迹内嵌反思token | $N$次独立critique生成 | $N/2$条反思，每条约500 tokens |
| Retry rollout | 无 | 轨迹内嵌重试 | $N$条完整refinement | $\leq N/2$条从枢纽点起的部分轨迹，仅反思判定需要时触发 |
| 总rollout数 | $N$ | $N$ | $2N$ | $\leq N$，取决于实际触发重试的比例 |
| 进入RL训练 | $N$条 | $N$条，反思token获奖励 | $N$条初始加$1$条最佳修正，其余$N-1$条refinement丢弃 | base加蒸馏轨迹，全部参与group对比 |
| 参考模型前向传播 | 需要 (KL) | 需要 (KL) | 需要 (KL) | 不需要 |

以下是逐环境、逐模型的实际token和时间开销。

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

从token budget角度看，Critique-GRPO的总token消耗最高，$2N$条完整轨迹加$N$次critique生成，且$N-1$条refinement被丢弃未进入训练，计算浪费严重。R³L的rollout端额外开销来自$N/2$次反思生成，但$N/2$ base带来的生成量减半和枢纽点重启带来的retry长度缩短显著抵消了这一开销。从wall-clock数据看，R³L的rollout时间在所有设置下均低于或接近GRPO，1.5B ALFWorld 372.4s对406.2s，4B DAPO 773.5s对1148.0s。训练端token等量，同时R³L移除了参考模型的前向传播。训练时间高于GRPO，在多步任务上增幅较小，在DAPO等单轮任务上由于base轨迹本身较短反思和SFT的固定开销占比更大。总体而言，R³L在与GRPO接近的wall-clock时间下达到了显著更高的性能。

我们再次感谢您提出的问题，希望这些回复可以解决您的疑惑，我们会将以上补充的实验和讨论整合到修订版论文中。
