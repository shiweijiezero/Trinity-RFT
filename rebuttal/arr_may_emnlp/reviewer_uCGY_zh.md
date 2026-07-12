# 审稿人 uCGY（中文翻译）

> 译注：OpenReview 导出的纯文本丢失了少量 LaTeX 符号。能够从论文和上下文唯一确定的符号（例如 R³L、α、mean±std）已恢复；无法唯一确定的数值或公式在下文标为“原文导出缺失”。

## 评分

- 置信度：4（相当确定）
- 可靠性（Soundness）：3（可接受）
- 兴奋度（Excitement）：3（有趣）
- 总体评价：3（Findings）
- 可复现性：3
- 数据集：2
- 软件：3

---

## 论文总结

R³L 是一种基于 GRPO、面向大语言模型推理和智能体任务的强化学习方案，针对作者识别出的三种失效模式：

1. **探索效率低下。** 在困难任务上，随机采样很少找到成功轨迹，而每次从头重新 rollout 的成本很高。
2. **信用分配粗糙。** 轨迹级奖励会因为后期错误而惩罚原本正确的前缀。
3. **失败样本占主导时训练不稳定。** 负样本压过少量正样本，使概率质量分散，并造成作者所谓的“熵坍塌”。

三个组件分别对应一种失效模式：

1. **语言引导的 Reflect-then-Retry。** 对每条基础轨迹，模型生成结构化反思，包括结果分类、根因分析、改进建议以及问题首次出现的 pivot turn $k_{pivot}$；随后，在 guidance 条件下从 $k_{pivot}$ 重新开始生成，以合成修正后的后缀。“上下文蒸馏”步骤将原始前缀与修正后缀组合成训练输入，同时移除 guidance，使修正能力能够迁移到没有 guidance 的推理阶段。方法还使用两个辅助 SFT 任务（学会反思和学会重试），并只在经过验证的成功修正上维护这些任务。
2. **Pivotal Credit Assignment。** 基础轨迹与重试轨迹在 $k_{pivot}$ 之前共享相同前缀，因此共享前缀不包含对比信号。二值 mask 将 pivot 之前所有 turn 的梯度置零，只更新产生分歧的后缀。
3. **Positive Amplification。** 使用单个系数 α>1（α=3.0）放大正 advantage，使建设性梯度占主导。作者还从 GRPO 中同时移除了 KL penalty 和 importance-sampling clipping，理由是 amplification 本身能够防止熵坍塌，而 importance sampling 对由 guidance 生成的 retry 数据并不可靠。

总预算仍为 N：其中 N/2 用于基础采样，另外 N/2 用于条件触发的局部 retry；只有当反思认为基础轨迹未成功时才触发 retry。

实验覆盖三个主干模型 Qwen2.5-1.5B-Instruct、Qwen2.5-7B-Instruct 和 Qwen3-4B，并在 Llama-3.2-3B-Instruct 上进行跨架构检查。智能体环境包括 ALFWorld、WebShop 和 ScienceWorld；数学推理在 DAPO 上训练，并在 GSM8K、Math500、MinervaMath、OlympiadBench、AMC23 和 DAPO test 上评测。Baseline 包括 RAFT、OPMD、GRPO、GSPO、Reflect-GRPO（作者对 Reflect-Retry-Reward 的复现）以及 Critique-GRPO，均在 Trinity-RFT 框架中复现。

论文的主要结论是：R³L 在全部 27 个“backbone×benchmark”设置中均为第一或第二，在 9 个智能体设置中均为第一；相对 baseline 提升 5%–52%，同时 rollout 时间更低，多步任务上的训练 token 约减半（表 9–10）。附录 A 给出了非正式理论论证，包括熵坍塌分解、关于 α 的梯度主导条件、前缀 masking 带来的方差降低界，以及局部收敛分析草图。代码已通过匿名仓库发布。

审稿人希望提醒作者注意一种可能的理解偏差：审稿人将数学任务的 guidance 理解为依据与 ground-truth answer 的比较而生成（附录 K.3），而不是来自环境错误消息。如果这一理解不正确，作者应当澄清，因为这会影响与 baseline 比较的公平性（见 W4）。

## 优点总结

### S1. 问题拆解清晰，与解决组件一一对应

三个失效模式——探索效率、信用分配粗糙、失败样本主导时的不稳定——动机充分，并分别清晰对应一个机制。论文容易理解，各项设计本身也都合理。

### S2. 实证覆盖广泛且结果一致

结果覆盖三个 backbone、一个跨架构检查、三个智能体环境、六个数学 benchmark、六个 baseline，以及多个消融维度，包括表 2 的组件移除、表 5 的 amplification factor、表 6 的同步频率、表 7 的 retry 次数和表 8 的 group size。R³L 在全部 9 个智能体设置中排名第一，并在全部 27 个设置中排名第一或第二。即便考虑 W1 中关于方差的保留意见，如此广泛实验网格上的一致性仍然是一个真实信号。

### S3. 计算效率是真正重要的优势，而不是附带结果

表 9–10 显示，R³L 在四个测量设置中的 rollout 时间均最低。例如，Qwen3-4B 在 DAPO 上为 773.5 秒/step，而 GRPO 为 1148.0，Critique-GRPO 为 2318.4。R³L 在多步任务上的训练 token 约减半，主要得益于基于 pivot 的局部 retry 以及移除了 reference model 的 forward pass。一个同时更准确且更便宜的方法更有可能得到实际采用。

### S4. 消融实验提供了真实洞见，而不只是“每个组件都有帮助”

表 7 的 retry 次数研究表明，一次 retry 已捕获大部分收益；在固定预算下，更多 retry 反而有害。作者给出了合理解释：更多 retry 会减少不同基础轨迹的数量，而且单次反思不会跨 retry 累积。“仅 Positive Amplification 能否达到完整 R³L”的分解也很有价值：ALFWorld 上 GRPO+PA 为 0.807，GRPO 为 0.720，完整 R³L 为 0.928，因此 amplification 约贡献了总提升的 40%。这正是审稿人希望看到的归因分析。

## 缺点总结

### W1. 只有单次运行，没有方差估计，直接削弱了稳定性主张

表 1 和消融表看起来都只报告了单次运行，没有多个 seed、误差条或显著性检验。RL fine-tuning 本身具有高方差，论文也承认这一点，而“稳定性”又是论文的核心贡献，因此缺少跨 seed 方差是最严重的问题。许多“胜出”的差距很小（具体范围在原文导出时缺失），例如 ScienceWorld-7B 上 R³L 为 0.403、Critique-GRPO 为 0.388，数学表中也有若干类似列。单次运行不足以支持“在全部 27 个设置中最佳”的主张。至少应在主表或其代表性子集上报告 3 个以上 seed 的 mean±std；理想情况下还应提供 seed 方差曲线，以实证支持稳定性，而不能只依赖单个任务上的图 4。

### W2. 理论部分与实际训练联系较弱，应作为直觉而不是保证

定理 3 的梯度主导条件假设正负样本的梯度范数具有可比性（具体公式在原文导出时缺失）。推论 4 声称 α=3“覆盖最实际的范围”，但这一说法依赖若干范围假设，包括 $|\bar{A}_{-}|/\bar{A}_{+}\in[1.0,2.0]$、$p_{retry}>0$，以及训练过程中的正样本比例和 advantage ratio。作者应报告实际训练期间的这些量，或者弱化理论表述。

### W3. 同时移除 KL regularization 和 importance sampling 是很强的改动，但证据不足

作者的理由是 Positive Amplification 能防止熵坍塌，而且 importance sampling 对 guidance 生成的 retry 数据不可靠。然而，实证支持只有单个任务 ALFWorld 上的图 4，其中展示了 KL 和梯度范数仍受控制。移除标准 off-policy safeguard 恰恰是最可能出现不稳定的地方，因此单任务稳定性曲线加非正式理论并不充分。建议至少在 WebShop 和一个数学设置上提供 KL、gradient norm 和 clip fraction 曲线，并最好增加重新加入 KL 或 IS 的消融，以分别说明移除它们会付出或节省什么。

### W4. 数学 retry guidance 可能存在公平性或泄漏问题，而且 pivot 定位准确率有限

数学任务没有环境错误消息，因此 retry guidance 是“根据与 ground truth 的比较生成的”（附录 K.3）；作者称它不会泄露答案，而只会“指出错误类型”。GRPO、GSPO、RAFT 等采样 baseline 并没有获得这种机制。因此，必须证明 R³L 的数学收益并非部分来自轨迹合成阶段对 ground-truth-derived hint 的特权访问。

与此相关，pivot localization 的 oracle agreement 只能算中等水平：表 11 中根据任务和训练 step 的不同为 43%–76%。对于 ALFWorld 和 WebShop，oracle 本身来自人工判断和 DeepSeek 对 100 条轨迹的标注，这也是相对宽松的 ground truth。请澄清：

1. Baseline 是否获得任何可比的、以 ground truth 为条件的信号？
2. 结果对 pivot localization error 有多敏感？

### W5. 对 baseline 低于 zero-shot 的情况，需要直接解释复现结果

某些复现 baseline 看起来弱得不太合理。例如表 13 中，Qwen2.5-7B 在 GSM8K 上的 Critique-GRPO 为 0.678，低于 GRPO 的 0.846，也低于 zero-shot reference 的 0.853；Reflect-GRPO 也有类似下降。一个 baseline 训练后低于自身 zero-shot 起点，意味着它可能没有充分调参，或者存在某种具体的不良交互。如果 baseline 调参不足，那么“27/27 最佳”的比较就被夸大了。附录 J 从总体上解释了跨分布退化，但仍应针对这些 baseline 低于 zero-shot 的具体行为给出解释，并明确说明为 baseline 投入了多少超参数调优工作。

### W6. 核心 amplification 规则确实存在定义不一致，并非渲染问题

公式 11 将最大回报轨迹的 amplified advantage 定义为 α，而表 5 caption 写的是“所有配置都将取得最大回报的轨迹 advantage 设为 1”；附录 K.3 也写成最大回报轨迹取 1、其他正 advantage 轨迹取 $\alpha A$。当 α=3 时，组内最强正信号会相差 3 倍，这并不是小差异。作者必须统一说明：最大回报轨迹得到的是 1 还是 α？这同时影响可复现性和对 amplification 机制的解释。

### W7. 表述略带宣传性，而且消融实验存在组件耦合

摘要以“相对提升 5%–52%”开头，其中 52% 来自 1.5B 模型在 GSM8K 上相对 GRPO 的提升；但该设置也是唯一一个更强 baseline Critique-GRPO（0.798）超过 R³L（0.721）的 benchmark。在 R³L 输给更强 baseline 的唯一 benchmark 上，挑选相对较弱 baseline 得到的最大提升作为开场数字，会产生误导。应报告相对最强 baseline 的典型提升。

此外，表 2 中的 “w/o Reflect” 必然同时关闭 Pivotal Credit，因为 Credit 依赖反思所识别的 pivot。因此这一行混合了两个组件的移除，无法单独识别 reflection 的贡献。作者虽然承认了这一点，但这限制了消融结论。

### W8. 影响范围受实验设定限制

所有 backbone 都不超过 7B，所有任务都有可验证奖励。论文关于“适用于任何存在 preference signal 的领域”的一般性主张尚未经过检验；开放式或主观奖励领域明确不在实验范围内。Limitations 已经承认这一点，但它仍然限制了贡献范围。此外，该方法依赖基础模型产生有用自我反思的能力，而论文也承认 1.5B 模型存在 cold-start，这表明该方案未必能顺利迁移到 reflection quality 较低的设置。

## 评论、建议与拼写问题

1. **统计报告（对应 W1）。** 即使只在主表的一小部分设置上增加 seed study，也会显著加强论文；这是 rebuttal 中价值最高的一项补充。
2. **统一 amplification 规则（对应 W6）。** 无论最终评审结果如何，camera-ready 都必须修正这一问题，因为读者一定会遇到这一正确性和清晰度问题。
3. **建议审稿人直接查看的图片。** 图 1（第 2 页，使用彩色 step 和概率变化展示 GRPO 与 R³L 的核心概念）、图 4（第 11 页，六个 panel 的稳定性证据）、图 5（第 20 页，四类轨迹）以及图 6（第 20 页，pivot point 漂移）。这些图包含纯文本层中缺失的信息。
4. **报告 retry trigger rate。** 建议报告触发 retry 的基础轨迹比例，因为它决定了 R³L 与其他方法之间的真实 rollout 成本，并能使表 9 的计算比较更加具体。
5. **拼写。** 投稿 checklist 中的 “voilate” 应为 “violate”，“not human-related data” 应为 “no human-related data”。建议进行一次通篇检查。
6. 建议在正文首次出现 “entropy collapse” 时就给出精确定义，而不是只在附录 A.2.1 中定义。
7. 将 Reflect-Retry-Reward 的复现称作 “Reflect-GRPO” 是合理的，但容易与 Critique-GRPO 混淆。建议首次出现时用一句话说明它是对 Bensal 等人方法的复现。

## 局限性与社会影响

论文对此讨论充分。Limitations 部分坦诚而具体：reflection 会增加一次 inference pass，但 pivot mechanism 能抵消 rollout 成本；1.5B 模型存在 cold-start；实验范围限于可验证奖励任务。这种坦诚应得到肯定，而不是惩罚。对于这项强化学习训练方法研究，没有社会影响方面的担忧。

## 伦理问题

本投稿不存在伦理方面的担忧。

是否需要伦理审查：否。

## 其他字段

- 是否知道或推测作者身份：否。
- 是否从外部来源了解本文：不适用；审稿人未从外部来源了解本文。
- 是否知道论文来源：不适用；审稿人未从外部来源了解论文来源。
- 上述知识是否影响评审：不适用；审稿人未从外部来源了解本文。
- 审稿人认证：审稿人确认其评审准确反映了本人对该工作的评价。若使用了任何自动化工具，其用途仅限于改善语法和文风，评审实质内容来自审稿人本人或已注明的第二审稿人。
- 出版伦理政策合规：审稿人仅在 PEC 政策允许的场景中使用了保护隐私的工具，例如语言编辑。
