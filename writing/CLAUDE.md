# CLAUDE.md - 论文修改指南

## 论文基本信息

**标题（已确定）**：
> LEPA: Language-Feedback Guided Exploration with Pivotal Credit and Positive Amplification

**缩写对应**：
- **L** = Language-Feedback
- **E** = Exploration
- **P** = Pivotal Credit
- **A** = Positive Amplification

**投稿目标**：ACL 格式

**核心文件**：
- 主文件：`acl_latex.tex`
- 参考文献：`custom.bib`
- 图表目录：`figure/`
- 数据目录：`draw/`

---

## 核心贡献（三组件）

| 组件 | 原名称 | 新名称（统一） | 对应 LEAP 字母 |
|------|--------|----------------|----------------|
| 语言引导探索 | Language-Guided Reflect-Then-Retry | Language-Feedback Guided Exploration | L, E |
| 关键点信用分配 | Pivotal Credit Assignment | Pivotal Credit | P |
| 正向放大 | Positive Preference Optimization (PPO) | Positive Amplification | A |

**注意**：原 "Positive Preference Optimization" 缩写 PPO 与 Proximal Policy Optimization 冲突，已改为 "Positive Amplification"。

---

## 故事线定位

**核心框架**：探索（Exploration）vs 利用（Exploitation）

**问题分类（两点）**：
- **探索**：随机采样成功率低 + 需要重复完整 rollout 成本高
- **利用**：scalar rewards 无法诊断错误 + trajectory-level credit 惩罚整条轨迹 + 失败主导导致训练不稳定

**方法分类（三点）**：
1. **Language-Feedback Guided Exploration**：从随机采样到主动合成，利用环境语言反馈诊断错误
2. **Pivotal Credit Assignment**：基于诊断定位，只更新修正后的后缀，保护有效前缀
3. **Positive Amplification**：放大正向信号，锚定优化方向，稳定 off-policy 训练

**核心差异化**：
1. **Language-Feedback**：使用语言形式的环境反馈（非 self-reflection），区别于纯 scalar reward
2. **主动合成 vs 被动采样**：不是等待随机成功，而是主动把失败变成成功
3. **不强绑 Agent**：方法适用于 agentic tasks 和 math reasoning，不在标题中限定

---

## 待修改清单

### 高优先级
- [x] **Abstract**：已完成，围绕探索/利用框架重写
- [ ] **Introduction**：调整叙事框架，突出 language-feedback 差异化
- [ ] **全文替换**：R³L → LEPA
- [ ] **方法章节**：Section 4.4 标题从 "Positive Preference Optimization" 改为 "Positive Amplification"

### 中优先级
- [ ] **数据补全**：`exp-main-result.tex` 中 Critique-GRPO 行的 `0.xx`
- [ ] **数据补全**：`sync_table.tex` 中的 `xx` 占位符

### 低优先级
- [ ] **图表引用检查**：`fig:case_studies`, `fig:feedback_ablation` 需确认定义
- [ ] **术语一致性**：全文检查 PPO → Positive Amplification 的一致性

---

## 术语规范

| 避免使用 | 推荐使用 | 原因 |
|----------|----------|------|
| R³L | LEPA | 新标题 |
| Positive Preference Optimization | Positive Amplification | 避免 PPO 缩写冲突 |
| LLM Agent | Agentic LLMs / 不提 | 表达更自然 |
| guided retry | language-feedback guided | 强调语言形式 |
| self-reflection | environment-feedback reflection | 强调依赖外部反馈 |

---

## 文件结构

```
writing/
├── acl_latex.tex          # 主论文文件
├── acl.sty                # ACL 样式文件
├── acl_natbib.bst         # 参考文献样式
├── custom.bib             # 参考文献
├── figure/                # 图表 LaTeX 文件
│   ├── framework.tex      # 框架图
│   ├── exp-main-result.tex # 主实验表格
│   ├── ablation.tex       # 消融实验
│   ├── algorithm.tex      # 算法伪代码
│   ├── implementation.tex # 实现细节
│   └── ...
├── draw/                  # 绘图数据文件
└── 参考论文/               # 参考文献原文
```

---

## 实验基准

**Agentic Environments**：
- ALFWorld（embodied decision-making）
- WebShop（online navigation）
- ScienceWorld（long-horizon scientific reasoning）

**Mathematical Reasoning**：
- GSM8K, Math500, MinervaMath, OlympiadBench, AMC23, DAPO

**模型**：
- Qwen2.5-1.5B-Instruct
- Qwen2.5-7B-Instruct
- Llama-3.2-3B-Instruct

---

## 写作风格与标准

### 核心原则

1. **每句话要有信息密度**：避免空泛词汇如 "fundamental challenges"、"essential"、"powerful"
2. **逻辑链条要清晰**：不能堆砌名词，要深挖因果关系
3. **每个组件描述要本质**：说清楚为什么要这样做、有什么好处、效果是什么
4. **对比要鲜明**：如 "从随机采样到主动合成" 这种对比很本质
5. **好处必须明确说出**：不能只说做了什么，要说清楚为什么这样做有用
6. **后果必须明确说出**：描述问题时，不能只说现象，要说明这个现象导致的后果（如"从头开始"→"成本高"）

### 句式规范

- **不要用括号**：如 "(ALFWorld, WebShop)" 改为自然的表达
- **不要用冒号引导解释**：如 "LEPA employs X: doing Y" 改为 "LEPA employs X that does Y"
- **不要用分号**：将分号连接的句子改写为独立句或用逗号连接
- **避免奇怪的搭配**：如 "alongside errors"、"costly retries from scratch"
- **每个组件只能一句话**：但这一句话要够本质、够完整
- **多个条件用一句话整合**：用 "Since X and Y, Z does A, achieving B" 的结构

### 术语使用

- **不要提前使用未铺垫的概念**：如 "retry costs" 在背景部分出现太突兀，因为 retry 是我们的方案
- **概念要有承接**：如 "failure points" 需要先在组件1中铺垫，组件2才能用
- **分类要准确**：scalar rewards 的问题属于 exploitation 而非 exploration
- **抽象术语要具体化**：如 "credit noise" 太抽象，应改为 "undeserved penalties on valid prefixes"
- **突兀的术语需要解释或替换**：如 "training groups" 没有铺垫就不要用

### 逻辑链条规范

- **独立逻辑线不能混淆**：off-policy 和 failure-dominated 是两条独立逻辑线，不能用 "where" 暗示因果
- **因果关系要准确**：如 "off-policy data where failures dominate" 是错误的，应该用 "and" 并列
- **问题描述要准确**：GRPO 不是 "只抑制错误"，而是 "正向信号被稀释"（failure-dominated 时）
- **机制要具体**：不能只说 "amplify"，要说 "reweights to amplify"

### 组件描述标准

- **组件1（Language-Feedback Guided Exploration）**：
  - 说清楚"从什么到什么"的转变（stochastic sampling → active synthesis）
  - 说明 language feedback 的作用（diagnose errors）
  - 说明效果（transform failed attempts into successful ones）

- **组件2（Pivotal Credit Assignment）**：
  - 要承接组件1（"With errors diagnosed and localized"）
  - 说清楚做什么（updates only the corrected suffix）
  - 两个好处都要明确：
    1. 保护前缀（preserving valid prefixes from undeserved penalties）
    2. 减少成本（eliminating the need to regenerate entire trajectories）

- **组件3（Positive Amplification）**：
  - 动机：两条独立逻辑线（off-policy data + failures dominate on difficult tasks）
  - 问题：正向信号被稀释（diluting positive signals）
  - 机制：通过 reweight 放大（reweights to amplify successful ones）
  - 效果：锚定方向 + 稳定训练（concentrating probability mass toward discovered solutions and stabilizing training）

### 数据表达

- 使用具体数值：如 "5% to 52%" 而非 "substantial improvement"

### 已确定的 Abstract

```
Reinforcement fine-tuning has emerged as a powerful technique for enhancing reasoning and agentic capabilities in large language models. However, current approaches struggle with both exploration and exploitation. Exploration suffers from low success rates under stochastic sampling, requiring repeated full rollouts to discover successful trajectories. Exploitation is hindered by uninformative scalar rewards that signal correctness but not causes of failure, trajectory-level credit that penalizes entire sequences for single errors, and failure-dominated gradients that destabilize training. To this end, we propose LEPA, Language-Feedback Guided Exploration with Pivotal Credit and Positive Amplification. To synthesize high-quality trajectories, LEPA shifts from stochastic sampling to active synthesis via reflect-then-retry, leveraging language feedback from the environment to diagnose errors and transform failed attempts into successful ones. With errors diagnosed and localized, Pivotal Credit Assignment updates only the corrected suffix, preserving valid prefixes from undeserved penalties and eliminating the need to regenerate entire trajectories. Since reflect-then-retry produces off-policy data and failures dominate on difficult tasks, diluting positive signals, Positive Amplification reweights to amplify successful ones, concentrating probability mass toward discovered solutions and stabilizing training. Experiments on agentic and reasoning tasks demonstrate 5% to 52% relative improvements over baselines while maintaining training stability.
```
