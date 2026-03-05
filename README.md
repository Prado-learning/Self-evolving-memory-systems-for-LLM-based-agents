# AutoEvolve: Self-Evolving Memory Systems for LLM-based Agents

基于LLM的智能体自进化记忆系统研究项目

## 项目简介

本项目研究如何让AI智能体在交互过程中不仅积累经验，还能**自主优化其记忆系统本身**，进而通过记忆提升智能体的能力。

传统agent记忆系统（如简单的RAG或固定模板）是静态的，这带来两个问题：
- **探索效率低下**：过度依赖随机尝试
- **适应性差**：无法针对特定领域（如Coding vs. Research）动态调整记忆权重

本项目通过自我进化机制构建包含"自我提问-自我导航-自我归因"的闭环，实现agent在无人工标注数据情况下的能力迭代。

## 四篇核心论文

| 论文 | 核心贡献 | 进化层级 |
|------|---------|---------|
| **AgentEvolver** | Self-QNA框架（自我提问-导航-归因） | 行为层进化 |
| **MemEvolve** | 记忆架构元进化（双层进化框架） | 架构层进化 |
| **MemGen** | Trigger-Weaver机制，动态记忆生成 | 表示层进化 |
| **MemRL** | M-MDP + Q-learning，记忆效用估计 | 效用层进化 |

## 项目结构

```
AutoEvolve/
├── README.md                 # 本文件
├── requirements.txt          # Python依赖
├── docs/                     # 文档
│   ├── presentation.md       # 组会分享大纲
│   ├── comparison_analysis.md # 论文对比分析
│   └── PPT_Script.md         # 演讲脚本
├── src/                      # 源代码
│   ├── simple/               # 简化实现
│   │   ├── agentevolve.py    # AgentEvolver复现
│   │   └── memevolve.py      # MemEvolve复现
│   └── utils/                # 工具函数
├── experiments/              # 实验实现
│   └── memrl/                # MemRL 核心复现
├── scripts/memrl/            # MemRL 运行脚本
├── outputs/                  # 实验结果输出
├── tests/                    # 单元测试
└── papers/                   # 论文源文件（保留原目录）
    ├── agentevolver/
    ├── memevolve/
    ├── MemGen/
    └── MemRL/
```

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行简化实现

**AgentEvolver演示：**
```bash
python src/simple/agentevolve.py
```

演示Self-QNA三个核心模块：
- **Self-Questioning**: 好奇心驱动的任务生成
- **Self-Navigating**: 经验复用与混合策略指导
- **Self-Attributing**: 差异化奖励归因

**MemEvolve演示：**
```bash
python src/simple/memevolve.py
```

**MemRL 复现（MVP）：**
```bash
python scripts/memrl/run_memrl_comparison.py --steps 6000 --eval-episodes 1200 --seed 42
python -m pytest -q tests/memrl/test_memrl_smoke.py
```

演示记忆效用估计的 baseline vs Task-Specific Q-Networks 对比，详见 `docs/MemRL_Repro.md`。

MemEvolve 演示双层进化框架：
- **内层循环（一阶）**: 固定架构下积累经验
- **外层循环（二阶）**: 进化记忆架构本身（Encode/Store/Retrieve/Manage四组件）

## 核心概念

### 1. AgentEvolver: Self-QNA框架

```python
# 三个自模块协同工作
agent = SimpleAgentEvolver()
agent.questioning.generate_task(feedback)   # 生成任务
agent.navigating.get_action_guidance(task)  # 获取指导
agent.attributing.attribute_rewards(traj)   # 奖励归因
```

### 2. MemEvolve: 记忆架构元进化

```python
# 定义记忆架构基因型
arch = MemoryArchitecture(
    encode_strategy="insight",    # raw/insight/skill/api
    store_format="vector_db",     # json/vector_db/graph/hybrid
    retrieve_method="contrastive", # semantic/contrastive/graph_search/hybrid
    manage_policy="prune"         # none/prune/consolidate/dedup
)

# 双层进化
memevolve = MemEvolve()
best_arch = memevolve.evolve(task_batches)  # 进化出最优架构
```

### 3. 四篇论文的关联

```
AgentEvolver (学什么任务)
    ↓ 提供多样化任务
MemEvolve (怎么存记忆)
    ↓ 确定最优架构
MemGen (存什么内容)
    ↓ 动态记忆注入
MemRL (怎么用记忆)
    ↓ 反馈驱动更新
统一框架: 自进化记忆系统
```

## 实验计划

1. **基线实验**: 在AppWorld/BFCL等基准上测试简化实现
2. **消融实验**: 分析各组件对性能的影响
3. **融合实验**: 探索MemGen/MemRL与AgentEvolver的结合
4. **跨任务泛化**: 验证进化架构的可迁移性

## 文档索引

- [组会分享大纲](docs/presentation.md) - 四篇论文核心思想与关联分析
- [对比分析](docs/comparison_analysis.md) - 深度方法论对比
- [演讲脚本](docs/PPT_Script.md) - 30-40分钟组会演讲稿

## 参考文献

1. AgentEvolver: Towards Efficient Self-Evolving Agent System (Tongyi Lab, COLM 2025)
2. MemEvolve: Meta-Evolution of Agent Memory Systems (OPPO + NUS)
3. MemGen: Memory Consolidation and Generation for Autonomous Agents (NUS)
4. MemRL: Memory-based Reinforcement Learning for Tool-using Agents (SJTU + MemTensor)

## License

学术研究用途
