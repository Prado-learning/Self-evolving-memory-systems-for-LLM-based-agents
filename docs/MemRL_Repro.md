# MemRL 核心机制复现（MVP）

最后更新：2026-03-05（已迁移至 AutoEvolve）

## 1) 项目背景梳理

- MemRL 是 AutoEvolve 项目四篇核心论文之一，主题为「记忆效用估计」。
- `papers/memrl/` 存放论文源文件，本复现提供**最小可运行 MemRL 机制**，用于快速验证机制与创新点。

## 2) 机制定义

### Baseline（MemRL 最小机制）

- 状态：`(intent, experience)`
- 记忆：将 experience 检索到离散 memory slot
- 决策：`Q(intent, memory_slot)`
- 更新：单步 RL（contextual bandit 风格）

### 改进版（你的创新点）

- 原问题：全局 Q 不区分任务类型，易产生负迁移
- 改进：

\[
Q(intent, experience) \rightarrow Q(intent, experience, task\_type)
\]

- 实现：**Task-Specific Q-Networks**（MVP里用每个 `task_type` 一张独立 Q 表，对应独立 Q 网络的最简实现）

## 3) 代码落地

- `experiments/memrl/memrl_core.py`
  - 环境：`ToyMemRLEnv`
  - 记忆检索：`MemoryBank`
  - Baseline：`BaselineMemRLAgent`
  - 改进版：`TaskSpecificMemRLAgent`
  - 训练/评估函数

- `scripts/memrl/run_memrl_comparison.py`
  - 一键训练 baseline + 改进版
  - 输出对比指标到 `outputs/memrl_comparison.json`

- `tests/memrl/test_memrl_smoke.py`
  - smoke test：检查改进版成功率不低于 baseline

## 4) 如何运行

### 4.1 直接运行对比实验

```bash
cd <project_root>
python scripts/memrl/run_memrl_comparison.py --steps 6000 --eval-episodes 1200 --seed 42
```

输出：
- 控制台打印关键指标
- 结构化结果写入：`outputs/memrl_comparison.json`

### 4.2 运行测试

```bash
cd <project_root>
python -m pytest -q tests/memrl/test_memrl_smoke.py
```

## 5) 指标说明

- `success_rate`：评估集上动作命中最优动作的比例
- `retrieval_accuracy`：experience 检索到正确 memory slot 的比例
- `convergence_step`：达到目标滑窗奖励阈值的最早训练步

## 6) 预期现象

在当前环境设计中，最优动作显式依赖 `task_type`，因此通常会观察到：

- baseline（全局 Q）被多任务干扰，success_rate 较低
- task-specific 版本显著更高，且收敛更快

## 7) 局限与下一步

- 当前是机制验证级别（toy env），不是生产级大模型训练。
- 下一步可升级：
  1. 把 Q 表替换成 MLP/DQN（每 task_type 一套参数）
  2. 接入真实轨迹特征（替代 toy experience）
  3. 引入离线数据与 replay buffer
  4. 与 AgentEvolver/MemEvolve 等模块融合，构建统一自进化记忆框架
