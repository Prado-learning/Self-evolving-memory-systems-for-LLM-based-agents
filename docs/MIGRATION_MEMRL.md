# MemRL 文件迁移步骤

> 从 ParathinkVLA 迁移至 AutoEvolve 的完整记录  
> 执行日期：2026-03-05

## 一、迁移背景

MemRL（Memory-based Reinforcement Learning）是 AutoEvolve 项目四篇核心论文之一，主题为「记忆效用估计」。此前复现代码临时存放在 ParathinkVLA 仓库中，现迁移至 AutoEvolve 以统一项目结构。

## 二、迁移文件清单

| 源路径 (ParathinkVLA) | 目标路径 (AutoEvolve) |
|----------------------|------------------------|
| `experiments/memrl/memrl_core.py` | `experiments/memrl/memrl_core.py` |
| `scripts/memrl/run_memrl_comparison.py` | `scripts/memrl/run_memrl_comparison.py` |
| `docs/MemRL_Repro.md` | `docs/MemRL_Repro.md` |
| `tests/memrl/test_memrl_smoke.py` | `tests/memrl/test_memrl_smoke.py` |
| `outputs/memrl_comparison.json` | `outputs/memrl_comparison.json` |

## 三、迁移步骤（已执行）

### 步骤 1：创建目录结构

```bash
cd /home/user/lvhuanzhu/AutoEvolve
mkdir -p experiments/memrl
mkdir -p scripts/memrl
mkdir -p tests/memrl
mkdir -p outputs
```

### 步骤 2：复制核心文件

```bash
# 核心实现
cp /home/user/lvhuanzhu/ParathinkVLA/experiments/memrl/memrl_core.py experiments/memrl/

# 运行脚本
cp /home/user/lvhuanzhu/ParathinkVLA/scripts/memrl/run_memrl_comparison.py scripts/memrl/

# 文档
cp /home/user/lvhuanzhu/ParathinkVLA/docs/MemRL_Repro.md docs/

# 测试
cp /home/user/lvhuanzhu/ParathinkVLA/tests/memrl/test_memrl_smoke.py tests/memrl/

# 已有实验结果（可选）
cp /home/user/lvhuanzhu/ParathinkVLA/outputs/memrl_comparison.json outputs/
```

### 步骤 3：添加 __init__.py 以支持包导入

```bash
touch experiments/__init__.py
touch experiments/memrl/__init__.py
```

### 步骤 4：更新文档中的路径引用

- `docs/MemRL_Repro.md`：将示例命令中的 `ParathinkVLA` 改为 `AutoEvolve`
- `README_PARATHINK.md`（ParathinkVLA）：移除或改为指向 AutoEvolve 的说明

### 步骤 5：验证迁移

```bash
cd /home/user/lvhuanzhu/AutoEvolve
python scripts/memrl/run_memrl_comparison.py --steps 1000 --eval-episodes 200 --seed 42
python -m pytest -q tests/memrl/test_memrl_smoke.py
```

### 步骤 6：从 ParathinkVLA 删除已迁移文件（可选）

迁移验证通过后，可从 ParathinkVLA 删除：

```bash
rm -rf /home/user/lvhuanzhu/ParathinkVLA/experiments/memrl
rm -rf /home/user/lvhuanzhu/ParathinkVLA/scripts/memrl
rm /home/user/lvhuanzhu/ParathinkVLA/docs/MemRL_Repro.md
rm /home/user/lvhuanzhu/ParathinkVLA/tests/memrl/test_memrl_smoke.py
rm /home/user/lvhuanzhu/ParathinkVLA/outputs/memrl_comparison.json
# 删除空目录
rmdir /home/user/lvhuanzhu/ParathinkVLA/tests/memrl 2>/dev/null || true
```

## 四、导入路径说明

所有代码使用 `from experiments.memrl.memrl_core import ...`，需在 **AutoEvolve 项目根目录** 下运行脚本或测试，确保 `experiments` 包可被正确解析。

## 五、快速运行

```bash
cd /home/user/lvhuanzhu/AutoEvolve
python scripts/memrl/run_memrl_comparison.py --steps 6000 --eval-episodes 1200 --seed 42
python -m pytest -q tests/memrl/test_memrl_smoke.py
```

输出结果写入 `outputs/memrl_comparison.json`。
