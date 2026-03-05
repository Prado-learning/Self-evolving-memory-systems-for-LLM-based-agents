"""
Simple AgentEvolver - 简化版复现
基于论文: "AgentEvolver: Towards Efficient Self-Evolving Agent System"

核心机制:
1. Self-questioning: 好奇心驱动的任务生成
2. Self-navigating: 经验复用与混合策略
3. Self-attributing: 差异化奖励归因
"""

import random
import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class Experience:
    """经验单元"""
    task: str                  # 任务描述
    trajectory: List[Dict]     # 执行轨迹 (state, action, reward)
    success: bool              # 是否成功
    total_reward: float        # 累计奖励
    timestamp: str             # 时间戳


@dataclass
class Task:
    """任务定义"""
    id: str
    description: str
    difficulty: float          # 0-1
    domain: str                # 领域 (e.g., "coding", "reasoning")


class SelfQuestioningModule:
    """
    Self-Questioning Module
    基于好奇心驱动的任务生成
    """

    def __init__(self, curiosity_threshold: float = 0.3):
        self.curiosity_threshold = curiosity_threshold
        self.known_domains: set = set()
        self.task_history: List[Task] = []

    def generate_task(self, environment_feedback: Optional[Dict] = None) -> Task:
        """
        基于环境反馈和好奇心生成新任务

        简化逻辑:
        1. 如果探索不足 -> 生成新领域的任务 (高好奇心)
        2. 如果某领域失败率高 -> 生成该领域的变体任务
        3. 如果表现良好 -> 增加难度
        """
        task_id = f"task_{len(self.task_history)}_{datetime.now().strftime('%H%M%S')}"

        if environment_feedback is None:
            # 初始阶段：探索新领域
            domain = random.choice(["coding", "reasoning", "planning", "tool_use"])
            difficulty = 0.3
            description = self._generate_description(domain, difficulty)
        else:
            # 基于反馈调整
            success_rate = environment_feedback.get("success_rate", 0.5)
            recent_domain = environment_feedback.get("recent_domain", "coding")

            if success_rate > 0.7:
                # 表现好，增加难度
                difficulty = min(1.0, environment_feedback.get("difficulty", 0.5) + 0.2)
                domain = recent_domain
            elif success_rate < 0.3:
                # 表现差，生成相似任务练习
                difficulty = max(0.1, environment_feedback.get("difficulty", 0.5) - 0.1)
                domain = recent_domain
            else:
                # 探索新领域
                domain = random.choice([d for d in ["coding", "reasoning", "planning", "tool_use"]
                                       if d != recent_domain])
                difficulty = 0.4

            description = self._generate_description(domain, difficulty)

        task = Task(id=task_id, description=description, difficulty=difficulty, domain=domain)
        self.task_history.append(task)
        self.known_domains.add(domain)
        return task

    def _generate_description(self, domain: str, difficulty: float) -> str:
        """根据领域和难度生成任务描述"""
        templates = {
            "coding": [
                "Write a function to {task}",
                "Debug the following code that {task}",
                "Optimize the implementation of {task}"
            ],
            "reasoning": [
                "Solve the logic puzzle: {task}",
                "Analyze the pattern in {task}",
                "Deduce the conclusion from {task}"
            ],
            "planning": [
                "Create a plan to {task}",
                "Schedule tasks for {task}",
                "Optimize the workflow for {task}"
            ],
            "tool_use": [
                "Use API to {task}",
                "Query database for {task}",
                "Automate the process of {task}"
            ]
        }

        template = random.choice(templates.get(domain, templates["coding"]))
        task_content = f"handle difficulty level {difficulty:.1f} challenge"
        return template.format(task=task_content)


class SelfNavigatingModule:
    """
    Self-Navigating Module
    经验复用 + 混合策略指导
    """

    def __init__(self, gamma: float = 0.9):
        self.gamma = gamma  # 折扣因子 (论文中的 γ)
        self.experience_buffer: List[Experience] = []
        self.short_term_memory: List[Dict] = []  # 短期经验
        self.long_term_memory: List[Experience] = []  # 长期经验

    def store_experience(self, experience: Experience):
        """存储经验到缓冲区"""
        self.experience_buffer.append(experience)

        if experience.success:
            # 成功经验存入长期记忆
            self.long_term_memory.append(experience)

    def retrieve_similar_experiences(self, task: Task, k: int = 3) -> List[Experience]:
        """
        检索相似经验 (简化版：基于领域和难度的匹配)
        """
        if not self.long_term_memory:
            return []

        # 计算相似度分数
        scored_experiences = []
        for exp in self.long_term_memory:
            # 简化的相似度计算
            domain_match = 1.0 if exp.trajectory and exp.trajectory[0].get("domain") == task.domain else 0.0
            difficulty_diff = abs(exp.trajectory[0].get("difficulty", 0.5) - task.difficulty) if exp.trajectory else 0.5
            success_bonus = 1.0 if exp.success else 0.0

            # 综合分数 (论文中的 hybrid policy)
            score = (domain_match * 0.4 +
                    (1 - difficulty_diff) * 0.3 +
                    success_bonus * 0.3)

            scored_experiences.append((exp, score))

        # 返回 top-k
        scored_experiences.sort(key=lambda x: x[1], reverse=True)
        return [exp for exp, _ in scored_experiences[:k]]

    def get_action_guidance(self, task: Task, current_state: Dict) -> Dict:
        """
        基于检索的经验提供动作指导
        """
        similar_exps = self.retrieve_similar_experiences(task)

        if not similar_exps:
            return {"mode": "explore", "guidance": "No prior experience. Explore freely."}

        # 分析成功经验中的共同模式
        successful_actions = []
        for exp in similar_exps:
            if exp.success and exp.trajectory:
                successful_actions.extend([step["action"] for step in exp.trajectory])

        if successful_actions:
            # 简化：返回最常见的动作作为指导
            action_counts = {}
            for action in successful_actions:
                action_counts[action] = action_counts.get(action, 0) + 1
            most_common = max(action_counts, key=action_counts.get)

            return {
                "mode": "exploit",
                "guidance": f"Based on {len(similar_exps)} similar experiences, consider: {most_common}",
                "reference_experiences": [exp.task for exp in similar_exps]
            }

        return {"mode": "explore", "guidance": "Prior experiences not directly applicable."}


class SelfAttributingModule:
    """
    Self-Attributing Module
    差异化奖励归因 - 识别轨迹中关键的步骤
    """

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha  # 归因权重参数 (论文中的 α)
        self.state_values: Dict[str, float] = {}  # 状态价值估计

    def attribute_rewards(self, trajectory: List[Dict], final_reward: float) -> List[float]:
        """
        为轨迹中的每个步骤分配归因奖励

        简化实现：使用基于最终奖励和位置的衰减
        (论文中的想法：关键步骤应该有更高的奖励归因)
        """
        n = len(trajectory)
        if n == 0:
            return []

        attributed_rewards = []
        for i, step in enumerate(trajectory):
            # 位置加权：早期步骤和关键步骤获得更高归因
            position_factor = 1.0 + self.alpha * (1.0 - i / n)  # 早期步骤更重要

            # 动作成功度
            action_success = step.get("reward", 0.0)

            # 综合归因奖励
            attributed_reward = final_reward * position_factor * (0.5 + 0.5 * action_success)
            attributed_rewards.append(attributed_reward)

            # 更新状态价值
            state_key = str(step.get("state", ""))
            self.state_values[state_key] = self.state_values.get(state_key, 0.0) + attributed_reward

        return attributed_rewards

    def get_important_steps(self, trajectory: List[Dict], top_k: int = 3) -> List[int]:
        """识别轨迹中最重要的步骤索引"""
        attributed_rewards = self.attribute_rewards(trajectory, 1.0)  # 使用单位奖励找重要性

        # 获取奖励最高的步骤
        indexed_rewards = [(i, r) for i, r in enumerate(attributed_rewards)]
        indexed_rewards.sort(key=lambda x: x[1], reverse=True)

        return [i for i, _ in indexed_rewards[:top_k]]


class SimpleAgentEvolver:
    """
    简化版 AgentEvolver 系统
    整合三个自模块
    """

    def __init__(self):
        self.questioning = SelfQuestioningModule()
        self.navigating = SelfNavigatingModule()
        self.attributing = SelfAttributingModule()

        self.iteration = 0
        self.performance_history: List[float] = []

    def run_episode(self, mock_environment: Optional[Dict] = None) -> Dict:
        """
        运行一个 episode
        """
        print(f"\n{'='*60}")
        print(f"Episode {self.iteration}")
        print(f"{'='*60}")

        # 1. Self-Questioning: 生成任务
        feedback = {
            "success_rate": sum(self.performance_history[-5:]) / min(len(self.performance_history), 5) if self.performance_history else 0.5,
            "recent_domain": getattr(self.questioning.task_history[-1], 'domain', 'coding') if self.questioning.task_history else 'coding',
            "difficulty": getattr(self.questioning.task_history[-1], 'difficulty', 0.5) if self.questioning.task_history else 0.5
        } if self.iteration > 0 else None

        task = self.questioning.generate_task(feedback)
        print(f"[Self-Questioning] Generated task: {task.description}")
        print(f"                     Domain: {task.domain}, Difficulty: {task.difficulty:.2f}")

        # 2. Self-Navigating: 获取指导
        guidance = self.navigating.get_action_guidance(task, {})
        print(f"[Self-Navigating] Mode: {guidance['mode']}")
        print(f"                   Guidance: {guidance['guidance'][:80]}...")

        # 模拟执行 (简化版)
        trajectory = self._simulate_execution(task, guidance)

        # 3. Self-Attributing: 奖励归因
        final_reward = 1.0 if random.random() < (0.3 + 0.4 * task.difficulty) else 0.0  # 简化成功率
        attributed_rewards = self.attributing.attribute_rewards(trajectory, final_reward)
        important_steps = self.attributing.get_important_steps(trajectory)

        print(f"[Self-Attributing] Final reward: {final_reward}")
        print(f"                   Important steps: {important_steps}")

        # 存储经验
        experience = Experience(
            task=task.description,
            trajectory=trajectory,
            success=final_reward > 0.5,
            total_reward=final_reward,
            timestamp=datetime.now().isoformat()
        )
        self.navigating.store_experience(experience)

        self.performance_history.append(final_reward)
        self.iteration += 1

        return {
            "task": task,
            "guidance": guidance,
            "success": final_reward > 0.5,
            "important_steps": important_steps
        }

    def _simulate_execution(self, task: Task, guidance: Dict) -> List[Dict]:
        """模拟任务执行，生成轨迹"""
        # 简化：生成随机轨迹
        n_steps = random.randint(3, 8)
        trajectory = []

        actions = ["analyze", "plan", "execute", "verify", "retry", "refine"]

        for i in range(n_steps):
            action = random.choice(actions)
            state = {"domain": task.domain, "difficulty": task.difficulty, "step": i}
            reward = random.uniform(0, 1)
            trajectory.append({"state": state, "action": action, "reward": reward})

        return trajectory

    def get_stats(self) -> Dict:
        """获取学习统计"""
        if not self.performance_history:
            return {"message": "No episodes completed yet"}

        recent_success = sum(self.performance_history[-10:]) / min(len(self.performance_history), 10)

        return {
            "total_episodes": self.iteration,
            "overall_success_rate": sum(self.performance_history) / len(self.performance_history),
            "recent_success_rate": recent_success,
            "domains_explored": len(self.questioning.known_domains),
            "experiences_stored": len(self.navigating.long_term_memory),
            "state_values_learned": len(self.attributing.state_values)
        }


def demo():
    """演示 AgentEvolver 的三个核心机制"""
    print("\n" + "="*70)
    print("AgentEvolver 简化版演示")
    print("="*70)
    print("\n核心机制:")
    print("  1. Self-Questioning: 好奇心驱动的任务生成")
    print("  2. Self-Navigating: 经验复用与混合策略指导")
    print("  3. Self-Attributing: 差异化奖励归因")
    print("="*70)

    agent = SimpleAgentEvolver()

    # 运行多个 episodes
    n_episodes = 10
    print(f"\n运行 {n_episodes} 个 episodes...")

    for _ in range(n_episodes):
        agent.run_episode()

    # 最终统计
    print(f"\n{'='*70}")
    print("学习统计")
    print(f"{'='*70}")
    stats = agent.get_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")

    print(f"\n{'='*70}")
    print("关键观察:")
    print("  - Self-Questioning 自适应生成不同难度和领域的任务")
    print("  - Self-Navigating 从经验中学习并提供策略指导")
    print("  - Self-Attributing 识别关键步骤，提升样本效率")
    print("  - 整个系统无需人工标注数据，实现自主进化")
    print("="*70)


if __name__ == "__main__":
    demo()
