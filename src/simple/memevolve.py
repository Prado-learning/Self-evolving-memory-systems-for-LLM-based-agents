"""
Simple MemEvolve - 简化版复现
基于论文: "MemEvolve: Meta-Evolution of Agent Memory Systems"

核心思想: 双层进化 - 不仅积累经验，还进化记忆架构本身
"""

import random
import copy
from typing import List, Dict, Callable, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime


class MemoryComponent(Enum):
    """记忆系统的四个核心组件"""
    ENCODE = "encode"      # 编码: 将经验转换为结构化表示
    STORE = "store"        # 存储: 整合到持久记忆
    RETRIEVE = "retrieve"  # 检索: 上下文感知的回忆
    MANAGE = "manage"      # 管理: 整合与遗忘


@dataclass
class MemoryArchitecture:
    """
    记忆架构基因型 (Genotype)
    对应论文中的四组件设计空间: (E, U, R, G)
    """
    name: str
    encode_strategy: str   # 编码策略: "raw", "insight", "skill", "api"
    store_format: str      # 存储格式: "vector_db", "graph", "json", "hybrid"
    retrieve_method: str   # 检索方法: "semantic", "contrastive", "graph_search", "hybrid"
    manage_policy: str     # 管理策略: "none", "prune", "consolidate", "dedup"

    # 性能统计
    fitness: Dict[str, float] = field(default_factory=dict)
    iteration: int = 0

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "encode": self.encode_strategy,
            "store": self.store_format,
            "retrieve": self.retrieve_method,
            "manage": self.manage_policy,
            "fitness": self.fitness
        }


@dataclass
class Experience:
    """经验单元"""
    task: str
    trajectory: List[Dict]
    success: bool
    reward: float
    timestamp: str


class EncodeModule:
    """编码模块: 将原始轨迹转换为结构化表示"""

    STRATEGIES = {
        "raw": lambda exp: f"Raw trajectory: {exp.task}",
        "insight": lambda exp: f"Insight: Learned from {exp.task} - Success: {exp.success}",
        "skill": lambda exp: f"Skill: How to handle {exp.task.split()[0]} tasks",
        "api": lambda exp: f"API: def solve_{exp.task.split()[0]}(): ..."
    }

    def __init__(self, strategy: str = "raw"):
        self.strategy = strategy
        self.encoder = self.STRATEGIES.get(strategy, self.STRATEGIES["raw"])

    def encode(self, experience: Experience) -> str:
        return self.encoder(experience)


class StoreModule:
    """存储模块: 整合编码后的经验"""

    def __init__(self, format_type: str = "json"):
        self.format_type = format_type
        self.storage: List[Dict] = []

    def store(self, encoded_exp: str, metadata: Dict):
        """存储经验"""
        entry = {
            "content": encoded_exp,
            "format": self.format_type,
            "metadata": metadata,
            "timestamp": datetime.now().isoformat()
        }

        if self.format_type == "vector_db":
            # 模拟向量存储
            entry["embedding"] = [random.random() for _ in range(128)]
        elif self.format_type == "graph":
            # 模拟图存储
            entry["nodes"] = [metadata.get("domain", "unknown")]
            entry["edges"] = []

        self.storage.append(entry)

    def get_size(self) -> int:
        return len(self.storage)


class RetrieveModule:
    """检索模块: 基于查询检索相关记忆"""

    def __init__(self, method: str = "semantic"):
        self.method = method

    def retrieve(self, query: str, storage: List[Dict], k: int = 3) -> List[str]:
        """检索相关记忆"""
        if not storage:
            return []

        if self.method == "semantic":
            # 模拟语义相似度检索
            scored = []
            for entry in storage:
                score = self._semantic_similarity(query, entry["content"])
                scored.append((entry["content"], score))

        elif self.method == "contrastive":
            # 对比检索: 区分成功和失败经验
            scored = []
            for entry in storage:
                base_score = self._semantic_similarity(query, entry["content"])
                success_bonus = 0.3 if entry["metadata"].get("success", False) else 0.0
                scored.append((entry["content"], base_score + success_bonus))

        elif self.method == "graph_search":
            # 模拟图搜索
            scored = [(entry["content"], random.random()) for entry in storage]

        else:  # hybrid
            scored = [(entry["content"], random.random() * 0.5 + 0.5)
                     for entry in storage]

        scored.sort(key=lambda x: x[1], reverse=True)
        return [content for content, _ in scored[:k]]

    def _semantic_similarity(self, a: str, b: str) -> float:
        """简化的语义相似度"""
        words_a = set(a.lower().split())
        words_b = set(b.lower().split())
        if not words_a or not words_b:
            return 0.0
        return len(words_a & words_b) / len(words_a | words_b)


class ManageModule:
    """管理模块: 离线整合与维护"""

    def __init__(self, policy: str = "none"):
        self.policy = policy

    def manage(self, storage: List[Dict]) -> List[Dict]:
        """执行记忆管理策略"""
        if self.policy == "none":
            return storage

        elif self.policy == "prune":
            # 剪枝: 删除低质量记忆
            return [entry for entry in storage
                   if entry["metadata"].get("utility", 0.5) > 0.3]

        elif self.policy == "consolidate":
            # 整合: 合并相似记忆
            if len(storage) > 10:
                # 简化: 只保留最近的一半
                return storage[len(storage)//2:]
            return storage

        elif self.policy == "dedup":
            # 去重
            seen = set()
            deduped = []
            for entry in storage:
                content_hash = hash(entry["content"])
                if content_hash not in seen:
                    seen.add(content_hash)
                    deduped.append(entry)
            return deduped

        return storage


class MemorySystem:
    """
    完整的记忆系统
    由四个模块组成: Encode, Store, Retrieve, Manage
    """

    def __init__(self, architecture: MemoryArchitecture):
        self.arch = architecture
        self.encoder = EncodeModule(architecture.encode_strategy)
        self.storage = StoreModule(architecture.store_format)
        self.retriever = RetrieveModule(architecture.retrieve_method)
        self.manager = ManageModule(architecture.manage_policy)

        self.experiences_processed = 0

    def process_experience(self, exp: Experience) -> bool:
        """处理新经验"""
        # Encode
        encoded = self.encoder.encode(exp)

        # Store
        metadata = {
            "success": exp.success,
            "reward": exp.reward,
            "domain": exp.task.split()[0] if exp.task else "unknown",
            "utility": exp.reward
        }
        self.storage.store(encoded, metadata)

        self.experiences_processed += 1

        # Manage (周期性执行)
        if self.experiences_processed % 5 == 0:
            self.storage.storage = self.manager.manage(self.storage.storage)

        return True

    def retrieve(self, query: str, k: int = 3) -> List[str]:
        """检索记忆"""
        return self.retriever.retrieve(query, self.storage.storage, k)

    def get_stats(self) -> Dict:
        return {
            "architecture": self.arch.to_dict(),
            "experiences_processed": self.experiences_processed,
            "storage_size": self.storage.get_size()
        }


class MemEvolve:
    """
    MemEvolve: Meta-Evolution of Memory Systems
    双层进化框架
    """

    def __init__(self, n_iterations: int = 3, n_candidates: int = 3):
        self.n_iterations = n_iterations
        self.n_candidates = n_candidates
        self.survivor_budget = 1  # 每轮保留的父代数量

        # 候选架构集合
        self.candidates: List[MemorySystem] = []
        self.best_architecture: Optional[MemoryArchitecture] = None

    def initialize_population(self):
        """初始化候选架构"""
        base_arch = MemoryArchitecture(
            name="BaseMemory",
            encode_strategy="raw",
            store_format="json",
            retrieve_method="semantic",
            manage_policy="none"
        )
        self.candidates = [MemorySystem(base_arch)]

    def inner_loop(self, memory_system: MemorySystem, task_batch: List[Experience]) -> Dict[str, float]:
        """
        内层循环: 一阶进化
        在固定架构下积累经验，评估性能
        """
        total_reward = 0.0
        total_cost = 0.0  # 模拟API成本
        total_latency = 0.0  # 模拟延迟

        for exp in task_batch:
            # 模拟检索和使用记忆
            retrieved = memory_system.retrieve(exp.task, k=2)

            # 模拟记忆对性能的影响
            # 好的架构应该能更好地利用记忆
            if retrieved:
                boost = random.uniform(0.1, 0.3)  # 记忆带来的性能提升
            else:
                boost = 0.0

            success = (exp.success and random.random() < (0.6 + boost))
            reward = 1.0 if success else 0.0

            # 处理经验
            memory_system.process_experience(exp)

            total_reward += reward
            total_cost += 0.01 * len(retrieved)  # 检索成本
            total_latency += random.uniform(0.5, 1.5)  # 处理延迟

        n = len(task_batch)
        return {
            "performance": total_reward / n,
            "cost": total_cost / n,
            "latency": total_latency / n
        }

    def outer_loop(self, fitness_results: List[Tuple[MemorySystem, Dict[str, float]]]) -> List[MemoryArchitecture]:
        """
        外层循环: 二阶进化
        基于性能反馈进化架构
        """
        # 帕累托排序 (简化版)
        scored = []
        for mem_sys, fitness in fitness_results:
            # 多目标: 高performance, 低cost, 低latency
            score = (fitness["performance"] -
                    0.1 * fitness["cost"] -
                    0.01 * fitness["latency"])
            scored.append((mem_sys, score, fitness))

        scored.sort(key=lambda x: x[1], reverse=True)

        # 选择 top-K 作为父代
        parents = scored[:self.survivor_budget]

        print(f"    Top performer: {parents[0][0].arch.name}")
        print(f"    Fitness: {parents[0][2]}")

        # 生成后代 (变异和重组)
        new_candidates = []
        for parent_mem, _, _ in parents:
            parent_arch = parent_mem.arch

            for i in range(self.n_candidates):
                child_arch = self._mutate_architecture(parent_arch, i)
                new_candidates.append(child_arch)

        return new_candidates

    def _mutate_architecture(self, parent: MemoryArchitecture, variant_id: int) -> MemoryArchitecture:
        """
        变异操作: 在四组件设计空间中产生变体
        """
        # 可选的策略
        encode_options = ["raw", "insight", "skill", "api"]
        store_options = ["json", "vector_db", "graph", "hybrid"]
        retrieve_options = ["semantic", "contrastive", "graph_search", "hybrid"]
        manage_options = ["none", "prune", "consolidate", "dedup"]

        # 根据variant_id选择变异位置
        new_arch = copy.deepcopy(parent)
        new_arch.name = f"{parent.name}_v{variant_id}"
        new_arch.iteration = parent.iteration + 1

        if variant_id == 0:
            new_arch.encode_strategy = random.choice(encode_options)
        elif variant_id == 1:
            new_arch.retrieve_method = random.choice(retrieve_options)
        elif variant_id == 2:
            new_arch.manage_policy = random.choice(manage_options)

        return new_arch

    def evolve(self, task_batches: List[List[Experience]]):
        """
        执行完整的双层进化过程
        """
        print("\n" + "="*70)
        print("MemEvolve: Meta-Evolution of Memory Systems")
        print("="*70)

        self.initialize_population()

        for iteration in range(self.n_iterations):
            print(f"\n{'='*70}")
            print(f"Evolution Iteration {iteration + 1}/{self.n_iterations}")
            print(f"{'='*70}")

            # 内层循环: 评估每个候选架构
            fitness_results = []
            task_batch = task_batches[iteration % len(task_batches)]

            print(f"\n  Inner Loop: Evaluating {len(self.candidates)} candidate(s)...")

            for mem_sys in self.candidates:
                print(f"\n    Testing: {mem_sys.arch.name}")
                fitness = self.inner_loop(mem_sys, task_batch)
                mem_sys.arch.fitness = fitness
                fitness_results.append((mem_sys, fitness))

            # 外层循环: 进化架构
            print(f"\n  Outer Loop: Evolving architectures...")
            new_architectures = self.outer_loop(fitness_results)

            # 更新候选集合
            self.candidates = [MemorySystem(arch) for arch in new_architectures]

        # 选择最佳架构
        best = max(fitness_results, key=lambda x: x[1]["performance"])
        self.best_architecture = best[0].arch

        print(f"\n{'='*70}")
        print("Evolution Complete!")
        print(f"Best Architecture: {self.best_architecture.name}")
        print(f"Final Fitness: {self.best_architecture.fitness}")
        print(f"Components: {self.best_architecture.to_dict()}")
        print("="*70)

        return self.best_architecture


def generate_mock_tasks(n: int) -> List[Experience]:
    """生成模拟任务"""
    tasks = []
    domains = ["coding", "reasoning", "planning", "web_search"]

    for i in range(n):
        domain = random.choice(domains)
        task = Experience(
            task=f"{domain} task {i}: solve problem",
            trajectory=[{"step": j, "action": f"action_{j}"} for j in range(5)],
            success=random.random() > 0.4,
            reward=random.random(),
            timestamp=datetime.now().isoformat()
        )
        tasks.append(task)

    return tasks


def demo():
    """演示 MemEvolve 的双层进化"""
    print("\n" + "="*70)
    print("MemEvolve 简化版演示")
    print("="*70)
    print("\n核心思想:")
    print("  双层进化框架:")
    print("    - 内层 (一阶): 固定架构下积累经验")
    print("    - 外层 (二阶): 进化记忆架构本身")
    print("\n  模块化设计空间:")
    print("    - Encode: raw | insight | skill | api")
    print("    - Store: json | vector_db | graph | hybrid")
    print("    - Retrieve: semantic | contrastive | graph_search | hybrid")
    print("    - Manage: none | prune | consolidate | dedup")
    print("="*70)

    # 准备任务批次
    print("\n准备任务数据...")
    task_batches = [
        generate_mock_tasks(10) for _ in range(3)
    ]

    # 创建 MemEvolve
    memevolve = MemEvolve(n_iterations=3, n_candidates=3)

    # 执行进化
    best_arch = memevolve.evolve(task_batches)

    # 演示最佳架构
    print(f"\n{'='*70}")
    print("使用进化后的最佳架构...")
    print(f"{'='*70}")

    best_system = MemorySystem(best_arch)
    test_tasks = generate_mock_tasks(5)

    for exp in test_tasks:
        best_system.process_experience(exp)

    # 测试检索
    query = "coding task"
    retrieved = best_system.retrieve(query, k=2)

    print(f"\n  Query: '{query}'")
    print(f"  Retrieved {len(retrieved)} memories:")
    for i, mem in enumerate(retrieved, 1):
        print(f"    {i}. {mem[:60]}...")

    print(f"\n{'='*70}")
    print("关键观察:")
    print("  - 记忆架构本身进化，而非仅仅积累经验")
    print("  - 四组件设计空间提供了可控的优化空间")
    print("  - Diagnose-and-Design: 基于性能诊断改进架构")
    print("  - 跨任务/跨模型泛化: 进化出的架构可迁移")
    print("="*70)


if __name__ == "__main__":
    demo()
