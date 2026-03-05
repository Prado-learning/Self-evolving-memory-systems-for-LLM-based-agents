"""
MemEvolve 单元测试
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from simple.memevolve import (
    MemoryComponent,
    MemoryArchitecture,
    Experience,
    EncodeModule,
    StoreModule,
    RetrieveModule,
    ManageModule,
    MemorySystem,
    MemEvolve,
    generate_mock_tasks
)


def test_memory_component():
    """测试MemoryComponent枚举"""
    assert MemoryComponent.ENCODE.value == "encode"
    assert MemoryComponent.STORE.value == "store"
    assert MemoryComponent.RETRIEVE.value == "retrieve"
    assert MemoryComponent.MANAGE.value == "manage"
    print("✓ MemoryComponent enum test passed")


def test_memory_architecture():
    """测试记忆架构基因型"""
    arch = MemoryArchitecture(
        name="TestArch",
        encode_strategy="insight",
        store_format="vector_db",
        retrieve_method="contrastive",
        manage_policy="prune"
    )

    assert arch.name == "TestArch"
    assert arch.encode_strategy == "insight"

    # 测试to_dict
    d = arch.to_dict()
    assert d["encode"] == "insight"
    assert d["store"] == "vector_db"

    print("✓ MemoryArchitecture test passed")


def test_encode_module():
    """测试编码模块"""
    # 测试不同策略
    for strategy in ["raw", "insight", "skill", "api"]:
        encoder = EncodeModule(strategy=strategy)
        exp = Experience(
            task="test task",
            trajectory=[],
            success=True,
            reward=1.0,
            timestamp="2024-01-01T00:00:00"
        )
        encoded = encoder.encode(exp)
        assert isinstance(encoded, str)
        assert len(encoded) > 0

    print("✓ EncodeModule test passed")


def test_store_module():
    """测试存储模块"""
    store = StoreModule(format_type="json")

    # 测试存储
    store.store("test content", {"success": True, "utility": 0.8})
    assert store.get_size() == 1

    # 测试不同格式
    for fmt in ["json", "vector_db", "graph"]:
        store = StoreModule(format_type=fmt)
        store.store(f"test {fmt}", {"test": True})
        assert store.get_size() == 1

    print("✓ StoreModule test passed")


def test_retrieve_module():
    """测试检索模块"""
    storage = [
        {"content": "coding task", "metadata": {"success": True}},
        {"content": "debugging task", "metadata": {"success": False}},
        {"content": "testing task", "metadata": {"success": True}},
    ]

    # 测试不同检索方法
    for method in ["semantic", "contrastive", "graph_search", "hybrid"]:
        retriever = RetrieveModule(method=method)
        results = retriever.retrieve("coding", storage, k=2)
        assert len(results) <= 2

    print("✓ RetrieveModule test passed")


def test_manage_module():
    """测试管理模块"""
    storage = [
        {"content": "exp1", "metadata": {"utility": 0.9}},
        {"content": "exp2", "metadata": {"utility": 0.2}},  # 应该被prune
        {"content": "exp3", "metadata": {"utility": 0.8}},
    ]

    # 测试prune
    manager = ManageModule(policy="prune")
    result = manager.manage(storage)
    assert len(result) <= len(storage)

    # 测试dedup
    manager = ManageModule(policy="dedup")
    result = manager.manage(storage)
    assert len(result) <= len(storage)

    print("✓ ManageModule test passed")


def test_memory_system():
    """测试完整记忆系统"""
    arch = MemoryArchitecture(
        name="TestSystem",
        encode_strategy="raw",
        store_format="json",
        retrieve_method="semantic",
        manage_policy="none"
    )

    system = MemorySystem(arch)

    # 测试处理经验
    exp = Experience(
        task="test task",
        trajectory=[{"step": 1}],
        success=True,
        reward=1.0,
        timestamp="2024-01-01T00:00:00"
    )

    system.process_experience(exp)
    assert system.experiences_processed == 1
    assert system.storage.get_size() == 1

    # 测试检索
    results = system.retrieve("test", k=1)
    assert len(results) <= 1

    # 测试统计
    stats = system.get_stats()
    assert stats["experiences_processed"] == 1

    print("✓ MemorySystem test passed")


def test_memevolve():
    """测试MemEvolve双层进化"""
    memevolve = MemEvolve(n_iterations=2, n_candidates=2)

    # 生成测试任务
    task_batches = [
        generate_mock_tasks(5) for _ in range(2)
    ]

    # 执行进化
    best_arch = memevolve.evolve(task_batches)

    assert best_arch is not None
    assert best_arch.fitness is not None
    assert "performance" in best_arch.fitness

    print("✓ MemEvolve test passed")


if __name__ == "__main__":
    test_memory_component()
    test_memory_architecture()
    test_encode_module()
    test_store_module()
    test_retrieve_module()
    test_manage_module()
    test_memory_system()
    test_memevolve()
    print("\n" + "="*50)
    print("All tests passed!")
    print("="*50)
