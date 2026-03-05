"""
AgentEvolver 单元测试
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from simple.agentevolve import (
    Experience,
    Task,
    SelfQuestioningModule,
    SelfNavigatingModule,
    SelfAttributingModule,
    SimpleAgentEvolver
)


def test_experience_creation():
    """测试经验单元创建"""
    exp = Experience(
        task="test task",
        trajectory=[{"state": {}, "action": "test", "reward": 1.0}],
        success=True,
        total_reward=1.0,
        timestamp="2024-01-01T00:00:00"
    )
    assert exp.task == "test task"
    assert exp.success is True
    print("✓ Experience creation test passed")


def test_task_creation():
    """测试任务创建"""
    task = Task(
        id="test_001",
        description="Test description",
        difficulty=0.5,
        domain="coding"
    )
    assert task.id == "test_001"
    assert task.domain == "coding"
    print("✓ Task creation test passed")


def test_self_questioning():
    """测试Self-Questioning模块"""
    module = SelfQuestioningModule(curiosity_threshold=0.3)

    # 测试初始任务生成
    task1 = module.generate_task()
    assert task1.id.startswith("task_")
    assert task1.domain in ["coding", "reasoning", "planning", "tool_use"]

    # 测试基于反馈的任务生成
    feedback = {
        "success_rate": 0.8,
        "recent_domain": "coding",
        "difficulty": 0.5
    }
    task2 = module.generate_task(feedback)
    assert task2.difficulty > 0.5  # 成功率高应该增加难度

    print("✓ Self-Questioning module test passed")


def test_self_navigating():
    """测试Self-Navigating模块"""
    module = SelfNavigatingModule(gamma=0.9)

    # 创建测试经验
    exp = Experience(
        task="coding task",
        trajectory=[{"state": {"domain": "coding", "difficulty": 0.5}, "action": "test", "reward": 1.0}],
        success=True,
        total_reward=1.0,
        timestamp="2024-01-01T00:00:00"
    )

    # 存储经验
    module.store_experience(exp)
    assert len(module.long_term_memory) == 1

    # 测试检索
    task = Task(id="t1", description="coding test", difficulty=0.5, domain="coding")
    similar = module.retrieve_similar_experiences(task, k=1)
    assert len(similar) <= 1

    # 测试指导生成
    guidance = module.get_action_guidance(task, {})
    assert "mode" in guidance
    assert guidance["mode"] in ["explore", "exploit"]

    print("✓ Self-Navigating module test passed")


def test_self_attributing():
    """测试Self-Attributing模块"""
    module = SelfAttributingModule(alpha=0.5)

    # 创建测试轨迹
    trajectory = [
        {"state": "s1", "action": "a1", "reward": 0.5},
        {"state": "s2", "action": "a2", "reward": 0.8},
        {"state": "s3", "action": "a3", "reward": 1.0},
    ]

    # 测试奖励归因
    attributed = module.attribute_rewards(trajectory, final_reward=1.0)
    assert len(attributed) == len(trajectory)
    assert all(r >= 0 for r in attributed)

    # 测试重要步骤识别
    important = module.get_important_steps(trajectory, top_k=2)
    assert len(important) <= 2
    assert all(isinstance(i, int) for i in important)

    print("✓ Self-Attributing module test passed")


def test_full_agent():
    """测试完整AgentEvolver系统"""
    agent = SimpleAgentEvolver()

    # 运行一个episode
    result = agent.run_episode()

    assert "task" in result
    assert "guidance" in result
    assert "success" in result
    assert "important_steps" in result

    # 测试统计
    stats = agent.get_stats()
    assert stats["total_episodes"] == 1
    assert stats["domains_explored"] >= 1

    print("✓ Full AgentEvolver system test passed")


if __name__ == "__main__":
    test_experience_creation()
    test_task_creation()
    test_self_questioning()
    test_self_navigating()
    test_self_attributing()
    test_full_agent()
    print("\n" + "="*50)
    print("All tests passed!")
    print("="*50)
