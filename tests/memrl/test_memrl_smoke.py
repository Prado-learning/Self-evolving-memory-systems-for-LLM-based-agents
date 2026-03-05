from experiments.memrl.memrl_core import MemoryBank, ToyMemRLEnv, evaluate_success_rate


def test_task_specific_not_worse_than_baseline():
    env = ToyMemRLEnv()
    memory = MemoryBank(num_contexts=env.num_contexts)

    baseline = evaluate_success_rate(env, memory, baseline=True, train_steps=2500, eval_episodes=400, seed=7)
    improved = evaluate_success_rate(env, memory, baseline=False, train_steps=2500, eval_episodes=400, seed=7)

    assert improved >= baseline
