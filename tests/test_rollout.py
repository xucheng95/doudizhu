import sys, os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'doudizhu', 'python', 'doudizhu'))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'doudizhu', 'python'))
import pytest
import torch
from training.config import TrainingConfig
from training.model import DoudizhuAgent
from training.rollout import collect_episode

def test_collect_episode_produces_trajectories():
    cfg = TrainingConfig(d_model=64, d_action=32, num_layers=2, num_heads=2,
                         ff_dim=64, max_actions=500, max_history=60,
                         episodes_per_batch=1)
    agents = {
        "landlord": DoudizhuAgent(cfg),
        "peasant0": DoudizhuAgent(cfg),
        "peasant1": DoudizhuAgent(cfg),
    }
    trajectories = collect_episode(agents, cfg, device=torch.device("cpu"))
    assert set(trajectories.keys()) == {"landlord", "peasant0", "peasant1"}
    total_steps = sum(len(v) for v in trajectories.values())
    assert total_steps > 0
    for role, steps in trajectories.items():
        for step in steps:
            for key in ["hand", "action", "log_prob", "value", "reward", "done"]:
                assert key in step, f"Missing {key} in {role} step"
        assert steps[-1]["done"] == True

def test_collect_episode_terminal_reward_nonzero():
    cfg = TrainingConfig(d_model=64, d_action=32, num_layers=2, num_heads=2,
                         ff_dim=64, max_actions=500, max_history=60)
    agents = {
        "landlord": DoudizhuAgent(cfg),
        "peasant0": DoudizhuAgent(cfg),
        "peasant1": DoudizhuAgent(cfg),
    }
    trajectories = collect_episode(agents, cfg, device=torch.device("cpu"))
    for role, steps in trajectories.items():
        assert steps[-1]["reward"] != 0.0, f"Terminal reward is 0 for {role}"
