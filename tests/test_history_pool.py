import torch
from training.config import TrainingConfig
from training.model import DoudizhuAgent
from training.history_pool import HistoryPool

def test_pool_saves_and_samples():
    cfg = TrainingConfig(d_model=64, d_action=32, num_layers=2, num_heads=2,
                         ff_dim=64, max_actions=10, max_history=5, history_pool_max_size=5)
    pool = HistoryPool(cfg)
    agent = DoudizhuAgent(cfg)
    pool.save(epoch=10, role="landlord", state_dict=agent.state_dict())
    pool.save(epoch=20, role="landlord", state_dict=agent.state_dict())
    assert pool.sample("landlord") is not None

def test_pool_respects_max_size():
    cfg = TrainingConfig(d_model=64, d_action=32, num_layers=2, num_heads=2,
                         ff_dim=64, max_actions=10, max_history=5, history_pool_max_size=5)
    pool = HistoryPool(cfg)
    agent = DoudizhuAgent(cfg)
    for i in range(10):
        pool.save(epoch=i * 10, role="landlord", state_dict=agent.state_dict())
    assert pool.size("landlord") <= cfg.history_pool_max_size

def test_pool_load_into_agent():
    cfg = TrainingConfig(d_model=64, d_action=32, num_layers=2, num_heads=2,
                         ff_dim=64, max_actions=10, max_history=5, history_pool_max_size=5)
    pool = HistoryPool(cfg)
    agent = DoudizhuAgent(cfg)
    pool.save(epoch=5, role="landlord", state_dict=agent.state_dict())
    new_agent = DoudizhuAgent(cfg)
    pool.load_into(new_agent, "landlord")
    for p1, p2 in zip(agent.parameters(), new_agent.parameters()):
        assert torch.allclose(p1, p2)
