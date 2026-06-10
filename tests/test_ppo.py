import pytest
import torch
from training.config import TrainingConfig
from training.model import DoudizhuAgent
from training.buffer import RolloutBuffer
from training.ppo import PPOUpdater

@pytest.fixture
def cfg():
    return TrainingConfig(d_model=64, d_action=32, num_layers=2, num_heads=4,
                          ff_dim=128, max_actions=10, max_history=5,
                          gamma=0.99, gae_lambda=0.95, clip_eps=0.2,
                          value_coeff=0.5, entropy_coeff=0.01,
                          lr=1e-3, max_grad_norm=0.5, ppo_epochs=2, batch_size=4)

def _fill_buffer(buf, cfg, n=8):
    for i in range(n):
        step = {
            "hand": torch.zeros(54),
            "num_cards": torch.zeros(3),
            "role": torch.zeros(3),
            "landlord_cards": torch.zeros(54),
            "history": torch.zeros(0, 62),
            "history_mask": torch.zeros(0, dtype=torch.bool),
            "all_hands": torch.zeros(3, 54),
            "move_type": torch.zeros(cfg.max_actions, dtype=torch.long),
            "move_rank": torch.zeros(cfg.max_actions, dtype=torch.long),
            "move_length": torch.zeros(cfg.max_actions, dtype=torch.long),
            "move_kickers": torch.zeros(cfg.max_actions, 15),
            "move_cards": torch.zeros(cfg.max_actions, 15),
            "num_legal": torch.tensor(3, dtype=torch.long),
            "action": torch.tensor(0, dtype=torch.long),
            "log_prob": torch.tensor(-1.0),
            "value": torch.tensor(0.5),
            "reward": 1.0 if i == n - 1 else 0.0,
            "done": (i == n - 1),
        }
        buf.add(step)
    buf.compute_gae(last_value=torch.tensor(0.0))

def test_ppo_update_returns_metrics(cfg):
    agent = DoudizhuAgent(cfg)
    buf = RolloutBuffer(cfg)
    _fill_buffer(buf, cfg)
    updater = PPOUpdater(agent, cfg)
    metrics = updater.update(buf)
    assert "policy_loss" in metrics
    assert "value_loss" in metrics
    assert "entropy" in metrics
    assert not torch.isnan(torch.tensor(metrics["policy_loss"]))
    assert not torch.isnan(torch.tensor(metrics["value_loss"]))

def test_ppo_update_clears_buffer(cfg):
    agent = DoudizhuAgent(cfg)
    buf = RolloutBuffer(cfg)
    _fill_buffer(buf, cfg)
    updater = PPOUpdater(agent, cfg)
    updater.update(buf)
    assert len(buf) == 0
