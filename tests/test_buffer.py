import pytest
import torch
from training.config import TrainingConfig
from training.buffer import RolloutBuffer

@pytest.fixture
def cfg():
    return TrainingConfig(d_model=64, max_actions=10, max_history=5,
                          gamma=0.99, gae_lambda=0.95, batch_size=4)

def _make_step(cfg):
    return {
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
        "reward": 0.0,
        "done": False,
    }

def test_buffer_stores_and_gae(cfg):
    buf = RolloutBuffer(cfg)
    for i in range(6):
        step = _make_step(cfg)
        step["reward"] = 1.0 if i == 5 else 0.0
        step["done"] = (i == 5)
        buf.add(step)
    buf.compute_gae(last_value=torch.tensor(0.0))
    assert len(buf) == 6
    # Last step gets the terminal reward — advantage should be highest
    advs = buf.advantages
    assert advs[-1].item() > advs[0].item()

def test_buffer_mini_batches(cfg):
    buf = RolloutBuffer(cfg)
    for _ in range(8):
        step = _make_step(cfg)
        step["done"] = False
        buf.add(step)
    buf.compute_gae(last_value=torch.tensor(0.0))
    batches = list(buf.iterate(batch_size=4))
    assert len(batches) == 2
    batch = batches[0]
    assert batch["action"].shape == (4,)
    assert batch["log_prob"].shape == (4,)
    assert batch["advantage"].shape == (4,)
    assert batch["returns"].shape == (4,)

def test_buffer_clear(cfg):
    buf = RolloutBuffer(cfg)
    buf.add(_make_step(cfg))
    buf.compute_gae(last_value=torch.tensor(0.0))
    buf.clear()
    assert len(buf) == 0
