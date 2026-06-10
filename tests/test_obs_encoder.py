import pytest
import torch
import sys, os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'doudizhu', 'python', 'doudizhu'))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'doudizhu', 'python'))
from training.config import TrainingConfig
from training.obs_encoder import encode_obs, encode_obs_batch

def _make_fake_obs(n_legal=5, n_history=3):
    import numpy as np
    return {
        "hands": np.zeros((3, 54), dtype=np.float32),
        "landlord_cards": np.zeros(54, dtype=np.float32),
        "num_cards": np.array([17, 17, 17], dtype=np.float32),
        "role": np.array([0, 0, 1], dtype=np.float32),
        "last_player": -1,
        "history": [
            {"player": 0, "type": "SINGLE", "rank": 5, "length": 0,
             "kickers": [], "cards": ["8"], "pass": False}
        ] * n_history,
        "legal_moves": [
            {"type": "SINGLE", "rank": i, "length": 0,
             "kickers": [], "cards": [str(i+3)], "pass": False}
            for i in range(n_legal)
        ],
    }

def test_encode_obs_shapes():
    cfg = TrainingConfig(max_actions=500, max_history=60)
    obs = _make_fake_obs(n_legal=5, n_history=3)
    enc = encode_obs(obs, cfg, device=torch.device('cpu'))

    assert enc['hand'].shape == (54,)
    assert enc['num_cards'].shape == (3,)
    assert enc['role'].shape == (3,)
    assert enc['landlord_cards'].shape == (54,)
    assert enc['history'].shape[1] == 62  # (T, 62)
    assert enc['history_mask'].shape == enc['history'].shape[:1]  # (T,)
    assert enc['all_hands'].shape == (3, 54)
    assert enc['move_type'].shape == (500,)
    assert enc['move_rank'].shape == (500,)
    assert enc['move_length'].shape == (500,)
    assert enc['move_kickers'].shape == (500, 15)
    assert enc['move_cards'].shape == (500, 15)
    assert enc['num_legal'].item() == 5

def test_encode_obs_batch():
    cfg = TrainingConfig(max_actions=500, max_history=60)
    obs_list = [_make_fake_obs(n_legal=3, n_history=2),
                _make_fake_obs(n_legal=7, n_history=5)]
    batch = encode_obs_batch(obs_list, cfg, device=torch.device('cpu'))
    assert batch['hand'].shape == (2, 54)
    assert batch['move_type'].shape == (2, 500)
    assert batch['num_legal'].shape == (2,)
    assert batch['num_legal'][0].item() == 3
    assert batch['num_legal'][1].item() == 7

def test_pad_positions_zero():
    cfg = TrainingConfig(max_actions=500, max_history=60)
    obs = _make_fake_obs(n_legal=3)
    enc = encode_obs(obs, cfg, device=torch.device('cpu'))
    assert enc['move_type'][3:].sum().item() == 0
    assert enc['move_rank'][3:].sum().item() == 0
