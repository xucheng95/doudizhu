import pytest
import torch
from training.config import TrainingConfig
from training.model import StateEncoder, ActionEncoder

@pytest.fixture
def cfg():
    return TrainingConfig(d_model=64, d_action=32, num_layers=2,
                          num_heads=4, ff_dim=128, max_history=10, max_actions=500)

def test_state_encoder_output_shape(cfg):
    enc = StateEncoder(cfg)
    hand = torch.zeros(2, 54)
    num_cards = torch.zeros(2, 3)
    role = torch.zeros(2, 3)
    landlord_cards = torch.zeros(2, 54)
    history = torch.zeros(2, 5, 62)
    history_mask = torch.zeros(2, 5, dtype=torch.bool)  # all valid

    state_emb = enc(hand, num_cards, role, landlord_cards, history, history_mask)
    assert state_emb.shape == (2, cfg.d_model), f"Expected (2, {cfg.d_model}), got {state_emb.shape}"

def test_state_encoder_handles_empty_history(cfg):
    enc = StateEncoder(cfg)
    hand = torch.zeros(2, 54)
    num_cards = torch.zeros(2, 3)
    role = torch.zeros(2, 3)
    landlord_cards = torch.zeros(2, 54)
    history = torch.zeros(2, 0, 62)  # empty history
    history_mask = torch.zeros(2, 0, dtype=torch.bool)

    state_emb = enc(hand, num_cards, role, landlord_cards, history, history_mask)
    assert state_emb.shape == (2, cfg.d_model)

def test_action_encoder_output_shape(cfg):
    enc = ActionEncoder(cfg)
    batch = 2
    move_type = torch.zeros(batch, cfg.max_actions, dtype=torch.long)
    move_rank = torch.zeros(batch, cfg.max_actions, dtype=torch.long)
    move_length = torch.zeros(batch, cfg.max_actions, dtype=torch.long)
    move_kickers = torch.zeros(batch, cfg.max_actions, 15)
    move_cards = torch.zeros(batch, cfg.max_actions, 15)
    pad_mask = torch.ones(batch, cfg.max_actions, dtype=torch.bool)  # all padded
    pad_mask[:, :3] = False  # first 3 are valid

    action_embs = enc(move_type, move_rank, move_length, move_kickers, move_cards, pad_mask)
    assert action_embs.shape == (batch, cfg.max_actions, cfg.d_model)
    # Padded positions should be zero
    assert action_embs[0, 5].abs().sum().item() == 0.0

from training.model import DoudizhuAgent

def test_agent_act_output_shapes(cfg):
    agent = DoudizhuAgent(cfg)
    B = 2
    hand = torch.zeros(B, 54)
    num_cards = torch.zeros(B, 3)
    role = torch.zeros(B, 3)
    landlord_cards = torch.zeros(B, 54)
    history = torch.zeros(B, 5, 62)
    history_mask = torch.zeros(B, 5, dtype=torch.bool)
    all_hands = torch.zeros(B, 3, 54)

    move_type = torch.zeros(B, 500, dtype=torch.long)
    move_rank = torch.zeros(B, 500, dtype=torch.long)
    move_length = torch.zeros(B, 500, dtype=torch.long)
    move_kickers = torch.zeros(B, 500, 15)
    move_cards = torch.zeros(B, 500, 15)
    num_legal = torch.tensor([3, 5])

    action, log_prob, entropy, value = agent.act(
        hand, num_cards, role, landlord_cards,
        history, history_mask, all_hands,
        move_type, move_rank, move_length, move_kickers, move_cards, num_legal
    )
    assert action.shape == (B,)
    assert log_prob.shape == (B,)
    assert entropy.shape == (B,)
    assert value.shape == (B,)
    assert action[0].item() < 3
    assert action[1].item() < 5

def test_agent_evaluate_actions(cfg):
    agent = DoudizhuAgent(cfg)
    B = 4
    hand = torch.zeros(B, 54)
    num_cards = torch.zeros(B, 3)
    role = torch.zeros(B, 3)
    landlord_cards = torch.zeros(B, 54)
    history = torch.zeros(B, 3, 62)
    history_mask = torch.zeros(B, 3, dtype=torch.bool)
    all_hands = torch.zeros(B, 3, 54)
    move_type = torch.zeros(B, 500, dtype=torch.long)
    move_rank = torch.zeros(B, 500, dtype=torch.long)
    move_length = torch.zeros(B, 500, dtype=torch.long)
    move_kickers = torch.zeros(B, 500, 15)
    move_cards = torch.zeros(B, 500, 15)
    num_legal = torch.tensor([4, 4, 4, 4])
    actions = torch.tensor([0, 1, 2, 3])

    log_prob, entropy, value = agent.evaluate_actions(
        hand, num_cards, role, landlord_cards,
        history, history_mask, all_hands,
        move_type, move_rank, move_length, move_kickers, move_cards, num_legal,
        actions
    )
    assert log_prob.shape == (B,)
    assert entropy.shape == (B,)
    assert value.shape == (B,)
    assert not torch.isnan(log_prob).any()
    assert not torch.isnan(value).any()
