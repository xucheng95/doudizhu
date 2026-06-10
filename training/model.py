from __future__ import annotations
import math
import torch
import torch.nn as nn
from training.config import TrainingConfig


# ---------- State Encoder ----------

class StateEncoder(nn.Module):
    """
    Encodes game state as: [state_token, hist_1, ..., hist_T] -> Transformer -> state_emb.

    Inputs:
        hand:           (B, 54)         current player's hand bits (hands[0])
        num_cards:      (B, 3)          card counts [self, next, prev]
        role:           (B, 3)          one-hot role
        landlord_cards: (B, 54)         landlord bonus cards bits
        history:        (B, T, 62)      encoded play records
        history_mask:   (B, T) bool     True = padding (ignored by attention)

    Output:
        state_emb: (B, D)
    """

    STATE_RAW_DIM = 54 + 3 + 3 + 54   # = 114
    HIST_RAW_DIM = 3 + 16 + 15 + 13 + 15  # player + type + rank + length + cards = 62

    def __init__(self, cfg: TrainingConfig) -> None:
        super().__init__()
        D = cfg.d_model
        self.state_proj = nn.Linear(self.STATE_RAW_DIM, D)
        self.hist_proj = nn.Linear(self.HIST_RAW_DIM, D)
        self.pos_emb = nn.Embedding(1 + cfg.max_history, D)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D, nhead=cfg.num_heads,
            dim_feedforward=cfg.ff_dim, dropout=cfg.dropout,
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_layers)

    def forward(
        self,
        hand: torch.Tensor,
        num_cards: torch.Tensor,
        role: torch.Tensor,
        landlord_cards: torch.Tensor,
        history: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> torch.Tensor:
        B = hand.size(0)
        T = history.size(1)
        device = hand.device

        state_raw = torch.cat([hand, num_cards, role, landlord_cards], dim=-1)  # (B, 114)
        state_tok = self.state_proj(state_raw).unsqueeze(1)  # (B, 1, D)

        if T > 0:
            hist_tok = self.hist_proj(history)  # (B, T, D)
            seq = torch.cat([state_tok, hist_tok], dim=1)  # (B, 1+T, D)
            pad_false = torch.zeros(B, 1, dtype=torch.bool, device=device)
            key_padding_mask = torch.cat([pad_false, history_mask], dim=1)
        else:
            seq = state_tok
            key_padding_mask = torch.zeros(B, 1, dtype=torch.bool, device=device)

        seq_len = seq.size(1)
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        seq = seq + self.pos_emb(positions)

        out = self.transformer(seq, src_key_padding_mask=key_padding_mask)
        return out[:, 0, :]  # (B, D)


# ---------- Action Encoder ----------

class ActionEncoder(nn.Module):
    """
    Encodes a batch of legal moves (padded to max_actions=500) into D-dim vectors.
    Padded positions are zeroed out.

    Inputs:
        move_type:    (B, 500) long
        move_rank:    (B, 500) long
        move_length:  (B, 500) long
        move_kickers: (B, 500, 15) float
        move_cards:   (B, 500, 15) float
        pad_mask:     (B, 500) bool  — True = padding slot

    Output:
        action_embs: (B, 500, D)  — pad slots are zero vectors
    """

    def __init__(self, cfg: TrainingConfig) -> None:
        super().__init__()
        Da = cfg.d_action
        D = cfg.d_model
        self.type_emb = nn.Embedding(16, Da, padding_idx=0)
        self.rank_emb = nn.Embedding(15, Da)
        self.len_emb = nn.Embedding(13, Da)
        self.kicker_proj = nn.Linear(15, Da, bias=False)
        self.cards_proj = nn.Linear(15, Da, bias=False)
        self.merge = nn.Sequential(
            nn.Linear(Da, D),
            nn.ReLU(),
            nn.Linear(D, D),
        )

    def forward(
        self,
        move_type: torch.Tensor,
        move_rank: torch.Tensor,
        move_length: torch.Tensor,
        move_kickers: torch.Tensor,
        move_cards: torch.Tensor,
        pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = (
            self.type_emb(move_type)
            + self.rank_emb(move_rank)
            + self.len_emb(move_length)
            + self.kicker_proj(move_kickers)
            + self.cards_proj(move_cards)
        )  # (B, 500, Da)
        x = self.merge(x)  # (B, 500, D)
        x = x.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        return x


# ---------- DoudizhuAgent ----------

class DoudizhuAgent(nn.Module):
    """
    Combined actor + critic for one role (Landlord / Peasant0 / Peasant1).

    Actor:   cross-attention(Q=state_emb, K/V=action_embs) -> logits -> Categorical
    Critic:  MLP(state_emb || hands_flat) -> scalar value
    """

    def __init__(self, cfg: TrainingConfig):
        super().__init__()
        self.cfg = cfg
        self.state_enc = StateEncoder(cfg)
        self.action_enc = ActionEncoder(cfg)
        D = cfg.d_model
        self.scale = math.sqrt(D)
        # Centralized critic: state_emb(D) + all_hands(3*54=162)
        self.critic = nn.Sequential(
            nn.Linear(D + 162, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def _build_pad_mask(self, num_legal: torch.Tensor) -> torch.Tensor:
        """Return (B, max_actions) bool mask where True = padding."""
        indices = torch.arange(self.cfg.max_actions, device=num_legal.device)
        return indices.unsqueeze(0) >= num_legal.unsqueeze(1)

    def _forward_shared(
        self,
        hand, num_cards, role, landlord_cards,
        history, history_mask, all_hands,
        move_type, move_rank, move_length, move_kickers, move_cards, num_legal,
    ):
        pad_mask = self._build_pad_mask(num_legal)
        state_emb = self.state_enc(
            hand, num_cards, role, landlord_cards, history, history_mask
        )  # (B, D)
        action_embs = self.action_enc(
            move_type, move_rank, move_length, move_kickers, move_cards, pad_mask
        )  # (B, max_actions, D)

        # Cross-attention logits
        q = state_emb.unsqueeze(1)  # (B, 1, D)
        logits = torch.bmm(q, action_embs.transpose(1, 2)).squeeze(1) / self.scale  # (B, max_actions)
        logits = logits.masked_fill(pad_mask, float('-inf'))

        # Centralized critic
        hands_flat = all_hands.flatten(1)  # (B, 162)
        value = self.critic(torch.cat([state_emb, hands_flat], dim=-1)).squeeze(-1)  # (B,)

        return logits, value

    def act(
        self,
        hand, num_cards, role, landlord_cards,
        history, history_mask, all_hands,
        move_type, move_rank, move_length, move_kickers, move_cards, num_legal,
    ):
        """Sample action during rollout. Returns (action, log_prob, entropy, value)."""
        logits, value = self._forward_shared(
            hand, num_cards, role, landlord_cards,
            history, history_mask, all_hands,
            move_type, move_rank, move_length, move_kickers, move_cards, num_legal,
        )
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

    def evaluate_actions(
        self,
        hand, num_cards, role, landlord_cards,
        history, history_mask, all_hands,
        move_type, move_rank, move_length, move_kickers, move_cards, num_legal,
        actions,
    ):
        """Evaluate given actions during PPO update. Returns (log_prob, entropy, value)."""
        logits, value = self._forward_shared(
            hand, num_cards, role, landlord_cards,
            history, history_mask, all_hands,
            move_type, move_rank, move_length, move_kickers, move_cards, num_legal,
        )
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), value

