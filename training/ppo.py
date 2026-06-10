from __future__ import annotations
import torch
import torch.nn as nn
from dataclasses import dataclass
from training.config import TrainingConfig
from training.model import DoudizhuAgent
from training.buffer import RolloutBuffer


@dataclass
class PPOMetrics:
    policy_loss: float
    value_loss: float
    entropy: float
    total_loss: float


class PPOUpdater:
    """Runs K epochs of PPO updates for one agent."""

    def __init__(self, agent: DoudizhuAgent, cfg: TrainingConfig):
        self.agent = agent
        self.cfg = cfg
        self.optimizer = torch.optim.Adam(agent.parameters(), lr=cfg.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=cfg.max_epochs
        )

    def update(self, buffer: RolloutBuffer) -> dict:
        """Run ppo_epochs of mini-batch updates. Clears buffer at end."""
        total_policy, total_value, total_entropy, n = 0.0, 0.0, 0.0, 0
        for ep in range(self.cfg.ppo_epochs):
            for batch in buffer.iterate(self.cfg.batch_size):
                m = self._update_batch(batch)
                total_policy += m.policy_loss
                total_value += m.value_loss
                total_entropy += m.entropy
                n += 1
        buffer.clear()
        n = max(n, 1)
        return {"policy_loss": total_policy / n,
                "value_loss": total_value / n,
                "entropy": total_entropy / n}

    def _update_batch(self, batch: dict) -> PPOMetrics:
        device = next(self.agent.parameters()).device

        def to(t): return t.to(device)

        new_log_prob, entropy, new_value = self.agent.evaluate_actions(
            to(batch["hand"]), to(batch["num_cards"]), to(batch["role"]),
            to(batch["landlord_cards"]), to(batch["history"]), to(batch["history_mask"]),
            to(batch["all_hands"]), to(batch["move_type"]), to(batch["move_rank"]),
            to(batch["move_length"]), to(batch["move_kickers"]), to(batch["move_cards"]),
            to(batch["num_legal"]), to(batch["action"]),
        )
        old_log_prob = to(batch["log_prob"])
        advantage = to(batch["advantage"])
        returns = to(batch["returns"])
        old_value = to(batch["value"])

        ratio = (new_log_prob - old_log_prob).exp()
        surr1 = ratio * advantage
        surr2 = ratio.clamp(1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps) * advantage
        policy_loss = -torch.min(surr1, surr2).mean()

        value_clipped = old_value + (new_value - old_value).clamp(
            -self.cfg.clip_eps, self.cfg.clip_eps)
        value_loss = torch.max(
            (new_value - returns).pow(2),
            (value_clipped - returns).pow(2)
        ).mean()

        loss = (policy_loss
                + self.cfg.value_coeff * value_loss
                - self.cfg.entropy_coeff * entropy.mean())
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.agent.parameters(), self.cfg.max_grad_norm)
        self.optimizer.step()

        return PPOMetrics(policy_loss.item(), value_loss.item(),
                          entropy.mean().item(), loss.item())

    def step_scheduler(self) -> None:
        self.scheduler.step()
