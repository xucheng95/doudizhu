from __future__ import annotations
import torch
from typing import Iterator
from training.config import TrainingConfig


class RolloutBuffer:
    """Stores one agent's trajectory, computes GAE, yields mini-batches."""

    def __init__(self, cfg: TrainingConfig):
        self.cfg = cfg
        self._steps: list[dict] = []
        self.advantages: torch.Tensor | None = None
        self.returns: torch.Tensor | None = None

    def add(self, step: dict) -> None:
        self._steps.append(step)

    def __len__(self) -> int:
        return len(self._steps)

    def compute_gae(self, last_value: torch.Tensor) -> None:
        T = len(self._steps)
        advantages = torch.zeros(T)
        gae = 0.0
        next_val = last_value.item()
        for t in reversed(range(T)):
            s = self._steps[t]
            r = float(s["reward"])
            done = float(s["done"])
            v = s["value"].item()
            delta = r + self.cfg.gamma * next_val * (1 - done) - v
            gae = delta + self.cfg.gamma * self.cfg.gae_lambda * (1 - done) * gae
            advantages[t] = gae
            next_val = v
        self.advantages = advantages
        self.returns = advantages + torch.tensor([s["value"].item() for s in self._steps])

    def iterate(self, batch_size: int) -> Iterator[dict]:
        assert self.advantages is not None, "Call compute_gae() before iterate()"
        # Only train on steps where agent had a choice (num_legal > 1)
        trainable = [i for i, s in enumerate(self._steps) if s["num_legal"].item() > 1]
        if not trainable:
            return
        T = len(trainable)
        perm = torch.randperm(T)
        adv_shuffled = self.advantages[trainable][perm]
        adv_norm = (adv_shuffled - adv_shuffled.mean()) / (adv_shuffled.std() + 1e-8)

        for start in range(0, T, batch_size):
            batch_perm = perm[start:start + batch_size]
            idx = [trainable[p] for p in batch_perm]
            steps = [self._steps[i] for i in idx]
            adv_batch = adv_norm[start:start + batch_size]

            max_T = max(s["history"].size(0) for s in steps)

            def _pad(s):
                t = s["history"].size(0)
                pad = max_T - t
                if pad > 0:
                    h = torch.cat([s["history"], torch.zeros(pad, 62)])
                    m = torch.cat([s["history_mask"], torch.ones(pad, dtype=torch.bool)])
                else:
                    h, m = s["history"], s["history_mask"]
                return h, m

            histories, masks = zip(*[_pad(s) for s in steps])

            yield {
                "hand": torch.stack([s["hand"] for s in steps]),
                "num_cards": torch.stack([s["num_cards"] for s in steps]),
                "role": torch.stack([s["role"] for s in steps]),
                "landlord_cards": torch.stack([s["landlord_cards"] for s in steps]),
                "history": torch.stack(list(histories)),
                "history_mask": torch.stack(list(masks)),
                "all_hands": torch.stack([s["all_hands"] for s in steps]),
                "move_type": torch.stack([s["move_type"] for s in steps]),
                "move_rank": torch.stack([s["move_rank"] for s in steps]),
                "move_length": torch.stack([s["move_length"] for s in steps]),
                "move_kickers": torch.stack([s["move_kickers"] for s in steps]),
                "move_cards": torch.stack([s["move_cards"] for s in steps]),
                "num_legal": torch.stack([s["num_legal"] for s in steps]),
                "action": torch.stack([s["action"] for s in steps]),
                "log_prob": torch.stack([s["log_prob"] for s in steps]),
                "value": torch.stack([s["value"] for s in steps]),
                "advantage": adv_batch,
                "returns": self.returns[idx],
            }

    def clear(self) -> None:
        self._steps.clear()
        self.advantages = None
        self.returns = None
