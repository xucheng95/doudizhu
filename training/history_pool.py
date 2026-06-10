from __future__ import annotations
import copy
import random
from training.config import TrainingConfig
from training.model import DoudizhuAgent


class HistoryPool:
    """Stores historical model checkpoints for self-play opponent sampling."""

    def __init__(self, cfg: TrainingConfig):
        self.cfg = cfg
        self._pool: dict[str, list[tuple[int, dict]]] = {
            "landlord": [], "peasant0": [], "peasant1": []
        }

    def save(self, epoch: int, role: str, state_dict: dict) -> None:
        """Add checkpoint; prune to max_size with even spacing."""
        self._pool[role].append((epoch, copy.deepcopy(state_dict)))
        if len(self._pool[role]) > self.cfg.history_pool_max_size:
            pool = self._pool[role]
            n = self.cfg.history_pool_max_size
            step = max(1, len(pool) // n)
            self._pool[role] = pool[::step][-n:]

    def size(self, role: str) -> int:
        return len(self._pool[role])

    def sample(self, role: str) -> dict | None:
        pool = self._pool[role]
        return random.choice(pool)[1] if pool else None

    def load_into(self, agent: DoudizhuAgent, role: str) -> bool:
        state_dict = self.sample(role)
        if state_dict is None:
            return False
        agent.load_state_dict(state_dict)
        return True
