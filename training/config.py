from __future__ import annotations
from dataclasses import dataclass, asdict
import logging
import warnings
import yaml


@dataclass
class TrainingConfig:
    # Model
    d_model: int = 256
    d_action: int = 64
    num_layers: int = 4
    num_heads: int = 4
    ff_dim: int = 512
    dropout: float = 0.1
    max_actions: int = 500
    max_history: int = 60

    # PPO
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_coeff: float = 0.5
    entropy_coeff: float = 0.01
    lr: float = 3e-4
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    batch_size: int = 256

    # Rollout
    num_workers: int = 8
    episodes_per_batch: int = 256

    # Training
    max_epochs: int = 10000
    eval_interval: int = 50
    checkpoint_interval: int = 100
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "runs/doudizhu"

    # Self-play
    history_pool_max_size: int = 100
    history_pool_start_epoch: int = 500
    history_ratio_phase2: float = 0.2
    history_ratio_phase3: float = 0.5

    # Reward (informational — actual rewards come from C++ env)
    landlord_win_reward: float = 2.0
    landlord_lose_reward: float = -2.0
    peasant_win_reward: float = 1.0
    peasant_lose_reward: float = -1.0

    @classmethod
    def from_yaml(cls, path: str) -> TrainingConfig:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        unknown = [k for k in data if k not in cls.__dataclass_fields__]
        if unknown:
            warnings.warn(f"Unknown config keys ignored: {unknown}", stacklevel=2)
        defaults = asdict(cls())
        defaults.update(data)
        return cls(**{k: v for k, v in defaults.items() if k in cls.__dataclass_fields__})
