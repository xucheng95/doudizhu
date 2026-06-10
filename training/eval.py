from __future__ import annotations
import sys, os
import torch

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'doudizhu', 'python', 'doudizhu'))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'doudizhu', 'python'))
from doudizhu_cpp import EnvConfig, Role
from env_wrapper import DoudizhuGymEnv

from training.config import TrainingConfig
from training.model import DoudizhuAgent
from training.obs_encoder import encode_obs

_ROLE_OF = {0: "peasant0", 1: "landlord", 2: "peasant1"}


@torch.no_grad()
def evaluate(
    agents: dict[str, DoudizhuAgent],
    cfg: TrainingConfig,
    n_games: int,
    device: torch.device,
) -> dict:
    """Run n_games with greedy (argmax) action selection."""
    for agent in agents.values():
        agent.eval()

    landlord_wins = 0
    total_length = 0

    env_config = EnvConfig()
    env_config.self_play = True
    gym_env = DoudizhuGymEnv(config=env_config)

    for _ in range(n_games):
        obs, _ = gym_env.reset()
        done = False
        game_len = 0

        while not done:
            player_idx = gym_env._env.game.current_player()
            role_key = _ROLE_OF[player_idx]
            agent = agents[role_key]

            enc = encode_obs(obs, cfg, device)
            batch = {k: v.unsqueeze(0) for k, v in enc.items()
                     if isinstance(v, torch.Tensor) and v.dim() >= 1}
            batch["num_legal"] = enc["num_legal"].unsqueeze(0)

            logits, _ = agent._forward_shared(
                batch["hand"], batch["num_cards"], batch["role"],
                batch["landlord_cards"], batch["history"], batch["history_mask"],
                batch["all_hands"], batch["move_type"], batch["move_rank"],
                batch["move_length"], batch["move_kickers"], batch["move_cards"],
                batch["num_legal"],
            )
            action_idx = logits.argmax(dim=-1).item()

            obs, _, done, _, _ = gym_env.step(int(action_idx))
            game_len += 1

        winner = gym_env._env.game.winner()
        if winner >= 0 and int(gym_env._env.game.role(winner)) == int(Role.LANDLORD):
            landlord_wins += 1
        total_length += game_len

    for agent in agents.values():
        agent.train()

    lw = landlord_wins / n_games
    return {
        "landlord_win_rate": lw,
        "peasant_win_rate": 1.0 - lw,
        "avg_game_length": total_length / n_games,
    }
