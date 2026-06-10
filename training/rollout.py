from __future__ import annotations
import sys, os
import torch

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'doudizhu', 'python', 'doudizhu'))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'doudizhu', 'python'))
from doudizhu_cpp import EnvConfig
from env_wrapper import DoudizhuGymEnv

from training.config import TrainingConfig
from training.model import DoudizhuAgent
from training.obs_encoder import encode_obs

_ROLE_OF = {0: "peasant0", 1: "landlord", 2: "peasant1"}


def collect_episode(
    agents: dict[str, DoudizhuAgent],
    cfg: TrainingConfig,
    device: torch.device,
    opponent_agents: dict[str, DoudizhuAgent] | None = None,
) -> dict[str, list[dict]]:
    """Run one complete self-play episode, return per-role trajectories."""
    env_config = EnvConfig()
    env_config.self_play = True
    gym_env = DoudizhuGymEnv(config=env_config)

    obs, _ = gym_env.reset()
    done = False
    active = opponent_agents if opponent_agents is not None else agents

    trajectories: dict[str, list[dict]] = {k: [] for k in agents}

    with torch.no_grad():
        while not done:
            player_idx = gym_env._env.game.current_player()
            role_key = _ROLE_OF[player_idx]
            # Use active (possibly historical) agent for inference
            agent = active.get(role_key, agents[role_key])

            enc = encode_obs(obs, cfg, device)
            batch = {k: v.unsqueeze(0) for k, v in enc.items()
                     if isinstance(v, torch.Tensor) and v.dim() >= 1}
            batch["num_legal"] = enc["num_legal"].unsqueeze(0)

            action, log_prob, entropy, value = agent.act(
                batch["hand"], batch["num_cards"], batch["role"],
                batch["landlord_cards"], batch["history"], batch["history_mask"],
                batch["all_hands"], batch["move_type"], batch["move_rank"],
                batch["move_length"], batch["move_kickers"], batch["move_cards"],
                batch["num_legal"],
            )
            action_idx = action.squeeze(0).item()

            obs, _reward, done, _, _ = gym_env.step(int(action_idx))

            def _unbatch(k: str, v: torch.Tensor) -> torch.Tensor:
                # history is [T, 62] — never squeeze it
                if k == "history" or k == "history_mask":
                    return v.cpu()
                return v.squeeze(0).cpu()

            step = {
                **{k: _unbatch(k, v) for k, v in enc.items()
                   if isinstance(v, torch.Tensor)},
                "action": action.squeeze(0).cpu(),
                "log_prob": log_prob.squeeze(0).cpu(),
                "value": value.squeeze(0).cpu(),
                "reward": 0.0,
                "done": done,
            }
            trajectories[role_key].append(step)

    # Backfill terminal rewards and done flag for all roles using per-player rewards.
    for player_idx, role_key in _ROLE_OF.items():
        steps = trajectories[role_key]
        if steps:
            steps[-1]["reward"] = float(gym_env._env.game.reward(player_idx))
            steps[-1]["done"] = True

    return trajectories


def collect_batch(
    agents: dict[str, DoudizhuAgent],
    cfg: TrainingConfig,
    device: torch.device,
    n_episodes: int,
    opponent_agents: dict[str, DoudizhuAgent] | None = None,
) -> dict[str, list[dict]]:
    """Collect n_episodes and concatenate per-role trajectories."""
    all_steps: dict[str, list[dict]] = {"landlord": [], "peasant0": [], "peasant1": []}
    for _ in range(n_episodes):
        ep = collect_episode(agents, cfg, device, opponent_agents)
        for role, steps in ep.items():
            all_steps[role].extend(steps)
    return all_steps
