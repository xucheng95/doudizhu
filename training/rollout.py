"""Parallel rollout for MAPPO training.

Uses ProcessPoolExecutor to run episodes across multiple CPU workers.
Each worker owns its own C++ env + model copy (CPU inference).
"""

from __future__ import annotations
import sys, os, copy
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'doudizhu', 'python', 'doudizhu'))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'doudizhu', 'python'))
from doudizhu_cpp import EnvConfig
from env_wrapper import DoudizhuGymEnv

from training.config import TrainingConfig
from training.model import DoudizhuAgent
from training.obs_encoder import encode_obs

_ROLE_OF = {0: "peasant0", 1: "landlord", 2: "peasant1"}


# ============================================================
# Top-level worker function (must be picklable for ProcessPool)
# ============================================================

def _worker_episodes(args: tuple) -> dict[str, list[dict]]:
    """Run n episodes on a single worker with its own env + model."""
    n_eps, state_dicts, cfg = args

    # Each worker imports its own env (C++ cannot be pickled)
    from doudizhu_cpp import EnvConfig as EC
    from env_wrapper import DoudizhuGymEnv as GE
    ec = EC()
    ec.self_play = True
    env = GE(config=ec)

    device = torch.device("cpu")
    agents = {
        role: DoudizhuAgent(cfg).to(device)
        for role in _ROLE_OF.values()
    }
    for role in _ROLE_OF.values():
        agents[role].load_state_dict(state_dicts[role])
        agents[role].eval()

    all_steps = {"landlord": [], "peasant0": [], "peasant1": []}
    for _ in range(n_eps):
        ep = _collect_one_episode(env, agents, cfg, device)
        for role, steps in ep.items():
            all_steps[role].extend(steps)
    return all_steps


# ============================================================
# Single-episode collection
# ============================================================

def _collect_one_episode(env, agents, cfg, device):
    """Run one self-play episode, return per-role trajectories."""
    obs, _ = env.reset()
    done = False
    trajectories = {r: [] for r in _ROLE_OF.values()}

    with torch.no_grad():
        while not done:
            player_idx = env._env.game.current_player()
            role_key = _ROLE_OF[player_idx]
            agent = agents[role_key]

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
            obs, _, done, _, _ = env.step(int(action_idx))

            def _unbatch(k, v):
                if k in ("history", "history_mask"):
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

    for player_idx, role_key in _ROLE_OF.items():
        steps = trajectories[role_key]
        if steps:
            steps[-1]["reward"] = float(env._env.game.reward(player_idx))
            steps[-1]["done"] = True

    return trajectories


# ============================================================
# Public API
# ============================================================

def collect_batch(
    agents: dict[str, DoudizhuAgent],
    cfg: TrainingConfig,
    device: torch.device,
    n_episodes: int,
    opponent_agents: dict[str, DoudizhuAgent] | None = None,
) -> dict[str, list[dict]]:
    """Collect n_episodes. Parallel if cfg.num_workers > 1."""
    n_workers = max(1, cfg.num_workers)
    if n_workers > 1 and n_episodes > 1:
        print(f"  Rolling out {n_episodes} episodes with {n_workers} workers...", flush=True)
        return _collect_parallel(agents, cfg, n_episodes)
    return _collect_sequential(agents, cfg, device, n_episodes, opponent_agents)


def cpu_device():
    return torch.device("cpu")


def _collect_sequential(agents, cfg, device, n_eps, opponent_agents):
    all_steps = {"landlord": [], "peasant0": [], "peasant1": []}
    ec = EnvConfig()
    ec.self_play = True
    env = DoudizhuGymEnv(config=ec)
    for i in range(n_eps):
        ep = _collect_one_episode(env, agents, cfg, device)
        for role, steps in ep.items():
            all_steps[role].extend(steps)
        if (i + 1) % 50 == 0:
            print(f"  rollout {i+1}/{n_eps} episodes", flush=True)
    return all_steps


def _collect_parallel(agents, cfg, n_eps):
    n_workers = max(1, cfg.num_workers)
    state_dicts = {r: {k: v.cpu().clone() for k, v in agents[r].state_dict().items()}
                   for r in _ROLE_OF.values()}

    # Distribute episodes
    per_worker = [n_eps // n_workers] * n_workers
    per_worker[-1] += n_eps % n_workers

    args_list = [(n, state_dicts, cfg) for n in per_worker if n > 0]

    all_steps = {"landlord": [], "peasant0": [], "peasant1": []}
    with ProcessPoolExecutor(max_workers=len(args_list)) as executor:
        futures = [executor.submit(_worker_episodes, args) for args in args_list]
        for f in as_completed(futures):
            result = f.result()
            for role in _ROLE_OF.values():
                all_steps[role].extend(result[role])

    return all_steps
