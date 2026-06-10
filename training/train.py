"""Main MAPPO training loop for Doudizhu."""
from __future__ import annotations
import sys, os, argparse, time, copy
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'doudizhu', 'python', 'doudizhu'))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'doudizhu', 'python'))

import torch
from torch.utils.tensorboard import SummaryWriter

from doudizhu_cpp import EnvConfig

from training.config import TrainingConfig
from training.model import DoudizhuAgent
from training.buffer import RolloutBuffer
from training.ppo import PPOUpdater
from training.rollout import collect_batch
from training.history_pool import HistoryPool
from training.eval import evaluate


ROLES = ["landlord", "peasant0", "peasant1"]


def build_agents(cfg: TrainingConfig, device: torch.device) -> dict[str, DoudizhuAgent]:
    return {role: DoudizhuAgent(cfg).to(device) for role in ROLES}


def save_checkpoint(agents: dict, updaters: dict, epoch: int, cfg: TrainingConfig) -> None:
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    path = os.path.join(cfg.checkpoint_dir, f"epoch_{epoch:05d}.pt")
    torch.save({
        "epoch": epoch,
        "agents": {r: a.state_dict() for r, a in agents.items()},
        "optimizers": {r: u.optimizer.state_dict() for r, u in updaters.items()},
    }, path)
    print(f"Checkpoint saved: {path}")


def _self_play_phase(epoch: int, cfg: TrainingConfig, pool: HistoryPool) -> float:
    if epoch < cfg.history_pool_start_epoch:
        return 0.0
    if pool.size("landlord") > 20:
        return cfg.history_ratio_phase3
    return cfg.history_ratio_phase2


def train(cfg: TrainingConfig) -> None:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Training on {device}", flush=True)
    print(f"Config: d_model={cfg.d_model}, layers={cfg.num_layers}, "
          f"ep_per_batch={cfg.episodes_per_batch}, ppo_epochs={cfg.ppo_epochs}",
          flush=True)

    agents = build_agents(cfg, device)
    updaters = {r: PPOUpdater(agents[r], cfg) for r in ROLES}
    buffers = {r: RolloutBuffer(cfg) for r in ROLES}
    pool = HistoryPool(cfg)
    writer = SummaryWriter(cfg.log_dir)

    for epoch in range(cfg.max_epochs):
        t0 = time.time()

        # Build opponent agents (historical mix)
        hist_ratio = _self_play_phase(epoch, cfg, pool)
        opponent_agents = None
        if hist_ratio > 0.0 and pool.size("landlord") > 0:
            import random
            opponent_agents = {}
            for role in ROLES:
                if random.random() < hist_ratio:
                    hist_agent = DoudizhuAgent(cfg).to(device)
                    pool.load_into(hist_agent, role)
                    opponent_agents[role] = hist_agent
                else:
                    opponent_agents[role] = agents[role]

        # Collect rollouts
        print(f"Epoch {epoch}: collecting...", end=" ", flush=True)
        roll_device = device if cfg.num_workers <= 1 else torch.device("cpu")
        all_steps = collect_batch(
            agents, cfg, roll_device,
            n_episodes=cfg.episodes_per_batch,
            opponent_agents=opponent_agents,
        )

        # Fill buffers and compute GAE
        for role in ROLES:
            buffers[role].clear()
            for step in all_steps[role]:
                buffers[role].add(step)
            buffers[role].compute_gae(last_value=torch.tensor(0.0))

        # PPO updates
        for role in ROLES:
            t0 = time.time()
            print(f"  PPO {role}...", end=" ", flush=True)
            metrics = updaters[role].update(buffers[role])
            dt = time.time() - t0
            writer.add_scalar(f"{role}/policy_loss", metrics["policy_loss"], epoch)
            writer.add_scalar(f"{role}/value_loss", metrics["value_loss"], epoch)
            writer.add_scalar(f"{role}/entropy", metrics["entropy"], epoch)
            updaters[role].step_scheduler()
            print(f"done ({dt:.1f}s)", flush=True)

        elapsed = time.time() - t0
        writer.add_scalar("train/epoch_time_s", elapsed, epoch)

        # Print progress every epoch
        steps = {r: len(all_steps.get(r, [])) for r in ROLES}
        print(f"Epoch {epoch:5d} | {elapsed:.1f}s | steps: {steps} | "
              f"p_loss={metrics['policy_loss']:.4f} v_loss={metrics['value_loss']:.4f} "
              f"ent={metrics['entropy']:.3f}", flush=True)

        # Evaluate
        if epoch % cfg.eval_interval == 0:
            result = evaluate(agents, cfg, n_games=200, device=device)
            writer.add_scalar("eval/landlord_win_rate", result["landlord_win_rate"], epoch)
            writer.add_scalar("eval/peasant_win_rate", result["peasant_win_rate"], epoch)
            writer.add_scalar("eval/avg_game_length", result["avg_game_length"], epoch)
            print(f"Epoch {epoch:5d} | landlord_wr={result['landlord_win_rate']:.3f} "
                  f"| {elapsed:.1f}s")

        # Checkpoint + history pool
        if epoch % cfg.checkpoint_interval == 0:
            save_checkpoint(agents, updaters, epoch, cfg)
            for role in ROLES:
                pool.save(epoch, role, copy.deepcopy(agents[role].state_dict()))

    writer.close()
    print("Training complete.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = TrainingConfig.from_yaml(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
