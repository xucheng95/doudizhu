"""Doudizhu MAPPO training — persistent workers for sampling + PPO."""
from __future__ import annotations
import sys, os, argparse, time, copy
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'doudizhu', 'python', 'doudizhu'))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'doudizhu', 'python'))

import torch
from torch.utils.tensorboard import SummaryWriter

from training.config import TrainingConfig
from training.model import DoudizhuAgent
from training.workers import PersistentWorkers, ROLES
from training.history_pool import HistoryPool
from training.eval import evaluate


def _eval_worker(state_dicts: dict, cfg, epoch: int, device):
    """Background eval — loads model, runs games, prints result."""
    agents = {r: DoudizhuAgent(cfg).to(device) for r in ROLES}
    for r in ROLES:
        agents[r].load_state_dict(state_dicts[r])
    result = evaluate(agents, cfg, n_games=200, device=device)
    print(f"  [eval epoch {epoch}] landlord_wr={result['landlord_win_rate']:.3f} "
          f"(avg_len={result['avg_game_length']:.0f})", flush=True)


def build_agents(cfg: TrainingConfig, device: torch.device) -> dict[str, DoudizhuAgent]:
    return {role: DoudizhuAgent(cfg).to(device) for role in ROLES}


def save_checkpoint(agents: dict, epoch: int, cfg: TrainingConfig) -> None:
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    path = os.path.join(cfg.checkpoint_dir, f"epoch_{epoch:05d}.pt")
    torch.save({
        "epoch": epoch,
        "agents": {r: a.state_dict() for r, a in agents.items()},
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
    print(f"d_model={cfg.d_model}, layers={cfg.num_layers}, "
          f"ep_per_batch={cfg.episodes_per_batch}", flush=True)

    agents = build_agents(cfg, device)
    pool = HistoryPool(cfg)
    writer = SummaryWriter(cfg.log_dir)

    # --- persistent workers ---
    workers = PersistentWorkers(cfg)

    for epoch in range(cfg.max_epochs):
        t0 = time.time()

        # ---- collect samples (wait for prefetched batch from previous epoch) ----
        t_collect = time.time()
        if epoch > 0:
            all_steps = workers.collect_work()
        else:
            state_dicts = {r: {k: v.cpu() for k, v in agents[r].state_dict().items()}
                           for r in ROLES}
            workers.submit_work(cfg.episodes_per_batch, state_dicts, cfg)
            all_steps = workers.collect_work()
        dt_collect = time.time() - t_collect

        # ---- PPO + prefetch overlap ----
        t_overlap = time.time()
        state_dicts = {r: {k: v.cpu() for k, v in agents[r].state_dict().items()}
                       for r in ROLES}
        workers.submit_work(cfg.episodes_per_batch, state_dicts, cfg)

        for role in ROLES:
            sd = {k: v.cpu() for k, v in agents[role].state_dict().items()}
            steps_data = list(all_steps[role])
            workers.submit_learn(role, sd, steps_data, cfg)

        t_learn = time.time()
        for role in ROLES:
            _, _, new_sd, _, dt = workers.collect_learn(role)
            agents[role].load_state_dict(new_sd)
        dt_learn = time.time() - t_learn

        print(f"  collect={dt_collect:.1f}s submit+learn={dt_learn:.1f}s "
              f"PPO={max(dt_learn,0):.1f}s", flush=True)

        elapsed = time.time() - t0
        steps = {r: len(all_steps.get(r, [])) for r in ROLES}
        print(f"Epoch {epoch:5d} | {elapsed:.1f}s | steps: {steps}", flush=True)

        # ---- evaluate (background process, non-blocking) ----
        if epoch % cfg.eval_interval == 0:
            print(f"  starting background eval...", flush=True)
            sd = {r: {k: v.cpu().clone() for k, v in agents[r].state_dict().items()}
                  for r in ROLES}
            ctx = __import__('torch').multiprocessing.get_context("spawn")
            p = ctx.Process(target=_eval_worker, args=(sd, cfg, epoch, device))
            p.start()

        # ---- checkpoint ----
        if epoch % cfg.checkpoint_interval == 0:
            save_checkpoint(agents, epoch, cfg)
            for role in ROLES:
                pool.save(epoch, role, copy.deepcopy(agents[role].state_dict()))

    workers.shutdown()
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
