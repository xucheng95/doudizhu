"""Persistent worker processes for PPO training and sampling.

- PPO workers: one per role (landlord/peasant0/peasant1), created once, reuse model/buffer
- Sampling workers: create env once, reuse across epochs

Communication via multiprocessing.Queue / Pipe.
"""

from __future__ import annotations
import sys, os, time, copy, pickle
import torch
import torch.multiprocessing as mp

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'doudizhu', 'python', 'doudizhu'))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'doudizhu', 'python'))

from training.config import TrainingConfig
from training.model import DoudizhuAgent
from training.buffer import RolloutBuffer
from training.ppo import PPOUpdater
from training.rollout import _collect_one_episode
from doudizhu_cpp import EnvConfig
from env_wrapper import DoudizhuGymEnv

_ROLE_OF = {0: "peasant0", 1: "landlord", 2: "peasant1"}
ROLES = ["landlord", "peasant0", "peasant1"]


# ============================================================
# Persistent PPO Worker
# ============================================================

def _ppo_worker_loop(role: str, task_queue: mp.Queue, result_queue: mp.Queue):
    """Persistent PPO worker: creates agent once, handles tasks in a loop."""
    # Dummy cfg — actual cfg sent with each task
    agent = None
    buffer = None
    updater = None

    while True:
        task = task_queue.get()
        if task is None:
            break

        state_dict, steps_data, cfg = task

        if agent is None:
            device = torch.device("cpu")
            agent = DoudizhuAgent(cfg).to(device)
            buffer = RolloutBuffer(cfg)
            updater = PPOUpdater(agent, cfg)
        elif cfg.d_model != agent.cfg.d_model:
            # Rebuild if config changed
            device = torch.device("cpu")
            agent = DoudizhuAgent(cfg).to(device)
            buffer = RolloutBuffer(cfg)
            updater = PPOUpdater(agent, cfg)

        agent.load_state_dict(state_dict)
        buffer.clear()
        for step in steps_data:
            buffer.add(step)
        buffer.compute_gae(last_value=torch.tensor(0.0))

        t1 = time.time()
        metrics = updater.update(buffer)
        dt = time.time() - t1

        result = (role, metrics, agent.state_dict(), updater.optimizer.state_dict(), dt)
        result_queue.put(result)


# ============================================================
# Persistent Sampling Worker
# ============================================================

def _sample_worker_loop(task_queue: mp.Queue, result_queue: mp.Queue):
    """Persistent sampling worker: creates env+model once, handles tasks in loop."""
    from doudizhu_cpp import EnvConfig as EC
    from env_wrapper import DoudizhuGymEnv as GE

    ec = EC()
    ec.self_play = True
    env = GE(config=ec)

    agents = None
    device = torch.device("cpu")

    while True:
        task = task_queue.get()
        if task is None:
            break

        n_eps, state_dicts, cfg = task

        if agents is None:
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

        result_queue.put(all_steps)


# ============================================================
# Manager for the persistent workers
# ============================================================

class PersistentWorkers:
    """Manages persistent PPO (3) and sampling (N) workers."""

    def __init__(self, cfg: TrainingConfig):
        self.cfg = cfg
        self.n_samplers = max(1, cfg.num_workers)
        ctx = mp.get_context("spawn")

        # PPO workers (one per role)
        self._ppo_tasks = {r: ctx.Queue() for r in ROLES}
        self._ppo_results = {r: ctx.Queue() for r in ROLES}
        self._ppo_procs = []
        for role in ROLES:
            p = ctx.Process(target=_ppo_worker_loop,
                            args=(role, self._ppo_tasks[role], self._ppo_results[role]))
            p.start()
            self._ppo_procs.append(p)

        # Sampling workers
        self._sample_tasks = ctx.Queue()
        self._sample_results = ctx.Queue()
        self._sample_procs = []
        for _ in range(self.n_samplers):
            p = ctx.Process(target=_sample_worker_loop,
                            args=(self._sample_tasks, self._sample_results))
            p.start()
            self._sample_procs.append(p)

    def submit_sample(self, n_eps: int, state_dicts: dict, cfg: TrainingConfig):
        """Send sampling tasks to all workers, return list of futures (queues to read)."""
        per_worker = [n_eps // self.n_samplers] * self.n_samplers
        per_worker[-1] += n_eps % self.n_samplers
        for n in per_worker:
            if n > 0:
                self._sample_tasks.put((n, state_dicts, cfg))
        return self._sample_results

    def collect_samples(self, n_workers: int) -> dict:
        """Collect results from N sampling workers, merge into one dict."""
        all_steps = {"landlord": [], "peasant0": [], "peasant1": []}
        for _ in range(max(1, n_workers)):
            result = self._sample_results.get()
            for role in _ROLE_OF.values():
                all_steps[role].extend(result[role])
        return all_steps

    def submit_ppo(self, role: str, state_dict: dict, steps_data: list, cfg: TrainingConfig):
        """Send PPO task to one role worker."""
        self._ppo_tasks[role].put((state_dict, steps_data, cfg))

    def collect_ppo(self, role: str) -> tuple:
        """Wait for PPO result from one role worker."""
        return self._ppo_results[role].get()

    def shutdown(self):
        for role in ROLES:
            self._ppo_tasks[role].put(None)
        for _ in self._sample_procs:
            self._sample_tasks.put(None)
        for p in self._ppo_procs + self._sample_procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
