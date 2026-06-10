"""Persistent processes: learners (PPO training) and workers (sampling).

- Learners: one per role (landlord/peasant0/peasant1), created once, reuse model/buffer
- Workers: create env once, reuse across epochs

Communication via multiprocessing.Queue.
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
# Persistent Learner (PPO)
# ============================================================

def _learner_loop(role: str, task_queue: mp.Queue, result_queue: mp.Queue):
    """Persistent learner: creates agent once, handles tasks in a loop."""
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
# Persistent Worker (sampling)
# ============================================================

def _worker_loop(task_queue: mp.Queue, result_queue: mp.Queue):
    """Persistent worker: creates env+model once, handles tasks in loop."""
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
    """Manages persistent learners (3) and workers (N) for async training."""

    def __init__(self, cfg: TrainingConfig):
        self.cfg = cfg
        self.n_workers = max(1, cfg.num_workers)
        ctx = mp.get_context("spawn")

        # Learners — one per role
        self._learner_tasks = {r: ctx.Queue() for r in ROLES}
        self._learner_results = {r: ctx.Queue() for r in ROLES}
        self._learner_procs = []
        for role in ROLES:
            p = ctx.Process(target=_learner_loop,
                            args=(role, self._learner_tasks[role], self._learner_results[role]))
            p.start()
            self._learner_procs.append(p)

        # Workers — sampling
        self._worker_tasks = ctx.Queue()
        self._worker_results = ctx.Queue()
        self._worker_procs = []
        for _ in range(self.n_workers):
            p = ctx.Process(target=_worker_loop,
                            args=(self._worker_tasks, self._worker_results))
            p.start()
            self._worker_procs.append(p)

    def submit_work(self, n_eps: int, state_dicts: dict, cfg: TrainingConfig):
        """Distribute sampling tasks to all workers."""
        per_worker = [n_eps // self.n_workers] * self.n_workers
        per_worker[-1] += n_eps % self.n_workers
        for n in per_worker:
            if n > 0:
                self._worker_tasks.put((n, state_dicts, cfg))

    def collect_work(self) -> dict:
        """Collect sampling results from all workers, merge."""
        all_steps = {"landlord": [], "peasant0": [], "peasant1": []}
        for _ in range(self.n_workers):
            result = self._worker_results.get()
            for role in _ROLE_OF.values():
                all_steps[role].extend(result[role])
        return all_steps

    def submit_learn(self, role: str, state_dict: dict, steps_data: list, cfg: TrainingConfig):
        """Send training task to one learner."""
        self._learner_tasks[role].put((state_dict, steps_data, cfg))

    def collect_learn(self, role: str) -> tuple:
        """Wait for training result from one learner."""
        return self._learner_results[role].get()

    def shutdown(self):
        for role in ROLES:
            self._learner_tasks[role].put(None)
        for _ in self._worker_procs:
            self._worker_tasks.put(None)
        for p in self._learner_procs + self._worker_procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
