#!/usr/bin/env python3
"""Full test for Doudizhu environment."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'doudizhu'))

import time
import random

# Test 1: Basic C++ random game
print("=" * 60)
print("Test 1: C++ environment - one random game")
print("=" * 60)

from doudizhu_cpp import DoudizhuEnv, EnvConfig

config = EnvConfig()
env = DoudizhuEnv(config)

obs = env.reset()
hand_count = sum(obs.hand)
print(f"Initial hand: {hand_count:.0f} cards")

done = False
moves = 0
while not done:
    actions = env.legal_actions()
    if not actions:
        print("No legal actions - breaking")
        break
    idx = random.randint(0, len(actions) - 1)
    result = env.step(idx)
    moves += 1
    done = result.terminated
    if moves > 200:
        print("Too many moves - breaking")
        break

print(f"Game finished in {moves} moves, winner={env.game.winner()}, reward={result.reward}")

# Test 2: Speed benchmark
print("\n" + "=" * 60)
print("Test 2: Speed benchmark (200 games)")
print("=" * 60)

env2 = DoudizhuEnv(EnvConfig())
num_games = 200
total_moves = 0
start = time.time()

for game in range(num_games):
    obs = env2.reset()
    done = False
    while not done:
        actions = env2.legal_actions()
        if not actions:
            break
        idx = random.randint(0, len(actions) - 1)
        result = env2.step(idx)
        done = result.terminated
        total_moves += 1

elapsed = time.time() - start
print(f"Played {num_games} games ({total_moves} total moves) in {elapsed:.2f}s")
print(f"Speed: {total_moves/elapsed:.0f} moves/s, {num_games/elapsed:.1f} games/s")
print(f"Avg moves/game: {total_moves/num_games:.1f}")

# Test 3: Gymnasium wrapper
print("\n" + "=" * 60)
print("Test 3: Gymnasium wrapper - basic ops")
print("=" * 60)

from doudizhu.env_wrapper import DoudizhuGymEnv
import numpy as np

env3 = DoudizhuGymEnv()
obs, info = env3.reset()
print(f"Obs keys: {list(obs.keys())}")
print(f"Hand sum: {obs['hand'].sum():.0f}")
print(f"Action mask sum: {obs['action_mask'].sum():.0f}")
print(f"Num cards: {obs['num_cards']}")
print(f"Role: {obs['role']}")

# Step 3 random moves
for i in range(3):
    mask = obs['action_mask']
    legal = np.where(mask > 0)[0]
    if len(legal) == 0:
        break
    action = np.random.choice(legal)
    obs, reward, done, truncated, info = env3.step(action)
    print(f"  Step {i}: action={action}, hand={obs['hand'].sum():.0f}, done={done}")

env3.close()

# Test 4: Multiple games via Gymnasium
print("\n" + "=" * 60)
print("Test 4: 100 games via Gymnasium wrapper")
print("=" * 60)

env4 = DoudizhuGymEnv()
num_games = 100
total_moves = 0
wins = 0
start = time.time()

for game in range(num_games):
    obs, _ = env4.reset()
    done = False
    while not done:
        mask = obs['action_mask']
        legal = np.where(mask > 0)[0]
        if len(legal) == 0:
            break
        action = np.random.choice(legal)
        obs, reward, done, truncated, info = env4.step(action)
        total_moves += 1
    if reward > 0:
        wins += 1

elapsed = time.time() - start
print(f"Played {num_games} games ({total_moves} moves) in {elapsed:.2f}s")
print(f"Speed: {total_moves/elapsed:.0f} moves/s, {num_games/elapsed:.1f} games/s")
print(f"Win rate (random agent): {wins/num_games:.1%}")

env4.close()
print("\n✅ All tests passed!")
