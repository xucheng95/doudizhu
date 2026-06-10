#!/usr/bin/env python3
"""Test script for Doudizhu environment."""

import sys
import os
# Add the directory containing the .so to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'doudizhu'))

import numpy as np
from doudizhu.env_wrapper import DoudizhuGymEnv, doudizhu_env_creator

# Test 1: Basic environment cycle
print("=" * 60)
print("Test 1: Basic environment cycle (single agent)")
print("=" * 60)

env = DoudizhuGymEnv()
obs, info = env.reset()
print(f"Observation keys: {list(obs.keys())}")
print(f"Hand shape: {obs['hand'].shape}")
print(f"Hand sum (cards in hand): {obs['hand'].sum()}")
print(f"Num cards: {obs['num_cards']}")
print(f"Role: {obs['role']}")
print(f"Action mask sum (legal actions): {obs['action_mask'].sum()}")
print(f"Last move rank: {obs['last_move_rank']}")
print(f"Last player: {obs['last_player']}")

# Test 2: Step through a few moves
print("\n" + "=" * 60)
print("Test 2: Stepping through moves")
print("=" * 60)

total_reward = 0
for i in range(100):  # Cap at 100 moves
    action_mask = obs['action_mask']
    legal_indices = np.where(action_mask > 0)[0]
    
    if len(legal_indices) == 0:
        print(f"Step {i}: No legal actions! Breaking.")
        break
    
    # Choose a random legal action
    action = np.random.choice(legal_indices)
    
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    
    if i < 5 or terminated:
        hand_count = obs['hand'].sum()
        mask_sum = obs['action_mask'].sum()
        print(f"Step {i}: action={action}, hand={hand_count:.0f}, legal={mask_sum:.0f}, "
              f"reward={reward:.1f}, done={terminated}")
    
    if terminated:
        print(f"\nGame over after {i+1} moves!")
        print(f"Winner: player {info['winner']}")
        print(f"Total reward: {total_reward}")
        break

env.close()

# Test 3: Benchmark speed
print("\n" + "=" * 60)
print("Test 3: Speed benchmark (100 games)")
print("=" * 60)

import time

env2 = DoudizhuGymEnv()
num_games = 100
total_moves = 0
start = time.time()

for game in range(num_games):
    obs, _ = env2.reset()
    done = False
    moves = 0
    while not done:
        action_mask = obs['action_mask']
        legal_indices = np.where(action_mask > 0)[0]
        if len(legal_indices) == 0:
            break
        action = np.random.choice(legal_indices)
        obs, reward, done, truncated, info = env2.step(action)
        moves += 1
    total_moves += moves

elapsed = time.time() - start
print(f"Played {num_games} games ({total_moves} total moves) in {elapsed:.2f}s")
print(f"Average: {total_moves/elapsed:.0f} moves/s, {num_games/elapsed:.1f} games/s")
print(f"Average moves per game: {total_moves/num_games:.1f}")

env2.close()

# Test 4: Self-play mode
print("\n" + "=" * 60)
print("Test 4: Self-play mode")
print("=" * 60)

from doudizhu_cpp import EnvConfig

config = EnvConfig()
config.self_play = True
config.use_auction = True

env3 = DoudizhuGymEnv(config)
obs, _ = env3.reset()
print(f"Self-play obs hand: {obs['hand'].sum():.0f} cards")
print(f"Role: {obs['role']}")

# In self-play mode, step() just advances one player at a time
for i in range(3):
    action_mask = obs['action_mask']
    legal_indices = np.where(action_mask > 0)[0]
    if len(legal_indices) > 0:
        action = np.random.choice(legal_indices)
        obs, reward, done, truncated, info = env3.step(action)
        print(f"Self-play step {i}: hand={obs['hand'].sum():.0f}, done={done}")

env3.close()

print("\n✅ All tests passed!")
