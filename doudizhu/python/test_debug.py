import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'doudizhu'))

print("--- Debug: C++ only ---")
from doudizhu_cpp import DoudizhuEnv, EnvConfig

config = EnvConfig()
env = DoudizhuEnv(config)
print(f"config created: use_auction={config.use_auction}, self_play={config.self_play}")

obs = env.reset()
print(f"reset done: hand sum = {sum(obs.hand)}")

import random
print("starting game loop...")
for step in range(300):
    actions = env.legal_actions()
    if not actions:
        print(f"step {step}: no actions!")
        break
    idx = random.randint(0, len(actions)-1)
    result = env.step(idx)
    if step == 0:
        print(f"first step: reward={result.reward}, term={result.terminated}")
    if result.terminated:
        print(f"game over at step {step+1}, winner={env.game.winner()}")
        break
else:
    print("loop finished without termination!")

print("C++ test OK")
