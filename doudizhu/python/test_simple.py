import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'doudizhu'))

print("Step 1: importing doudizhu_cpp...")
from doudizhu_cpp import (
    DoudizhuEnv, EnvConfig, MoveType, Game,
    NUM_CARDS, NUM_PLAYERS
)
print("OK - imported successfully")

print("\nStep 2: creating environment...")
config = EnvConfig()
env = DoudizhuEnv(config)
print("OK - env created")

print("\nStep 3: resetting...")
obs = env.reset()
print(f"OK - reset. Hand sum: {sum(obs.hand)}")

print("\nStep 4: getting legal actions...")
actions = env.legal_actions()
print(f"OK - {len(actions)} legal actions")
for i, m in enumerate(actions[:5]):
    print(f"  {i}: type={m.type}, rank={m.rank}, len={m.length}")

print("\nStep 5: stepping...")
result = env.step(0)
print(f"OK - step done. term={result.terminated}, reward={result.reward}")

print("\nAll basic tests passed!")
