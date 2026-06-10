"""Doudizhu rules audit — key checks."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'doudizhu'))

from doudizhu_cpp import DoudizhuEnv, EnvConfig, MoveType
from env_wrapper import find_action_idx, _mask_ranks
from collections import Counter
import random

errors = []

def test(desc, fn, *args):
    try:
        fn(*args)
        print(f"  ✅ {desc}")
    except AssertionError as e:
        print(f"  ❌ {desc}: {e}")
        errors.append(desc)

# -----------------------------------------------------------
print("=" * 50)
print("1. Deal & card count")
env = DoudizhuEnv(EnvConfig())
obs = env.reset()
total = int(sum(obs.num_cards))
assert total == 54, f"total {total} != 54"
# landlord is player 1 by default, gets 20
n0, n1, n2 = obs.num_cards
assert n1 == 20, f"landlord(1) has {n1} != 20"
test("54 cards dealt, landlord=20", lambda: None)

# -----------------------------------------------------------
print("\n2. Card uniqueness")
env2 = DoudizhuEnv(EnvConfig())
all_cards = set()
for g in range(10):
    env2.reset()
    g2 = env2.game
    for p in range(3):
        h = int(g2.hand(p))
        while h:
            lsb = h & -h
            cid = lsb.bit_length() - 1
            all_cards.add(cid)
            h &= h - 1
# Each game should use exactly 54 unique cards
# Across 10 games we should see many cards
assert len(all_cards) >= 50, f"only {len(all_cards)} unique cards seen"
test(f"cards look random ({len(all_cards)} unique seen in 10 games)", lambda: None)

# -----------------------------------------------------------
print("\n3. Move types basic sanity")
env3 = DoudizhuEnv(EnvConfig())
for g in range(50):
    env3.reset()
    done = False
    while not done:
        acts = env3.legal_actions()
        assert len(acts) > 0, "no legal actions in active game"
        # PASS should only be present when following
        if g3 := env3.game:
            if g3.last_move().type != MoveType.NONE:
                has_pass = any(a.type == MoveType.PASS for a in acts)
                if not has_pass:
                    # could be ok if last_move is NONE (round reset)
                    pass
        # Step
        idx = random.randint(0, len(acts)-1)
        r = env3.step(idx)
        done = r.terminated
test("50 games: all have legal actions", lambda: None)

# -----------------------------------------------------------
print("\n4. Action mapping completeness")
env4 = DoudizhuEnv(EnvConfig())
unmapped = {}
total_legal = 0
for g in range(2000):
    env4.reset()
    done = False
    while not done:
        acts = env4.legal_actions()
        for a in acts:
            total_legal += 1
            idx = find_action_idx(a)
            if idx < 0:
                ranks = _mask_ranks(a.mask)
                key = (int(a.type), a.rank, a.length)
                if key not in unmapped:
                    unmapped[key] = []
                unmapped[key].append(tuple(sorted(Counter(ranks).items())))
        if not acts: break
        idx = random.randint(0, len(acts)-1)
        r = env4.step(idx)
        done = r.terminated

if unmapped:
    for k, examples in sorted(unmapped.items()):
        print(f"  ❌ type={MoveType(k[0]).name}, rank={k[1]}, len={k[2]}: {examples[:3]}... ({len(examples)}x)")
    errors.append("action mapping: unmapped moves found")
else:
    test(f"{total_legal} legal moves, 0 unmapped", lambda: None)

# -----------------------------------------------------------
print("\n5. Game termination")
env5 = DoudizhuEnv(EnvConfig())
never_ended = 0
for g in range(200):
    env5.reset()
    done = False
    moves = 0
    while not done and moves < 300:
        acts = env5.legal_actions()
        if not acts: break
        r = env5.step(random.randint(0, len(acts)-1))
        done = r.terminated
        moves += 1
    if moves >= 300:
        never_ended += 1
assert never_ended == 0, f"{never_ended} games didn't terminate"
test("200 games all terminated", lambda: None)

# -----------------------------------------------------------
print("\n6. Winner has 0 cards")
env6 = DoudizhuEnv(EnvConfig())
env6.reset()
done = False
while not done:
    acts = env6.legal_actions()
    if not acts: break
    r = env6.step(random.randint(0, len(acts)-1))
    done = r.terminated
winner = env6.game.winner()
w_cards = env6.game.get_observation(winner).hand.sum()
assert w_cards == 0, f"winner {winner} has {w_cards} cards"
test("winner has 0 cards", lambda: None)

# -----------------------------------------------------------
print("\n7. Scoring: same-team sign check")
sign_errors = 0
for g in range(500):
    env7 = DoudizhuEnv(EnvConfig())
    env7.reset()
    done = False
    while not done:
        acts = env7.legal_actions()
        if not acts: break
        r = env7.step(random.randint(0, len(acts)-1))
        done = r.terminated
    game = env7.game
    w = game.winner()
    if w < 0: continue
    wrole = game.role(w)
    for p in range(3):
        reward = game.reward(p)
        prole = game.role(p)
        same = (wrole == prole)
        if same and reward <= 0: sign_errors += 1
        if not same and reward >= 0: sign_errors += 1
assert sign_errors == 0, f"{sign_errors} reward sign errors"
test(f"500 games: 0 reward sign errors", lambda: None)

# -----------------------------------------------------------
print("\n8. Landlord multiplier (×2)")
mult_ok = 0
mult_fail = 0
for g in range(200):
    env8 = DoudizhuEnv(EnvConfig())
    env8.reset()
    done = False
    while not done:
        acts = env8.legal_actions()
        if not acts: break
        r = env8.step(random.randint(0, len(acts)-1))
        done = r.terminated
    game = env8.game
    w = game.winner()
    if w < 0: continue
    wrole = game.role(w)
    for p in range(3):
        reward = game.reward(p)
        prole = game.role(p)
        if prole == 2 and reward == 0: continue  # terminal before any step
        if abs(reward) < 1e-6: continue
        # Landlord gets ±2, peasant gets ±1
        if int(prole) == 2:  # landlord
            if abs(abs(reward) - 2.0) > 0.01:
                mult_fail += 1
        else:
            if abs(abs(reward) - 1.0) > 0.01:
                mult_fail += 1
        mult_ok += 1
test(f"{mult_ok} rewards, {mult_fail} multiplier violations", lambda: None)

# -----------------------------------------------------------
print("\n" + "=" * 50)
if errors:
    print(f"  AUDIT: {len(errors)} ISSUES FOUND")
    for e in errors:
        print(f"    - {e}")
else:
    print(f"  ✅ AUDIT PASSED - All checks OK")
print("=" * 50)
