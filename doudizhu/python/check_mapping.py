import sys, os
PKG = os.path.join(os.path.dirname(__file__), 'doudizhu')
sys.path.insert(0, PKG)  # for doudizhu_cpp.so
sys.path.insert(0, os.path.dirname(PKG))  # for env_wrapper
import numpy as np

from doudizhu_cpp import DoudizhuEnv, EnvConfig, MoveType, NUM_RANKS
# Rank constants (from cards.hpp)
RANK_3, RANK_A, RANK_SJ, RANK_BJ = 0, 11, 13, 14
from env_wrapper import find_action_idx, ACTION_LOOKUP, ACTION_TABLE

# Check: straight range in Python vs C++
print("=== STRAIGHT 对比 ===")
# C++ side (correct):
#   for len in 5..12:
#       for start in RANK_3 .. RANK_A - len + 1:
# Python side (build_action_table):
#   for length in 5..13:
#       for start in range(0, 12 - length + 1 + 1):
print("C++ (correct):")
cpp_straights = set()
for length in range(5, 13):
    max_start = RANK_A - length + 1  # 11 - length + 1 = 12 - length
    for start in range(0, max_start + 1):
        cpp_straights.add((MoveType.STRAIGHT, start, length))
        if length >= 11:
            print(f"  STRAIGHT rank={start} len={length}")

print("\nPython table:")
py_straights = set()
for length in range(5, 13):
    for start in range(0, 12 - length + 1 + 1):
        tup = (MoveType.STRAIGHT, start, length)
        py_straights.add(tup)
        if length >= 11:
            ok = "✓" if tup in ACTION_LOOKUP else "✗"
            print(f"  STRAIGHT rank={start} len={length}  {ok}")

missing = cpp_straights - py_straights
extra = py_straights - cpp_straights
print(f"\nC++有但Python缺: {len(missing)}")
for m in sorted(missing, key=lambda x: (x[2], x[1])):
    print(f"  STRAIGHT rank={m[1]} len={m[2]}")
print(f"Python有但C++没有: {len(extra)}")

print("\n\n=== DOUBLE_STRAIGHT 对比 ===")
cpp_ds = set()
for length in range(3, 11):
    max_start = RANK_A - length + 1
    for start in range(0, max_start + 1):
        cpp_ds.add((MoveType.DOUBLE_STRAIGHT, start, length))

py_ds = set()
for length in range(3, 11):
    for start in range(0, 12 - length + 1 + 1):
        py_ds.add((MoveType.DOUBLE_STRAIGHT, start, length))

missing_ds = cpp_ds - py_ds
print(f"C++有但Python缺: {len(missing_ds)}")
for m in sorted(missing_ds, key=lambda x: (x[2], x[1])):
    print(f"  DOUBLE_STRAIGHT rank={m[1]} len={m[2]}")

print("\n\n=== PLANE 对比 ===")
cpp_pl = set()
for length in range(2, 7):
    max_start = RANK_A - length + 1
    for start in range(0, max_start + 1):
        for mt in [MoveType.PLANE, MoveType.PLANE_WING_SINGLE, MoveType.PLANE_WING_PAIR]:
            cpp_pl.add((mt, start, length))

py_pl = set()
for length in range(2, 7):
    for start in range(0, 12 - length + 1 + 1):
        py_pl.add((MoveType.PLANE, start, length))
        py_pl.add((MoveType.PLANE_WING_SINGLE, start, length))
        py_pl.add((MoveType.PLANE_WING_PAIR, start, length))

missing_pl = cpp_pl - py_pl
print(f"C++有但Python缺: {len(missing_pl)}")
for m in sorted(missing_pl, key=lambda x: (x[2], x[1])):
    print(f"  {m[0].name} rank={m[1]} len={m[2]}")

print("\n\n=== ROCKET 对比 ===")
print(f"ROCKET in LOOKUP: {(MoveType.ROCKET, 13, 0) in ACTION_LOOKUP}")
print(f"ROCKET C++: rank=RANK_BJ({RANK_BJ}), len=0")
print(f"Key in table: check table[171] = {ACTION_TABLE[171]}")
