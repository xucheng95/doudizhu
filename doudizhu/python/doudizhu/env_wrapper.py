"""
Doudizhu Gymnasium Environment Wrapper

Observation (Dict):
    hands:          (3,54)  float32 — all 3 players' hand bits (row 0 = self, critic uses all)
    landlord_cards: (54,)   float32 — landlord's bonus cards
    num_cards:      (3,)    float32 — card count per player (reordered, index 0 = self)
    role:           (3,)    float32 — own role one-hot [PEASANT_0, PEASANT_1, LANDLORD]
    last_player:    int     — who played last
    history:        list    — all plays this round [{player, type, rank, length, kickers, cards}, ...]
    legal_moves:    list    — currently allowed plays [{type, rank, length, kickers, cards, pass}, ...]

Action: Discrete(2048) — index into legal_moves list (pick 0 .. len(legal_moves)-1)
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Dict, Any, List
from collections import Counter

from doudizhu_cpp import (
    DoudizhuEnv,
    EnvConfig,
    MoveType,
    NUM_CARDS,
    NUM_PLAYERS,
    NUM_RANKS,
)

_RANK_NAMES = ["3","4","5","6","7","8","9","10","J","Q","K","A","2","SJ","BJ"]

MAX_ACTIONS = 2048  # upper bound for legal_moves count

# ----- move helpers -----
def _mask_ranks(mask) -> list:
    """Extract sorted list of card ranks from a CardMask."""
    ranks = []
    m = int(mask)
    while m:
        lsb = m & -m
        cid = lsb.bit_length() - 1
        m &= m - 1
        if cid >= 52:
            ranks.append(13 + (cid - 52))
        else:
            ranks.append(cid // 4)
    return sorted(ranks)


def _kicker_ranks(move) -> list:
    """Return sorted kicker ranks for a move."""
    ranks = _mask_ranks(move.mask)
    cnt = Counter(ranks)
    t, r, l = move.type, move.rank, move.length
    if t == MoveType.PASS:
        return []
    # Remove primary cards
    if t in (MoveType.TRIPLE_ONE, MoveType.TRIPLE_TWO):
        cnt[r] -= 3
        if cnt[r] <= 0: del cnt[r]
    elif t in (MoveType.FOUR_TWO_SINGLE, MoveType.FOUR_TWO_PAIR):
        cnt[r] -= 4
        if cnt[r] <= 0: del cnt[r]
    elif t in (MoveType.PLANE_WING_SINGLE, MoveType.PLANE_WING_PAIR):
        for pr in range(r, r + l):
            cnt[pr] -= 3
            if cnt[pr] <= 0: del cnt[pr]
    return sorted(cnt.keys())


def _cards_str(mask) -> List[str]:
    """Mask → sorted list of rank names (no suits)."""
    ids = []
    m = int(mask)
    while m:
        lsb = m & -m
        ids.append(lsb.bit_length() - 1)
        m &= m - 1
    ranks = []
    for cid in sorted(ids):
        if cid >= 52:
            ranks.append("SJ" if cid == 52 else "BJ")
        else:
            ranks.append(_RANK_NAMES[cid // 4])
    return ranks


_KICKER_TYPES = {MoveType.TRIPLE_ONE, MoveType.TRIPLE_TWO,
                 MoveType.PLANE_WING_SINGLE, MoveType.PLANE_WING_PAIR,
                 MoveType.FOUR_TWO_SINGLE, MoveType.FOUR_TWO_PAIR}

def move_to_dict(move) -> dict:
    """C++ Move → human-readable dict."""
    if move.type == MoveType.PASS:
        return {"type": "PASS", "rank": -1, "length": 0,
                "kickers": [], "cards": [], "pass": True}
    return {
        "type": move.type.name,
        "rank": int(move.rank),
        "length": int(move.length),
        "kickers": _kicker_ranks(move) if move.type in _KICKER_TYPES else [],
        "cards": _cards_str(move.mask),
        "pass": False,
    }


def record_to_dict(rec) -> dict:
    """C++ PlayRecord → dict."""
    d = move_to_dict(rec.move)
    d["player"] = int(rec.player)
    return d


# ============================================================
class DoudizhuGymEnv(gym.Env):
    """Doudizhu with card-level legal-move list and play history."""

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(self, config: Optional[EnvConfig] = None, render_mode=None):
        super().__init__()
        self.config = config or EnvConfig()
        self._env = DoudizhuEnv(self.config)
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(2048)
        self.observation_space = spaces.Dict({
            "hands": spaces.Box(0, 1, (NUM_PLAYERS, NUM_CARDS), dtype=np.float32),
            "landlord_cards": spaces.Box(0, 1, (NUM_CARDS,), dtype=np.float32),
            "num_cards": spaces.Box(0, 17, (3,), dtype=np.float32),
            "role": spaces.Box(0, 1, (3,), dtype=np.float32),
            "last_player": spaces.Box(0, 2, (1,), dtype=np.int32),
            "history": spaces.Sequence(spaces.Dict({
                "player": spaces.Discrete(3),
                "type": spaces.Discrete(16),
                "rank": spaces.Discrete(15),
                "length": spaces.Discrete(13),
                "kickers": spaces.Sequence(spaces.Discrete(15)),
                "cards": spaces.Sequence(spaces.Discrete(54)),
                "pass": spaces.Discrete(2),
            })),
            "legal_moves": spaces.Sequence(spaces.Dict({
                "type": spaces.Discrete(16),
                "rank": spaces.Discrete(15),
                "length": spaces.Discrete(13),
                "kickers": spaces.Sequence(spaces.Discrete(15)),
                "cards": spaces.Sequence(spaces.Discrete(54)),
                "pass": spaces.Discrete(2),
            })),
        })

    def _obs_dict(self, obs_cpp) -> dict:
        # numerical arrays — reorder so index 0 = observing player
        obs_p = self._env.observe_player
        p_order = [obs_p, (obs_p + 1) % 3, (obs_p + 2) % 3]

        hands = [np.array(self._env.game.get_observation(p).hand, dtype=np.float32)
                 for p in p_order]
        cards = [sum(self._env.game.get_observation(p).hand) for p in p_order]

        d = {
            "hands": np.stack(hands),
            "landlord_cards": np.array(obs_cpp.landlord_cards, dtype=np.float32),
            "num_cards": np.array(cards, dtype=np.float32),
            "role": np.array(obs_cpp.role, dtype=np.float32),
            "last_player": int(self._env.game.last_player()),
        }

        # history
        d["history"] = [record_to_dict(r) for r in self._env.game.play_history()]

        # legal moves
        d["legal_moves"] = [move_to_dict(m) for m in self._env.legal_actions()]
        return d

    def reset(self, seed: Optional[int] = None,
              options: Optional[dict] = None) -> tuple:
        super().reset(seed=seed)
        if seed is not None:
            self._env.set_seed(seed)
        player_id = -1
        if options and "player_id" in options:
            player_id = options["player_id"]
        obs_cpp = self._env.reset(player_id)
        return self._obs_dict(obs_cpp), {"num_moves": 0}

    def step(self, action: int) -> tuple:
        """Execute action — index into legal_moves list."""
        legal = self._env.legal_actions()
        if 0 <= action < len(legal):
            result = self._env.step(action)
        else:
            result = self._env.step(0)
        obs = self._obs_dict(result.obs)
        info = {
            "num_moves": self._env.num_moves,
            "winner": self._env.game.winner() if result.terminated else -1,
        }
        return obs, result.reward, result.terminated, result.truncated, info

    def render(self):
        if self.render_mode in ("human", "ansi"):
            self._env.game.print_state()
            legal = self._env.legal_actions()
            if legal:
                print(f"\nLegal moves ({len(legal)}):")
                for i, m in enumerate(legal[:15]):
                    from doudizhu_cpp import move_to_string
                    print(f"  {i}: {move_to_string(m)}")
                if len(legal) > 15:
                    print(f"  ... and {len(legal) - 15} more")

    def close(self):
        pass
