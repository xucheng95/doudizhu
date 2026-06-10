"""Convert DoudizhuGymEnv obs dicts into tensors for DoudizhuAgent."""
from __future__ import annotations

import torch

from training.config import TrainingConfig

_MOVE_TYPE_IDX = {
    "NONE": 0, "SINGLE": 1, "PAIR": 2, "TRIPLE": 3,
    "TRIPLE_ONE": 4, "TRIPLE_TWO": 5, "STRAIGHT": 6,
    "DOUBLE_STRAIGHT": 7, "PLANE": 8, "PLANE_WING_SINGLE": 9,
    "PLANE_WING_PAIR": 10, "BOMB": 11, "ROCKET": 12,
    "FOUR_TWO_SINGLE": 13, "FOUR_TWO_PAIR": 14, "PASS": 15,
}

_RANK_STR_MAP = {
    "3": 0, "4": 1, "5": 2, "6": 3, "7": 4, "8": 5, "9": 6, "10": 7,
    "J": 8, "Q": 9, "K": 10, "A": 11, "2": 12, "SJ": 13, "BJ": 14,
}


def _card_str_to_rank(s: str) -> int:
    return _RANK_STR_MAP.get(str(s), -1)


def _encode_history_token(record: dict) -> list[float]:
    """Encode one play record into a 62-dim float vector.

    Layout: player(3) + type(16) + rank(15) + length(13) + cards(15) = 62
    """
    player_oh = [0.0] * 3
    p = int(record.get("player", 0))
    if 0 <= p < 3:
        player_oh[p] = 1.0

    t_idx = _MOVE_TYPE_IDX.get(record.get("type", "PASS"), 15)
    type_oh = [0.0] * 16
    type_oh[t_idx] = 1.0

    rank = max(0, min(int(record.get("rank", 0)), 14))
    rank_oh = [0.0] * 15
    rank_oh[rank] = 1.0

    length = max(0, min(int(record.get("length", 0)), 12))
    len_oh = [0.0] * 13
    len_oh[length] = 1.0

    cards_mh = [0.0] * 15
    for c in record.get("cards", []):
        r = _card_str_to_rank(c)
        if 0 <= r < 15:
            cards_mh[r] = 1.0

    return player_oh + type_oh + rank_oh + len_oh + cards_mh  # 62


def _encode_move(move: dict) -> dict:
    """Encode one move dict into structured integer/float fields."""
    t_idx = _MOVE_TYPE_IDX.get(move.get("type", "PASS"), 15)
    rank = max(0, min(int(move.get("rank", 0)), 14))
    length = max(0, min(int(move.get("length", 0)), 12))

    kickers_mh = [0.0] * 15
    for k in move.get("kickers", []):
        idx = int(k)
        if 0 <= idx < 15:
            kickers_mh[idx] = 1.0

    cards_mh = [0.0] * 15
    for c in move.get("cards", []):
        r = _card_str_to_rank(str(c))
        if 0 <= r < 15:
            cards_mh[r] = 1.0

    return {"type": t_idx, "rank": rank, "length": length,
            "kickers": kickers_mh, "cards": cards_mh}


def encode_obs(obs: dict, cfg: TrainingConfig, device: torch.device) -> dict:
    """Encode a single obs dict into a dict of unbatched tensors."""
    hand = torch.tensor(obs["hands"][0], dtype=torch.float32, device=device)
    all_hands = torch.tensor(obs["hands"], dtype=torch.float32, device=device)
    landlord_cards = torch.tensor(obs["landlord_cards"], dtype=torch.float32, device=device)
    num_cards = torch.tensor(obs["num_cards"], dtype=torch.float32, device=device)
    role = torch.tensor(obs["role"], dtype=torch.float32, device=device)

    raw_hist = obs.get("history", [])
    T = min(len(raw_hist), cfg.max_history)
    if T > 0:
        hist_vecs = [_encode_history_token(r) for r in raw_hist[-T:]]
        history = torch.tensor(hist_vecs, dtype=torch.float32, device=device)
        history_mask = torch.zeros(T, dtype=torch.bool, device=device)
    else:
        history = torch.zeros(0, 62, dtype=torch.float32, device=device)
        history_mask = torch.zeros(0, dtype=torch.bool, device=device)

    legal = obs.get("legal_moves", [])
    n_legal = min(len(legal), cfg.max_actions)
    move_type = torch.zeros(cfg.max_actions, dtype=torch.long, device=device)
    move_rank = torch.zeros(cfg.max_actions, dtype=torch.long, device=device)
    move_length = torch.zeros(cfg.max_actions, dtype=torch.long, device=device)
    move_kickers = torch.zeros(cfg.max_actions, 15, dtype=torch.float32, device=device)
    move_cards = torch.zeros(cfg.max_actions, 15, dtype=torch.float32, device=device)

    for i, mv in enumerate(legal[:n_legal]):
        enc = _encode_move(mv)
        move_type[i] = enc["type"]
        move_rank[i] = enc["rank"]
        move_length[i] = enc["length"]
        move_kickers[i] = torch.tensor(enc["kickers"], device=device)
        move_cards[i] = torch.tensor(enc["cards"], device=device)

    return {
        "hand": hand,
        "num_cards": num_cards,
        "role": role,
        "landlord_cards": landlord_cards,
        "history": history,
        "history_mask": history_mask,
        "all_hands": all_hands,
        "move_type": move_type,
        "move_rank": move_rank,
        "move_length": move_length,
        "move_kickers": move_kickers,
        "move_cards": move_cards,
        "num_legal": torch.tensor(n_legal, dtype=torch.long, device=device),
    }


def encode_obs_batch(obs_list: list[dict], cfg: TrainingConfig, device: torch.device) -> dict:
    """Stack a list of encoded obs into a batched dict of tensors."""
    encoded = [encode_obs(o, cfg, device) for o in obs_list]

    max_T = max(e["history"].size(0) for e in encoded)
    histories, masks = [], []
    for e in encoded:
        T = e["history"].size(0)
        pad_len = max_T - T
        if pad_len > 0:
            histories.append(torch.cat([e["history"], torch.zeros(pad_len, 62, device=device)]))
            masks.append(torch.cat([e["history_mask"], torch.ones(pad_len, dtype=torch.bool, device=device)]))
        else:
            histories.append(e["history"])
            masks.append(e["history_mask"])

    def stack(key: str) -> torch.Tensor:
        return torch.stack([e[key] for e in encoded], dim=0)

    return {
        "hand": stack("hand"),
        "num_cards": stack("num_cards"),
        "role": stack("role"),
        "landlord_cards": stack("landlord_cards"),
        "history": torch.stack(histories),
        "history_mask": torch.stack(masks),
        "all_hands": stack("all_hands"),
        "move_type": stack("move_type"),
        "move_rank": stack("move_rank"),
        "move_length": stack("move_length"),
        "move_kickers": stack("move_kickers"),
        "move_cards": stack("move_cards"),
        "num_legal": stack("num_legal"),
    }
