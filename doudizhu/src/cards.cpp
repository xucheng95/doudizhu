#include "cards.hpp"
#include <cstring>
#include <sstream>

namespace doudizhu {

const char* RANK_NAMES[NUM_RANKS] = {
    "3", "4", "5", "6", "7", "8", "9", "10",
    "J", "Q", "K", "A", "2", "SJ", "BJ"
};

const char* SUIT_NAMES[NUM_SUITS] = {
    "♠", "♥", "♣", "♦"
};

std::string card_name(int card_id) {
    if (card_id < 0 || card_id >= NUM_CARDS) return "?";
    int r = card_rank(card_id);
    if (r >= RANK_SJ) {
        return (r == RANK_SJ) ? "SJ" : "BJ";
    }
    int s = card_suit(card_id);
    return std::string(SUIT_NAMES[s]) + RANK_NAMES[r];
}

std::string mask_to_string(CardMask mask) {
    std::string result;
    std::vector<int> ids = mask_to_ids(mask);
    std::sort(ids.begin(), ids.end(), [](int a, int b) {
        return card_rank(a) < card_rank(b);
    });
    for (size_t i = 0; i < ids.size(); ++i) {
        if (i > 0) result += " ";
        result += card_name(ids[i]);
    }
    return result;
}

std::string move_to_string(const Move& move) {
    if (move.type == MoveType::NONE) return "None";
    if (move.type == MoveType::PASS) return "Pass";

    static const char* TYPE_NAMES[] = {
        "NONE", "Single", "Pair", "Triple", "Triple+1", "Triple+2",
        "Straight", "DoubleStraight", "Plane", "Plane+Singles", "Plane+Pairs",
        "Bomb", "Rocket", "Four+2Singles", "Four+2Pairs", "Pass"
    };

    std::string s = TYPE_NAMES[static_cast<int>(move.type)];
    s += "[" + std::string(RANK_NAMES[move.rank]);
    if (move.length > 0) {
        s += " x" + std::to_string(move.length);
    }
    s += "]: " + mask_to_string(move.mask);
    return s;
}

std::string rank_name(int rank) {
    if (rank < 0 || rank >= NUM_RANKS) return "?";
    return RANK_NAMES[rank];
}

} // namespace doudizhu
