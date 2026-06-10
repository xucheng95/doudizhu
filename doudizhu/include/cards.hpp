#ifndef DOUDIZHU_CARDS_HPP
#define DOUDIZHU_CARDS_HPP

#include <cstdint>
#include <string>
#include <vector>
#include <array>
#include <algorithm>

namespace doudizhu {

// ============================================================
// Card representation
// ============================================================

// 54 cards: 52 suit cards + 2 jokers
constexpr int NUM_CARDS = 54;
constexpr int NUM_RANKS = 15;   // 3,4,...,K,A,2,SJoker,BJoker
constexpr int NUM_SUITS = 4;

// Rank constants (internal rank, 0-14)
constexpr int RANK_3  = 0;
constexpr int RANK_4  = 1;
constexpr int RANK_5  = 2;
constexpr int RANK_6  = 3;
constexpr int RANK_7  = 4;
constexpr int RANK_8  = 5;
constexpr int RANK_9  = 6;
constexpr int RANK_10 = 7;
constexpr int RANK_J  = 8;
constexpr int RANK_Q  = 9;
constexpr int RANK_K  = 10;
constexpr int RANK_A  = 11;
constexpr int RANK_2  = 12;
constexpr int RANK_SJ = 13;  // Small Joker
constexpr int RANK_BJ = 14;  // Big Joker

// Number of cards per rank (4 for normal ranks, 1 for jokers)
constexpr int RANK_COUNT[NUM_RANKS] = {
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 1
};

// Rank display name
extern const char* RANK_NAMES[NUM_RANKS];
extern const char* SUIT_NAMES[NUM_SUITS];

// Card ID: 0-53
// For rank in [0,12]: id = rank * 4 + suit  (suit 0-3)
// For jokers: 52 = Small Joker, 53 = Big Joker
inline int card_id(int rank, int suit = 0) {
    if (rank >= RANK_SJ) return 52 + (rank - RANK_SJ);  // 52 or 53
    return rank * 4 + suit;
}

inline int card_rank(int id) {
    if (id >= 52) return RANK_SJ + (id - 52);  // 13 or 14
    return id / 4;
}

inline int card_suit(int id) {
    if (id >= 52) return 0;  // jokers have no suit
    return id % 4;
}

// Bitmask type: bit i = card id i
using CardMask = uint64_t;

constexpr CardMask EMPTY_MASK = 0;

// Precomputed mask for all cards of each rank
inline CardMask make_rank_mask(int rank) {
    if (rank >= RANK_SJ) return 1ULL << (52 + rank - RANK_SJ);
    CardMask m = 0;
    for (int s = 0; s < NUM_SUITS; ++s) m |= 1ULL << (rank * 4 + s);
    return m;
}

inline const std::array<CardMask, NUM_RANKS>& rank_masks() {
    static const std::array<CardMask, NUM_RANKS> table = [] {
        std::array<CardMask, NUM_RANKS> t{};
        for (int r = 0; r < NUM_RANKS; ++r) t[r] = make_rank_mask(r);
        return t;
    }();
    return table;
}

inline CardMask rank_mask(int rank) { return rank_masks()[rank]; }

inline bool mask_has(CardMask mask, int card_id) {
    return (mask >> card_id) & 1;
}

inline CardMask mask_add(CardMask mask, int card_id) {
    return mask | (1ULL << card_id);
}

inline CardMask mask_remove(CardMask mask, int card_id) {
    return mask & ~(1ULL << card_id);
}

inline CardMask mask_from_ids(const std::vector<int>& ids) {
    CardMask m = 0;
    for (int id : ids) m |= (1ULL << id);
    return m;
}

// Pops the lowest set bit and returns its index
inline int mask_pop_lsb(CardMask& mask) {
    int idx = __builtin_ctzll(mask);
    mask &= (mask - 1);
    return idx;
}

inline int mask_count(CardMask mask) {
    return __builtin_popcountll(mask);
}

// Get all card ids from a mask
inline std::vector<int> mask_to_ids(CardMask mask) {
    std::vector<int> ids;
    ids.reserve(mask_count(mask));
    while (mask) {
        ids.push_back(mask_pop_lsb(mask));
    }
    return ids;
}

// Get the rank-hashed count vector for a mask
// returns [count of rank_0, count of rank_1, ..., count of rank_14]
inline std::array<int, NUM_RANKS> mask_rank_counts(CardMask mask) {
    std::array<int, NUM_RANKS> counts = {};
    while (mask) {
        int id = mask_pop_lsb(mask);
        counts[card_rank(id)]++;
    }
    return counts;
}

// ============================================================
// Deck operations
// ============================================================

// Create a full deck (54 cards)
inline CardMask full_deck() {
    return (1ULL << NUM_CARDS) - 1;
}

// Shuffle the deck (Fisher-Yates, in-place on card IDs)
inline void shuffle_deck(std::vector<int>& deck) {
    static bool seeded = false;
    if (!seeded) {
        std::srand(std::time(nullptr));
        seeded = true;
    }
    for (int i = deck.size() - 1; i > 0; --i) {
        int j = std::rand() % (i + 1);
        std::swap(deck[i], deck[j]);
    }
}

inline std::vector<int> create_and_shuffle_deck() {
    std::vector<int> deck(NUM_CARDS);
    for (int i = 0; i < NUM_CARDS; ++i) deck[i] = i;
    shuffle_deck(deck);
    return deck;
}

// ============================================================
// Move types
// ============================================================

enum class MoveType : uint8_t {
    NONE = 0,
    SINGLE,          // 单张
    PAIR,            // 对子
    TRIPLE,          // 三张
    TRIPLE_ONE,      // 三带一
    TRIPLE_TWO,      // 三带二
    STRAIGHT,        // 顺子（5+连续单张）
    DOUBLE_STRAIGHT, // 连对（3+连续对子）
    PLANE,           // 飞机（2+连续三张）
    PLANE_WING_SINGLE, // 飞机带单
    PLANE_WING_PAIR,   // 飞机带对
    BOMB,            // 炸弹（4张同点）
    ROCKET,          // 火箭（王炸）
    FOUR_TWO_SINGLE, // 四带二单
    FOUR_TWO_PAIR,   // 四带二对
    PASS             // 过牌
};

constexpr int NUM_MOVE_TYPES = 15;

// Move: a set of cards + type
struct Move {
    MoveType type = MoveType::NONE;
    CardMask mask = 0;
    int rank = 0;       // Primary rank (e.g., for single it's the card rank)
    int length = 0;     // Extra param (e.g., straight length, wing count)
    int count = 0;      // Number of cards in this move
    int action_idx = -1; // Flat action-space index (0..14635)

    bool is_valid() const { return type != MoveType::NONE; }
};

// Play record: who played what
struct PlayRecord {
    int player = -1;
    Move move;
};

// Display helpers
std::string card_name(int card_id);
std::string mask_to_string(CardMask mask);
std::string move_to_string(const Move& move);
std::string rank_name(int rank);

} // namespace doudizhu

#endif // DOUDIZHU_CARDS_HPP
