#include "game.hpp"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <mutex>
#include <random>
#include <unordered_map>

namespace doudizhu {

// ============================================================
// Action table — static, built once per process
// ============================================================
static std::unordered_map<uint64_t, int> action_lookup_;
static std::once_flag action_table_flag_;

static uint64_t pack_key(int type, int rank, int length,
                          int k0, int k1, int k2, int k3, int k4, int k5) {
    return ((uint64_t)type   << 36) |
           ((uint64_t)rank   << 32) |
           ((uint64_t)length << 28) |
           ((uint64_t)k0     << 24) |
           ((uint64_t)k1     << 20) |
           ((uint64_t)k2     << 16) |
           ((uint64_t)k3     << 12) |
           ((uint64_t)k4     << 8)  |
           ((uint64_t)k5);
}

void Game::init_action_table() {
    std::call_once(action_table_flag_, []() {
    action_lookup_.reserve(14636);
    int idx = 0;

    // 1. SINGLE (0-14)
    for (int r = 0; r < NUM_RANKS; ++r)
        action_lookup_[pack_key((int)MoveType::SINGLE, r, 0, 0,0,0,0,0,0)] = idx++;

    // 2. PAIR (15-27)
    for (int r = 0; r < 13; ++r)
        action_lookup_[pack_key((int)MoveType::PAIR, r, 0, 0,0,0,0,0,0)] = idx++;

    // 3. TRIPLE (28-40)
    for (int r = 0; r < 13; ++r)
        action_lookup_[pack_key((int)MoveType::TRIPLE, r, 0, 0,0,0,0,0,0)] = idx++;

    // 4. TRIPLE_ONE (41-222)
    for (int r = 0; r < 13; ++r)
        for (int k = 0; k < NUM_RANKS; ++k)
            if (k != r)
                action_lookup_[pack_key((int)MoveType::TRIPLE_ONE, r, 0, k,0,0,0,0,0)] = idx++;

    // 5. TRIPLE_TWO (223-378)
    for (int r = 0; r < 13; ++r)
        for (int k = 0; k < 13; ++k)
            if (k != r)
                action_lookup_[pack_key((int)MoveType::TRIPLE_TWO, r, 0, k,0,0,0,0,0)] = idx++;

    // 6. STRAIGHT (379-414)
    for (int len = 5; len <= 12; ++len)
        for (int start = 0; start <= 12 - len; ++start)
            action_lookup_[pack_key((int)MoveType::STRAIGHT, start, len, 0,0,0,0,0,0)] = idx++;

    // 7. DOUBLE_STRAIGHT (415-466)
    for (int len = 3; len <= 10; ++len)
        for (int start = 0; start <= 12 - len; ++start)
            action_lookup_[pack_key((int)MoveType::DOUBLE_STRAIGHT, start, len, 0,0,0,0,0,0)] = idx++;

    // 8. PLANE (467-511)
    for (int len = 2; len <= 6; ++len)
        for (int start = 0; start <= 12 - len; ++start)
            action_lookup_[pack_key((int)MoveType::PLANE, start, len, 0,0,0,0,0,0)] = idx++;

    // 9. PLANE_WING_SINGLE (512-9143): generate C(15-len, len) combos
    for (int len = 2; len <= 6; ++len) {
        for (int start = 0; start <= 12 - len; ++start) {
            std::vector<int> avail;
            for (int r = 0; r < NUM_RANKS; ++r)
                if (r < start || r >= start + len) avail.push_back(r);
            int n = (int)avail.size();
            if (n < len) continue;
            std::vector<int> comb(len);
            for (int i = 0; i < len; ++i) comb[i] = i;
            while (true) {
                int ks[6] = {0};
                for (int i = 0; i < len; ++i) ks[i] = avail[comb[i]];
                action_lookup_[pack_key((int)MoveType::PLANE_WING_SINGLE, start, len,
                                         ks[0], ks[1], ks[2], ks[3], ks[4], ks[5])] = idx++;
                int i = len - 1;
                while (i >= 0 && comb[i] == n - len + i) --i;
                if (i < 0) break;
                comb[i]++;
                for (int j = i + 1; j < len; ++j) comb[j] = comb[j-1] + 1;
            }
        }
    }

    // 10. PLANE_WING_PAIR (9144-12579)
    for (int len = 2; len <= 6; ++len) {
        for (int start = 0; start <= 12 - len; ++start) {
            std::vector<int> avail;
            for (int r = 0; r < 13; ++r)
                if (r < start || r >= start + len) avail.push_back(r);
            int n = (int)avail.size();
            if (n < len) continue;
            std::vector<int> comb(len);
            for (int i = 0; i < len; ++i) comb[i] = i;
            while (true) {
                int ks[6] = {0};
                for (int i = 0; i < len; ++i) ks[i] = avail[comb[i]];
                action_lookup_[pack_key((int)MoveType::PLANE_WING_PAIR, start, len,
                                         ks[0], ks[1], ks[2], ks[3], ks[4], ks[5])] = idx++;
                int i = len - 1;
                while (i >= 0 && comb[i] == n - len + i) --i;
                if (i < 0) break;
                comb[i]++;
                for (int j = i + 1; j < len; ++j) comb[j] = comb[j-1] + 1;
            }
        }
    }

    // 11. BOMB (12580-12592)
    for (int r = 0; r < 13; ++r)
        action_lookup_[pack_key((int)MoveType::BOMB, r, 0, 0,0,0,0,0,0)] = idx++;

    // 12. ROCKET (12593)
    action_lookup_[pack_key((int)MoveType::ROCKET, 14, 0, 0,0,0,0,0,0)] = idx++;

    // 13. FOUR_TWO_SINGLE (12594-13776)
    for (int r = 0; r < 13; ++r)
        for (int k1 = 0; k1 < NUM_RANKS; ++k1)
            if (k1 != r)
                for (int k2 = k1 + 1; k2 < NUM_RANKS; ++k2)
                    if (k2 != r)
                        action_lookup_[pack_key((int)MoveType::FOUR_TWO_SINGLE, r, 0, k1,k2,0,0,0,0)] = idx++;

    // 14. FOUR_TWO_PAIR (13777-14634)
    for (int r = 0; r < 13; ++r)
        for (int k1 = 0; k1 < 13; ++k1)
            if (k1 != r)
                for (int k2 = k1 + 1; k2 < 13; ++k2)
                    if (k2 != r)
                        action_lookup_[pack_key((int)MoveType::FOUR_TWO_PAIR, r, 0, k1,k2,0,0,0,0)] = idx++;

    // 15. PASS (14635)
    action_lookup_[pack_key((int)MoveType::PASS, 0, 0, 0,0,0,0,0,0)] = idx++;

    }); // end call_once
}

int Game::encode_action_idx(MoveType type, int rank, int length,
                             int k0, int k1, int k2, int k3, int k4, int k5) {
    init_action_table();
    uint64_t key = pack_key((int)type, rank, length, k0, k1, k2, k3, k4, k5);
    auto it = action_lookup_.find(key);
    if (it != action_lookup_.end()) return it->second;
    return -1;
}

// ============================================================
// Constructor
// ============================================================
Game::Game()
    : phase_(Phase::WAITING)
    , current_player_(0)
    , last_player_(-1)
    , pass_count_(0)
    , winner_(-1)
    , landlord_idx_(-1)
    , legal_actions_player_(-1)
{
    hands_.fill(0);
    roles_ = {Role::PEASANT_0, Role::PEASANT_1, Role::PEASANT_0};
    landlord_bonus_ = 0;
}

// ============================================================
// Reset / deal
// ============================================================
void Game::reset() {
    phase_ = Phase::PLAYING;
    landlord_idx_ = 1; // default: player 1 is landlord
    roles_ = {Role::PEASANT_0, Role::LANDLORD, Role::PEASANT_1};
    current_player_ = landlord_idx_;  // landlord goes first
    last_player_ = -1;
    pass_count_ = 0;
    last_move_ = Move{};
    winner_ = -1;
    legal_actions_player_ = -1;
    legal_actions_.clear();
    play_history_.clear();

    deal_cards();

    // Give landlord bonus
    hands_[landlord_idx_] |= landlord_bonus_;
}

void Game::reset_with_auction() {
    phase_ = Phase::AUCTION;
    current_player_ = 0;
    last_player_ = -1;
    pass_count_ = 0;
    last_move_ = Move{};
    winner_ = -1;
    landlord_idx_ = -1;
    roles_ = {Role::PEASANT_0, Role::PEASANT_0, Role::PEASANT_0};
    legal_actions_player_ = -1;
    legal_actions_.clear();
    play_history_.clear();

    deal_cards();
}

void Game::deal_cards() {
    // Reuse a stack-local array to avoid heap allocation every episode
    int deck[NUM_CARDS];
    for (int i = 0; i < NUM_CARDS; ++i) deck[i] = i;

    // Fisher-Yates using the game's own rng (initialised in Game::Game via reset)
    static std::mt19937 rng(std::random_device{}());
    for (int i = NUM_CARDS - 1; i > 0; --i) {
        std::uniform_int_distribution<int> d(0, i);
        std::swap(deck[i], deck[d(rng)]);
    }

    hands_.fill(0);
    landlord_bonus_ = 0;

    for (int i = 0; i < 51; ++i)
        hands_[i / 17] |= (1ULL << deck[i]);
    for (int i = 51; i < 54; ++i)
        landlord_bonus_ |= (1ULL << deck[i]);
}

// ============================================================
// Auction simulation (simple: last player becomes landlord)
// ============================================================
void Game::simulate_auction() {
    // Simple: current_player_ (starting bidder) bids, loops
    // For simplicity: random landlord
    static std::mt19937 rng2(std::random_device{}());
    std::uniform_int_distribution<int> dist(0, 2);
    landlord_idx_ = dist(rng2);

    roles_[landlord_idx_] = Role::LANDLORD;
    // other roles already PEASANT_0/PEASANT_1, but ensure correct labeling
    int peasant_idx = 0;
    for (int i = 0; i < NUM_PLAYERS; ++i) {
        if (i != landlord_idx_) {
            roles_[i] = (peasant_idx++ == 0) ? Role::PEASANT_0 : Role::PEASANT_1;
        }
    }

    hands_[landlord_idx_] |= landlord_bonus_;

    phase_ = Phase::PLAYING;
    current_player_ = landlord_idx_;
    last_player_ = -1;
    pass_count_ = 0;
    last_move_ = Move{};
    winner_ = -1;
    legal_actions_player_ = -1;
    legal_actions_.clear();
}

// ============================================================
// Step
// ============================================================
bool Game::step(int action_idx) {
    if (phase_ == Phase::AUCTION) {
        // Simple auction step: just pass/bid
        // For simplicity we auto-resolve auction and proceed
        simulate_auction();
        return true;
    }

    if (is_terminal()) return false;
    const auto& actions = get_current_player_moves();
    if (action_idx < 0 || (size_t)action_idx >= actions.size()) return false;

    const Move move = actions[action_idx];  // value copy: step_move clears legal_actions_
    return step_move(move);
}

bool Game::step_move(const Move& move) {
    if (is_terminal()) return false;
    if (!move.is_valid()) return false;

    // Apply move
    play_history_.push_back({current_player_, move});
    if (move.type != MoveType::PASS) {
        hands_[current_player_] &= ~move.mask;
        last_move_ = move;
        last_player_ = current_player_;
        pass_count_ = 0;
    } else {
        pass_count_++;
    }

    // Check for win
    if (hands_[current_player_] == 0 && move.type != MoveType::PASS) {
        phase_ = Phase::FINISHED;
        winner_ = current_player_;
        return true;
    }

    // Advance to next player
    current_player_ = (current_player_ + 1) % NUM_PLAYERS;

    // If two consecutive passes, the last non-pass player leads again
    if (pass_count_ >= 2 && last_move_.type != MoveType::NONE) {
        last_move_ = Move{};
        pass_count_ = 0;
    }

    // Clear legal actions cache for new player
    legal_actions_player_ = -1;
    legal_actions_.clear();

    return true;
}

// ============================================================
// Reward
// ============================================================
float Game::reward(int player) const {
    if (!is_terminal()) return 0.0f;
    if (winner_ < 0) return 0.0f;

    Role winner_role = roles_[winner_];
    Role player_role = roles_[player];

    bool same_team = (winner_role == Role::LANDLORD) == (player_role == Role::LANDLORD);
    if (same_team) {
        // Winner's team
        if (winner_role == Role::LANDLORD) return 2.0f;
        return 1.0f;
    } else {
        // Loser's team
        if (winner_role == Role::LANDLORD) return -1.0f;
        return -2.0f;
    }
}

// ============================================================
// Observation
// ============================================================
Observation Game::get_observation(int player) const {
    Observation obs{};
    obs.fill_hand(hands_[player]);
    obs.fill_landlord(landlord_bonus_);

    for (int i = 0; i < NUM_PLAYERS; ++i)
        obs.num_cards[i] = static_cast<float>(mask_count(hands_[i]));

    obs.role.fill(0.0f);
    if (roles_[player] == Role::PEASANT_0) obs.role[0] = 1.0f;
    else if (roles_[player] == Role::PEASANT_1) obs.role[1] = 1.0f;
    else obs.role[2] = 1.0f;

    // Last move info
    obs.last_move_rank.fill(0);
    if (last_move_.type != MoveType::NONE && last_move_.type != MoveType::PASS) {
        obs.last_move_rank[last_move_.rank] = 1.0f;
    }
    if (last_player_ >= 0) {
        obs.last_player[last_player_] = 1.0f;
    }

    return obs;
}

// ============================================================
// Legal action generation
// ============================================================
const std::vector<Move>& Game::get_current_player_moves() const {
    // Lazily compute
    if (legal_actions_player_ != current_player_) {
        const_cast<Game*>(this)->compute_legal_actions();
        legal_actions_player_ = current_player_;
    }
    return legal_actions_;
}

void Game::compute_legal_actions() {
    legal_actions_.clear();
    legal_actions_.reserve(512);  // full kicker enumeration can exceed 200+ moves
    CardMask hand = hands_[current_player_];
    auto counts = mask_rank_counts(hand);

    if (last_move_.type == MoveType::NONE || pass_count_ >= 2) {
        // Lead: any valid combo
        generate_lead_moves(hand, counts, legal_actions_);
    } else {
        // Follow: must beat last move
        generate_follow_moves(hand, counts, last_move_, legal_actions_);
        // Always include pass
        legal_actions_.push_back(make_move(MoveType::PASS, 0, 0, 0));
    }
}

// ============================================================
// Move construction helpers
// ============================================================
Move Game::make_move(MoveType type, CardMask mask, int rank, int length) {
    Move m;
    m.type = type;
    m.mask = mask;
    m.rank = rank;
    m.length = length;
    m.count = mask_count(mask);
    m.action_idx = encode_action_idx(type, rank, length, 0,0,0,0,0,0);
    return m;
}

Move Game::make_move_idx(MoveType type, CardMask mask, int rank, int length,
                          int k0, int k1, int k2, int k3, int k4, int k5) {
    Move m;
    m.type = type;
    m.mask = mask;
    m.rank = rank;
    m.length = length;
    m.count = mask_count(mask);
    m.action_idx = encode_action_idx(type, rank, length, k0, k1, k2, k3, k4, k5);
    return m;
}

CardMask Game::cards_of_rank(int rank) {
    return rank_mask(rank);
}

CardMask Game::extract_n_from_rank(CardMask hand, int rank, int n) {
    CardMask rc = cards_of_rank(rank) & hand;
    CardMask result = 0;
    int found = 0;
    while (rc && found < n) {
        int id = __builtin_ctzll(rc);
        result |= (1ULL << id);
        rc &= (rc - 1);
        found++;
    }
    return result;
}

// ============================================================
// Recognize move from a card mask
// ============================================================
Move Game::recognize_move(CardMask mask) {
    if (mask == 0) return Move{};

    Move m;
    m.mask = mask;
    m.count = mask_count(mask);

    auto counts = mask_rank_counts(mask);
    int non_zero = 0;
    int last_nz = -1;
    int nz_run = 0;
    int nz_run_max = 0;
    int quad_count = 0, triple_count = 0, pair_count = 0, single_count = 0;
    int quad_rank = -1, triple_rank = -1, pair_rank = -1, single_rank = -1;

    for (int r = 0; r < NUM_RANKS; ++r) {
        if (counts[r] > 0) {
            non_zero++;
            if (counts[r] == 4) { quad_count++; quad_rank = r; }
            if (counts[r] == 3) { triple_count++; if (triple_rank < 0) triple_rank = r; }
            if (counts[r] == 2) { pair_count++; if (pair_rank < 0) pair_rank = r; }
            if (counts[r] == 1) { single_count++; if (single_rank < 0) single_rank = r; }

            if (last_nz >= 0 && r == last_nz + 1) {
                nz_run++;
            } else {
                nz_run = 1;
            }
            nz_run_max = std::max(nz_run_max, nz_run);
            last_nz = r;
        } else {
            nz_run = 0;
        }
    }

    // Rocket: 2 jokers
    if (mask == ((1ULL << 52) | (1ULL << 53))) {
        return make_move(MoveType::ROCKET, mask, RANK_BJ, 0);
    }

    // Single
    if (non_zero == 1 && m.count == 1) {
        return make_move(MoveType::SINGLE, mask, single_rank, 0);
    }

    // Pair
    if (non_zero == 1 && m.count == 2 && counts[pair_rank] == 2) {
        return make_move(MoveType::PAIR, mask, pair_rank, 0);
    }

    // Triple
    if (non_zero == 1 && m.count == 3 && counts[triple_rank] == 3) {
        return make_move(MoveType::TRIPLE, mask, triple_rank, 0);
    }

    // Bomb
    if (non_zero == 1 && m.count == 4 && counts[quad_rank] == 4) {
        return make_move(MoveType::BOMB, mask, quad_rank, 0);
    }

    // Triple + 1: exactly one triple, one single (4 cards)
    if (non_zero == 2 && m.count == 4 && triple_count == 1 && single_count == 1) {
        return make_move(MoveType::TRIPLE_ONE, mask, triple_rank, 0);
    }

    // Triple + 2: exactly one triple, one pair (5 cards)
    if (non_zero == 2 && m.count == 5 && triple_count == 1 && pair_count == 1) {
        return make_move(MoveType::TRIPLE_TWO, mask, triple_rank, 0);
    }

    // Straight: 5+ single cards, consecutive
    if (nz_run_max >= 5 && non_zero == nz_run_max && triple_count == 0 && pair_count == 0 && quad_count == 0) {
        // Check range: only ranks 0-11 can be in straight (3-A), not 2, SJ, BJ
        int first = -1, last = -1;
        for (int r = 0; r < NUM_RANKS; ++r) {
            if (counts[r] > 0) { if (first < 0) first = r; last = r; }
        }
        if (first >= RANK_3 && last <= RANK_A && last - first + 1 == nz_run_max) {
            return make_move(MoveType::STRAIGHT, mask, first, nz_run_max);
        }
    }

    // Double straight: 3+ consecutive pairs
    int pair_run = 0, pair_run_max = 0;
    int pair_first = -1, pair_last = -1;
    for (int r = 0; r < NUM_RANKS; ++r) {
        if (counts[r] == 2) {
            if (pair_first < 0) pair_first = r;
            pair_run++;
            pair_run_max = std::max(pair_run_max, pair_run);
            pair_last = r;
        } else {
            pair_run = 0;
        }
    }
    if (pair_run_max >= 3 && non_zero == pair_run_max && m.count == pair_run_max * 2) {
        if (pair_first >= RANK_3 && pair_last <= RANK_A) {
            return make_move(MoveType::DOUBLE_STRAIGHT, mask, pair_first, pair_run_max);
        }
    }

    // Plane: 2+ consecutive triples
    int triple_run = 0, triple_run_max = 0;
    int triple_first = -1, triple_last = -1;
    for (int r = 0; r < NUM_RANKS; ++r) {
        if (counts[r] == 3) {
            if (triple_first < 0) triple_first = r;
            triple_run++;
            triple_run_max = std::max(triple_run_max, triple_run);
            triple_last = r;
        } else {
            triple_run = 0;
        }
    }

    if (triple_run_max >= 2) {
        int plane_cards = triple_run_max * 3;
        int extra_cards = m.count - plane_cards;

        // Pure plane
        if (extra_cards == 0 && non_zero == triple_run_max) {
            return make_move(MoveType::PLANE, mask, triple_first, triple_run_max);
        }
        // Plane + singles
        if (extra_cards == triple_run_max && non_zero == triple_run_max + extra_cards) {
            return make_move(MoveType::PLANE_WING_SINGLE, mask, triple_first, triple_run_max);
        }
        // Plane + pairs
        if (extra_cards == triple_run_max * 2 && pair_count == triple_run_max && non_zero == triple_run_max + pair_count) {
            return make_move(MoveType::PLANE_WING_PAIR, mask, triple_first, triple_run_max);
        }
    }

    // Four + 2 singles (6 cards): one quad + exactly 2 more individual cards
    if (quad_count == 1 && m.count == 6) {
        int extra = 0;
        for (int r = 0; r < NUM_RANKS; ++r)
            if (r != quad_rank) extra += counts[r];
        if (extra == 2) {
            return make_move(MoveType::FOUR_TWO_SINGLE, mask, quad_rank, 0);
        }
    }

    // Four + 2 pairs (8 cards): one quad + exactly 2 pairs
    if (quad_count == 1 && m.count == 8) {
        int pair_check = 0;
        bool valid = true;
        for (int r = 0; r < NUM_RANKS; ++r) {
            if (r == quad_rank) continue;
            if (counts[r] == 0) continue;
            if (counts[r] == 2) pair_check++;
            else { valid = false; break; }
        }
        if (valid && pair_check == 2) {
            return make_move(MoveType::FOUR_TWO_PAIR, mask, quad_rank, 0);
        }
    }

    return Move{};  // Invalid
}

// ============================================================
// Can beat: does m1 beat m2?
// ============================================================
bool Game::can_beat(const Move& m1, const Move& m2) {
    // Rocket beats everything
    if (m1.type == MoveType::ROCKET) return true;

    // Bomb beats non-bomb, non-rocket
    if (m1.type == MoveType::BOMB && m2.type != MoveType::BOMB && m2.type != MoveType::ROCKET) return true;

    // Same type: compare by rank (and length for straights)
    if (m1.type == m2.type && m1.length == m2.length) {
        return m1.rank > m2.rank;
    }

    return false;
}

// ============================================================
// Generate lead moves
// ============================================================
void Game::generate_lead_moves(CardMask hand, const std::array<int, NUM_RANKS>& counts, std::vector<Move>& out) {
    // 1. Singles
    for (int r = 0; r < NUM_RANKS; ++r) {
        if (counts[r] >= 1) {
            CardMask mask = extract_n_from_rank(hand, r, 1);
            out.push_back(make_move(MoveType::SINGLE, mask, r, 0));
        }
    }

    // 2. Pairs (not BJ which has only 1)
    for (int r = 0; r <= RANK_SJ; ++r) {
        if (counts[r] >= 2) {
            CardMask mask = extract_n_from_rank(hand, r, 2);
            out.push_back(make_move(MoveType::PAIR, mask, r, 0));
        }
    }

    // 3. Triples
    for (int r = 0; r <= RANK_2; ++r) {
        if (counts[r] >= 3) {
            CardMask mask = extract_n_from_rank(hand, r, 3);
            out.push_back(make_move(MoveType::TRIPLE, mask, r, 0));
        }
    }

    // 4. Triple + 1 — enumerate all kicker ranks
    for (int r = 0; r <= RANK_2; ++r) {
        if (counts[r] >= 3) {
            CardMask triple = extract_n_from_rank(hand, r, 3);
            for (int k = 0; k < NUM_RANKS; ++k) {
                if (k != r && counts[k] >= 1) {
                    CardMask kicker = extract_n_from_rank(hand, k, 1);
                    out.push_back(make_move_idx(MoveType::TRIPLE_ONE, triple | kicker, r, 0, k));
                }
            }
        }
    }

    // 5. Triple + 2 — enumerate all pair kicker ranks
    for (int r = 0; r <= RANK_2; ++r) {
        if (counts[r] >= 3) {
            CardMask triple = extract_n_from_rank(hand, r, 3);
            for (int k = 0; k <= RANK_SJ; ++k) {
                if (k != r && counts[k] >= 2) {
                    CardMask pair = extract_n_from_rank(hand, k, 2);
                    out.push_back(make_move_idx(MoveType::TRIPLE_TWO, triple | pair, r, 0, k));
                }
            }
        }
    }

    // 6. Straights (5-12 consecutive singles, ranks 0-11 only)
    for (int len = 5; len <= 12; ++len) {
        for (int start = RANK_3; start + len - 1 <= RANK_A; ++start) {
            bool ok = true;
            for (int r = start; r < start + len; ++r) {
                if (counts[r] < 1) { ok = false; break; }
            }
            if (ok) {
                CardMask mask = 0;
                for (int r = start; r < start + len; ++r) {
                    mask |= extract_n_from_rank(hand, r, 1);
                }
                out.push_back(make_move(MoveType::STRAIGHT, mask, start, len));
            }
        }
    }

    // 7. Double straights (3-10 consecutive pairs)
    for (int len = 3; len <= 10; ++len) {
        for (int start = RANK_3; start + len - 1 <= RANK_A; ++start) {
            bool ok = true;
            for (int r = start; r < start + len; ++r) {
                if (counts[r] < 2) { ok = false; break; }
            }
            if (ok) {
                CardMask mask = 0;
                for (int r = start; r < start + len; ++r) {
                    mask |= extract_n_from_rank(hand, r, 2);
                }
                out.push_back(make_move(MoveType::DOUBLE_STRAIGHT, mask, start, len));
            }
        }
    }

    // 8. Planes (2-6 consecutive triples)
    for (int len = 2; len <= 6; ++len) {
        for (int start = RANK_3; start + len - 1 <= RANK_A; ++start) {
            bool ok = true;
            for (int r = start; r < start + len; ++r) {
                if (counts[r] < 3) { ok = false; break; }
            }
            if (ok) {
                CardMask plane_mask = 0;
                for (int r = start; r < start + len; ++r) {
                    plane_mask |= extract_n_from_rank(hand, r, 3);
                }
                // Pure plane
                out.push_back(make_move(MoveType::PLANE, plane_mask, start, len));

                // Plane + singles — enumerate by RANK combos (match Python table)
                {
                    std::vector<int> avail;
                    for (int r = 0; r < NUM_RANKS; ++r)
                        if ((r < start || r >= start + len) && counts[r] >= 1)
                            avail.push_back(r);
                    int n = (int)avail.size();
                    if (n >= len) {
                        std::vector<int> comb(len);
                        for (int i = 0; i < len; ++i) comb[i] = i;
                        while (true) {
                            CardMask wings = 0;
                            int ks[6] = {0};
                            for (int i = 0; i < len; ++i) {
                                int wr = avail[comb[i]];
                                ks[i] = wr;
                                wings |= extract_n_from_rank(hand, wr, 1);
                            }
                            out.push_back(make_move_idx(MoveType::PLANE_WING_SINGLE,
                                          plane_mask | wings, start, len,
                                          ks[0], ks[1], ks[2], ks[3], ks[4], ks[5]));
                            int i = len - 1;
                            while (i >= 0 && comb[i] == n - len + i) --i;
                            if (i < 0) break;
                            comb[i]++;
                            for (int j = i + 1; j < len; ++j) comb[j] = comb[j-1] + 1;
                        }
                    }
                }

                // Plane + pairs — enumerate by RANK combos (match Python table)
                {
                    std::vector<int> avail;
                    for (int r = 0; r <= RANK_SJ; ++r) {
                        if ((r < start || r >= start + len) && counts[r] >= 2)
                            avail.push_back(r);
                    }
                    int n = (int)avail.size();
                    if (n >= len) {
                        std::vector<int> comb(len);
                        for (int i = 0; i < len; ++i) comb[i] = i;
                        while (true) {
                            CardMask pw = 0;
                            int ks[6] = {0};
                            for (int i = 0; i < len; ++i) {
                                int pr = avail[comb[i]];
                                ks[i] = pr;
                                pw |= extract_n_from_rank(hand, pr, 2);
                            }
                            if ((pw & plane_mask) == 0) {
                                out.push_back(make_move_idx(MoveType::PLANE_WING_PAIR,
                                              plane_mask | pw, start, len,
                                              ks[0], ks[1], ks[2], ks[3], ks[4], ks[5]));
                            }
                            int i = len - 1;
                            while (i >= 0 && comb[i] == n - len + i) --i;
                            if (i < 0) break;
                            comb[i]++;
                            for (int j = i + 1; j < len; ++j) comb[j] = comb[j-1] + 1;
                        }
                    }
                }
            }
        }
    }

    // 9. Bombs
    for (int r = 0; r <= RANK_2; ++r) {
        if (counts[r] >= 4) {
            CardMask mask = extract_n_from_rank(hand, r, 4);
            out.push_back(make_move(MoveType::BOMB, mask, r, 0));
        }
    }

    // 10. Rocket (both jokers)
    if (counts[RANK_SJ] >= 1 && counts[RANK_BJ] >= 1) {
        CardMask mask = extract_n_from_rank(hand, RANK_SJ, 1) | extract_n_from_rank(hand, RANK_BJ, 1);
        out.push_back(make_move(MoveType::ROCKET, mask, RANK_BJ, 0));
    }

    // 11. Four + 2 singles — enumerate all kicker rank pairs
    for (int r = 0; r <= RANK_2; ++r) {
        if (counts[r] >= 4) {
            CardMask bomb = extract_n_from_rank(hand, r, 4);
            for (int k1 = 0; k1 < NUM_RANKS; ++k1) {
                if (k1 == r || counts[k1] < 1) continue;
                for (int k2 = k1 + 1; k2 < NUM_RANKS; ++k2) {
                    if (k2 == r || counts[k2] < 1) continue;
                    CardMask kickers = extract_n_from_rank(hand, k1, 1) | extract_n_from_rank(hand, k2, 1);
                    out.push_back(make_move_idx(MoveType::FOUR_TWO_SINGLE, bomb | kickers, r, 0, k1, k2));
                }
            }
        }
    }

    // 12. Four + 2 pairs — enumerate all pair-of-pair-ranks combos
    for (int r = 0; r <= RANK_2; ++r) {
        if (counts[r] >= 4) {
            CardMask bomb = extract_n_from_rank(hand, r, 4);
            for (int k1 = 0; k1 <= RANK_SJ; ++k1) {
                if (k1 == r || counts[k1] < 2) continue;
                for (int k2 = k1 + 1; k2 <= RANK_SJ; ++k2) {
                    if (k2 == r || counts[k2] < 2) continue;
                    CardMask pairs = extract_n_from_rank(hand, k1, 2) | extract_n_from_rank(hand, k2, 2);
                    out.push_back(make_move_idx(MoveType::FOUR_TWO_PAIR, bomb | pairs, r, 0, k1, k2));
                }
            }
        }
    }
}

// ============================================================
// Generate follow moves (beat last_move_)
// ============================================================
void Game::generate_follow_moves(CardMask hand, const std::array<int, NUM_RANKS>& counts,
                                  const Move& ref, std::vector<Move>& out) {
    auto try_beat = [&](MoveType type, int rank, int length, CardMask mask) {
        Move candidate = make_move(type, mask, rank, length);
        if (can_beat(candidate, ref)) {
            out.push_back(candidate);
        }
    };
    auto try_beat_idx = [&](MoveType type, int rank, int length, CardMask mask,
                             int k0=0, int k1=0, int k2=0, int k3=0, int k4=0, int k5=0) {
        Move candidate = make_move_idx(type, mask, rank, length, k0, k1, k2, k3, k4, k5);
        if (can_beat(candidate, ref)) {
            out.push_back(candidate);
        }
    };

    switch (ref.type) {
    case MoveType::SINGLE:
        for (int r = ref.rank + 1; r < NUM_RANKS; ++r) {
            if (counts[r] >= 1) {
                CardMask m = extract_n_from_rank(hand, r, 1);
                try_beat(MoveType::SINGLE, r, 0, m);
            }
        }
        break;

    case MoveType::PAIR:
        for (int r = ref.rank + 1; r <= RANK_SJ; ++r) {
            if (counts[r] >= 2) {
                CardMask m = extract_n_from_rank(hand, r, 2);
                try_beat(MoveType::PAIR, r, 0, m);
            }
        }
        break;

    case MoveType::TRIPLE:
        for (int r = ref.rank + 1; r <= RANK_2; ++r) {
            if (counts[r] >= 3) {
                CardMask m = extract_n_from_rank(hand, r, 3);
                try_beat(MoveType::TRIPLE, r, 0, m);
            }
        }
        break;

    case MoveType::TRIPLE_ONE:
    case MoveType::TRIPLE_TWO: {
        bool need_pair = (ref.type == MoveType::TRIPLE_TWO);
        for (int r = ref.rank + 1; r <= RANK_2; ++r) {
            if (counts[r] >= 3) {
                CardMask triple = extract_n_from_rank(hand, r, 3);
                if (need_pair) {
                    for (int k = 0; k <= RANK_SJ; ++k) {
                        if (k != r && counts[k] >= 2) {
                            CardMask pair = extract_n_from_rank(hand, k, 2);
                            try_beat_idx(MoveType::TRIPLE_TWO, r, 0, triple | pair, k);
                        }
                    }
                } else {
                    for (int k = 0; k < NUM_RANKS; ++k) {
                        if (k != r && counts[k] >= 1) {
                            CardMask kicker = extract_n_from_rank(hand, k, 1);
                            try_beat_idx(MoveType::TRIPLE_ONE, r, 0, triple | kicker, k);
                        }
                    }
                }
            }
        }
        break;
    }

    case MoveType::STRAIGHT: {
        int len = ref.length;
        for (int start = RANK_3; start + len - 1 <= RANK_A; ++start) {
            if (start <= ref.rank) continue;
            bool ok = true;
            for (int r = start; r < start + len; ++r)
                if (counts[r] < 1) { ok = false; break; }
            if (ok) {
                CardMask mask = 0;
                for (int r = start; r < start + len; ++r)
                    mask |= extract_n_from_rank(hand, r, 1);
                try_beat(MoveType::STRAIGHT, start, len, mask);
            }
        }
        break;
    }

    case MoveType::DOUBLE_STRAIGHT: {
        int len = ref.length;
        for (int start = RANK_3; start + len - 1 <= RANK_A; ++start) {
            if (start <= ref.rank) continue;
            bool ok = true;
            for (int r = start; r < start + len; ++r)
                if (counts[r] < 2) { ok = false; break; }
            if (ok) {
                CardMask mask = 0;
                for (int r = start; r < start + len; ++r)
                    mask |= extract_n_from_rank(hand, r, 2);
                try_beat(MoveType::DOUBLE_STRAIGHT, start, len, mask);
            }
        }
        break;
    }

    case MoveType::PLANE:
    case MoveType::PLANE_WING_SINGLE:
    case MoveType::PLANE_WING_PAIR: {
        int len = ref.length;
        MoveType ft = ref.type;
        bool need_singles = (ft == MoveType::PLANE_WING_SINGLE);
        bool need_pairs = (ft == MoveType::PLANE_WING_PAIR);
        for (int start = RANK_3; start + len - 1 <= RANK_A; ++start) {
            if (start <= ref.rank) continue;
            bool ok = true;
            for (int r = start; r < start + len; ++r)
                if (counts[r] < 3) { ok = false; break; }
            if (!ok) continue;

            CardMask plane = 0;
            for (int r = start; r < start + len; ++r)
                plane |= extract_n_from_rank(hand, r, 3);

            if (!need_singles && !need_pairs) {
                try_beat(MoveType::PLANE, start, len, plane);
                continue;
            }

            if (need_singles) {
                std::vector<int> avail;
                for (int r = 0; r < NUM_RANKS; ++r)
                    if ((r < start || r >= start + len) && counts[r] >= 1)
                        avail.push_back(r);
                int n = (int)avail.size();
                if (n >= len) {
                    std::vector<int> comb(len);
                    for (int i = 0; i < len; ++i) comb[i] = i;
                    while (true) {
                        CardMask wings = 0;
                        int ks[6] = {0};
                        for (int i = 0; i < len; ++i) {
                            ks[i] = avail[comb[i]];
                            wings |= extract_n_from_rank(hand, ks[i], 1);
                        }
                        try_beat_idx(MoveType::PLANE_WING_SINGLE, start, len, plane | wings,
                                     ks[0], ks[1], ks[2], ks[3], ks[4], ks[5]);
                        int i = len - 1;
                        while (i >= 0 && comb[i] == n - len + i) --i;
                        if (i < 0) break;
                        comb[i]++;
                        for (int j = i + 1; j < len; ++j) comb[j] = comb[j-1] + 1;
                    }
                }
            }
            if (need_pairs) {
                std::vector<int> avail;
                for (int r = 0; r <= RANK_SJ; ++r) {
                    if (r >= start && r < start + len) continue;
                    if (counts[r] >= 2) avail.push_back(r);
                }
                int n = (int)avail.size();
                if (n >= len) {
                    std::vector<int> comb(len);
                    for (int i = 0; i < len; ++i) comb[i] = i;
                    while (true) {
                        CardMask pw = 0;
                        int ks[6] = {0};
                        for (int i = 0; i < len; ++i) {
                            ks[i] = avail[comb[i]];
                            pw |= extract_n_from_rank(hand, ks[i], 2);
                        }
                        if ((pw & plane) == 0)
                            try_beat_idx(MoveType::PLANE_WING_PAIR, start, len, plane | pw,
                                         ks[0], ks[1], ks[2], ks[3], ks[4], ks[5]);
                        int i = len - 1;
                        while (i >= 0 && comb[i] == n - len + i) --i;
                        if (i < 0) break;
                        comb[i]++;
                        for (int j = i + 1; j < len; ++j) comb[j] = comb[j-1] + 1;
                    }
                }
            }
        }
        break;
    }

    case MoveType::BOMB:
        for (int r = ref.rank + 1; r <= RANK_2; ++r) {
            if (counts[r] >= 4) {
                CardMask m = extract_n_from_rank(hand, r, 4);
                try_beat(MoveType::BOMB, r, 0, m);
            }
        }
        break;

    case MoveType::ROCKET:
        break;

    case MoveType::FOUR_TWO_SINGLE:
    case MoveType::FOUR_TWO_PAIR: {
        bool need_pairs = (ref.type == MoveType::FOUR_TWO_PAIR);
        for (int r = ref.rank + 1; r <= RANK_2; ++r) {
            if (counts[r] >= 4) {
                CardMask bomb = extract_n_from_rank(hand, r, 4);
                if (need_pairs) {
                    for (int k1 = 0; k1 <= RANK_SJ; ++k1) {
                        if (k1 == r || counts[k1] < 2) continue;
                        for (int k2 = k1 + 1; k2 <= RANK_SJ; ++k2) {
                            if (k2 == r || counts[k2] < 2) continue;
                            CardMask pairs = extract_n_from_rank(hand, k1, 2) | extract_n_from_rank(hand, k2, 2);
                            if ((pairs & bomb) == 0)
                                try_beat_idx(MoveType::FOUR_TWO_PAIR, r, 0, bomb | pairs, k1, k2);
                        }
                    }
                } else {
                    for (int k1 = 0; k1 < NUM_RANKS; ++k1) {
                        if (k1 == r || counts[k1] < 1) continue;
                        for (int k2 = k1 + 1; k2 < NUM_RANKS; ++k2) {
                            if (k2 == r || counts[k2] < 1) continue;
                            CardMask kickers = extract_n_from_rank(hand, k1, 1) | extract_n_from_rank(hand, k2, 1);
                            try_beat_idx(MoveType::FOUR_TWO_SINGLE, r, 0, bomb | kickers, k1, k2);
                        }
                    }
                }
            }
        }
        break;
    }

    default:
        break;
    }

    // Bombs can always override non-rocket moves; rocket always beats everything
    if (ref.type != MoveType::ROCKET) {
        for (int r = 0; r <= RANK_2; ++r) {
            if (counts[r] >= 4) {
                CardMask m = extract_n_from_rank(hand, r, 4);
                try_beat(MoveType::BOMB, r, 0, m);
            }
        }
        if (counts[RANK_SJ] >= 1 && counts[RANK_BJ] >= 1) {
            CardMask m = extract_n_from_rank(hand, RANK_SJ, 1) | extract_n_from_rank(hand, RANK_BJ, 1);
            try_beat(MoveType::ROCKET, RANK_BJ, 0, m);
        }
    }
}

// ============================================================
// Print state
// ============================================================
void Game::print_state() const {
    std::cout << "=== Game State ===" << std::endl;
    if (phase_ == Phase::WAITING) { std::cout << "Phase: WAITING" << std::endl; return; }
    if (phase_ == Phase::AUCTION) { std::cout << "Phase: AUCTION" << std::endl; }
    if (phase_ == Phase::PLAYING) { std::cout << "Phase: PLAYING" << std::endl; }
    if (phase_ == Phase::FINISHED) {
        std::cout << "Phase: FINISHED, winner=" << winner_ << std::endl;
    }

    for (int p = 0; p < NUM_PLAYERS; ++p) {
        std::cout << "Player " << p << " (" << (roles_[p] == Role::LANDLORD ? "L" : "F") << "): "
                  << mask_to_string(hands_[p])
                  << "  (" << mask_count(hands_[p]) << " cards)" << std::endl;
    }
    std::cout << "Bonus: " << mask_to_string(landlord_bonus_) << std::endl;
    std::cout << "Current player: " << current_player_ << std::endl;
    if (last_move_.is_valid()) {
        std::cout << "Last move: " << move_to_string(last_move_) << std::endl;
    }
    std::cout << "Pass count: " << pass_count_ << std::endl;
}

} // namespace doudizhu
