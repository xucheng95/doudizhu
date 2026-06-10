#ifndef DOUDIZHU_GAME_HPP
#define DOUDIZHU_GAME_HPP

#include "cards.hpp"
#include <vector>
#include <array>
#include <functional>
#include <unordered_map>

namespace doudizhu {

// ============================================================
// Game phases
// ============================================================
enum class Phase : uint8_t {
    WAITING,   // Before game starts
    AUCTION,   // Landlord bidding
    PLAYING,   // Card play
    FINISHED   // Game over
};

// Player roles
enum class Role : uint8_t {
    PEASANT_0 = 0,
    PEASANT_1 = 1,
    LANDLORD  = 2
};

constexpr int NUM_PLAYERS = 3;

// ============================================================
// Observation data (for RL)
// ============================================================
struct Observation {
    std::array<float, NUM_CARDS> hand;          // 54-dim: current player's hand (bit binary)
    std::array<float, NUM_CARDS> landlord_cards; // 54-dim: landlord's bonus cards
    std::array<float, NUM_PLAYERS> num_cards;    // 3-dim: card count each player has
    std::array<float, NUM_PLAYERS> role;         // 3-dim one-hot: PEASANT_0, PEASANT_1, LANDLORD
    // Last play: encode the last non-pass move that was played
    std::array<float, NUM_RANKS> last_move_rank; // 15-dim: one-hot rank of last move
    std::array<float, NUM_PLAYERS> last_player;  // 3-dim: who played last

    void fill_hand(const CardMask& mask) { hand.fill(0); _mask_to_bin(mask, hand); }
    void fill_landlord(const CardMask& mask) { landlord_cards.fill(0); _mask_to_bin(mask, landlord_cards); }

private:
    static void _mask_to_bin(CardMask m, std::array<float, NUM_CARDS>& arr) {
        while (m) { int id = mask_pop_lsb(m); arr[id] = 1.0f; }
    }
};

// ============================================================
// Game class: core state machine
// ============================================================
class Game {
public:
    Game();

    // --- Lifecycle ---
    void reset();                                    // Start a new game (deal, no auction)
    void reset_with_auction();                       // Full reset including a simulated auction

    // --- Step ---
    // Apply the action for the current player.
    // `action_idx` selects the move from `get_legal_actions()`.
    bool step(int action_idx);
    bool step_move(const Move& move);                // Step with a Move object

    // --- Query ---
    const std::vector<Move>& get_legal_actions() const { return legal_actions_; }
    const std::vector<Move>& get_current_player_moves() const;  // re-compute if needed

    bool is_terminal() const { return phase_ == Phase::FINISHED; }
    bool is_auction() const { return phase_ == Phase::AUCTION; }

    int current_player() const { return current_player_; }
    Phase phase() const { return phase_; }
    Role role(int player) const { return roles_[player]; }

    const std::array<CardMask, NUM_PLAYERS>& hands() const { return hands_; }
    CardMask hand(int player) const { return hands_[player]; }
    CardMask landlord_bonus() const { return landlord_bonus_; }

    const Move& last_move() const { return last_move_; }
    int last_player() const { return last_player_; }
    int pass_count() const { return pass_count_; }  // consecutive passes since last play

    // Rewards: +1/-1 from perspective of the given player
    float reward(int player) const;

    // Winner
    int winner() const { return winner_; }

    // --- History ---
    const std::vector<PlayRecord>& play_history() const { return play_history_; }

    // --- Observation builder ---
    Observation get_observation(int player) const;

    // --- Debug ---
    void print_state() const;

    // --- Static utilities ---
    static Move make_move(MoveType type, CardMask mask, int rank, int length = 0);
    static Move make_move_idx(MoveType type, CardMask mask, int rank, int length,
                               int k0=0, int k1=0, int k2=0, int k3=0, int k4=0, int k5=0);
    static CardMask cards_of_rank(int rank);
    static CardMask extract_n_from_rank(CardMask hand, int rank, int n);
    static bool can_beat(const Move& m1, const Move& m2);
    static Move recognize_move(CardMask mask);

    // --- Action index computation (match Python action-table layout) ---
    static int encode_action_idx(MoveType type, int rank, int length,
                                  int k0=0, int k1=0, int k2=0, int k3=0, int k4=0, int k5=0);
    static void init_action_table();
    static int num_actions() { return 14636; }

private:
    // State
    Phase phase_;
    int current_player_;
    int last_player_;
    int pass_count_;              // how many players passed consecutively
    Move last_move_;
    int winner_;

    std::array<CardMask, NUM_PLAYERS> hands_;
    std::array<Role, NUM_PLAYERS> roles_;
    CardMask landlord_bonus_;

    // Landlord index (set after auction)
    int landlord_idx_;

    // Legal actions cache
    mutable std::vector<Move> legal_actions_;
    mutable int legal_actions_player_;

    // Play history
    std::vector<PlayRecord> play_history_;

    // --- Internal helpers ---
    void deal_cards();
    void simulate_auction();

    // --- Move generation ---
    // Given a hand and a target move to beat (nullptr = lead), produce all legal moves
    void compute_legal_actions();

    // Generate lead moves (no last move to beat)
    void generate_lead_moves(CardMask hand, const std::array<int, NUM_RANKS>& counts, std::vector<Move>& out);
    // Generate follow moves that beat reference_move
    void generate_follow_moves(CardMask hand, const std::array<int, NUM_RANKS>& counts,
                               const Move& ref, std::vector<Move>& out);

};

} // namespace doudizhu

#endif // DOUDIZHU_GAME_HPP
