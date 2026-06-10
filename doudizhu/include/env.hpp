#ifndef DOUDIZHU_ENV_HPP
#define DOUDIZHU_ENV_HPP

#include "game.hpp"
#include <vector>
#include <random>

namespace doudizhu {

// ============================================================
// DoudizhuEnv - RL environment (C++ side)
//
// Supports both 1v2 and single-player agent-against-random modes.
// Gym-style API: reset, step, get_obs, get_reward, is_terminal.
// ============================================================

// Configuration
struct EnvConfig {
    bool use_auction = false;  // Whether to simulate auction phase
    bool self_play = false;    // If true, step() takes 3 actions per loop
};

class DoudizhuEnv {
public:
    explicit DoudizhuEnv(const EnvConfig& config = EnvConfig{});

    // Reset the environment. Returns initial observation for the current player.
    // If player_id is valid (0-2), returns obs for that player.
    Observation reset(int player_id = -1);

    // Take an action (index into current player's legal actions).
    // Returns {observation, reward, terminal, legal_actions} for the player
    // whose turn it is (or the observing player, if configured).
    //
    // In self-play mode, call step() for each action; internally advances the game.
    struct StepResult {
        Observation obs;
        float reward = 0.0f;
        bool terminated = false;
        bool truncated = false;
        std::vector<Move> legal_actions;
    };

    StepResult step(int action_idx);

    // Get legal actions for the current player (as Move objects)
    const std::vector<Move>& legal_actions() const;

    // Get legal actions encoded as a binary mask over the full action space
    // 0 = invalid, 1 = valid
    std::vector<float> legal_action_mask() const;

    // Access to underlying game
    const Game& game() const { return game_; }
    Game& game() { return game_; }

    // Current observing player
    int observe_player() const { return observe_player_; }
    void set_observe_player(int p) { observe_player_ = p; }

    // Episodic info
    int num_moves() const { return num_moves_; }
    void set_seed(unsigned s);

private:
    Game game_;
    EnvConfig config_;
    int observe_player_;  // Which player's perspective we observe
    int num_moves_;

    // RNG for random opponents
    std::mt19937 rng_;
    bool use_rng_;

    void advance_game_random();  // Fill opponent moves with random actions
};

} // namespace doudizhu

#endif // DOUDIZHU_ENV_HPP
