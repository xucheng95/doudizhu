#include "env.hpp"
#include <algorithm>
#include <cassert>

namespace doudizhu {

DoudizhuEnv::DoudizhuEnv(const EnvConfig& config)
    : config_(config)
    , observe_player_(0)
    , num_moves_(0)
    , rng_(std::random_device{}())
    , use_rng_(true)
{}

void DoudizhuEnv::set_seed(unsigned s) {
    rng_.seed(s);
    use_rng_ = true;
}

Observation DoudizhuEnv::reset(int player_id) {
    if (player_id >= 0 && player_id < NUM_PLAYERS) {
        observe_player_ = player_id;
    }

    // Reset game
    if (config_.use_auction) {
        game_.reset_with_auction();
        game_.step(0);  // Auto-resolve auction
    } else {
        game_.reset();
    }

    num_moves_ = 0;
    if (!config_.self_play) advance_game_random();
    return game_.get_observation(observe_player_);
}

DoudizhuEnv::StepResult DoudizhuEnv::step(int action_idx) {
    StepResult result;

    if (game_.is_terminal()) {
        result.terminated = true;
        result.obs = game_.get_observation(observe_player_);
        result.reward = game_.reward(observe_player_);
        return result;
    }

    // Get legal actions
    const auto& actions = legal_actions();

    // Apply action
    if (action_idx >= 0 && (size_t)action_idx < actions.size()) {
        game_.step(action_idx);
        num_moves_++;
    }

    // If not self-play, let opponent players make random moves
    if (!config_.self_play) {
        advance_game_random();
    }

    // Check terminal
    result.terminated = game_.is_terminal();
    result.obs = game_.get_observation(observe_player_);
    result.reward = game_.reward(observe_player_);
    result.legal_actions = legal_actions();
    return result;
}

const std::vector<Move>& DoudizhuEnv::legal_actions() const {
    return game_.get_current_player_moves();
}

std::vector<float> DoudizhuEnv::legal_action_mask() const {
    // In the C++ version, we return a mask over legal actions vector
    // The Python side maps this differently based on the action space definition
    const auto& actions = legal_actions();
    std::vector<float> mask(actions.size(), 1.0f);
    return mask;
}

void DoudizhuEnv::advance_game_random() {
    // Fill remaining opponent moves with random actions until
    // it's the observed player's turn (in non-self-play mode)
    while (!game_.is_terminal() && game_.current_player() != observe_player_) {
        const auto& actions = game_.get_current_player_moves();
        if (actions.empty()) break;  // Shouldn't happen

        // Choose random action
        if (use_rng_) {
            std::uniform_int_distribution<int> dist(0, (int)actions.size() - 1);
            int idx = dist(rng_);
            game_.step(idx);
        } else {
            game_.step(0);
        }
        num_moves_++;
    }
}

} // namespace doudizhu
