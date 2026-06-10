#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "cards.hpp"
#include "game.hpp"
#include "env.hpp"

namespace py = pybind11;
namespace dd = doudizhu;

// Helper: Convert std::array<float, N> to numpy array
template <size_t N>
py::array_t<float> array_to_numpy(const std::array<float, N>& arr) {
    auto result = py::array_t<float>(N);
    auto buf = result.request();
    float* ptr = static_cast<float*>(buf.ptr);
    std::copy(arr.begin(), arr.end(), ptr);
    return result;
}

PYBIND11_MODULE(doudizhu_cpp, m) {
    m.doc() = "Doudizhu (Fight the Landlord) RL environment - C++ backend";

    // ===== Cards =====
    py::class_<dd::PlayRecord>(m, "PlayRecord")
        .def(py::init<>())
        .def_readonly("player", &dd::PlayRecord::player)
        .def_readonly("move", &dd::PlayRecord::move);

    py::class_<dd::Move>(m, "Move")
        .def(py::init<>())
        .def_readonly("type", &dd::Move::type)
        .def_readonly("rank", &dd::Move::rank)
        .def_readonly("length", &dd::Move::length)
        .def_readonly("count", &dd::Move::count)
        .def_readonly("mask", &dd::Move::mask)
        .def_readonly("action_idx", &dd::Move::action_idx)
        .def("is_valid", &dd::Move::is_valid)
        .def("__repr__", [](const dd::Move& m) {
            return dd::move_to_string(m);
        });

    py::enum_<dd::MoveType>(m, "MoveType")
        .value("NONE", dd::MoveType::NONE)
        .value("SINGLE", dd::MoveType::SINGLE)
        .value("PAIR", dd::MoveType::PAIR)
        .value("TRIPLE", dd::MoveType::TRIPLE)
        .value("TRIPLE_ONE", dd::MoveType::TRIPLE_ONE)
        .value("TRIPLE_TWO", dd::MoveType::TRIPLE_TWO)
        .value("STRAIGHT", dd::MoveType::STRAIGHT)
        .value("DOUBLE_STRAIGHT", dd::MoveType::DOUBLE_STRAIGHT)
        .value("PLANE", dd::MoveType::PLANE)
        .value("PLANE_WING_SINGLE", dd::MoveType::PLANE_WING_SINGLE)
        .value("PLANE_WING_PAIR", dd::MoveType::PLANE_WING_PAIR)
        .value("BOMB", dd::MoveType::BOMB)
        .value("ROCKET", dd::MoveType::ROCKET)
        .value("FOUR_TWO_SINGLE", dd::MoveType::FOUR_TWO_SINGLE)
        .value("FOUR_TWO_PAIR", dd::MoveType::FOUR_TWO_PAIR)
        .value("PASS", dd::MoveType::PASS)
        .export_values();

    py::enum_<dd::Role>(m, "Role")
        .value("PEASANT_0", dd::Role::PEASANT_0)
        .value("PEASANT_1", dd::Role::PEASANT_1)
        .value("LANDLORD", dd::Role::LANDLORD)
        .export_values();

    // ===== Observation =====
    py::class_<dd::Observation>(m, "Observation")
        .def(py::init<>())
        .def_property_readonly("hand", [](const dd::Observation& obs) {
            return array_to_numpy<dd::NUM_CARDS>(obs.hand);
        })
        .def_property_readonly("landlord_cards", [](const dd::Observation& obs) {
            return array_to_numpy<dd::NUM_CARDS>(obs.landlord_cards);
        })
        .def_property_readonly("num_cards", [](const dd::Observation& obs) {
            return array_to_numpy<dd::NUM_PLAYERS>(obs.num_cards);
        })
        .def_property_readonly("role", [](const dd::Observation& obs) {
            return array_to_numpy<dd::NUM_PLAYERS>(obs.role);
        })
        .def_property_readonly("last_move_rank", [](const dd::Observation& obs) {
            return array_to_numpy<dd::NUM_RANKS>(obs.last_move_rank);
        })
        .def_property_readonly("last_player", [](const dd::Observation& obs) {
            return array_to_numpy<dd::NUM_PLAYERS>(obs.last_player);
        });

    // ===== EnvConfig =====
    py::class_<dd::EnvConfig>(m, "EnvConfig")
        .def(py::init<>())
        .def_readwrite("use_auction", &dd::EnvConfig::use_auction)
        .def_readwrite("self_play", &dd::EnvConfig::self_play);

    // ===== StepResult =====
    py::class_<dd::DoudizhuEnv::StepResult>(m, "StepResult")
        .def(py::init<>())
        .def_readwrite("obs", &dd::DoudizhuEnv::StepResult::obs)
        .def_readwrite("reward", &dd::DoudizhuEnv::StepResult::reward)
        .def_readwrite("terminated", &dd::DoudizhuEnv::StepResult::terminated)
        .def_readwrite("truncated", &dd::DoudizhuEnv::StepResult::truncated)
        .def_readwrite("legal_actions", &dd::DoudizhuEnv::StepResult::legal_actions);

    // ===== DoudizhuEnv =====
    py::class_<dd::DoudizhuEnv>(m, "DoudizhuEnv")
        .def(py::init<const dd::EnvConfig&>(), py::arg("config") = dd::EnvConfig{})

        .def("reset", &dd::DoudizhuEnv::reset,
             py::arg("player_id") = -1,
             "Reset the environment. Returns observation for the given player.")

        .def("step", &dd::DoudizhuEnv::step,
             py::arg("action_idx"),
             "Take an action. Returns StepResult with obs, reward, done, info.")

        .def("legal_actions", &dd::DoudizhuEnv::legal_actions,
             "Get legal actions for the current player as list of Moves.")

        .def("legal_action_mask", &dd::DoudizhuEnv::legal_action_mask,
             "Get binary mask over legal actions.")

        .def("get_observation", [](dd::DoudizhuEnv& env, int player) {
            return env.game().get_observation(player);
        }, py::arg("player"),
           "Get observation for a specific player.")

        .def_property_readonly("game", [](dd::DoudizhuEnv& env) -> dd::Game& {
            return env.game();
        }, py::return_value_policy::reference_internal)

        .def_property("observe_player",
            &dd::DoudizhuEnv::observe_player,
            &dd::DoudizhuEnv::set_observe_player)

        .def("set_seed", &dd::DoudizhuEnv::set_seed,
             py::arg("seed"))

        .def_property_readonly("num_moves", &dd::DoudizhuEnv::num_moves);

    // ===== Game (for advanced access) =====
    py::class_<dd::Game>(m, "Game")
        .def(py::init<>())
        .def("reset", &dd::Game::reset)
        .def("reset_with_auction", &dd::Game::reset_with_auction)
        .def("step", &dd::Game::step, py::arg("action_idx"))
        .def("step_move", &dd::Game::step_move, py::arg("move"))
        .def("is_terminal", &dd::Game::is_terminal)
        .def("current_player", &dd::Game::current_player)
        .def("winner", &dd::Game::winner)
        .def("reward", &dd::Game::reward, py::arg("player"))
        .def("get_observation", &dd::Game::get_observation, py::arg("player"))
        .def("play_history", &dd::Game::play_history)
        .def("last_move", &dd::Game::last_move, py::return_value_policy::reference_internal)
        .def("last_player", &dd::Game::last_player)
        .def("print_state", &dd::Game::print_state)
        .def("legal_actions", [](dd::Game& g) -> const std::vector<dd::Move>& {
            return g.get_current_player_moves();
        }, py::return_value_policy::reference_internal);

    // ===== Utility functions =====
    m.def("full_deck", &dd::full_deck);
    m.def("mask_to_string", &dd::mask_to_string);
    m.def("card_name", &dd::card_name);
    m.def("recognize_move", &dd::Game::recognize_move);

    // Constants
    m.attr("NUM_CARDS") = dd::NUM_CARDS;
    m.attr("NUM_PLAYERS") = dd::NUM_PLAYERS;
    m.attr("NUM_RANKS") = dd::NUM_RANKS;
}
