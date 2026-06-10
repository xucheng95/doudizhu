import sys, os, torch
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'doudizhu', 'python', 'doudizhu'))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'doudizhu', 'python'))
from training.config import TrainingConfig
from training.model import DoudizhuAgent
from training.eval import evaluate

def test_evaluate_returns_win_rates():
    cfg = TrainingConfig(d_model=64, d_action=32, num_layers=2, num_heads=2,
                         ff_dim=64, max_actions=500, max_history=60)
    agents = {
        "landlord": DoudizhuAgent(cfg),
        "peasant0": DoudizhuAgent(cfg),
        "peasant1": DoudizhuAgent(cfg),
    }
    result = evaluate(agents, cfg, n_games=10, device=torch.device("cpu"))
    assert "landlord_win_rate" in result
    assert "peasant_win_rate" in result
    assert "avg_game_length" in result
    assert 0.0 <= result["landlord_win_rate"] <= 1.0
    assert 0.0 <= result["peasant_win_rate"] <= 1.0
    assert abs(result["landlord_win_rate"] + result["peasant_win_rate"] - 1.0) < 1e-5
