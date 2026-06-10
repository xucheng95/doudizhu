import pytest
from training.config import TrainingConfig

def test_default_config_fields():
    cfg = TrainingConfig()
    assert cfg.d_model == 256
    assert cfg.d_action == 64
    assert cfg.num_layers == 4
    assert cfg.num_heads == 4
    assert cfg.ff_dim == 512
    assert cfg.dropout == 0.1
    assert cfg.max_actions == 500
    assert cfg.max_history == 60
    assert cfg.gamma == 0.99
    assert cfg.gae_lambda == 0.95
    assert cfg.clip_eps == 0.2
    assert cfg.value_coeff == 0.5
    assert cfg.entropy_coeff == 0.01
    assert cfg.lr == 3e-4
    assert cfg.max_grad_norm == 0.5
    assert cfg.ppo_epochs == 4
    assert cfg.batch_size == 256
    assert cfg.num_workers == 8
    assert cfg.episodes_per_batch == 256
    assert cfg.max_epochs == 10000
    assert cfg.eval_interval == 50
    assert cfg.checkpoint_interval == 100

def test_config_from_yaml(tmp_path):
    import yaml
    yaml_content = {"d_model": 128, "num_layers": 2}
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.dump(yaml_content))
    cfg = TrainingConfig.from_yaml(str(p))
    assert cfg.d_model == 128
    assert cfg.num_layers == 2
    assert cfg.gamma == 0.99  # default preserved
