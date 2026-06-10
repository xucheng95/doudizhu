from doudizhu.env_wrapper import DoudizhuGymEnv, doudizhu_env_creator
import gymnasium

# Register the environment
gymnasium.register(
    id="Doudizhu-v0",
    entry_point="doudizhu.env_wrapper:DoudizhuGymEnv",
)

__all__ = ["DoudizhuGymEnv", "doudizhu_env_creator"]
