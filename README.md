# DouDizhu RL Environment

斗地主强化学习环境，C++ 核心 + pybind11 绑定 + Gymnasium 接口。主打高性能采样，支持单智能体和多智能体训练。

## 性能

| 模式 | moves/s | games/s |
|---|---|---|
| C++ 纯环境（随机走子） | 450K | 23K |
| Gymnasium 包装（单智能体） | 10.5K | 522 |

推理 20ms/步时，环境开销占比 < 1%，瓶颈在推理。

## 项目结构

```
doudizhu/
├── include/
│   ├── cards.hpp       # 牌面编码、位操作、Move/PlayRecord 类型
│   ├── game.hpp        # 游戏状态机、规则引擎、动作枚举
│   └── env.hpp         # RL 环境接口（step/reset/obs）
├── src/
│   ├── cards.cpp       # 牌面字符串转换
│   ├── game.cpp        # 规则逻辑 + action 查找表
│   └── env.cpp         # 环境实现
├── pybind/
│   └── bindings.cpp    # pybind11 绑定
├── python/doudizhu/
│   ├── env_wrapper.py  # Gymnasium.Env 实现
│   └── __init__.py     # 环境注册
├── build.sh            # 一键编译
├── training/           # MAPPO 训练
└── tests/              # 单元测试
```

## 快速开始

### 编译

```bash
cd doudizhu
bash build.sh          # macOS (clang + Python framework)

# 或 CMake 跨平台：
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
```

### 安装

```bash
cd doudizhu/python
pip install -e .
```

### 使用

```python
import gymnasium as gym
import doudizhu  # 注册 Doudizhu-v0

env = gym.make("Doudizhu-v0")
obs, info = env.reset()

# obs 结构见下文

done = False
while not done:
    n = len(obs["legal_moves"])        # 可选动作数
    action = model(obs)                 # 模型输出 0 ~ n-1
    obs, reward, done, _, info = env.step(action)
```

## Observation

| 字段 | 类型 | 维度 | 说明 |
|---|---|---|---|
| `hands` | float32 | (3,54) | 玩家手牌：[0]=self, [1]=下家, [2]=上家 |
| `landlord_cards` | float32 | (54,) | 地主底牌（3张）bit 掩码 |
| `num_cards` | float32 | (3,) | 剩余牌数：[0]=self, [1]=下家, [2]=上家 |
| `role` | float32 | (3,) | 自身角色 one-hot：[农民0, 农民1, 地主] |
| `last_player` | int | — | 上家是谁（0/1/2） |
| `history` | list[dict] | 变长 | 整局出牌记录，最后一条非 PASS = 上家出牌 |
| `legal_moves` | list[dict] | 变长 | 当前可选动作 `{type, rank, length, kickers, cards, pass}` |

### legal_moves 示例

```python
[
    {"type": "SINGLE",  "rank": 6,  "cards": ["9"],              "pass": False},
    {"type": "PAIR",    "rank": 3,  "cards": ["6", "6"],         "pass": False},
    {"type": "BOMB",    "rank": 0,  "cards": ["3","3","3","3"], "pass": False},
    {"type": "TRIPLE_ONE","rank":5,"cards": ["8","8","8","A"],  "kickers": [11], "pass": False},
    {"type": "PASS",    "rank": -1, "cards": [],                 "pass": True},
]
```

## Action

`Discrete(2048)` — 索引直接对应 `legal_moves` 列表位置，模型输出 `0 ~ len(legal_moves)-1`。

## RL 训练 (MAPPO)

训练代码在 `training/`，基于 MAPPO（Multi-Agent PPO）+ Centralized Critic + Self-Play：

```
training/
├── config.py         # TrainingConfig 配置类
├── model.py          # DoudizhuAgent: StateEncoder + ActionEncoder + Critic
├── obs_encoder.py    # 观测 → tensor 编码
├── buffer.py         # RolloutBuffer: 存储轨迹、计算 GAE、生成 mini-batch
├── ppo.py            # PPOUpdater: K 轮 mini-batch 更新
├── rollout.py        # collect_episode: 自对弈采样
├── train.py          # 训练主循环 + TensorBoard 日志
├── eval.py           # evaluate: 贪婪策略评估胜率
└── history_pool.py   # HistoryPool: 历史模型池（Self-Play 对手采样）
```

### 架构

| 组件 | 说明 |
|---|---|
| StateEncoder | Transformer 编码 [state_token, hist_1, ..., hist_T] → state_emb |
| ActionEncoder | 合法动作编码（type/rank/length/kickers/cards embedding） → action_embs |
| Actor | Cross-attention(Q=state_emb, K/V=action_embs) → Categorical logits |
| Critic | MLP(state_emb ‖ all_hands) → scalar value（centralized，看全局面） |

### 启动训练

```bash
python training/train.py --config configs/default.yaml
```

### 评估

```python
from training.eval import evaluate
result = evaluate(agents, config, n_games=500, device=device)
# result: {"landlord_win_rate": 0.42, "peasant_win_rate": 0.58, "avg_game_length": 22.3}
```

## 牌型覆盖

全部斗地主合法牌型，带牌变体展开为独立动作：

| 牌型 | 说明 |
|---|---|
| SINGLE / PAIR / TRIPLE | 单张、对子、三张（3-K） |
| TRIPLE_ONE / TRIPLE_TWO | 三带一、三带二（带牌 rank 独立） |
| STRAIGHT | 顺子（5-12张连续单牌） |
| DOUBLE_STRAIGHT | 连对（3-10对连续对子） |
| PLANE / WSINGLE / WPAIR | 飞机、飞机带单、飞机带对（翼牌 rank 独立） |
| BOMB / ROCKET | 炸弹（四张同点）、火箭（王炸） |
| FOUR_TWO_SINGLE / PAIR | 四带二单、四带二对（带牌 rank 独立） |
| PASS | 过牌 |

## 牌面编码

- Card ID：0-53。0-3 = ♠3 ♥3 ♣3 ♦3，...，52 = 小王，53 = 大王
- 手牌用 `uint64_t` 位掩码存储
- `card_rank(id) = id / 4`（52/53 返回 13/14）

## 奖励

基础分 ×1，地主翻倍：

| 结果 | 地主 | 农民 |
|---|---|---|
| 地主胜 | +2 | -1 |
| 农民胜 | -2 | +1 |

## 测试

```bash
cd doudizhu/python
python test_full.py     # 速度基准 + 环境循环
python audit_rules.py   # 规则正确性审计
```

## 许可

MIT
