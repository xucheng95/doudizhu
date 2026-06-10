# DouDizhu RL Environment

斗地主强化学习环境，C++ 核心 + pybind11 绑定 + Gymnasium 接口。含完整 MAPPO 训练管线。

## 快速开始

```bash
git clone https://github.com/xucheng95/doudizhu.git
cd doudizhu

# 一键搭建虚拟环境（创建 venv、安装依赖、编译 C++ 扩展）
bash scripts/setup.sh
source venv/bin/activate

# 启动训练
bash scripts/train.sh
```

## 训练架构

```
                     ┌─ Worker 1 (常驻, 持env+model副本) ─┐
主进程 ──submit_work─┼─ Worker 2 ────────────────────────┼── collect_work → 轨迹
                     └─ Worker 3/4 ──────────────────────┘
                     
                     ┌─ Learner landlord  (常驻, 持model) ─┐
主进程 ──submit_learn┼─ Learner peasant0 ─────────────────┼── collect_learn → 梯度更新
                     └─ Learner peasant1 ─────────────────┘

eval 后台进程（每50轮），checkpoint 每100轮
```

| 进程 | 数量 | 职责 |
|---|---|---|
| 主进程 | 1 | 调度 + 日志 + TensorBoard |
| Worker | 4 | 自对弈采样，返回轨迹 |
| Learner | 3 | 各角色独立 PPO 训练 |
| Eval | 1（后台） | 200局贪婪策略评估 |

## 观测

| 字段 | 类型 | 维度 | 说明 |
|---|---|---|---|
| `hands` | float32 | (3,54) | [0]=self, [1]=下家, [2]=上家 |
| `landlord_cards` | float32 | (54,) | 地主底牌 |
| `num_cards` | float32 | (3,) | [0]=self, [1]=下家, [2]=上家 |
| `role` | float32 | (3,) | one-hot: [农民0, 农民1, 地主] |
| `last_player` | int | — | 上家 0/1/2 |
| `history` | list[dict] | 变长 | 整局出牌记录 |
| `legal_moves` | list[dict] | 变长 | `{type, rank, length, kickers, cards, pass}` |

## 动作

`Discrete(2048)` — 索引对应 `legal_moves` 列表，模型输出 `0 ~ len(legal_moves)-1`。

## 模型

| 组件 | 结构 |
|---|---|
| StateEncoder | Transformer (state_token + history) → state_emb |
| ActionEncoder | Embedding(type/rank/length/kickers/cards) → action_embs |
| Actor | Cross-Attention(Q=state_emb, K/V=action_embs) → logits |
| Critic | MLP(state_emb ‖ all_hands) → value |

## 配置

`configs/default.yaml`:

```yaml
d_model: 192          # Transformer 隐层维度
num_layers: 3         # Transformer 层数
num_workers: 4        # 采样并行数
episodes_per_batch: 128  # 每轮采样局数
ppo_epochs: 2         # PPO 更新轮数
batch_size: 128       # mini-batch 大小
eval_interval: 50     # 评估间隔（轮）
checkpoint_interval: 100
```

## 牌型覆盖

| 牌型 | 说明 |
|---|---|
| SINGLE / PAIR / TRIPLE | 单张、对子、三张 |
| TRIPLE_ONE / TRIPLE_TWO | 三带一/二（带牌独立） |
| STRAIGHT | 顺子（5-12张） |
| DOUBLE_STRAIGHT | 连对（3-10对） |
| PLANE / WSINGLE / WPAIR | 飞机/带单/带对 |
| BOMB / ROCKET | 炸弹/火箭 |
| FOUR_TWO_SINGLE / PAIR | 四带二（带牌独立） |
| PASS | 过牌 |

## 牌面编码

- Card ID: 0-53。0-3 = 4张3，...，52=小王，53=大王
- 手牌: `uint64_t` 位掩码
- `rank = id / 4`

## 奖励

| | 地主胜 | 农民胜 |
|---|---|---|
| 地主 | +2 | -2 |
| 农民 | -1 | +1 |

## 环境性能

| 模式 | moves/s | games/s |
|---|---|---|
| C++ 核心 | 450K | 23K |
| Gymnasium 包装 | 10.5K | 522 |

## 测试

```bash
python doudizhu/python/test_full.py
```

## 许可

MIT
