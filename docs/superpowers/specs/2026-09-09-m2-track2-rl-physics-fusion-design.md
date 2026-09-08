# M2 Track 2: Plan B RL 源重跑 → 融合重算 → fusion-level 评估 — 设计文档

**Goal:** 在 Plan B 对称物理(PhysicsEnv) + PPO ratio 修复就位后,**重跑 `rl` 源**、**重算三源融合**、并在**留出局**上做 **fusion-level 评估**,回答:"Plan B 物理 RL 源是否让融合后的宏观决策(global_mode)更贴合胜负?"以及"RL 源是否不再退化(学得动)?"。

**范围:** 只动 `rl` 源 + `labels/fusion.py` 的调用 + `eval/fusion_eval.py`(新). 这是 plan-b-physics-design §154 与 retrain-eval-design §9 中被明确**另立**的下游"轨2",必须在对称环境+PPO 修复完成后单独跑。

**Tech Stack:** RL head 纯 numpy + CPU IPPO(torch 惰性,`rl/train.py`);fusion-level 指标纯 numpy(无 torch/transformers/sklearn),单测离线可跑。

---

## 1. 为什么值得做(承接前一阶段)

Plan B 已确认修好 rollout 标注退化(标签分布从 100%/2.9% 守 → 24.8% 攻+均)。但 Plan B 的物理价值对 **LLM 蒸馏链路**只依赖其产出的 rollout 标签;对 **RL 源**则直接依赖"真 MDP 前向"。PPO ratio bug(修复前 `ratio` 恒 1 → 退化 A2C/REINFORCE)意味着**旧 RL 策略根本没被 PPO 训过**,RL 源在融合里约等于噪声。本轨在 PPO 修复后重跑 RL,衡量其能否作为真第三源提升融合。

## 2. 现状锚点(基线)

- 融合: `labels/fusion.fuse(rule, rl, llm=None)`,当前 `llm` 源在 `run_m2.py` 默认 `None`(DeepSeek 慢环跳过)→ 实际是 **rule + rl 双源**。
- 默认置信: `DEFAULT_CONFS = {'rule':0.5,'rl':0.3,'llm':0.2}`;哨兵通道 `CHANNEL_CONFS={'哨兵':{'rule':0.4,'rl':0.6,'llm':0.0}}`(RL 在哨兵有决策权,LLM 排空防与规则勾结)。
- 留出局: `out_m2/m2_report.json` 的 `eval_games` = `[1778661259223, 1778661970735, 1778663517697, 1778664469370]`(4 局,超出训练窗,用于公平对照)。
- RL 头: `run_m2.rl_head(dps, b, train_game_ids, rl_updates, physics=True)` → 在 PhysicsEnv 上训 IPPO → `rollout_to_decisions`。
- 指标地基: `eval.decision_gap.global_mode_align_gap`(`gap = mean_fav(进攻) − mean_fav(防守)`),fav 来自 `outcome_backtest.fav_from_worldstate(w,'红')`。

## 3. fusion-level 评估口径(本轨新增)

对每个留出局,产三套决策序列作对照:

1. **rule-only** 基线: `fuse(rule, rule, None)` — 确定性安全底座。
2. **rl-physics**: `fuse(rule, rl_physics, None)` — Plan B 物理 RL 源接入。
3. (参照,可选) **rl-bandit-M2Env**: `rl_head(..., physics=False)` 旧模式,用于对比"物理前向是否真的更好"。

对每套,用 `eval/fusion_eval` 量:

| 指标 | 定义 | 期望 |
|---|---|---|
| `legal_rate` | 合法分层决策占比(`sl.validate`) | ≥0.9(融合需保持合法) |
| `conflict_rate` | 融合冲突标记占比 | 记录, 不硬门控 |
| `mode_dist` | `global_mode` 计数归一 | **不再退化单模**(攻/守/均并存) |
| `align_gap` | `global_mode_align_gap` = mean_fav(进攻)−mean_fav(防守) | 期望 **rl-physics ≥ rule-only** |
| `base_atk_gap_等` | 见 `fav_from_worldstate` 组件(可选细粒度) | 记录 |

**承重判据(§9.4 的语义,因果诚实):** `align_gap(rl_fusion) ≥ align_gap(rule_only)` 且 `mode_dist` 含进攻+均衡(非单模)。这证明"物理 RL 源把进攻放到更有利的胜势区间、且不把分布挤成单模"。

**诚实上限:** `align_gap` 是**关联性/条件有利度**,非因果;`fav` 与 win 的可靠性地基见 `win_model_cv`(状态确实携带胜负)。留出仅 4 局,数值以方向与量级为准,不做精确统计检验。

## 4. 交付物

- `eval/fusion_eval.py`(新,纯 numpy,可单测): `mode_distribution`, `legal_rate`, `fusion_align_gap`(包装 decision_gap), `summarize_confront(rule_only, rl_fused, worldstates)`。
- `run_track2.py`(新,top-level runner,GPU-free): 对留出局产三套决策 → 融合评估 → 聚合 → 写报告。
- `tests/test_fusion_eval.py`(新,离线纯函数单测)。
- 报告: `out_m2/track2_rl_fusion_report.json`(不 commit)。

## 5. 命令

```bash
python run_track2.py --rl-updates 80 --out out_m2/track2_rl_fusion_report.json
```

## 6. 约束(承继项目)

- **不改 `data.db`**;只读。
- **不训/不评 LLM**(那是 Track 1);RL 源 CPU,不占 8G GPU。**若与 Track 1 同时段,错峰避免 8G 卡并发**(口诀: 别共享 8G 卡)。
- **不 commit** `out_m*`/`_work`/`out_m2/track2_*` 产物;`.gitignore` 只做增量添加。
- 每任务完成即校验(用户 workflow): 先跑小样本看方向,再全量。

## 7. Out of scope

- 不做新的 RL 网络/超参扫描(8G + IaaS 预算)。
- 不引入 LLM(`llm` 源保持 `None`,本轨只证明 RL 源)。
- 不改 `labels/fusion.py` 的 `CHANNEL_CONFS`/`DEFAULT_CONFS`(本轨复用,不调优权重)。
- 不做融合的 win-model CV 因果回归(留作后续;本轨用 align_gap 关联口径)。
