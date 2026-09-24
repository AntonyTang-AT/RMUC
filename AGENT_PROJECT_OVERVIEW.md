# AGENT PROJECT OVERVIEW — RMUC 2026 · M2 项目导读

> **给 Agent 的完整自足导读**。读完本文件即可理解项目思路、路径、仓库布局、已完成/未完成状态,以及**每个待办任务的确切启动方式与验收标准**。
> **超详细 Codex 交接**：见根目录 [`CODEX_HANDOFF.md`](CODEX_HANDOFF.md)（文件清单、命令大全、数字结论、下一步）。
> 最后更新: 2026-09-24。硬件: RTX 5060 8G / Windows 11。推理: 本机倾向 3B/无卡 Runtime；7B Tier2 在 AutoDL/有卡时打开。

---

## 0. 一句话

**交付主线**：规则头保底 + 通道修复融合（可选 RL）+ **lookahead 标签** → QLoRA 微调 LLM，从 WorldState 产出合法分层决策（`global_mode` + 六类单位动作）。

**延时分层（Phase A）**：简单帧走 Tier0/1（规则/启发/后续小模型·RL）；复杂帧标 `need_llm` 走 Tier2 微调大模型。敌方侧输出 **下一步意图概率**（`labels/opp_intent.py`），注入 digest；旁路挂 `decision_meta` / `opponent_intent`。叙事 = **重标注器（可慢）→ 轻学生（追延时）**，本阶段不重训全量 7B。

**研究附录**：Track2（Physics IPPO 仿真/硬门）与 Track1（rollout vs lookahead 对照，**已结案负结果**）。硬门过关 ≠「RL 已优于规则」。

---

## 1. 项目思路(在做什么)

**输入**: 比赛时间序列 → 决策点(每帧 WorldState)。**输出**: 每决策点的分层决策 = `{t, global_mode, units{英雄/工程/步兵3/步兵4/空中/哨兵: {move, target, fire, risk}}}`。

**融合源**(`labels/fusion.py`) — 主线默认 **不开 LLM 慢环**:
| 源 | 实现 | 主线角色 |
|---|---|---|
| 规则头 rule | `tree/global_mode.py` + `run_m2.rule_head` | 保底（必选） |
| RL rl | PhysicsEnv/M2Env + IPPO | 可选通道发言（哨兵 CHANNEL_CONFS） |
| LLM llm（慢环） | DeepSeek 等 | **默认剔除出主线**；仅 `--profile research` |

**蒸馏目标 LLM**：`finetune/lora_train.py`（Qwen2.5-3B QLoRA）— 这是考核交付物。

**诚实问题（附录回答）**:
- Track1：`rollout_gap_ge_lookahead` → **false**（旧 7B 与 v4 均未过，§7.11）
- Track2：Physics 是否学得动 / free 是否非退化 → v3fix2 **loaded=true**，但含进攻偏向启发式 caveat（§7.12–7.13）

---

## 2. 分阶段路径(在哪一步)

```
[✅] M0/M1   提示工程基线 + per_class 指标 + 诚实规则基线
[✅] M3      博弈论 + 分层 schema + 行为树先验
[✅] M2 定版 ★ 当前
        ├─[✅] 交付主线: --profile mainline（skip_llm + lookahead_delta=5）
        ├─[✅] 基建: confs 透传 / RL reward / Plan B 物理 / lookahead 标注
        ├─[✅] Track1: 结案负结果（rollout ≱ lookahead）
        └─[✅] Track2: 附录 hardgate 通过（caveat: 非「优于规则」证明）
```

**保留**: schema、规则头、CHANNEL_CONFS、QLoRA+模板 eval、Track2 底座（delta/coh0.25/fav0.02/from_units）、限幅 max_upsample=3。

**剔除出主线**: 默认 LLM 第三源、game_theory 作 Track2 主结论、fav≥0.08、无上限配平翻盘、把 hardgate 当产品成功定义。

---

## 3. 仓库布局(关键文件是什么)

**入口(`*.py` 顶层)**:
- `run_m0.py` / `run_m1.py` / `run_m3.py` / `run_llm_m3.py` / `run_llm_api.py`
- `run_m2.py`(**交付主线** orchestration；`--profile mainline|research`)
- `run_track2.py`(**研究附录** Track2，GPU-free)
- `config.py` / `blackboard.py`

**核心模块**:
| 目录 | 职责 |
|---|---|
| `data/` | `db.py`(只读)、`features.py` |
| `worldstate/` | `worldstate.py`、`zone.py` |
| `tree/` | `global_mode.py` |
| `game/` | `game_theory.py` |
| `labels/` | `fusion.py`、`strategy_label.py`(lookahead)、`rollout_label.py`(附录)、`opp_intent.py`(敌方意图概率)、`router.py`(Tier0/1/2) |
| `rl/` | `env.py`、`physics.py`、`ipoppo.py`、`reward.py`/`reward_learn.py` |
| `finetune/` | `make_dataset.py`、`lora_train.py`、`eval.py`、`retrain_eval.py`(Track1 附录) |
| `eval/` | `decision_gap.py`、`fusion_eval.py`、… |
| `schema/` | `layered.py` |

**产物**(gitignored): `out_m2/`、`_work/`

---

## 4. 环境与硬约束(必须遵守)

- **GPU**: RTX 5060 8G。**严禁两个大模型任务同时占卡**。
- **单元测试无 torch/transformers**（SIM 离线）。
- **`data.db` 只读**；产物不 commit；密钥不入库。
- **汇报**中文；每任务自验。

---

## 5. 已有结果(已核实, 诚实版)

| 里程碑 | 结果 |
|---|---|
| 微调 LLM 合法输出 | raw≈0 → finetuned legal≈1（考核交付质变） |
| 融合 confs 透传 | 修复后 RL 可进哨兵通道 |
| Track1 | `rollout_gap_ge_lookahead=false`（v4 更差）→ **结案** |
| Track2 v3fix2 | hardgate loaded=true；mode_diff≈258；**非**优于 rule 证明 |
| 消融 | delta/coh0.25 必要；fav0.02 弱有用；mode_gt 假通过 |

---

## 6. 当前检查点 & 启动方式 ★

### 交付主线（默认）

```bash
python run_m2.py --profile mainline --stage dataset   # skip_llm + lookahead_delta=5
python run_m2.py --profile mainline --stage train
python run_m2.py --profile mainline --stage eval
# 或一键重训脚本
python _work/retrain_mainline.py --device cuda
```

**已落盘交付物（2026-09-09）**：
- adapter：`out_m2/finetuned_mainline/adapter`（7B）
- 数据：`out_m2/m2_dataset_mainline.jsonl`
- 报告：`out_m2/mainline_eval_report.json` → **legal=1.0，deliverable_pass=true**；gap≈1.77（分析）；rule-agreement≈0.47

验收：合法分层决策为主；**不以** Track1 gap / Track2 hardgate 为交付门闩。

### 延时分层 & 敌方意图（Phase A，已落地）

| 模块 | 作用 |
|---|---|
| `labels/opp_intent.py` | Δs 似然 + 本局 top 抬升 → `opponent_intent.probs`；digest 行 `敌方意图(top=…)` |
| `labels/router.py` | 复杂度门控 → `decision_meta.{tier,need_llm,confidence}` |
| `finetune/make_dataset.situation_digest` | 注入意图行（训练输入条件） |
| `run_m2.stage_fusion` | 融合后挂旁路字段；统计 `n_need_llm` |

意图词表：`压前哨|拆基地|盯哨兵|换血前排|回防收缩|摸资源/中立`。大模型只读入分布，不在热路径反演敌方决策树。

### Phase B：Tier1 Mode 蒸馏（已落地）

教师 = 主线 lookahead 硬标签；学生 = 40 维 `M2Env.obs('哨兵')` → 3 类 `global_mode` MLP。  
说明：当前 router 下 `need_llm` 占比很高（~98%），**默认全量标签训练**；热路径挂载学生时覆盖全部帧 mode（7B 尚未进 fusion），`need_llm` 仍写入 meta 供后续 Tier2。

```bash
python _work/distill_mode_tier1.py --device cuda   # 建集+训练+报告
# 热路径挂载（简单帧学生出 mode，规则填单位）
python run_m2.py --profile mainline --stage fusion --skip-rl \
  --mode-student out_m2/mode_student --gid <留出局>
```

| 产物 | 路径 |
|---|---|
| 蒸馏集 | `out_m2/mode_distill.jsonl` |
| 学生权重 | `out_m2/mode_student/adapter.pt` |
| 报告 | `out_m2/mode_distill_report.json` |

模块：`rl/mode_features.py`、`rl/mode_student.py`、`finetune/make_mode_distill.py`。`need_llm=True` 仍留给 7B；本阶段不蒸馏文本 LLM / 六类 move。

**路由校准（全量对照后）**：主线数据 `base_gap` 中位≈0，旧阈值 `base_gap<0.12` 会把 ~98% 帧打进 Tier2。已改为 `base_gap_thr=0.0`（关闭纯 gap 门控），保留 critical / mode_jitter / multi_low_hp。校准后约 **84% Tier1 / 16% Tier2**，Tier1 上学生 mode 准确率 ≈0.75。脚本：`python _work/route_calibration.py --skip-llm`。

**稳定性门控（质量优先、轻量优先）**：
- 学生 `predict_proba` + **3 帧平滑**；置信 ≥0.55 → Tier1 学生；低置信 → **规则回退**（不盲目调 7B）
- 仅「危急且学生低置信」标 `need_llm`
- 扩大留出对照：`python _work/expand_leaveout_compare.py --eval-games 8 --limit 30`
- 教师是 lookahead 硬标签而非 API；**当前不需要 API 蒸馏 mode**（学生已常优于新 7B）
- **服务器无人值守**：`_work/auto_pipeline_shutdown.sh` 等评估结束 → 写 `out_m2/pipeline_status.json` → `shutdown -h now`。日志：`_work/auto_pipeline_shutdown.log`

### Runtime 真分层执行（无卡优先，2026-09-10）

热路径已从「只打 `need_llm` 标」升级为可执行器 [`runtime/decide.py`](runtime/decide.py) `TieredDecider`：

| 路径 | 行为 |
|---|---|
| Tier0 | 学生低置信 → 规则 mode + 单位 refill |
| Tier1 | 学生高置信 → 3 帧平滑 mode + refill |
| Tier2 | `need_llm` → `tier2_backend.generate_one`；**无 CUDA / `--tier2-adapter none` → NullTier2 规则回退** |

- Tier2 默认交付 adapter 路径：`out_m2/finetuned_lookahead_7b/adapter`（`runtime/tier2.DEFAULT_ADAPTER`）；本机无卡勿加载。
- 单点推理：`finetune/infer_one.py`；后端：`runtime/tier2.py`（`NullTier2` / `LazyLoraTier2`）。
- CLI：`run_m2.py --mode-student ... --tier2-adapter none|auto|<path> --tier2-device auto`
- 意图驱动单位偏置：`labels/intent_policy.py`（挂在 `refill_units_for_mode(..., opp_intent=)`）。
- 单位学生脚手架（未默认启用）：`rl/unit_features.py`、`rl/unit_student.py`、`finetune/make_unit_distill.py`；`TieredDecider(unit_student=...)` 钩子已留。
- 无卡验收：`python _work/bench_runtime_cpu.py --limit 20 --skip-tier2` → `out_m2/runtime_cpu_bench.json`

**本阶段不做**：本机加载 7B、API 蒸馏、时序学生实装、Physics 闭环大战（仅代码注释钩子）。

### 考核原图对齐（2026-09-10）

对照 `项目要求/` 两张 JPEG（WorldState 语义态势 / 状态机·行为树）：

| 已对齐 | 说明 |
|---|---|
| 哨兵三态 | `derive_sentry_state` 瞬时几何 → `def`/`hitbuilding`/`cruise`；`sentry_to_strategic` 映射图 2 雷达意图名 |
| 语义态势旁路 | `worldstate/semantic.py`：`base_threat`/`enemy_trend`/`reachable`/`lost_target`（`source=replay_approx`） |
| DecisionRequest | `runtime/decision_request.to_decision_request`；`TieredDecider(as_decision_request=True)` |
| Runtime meta | 每帧写入 `sentry_state` + `strategic_intent` + `semantic_situation` |

| 明确不做（与原图「不为 LLM 迁移执行框架」一致） | 说明 |
|---|---|
| 真雷达/激光/裁判实时融合 | 输入仍为官方 SQLite 回放建 WorldState |
| 实车行为树 T0–T3 主执行器 | 决策栈可插拔，不替代上车执行框架 |

### 研究附录

```bash
python run_m2.py --profile research --stage fusion
python run_track2.py --rl-updates 200 --hardgate --out out_m2/track2_rl_fusion_hardgate_v3fix2.json
```

Track1 / Track2 v3fix2 已结案（分析 §7.11–7.13、§9）。

---

## 7. 启动前的检查清单

1. `nvidia-smi` —— 无残留大模型进程。
2. `python -m pytest tests/test_m2.py tests/test_run_track2.py tests/test_make_dataset.py tests/test_run_m2_profile.py tests/test_opp_intent.py tests/test_router.py tests/test_mode_features.py tests/test_intent_policy.py tests/test_runtime_decide.py tests/test_unit_student_shapes.py tests/test_global_mode.py tests/test_semantic_situation.py -q`
3. 产物只写 `out_m2/` / `_work/`。

---

## 8. 解读口径 & 诚实上限

- `gap` / `fav_align` = 关联性，非因果；4 留出局只看方向与量级。
- Track2 `mode_from_units` 进攻偏向 = **防偏守启发式**，不得单独当「优于规则」。
- 符号物理 + 模式级敌方 ≠ 完整裁判。
- Plan B rollout 标签偏防守是方法产物；**不作**主线蒸馏默认源。

## 9. 因果、可观测、多时域预测评估（2026-09-25）

新增预测评估契约见 `docs/superpowers/specs/2026-09-25-causal-observable-forecasting-design.md`，执行记录见 `.superpowers/sdd/2026-09-25-causal-observable-forecasting/progress.md`。

- `data/features.py` 的回放对齐只使用当前时刻及之前的观测；未来状态不再回填。
- `data/observable.py` 定义车载传感器/雷达快照及可见性边界。当前 SQLite 回放仍是全局真值，不能冒充实车可观测输入。
- `data/forecast_targets.py` 按 10/30/60/120 秒精确时刻生成目标，比赛结束前没有目标的样本按删失处理；`data/forecast_splits.py` 按比赛顺序整体划分比赛组。
- `eval/forecast_schema.py` 与 `eval/prediction_ledger.py` 校验并保存概率预测；`eval/benchmark.py` 支持红蓝两侧，概率评分报告 Brier、log loss、CRPS/energy score、校准统计、侧别宏平均、样本池化微平均和整局聚类区间。
- `eval/forecast_report.py` 输出数据清单和 readiness 门禁。当前缺少经验证的车载传感器/可见性回放提供器，因此部署就绪状态必须为 false；基于 SQLite 全局回放得到的数值只能标为 oracle replay 结果。
- 当前验收：`python -m pytest tests -q` → **344 passed**。这证明实现与接口测试通过，不代表模型已经通过真实传感器输入的部署评估。
