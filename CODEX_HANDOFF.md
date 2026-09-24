# RMUC 2026 M2 · Codex 完整交接文档

> **读者**：接手本仓库的 Codex / 其他 Agent / 人类协作者  
> **目的**：不依赖口头上下文，仅凭本文即可理解考核目标、架构、文件布局、运行命令、已有结论、诚实边界与推荐下一步  
> **仓库路径**：`F:\Desktop\学校\rm考核项目`  
> **文档日期**：2026-09-24  
> **配套短导读**：`AGENT_PROJECT_OVERVIEW.md`（给 Agent 的压缩版；本文是展开版）  
> **分析长文**：`RMUC2026_M2_优化分析与方案.md`（实验过程与负面结果叙事）

---

## 目录

1. [项目是什么（考核原文 + 原图）](#1-项目是什么考核原文--原图)
2. [一句话当前定位](#2-一句话当前定位)
3. [总体架构与数据流](#3-总体架构与数据流)
4. [核心思路（为什么这样设计）](#4-核心思路为什么这样设计)
5. [里程碑与当前进度总表](#5-里程碑与当前进度总表)
6. [仓库文件清单（按目录）](#6-仓库文件清单按目录)
7. [关键模块 API / 行为说明](#7-关键模块-api--行为说明)
8. [可输出内容规格](#8-可输出内容规格)
9. [环境、数据与硬约束](#9-环境数据与硬约束)
10. [使用方法（命令大全）](#10-使用方法命令大全)
11. [已落盘产物与关键数字](#11-已落盘产物与关键数字)
12. [AutoDL / 服务器相关](#12-autodl--服务器相关)
13. [测试与验收](#13-测试与验收)
14. [诚实边界与禁止事项](#14-诚实边界与禁止事项)
15. [推荐下一步（给 Codex）](#15-推荐下一步给-codex)
16. [附录：常见坑](#16-附录常见坑)

---

## 1. 项目是什么（考核原文 + 原图）

### 1.1 考核文字（`项目要求/项目要求.txt`）

**基础任务**
- 部署 / 微调开源大模型，做局势分析（可抄 TDT / 东北大学青工会分享思路）
- 数据：官方新赛季全场数据 `rmuc_2026_region_dataset.sqlite`
- 输入：处理后的全场数据构建 **WorldState**
- 输出：哨兵宏观决策三态 **`def` / `hitbuilding` / `cruise`**

**进阶任务（promax）**
- 输入：针对新规则调整后的数据（考核方未给出具体调法）
- 输出：每个机器人更详细宏观决策，**严格结构化**

**参考**
- https://b23.tv/2dXWsc4
- https://www.bilibili.com/video/BV1i88A65E7r

### 1.2 两张原图（同目录 JPEG，文件名可能乱码）

**图 A · 雷达站如何构建全局 WorldState**
- 传感：相机 / 激光雷达 / 无线电·裁判
- 四层：静态地图 → 动态目标 → 战局状态 → **语义态势**
- 语义示例字段：`base_threat`、`enemy_trend`、`reachable`、`lost_target`
- 本仓库实现方式：**SQLite 回放近似**（`source=replay_approx`），**不是**真雷达栈

**图 B · 状态机 vs 行为树**
- 标题精神：**不为接入 LLM 而整体迁移执行框架**
- 状态机首期：`StrategicDefendBase→def`，`StrategicAttackOutpost→hitbuilding`，`StrategicPatrol→cruise`
- 行为树长期：T0 安全 > T1 生存 > T2 战略（雷达意图）> T3 智能战斗
- 统一接口名：`DecisionRequest`；实车只允许一个主执行器
- 本仓库：**做决策栈 + 薄 DecisionRequest 适配**；**不做**实车 BT 主执行器

### 1.3 我们如何映射考核

| 考核项 | 本仓库对应 |
|--------|------------|
| WorldState 输入 | `worldstate/worldstate.py` + `data/features.py` 决策点 |
| 基础三态 | `tree/global_mode.derive_sentry_state` + meta / DecisionRequest |
| 进阶结构化 | `schema/layered.py`：`global_mode` + 六单位四通道 |
| 微调 LLM | QLoRA（主线曾用 7B；本机 5060 8G 默认倾向 3B/4bit） |
| TDT 语义态势 | `worldstate/semantic.py` 旁路近似 |

---

## 2. 一句话当前定位

**交付主线已过门**：规则保底 + lookahead(Δ=5) 标签 → QLoRA → 合法分层 JSON（`deliverable_pass=true`，legal≈1.0）。

**增强线已落地**：延时分层路由 + 敌方意图 + Tier1 mode 学生 + 置信门控/平滑 + **真分层 Runtime**（无卡 Tier2=Null 回退）+ 考核原图口径对齐。

**研究附录已结案**：Track1 rollout ≱ lookahead（负结果）；Track2 Physics hardgate 通过但 ≠「RL 优于规则」。

**定版建议（有证据）**：
- 交付 / Tier2 adapter 优先 **`finetuned_lookahead_7b`**（旧），不是新 `finetuned_mainline`（mode 留出更弱）
- **不做** API 蒸馏 mode（教师是 lookahead，学生已常优于新 7B mode）
- 热路径：**学生高置信 → 平滑 mode；低置信 → 规则；仅危急且弱 → need_llm**

---

## 3. 总体架构与数据流

```text
SQLite(timeseries)
    │
    ▼
features.build_decision_points  ──► DecisionPoint 序列
    │
    ▼
worldstate.build_worldstate     ──► robots/buildings/team
    │
    ├─► semantic.derive_semantic_situation   (旁路)
    ├─► opp_intent.infer_opponent_intent     (旁路)
    ├─► router.route_tier                    (tier / need_llm)
    └─► derive_sentry_state                  (def|hitbuilding|cruise)
    │
    ▼
runtime.TieredDecider / run_m2.stage_fusion
    │
    ├─ Tier0: 规则 global_mode + refill_units(+intent_policy)
    ├─ Tier1: ModeStudent + ModeSmoother + refill
    └─ Tier2: LazyLoraTier2.generate_one 或 Null→规则回退
    │
    ▼
分层决策 JSON + decision_meta + opponent_intent + semantic_situation
    │
    ├─► make_dataset → instruction/completion → QLoRA
    ├─► DecisionRequest 适配（可选）
    └─► eval / leaveout / bench 报告
```

### 3.1 Profile

| Profile | 行为 |
|---------|------|
| `mainline`（默认） | 强制 `skip_llm`（融合第三源 DeepSeek 慢环关闭）；`lookahead_delta` 默认注入 5 |
| `research` | 不强制改 flags，用于旧实验复现 |

实现：`run_m2.apply_profile`。

### 3.2 延时分层语义

| Tier | 含义 | 典型执行 |
|------|------|----------|
| Tier0 | 规则快路径 | `derive_global_mode` + 单位规则/意图偏置 |
| Tier1 | 轻学生 | ModeStudent（CPU 可跑） |
| Tier2 | 大模型 | 有 CUDA 时 LoRA；无卡 `NullTier2` 规则回退 |

路由阈值见 `labels/router.DEFAULT_THRESHOLDS`：`base_gap_thr=0.0`（关闭纯 gap 门控），`student_conf_thr=0.55`。

---

## 4. 核心思路（为什么这样设计）

### 4.1 标签先于模型

主线教师不是「更大的 API 模型」，而是：
1. 规则头产出合法分层决策
2. `strategy_label.relabel_decision_series` 用未来 Δ=5 帧重标 `global_mode`，再按 mode 重填单位

这样蒸馏目标稳定、可复现、不依赖外网 API。

### 4.2 重标注器慢、热路径轻

- 训练 / 离线：lookahead、可选 7B 批评估可以慢
- 在线：多数帧 ms 级（规则 / MLP）；稀有危急帧才考虑秒级 7B

叙事：**重标注器（可慢）→ 轻学生（追延时）**。

### 4.3 质量优先、轻量优先的门控

- 学生 softmax 置信 ≥0.55 → 用学生（再 3 帧平滑）
- 低置信且非危急 → **规则回退**（不盲目调 7B）
- 危急且低置信 → `need_llm`

校准后约 **84% Tier1 / 16% Tier2**（相对旧阈值 ~98% need_llm）。

### 4.4 主线剔除的东西

- 默认 DeepSeek 三源慢环（融合里 LLM 头常为 stub）
- 把 Track1 gap / Track2 hardgate 当交付门闩
- 无上限配平、fav≥0.08 等曾导致假通过的设定
- API 蒸馏 mode（当前证据不支持）

### 4.5 原图对齐哲学

- **对齐**：三态、战略意图名、语义态势字段、DecisionRequest 名
- **不对齐也不假装**：真雷达融合、实车 BT T0–T3

---

## 5. 里程碑与当前进度总表

| 阶段 | 状态 | 说明 |
|------|------|------|
| M0 提示/可视化基线 | ✅ | `run_m0.py`，`out_m0/` |
| M1 哨兵三态 + 规则基线 | ✅ | `run_m1.py`；现已修 hitbuilding |
| M3 分层 schema + 博弈/先验 | ✅ | `schema/layered.py`，`run_m3.py` |
| M2 融合 + QLoRA 主线 | ✅ | `run_m2.py --profile mainline`；legal=1.0 |
| Track1 rollout vs lookahead | ✅ 结案负 | `rollout_gap_ge_lookahead=false` |
| Track2 Physics IPPO hardgate | ✅ 附录通过 | ≠ 优于规则 |
| Phase A 意图 + 路由 | ✅ | `opp_intent` / `router` |
| Phase B mode 蒸馏 | ✅ | `out_m2/mode_student`，~0.17ms |
| 路由校准 + 稳定性门控 | ✅ | conf + 3 帧平滑 |
| 扩大留出 8 局对照 | ✅ | 建议旧 7B；勿 API 蒸馏 |
| Runtime 真分层执行 | ✅ | `runtime/decide.py`；无卡可跑 |
| 意图驱动单位偏置 | ✅ | `intent_policy.py` |
| 单位学生脚手架 | ✅ 未训默认关 | `unit_student` 钩子已留 |
| 考核原图对齐 | ✅ | 三态 + semantic + DecisionRequest |
| 本机挂真 7B Tier2 | ⏸ 未做 | 用户决定暂不下沉本机 |
| 单位学生全量训练 | ⏸ 未做 | 有卡/需要时再训 |
| 时序学生 / Physics 闭环大战 | ⏸ 仅钩子 | P3/P4 |

**考核完成度判断**：基础 + 进阶结构化 **可视为完成**；增强线超额；执行框架层按原图精神不做。

---

## 6. 仓库文件清单（按目录）

### 6.1 顶层入口

| 文件 | 作用 |
|------|------|
| `run_m0.py` | 早期可视化 / 报告 |
| `run_m1.py` | 哨兵三态；`--rule-baseline` 用 `derive_sentry_state` |
| `run_m3.py` | 规则分层决策序列 |
| `run_llm_m3.py` / `run_llm_api.py` | LLM 慢环 / API 实验（非 mainline 默认） |
| `run_m2.py` | **交付主线编排**：fusion→dataset→train→eval |
| `run_track2.py` | Track2 Physics+融合 hardgate（附录） |
| `config.py` | `DB_PATH`、`GAME_REF`、`DF_STEP=3`、`TREND=15` |
| `blackboard.py` | 黑板结构 |
| `requirements.txt` | 极简（pytest/matplotlib）；训练依赖另装 |
| `AGENT_PROJECT_OVERVIEW.md` | Agent 短导读 |
| `RMUC2026_M2_优化分析与方案.md` | 长分析 |
| `CODEX_HANDOFF.md` | **本文** |

### 6.2 `data/`

| 文件 | 作用 |
|------|------|
| `db.py` | 只读 SQLite |
| `features.py` | 决策点、场地 bounds；建筑类型常量 |

### 6.3 `worldstate/`

| 文件 | 作用 |
|------|------|
| `worldstate.py` | `build_worldstate(dp, bounds)` |
| `zone.py` | 区划 / depth |
| `semantic.py` | **语义态势旁路**（原图对齐） |

### 6.4 `schema/`

| 文件 | 作用 |
|------|------|
| `layered.py` | 分层决策契约、`validate`、`conflict_rate`、`UNIT_CLASSES` 等 |

### 6.5 `tree/`

| 文件 | 作用 |
|------|------|
| `global_mode.py` | `derive_global_mode`、`derive_unit_intent`、**三态** `derive_sentry_state`、`sentry_to_strategic` |
| `behavior_prior.py` | 行为先验 |

### 6.6 `game/`

| 文件 | 作用 |
|------|------|
| `game_theory.py` | 3×3 payoff / best response（非主线默认结论源） |

### 6.7 `labels/`

| 文件 | 作用 |
|------|------|
| `fusion.py` | 多源融合（规则/RL/LLM 通道） |
| `strategy_label.py` | **lookahead 重标**（主线教师） |
| `rollout_label.py` | Plan B / Track1 标签（附录） |
| `sentry_labels.py` | 基于原始 dp 的三态标签（outcome_backtest 默认） |
| `opp_intent.py` | 敌方意图概率 |
| `router.py` | Tier 路由 + `attach_decision_meta` / `strip_sidecar` |
| `intent_policy.py` | **意图→单位 move/target 偏置** |
| `transfer.py` | 迁移相关辅助 |

### 6.8 `rl/`

| 文件 | 作用 |
|------|------|
| `env.py` | `M2Env` / `PhysicsEnv` |
| `physics.py` | 符号物理前向 |
| `train.py` | IPPO |
| `reward.py` / `reward_learn.py` | 奖励组件 / 可学权重 |
| `distill_labels.py` | RL rollout → 分层决策 |
| `forward.py` | 前向辅助 |
| `mode_features.py` | 40 维 obs、`refill_units_for_mode` |
| `mode_student.py` | Tier1 mode MLP + `ModeSmoother` |
| `unit_features.py` / `unit_student.py` | 单位学生脚手架（默认关） |

### 6.9 `finetune/`

| 文件 | 作用 |
|------|------|
| `make_dataset.py` | instruction/completion；digest 含意图+语义行 |
| `lora_train.py` | QLoRA 训练；`LORA_BASE` 自动选 7B/3B |
| `eval.py` | 留出 legal/agreement；复用 `infer_one` |
| `infer_one.py` | **单点 LoRA 生成**（Tier2） |
| `make_mode_distill.py` | mode 蒸馏集 |
| `make_unit_distill.py` | 单位蒸馏集脚手架 |
| `make_rollout_dataset.py` | rollout 数据集 |
| `retrain_eval.py` | Track1 重训评估 |

### 6.10 `runtime/`（系统升级核心）

| 文件 | 作用 |
|------|------|
| `decide.py` | **`TieredDecider`** 真分层执行 |
| `tier2.py` | `NullTier2` / `LazyLoraTier2` / `resolve_tier2_backend` |
| `decision_request.py` | 薄 `DecisionRequest` 适配 |

### 6.11 `eval/`

| 文件 | 作用 |
|------|------|
| `metrics.py` | per_class 等 |
| `decision_gap.py` | mode-fav gap |
| `fusion_eval.py` | 融合评估 |
| `outcome_backtest.py` | 三态有利度回测 |
| `win_proba.py` / `llm_backtest.py` / `forward_eval.py` / `strategy_lag.py` / `viz.py` | 各类分析 |

### 6.12 `model/` / `fastloop/`

早期 M0/M1 提示与快环行为（非 M2 主线核心）。

### 6.13 `tests/`

大量无 torch 单测。与新系统强相关：
- `test_global_mode.py`、`test_semantic_situation.py`
- `test_router.py`、`test_opp_intent.py`、`test_intent_policy.py`
- `test_runtime_decide.py`、`test_mode_features.py`、`test_mode_student_shapes.py`
- `test_unit_student_shapes.py`、`test_run_m2_profile.py`、`test_m2.py` …

### 6.14 `_work/`（脚本与临时；多数 gitignore）

重要脚本（非完整列表）：
- `retrain_mainline.py` — 一键主线重训
- `distill_mode_tier1.py` — mode 蒸馏流水线
- `route_calibration.py` — 路由校准
- `expand_leaveout_compare.py` — 8 局扩大留出
- `bench_runtime_cpu.py` — 无卡 Runtime 验收
- `auto_pipeline_shutdown.sh` — 服务器等评估完关机
- `check_server_status.py` / `force_start_pipeline.py` / `start_auto_pipeline_remote.py` — AutoDL 运维

### 6.15 `out_m2/`（产物；通常不提交）

见第 11 节。本地可能有多份 adapter 目录（`finetuned_*`）；**服务器上**还有完整 7B 权重与 `finetuned_lookahead_7b`。

### 6.16 `项目要求/` / `数据与资料/`

- 考核 txt + 两张 JPEG
- 数据集说明 md；sqlite 本体路径见 `config.DB_PATH`（大文件，只读）

### 6.17 `docs/superpowers/`

历史 design / plan（M0–M2、rollout、Track2、Plan B）。**以 AGENT + 本文 + 分析文档为准**；旧 plan 可能过时。

---

## 7. 关键模块 API / 行为说明

### 7.1 `schema.layered`

- `GLOBAL_MODES = (进攻, 均衡, 防守)`
- `UNIT_CLASSES = (英雄, 工程, 步兵3, 步兵4, 空中, 哨兵)`
- `TARGETS` / `MOVES` / `RISKS` 枚举
- `validate(decision) -> (ok, errs)`
- 旁路字段（`decision_meta` 等）validate 前应 `strip_sidecar_for_validate`

### 7.2 `derive_sentry_state`（对齐后）

瞬时几何，优先级：
1. 敌方单位进入己半场 → `def`
2. 己方哨兵敌半场且 `dfire>0` → `hitbuilding`
3. 否则 → `cruise`

映射：
- `def` → `StrategicDefendBase`
- `hitbuilding` → `StrategicAttackOutpost`
- `cruise` → `StrategicPatrol`

### 7.3 `TieredDecider.decide`

每帧：
1. build worldstate；意图；可选学生 `predict_proba`
2. `route_tier`
3. need_llm → tier2，失败规则 refill
4. 否则学生高置信平滑 / 低置信规则
5. `refill_units_for_mode(..., opp_intent=)`（含 intent_policy）
6. 挂 semantic + sentry_state + strategic_intent + meta
7. 可选 `as_decision_request=True`

### 7.4 Tier2 无卡行为

`resolve_tier2_backend(adapter='none'|'auto'|path, device='auto')`：
- 无 CUDA 或 `none` → `NullTier2`（不加载权重）
- `auto` 且有 CUDA → `LazyLoraTier2(DEFAULT_ADAPTER=out_m2/finetuned_lookahead_7b/adapter)`

### 7.5 主线数据集

`make_records`：
- instruction = `situation_digest`（可含敌方意图行 + 语义行）
- completion 默认 **剥旁路**，只留主 schema JSON

---

## 8. 可输出内容规格

### 8.1 主交付（每决策点）

```json
{
  "t": 123.0,
  "global_mode": "进攻",
  "units": {
    "英雄": {"target": "...", "move": "...", "fire": true, "risk": "中"},
    "工程": {"...": "..."},
    "步兵3": {"...": "..."},
    "步兵4": {"...": "..."},
    "空中": {"...": "..."},
    "哨兵": {"...": "..."}
  }
}
```

### 8.2 旁路（Runtime 默认挂）

- `decision_meta`：`tier`、`need_llm`、`confidence`、`reasons`、`mode_source`、`student_conf`、`tier2_attempted`、`tier2_ok`、`sentry_state`、`strategic_intent`
- `opponent_intent`：`probs`、`top`、`source`
- `semantic_situation`：`base_threat`、`enemy_trend`、`reachable`、`lost_target`、`source=replay_approx`

### 8.3 DecisionRequest

`runtime.decision_request.to_decision_request(decision)` →  
`{interface, sentry_state, strategic_intent, global_mode, units, t, ...}`

---

## 9. 环境、数据与硬约束

### 9.1 硬件

- 本机历史设定：Windows 11，**RTX 5060 8GB**
- 7B：**4bit 推理**可试；**7B 全量微调**很勉强 → 训练常用 AutoDL
- **严禁两个大模型任务同时占卡**

### 9.2 数据

- `config.DB_PATH` → `数据与资料/rmuc_2026_region_dataset.sqlite`
- **只读**；不要写入 DB

### 9.3 依赖约束

- 单元测试路径：**不得**依赖 torch/transformers（学生形状测可 `importorskip`）
- 训练：`transformers` / `peft` / `bitsandbytes` / `accelerate` / `datasets` 等需环境自备
- 密钥不入库；`out_m2/` / `_work/` 大产物通常 gitignore

### 9.4 语言

- 对用户汇报：**简体中文**

---

## 10. 使用方法（命令大全）

以下均在仓库根目录执行。

### 10.1 测试（先跑）

```bash
python -m pytest tests/test_global_mode.py tests/test_semantic_situation.py tests/test_runtime_decide.py tests/test_intent_policy.py tests/test_router.py tests/test_opp_intent.py tests/test_mode_features.py tests/test_run_m2_profile.py -q
```

更全：

```bash
python -m pytest tests/test_m2.py tests/test_run_track2.py tests/test_make_dataset.py tests/test_run_m2_profile.py tests/test_opp_intent.py tests/test_router.py tests/test_mode_features.py tests/test_intent_policy.py tests/test_runtime_decide.py tests/test_unit_student_shapes.py tests/test_global_mode.py tests/test_semantic_situation.py -q
```

### 10.2 交付主线

```bash
python run_m2.py --profile mainline --stage dataset
python run_m2.py --profile mainline --stage train
python run_m2.py --profile mainline --stage eval
# 或
python _work/retrain_mainline.py --device cuda
```

常用参数：
- `--skip-rl`：跳过 RL 头
- `--mode-student out_m2/mode_student`
- `--mode-student-device cpu`
- `--tier2-adapter none|auto|<path>`（默认 `none`）
- `--tier2-device auto`
- `--limit N`：每局决策点数上限
- `--out out_m2`

### 10.3 无卡 Runtime 验收

```bash
python _work/bench_runtime_cpu.py --limit 20 --skip-tier2
# → out_m2/runtime_cpu_bench.json
```

挂学生（若有权重）：

```bash
python _work/bench_runtime_cpu.py --limit 20 --mode-student out_m2/mode_student
```

### 10.4 Mode 蒸馏

```bash
python _work/distill_mode_tier1.py --device cuda
```

### 10.5 路由校准 / 扩大留出

```bash
python _work/route_calibration.py --skip-llm
python _work/expand_leaveout_compare.py --eval-games 8 --limit 30 --device cuda
# 无卡可加 --skip-llm 只跑快路径（脚本支持）
```

### 10.6 Track2 附录

```bash
python run_track2.py --rl-updates 200 --hardgate --out out_m2/track2_rl_fusion_hardgate_v3fix2.json
```

### 10.7 有卡打开 Tier2（示例）

```bash
python run_m2.py --profile mainline --stage fusion --skip-rl \
  --mode-student out_m2/mode_student \
  --tier2-adapter out_m2/finetuned_lookahead_7b/adapter \
  --tier2-device cuda \
  --limit 20
```

### 10.8 原图对齐后是否需要重训？

**一般不需要。**  
三态/语义/DecisionRequest 主要改规则与旁路；completion 默认仍无旁路。  
仅当希望 7B **显式吃 digest 新语义行** 时，才重跑 `dataset → train`。

---

## 11. 已落盘产物与关键数字

### 11.1 主线交付

| 产物 | 路径 | 要点 |
|------|------|------|
| 评估报告 | `out_m2/mainline_eval_report.json` | legal=1.0，deliverable_pass=true，gap≈1.77，rule-agreement≈0.47 |
| 数据集 | `out_m2/m2_dataset_mainline.jsonl`（若存在） | lookahead 主线 |
| adapter | `out_m2/finetuned_mainline/adapter`（服务器完整；本地未必齐） | 新 mainline；**mode 留出弱于旧 7B** |

### 11.2 留出对照（定版建议来源）

`out_m2/expand_leaveout_compare.json` / `out_m2/pipeline_status.json`：

| 路径 | mode≈教师准确率 |
|------|----------------|
| 规则 | 0.308 |
| 学生平滑 | 0.488 |
| 置信门控 | 0.438 |
| 新 mainline 7B | 0.505（legal≈0.94） |
| **旧 lookahead 7B** | **0.614（legal≈0.996）** |

`need_llm_rate`≈0.0375；建议 `prefer_adapter=old_lookahead7b`，`use_api_distill=false`。

### 11.3 Mode 蒸馏

`out_m2/mode_distill_report.json`：
- 训练 n=3317；eval acc≈0.71；latency_ms_median≈0.17
- `out_m2/mode_student/adapter.pt`

### 11.4 Runtime CPU bench

`out_m2/runtime_cpu_bench.json`：legal=1.0，~0.2ms/点（无学生时全 tier0_rule）。

### 11.5 其他重要报告

- `out_m2/route_calibration_report.json`
- `out_m2/retrain_eval_report*.json`（Track1）
- `out_m2/track2_rl_fusion_hardgate_v3fix2.json`
- `out_m2/ablate_*.json` / `ablate_summary.json`

### 11.6 Adapter 目录名对照（易混）

| 目录名 | 含义 | 建议 |
|--------|------|------|
| `finetuned_lookahead_7b` | 旧 lookahead 7B（服务器；对照最优） | **Tier2/交付优先** |
| `finetuned_mainline` | 新主线重训 7B | 实验；mode 弱于旧 |
| `finetuned_lookahead` / `_v2` | 更早 3B/实验 | 历史 |
| `finetuned_rollout*` | rollout 教师线 | Track1 负结果，非主线 |
| `mode_student` | Tier1 MLP | 热路径轻量 |

---

## 12. AutoDL / 服务器相关

历史实例（可能已变）：`connect.westd.seetacloud.com`，项目常在 `/root/autodl-tmp/rmuc`。

无人值守：
- `_work/auto_pipeline_shutdown.sh`：等 `expand_leaveout_compare` → 写 `pipeline_status.json` → `shutdown -h now`
- 日志：`_work/auto_pipeline_shutdown.log`

运维脚本：`check_server_status.py`、`force_start_pipeline.py` 等（需 `SEETA_PASS` 环境变量；**勿把密码写进仓库**）。

本机用户曾决定：**暂不把 7B 下沉到 5060**；无卡开发以 Runtime NullTier2 + 学生/规则为主。

---

## 13. 测试与验收

### 13.1 交付门闩

- **主门闩**：微调模型 `legal_rate`（主线报告 legal=1.0）
- **非门闩**：Track1 gap、Track2 hardgate、mode 对齐教师准确率（分析用）

### 13.2 原图对齐验收点

- `derive_sentry_state` 能产出 hitbuilding（越线+开火构造）
- meta 含 `sentry_state` + `strategic_intent`
- `semantic_situation.source == replay_approx`
- `to_decision_request` 字段齐全

### 13.3 Runtime 验收

- 无卡：`bench_runtime_cpu.py` legal=1.0，不加载 7B
- need_llm 帧：`mode_source=tier2_fallback_rule`（NullTier2）

---

## 14. 诚实边界与禁止事项

1. **gap / fav_align = 关联非因果**  
2. Track2 `mode_from_units` 进攻偏向 = 防偏守启发式，**不是**「RL 赢了规则」  
3. 符号物理 ≠ 完整裁判  
4. Plan B rollout 标签偏防守 → **不作主线默认教师**  
5. 语义态势是 **replay_approx**，不要写成雷达真融合  
6. 对齐原图提升的是 **规格完整度**，不自动提升竞技战力  
7. 不要同时占卡跑两个大模型  
8. 不要把密钥、sqlite、大 adapter 权重 commit 进 git（除非团队明确要求）  
9. 不要编辑用户未要求的 plan 文件；用户规则：少写无关 md——但本文是用户明确要求的交接文档  

---

## 15. 推荐下一步（给 Codex）

按优先级（默认无卡 / 少算力）：

### P0 定版收口（半小时级）
1. 文档与 CLI 默认明确：`prefer_adapter=finetuned_lookahead_7b`（或软链/复制说明）
2. 确认 `run_m2` 默认 `--tier2-adapter none`（已是）在无卡环境

### P1 演示包（答辩用）
1. 导出 1 局 20 帧：layered + meta + intent + semantic + DecisionRequest  
2. 一页数字：legal、留出对照表、延迟对比  

### P2 有卡时
1. 打开 Tier2 LazyLora 做稀有帧真推理 bench  
2. 可选：单位学生小训并挂 `unit_student`  

### P3 明确不做除非用户要求
- API 蒸馏 mode  
- 再全量重训新 mainline 指望碾压  
- 实车 BT T0–T3  
- 为「对齐」强制重训 7B（非必须）

---

## 16. 附录：常见坑

1. **PowerShell 展开 `$(...)`**：远程写脚本时慎用；用本地文件 SFTP 上传更稳  
2. **`pgrep -f` 误匹配 SSH 命令行**：用 pidfile  
3. **eval 必须用训练同款 chat 模板**（`eval_prompt`），否则 legal 崩  
4. **旧 `derive_sentry_state` 丢 hitbuilding**：已修；若看到旧注释以代码为准  
5. **router `base_gap_thr=0.12` 会让 ~98% need_llm**：已改为 0.0  
6. **融合 confs 透传**：历史 bug 已修；读分析文档了解背景  
7. **本地 adapter 可能只有 tokenizer 无权重**：完整权重常在 AutoDL  
8. **中文路径**：Windows 下注意编码；Python `pathlib` 优于乱码控制台  

---

## 快速对照卡（贴在对话开头给 Codex）

```text
项目：RMUC 2026 M2 局势分析 LLM
主线：lookahead 标签 → QLoRA → 合法分层 JSON（已过门）
热路径：TieredDecider（规则/学生/可选7B）；无卡 Tier2=Null
定版 adapter：旧 finetuned_lookahead_7b > 新 finetuned_mainline
不做：API蒸馏mode、实车BT、假装雷达融合
考核：基础三态 + 进阶六单位结构化
入口：run_m2.py --profile mainline
Runtime：runtime/decide.py
导读：AGENT_PROJECT_OVERVIEW.md
交接：CODEX_HANDOFF.md（本文）
```

---

*文档结束。若与代码冲突，以仓库当前源码与 `out_m2/*_report.json` 为准，并回写本文。*

## 预测评估实现更新（2026-09-25）

本项目新增一条因果、可观测的多时域概率预测评估链路，设计规格见 `docs/superpowers/specs/2026-09-25-causal-observable-forecasting-design.md`，工作决定和复核记录见 `.superpowers/sdd/2026-09-25-causal-observable-forecasting/progress.md`。

- 输入与标签分离：回放特征只按 `observed_at <= issued_at` 构造；部署观测通过 `data/observable.py` 的显式传感器/雷达快照接口提供。`timeseries` SQLite 仍是全局回放数据，不是实车可见性记录。
- 目标时间固定为发布时刻后 10/30/60/120 秒，按精确时间查找；比赛结束或对应真值缺失时记为删失，不找最近帧替代。比赛组按比赛时间顺序整体进入 train/validation/test。
- 预测先写不可变 JSONL ledger，经 schema 校验后，在目标时刻成熟时评分。分类使用 Brier、log loss、校准统计；连续概率分布支持 CRPS 和 energy score。报告同时列出红方、蓝方、等权侧别宏平均和按样本量池化微平均，并以整局聚类重采样估计差值区间。
- 决策基准 CLI 的旧 JSONL 输入契约是 `game_id,side,index,decision`；概率预测通过 `eval.prediction_ledger` 和 `evaluate_forecast_ledger`，不要把两类接口混用。
- 部署门禁由 `eval/forecast_report.py` 汇总。当前可见性提供器缺失，故 deployment readiness 为 false；SQLite 全局回放结果只能称为 oracle replay，不能作为车载传感器效果结论。
- 最近全量测试：`python -m pytest tests -q` → **344 passed**，包含整数机器人 ID 回归测试。
