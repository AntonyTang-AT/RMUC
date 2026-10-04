# RMUC 2026 项目要求符合性审计

审计日期：2026-09-29

审计源：`项目要求/项目要求.txt`、`项目要求/雷达站如何构建全局WorldState.JPEG`、`项目要求/状态机和行为树.JPEG`。

## 结论摘要

- 基础项目的工程链路（官方数据 -> WorldState -> 微调开源大模型候选 -> 哨兵三态 -> 可视化）已接通。
- 进阶项目的逐机器人结构化决策、状态机/行为树优先级和单一 `DecisionRequest` 执行器边界均已实现，并已接入运行时。
- 主线已做真实端到端验证：比赛 `1778661970735` 的 `t=52s` 触发 Tier2，使用 `LazyLoraTier2`，最终 `n_tier2_ok=1`、`mode_source=tier2_llm`。
- 不能声称“所有严格验收都已通过”：小样本四场景验收中，红方带预测的原始模型输出仍有 1/4 需要窄枚举修复，因此原始生成可靠性和结构可靠性门仍为 `fail`；策略质量与因果收益也未获推广资格。

## 逐项对照

| 要求 | 状态 | 可复核证据 | 审计判断 |
|---|---|---|---|
| 使用官方新赛季全场数据 | 通过 | `config.py` 指向 `数据与资料/rmuc_2026_region_dataset.sqlite`；数据文件存在 | 主线读取官方 SQLite，不使用空的 `data/rm.db` 或 `data.db`。 |
| 处理全场数据构建 WorldState 并做局势分析 | 通过 | `data/features.py`、`worldstate/worldstate.py`、`worldstate/semantic.py` | 静态地图、动态单位/建筑状态和语义态势（`base_threat`、`enemy_trend`、`reachable`、`lost_target`）均在运行时形成可审计字段。 |
| 部署微调开源大模型 | 通过（结构候选） | `out_m2/finetuned_visible_v5_risk_anchor_fast/adapter`；`meta.json` | Qwen2.5-3B-Instruct + LoRA，214 条结构补强记录、1 epoch、327 秒。该候选是正式 Tier2 默认路径，但不是策略效果已证明的候选。 |
| 输出哨兵 `def` / `hitbuilding` / `cruise` 三态宏观决策 | 通过 | `tree/sentry_state_machine.py`、`runtime/decide.py` | 显式 `StateEnum`、`EventEnum` 和转移表；运行时决策元数据含状态、原因和战略意图。 |
| 每个机器人输出严格结构化宏观决策 | 条件通过 | `schema/layered.py`、`finetune/infer_one.py`、`runtime/decision_guard.py` | 最终运行时输出固定含六个单位槽位和 `target/move/fire/risk`。四场景小样本运行时交付率为 16/16；原始模型输出为 15/16，红方带预测有 1/4 经 `risk: 无 -> 低` 窄修复，故原始严格可靠性尚未通过。 |
| 状态机优先、行为树 T0--T3、统一 DecisionRequest、单一执行器 | 通过（软件接口） | `tree/priority_behavior_tree.py`、`runtime/decision_request.py`、`runtime/single_executor.py` | T0 系统安全、T1 任务协调可覆盖 T2 战略意图，T3 只能细化；执行器注册被限制为唯一。未宣称已接入实体机器人控制器。 |
| 正确接入交付主线 | 通过 | `run_m2.py`、`runtime/tier2.py`、`runtime/decide.py` | `run_m2 --profile mainline` 默认 `--tier2-adapter auto`；无 CUDA 时安全回退。Tier2 启用时自动加载并按决策时刻对齐可见快照。`t=52s` 真正调用、可见状态投影后通过守卫。 |
| 用可视化呈现结果 | 通过 | `eval/build_battle_replay_visual.py`、回放 HTML | 回放共 419 帧、加载 9 条真实模型决策；其中包含 `t=52s` 已被主线接受的 Tier2 决策，并保留红蓝评估行。 |

## 运行与测试证据

- 完整测试：`python -m pytest tests -q` -> `643 passed, 2 warnings`。警告来自既有 PyTorch nested-tensor 配置，不是本次改动失败。
- 主线运行检查：`out_m2/mainline_v5_tier2_route_smoke/fused_decisions.json` 中 `t=52s` 的决策元数据记录 `tier2_attempted=true`、`tier2_ok=true`、`mode_source=tier2_llm`。
- 候选验收：`out_m2/llm_v5_risk_anchor_fast_smoke/acceptance.json`。它证明可交付与可见性覆盖，不证明策略价值。
- 可视化：`C:/Users/AntonyTang/.codex/visualizations/2026/09/28/01a0e628-f349-7361-8f55-0c9c8285748e/rm-battle-replay-v5-mainline-integrated.html`。

## 尚未通过的门槛

1. 原始模型生成的严格结构可靠性：红方带预测 4 条中有 1 条需要 `risk` 枚举修复；当前四场景小样本修复比例高于 5% 门槛。
2. 策略质量和因果有效性：HDQ/因果门未通过或不可验证；这不影响“已接入和可视化展示”的工程结论，但禁止把候选描述为优于规则或可直接上车的策略。
