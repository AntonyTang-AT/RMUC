# M2 LLM 自动重训+评估 (Track 1) — 设计文档

**Goal:** 把"M2 LLM 策略"的**重平衡 → QLoRA 重训 → 留出评估**固化成一条单趟、可重放的流水线,并对 **rollout 标签 vs lookahead 标签**两种训练源各训一版做 head-to-head,回答"rollout 标签是否更优"(spec §9.4)。

**范围:** 只覆盖三源融合中的 **`llm` 源**(`finetune/lora_train.py`)。`rl` 源的重跑另立轨2(依赖 Plan B 对称环境 + PPO 修复),`rule` 源为确定性,不在本指令范围内。

**Tech Stack:** QLoRA 4bit NF4 Qwen2.5-3B(OOM→1.5B 兜底),torch/transformers/peft/datasets 仅在训练时惰性 import;评估侧 metric 纯 numpy(无 torch)。

**Spec:** 本文自身。承接 `docs/superpowers/specs/2026-09-09-rollout-labeling-design.md`(Plan A)与 `docs/superpowers/plans/2026-09-09-rollout-labeling.md`。

---

## 1. 为什么要自动重训+评估

Plan A 只把标签从"行为锚定"升格为"最优锚定",但没有验证**用这套新标签训出的 LLM 是否真的更好**。§9.4 要求的"重训回归"目前全靠手工:人手动 `rebalance → train → eval`,再拼报告。本轨把这条链路脚本化,并输出一个**同数据、两标签**的对照,让"标签方法 A/B 谁更优"有可重放的证据。

## 2. 一次运行做几件事(单趟)

```
输入: --method <lookahead|rollout>  (或顶层 --both 跑两种并合并)
 1. 读数据集 jsonl(records)
 2. rebalance_records(攻/均/守=45/30/25, 只删不增, 均衡作锚)
 3. train_lora(rebalanced_records, --out-dir, epochs=1, lr=2e-4, ...)
      → adapter_path + meta
 4. finetune.eval.eval_model(adapter_path, eval_dps, eval_bounds)
      → legal_rate / agreement_with_rule / conflict_rate
 5. global_mode alignment gap(新纯函数, 见 §4)
 6. 写 report json
```

`--both` = 依次跑 lookahead、rollout 两趟,合并成 `retrain_eval_report.json`(两行 + delta)。

## 3. 评估集与数据口径(必须公平可比)

- **训练源**(两行共用):同一批训练决策点,仅**标签方法**不同。
  - lookahead 行 ← `out_m2/m2_dataset.jsonl`(=重平衡前的回看标签;重平衡在 runner 内做)
  - rollout 行 ← `out_m2/m2_dataset_rollout.jsonl`(Plan A 产出)
  - 两者都经 `rebalance_records` 同一参数重平衡 → 只差标签。
- **评估集**(两行共用,保证头部可比):用 `m2_report.json` 的 `eval_games`(4 局),只读 `data.db` 经 `features.build_decision_points`/`field_bounds` 构建 `eval_dps`/`eval_bounds`。**同一批 dps 喂给两个模型 decode**,避免"各自挑了点"导致对照失衡。
- 训练/评估不重叠(训练取 dataset_games,评估取 eval_games)。

## 4. 评估 rig(§9.4 的承重判据)

共四个指标,三个报告性 + 一个门控:

1. **legal_rate** & **conflict_rate** —— `finetune.eval.eval_model`(进程内 decode 六个类意图,`sl.validate`)。
   - 门控:legal_rate 必须 ≥ `--legal-floor`(默认 0.8),防模型崩。
2. **global_mode alignment gap**(新增纯函数,无 torch)—— 承重的"outcome-align"信号。
   - 对 eval 集每个决策点,用模型 decode 出的 `global_mode` 记为 `m`;用 `outcome_backtest.fav_from_worldstate(w)` 算该点红方有利度 `fav`;`gap = mean_fav(进攻点) − mean_fav(防守点)`。
   - **期望:对齐好的模型 gap > 0(进攻落在高胜势);rollout 模型 gap ≥ lookahead 模型*(即"新标签训出的模型把进攻放对了").
   - 为什么用 `fav_from_worldstate` 而非 Plan A 单点胜率机:§5(rollout-labeling 设计)已警告胜率机用 T 窗特征训,喂单帧会塌到先验 `sigmoid(b)` 失去区分度;`fav_from_worldstate` 是逐状态标量(win 相关组件和),无分布错配、无窗、GPU-free。
3. **agreement_with_rule** —— 报告性(与规则头吻合;非门控,因为 rollout 标签本就更权威,一致性下降不视为退化,应同时看 gap 是否改善)。
4. **win-model CV AUC(旁证)** —— 可选:在 eval 集上报 `outcome_backtest.cv_auc_win_model` 的基线,证明"状态真的携带胜负信号",给 gap 一个可信地基。非门控。

## 5. 诚实上限(§9.4 的语言)

- 这是**关联性**度量:gap 为正只说明"模型把进攻放在高有利度区间",不证明"是这些决策赢的"。真正的因果需前向动力学(轨2/Plan B 后的 RL 环境)。
- 模型== 评估集仍是**留出**(4 局),样本少;数值小,结论以方向和量级为准,不当作精确统计检验。

## 6. 命名与产物(禁止覆盖旧名,产物不 commit)

- `out_m2/finetuned_lookahead/`、`out_m2/finetuned_rollout/`(adapter + meta;train_lora 自带 `meta.json`)
- `out_m2/retrain_eval_report.json`:`{method, train_stats, eval{legal_rate, conflict_rate, agreement_with_rule, global_mode_align_gap}, win_model_cv, rows:[...], delta:{...}}`
- 产物路径一律在 `out_m2/*`/`_work/*`(gitignored);git 只收入 `finetune/retrain_eval.py` + 新 eval helper + 测试。

## 7. GPU 预算

2 趟(rollout + lookahead),各 1 epoch,3B QLoRA(4 bit NF4)/1.5B 兜底,与既有 `m2_report.json` 训练一致。`train_lora` 已有 OOM→1.5B fallback。

## 8. File map

| File | Action |
|---|---|
| `finetune/retrain_eval.py` | Create(单趟 runner + `--both` 合并对照) |
| `eval/decision_gap.py` | Create(纯函数 `global_mode_align_gap(labels, worldstates, team)`,无 torch) |
| `tests/test_decision_gap.py` | Create(纯函数单测) |
| `finetune/lora_train.py` | 不改(现有 train_lora 复用) |
| `RMUC2026_M2_优化分析与方案.md` | Modify(记录此重训+评估结果) |

## 9. Out of scope

- 不动 `rl` 源/m2 融合;`rl` 重跑(轨2)在 Plan B 对称环境 + PPO 修复后进行。
- 不训"滚动标签评分函数"以外的任何新模型;不做多超参扫描(单趟,8G 预算)。
- 不改 `data.db`;不 commit `out_m*`/`_work` 产物。
