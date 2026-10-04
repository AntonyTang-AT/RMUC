"""Train-only 10/30s fixed LightGBM ablation of visible combat-event history.

The baseline probabilities are the saved motion+zone/score whole-game OOF.
Candidate inputs append only the separately audited, issued-at visible-event
features.  No validation/test file, JSONL target, threshold search, or model
configuration search is used.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from eval.experiment_death_aware_mode_lightgbm import features as direct_features
from eval.experiment_death_aware_confident_switch_oof import game_cells, paired_summary
from forecast.temporal import load_compact

SPLIT = ROOT / "out_m2/temporal_forecast_death_aware_candidate/train.npz"
ZONE_CACHE = ROOT / "out_m2/death_aware_global_mode_experiments/nested_transition_expert_30s_5outer_pilot/causal_train_features.npz"
MOTION_CACHE = ROOT / "out_m2/death_aware_global_mode_experiments/visible_events_train_audit/motion_features.npz"
EVENT_CACHE = ROOT / "out_m2/death_aware_global_mode_experiments/visible_events_train_audit/visible_event_features.npz"
BASELINE = ROOT / "out_m2/death_aware_global_mode_experiments/motion_zone_score_oof/motion_candidate_oof.npz"
OUTPUT = ROOT / "out_m2/death_aware_global_mode_experiments/visible_events_oof"
DOC = ROOT / "docs/analysis/2026-09-27-death-aware-visible-events-oof.md"
HORIZONS = (10, 30)
SEED = 1927


def identical(a: np.ndarray, b: np.ndarray, name: str) -> None:
    if not np.array_equal(a, b):
        raise AssertionError(f"train cache alignment failure: {name}")


def load_features() -> tuple[dict[str, np.ndarray], np.ndarray, dict]:
    d = load_compact(SPLIT)
    with np.load(ZONE_CACHE, allow_pickle=False) as z, \
         np.load(MOTION_CACHE, allow_pickle=False) as motion, \
         np.load(EVENT_CACHE, allow_pickle=False) as event, \
         np.load(BASELINE, allow_pickle=False) as baseline:
        for name in ("game_ids", "sides", "issued_at", "mode_y", "base_mode"):
            identical(np.asarray(d[name]), np.asarray(baseline[name]), name + ": baseline")
        for name in ("game_ids", "sides", "issued_at"):
            identical(np.asarray(d[name]), np.asarray(event[name]), name + ": event")
        for name in ("game_ids", "sides", "issued_at", "mode_y", "base_mode"):
            identical(np.asarray(d[name]), np.asarray(z[name]), name + ": zone")
        base_x = np.concatenate((direct_features(d), z["extra_features"], motion["x"]), axis=1).astype(np.float32)
        event_x = np.asarray(event["x"], dtype=np.float32)
        x = np.concatenate((base_x, event_x), axis=1)
        reference = {name: np.asarray(baseline[name]) for name in
                     ("motion_candidate", "fold_ids", "mode_y", "base_mode", "game_ids", "sides", "issued_at")}
        info = {"base_feature_dim": base_x.shape[1], "event_feature_dim": event_x.shape[1],
                "candidate_feature_dim": x.shape[1], "event_feature_names": event["names"].tolist()}
    if not np.all(np.isfinite(x)):
        raise AssertionError("nonfinite causal feature")
    return d, x, {"reference": reference, "features": info}


def main() -> None:
    started = time.perf_counter()
    d, x, context = load_features()
    ref = context["reference"]
    gids = np.asarray(d["game_ids"], dtype=np.int64)
    sides = np.asarray(d["sides"]).astype(str)
    labels = np.asarray(d["mode_y"], dtype=np.int64)
    current = np.asarray(d["base_mode"], dtype=np.int64)
    predictions = np.full((len(gids), len(HORIZONS), 3), np.nan, np.float32)
    fold_ids = np.full(len(gids), -1, np.int8)
    folds = []
    for fold, (train_index, held_index) in enumerate(GroupKFold(n_splits=5).split(x, groups=gids)):
        fold_ids[held_index] = fold
        if not np.all(ref["fold_ids"][held_index] == fold):
            raise AssertionError(f"saved motion baseline has different outer fold at {fold}")
        row = {"fold": fold, "train_games": int(len(np.unique(gids[train_index]))),
               "held_games": int(len(np.unique(gids[held_index]))), "horizons": {}}
        for j, horizon in enumerate(HORIZONS):
            train = train_index[(labels[train_index, j] >= 0) & (current[train_index] >= 0) & (current[train_index] < 3)]
            held = held_index[(labels[held_index, j] >= 0) & (current[held_index] >= 0) & (current[held_index] < 3)]
            model = LGBMClassifier(objective="multiclass", num_class=3,
                n_estimators=260, learning_rate=.045, num_leaves=15,
                max_depth=-1, min_child_samples=90, reg_lambda=4.,
                colsample_bytree=.8, subsample=.85, subsample_freq=1,
                class_weight=None, verbosity=-1, n_jobs=4, random_state=SEED + fold)
            model.fit(x[train], labels[train, j])
            held_probability = model.predict_proba(x[held])
            for class_index, cls in enumerate(model.classes_):
                predictions[held, j, int(cls)] = held_probability[:, class_index]
            row["horizons"][str(horizon)] = {"train_rows": len(train), "held_rows": len(held)}
        folds.append(row)
        print(json.dumps({"fold": fold, "elapsed_seconds": round(time.perf_counter() - started, 1)}), flush=True)
    identical(fold_ids, ref["fold_ids"], "all saved baseline folds")
    scores = {}
    for j, horizon in enumerate(HORIZONS):
        valid = labels[:, j] >= 0
        if not np.all(np.isfinite(predictions[valid, j])):
            raise AssertionError(f"missing candidate OOF predictions: {horizon}")
        reference_probability = ref["motion_candidate"][:, j]
        candidate_cells = game_cells(d, valid, labels[:, j], np.argmax(predictions[:, j], axis=1))
        baseline_cells = game_cells(d, valid, labels[:, j], np.argmax(reference_probability, axis=1))
        persistence_cells = game_cells(d, valid, labels[:, j], current)
        scores[str(horizon)] = {"scorable_rows": int(valid.sum()),
            "candidate_minus_motion_baseline": paired_summary(candidate_cells, baseline_cells, SEED + horizon),
            "candidate_minus_persistence": paired_summary(candidate_cells, persistence_cells, SEED + horizon + 100)}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    report = {"schema": "death-aware-visible-events-10-30s-train-oof-v1",
              "opened_splits": ["train"], "validation_access": False, "test_access": False,
              "label_contract": str(d["global_mode_label_version"][0]),
              "train_rows": len(gids), "train_games": len(np.unique(gids)),
              "grouping": "five-fold GroupKFold by complete game; red/blue views shared same held fold",
              "comparator": "saved same-fold, same-LightGBM-configuration motion+zone/score OOF; candidate only appends causal visible-event features",
              "event_audit": str(OUTPUT.parent / "visible_events_train_audit/report.json"),
              "features": context["features"], "folds": folds, "horizons": scores,
              "runtime_seconds": time.perf_counter() - started}
    (OUTPUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(OUTPUT / "candidate_oof.npz", game_ids=gids, sides=sides,
        issued_at=np.asarray(d["issued_at"]), mode_y=labels, base_mode=current,
        fold_ids=fold_ids, candidate=predictions,
        motion_baseline=ref["motion_candidate"][:, :len(HORIZONS)])
    lines = ["# 可见战斗事件增强：train 整局 OOF", "",
             "仅训练集 363 场五折整局 OOF；红蓝视角同折，比赛等权且比赛内红蓝等权。候选沿用已保存 motion+zone/score 模型的分折、LightGBM 参数与随机种子，仅追加因果事件特征。没有访问 validation/test，也没有选择阈值或调参。",
             "", "| 时距 | 方法 | Accuracy | Macro-F1 | Balanced Accuracy | 相对持久性 Accuracy Δ [95% CI] |",
             "|---:|---|---:|---:|---:|---:|"]
    for h in HORIZONS:
        row = scores[str(h)]
        m = row["candidate_minus_motion_baseline"]
        p = row["candidate_minus_persistence"]
        for name, values, field in (("事件增强", p, "candidate"), ("原 motion", m, "baseline"),
                                    ("持久性", p, "baseline")):
            a = values["accuracy"][field]
            f = values["macro_f1"][field]
            b = values["balanced_accuracy"][field]
            delta = p["accuracy"] if name == "事件增强" else None
            diff = (f"{100*delta['delta']:+.2f}pp [{100*delta['delta_ci95'][0]:+.2f}, "
                    f"{100*delta['delta_ci95'][1]:+.2f}]") if delta else "—"
            lines.append(f"| {h}s | {name} | {a:.2%} | {f:.2%} | {b:.2%} | {diff} |")
        lines.append(f"事件增强减原 motion：{h}s Accuracy {100*m['accuracy']['delta']:+.2f}pp "
                     f"[{100*m['accuracy']['delta_ci95'][0]:+.2f}, {100*m['accuracy']['delta_ci95'][1]:+.2f}]；"
                     f"Macro-F1 {100*m['macro_f1']['delta']:+.2f}pp "
                     f"[{100*m['macro_f1']['delta_ci95'][0]:+.2f}, {100*m['macro_f1']['delta_ci95'][1]:+.2f}]；"
                     f"Balanced Acc {100*m['balanced_accuracy']['delta']:+.2f}pp "
                     f"[{100*m['balanced_accuracy']['delta_ci95'][0]:+.2f}, {100*m['balanced_accuracy']['delta_ci95'][1]:+.2f}]。")
    lines += ["", "事件特征仅由当前或更早的可见快照生成；己方发弹来自同 ID 累计发弹差分；敌方只使用可见 HP，不能推断敌方发弹。HP 缺失和雷达消失不视为死亡。完整覆盖率和观测时间检查见 `visible_events_train_audit/report.json`。",
              "", "该训练集此前经过多轮候选探索，OOF 为开发证据，不能当作新的独立盲测。",
              "", "复现：先运行 `python eval/audit_death_aware_visible_events_train.py`，再运行 `python eval/experiment_death_aware_visible_events_oof.py`；motion 缓存由原 `extract_motion(train.npz, train.death_aware.jsonl.gz)` 生成。"]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUTPUT / "report.json"), "document": str(DOC),
                      "runtime_seconds": report["runtime_seconds"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
