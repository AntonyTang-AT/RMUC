"""Build an interactive map replay for one RM match.

The replay intentionally separates:
* current observable entities (what the selected side can read now),
* model forecasts at 10/30/60/120 seconds, and
* exact future truth used only for evaluation.

The generated HTML embeds the supplied RM floor-plan image and a compact
three-second decision timeline, so it can be opened without a local server.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import io
import json
import mimetypes
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.db import connect
from data.forecast_dataset import build_game_forecast_examples
from forecast.temporal import HORIZONS, MOBILE_TYPES, target_labels
from runtime.temporal_forecast import (resolve_temporal_forecaster,
                                       VALIDATION_POSITION_FUSION_WEIGHTS)
from worldstate.functional_zone import (
    FIELD_HEIGHT_M,
    FIELD_WIDTH_M,
    FUNCTIONAL_ZONE_LABELS,
    ZONE_POLYGONS,
    assign_functional_zone,
    zone_centroid,
)
from worldstate.canonical_zone import locate_zone, neighbors, is_legal_transition
from worldstate.unit_state import derive_unit_special_state
from worldstate.narrative import build_global_narrative


DEFAULT_MAP = Path(r"F:\documents\Tencent Files\2820916363\nt_qq\nt_data\Pic\2026-09\Ori\50f788026ad4944f212aebc17658b3a7.jpg")
DEFAULT_OUT = Path(r"C:\Users\AntonyTang\.codex\visualizations\2026\09\24\01a0d26c-e553-7562-9bc2-455ceecdbfaa\rm-battle-replay.html")
CANONICAL_REGION_PATH = ROOT / "out_m2" / "official" / "canonical_zone_regions.json"
CANONICAL_LABEL_PATH = ROOT / "out_m2" / "official" / "canonical_zone_labels.json"

# Display-only footprints calibrated against the supplied top-down plan.  The
# runtime's rule polygons remain the stable prediction vocabulary; these
# footprints are used to keep the overlay aligned with visible structures and
# to avoid presenting the old coarse rectangles as physical walls.
DISPLAY_ZONE_POLYGONS = {
    '红基地': [(0.90, 9.40), (2.35, 10.30), (3.80, 9.40),
               (3.80, 5.35), (2.35, 4.45), (0.90, 5.45)],
    '红停机坪': [(0.55, 13.05), (0.95, 14.05), (1.75, 14.35),
                 (2.25, 13.80), (2.25, 12.85), (1.55, 12.35),
                 (0.75, 12.65)],
    '红补给区': [(0.35, 0.35), (3.30, 0.35), (3.30, 2.55),
                 (0.35, 2.55)],
    # The R-marked footprint in the source plan is the red assembly area.
    '红上装配区': [(5.20, 10.85), (9.55, 10.85), (10.55, 12.55),
                   (9.55, 14.95), (5.20, 14.95)],
    '红前哨区': [(5.45, 6.05), (6.65, 5.65), (7.75, 6.35),
                 (7.75, 8.35), (6.65, 8.95), (5.45, 8.35)],
    '红梯形高地区': [(4.55, 9.20), (6.95, 9.45), (7.55, 11.25),
                     (5.05, 12.25), (4.10, 11.05)],
    '红上公路': [(7.10, 12.60), (9.10, 12.60), (11.75, 14.95),
                 (9.85, 14.95)],
    '红下公路': [(7.15, 0.05), (11.85, 0.05), (10.65, 2.30),
                 (7.00, 2.30)],
    '红中路': [(7.15, 3.15), (10.30, 4.00), (10.30, 11.00),
               (7.15, 11.85)],
    '中央北隧道': [(11.80, 11.70), (16.20, 11.70),
                    (16.20, 13.05), (11.80, 13.05)],
    '中央南隧道': [(11.80, 1.95), (16.20, 1.95),
                    (16.20, 3.30), (11.80, 3.30)],
    '中央高地': [(10.20, 4.00), (11.40, 2.80), (16.60, 2.80),
                 (17.80, 4.00), (17.80, 11.00), (16.60, 12.20),
                 (11.40, 12.20), (10.20, 11.00)],
}


def _num(value, default=None):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if value == value else default


def _matured_horizons(issued_at, current_time, horizons=HORIZONS):
    """Return only future targets whose exact timestamps have been reached."""
    issued_at = float(issued_at)
    current_time = float(current_time)
    return [int(horizon) for horizon in horizons
            if current_time >= issued_at + float(horizon) - 1e-6]


def _pos(value):
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        x, y = _num(value[0]), _num(value[1])
        if x is not None and y is not None:
            return [round(x, 3), round(y, 3)]
    return None


def _top_probability(values):
    if not isinstance(values, dict) or not values:
        return None, None
    label = max(values, key=lambda key: float(values[key]))
    return label, round(float(values[label]), 3)


@lru_cache(maxsize=128)
def _canonical_centroid(zone_id):
    """Return a canonical-zone centroid in the runtime (y-up) frame."""
    if not zone_id:
        return None
    try:
        data = json.loads(CANONICAL_REGION_PATH.read_text(encoding="utf-8"))
        row = next((item for item in data.get("regions", [])
                    if item.get("zone_id") == zone_id), None)
        if not row:
            return None
        x, y_down = row.get("centroid_field_m", [None, None])
        if x is None or y_down is None:
            return None
        return [float(x), FIELD_HEIGHT_M - float(y_down)]
    except Exception:
        return None


@lru_cache(maxsize=4096)
def _canonical_reachable(current_zone, horizon_seconds, speed_mps):
    """Return canonical zones reachable through legal edges by a horizon.

    The replay is deliberately conservative at +10s (stay or one edge), then
    lets longer forecasts traverse several edges when the robot's speed and
    the accumulated centroid route length allow it.
    """
    if not current_zone:
        return {}
    horizon = max(0.0, float(horizon_seconds or 10.0))
    speed = max(0.05, float(speed_mps or 0.4))
    max_hops = max(1, int(horizon // 10.0))
    if horizon <= 10.0 + 1e-9:
        max_hops = 1
    max_distance = speed * horizon
    reachable = {current_zone: (0.0, [current_zone])}
    queue = [(current_zone, 0, 0.0, [current_zone])]
    while queue:
        zone, hops, distance, path = queue.pop(0)
        if hops >= max_hops:
            continue
        for nxt in neighbors(zone):
            a, b = _canonical_centroid(zone), _canonical_centroid(nxt)
            edge = (((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
                    if a is not None and b is not None else 0.0)
            total = distance + edge
            if total > max_distance + 1e-6:
                continue
            candidate_path = path + [nxt]
            old = reachable.get(nxt)
            if old is None or total < old[0]:
                reachable[nxt] = (total, candidate_path)
                queue.append((nxt, hops + 1, total, candidate_path))
    return reachable


def _project_canonical_target(current_zone, predicted_zone, predicted_position,
                              horizon_seconds=10, speed_mps=None):
    """Project a target onto the speed- and horizon-reachable canonical graph."""
    if not current_zone:
        return predicted_zone
    reachable = _canonical_reachable(current_zone, horizon_seconds, speed_mps)
    if not reachable:
        return current_zone
    if predicted_zone in reachable:
        return predicted_zone
    if not predicted_position:
        return min(reachable, key=lambda z: (len(reachable[z][1]), reachable[z][0]))
    px, py = float(predicted_position[0]), float(predicted_position[1])
    scored = []
    for candidate, (route_distance, path) in reachable.items():
        centroid = _canonical_centroid(candidate)
        if centroid is None:
            continue
        scored.append((((centroid[0] - px) ** 2 + (centroid[1] - py) ** 2) ** 0.5,
                       len(path), route_distance, candidate))
    return min(scored)[3] if scored else current_zone


def _special_summary(special):
    if not isinstance(special, dict):
        return "unknown"
    for key in ("attack_status_probabilities", "resource_status_probabilities",
                "flight_status_probabilities", "resource_phase_probabilities"):
        label, probability = _top_probability(special.get(key))
        if label is not None:
            return f"{label} ({probability:.2f})"
    return str(special.get("state") or special.get("kind") or "unknown")


def _guidance_display(guidance):
    """Compactly explain rule advice without implying calibration or execution."""
    guidance = guidance if isinstance(guidance, dict) else {}
    sources = []
    for item in guidance.get("evidence") or []:
        source = item.get("source") if isinstance(item, dict) else None
        label = ("传感器观测" if source in {
            "current_observation", "current_position", "direct_sensor_field", "self_sensor"
        } else "区域图计算" if source == "functional_zone_graph"
            else "几何推断" if source in {
                "visible_geometry_inference", "functional_zone_prior"
            } else "数据缺失/来源未注明")
        if label not in sources:
            sources.append(label)
    scorecard = guidance.get('decision_scorecard') or {}
    feasibility = scorecard.get('feasibility') or {}
    if feasibility.get('status') == 'scored':
        feasibility_text = f"路线可行性{float(feasibility.get('score') or 0):.0f}%"
    elif feasibility.get('status') == 'not_required':
        feasibility_text = '无需跨区'
    elif feasibility.get('status') == 'failed':
        feasibility_text = '路线不可行'
    else:
        feasibility_text = '路线未知'
    execution = scorecard.get('execution') or {}
    execution_text = ('执行未知（无命令日志）' if execution.get('status') == 'unknown'
                      else '不适用' if execution.get('status') == 'not_applicable'
                      else '待核验')
    consequence = scorecard.get('consequence') or {}
    metrics = consequence.get('metrics') or []
    metric_labels = {
        'current_zone_retention': '是否留在当前区', 'own_survival': '我方存活',
        'own_firing_event': '本机是否有发弹记录（行为响应，不代表命令执行）',
        'own_hp_change': '我方血量变化', 'identified_target_hp_change': '目标血量变化',
        'target_alive': '目标存活', 'progress_toward_safe_zone': '是否靠近安全区',
        'resource_arrival': '是否到资源区', 'resource_exchange_completion': '是否完成兑换',
        'resource_collection': '采集结果', 'defense_zone_occupancy': '是否留守防区',
        'facility_damage': '设施损伤',
    }
    consequence_text = ('待3/10秒核对' + '、'.join(metric_labels.get(m,m) for m in metrics[:2])
                        if consequence.get('status') == 'pending' and metrics
                        else '后果未知' if consequence.get('status') == 'unknown'
                        else '无结果指标')
    causal = scorecard.get('causal_benefit') or {}
    return {
        "status": "规则建议（未校准，不代表已执行）",
        "trigger": guidance.get("trigger_condition") or "未提供，等待新观测确认",
        "stop": guidance.get("cancel_condition") or "未提供，需随新观测重新评估",
        "evidence": "、".join(sources) if sources else "数据缺失/来源未注明",
        "feasibility": feasibility_text,
        "execution": execution_text,
        "consequence": consequence_text,
        "causal_benefit": ('收益因果未知' if causal.get('status') == 'not_identifiable'
                           else '不适用'),
    }


def _compact_current(entity, all_entities, team):
    pos = _pos(entity.get("position"))
    raw_hp = entity.get("hp")
    hp = _num(raw_hp)
    # Only explicit HP or an upstream lifecycle label can establish death.
    # A stationary position, a stale/missing track, or zero coordinates do not
    # establish it; absent/invalid HP without a valid upstream label is unknown.
    upstream_life = entity.get("life_state")
    if isinstance(raw_hp, bool):
        life_state, life_source = "unknown", "invalid_hp"
    elif hp is not None:
        life_state = "destroyed" if hp <= 0 else "active"
        life_source = "explicit_current_hp"
    elif upstream_life in ("active", "destroyed", "unknown"):
        life_state, life_source = upstream_life, "upstream_lifecycle"
    else:
        life_state, life_source = "unknown", "hp_missing"
    if life_state not in ("active", "destroyed", "unknown"):
        life_state = "unknown"
        life_source = "invalid_lifecycle"
    special = derive_unit_special_state(entity, all_entities, self_team=team)
    guidance = special.get("decision_guidance")
    if isinstance(guidance, dict) and guidance.get("applicable"):
        special["guidance_display"] = _guidance_display(guidance)
    attack_target = special.get("attacking_target") if isinstance(special, dict) else None
    attack_target_entity = None
    if isinstance(attack_target, dict) and attack_target.get("entity_id") is not None:
        attack_target_entity = next(
            (candidate for candidate in all_entities
             if str(candidate.get("entity_id")) == str(attack_target.get("entity_id"))),
            None,
        )
    attack_target_position = _pos((attack_target_entity or {}).get("position"))
    target_life_state = "unknown"
    if attack_target_entity:
        target_hp_raw = attack_target_entity.get("hp")
        target_hp = _num(target_hp_raw)
        target_upstream_life = attack_target_entity.get("life_state")
        if isinstance(target_hp_raw, bool):
            target_life_state = "unknown"
        elif target_hp is not None:
            target_life_state = "destroyed" if target_hp <= 0 else "active"
        elif target_upstream_life in ("active", "destroyed", "unknown"):
            target_life_state = target_upstream_life
    return {
        "entity_id": entity.get("entity_id"),
        "team": entity.get("team"),
        "role": entity.get("role"),
        "type": entity.get("entity_type"),
        "position": pos,
        "zone": assign_functional_zone(*pos) if pos else "场地未知区",
        "canonical_zone": locate_zone(*pos) if pos else None,
        "hp": None if isinstance(raw_hp, bool) else hp,
        "maxhp": _num(entity.get("maxhp")),
        "hp_ratio": (hp / _num(entity.get("maxhp"))
                     if not isinstance(raw_hp, bool) and hp is not None and _num(entity.get("maxhp"))
                     and _num(entity.get("maxhp")) > 0 else None),
        "life_state": life_state,
        "life_state_source": life_source,
        "yaw_deg": _num(entity.get("yaw")),
        "velocity": [_num(entity.get("velocity_x")), _num(entity.get("velocity_y"))]
                    if entity.get("velocity_x") is not None and entity.get("velocity_y") is not None else None,
        "speed_mps": _num(entity.get("speed_mps")),
        "motion_direction_deg": _num(entity.get("motion_direction_deg")),
        "orientation_source": "muzzle_yaw_only" if entity.get("yaw") is not None else None,
        "source": entity.get("source"),
        "observability": entity.get("observability_state"),
        "special": _special_summary(special),
        "special_state": special,
        "attack_target": ({
            "entity_id": attack_target.get("entity_id"),
            "entity_type": attack_target.get("entity_type"),
            "position": attack_target_position,
            "life_state": target_life_state,
        } if attack_target and attack_target_position else None),
    }


def _compact_global_prediction(horizon_prediction, adaptive_by_task=None, horizon=None):
    """Keep active-head provenance visible in the replay's compact payload."""
    prediction = horizon_prediction or {}
    head = prediction.get("global_mode") or {}
    baseline = prediction.get("global_mode_neural_baseline") or {}
    return {
        "mode": head.get("top"),
        "mode_confidence": head.get("confidence"),
        "mode_probabilities": head.get("probabilities", {}),
        "mode_source": prediction.get("global_mode_source"),
        "mode_neural_baseline": baseline.get("top"),
        "trend": (prediction.get("opponent_trend") or {}).get("top"),
        "trend_confidence": (prediction.get("opponent_trend") or {}).get("confidence"),
        "trend_probabilities": (prediction.get("opponent_trend") or {}).get("probabilities", {}),
        "region": (prediction.get("enemy_region") or {}).get("top"),
        "region_confidence": (prediction.get("enemy_region") or {}).get("confidence"),
        "region_probabilities": (prediction.get("enemy_region") or {}).get("probabilities", {}),
        "adaptive_by_task": {
            task: dict((adaptive_by_task or {}).get(f"{task}@{int(horizon)}s") or {})
            for task in ("global_mode", "opponent_trend", "enemy_region")
        } if horizon is not None else {},
    }


def _adaptive_feedback_text(adaptive_by_task):
    """Describe within-match scoring/routing without implying weight updates."""
    names = {"global_mode": "全局态势", "opponent_trend": "敌方趋势",
             "enemy_region": "敌方区域"}
    reason_text = {
        "warmup": "结果积累中", "model_kept": "暂沿用模型",
        "model_underperforms_persistence": "近期误差较大，切到持久性基线",
        "persistence_kept": "近期基线更稳，继续用持久性",
        "model_recovered": "模型近期恢复，切回模型",
    }
    if not isinstance(adaptive_by_task, dict) or not adaptive_by_task:
        return "局内反馈状态未知"
    parts = []
    for task in ("global_mode", "opponent_trend", "enemy_region"):
        state = adaptive_by_task.get(task) or {}
        if not state:
            continue
        strategy = state.get("strategy", "model")
        selected = "持久性基线" if strategy == "persistence" else "时序模型"
        samples = int(state.get("samples") or 0)
        reason = reason_text.get(state.get("reason"), "继续观察")
        parts.append(f"{names[task]}用{selected}，已核验{samples}次（{reason}）")
    if not parts:
        return "局内反馈状态未知"
    return "局内复盘：" + "；".join(parts) + "。只切换预测来源，不更新模型权重。"


def _replay_global_narrative(side, current_entities, forecast, horizon_seconds,
                            unit_forecasts=None, include_details=True,
                            current_description=None):
    """Build the replay's auditable three-part brief from its actual inputs.

    Only mobile robots enter the side summaries. Building status remains in
    the evidence payload but is not counted as a robot. Forecast confidences
    remain tied to their own tasks instead of sharing one headline score.
    """
    opposite = "蓝" if side == "红" else "红"
    robots = [entity for entity in (current_entities or [])
              if entity.get("type") in MOBILE_TYPES]
    own_zones = {entity.get("zone") for entity in robots
                 if entity.get("team") == side and entity.get("position")
                 and entity.get("life_state") != "destroyed"}
    enemy_zones = {entity.get("zone") for entity in robots
                   if entity.get("team") == opposite and entity.get("position")
                   and entity.get("life_state") != "destroyed"}
    current_contested = sorted(zone for zone in own_zones & enemy_zones if zone)
    forecast_contested = list((forecast.get("outlook") or {}).get("contested_zones") or [])[:3]

    visible_unit_states = [{
        "entity_id": entity.get("entity_id"), "team": entity.get("team"),
        "type": entity.get("type"), "zone": entity.get("zone"),
        "life_state": entity.get("life_state"),
        "observability": entity.get("observability", "unavailable"),
        "state": entity.get("special_state") or {},
    } for entity in robots]
    unit_locations = [{
        "entity_id": entity.get("entity_id"), "hp_ratio": entity.get("hp_ratio"),
        "vulnerable": (entity.get("hp_ratio") is not None
                       and 0 < float(entity.get("hp_ratio")) < .3),
    } for entity in robots]
    buildings = {entity.get("type"): entity for entity in (current_entities or [])
                 if entity.get("team") == side and entity.get("type") in ("基地", "前哨站")}

    trend_names = {"推进": "向我方推进", "press": "向我方推进",
                   "稳定": "暂时维持位置", "hold": "暂时维持位置",
                   "回撤": "正在回撤", "retreat": "正在回撤"}
    trend = forecast.get("trend")
    mode = forecast.get("mode")
    mode_confidence = forecast.get("mode_confidence")
    trend_confidence = forecast.get("trend_confidence")
    source = ("30秒分类候选" if forecast.get("mode_source") == "mode_change_xgb_v1"
              else "时序模型")
    horizon_use = {
        10: "短时路线参考", 30: "局部控制准备",
        60: "中期走势参考", 120: "只看总体方向",
    }
    purpose = horizon_use.get(int(horizon_seconds), "趋势参考")
    prediction_parts = []
    if trend in trend_names and isinstance(trend_confidence, (int, float)):
        prediction_parts.append(
            f"敌方趋势偏向{trend_names[trend]}（{float(trend_confidence):.0%}）")
    elif trend in trend_names:
        prediction_parts.append(f"敌方趋势偏向{trend_names[trend]}（置信度未知）")
    mode_names = {"进攻": "整体态势偏进攻", "防守": "整体态势偏防守",
                  "均衡": "整体态势暂时均衡"}
    if mode in mode_names and isinstance(mode_confidence, (int, float)):
        prediction_parts.append(f"{mode_names[mode]}（{float(mode_confidence):.0%}）")
    elif mode in mode_names:
        prediction_parts.append(f"{mode_names[mode]}（置信度未知）")
    region = forecast.get("region")
    region_confidence = forecast.get("region_confidence")
    if (region and isinstance(region_confidence, (int, float))
            and float(region_confidence) >= .45):
        prediction_parts.append(
            f"区域倾向{region}（{float(region_confidence):.0%}）")
    elif region:
        prediction_parts.append(f"区域方向置信度偏低")
    if not prediction_parts:
        prediction_parts.append("趋势和态势方向都不确定，先观察")
    # Keep each task's confidence tied to that task. Explain how much detail
    # this horizon supports, then remind the reader to check future evidence.
    prediction_text = (f"+{int(horizon_seconds)}秒（{purpose}）："
                       f"{'；'.join(prediction_parts)}；后续按实况核验。")

    outlook = forecast.get("outlook") or {}
    original_actions = outlook.get("recommended_actions") or []
    first_action = dict(original_actions[0]) if original_actions else {}
    objective = first_action.get("objective")
    if objective and current_contested and "区域控制" in objective:
        zone_text = "、".join(current_contested[:2])
        objective = objective.replace("当前区域控制", f"争夺区（{zone_text}）内阵型")
        first_action["objective"] = objective
    elif objective and forecast_contested and "区域控制" in objective:
        objective = objective.replace("保持当前区域控制", "优先关注模型指出的" +
                                      "、".join(forecast_contested[:2]))
        first_action["objective"] = objective
    if first_action:
        first_action.setdefault("action", "replay_conditional_guidance")

    intruders = [{"type": entity.get("type"), "zone": entity.get("zone"),
                  "entity_id": entity.get("entity_id")}
                 for entity in robots if entity.get("team") == opposite
                 and entity.get("position") and entity.get("zone", "").startswith(side)
                 and entity.get("life_state") != "destroyed"]
    analysis = {
        "current_mode": "态势未知",
        "source": "replay_current_observations_and_model_forecast",
        "confidence": 0,
        "recommendations": [first_action] if first_action else [],
        "detailed_posture": {
            "label": (outlook.get("tactical_situation") or {}).get("label"),
            "objective": objective,
            "trigger": first_action.get("trigger"),
            "fallback": first_action.get("fallback"),
        },
        "state": {
            "team": side, "visible_unit_states": visible_unit_states,
            "unit_locations": unit_locations,
            "zone_control": {"contested_zones": current_contested},
            "control_posture": "contested" if current_contested else "unknown",
            "enemy_trend": "unknown",
            "risk_register": {"enemy_intruders": intruders,
                              "own_vulnerable_units": []},
            "own_base_hp": (buildings.get("基地") or {}).get("hp_ratio"),
            "own_outpost_hp": (buildings.get("前哨站") or {}).get("hp_ratio"),
        },
        "forecast_context": {"predictions": [{
            "horizon_seconds": int(horizon_seconds),
            "global_mode": {"top": mode, "confidence": mode_confidence},
            "opponent_trend": {"top": trend, "confidence": trend_confidence},
            "enemy_region": {"top": forecast.get("region"),
                             "confidence": forecast.get("region_confidence")},
            "outlook": outlook,
            "entities": [{
                "side": "self" if item.get("role") == "self" else "opponent",
                "entity_id": item.get("entity_id"),
                "entity_type": item.get("type"),
                "from_zone": item.get("from_zone"), "to_zone": item.get("to_zone"),
                "presence": item.get("presence"),
                "forecast_suppressed": item.get("forecast_suppressed"),
                "life_state": item.get("life_state"),
            } for item in (unit_forecasts or {}).values()],
        }]},
    }
    narrative = (build_global_narrative(analysis)
                 if current_description is None or include_details else None)
    sections = (narrative.get("sections") or {}) if narrative else {}
    description = (current_description if current_description is not None else
                   narrative.get("description", "现况：当前信息不足"))
    description = description.replace("现况（态势未知）：阶段：", "现况：战术阶段为")
    description = description.replace("现况（态势未知；阶段：", "现况（战术阶段：")
    for zone in current_contested:
        repeated_location = f"；{zone}争夺中。"
        if description.endswith(repeated_location):
            description = description[:-len(repeated_location)] + "。"
            description = description.replace(
                "争夺关键区", f"争夺关键区（{zone}）", 1)
    forecast_detail = sections.get("forecast_analysis") or {}
    forecast_detail["summary"] = prediction_text
    forecast_detail["model_source"] = source
    if forecast.get("mode_neural_baseline"):
        forecast_detail["neural_baseline"] = forecast["mode_neural_baseline"]
    for record in forecast_detail.get("horizons") or []:
        # The UI reports each task confidence separately. Remove the old
        # max-of-different-tasks convenience field from expanded evidence.
        record.pop("confidence", None)
    # Per-unit facts and routes are rendered in the dedicated unit table.
    # Keeping its duplicate narrative rows in every horizon multiplies the
    # replay payload without adding evidence to the global brief.
    sections.pop("unit_analysis", None)
    recommendation_detail = sections.get("overall_recommendation") or {}
    objective = (recommendation_detail.get("objective") or first_action.get("objective")
                 or "等待新观测后再判断")
    trigger = (recommendation_detail.get("trigger") or first_action.get("trigger")
               or "暂缺明确触发条件")
    fallback = (recommendation_detail.get("fallback") or first_action.get("fallback")
                or "保持可回撤位置并继续观察")
    objective_raw = str(objective or "")
    if any(term in objective_raw for term in ("区域控制", "争夺区", "争夺区域")):
        objective_short = (f"稳住{current_contested[0]}" if current_contested
                           else "稳住争夺区")
    elif "基地入口" in objective_raw or "回防路线" in objective_raw:
        objective_short = "守住基地入口"
    elif "资源" in objective_raw or "兑换" in objective_raw:
        objective_short = "完成资源兑换"
    elif "压制" in objective_raw or "攻击" in objective_raw:
        objective_short = "压制关键目标"
    else:
        objective_short = objective_raw.split("，", 1)[0].split("；", 1)[0]
        if len(objective_short) > 16:
            objective_short = "稳住关键位置"
        objective_short = objective_short or "继续观察"

    trigger_raw = str(trigger or "")
    if "连续帧推进" in trigger_raw and "建筑受击" in trigger_raw:
        trigger_short = "敌方持续推进或建筑受击"
    elif "进入防区" in trigger_raw:
        trigger_short = "敌方进入防区"
    elif "持矿" in trigger_raw or "携带资源" in trigger_raw:
        trigger_short = "携带资源且路线通畅"
    elif "目标可见" in trigger_raw and "路线可达" in trigger_raw:
        trigger_short = "目标可见且路线可达"
    elif "确认目标" in trigger_raw or "可攻击目标" in trigger_raw:
        trigger_short = "发现可攻击目标"
    elif "建筑受击" in trigger_raw:
        trigger_short = "己方建筑受击"
    elif "持续推进" in trigger_raw or "连续帧推进" in trigger_raw:
        trigger_short = "敌方持续推进"
    else:
        trigger_short = "出现明确变化"
    recommendation_text = f"建议：{objective_short}；触发：{trigger_short}。"
    adaptive_feedback = _adaptive_feedback_text(forecast.get("adaptive_by_task"))
    return {
        "description": description,
        "prediction_analysis": prediction_text,
        "recommendation": recommendation_text,
        "adaptive_feedback": adaptive_feedback,
        "details": sections if include_details else {},
    }


def _group_entities(entities, side, entity_type):
    target_team = side if side in ("红", "蓝") else None
    # The model uses role-relative slots.  Its self slot is the observing team;
    # its opponent slot is the other team.
    role = "self" if side == target_team else "opponent"
    return [e for e in entities
            if e.get("entity_type") == entity_type
            and (e.get("team") == target_team if role == "self"
                 else e.get("team") != target_team)]


def _aggregate_truth(truth, observing_side):
    entities = list((truth or {}).get("entities", {}).values())
    result = {}
    for role, team in (("self", observing_side),
                       ("opponent", "蓝" if observing_side == "红" else "红")):
        for entity_type in MOBILE_TYPES:
            values = [e for e in entities
                      if e.get("team") == team and e.get("机器人类型", e.get("entity_type")) == entity_type]
            # Truth entities use entity_type; the fallback above keeps this
            # helper tolerant of a hand-built evaluation fixture.
            if not values:
                values = [e for e in entities
                          if e.get("team") == team and e.get("entity_type") == entity_type]
            # Historical rows persist after destruction. For trajectory
            # scoring, explicit future HP=0 means the unit is no longer alive.
            values = [e for e in values
                      if not (isinstance(e.get("hp"), (int, float))
                              and not isinstance(e.get("hp"), bool) and e.get("hp") <= 0)]
            positions = [_pos(e.get("position")) for e in values]
            positions = [p for p in positions if p is not None]
            centroid = None
            if positions:
                centroid = [round(sum(p[0] for p in positions) / len(positions), 3),
                            round(sum(p[1] for p in positions) / len(positions), 3)]
            key = f"{role}:{entity_type}"
            result[key] = {
                "present": bool(values),
                "position": centroid,
                "zone": assign_functional_zone(*centroid) if centroid else "场地未知区",
                "count": len(values),
            }
    return result


def _forecast_entities(prediction, horizon, current_entities, selected_only=True):
    values = prediction.get("horizons", {}).get(str(horizon), {}).get("entities", {})
    current_by_slot = {}
    current_lifecycle_by_slot = {}
    for entity in current_entities:
        if entity.get("type") not in MOBILE_TYPES:
            continue
        key = f"{entity.get('role')}:{entity.get('type')}"
        current_by_slot.setdefault(key, []).append(entity.get("position"))
        current_lifecycle_by_slot.setdefault(key, []).append(entity.get("life_state", "unknown"))
    output = {}
    for slot, entity in values.items():
        typ = entity.get("entity_type")
        if typ not in MOBILE_TYPES:
            continue
        # The map always shows both sides' current observations.  Forecast
        # markers are restricted to the selected observer's own slots by
        # default, so a dashed marker cannot be mistaken for a second team's
        # current robot.  Selecting the other side runs the other forecast.
        if selected_only and entity.get("side") != "self":
            continue
        slot_lifecycle = current_lifecycle_by_slot.get(slot, [])
        explicitly_destroyed = bool(slot_lifecycle) and all(
            value == "destroyed" for value in slot_lifecycle)
        presence = round(float(entity.get("presence_probability", 0.0)), 3)
        pos = entity.get("position") or {}
        point = [_num(pos.get("x")), _num(pos.get("y"))]
        if point[0] is None or point[1] is None:
            point = None
        current_points = [p for p in current_by_slot.get(slot, []) if p]
        current_point = None
        if current_points:
            current_point = [round(sum(p[0] for p in current_points) / len(current_points), 3),
                             round(sum(p[1] for p in current_points) / len(current_points), 3)]
        forecast_destroyed = (entity.get("life_state") == "destroyed" or
                              entity.get("state") == "destroyed")
        suppressed = bool(entity.get("forecast_suppressed") or
                          explicitly_destroyed or forecast_destroyed)
        raw_from_canonical = entity.get("from_canonical_zone")
        # A few legacy forecast rows omitted the source zone.  Recover it
        # from the latest observed position so the rendered arrow still has
        # a legal, local origin instead of jumping across the map.
        if not suppressed and not raw_from_canonical and current_point:
            raw_from_canonical = locate_zone(*current_point)
        if not suppressed and not raw_from_canonical and point:
            raw_from_canonical = locate_zone(*point)
        raw_to_canonical = entity.get("to_canonical_zone")
        speed_mps = entity.get("speed_mps")
        if speed_mps is None:
            speed_mps = (entity.get("movement") or {}).get("speed_mps")
        reachable = {} if suppressed else _canonical_reachable(raw_from_canonical, horizon, speed_mps)
        display_to_canonical = (None if suppressed else _project_canonical_target(
            raw_from_canonical, raw_to_canonical, point, horizon, speed_mps))
        output[slot] = {
            "role": entity.get("side"),
            "team": entity.get("team"),
            "type": typ,
            "presence": presence,
            "life_state": "destroyed" if (explicitly_destroyed or forecast_destroyed)
                          else entity.get("life_state", "unknown"),
            "forecast_suppressed": suppressed,
            "suppression_reason": (entity.get("suppression_reason") or
                                   ("explicit_current_hp_zero" if explicitly_destroyed else None)),
            "position": None if suppressed else point,
            "from_zone": None if suppressed else (entity.get("from_zone") or "场地未知区"),
            "to_zone": None if suppressed else (entity.get("to_zone") or "场地未知区"),
            "from_canonical_zone": None if suppressed else raw_from_canonical,
            "to_canonical_zone": display_to_canonical,
            "raw_to_canonical_zone": None if suppressed else raw_to_canonical,
            # Store intermediate waypoints only when the horizon actually
            # spans more than one edge; one-edge routes are already clear from
            # from_zone → to_zone and would needlessly inflate the HTML.
            "canonical_path": ([] if suppressed else (reachable.get(display_to_canonical) or (None, []))[1]
                               if len((reachable.get(display_to_canonical) or (None, []))[1]) > 2
                               else []),
            "movement": None if suppressed else (entity.get("movement") or {}),
            "speed_mps": None if suppressed else entity.get("speed_mps"),
            "motion_direction_deg": None if suppressed else (entity.get("motion_direction_deg") or
                                    (entity.get("movement") or {}).get("motion_direction_deg")),
            "heading_deg": None if suppressed else (entity.get("heading_deg") or
                           (entity.get("movement") or {}).get("heading_deg")),
            "direction_correction_applied": bool(
                entity.get("direction_correction_applied") or
                (entity.get("movement") or {}).get("direction_correction_applied")),
            "special": None if suppressed else _special_summary(entity.get("special_state_forecast")),
            "special_state": {} if suppressed else (entity.get("special_state_forecast") or entity.get("special_state") or {}),
            "state": entity.get("state"),
            "current_position": current_point,
        }
    return output


def _evaluate(predicted, truth_by_slot):
    presence = []
    zones = []
    errors = []
    for slot, item in predicted.items():
        truth = truth_by_slot.get(slot, {"present": False, "position": None, "zone": "场地未知区"})
        pred_present = item["presence"] >= 0.35
        presence.append(float(pred_present == bool(truth["present"])))
        if pred_present and truth["present"]:
            zones.append(float(item["to_zone"] == truth["zone"]))
            if item.get("position") and truth.get("position"):
                dx = item["position"][0] - truth["position"][0]
                dy = item["position"][1] - truth["position"][1]
                errors.append((dx * dx + dy * dy) ** 0.5)
    return {
        "presence_accuracy": round(sum(presence) / len(presence), 3) if presence else None,
        "zone_accuracy": round(sum(zones) / len(zones), 3) if zones else None,
        "position_mae_m": round(sum(errors) / len(errors), 3) if errors else None,
    }


def _zone_payload():
    """Return the user-approved canonical 59-zone planar partition.

    ``canonical_zone_regions.json`` is the geometry source of truth.  Its
    polygons are kept in image coordinates (y down), while the replay uses
    world coordinates (y up), so this function flips y once and clips the
    few anti-aliased line overlaps deterministically.  The uncovered floor is
    emitted as one or more ``场地开放区`` pieces so the SVG remains a complete,
    non-overlapping partition.
    """
    try:
        from shapely.geometry import Polygon, box
        from shapely.ops import triangulate, unary_union
    except ImportError:
        # The normal workstation has Shapely.  Keep a deterministic fallback
        # for stripped-down environments: emit the canonical polygons without
        # fill ownership; the renderer still draws the exact new boundaries.
        data = json.loads(CANONICAL_REGION_PATH.read_text(encoding="utf-8"))
        labels = {}
        if CANONICAL_LABEL_PATH.exists():
            labels = {r.get("zone_id"): r for r in json.loads(
                CANONICAL_LABEL_PATH.read_text(encoding="utf-8")).get("regions", [])}
        payload = []
        for row in data.get("regions", []):
            zone_id = row["zone_id"]
            label = labels.get(zone_id, {})
            points = [[float(x), FIELD_HEIGHT_M - float(y)]
                      for x, y in row.get("polygon_field_m", [])]
            centroid = [float(row.get("centroid_field_m", [0, 0])[0]),
                        FIELD_HEIGHT_M - float(row.get("centroid_field_m", [0, 0])[1])]
            payload.append({
                "id": zone_id, "side": row.get("side", "purple"),
                "points": points, "centroid": centroid,
                "label": True,
                "label_text": f"{label.get('side_name', row.get('side', '中央'))}·{zone_id}",
            })
        return payload

    data = json.loads(CANONICAL_REGION_PATH.read_text(encoding="utf-8"))
    label_rows = {}
    if CANONICAL_LABEL_PATH.exists():
        label_rows = {r.get("zone_id"): r for r in json.loads(
            CANONICAL_LABEL_PATH.read_text(encoding="utf-8")).get("regions", [])}
    frame = box(0.0, 0.0, FIELD_WIDTH_M, FIELD_HEIGHT_M)
    raw = []
    for row in data.get("regions", []):
        zone_id = row["zone_id"]
        points = [(float(x), FIELD_HEIGHT_M - float(y))
                  for x, y in row.get("polygon_field_m", [])]
        if len(points) < 3:
            continue
        geom = Polygon(points).intersection(frame).buffer(0)
        if geom.is_empty:
            continue
        meta = label_rows.get(zone_id, {})
        raw.append({
            "id": zone_id,
            "side": row.get("side", "purple"),
            "geom": geom,
            "centroid": [float(row.get("centroid_field_m", [0, 0])[0]),
                         FIELD_HEIGHT_M - float(row.get("centroid_field_m", [0, 0])[1])],
            "label_text": f"{meta.get('side_name', {'red':'红侧','blue':'蓝侧','purple':'中央'}.get(row.get('side'), '中央'))}·{zone_id}",
        })

    # Ownership clipping turns the raster-extracted boundary strokes into a
    # strict planar partition without changing the canonical outer outlines.
    used = None
    pieces = []
    # Claim smaller enclosed modules first (bases, outposts, resource pads),
    # then let larger surrounding polygons own the remaining floor.  This
    # preserves all 59 canonical IDs instead of clipping an inner module away
    # when the raster extraction contains a nested boundary.
    for item in sorted(raw, key=lambda value: value["geom"].area):
        geom = item["geom"] if used is None else item["geom"].difference(used).buffer(0)
        if geom.is_empty:
            continue
        used = geom if used is None else unary_union([used, geom]).buffer(0)
        pieces.append((item, geom))
    remainder = frame if used is None else frame.difference(used).buffer(0)
    if not remainder.is_empty:
        pieces.append(({
            "id": "场地开放区", "side": "open",
            "centroid": None, "label_text": "开放地面",
        }, remainder))

    payload = []
    for item, geom in pieces:
        parts = list(getattr(geom, "geoms", (geom,)))
        for part_index, part in enumerate(parts):
            if part.is_empty or part.area <= 1e-5:
                continue
            # SVG polygons cannot encode holes.  Triangulate only the pieces
            # that have interiors and clip the triangles back to the piece.
            simple = []
            if getattr(part, "interiors", ()):
                for triangle in triangulate(part):
                    clipped = triangle.intersection(part)
                    if not clipped.is_empty and clipped.area > 1e-5:
                        simple.extend(list(getattr(clipped, "geoms", (clipped,))))
            else:
                simple = [part]
            for tri_index, polygon in enumerate(simple):
                if polygon.is_empty or polygon.area <= 1e-5:
                    continue
                coords = list(polygon.exterior.coords)[:-1]
                # Use the source centroid for canonical labels when it remains
                # in this ownership piece; otherwise use a representative point.
                representative = polygon.representative_point()
                source_centroid = item.get("centroid")
                label_here = item["id"] != "场地开放区" and (
                    part_index == 0 and tri_index == 0 and polygon.area >= 0.20)
                payload.append({
                    "id": item["id"], "side": item.get("side", "open"),
                    "points": [[round(float(x), 6), round(float(y), 6)] for x, y in coords],
                    "centroid": ([round(float(source_centroid[0]), 6), round(float(source_centroid[1]), 6)]
                                 if source_centroid else [round(float(representative.x), 6), round(float(representative.y), 6)]),
                    "label": bool(label_here),
                    "label_text": item.get("label_text", "开放地面"),
                })
    return payload


def _position_fusion_configuration(enabled):
    enabled = bool(enabled)
    return {
        "enabled": enabled,
        "label": ("validation-driven experimental group-centroid position fusion"
                  if enabled else None),
        "model_weight_by_horizon": ({f"{h}s": w for h, w in VALIDATION_POSITION_FUSION_WEIGHTS.items()}
                                     if enabled else None),
        "velocity_source": ("current and exact previous 3-second observable XY group-centroid difference; yaw excluded"
                            if enabled else None),
        "default_changed": False,
    }


def _load_llm_replay_decisions(paths, game_id):
    """Load validated candidate decisions keyed by replay side and issue time."""
    rows = {}
    for path in paths or ():
        for line in Path(path).read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if int(row.get('game_id', -1)) != int(game_id):
                continue
            side = row.get('perspective')
            decision = row.get('contract_decision')
            if side not in ('红', '蓝') or not row.get('contract_valid'):
                continue
            if not isinstance(decision, dict):
                continue
            issued_at = round(float(row.get('issued_at_s', 0.0)), 5)
            rows[(side, issued_at)] = {
                'adapter': Path(str(row.get('adapter') or '')).parent.name,
                'strict_parse_valid': bool(row.get('strict_parse_valid')),
                'contract_repairs': list(row.get('contract_repairs') or []),
                'decision': decision,
            }
    return rows


def build_payload(game_id, max_time, map_path, display_step=1.0, forecast_checkpoint="auto",
                  position_fusion_candidate=False, llm_rows=None):
    con = connect()
    rows = con.execute('SELECT * FROM timeseries WHERE game_id=? ORDER BY "时刻秒"', (game_id,)).fetchall()
    if not rows:
        raise ValueError(f"game {game_id} has no timeseries rows")
    examples, snapshots = build_game_forecast_examples(
        rows, game_id, horizons=HORIZONS, step=3, return_snapshots=True)
    snapshot_maps = {
        side: {round(float(s["issued_at"]), 3): s for s in values}
        for side, values in snapshots.items()
    }
    example_maps = {
        side: {round(float(e["issued_at"]), 3): e for e in values}
        for side, values in examples.items()
    }
    decision_times = sorted(set(example_maps["红"]) & set(example_maps["蓝"]))
    decision_times = [t for t in decision_times if t <= float(max_time)]
    replay_times = sorted(set(snapshot_maps["红"]) & set(snapshot_maps["蓝"]))
    replay_times = [t for t in replay_times if t <= float(max_time)]
    if not decision_times:
        raise ValueError("no decision points in requested time range")
    if display_step < 1 or abs(display_step - round(display_step)) > 1e-9:
        raise ValueError("display_step must be a positive integer number of seconds")
    # Store every source second.  The browser can subsample these frames, but
    # a one-second replay is always available; forecasts are only refreshed
    # on the three-second decision grid and held between decision ticks.
    display_times = replay_times

    runtimes = {side: resolve_temporal_forecaster(
                    forecast_checkpoint, device="cpu", team=side, context=12,
                    adaptive_router="auto", auto_mature_feedback=True,
                    apply_validation_position_fusion=position_fusion_candidate)
                for side in ("红", "蓝")}
    frames = []
    for side in ("红", "蓝"):
        runtimes[side].reset()
    decision_views = {"红": {}, "蓝": {}}
    for t in decision_times:
        for side in ("红", "蓝"):
            snap = snapshot_maps[side][t]
            example = example_maps[side][t]
            # The runtime matures pending horizons from this sensor/radar
            # snapshot before producing the next forecast.
            prediction = runtimes[side].predict(snap)
            raw_entities = list(snap["entities"].values())
            current_entities = [_compact_current(e, raw_entities, side)
                                for e in raw_entities]
            view = {
                "side": side,
                "forecast_issued_at": float(t),
                "decision_updated": True,
                "visible_entities": current_entities,
                "visibility": snap.get("visibility_summary", {}),
                "adaptive_strategy": (prediction or {}).get("adaptive_strategy", {}),
                "horizons": {},
            }
            for horizon in HORIZONS:
                targets = example.get("targets", {})
                target = targets.get(str(horizon), targets.get(horizon))
                truth = target.get("truth") if isinstance(target, dict) and not target.get("censored") else None
                truth_by_slot = _aggregate_truth(truth, side) if truth else {}
                p = prediction or {"horizons": {}}
                hpred = p.get("horizons", {}).get(str(horizon), {})
                entities = _forecast_entities(p, horizon, current_entities, selected_only=True)
                unit_forecasts = _forecast_entities(p, horizon, current_entities, selected_only=False)
                pred_global = _compact_global_prediction(
                    hpred, (p.get("adaptive_strategy") or {}).get("by_task_horizon"),
                    horizon)
                outlook = hpred.get("global_outlook") or {}
                summary = outlook.get("decision_summary") or {}
                pred_global["outlook"] = {
                    "control_trend": summary.get("control_trend"),
                    "priority": summary.get("priority"),
                    "tactical_situation": outlook.get("tactical_situation"),
                    "trigger": summary.get("commitment_rule"),
                    "fallback": summary.get("fallback_rule"),
                    # The replay only renders the first actionable item and
                    # the contested-zone names.  Keep those fields compact
                    # instead of serializing every nested outlook component
                    # into every repeated one-second frame.
                    "contested_zones": ((outlook.get("control") or {}).get(
                        "contested_zones") or [])[:3],
                    "recommended_actions": [
                        {key: value for key, value in (outlook.get("recommended_actions") or [{}])[0].items()
                         if key in ("objective", "trigger", "fallback")}
                    ] if outlook.get("recommended_actions") else [],
                }
                actual_global = target_labels(example, horizon) if truth else {}
                narrative = _replay_global_narrative(
                    side, current_entities, pred_global, horizon, unit_forecasts)
                view["horizons"][str(horizon)] = {
                    "forecast": pred_global,
                    "narrative": narrative,
                    "actual": {
                        "mode": actual_global.get("global_mode"),
                        "trend": actual_global.get("opponent_trend"),
                        "region": actual_global.get("enemy_region"),
                    },
                    "entities": entities,
                    "unit_forecasts": unit_forecasts,
                    "evaluation": _evaluate(entities, truth_by_slot) if truth else None,
                }
            decision_views[side][float(t)] = view

    # Materialise a frame for every source second.  The current observable
    # entities move at 1 Hz, while the forecast/decision view is copied from
    # the most recent 3-second tick.  This makes the hold behaviour explicit
    # and keeps evaluation anchored to the actual forecast issue time.
    for t in display_times:
        frame_views = {}
        decision_time = max((d for d in decision_times if d <= t), default=decision_times[0])
        for side in ("红", "蓝"):
            # Forecasts are immutable between decision ticks.  The compact
            # frame keeps only this second's observations and references the
            # shared decision record below.
            base_view = decision_views[side][decision_time]
            snap = snapshot_maps[side][t]
            raw_entities = list(snap["entities"].values())
            current_entities = [_compact_current(e, raw_entities, side)
                                for e in raw_entities]
            frame_narratives = {}
            shortest_view = base_view.get("horizons", {}).get("10", {})
            current_brief = _replay_global_narrative(
                side, current_entities, shortest_view.get("forecast") or {},
                10, shortest_view.get("unit_forecasts"), include_details=False)
            for horizon in HORIZONS:
                horizon_view = base_view.get("horizons", {}).get(str(horizon), {})
                compact_narrative = (current_brief if horizon == 10 else
                                     _replay_global_narrative(
                                         side, current_entities,
                                         horizon_view.get("forecast") or {},
                                         horizon, horizon_view.get("unit_forecasts"),
                                         include_details=False,
                                         current_description=current_brief["description"]))
                frame_narratives[str(horizon)] = {
                    key: compact_narrative[key]
                    for key in ("description", "prediction_analysis", "recommendation",
                                "adaptive_feedback")
                }
            frame_views[side] = {
                "side": side,
                "visible_entities": current_entities,
                "visibility": snap.get("visibility_summary", {}),
                "narratives": frame_narratives,
                "decision_updated": bool(abs(float(t) - float(decision_time)) < 1e-9),
                "held_decision": not bool(abs(float(t) - float(decision_time)) < 1e-9),
                "current_observation_at": float(t),
            }
        frames.append({"t": float(t), "decision_time": float(decision_time),
                       "decision_updated": bool(abs(float(t) - float(decision_time)) < 1e-9),
                       "held_decision": bool(abs(float(t) - float(decision_time)) >= 1e-9),
                       "views": frame_views})

    # A resized copy keeps the inline replay small enough for the conversation
    # surface while preserving the field-plan aspect ratio.  The coordinate
    # system remains the 1920x1024 SVG viewBox.
    try:
        from PIL import Image
        with Image.open(map_path) as image:
            image = image.convert("RGB")
            image.thumbnail((960, 515))
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=68, optimize=True)
            map_bytes = buffer.getvalue()
        mime = "image/jpeg"
    except Exception:
        mime = mimetypes.guess_type(map_path.name)[0] or "image/jpeg"
        map_bytes = map_path.read_bytes()
    map_data = base64.b64encode(map_bytes).decode("ascii")
    from eval.temporal_consistency import evaluate_replay_frames
    llm_decisions = _load_llm_replay_decisions(llm_rows, game_id)
    for frame in frames:
        issued_at = round(float(frame['decision_time']), 5)
        frame['llm_decisions'] = {
            side: llm_decisions[(side, issued_at)]
            for side in ('红', '蓝')
            if (side, issued_at) in llm_decisions
        }
    payload = {
        "schema": "rm-battle-replay-visual-v1",
        "game_id": int(game_id),
        "field": {"width_m": FIELD_WIDTH_M, "height_m": FIELD_HEIGHT_M,
                   "image_width": 1920, "image_height": 1029},
        "map_data_uri": f"data:{mime};base64,{map_data}",
        "decision_step_seconds": 3,
        "display_step_seconds": display_step,
        "default_display_step_seconds": display_step,
        "horizons": list(HORIZONS),
        "horizon_policy": {},
        "mobile_types": sorted(MOBILE_TYPES),
        "zones": _zone_payload(),
        "forecast_configuration": {
            "position_fusion": _position_fusion_configuration(position_fusion_candidate),
        },
        "frames": frames,
        "llm_decision_rows_loaded": len(llm_decisions),
    }
    diagnostic_path = ROOT / "out_m2" / "horizon_diagnostic.json"
    if diagnostic_path.exists():
        try:
            payload["horizon_policy"] = json.loads(
                diagnostic_path.read_text(encoding="utf-8"))
        except Exception:
            payload["horizon_policy"] = {}
    payload["adaptive_calibration"] = {
        side: {
            "matured_forecast_records": runtimes[side].feedback_status()[
                "completed_forecast_records"],
            "matured_task_horizon_outcomes": runtimes[side].feedback_status()[
                "matured_task_horizon_outcomes"],
            "router_status": (runtimes[side].adaptive_router.status()
                               if runtimes[side].adaptive_router is not None else {}),
            "feedback_contract": "runtime automatically scores each horizon on the first subsequent decision snapshot at or after issue_time + horizon; labels use sensor/radar observations only",
        }
        for side in ("红", "蓝")
    }
    # Reattach shared forecast views only in a short-lived in-memory structure
    # for diagnostics. The serialized frames remain observation-only.
    evaluation_frames = []
    for frame in frames:
        issued = float(frame["decision_time"])
        evaluation_views = {}
        for side, current_view in frame["views"].items():
            merged = dict(decision_views[side][issued])
            merged.update(current_view)
            evaluation_views[side] = merged
        evaluation_frames.append({**frame, "views": evaluation_views})
    payload["temporal_consistency"] = evaluate_replay_frames(
        {**payload, "frames": evaluation_frames})
    return _share_replay_decisions(payload, decision_views)


def _share_replay_decisions(payload, decision_views):
    """Store each three-second forecast once; one-second frames keep observations.

    The forecast is immutable between decision ticks.  Sharing it by index
    avoids serializing the same nested horizons repeatedly in the HTML JSON.
    """
    frames = payload.get("frames") or []
    decision_times = sorted({float(frame.get("decision_time", frame.get("t", 0.0)))
                             for frame in frames})
    decision_index = {time: index for index, time in enumerate(decision_times)}
    decisions = []
    for time in decision_times:
        decisions.append({
            "t": time,
            "views": {
                side: {
                    "forecast_issued_at": view.get("forecast_issued_at", time),
                    "adaptive_strategy": view.get("adaptive_strategy", {}),
                    "horizons": view.get("horizons", {}),
                }
                for side, side_decisions in decision_views.items()
                for view in [side_decisions.get(time, {})]
            },
        })

    for frame in frames:
        time = float(frame.get("decision_time", frame.get("t", 0.0)))
        frame["decision_ref"] = decision_index[time]
        for side, view in list((frame.get("views") or {}).items()):
            visible_entities = view.get("visible_entities", [])
            for entity in visible_entities:
                _compact_browser_guidance(entity)
            frame["views"][side] = {
                key: view[key] for key in (
                    "side", "visible_entities", "visibility", "decision_updated",
                    "held_decision", "current_observation_at", "narratives",
                ) if key in view
            }
    payload["decisions"] = decisions
    return payload


def _compact_browser_guidance(entity):
    """Drop recommendation fields the replay browser never reads.

    This runs only after Python has built the per-frame narrative and scores.
    The user-visible summary, triggers, stop conditions, recommendation and
    every metric rendered by ``metricText`` remain in the compact mapping.
    Raw guidance evidence and duplicate prose stay available during assembly,
    then are omitted from the embedded, repeated one-second observations.
    """
    if not isinstance(entity, dict):
        return entity
    special = entity.get("special_state")
    if not isinstance(special, dict):
        return entity
    guidance = special.get("decision_guidance")
    if not isinstance(guidance, dict):
        return entity

    guidance_fields = (
        "applicable", "unit_ref", "action", "action_label", "display_summary",
        "trigger_condition", "cancel_condition", "action_reason",
        "first_step_zone", "movement_goal_zone", "target_zone",
        "target_unit_id", "target_unit_type", "evaluation", "decision_scorecard",
    )
    compact = {key: guidance[key] for key in guidance_fields if key in guidance}

    evaluation = compact.get("evaluation")
    if isinstance(evaluation, dict):
        compact["evaluation"] = {
            key: evaluation[key] for key in (
                "scorable", "metric", "expected", "target_linkage",
                "expected_target_id",
            )
            if key in evaluation
        }

    scorecard = compact.get("decision_scorecard")
    if isinstance(scorecard, dict):
        score_fields = {
            "feasibility": ("status", "score"),
            "execution": ("status",),
            "consequence": ("status", "metrics"),
            "causal_benefit": ("status",),
            "target_confirmation": ("status",),
        }
        compact["decision_scorecard"] = {
            section: {key: section_data[key] for key in keys if key in section_data}
            for section, keys in score_fields.items()
            if isinstance((section_data := scorecard.get(section)), dict)
        }
    special["decision_guidance"] = compact
    return entity


HTML_PREFIX = r'''<meta charset="utf-8"><div id="rm-battle-replay" aria-label="RM比赛时序回放">
<style>
#rm-battle-replay{--background:#0b1020;--foreground:#edf2ff;--muted:#18233a;--muted-foreground:#a9b5cf;--border:#34415d;--red:#ef5b6b;--blue:#5aa7ff;--green:#48d597;--yellow:#f3c969;--purple:#ba91ff;font:14px/1.26 system-ui,-apple-system,"Microsoft YaHei",sans-serif;color:var(--foreground);background:var(--background);padding:8px;max-width:1760px;width:calc(100% - 16px);margin:0 auto;box-sizing:border-box}
#rm-battle-replay .toolbar{display:flex;flex-wrap:wrap;align-items:center;gap:5px;margin-bottom:5px}
#rm-battle-replay .rm-experiment-banner{margin:6px 0 10px;padding:8px 10px;border:1px solid #c695ff;background:#211638;color:#f0ddff;border-radius:5px;font-size:12px}
#rm-battle-replay button,#rm-battle-replay select,#rm-battle-replay input{font:inherit;color:var(--foreground);background:var(--muted);border:1px solid var(--border);border-radius:5px;padding:4px 6px}
#rm-battle-replay button{cursor:pointer}.rm-label{color:var(--muted-foreground);font-size:12px}.rm-time{font-variant-numeric:tabular-nums;min-width:100px}
#rm-battle-replay .replay-main{display:grid;grid-template-columns:minmax(0,2.25fr) minmax(400px,1fr);gap:6px;align-items:start}
#rm-battle-replay .map-wrap{position:relative;border:1px solid var(--border);background:#080b13;overflow:visible}
#rm-battle-replay svg{display:block;width:100%;height:auto;aspect-ratio:2.05/1}
#rm-battle-replay .zone{fill:rgba(90,167,255,.13);stroke-width:1.5;stroke-dasharray:none;vector-effect:non-scaling-stroke}.zone-label{fill:#f4f7ff;font-size:16px;font-weight:600;text-anchor:middle;pointer-events:none;paint-order:stroke;stroke:rgba(4,9,20,.95);stroke-width:3px;stroke-linejoin:round}.unit-label{fill:#fff;font-size:15px;font-weight:600;text-anchor:middle;dominant-baseline:middle;pointer-events:none;paint-order:stroke;stroke:rgba(3,8,18,.95);stroke-width:3px;stroke-linejoin:round}.unit-box{stroke-width:3}
#rm-battle-replay .team-red{fill:var(--red);stroke:#ffe6e9}.team-blue{fill:var(--blue);stroke:#e5f1ff}.team-destroyed{fill:#7c8494;stroke:#e2e5eb;opacity:.88}.marker{stroke-width:.05}.pred-link{stroke-width:2.4;opacity:.95}.attack-trace{stroke:#ff8c42;stroke-width:3;opacity:.98}.attack-target{fill:none;stroke:#ff8c42;stroke-width:2.6}.state-bubble{fill:rgba(10,15,30,.97);stroke:#ffd166;stroke-width:1.8}.state-bubble-text{fill:#fff4cf;font-size:14px;font-weight:700;text-anchor:middle;dominant-baseline:middle;paint-order:stroke;stroke:rgba(5,9,18,.95);stroke-width:2.5px}.unknown{opacity:.45}.unit-guidance-meta{margin-top:5px;padding:5px 7px;border-left:2px solid #79c7ff;background:#16243a;color:#dceeff;font-size:12px;line-height:1.5}.unit-guidance-meta b{color:#ffd98a}
#rm-battle-replay .unit-marker{cursor:pointer}.unit-marker:focus{outline:none}.unit-marker.is-selected .unit-box{stroke:#fff3a1;stroke-width:6;filter:drop-shadow(0 0 5px rgba(255,230,120,.9))}
#rm-battle-replay .unit-popover{position:absolute;z-index:5;top:10px;right:10px;width:min(285px,calc(100% - 20px));max-height:calc(100% - 20px);overflow:auto;background:rgba(13,20,38,.97);border:1px solid #ffd166;border-radius:6px;padding:10px;box-shadow:0 8px 24px rgba(0,0,0,.38);font-size:12px}.unit-popover[hidden]{display:none}.unit-popover .popover-title{margin:0 26px 7px 0;color:#fff3bf;font-size:14px;font-weight:600}.unit-popover p{margin:4px 0;color:var(--foreground);line-height:1.45}.unit-popover .close{position:absolute;right:7px;top:5px;border:0;background:transparent;padding:2px 5px;color:var(--muted-foreground);font-size:18px;line-height:1;cursor:pointer}.unit-popover .close:hover{color:#fff}.unit-popover .key{color:#a9b5cf}.unit-popover .value{color:#f2f6ff}.unit-popover .forecast{border-top:1px solid rgba(150,170,210,.25);margin-top:7px;padding-top:7px}
#rm-battle-replay .legend{display:flex;flex-wrap:wrap;gap:8px 13px;margin:7px 0;color:#dbe6ff;font-size:12px}.legend span{display:inline-flex;align-items:center;gap:4px}.swatch{width:11px;height:11px;display:inline-block;border:1px solid currentColor}.swatch.red{color:var(--red);background:var(--red)}.swatch.blue{color:var(--blue);background:var(--blue)}
#rm-battle-replay .global-panel,.unit-panel{background:var(--muted);border:1px solid var(--border);padding:6px;min-width:0}.global-panel{max-height:calc(100vw * .45);overflow:auto}.unit-grid{display:grid;grid-template-columns:minmax(0,1.45fr) minmax(230px,.55fr);gap:6px;margin-top:6px}.own-unit-columns{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:5px}.readout h3{font-size:14px;font-weight:500;margin:0 0 3px;color:var(--foreground)}.readout h4{margin:5px 0 2px;line-height:1.15}.readout h4:first-child{margin-top:0}.readout table{width:100%;border-collapse:collapse;font-size:11px;line-height:1.22}.readout td,.readout th{text-align:left;border-bottom:1px solid rgba(150,170,210,.18);padding:2px 3px;vertical-align:top}.readout td,.readout th{overflow-wrap:anywhere}.readout td,.readout td *{color:var(--foreground)}.readout th{color:var(--muted-foreground);font-weight:400}.readout .rm-label{color:#b9c8e7}.readout p{color:var(--foreground)!important;margin:2px 0;line-height:1.25}.readout .summary-detail{margin:2px 0 4px;padding-left:5px;border-left:2px solid rgba(121,199,255,.6);color:#c8d8f2;line-height:1.25}.readout .unit-analysis-line{margin:1px 0;line-height:1.22}.readout .unit-analysis-line b{color:#ffd98a}.metric{font-variant-numeric:tabular-nums;color:var(--green)!important}.bad{color:var(--yellow)!important}.unit-panel .rm-label{margin-bottom:3px}.unit-panel table{table-layout:fixed}.unit-panel th:first-child{width:22%}.unit-panel th:nth-child(2){width:30%}
@media(max-width:1050px){#rm-battle-replay .replay-main{grid-template-columns:minmax(0,1.6fr) minmax(280px,.85fr)}}
@media(max-width:850px){#rm-battle-replay .replay-main{grid-template-columns:1fr}.global-panel{max-height:none}.unit-grid{grid-template-columns:1fr}}
@media(max-width:620px){.own-unit-columns{grid-template-columns:1fr}}
@media(max-width:700px){.readout table{font-size:11px}.unit-panel th:first-child{width:25%}.unit-panel th:nth-child(2){width:30%}}
</style>
<div class="toolbar">
  <button id="rm-play" type="button" aria-pressed="false">播放</button>
  <label class="rm-label" for="rm-frame">帧</label><input id="rm-frame" type="range" min="0" value="0" step="1" style="flex:1;min-width:160px">
  <span id="rm-time" class="rm-time"></span>
  <label class="rm-label" for="rm-step">回放步长</label><select id="rm-step"><option value="1" selected>1秒</option><option value="3">3秒</option><option value="6">6秒</option><option value="12">12秒</option><option value="30">30秒</option><option value="60">60秒</option></select>
  <label class="rm-label" for="rm-interval">每帧停留</label><select id="rm-interval"><option value="10">10ms</option><option value="20" selected>20ms</option><option value="33">33ms</option><option value="50">50ms</option><option value="80">80ms</option><option value="180">180ms</option><option value="350">350ms</option><option value="600">600ms</option><option value="1000">1秒</option></select>
  <label class="rm-label" for="rm-side">视角</label><select id="rm-side"><option value="红">红方</option><option value="蓝">蓝方</option></select>
  <label class="rm-label" for="rm-horizon">预测</label><select id="rm-horizon"><option value="10">+10秒</option><option value="30">+30秒</option><option value="60">+60秒</option><option value="120">+120秒</option></select>
  <label class="rm-label"><input id="rm-zones" type="checkbox" checked> 功能区</label>
</div>
<div id="rm-experiment-banner" class="rm-experiment-banner" hidden></div>
<div class="replay-main"><div class="map-column"><div class="map-wrap"><svg id="rm-map" viewBox="0 0 1920 1029" role="img" aria-label="场地平面图上的红蓝兵种位置、功能区和可能移动方向"><defs><marker id="rm-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="8" markerHeight="8" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#f3c969"></path></marker><marker id="rm-attack-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="9" markerHeight="9" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#ff8c42"></path></marker></defs><image id="rm-map-image" x="0" y="0" width="1920" height="1029" preserveAspectRatio="none"></image><g id="rm-zone-layer"></g><g id="rm-links"></g><g id="rm-attacks"></g><g id="rm-markers"></g></svg><div id="rm-unit-popover" class="unit-popover" role="dialog" aria-live="polite" hidden></div></div></div><section class="global-panel readout"><h3 id="rm-global-title">全局预测与实际</h3><div id="rm-global"></div></section></div>
<div class="legend"><span><i class="swatch red"></i>红方当前</span><span><i class="swatch blue"></i>蓝方当前</span><span>黄色粗箭头：按预测时距和速度显示可达路线</span><span style="color:#ffb36b">橙色粗线：正在攻击及目标</span><span>机器人上方气泡：特殊状态</span><span>细线为区域边界</span></div>
<div class="unit-grid readout"><section class="unit-panel"><h3 id="rm-own-title">我方单位预测与建议</h3><div id="rm-eval-own"></div></section><section class="unit-panel"><h3 id="rm-enemy-title">敌方单位预测</h3><div id="rm-eval-enemy"></div></section></div>
<script>
(async()=>{
const compressed=atob("__DATA_GZIP__"), packed=new Uint8Array(compressed.length);
for(let i=0;i<compressed.length;i++)packed[i]=compressed.charCodeAt(i);
if(typeof DecompressionStream!=="function")throw new Error("该浏览器不支持gzip回放，请使用新版Chrome/Edge");
const jsonText=await new Response(new Blob([packed]).stream().pipeThrough(new DecompressionStream("gzip"))).text();
const DATA=JSON.parse(jsonText);
const root=document.getElementById('rm-battle-replay');
const experiment=DATA.forecast_configuration?.position_fusion;
const MOBILE_TYPES=DATA.mobile_types||[];
const zonesById=new Map();
for(const zone of DATA.zones){const prior=zonesById.get(zone.id);if(!prior||(!prior.label&&zone.label))zonesById.set(zone.id,zone)}
const narrativeEvidenceCache=new WeakMap();
const experimentBanner=document.getElementById('rm-experiment-banner');
if(experiment?.enabled){experimentBanner.hidden=false;experimentBanner.textContent=`验证驱动实验：validation-driven experimental group-centroid position fusion；模型权重 ${Object.entries(experiment.model_weight_by_horizon||{}).map(([h,w])=>`${h} ${Math.round(Number(w)*100)}%`).join(' / ')}；速度来自当前及前3秒可见XY质心差分；不使用yaw作为车头方向。生产默认未改变。`}
const frameInput=document.getElementById('rm-frame'), sideInput=document.getElementById('rm-side'), horizonInput=document.getElementById('rm-horizon');
const playButton=document.getElementById('rm-play'), zoneInput=document.getElementById('rm-zones'), stepInput=document.getElementById('rm-step'), intervalInput=document.getElementById('rm-interval');
const mapImage=document.getElementById('rm-map-image'), zoneLayer=document.getElementById('rm-zone-layer'), linkLayer=document.getElementById('rm-links'), attackLayer=document.getElementById('rm-attacks'), markerLayer=document.getElementById('rm-markers');
const timeLabel=document.getElementById('rm-time'), globalBox=document.getElementById('rm-global'), ownEvalBox=document.getElementById('rm-eval-own'), enemyEvalBox=document.getElementById('rm-eval-enemy'), unitPopover=document.getElementById('rm-unit-popover');
let selectedEntityId=null;
mapImage.setAttribute('href',DATA.map_data_uri);
let frameIndices=[];
function rebuildFrameIndices(preserveTime=true){
  const oldFrame=DATA.frames[frameIndices[Number(frameInput.value)||0]];
  const oldTime=oldFrame?.t;
  const requested=Math.max(1,Number(stepInput.value)||DATA.default_display_step_seconds||1);
  const startTime=Number(DATA.frames[0]?.t||0);
  frameIndices=DATA.frames.map((frame,i)=>({frame,i})).filter(({frame,i})=>
    i===0 || Math.abs((Number(frame.t)-startTime)/requested-Math.round((Number(frame.t)-startTime)/requested))<1e-6
  ).map(({i})=>i);
  frameInput.max=Math.max(0,frameIndices.length-1);
  if(preserveTime&&oldTime!=null){
    let best=0, distance=Infinity;
    frameIndices.forEach((index,i)=>{const d=Math.abs(DATA.frames[index].t-oldTime);if(d<distance){distance=d;best=i}});
    frameInput.value=best;
  } else frameInput.value=0;
}
rebuildFrameIndices(false);
const px=x=>x/DATA.field.width_m*DATA.field.image_width, py=y=>(DATA.field.height_m-y)/DATA.field.height_m*DATA.field.image_height;
const esc=v=>String(v??'—').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const pointString=pts=>pts.map(p=>`${px(p[0])},${py(p[1])}`).join(' ');
const zoneColor=z=>z.side==='red'?'var(--red)':z.side==='blue'?'var(--blue)':z.side==='open'?'var(--muted-foreground)':'var(--purple)';
const typeShape=(entity,cls,r=22)=>{
  const x=entity.position[0],y=entity.position[1],type=entity.type;
  const X=px(x),Y=py(y),label=({英雄:'英雄',工程:'工程',步兵3:'步3',步兵4:'步4',空中:'空中',哨兵:'哨兵'})[type]||type;
  const id=String(entity.entity_id??`${entity.team}:${type}`);
  const selected=selectedEntityId===id?' is-selected':'';
  return `<g class="${cls} unit-marker${selected}" data-entity-id="${esc(id)}" tabindex="0" role="button" aria-label="查看${esc(entity.team||'')}${esc(type)}详细信息"><rect class="unit-box" x="${X-r}" y="${Y-r}" width="${2*r}" height="${2*r}" rx="2"/><text class="unit-label" x="${X}" y="${Y}">${esc(label)}</text></g>`;
};
const bubbleText=(entity)=>{
  if(entity?.life_state==='destroyed')return '阵亡';
  if(entity?.life_state==='unknown'&&entity?.observability==='stale')return '信号失联';
  if(entity?.life_state==='unknown'&&entity?.observability==='absent')return '当前未观测';
  if(entity?.life_state==='unknown')return '存活状态未知';
  const s=entity?.special_state||{};
  if(s.attack_status==='attacking')return '攻击中';
  if(s.attack_status==='possible_attack')return '附近有疑似目标';
  if(s.resource_phase==='exchanging')return '兑换中';
  if(s.resource_phase==='collecting_energy_unit')return '取能量';
  if(s.resource_phase==='returning_to_exchange')return '回兑换';
  if(s.defense_status==='guarding_base')return '守基地';
  if(s.defense_status==='guarding_outpost')return '守前哨';
  if(s.flight_status==='airborne')return '飞行中';
  return '';
};
const stateBubble=(entity)=>{
  const text=bubbleText(entity); if(!text||!entity.position)return '';
  const X=px(entity.position[0]),Y=py(entity.position[1])-31;
  return `<g><rect class="state-bubble" x="${X-36}" y="${Y-16}" width="72" height="28" rx="6"/><text class="state-bubble-text" x="${X}" y="${Y-2}">${esc(text)}</text></g>`;
};
function drawZones(){
  zoneLayer.innerHTML=zoneInput.checked?DATA.zones.map(z=>{const open=z.id==='场地开放区';const color=zoneColor(z);return `<polygon class="zone" points="${pointString(z.points)}" style="stroke:${open?'transparent':color};fill:${open?'transparent':color};fill-opacity:${open?0:.10}"/>${z.label?`<text class="zone-label" x="${px(z.centroid[0])}" y="${py(z.centroid[1])}">${esc(z.label_text||z.id)}</text>`:''}`}).join(''):'';
}
function currentAnchor(view,slot){const live=(view.visible_entities||[]).filter(e=>`${e.role}:${e.type}`===slot&&e.position&&e.life_state!=='destroyed');if(!live.length)return null;return [live.reduce((sum,e)=>sum+e.position[0],0)/live.length,live.reduce((sum,e)=>sum+e.position[1],0)/live.length]}
function roleSummary(balance,side){const value=(balance||{})[side]||{};const by=value.by_type||{};return Object.entries(by).map(([k,v])=>`${k} ${v.present??0}台${v.critical?`，危险${v.critical}`:''}`).join('；')||'暂无可用编组信息'}
const entityIdOf=e=>String(e?.entity_id??`${e?.team||''}:${e?.type||''}`);
const entityDetail=(entity,pred,h,side)=>{
  if(!entity){unitPopover.hidden=true;return}
  const special=entity.special_state||{};
  const state=entity.life_state==='destroyed'?'已阵亡':entity.life_state==='unknown'?'状态未知':'存活';
  const position=entity.position?`(${Number(entity.position[0]).toFixed(1)}, ${Number(entity.position[1]).toFixed(1)})`:'未观测';
  const hp=entity.hp_ratio==null?'未知':`${Math.round(Number(entity.hp_ratio)*100)}%`;
  const speed=entity.speed_mps==null?'未知':`${Number(entity.speed_mps).toFixed(2)} m/s`;
  const behavior=special.attack_status==='attacking'?'正在攻击':special.resource_phase==='collecting_energy_unit'?'采集能量':special.resource_phase==='returning_to_exchange'?'携带资源回兑换':special.resource_phase==='exchanging'?'正在兑换':special.defense_status==='guarding_base'?'守基地':special.defense_status==='guarding_outpost'?'守前哨':'未确认';
  let forecast='暂无单体预测';
  if(pred?.forecast_suppressed)forecast=`${h}秒预测已抑制：${pred.suppression_reason||'当前状态不支持外推'}`;
  else if(pred){
    const route=(pred.canonical_path||[]).join(' → ')||`${pred.from_canonical_zone||pred.from_zone||'当前区域'} → ${pred.to_canonical_zone||pred.to_zone||'区域未知'}`;
    forecast=`${h}秒候选路线：${route}；存在概率${pred.presence==null?'未知':Math.round(Number(pred.presence)*100)+'%'}；只作参考`;
  }
  unitPopover.innerHTML=`<button class="close" type="button" aria-label="关闭详情">×</button><div class="popover-title">${esc(entity.team||'')}${esc(entity.type||'单位')} <span class="key">#${esc(entity.entity_id||'未知')}</span></div><p><span class="key">当前区域：</span><span class="value">${esc(entity.zone||'未知')}</span></p><p><span class="key">坐标：</span><span class="value">${esc(position)}</span>　<span class="key">血量：</span><span class="value">${esc(hp)}</span></p><p><span class="key">速度：</span><span class="value">${esc(speed)}</span>　<span class="key">状态：</span><span class="value">${esc(state)}</span></p><p><span class="key">行为：</span><span class="value">${esc(behavior)}</span>　<span class="key">数据：</span><span class="value">${esc(entity.observability||'未知')}</span></p><div class="forecast"><p><span class="key">${esc(h)}秒单体预测：</span><span class="value">${esc(forecast)}</span></p></div>`;
  unitPopover.hidden=false;
  unitPopover.querySelector('.close').onclick=()=>{selectedEntityId=null;unitPopover.hidden=true;draw()};
};
markerLayer.addEventListener('click',event=>{
  const marker=event.target.closest?.('.unit-marker');
  if(!marker)return;
  selectedEntityId=marker.dataset.entityId;
  draw();
});
markerLayer.addEventListener('keydown',event=>{
  if(event.key!=='Enter'&&event.key!==' ')return;
  const marker=event.target.closest?.('.unit-marker');
  if(!marker)return;
  event.preventDefault();selectedEntityId=marker.dataset.entityId;draw();
});
unitPopover.addEventListener('click',event=>event.stopPropagation());
function draw(){
  const frame=DATA.frames[frameIndices[Number(frameInput.value)||0]]||DATA.frames[0], side=sideInput.value, h=String(horizonInput.value), frameView=frame.views[side], decisionView=DATA.decisions?.[frame.decision_ref]?.views?.[side]||{}, view={...decisionView,...frameView}, horizon=view.horizons[h], llmDecision=frame.llm_decisions?.[side]||null;
  timeLabel.textContent=`t=${frame.t.toFixed(0)}s / 数据1s，决策3s，${frame.held_decision?'沿用上一决策':'本帧更新决策'}，回放步长${Number(stepInput.value)||DATA.default_display_step_seconds}s`;
  const current=view.visible_entities||[];
  const mobile=current.filter(e=>e.position&&['英雄','工程','步兵3','步兵4','空中','哨兵'].includes(e.type));
  if(selectedEntityId&&!mobile.some(e=>entityIdOf(e)===selectedEntityId)){selectedEntityId=null;unitPopover.hidden=true}
  markerLayer.innerHTML=mobile.map(e=>typeShape(e,e.life_state==='destroyed'?'team-destroyed':(e.team==='红'?'team-red':'team-blue'),22)+stateBubble(e)).join('');
  linkLayer.innerHTML='';
  attackLayer.innerHTML=mobile.map(e=>{
    const s=e.special_state||{}, target=e.attack_target;
    if(e.life_state==='destroyed'||s.attack_status!=='attacking'||!target?.position||target.life_state==='destroyed')return '';
    const x1=px(e.position[0]),y1=py(e.position[1]),x2=px(target.position[0]),y2=py(target.position[1]);
    return `<line class="attack-trace" x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" marker-end="url(#rm-attack-arrow)"/><circle class="attack-target" cx="${x2}" cy="${y2}" r="13"/>`;
  }).join('');
  const preds=horizon.entities||{};
  const forecastLinks=[];
  for(const [slot,p] of Object.entries(preds)){
    const currentSlot=current.filter(e=>`${e.role}:${e.type}`===slot);
    if(currentSlot.length&&currentSlot.every(e=>e.life_state==='destroyed'))continue;
    if(p.forecast_suppressed||!p.position||p.presence<.25)continue;
    const liveAnchor=currentAnchor(view,slot);
    if(currentSlot.some(e=>e.life_state==='destroyed')&&!liveAnchor)continue;
    const anchor=liveAnchor||p.current_position;
    const canonicalTarget=p.to_canonical_zone||p.to_zone;
    const targetZone=zonesById.get(canonicalTarget);
    const target=targetZone?.centroid||p.position;
    if(anchor&&target){
      const pathPoints=(p.canonical_path||[]).map(id=>zonesById.get(id)?.centroid).filter(Boolean);
      const points=[anchor,...pathPoints];
      if(!points.length||points[points.length-1][0]!==target[0]||points[points.length-1][1]!==target[1])points.push(target);
      for(let i=0;i<points.length-1;i++){
        const a=points[i],b=points[i+1];
        forecastLinks.push(`<line class="pred-link" stroke="#f3c969" x1="${px(a[0])}" y1="${py(a[1])}" x2="${px(b[0])}" y2="${py(b[1])}" ${i===points.length-2?'marker-end="url(#rm-arrow)"':''}/>`);
      }
    }
  }
  linkLayer.innerHTML=forecastLinks.join('');
  const f=horizon.forecast||{}, a=horizon.actual||{}, visibility=view.visibility||{};
  const outlook=f.outlook||{}, control=outlook.control||{}, contested=(outlook.contested_zones||control.contested_zones||[]).slice(0,3);
  const tactical=outlook.tactical_situation||{};
  const action=outlook.recommended_actions?.[0]||{};
  const ownNow=current.filter(e=>e.team===side&&MOBILE_TYPES.includes(e.type)), enemyNow=current.filter(e=>e.team!==(side==='红'?'红':'蓝')&&MOBILE_TYPES.includes(e.type));
  const phrase=e=>{if(e.life_state==='destroyed')return '已阵亡';if(e.life_state==='unknown'&&e.observability==='stale')return '信号失联';if(e.life_state==='unknown'&&e.observability==='absent')return '当前未观测';if(e.life_state==='unknown')return '存活状态未知';const s=e.special_state||{};if(s.attack_status==='attacking')return '正在攻击';if(s.attack_status==='possible_attack')return '附近有疑似目标（距离推断）';if(s.resource_phase==='returning_to_exchange')return '携带资源回兑换';if(s.resource_phase==='collecting_energy_unit')return '在资源点采集';if(s.resource_phase==='exchanging')return '正在兑换';if(s.defense_status==='guarding_base')return '守基地';if(s.defense_status==='guarding_outpost')return '守前哨';return '状态未确认'};
  const clip=(value,max)=>{const text=String(value??'');return text.length>max?text.slice(0,max-1)+'…':text};
  const summarize=(items)=>clip(items.slice(0,3).map(e=>`${e.type}${e.zone?`在${e.zone}`:''}${phrase(e)==='常规机动/驻守'?'':`（${phrase(e)}）`}`).join('、')||'观测不足',42);
  const ownState=summarize(ownNow), enemyState=summarize(enemyNow);
  const freshOwn=ownNow.filter(e=>e.position&&e.observability==='current');
  const freshEnemy=enemyNow.filter(e=>e.position&&e.observability==='current');
  const currentTactical=(()=>{
    const intruder=freshEnemy.find(e=>String(e.canonical_zone||'').startsWith(side));
    if(intruder)return `纵深拦截：${intruder.type}已在${intruder.zone}`;
    const contestedUnit=freshOwn.find(e=>e.zone&&e.zone!=='场地未知区'&&freshEnemy.some(o=>o.zone===e.zone));
    if(contestedUnit)return `关键区域争夺：双方位于${contestedUnit.zone}`;
    const resourceUnit=freshOwn.find(e=>e.type==='工程'&&['collecting_energy_unit','returning_to_exchange','exchanging'].includes(e.special_state?.resource_phase));
    if(resourceUnit)return `资源运营：工程${phrase(resourceUnit)}`;
    const attacker=freshOwn.find(e=>e.special_state?.attack_status==='attacking');
    if(attacker)return `交战压制：${attacker.type}已确认攻击`;
    return '细分战术态势待连续观测';
  })();
  const specialEvents=[...ownNow,...enemyNow].filter(e=>e.life_state==='destroyed'||phrase(e)!=='常规机动/驻守').slice(0,2).map(e=>`${e.team}${e.type}${phrase(e)}`);
  const fMode=f.mode||'未知', fTrend=f.trend||'未知';
  const modeSource=f.mode_source==='mode_change_xgb_v1'?'30秒XGBoost候选':'时序模型';
  const modeComparison=f.mode_neural_baseline?`；原时序模型判断${f.mode_neural_baseline}`:'';
  const actionText=clip(action.objective||'保持关键区并等待连续观测',46);
  const trigger=clip(action.trigger||'敌方持续推进或设施受击',28);
  const fallback=clip(action.fallback||'退回最近己方可控区',24);
  const modeConfidence=f.mode_confidence==null?'未知':`${Math.round(Number(f.mode_confidence)*100)}%`;
  const trendConfidence=f.trend_confidence==null?'未知':`${Math.round(Number(f.trend_confidence)*100)}%`;
  const confidenceNote=(Number(f.mode_confidence||0)<.45||Number(f.trend_confidence||0)<.45)?'把握偏低，先观察确认':'';
  const matured=frame.t>=Number(view.forecast_issued_at)+Number(h)-1e-6;
  const calibration=matured?[['态势',f.mode,a.mode],['趋势',f.trend,a.trend]].map(([label,pred,actual])=>`${label}${pred||'—'}→${actual||'目标缺失'}${actual?(pred===actual?'✓':'×'):''}`).join('；'):`到 t=${Math.round(Number(view.forecast_issued_at)+Number(h))}s 后核验`;
  const currentFocus=clip(specialEvents.join('、')||contested.join('、')||'暂无明确特殊事件',24);
  const recommendation=clip(`${actionText}；触发：${trigger}；否则：${fallback}`,94);
  const replayNarrative=frameView.narratives?.[h]||horizon.narrative||{};
  const adaptiveFeedback=replayNarrative.adaptive_feedback||'局内反馈状态未知';
  const narrative=horizon.narrative||{};
  let narrativeEvidence=narrativeEvidenceCache.get(narrative);
  if(narrativeEvidence===undefined){narrativeEvidence=JSON.stringify(narrative.details||{},null,2);narrativeEvidenceCache.set(narrative,narrativeEvidence)}
  const uniqueZones=(items)=>[...new Set(items.map(e=>e.zone).filter(Boolean))];
  const ownZoneText=uniqueZones(ownNow).slice(0,4).join('、')||'暂无明确区域';
  const enemyZoneText=uniqueZones(enemyNow).slice(0,4).join('、')||'暂无明确区域';
  const ownActive=ownNow.filter(e=>e.life_state==='active').length;
  const enemyActive=enemyNow.filter(e=>e.life_state==='active').length;
  const ownRisk=ownNow.filter(e=>e.hp_ratio!=null&&Number(e.hp_ratio)<.3).map(e=>e.type);
  const statusDetail=clip(`当前时刻 t=${Math.round(Number(frame.t)||0)}s：我方可见${ownNow.length}台，确认存活${ownActive}台，分布在${ownZoneText}；敌方雷达可见${enemyNow.length}台，确认存活${enemyActive}台，主要在${enemyZoneText}。${currentTactical}。${ownRisk.length?`我方低血量关注：${ownRisk.join('、')}；`:''}${specialEvents.length?`特殊状态：${specialEvents.join('、')}。`:'暂未发现需要单独升级的特殊事件。'}`,360);
  const regionConfidence=f.region_confidence==null?'未知':`${Math.round(Number(f.region_confidence)*100)}%`;
  const regionPart=f.region?`${f.region}（${regionConfidence}）`:'尚未形成稳定区域指向';
  const contestPart=contested.length?`重点关注${contested.join('、')}`:'当前未识别明确争夺区';
  const predictionDetail=clip(`预测来源：${modeSource}；未来${h}秒全局态势为${fMode}（${modeConfidence}），敌方趋势为${fTrend}（${trendConfidence}），区域方向为${regionPart}。${modeComparison||'当前没有可用的模型/持久性分歧说明'}；${contestPart}。${confidenceNote||'当前置信度达到可参考水平，但仍需在目标时刻用新观测核验'}。${matured?'本时距已到核验点，可对照实际态势。':`本时距尚未到达，暂不把预测当作事实；预计在 t=${Math.round(Number(view.forecast_issued_at)+Number(h))}s 复核。`}`,420);
  const actionUnits=ownNow.map(e=>{
    const g=e.special_state?.decision_guidance||{};
    return g.display_summary?.action||g.action||g.action_label?`${e.type}：${g.display_summary?.action||g.action||g.action_label}`:null;
  }).filter(Boolean);
  const actionUnitText=actionUnits.length?clip(actionUnits.slice(0,4).join('；'),150):'暂未形成可直接下发的单体动作';
  const recommendationDetail=clip(`执行顺序：先${actionText}；当前单体建议：${actionUnitText}。触发条件：${trigger}；停止或重算条件：${fallback}，以及区域、血量、目标可见性发生变化。决策依据：${currentTactical}；${currentFocus==='暂无明确特殊事件'?'继续观察连续帧变化。':`优先处理${currentFocus}。`}建议只作为条件化参考，必须以最新可见事实和路线可行性复核后执行。`,420);
  const llmDecisionText=(()=>{
    if(!llmDecision)return '当前决策点没有已验证的微调模型输出。';
    const d=llmDecision.decision||{}, units=Object.entries(d.units||{}).map(([name,item])=>`${name}：${item.move||'无'} / ${item.target||'无'} / ${item.fire?'开火':'不开火'} / 风险${item.risk||'未知'}`).join('；');
    const repair=(llmDecision.contract_repairs||[]).length?`；合同修复${llmDecision.contract_repairs.join('、')}`:'；原始输出无需合同修复';
    return `候选 ${llmDecision.adapter||'未知'}：全局${d.global_mode||'未知'}；${units||'单位字段缺失'}${repair}。该区块展示真实候选输出，不代表策略收益已验证。`;
  })();
  globalBox.innerHTML=`<h4>现状</h4><p>${esc(replayNarrative.description||`现况：我方${ownState}；敌方${enemyState}。`)}</p><div class="summary-detail">${esc(statusDetail)}</div><h4>预测解析</h4><p>${esc(replayNarrative.prediction_analysis||`约${h}秒后：趋势${fTrend||'未知'}；全局态势${fMode||'未知'}，概率见展开明细。`)}</p><div class="summary-detail">${esc(predictionDetail)}</div><h4>整体建议</h4><p>${esc(replayNarrative.recommendation||`${actionText}；满足条件：${trigger}。不满足时：${fallback}。`)}</p><div class="summary-detail">${esc(recommendationDetail)}</div><h4>微调模型决策</h4><div class="summary-detail">${esc(llmDecisionText)}</div><details class="narrative-evidence"><summary>展开现状依据、各预测概率和建议条件</summary><pre>${esc(narrativeEvidence||'当前没有可展开的结构化明细')}</pre></details><div class="rm-label">到点核验：${esc(calibration)}。3秒/10秒反馈是位置与事件代理，不证明建议带来收益。</div><div class="rm-label">${esc(adaptiveFeedback)}</div>`;
  const ev=horizon.evaluation||{};
  const unitPredictions=horizon.unit_forecasts||preds;
  const ownTypes=['英雄','工程','步兵3','步兵4','空中','哨兵'];
  const visibleUnits=current.filter(e=>ownTypes.includes(e.type));
  const countByType=(unit)=>visibleUnits.filter(e=>e.team===unit.team&&e.type===unit.type).length;
  const metricText=(g)=>{
    const scorecard=g?.decision_scorecard||{};
    if(scorecard.feasibility||scorecard.execution||scorecard.consequence){
      const f=scorecard.feasibility||{},x=scorecard.execution||{},o=scorecard.consequence||{};
      const feasibility=f.status==='scored'?`路线可行性${Number(f.score).toFixed(0)}%`:
        f.status==='failed'?'路线不可行':f.status==='not_required'?'无需跨区':'路线未知';
      const execution=x.status==='unknown'?'执行未知（无命令日志）':x.status==='not_applicable'?'无动作':'执行待核验';
      const labels={current_zone_retention:'区域驻留',own_firing_event:'本机发弹记录（仅行为响应）',own_survival:'我方存活',own_hp_change:'我方血量',
        identified_target_hp_change:'目标血量',target_alive:'目标存活',progress_toward_safe_zone:'向安全区靠近',
        resource_arrival:'到达资源区',resource_exchange_completion:'完成兑换',resource_collection:'采集结果',
        defense_zone_occupancy:'防区驻留',facility_damage:'设施损伤'};
      const effects=(o.metrics||[]).map(m=>labels[m]||m).join('、')||'无可用结果指标';
      const consequence=o.status==='pending'?`待3/10秒核对${effects}`:
        o.status==='unknown'?'后果未知':effects;
      const causal=scorecard.causal_benefit?.status==='not_identifiable'?'收益归因未知':'';
      const confirm=o.target_confirmation?.status==='unknown'?'；目标确认未知':'';
      return `${feasibility}；${execution}；${consequence}${confirm}${causal?`；${causal}`:''}`;
    }
    const evaluation=g?.evaluation||{};
    if(evaluation.scorable===false)return '当前缺少可核验的目标或路径';
    const metric=evaluation.metric;
    const expected=evaluation.expected||evaluation.first_step_zone||g?.first_step_zone||g?.target_zone;
    if(metric==='next_zone')return `3秒核对是否向${expected||'首步区域'}靠近，10秒核对是否到达`;
    if(metric==='reaches_target_zone')return `10秒核对是否到达${expected||'目标区域'}`;
    if(metric==='zone_occupancy')return `10秒核对是否仍在${expected||'原区域'}；触发条件出现后重新评估`;
    if(metric==='attack_confirmation_within_3s'||metric==='target_visible_and_attack_state')return '3秒内核对目标是否仍可见、攻击状态是否持续';
    if(metric==='life_state')return '下一次观测核对生命状态';
    return '3秒后核对行动是否执行，10秒后核对目标区域变化';
  };
  const renderUnitRow=unit=>{
    const role=unit.team===side?'self':'opponent';
    const pred=unitPredictions[`${role}:${unit.type}`];
    const g=unit.special_state?.decision_guidance||{};
    const guidanceDisplay=unit.special_state?.guidance_display;
    const dead=unit.life_state==='destroyed';
    const duplicate=countByType(unit)>1;
    const title=`${role==='self'?'我方':'敌方'}${unit.type}${duplicate&&unit.entity_id!=null?` #${unit.entity_id}`:''}`;
    const currentText=`${unit.zone||'位置未知'}，${phrase(unit)}${unit.hp_ratio==null?'':`，血量${Math.round(unit.hp_ratio*100)}%`}`;
    const infoLine=(label,value)=>`<div class="unit-analysis-line"><b>${label}：</b>${esc(value)}</div>`;
    let analysis;
    if(dead)analysis=infoLine('状态','已阵亡；停止未来预测和行动建议');
    else if(role==='opponent'){
      let futureText;
      if(!pred||pred.forecast_suppressed)futureText=`未来${h}秒暂无可用单体预测`;
      else if(duplicate)futureText=`未来${h}秒仅有${unit.type}编组预测，无法确定本机去向`;
      else{
        const route=(pred.canonical_path||[]).join(' → ')||`${pred.from_canonical_zone||pred.from_zone||'当前区域'} → ${pred.to_canonical_zone||pred.to_zone||'区域未知'}`;
        futureText=`未来${h}秒候选路线：${route}；只是可能去向，到时核对`;
      }
      analysis=infoLine('未来预测',futureText);
    }else{
      const compact=g.display_summary||{};
      const movementLabels=['retreat_low_hp','return_resource_to_supply','continue_resource_route'];
      const goal=g.movement_goal_zone||(movementLabels.includes(g.action_label)?g.target_zone:null);
      const step=g.first_step_zone;
      const routeText=goal?`先到${step&&step!==unit.zone?`${step}，再到`:''}${goal}`:'';
      const targetType=g.target_unit_type;
      const targetLink=g.evaluation?.target_linkage;
      const targetSource=targetLink==='explicit_sensor_target'?'传感器指定':
        targetLink==='visible_candidate_only'?'可见候选':'来源未知';
      const targetText=g.action_label==='continue_attack_visible_target'
        ?targetType?`${targetSource}：${targetType}${g.target_unit_id!=null?` #${g.target_unit_id}`:''}`:'攻击对象：未确认':'';
      const objectText=[routeText&&`路线${routeText}`,targetText&&`攻击对象${targetText}`].filter(Boolean).join('；');
      if(g.action_label==='no_action_evidence'||!g.applicable||!g.action){
        analysis=infoLine('建议动作',compact.action||'暂不建议具体移动或攻击')+
          infoLine('理由与触发条件',`${compact.reason||'任务或交战状态未确认'}；重新评估：${compact.reassess_when||g.trigger_condition||'收到明确任务或确认可见目标'}`);
      }else{
        const actionText=[compact.action||g.action,objectText].filter(Boolean).join('；');
        const rationale=[compact.reason||g.action_reason||'依据当前观测',
          g.trigger_condition&&`触发条件：${g.trigger_condition}`].filter(Boolean).join('；');
        analysis=infoLine('建议动作',actionText)+
          infoLine('理由与触发条件',rationale)+
          infoLine('停止/重算',compact.reassess_when||g.cancel_condition||'关键状态发生变化');
      }
    }
    const guidanceDetails=role==='self'&&!dead&&guidanceDisplay?`<details class="unit-guidance-meta"><summary>评分与证据</summary><div><b>${esc(guidanceDisplay.status)}</b>；可行性：${esc(guidanceDisplay.feasibility||'未知')}；执行：${esc(guidanceDisplay.execution||'未知')}；后果：${esc(guidanceDisplay.consequence||'未知')}；${esc(guidanceDisplay.causal_benefit||'收益因果未知')}；触发条件：${esc(guidanceDisplay.trigger)}；停止条件：${esc(guidanceDisplay.stop)}；依据：${esc(guidanceDisplay.evidence)}；核对标准：${esc(metricText(g))}</div></details>`:'';
    return `<tr><td>${esc(title)}</td><td>${esc(currentText)}</td><td>${analysis}${guidanceDetails}</td></tr>`;
  };
  const scoreText=matured?`未来 ${h}s：在场判断正确率 <b class="metric">${ev.presence_accuracy==null?'—':(ev.presence_accuracy*100).toFixed(1)+'%'}</b>　区域命中率 <b class="metric">${ev.zone_accuracy==null?'—':(ev.zone_accuracy*100).toFixed(1)+'%'}</b>　平均位置误差 <b class="metric">${ev.position_mae_m==null?'—':ev.position_mae_m.toFixed(2)+'米'}</b>`:`未来 ${h}s：目标尚未到时，预测评分待核验`;
  const unitTable=rows=>rows.length?`<table><tr><th>单位</th><th>当前状态</th><th>敌方未来预测 / 我方建议动作与理由</th></tr>${rows}</table>`:'<p class="rm-label">当前没有可展示的单位。</p>';
  // Render rows by entity so the two panels stay correct when duplicate unit
  // classes are present.
  const unitRows=visibleUnits.map(unit=>({unit,row:renderUnitRow(unit)}));
  const enemyUnitRows=unitRows.filter(({unit})=>unit.team!==side).map(({row})=>row).join('');
  const ownRows=unitRows.filter(({unit})=>unit.team===side).map(({row})=>row);
  const ownColumn=(rows)=>unitTable(rows.join(''));
  ownEvalBox.innerHTML=`<div class="rm-label">${scoreText}</div><div class="own-unit-columns">${ownColumn(ownRows.slice(0,3))}${ownColumn(ownRows.slice(3,6))}</div>`;
  enemyEvalBox.innerHTML=unitTable(enemyUnitRows);
  const selected=mobile.find(e=>entityIdOf(e)===selectedEntityId);
  entityDetail(selected,selected?unitPredictions[`${selected.role}:${selected.type}`]:null,h,side);
}
let playbackTimer=null;
function tick(){
  playbackTimer=null;
  if(playButton.dataset.playing!=='1')return;
  const v=Number(frameInput.value||0)+1;
  if(v>=frameIndices.length){playButton.dataset.playing='0';playButton.textContent='播放';return}
  frameInput.value=v;draw();
  playbackTimer=setTimeout(tick,Number(intervalInput.value)||20);
}
playButton.onclick=()=>{
  const on=playButton.dataset.playing==='1';
  playButton.dataset.playing=on?'0':'1';playButton.textContent=on?'播放':'暂停';
  if(playbackTimer!==null){clearTimeout(playbackTimer);playbackTimer=null}
  if(!on)tick();
};
for(const el of [frameInput,sideInput,horizonInput])el.addEventListener('input',draw);
zoneInput.addEventListener('change',drawZones);
stepInput.addEventListener('change',()=>{rebuildFrameIndices(true);draw()});
drawZones();draw();
})();
</script>
</div>'''


def _argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game-id", type=int, default=None)
    parser.add_argument("--max-time", type=float, default=420.0)
    parser.add_argument("--display-step", type=float, default=1.0)
    parser.add_argument("--checkpoint", default="auto",
                        help="forecast checkpoint path or auto (runtime default)")
    parser.add_argument("--position-fusion-candidate", action="store_true", default=False,
                        help="enable validation-driven experimental group-centroid position fusion (not the production default)")
    parser.add_argument('--llm-rows', nargs='*', type=Path, default=None,
                        help='validated candidate row JSONL files to show in the replay')
    parser.add_argument("--map", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser


def main():
    args = _argument_parser().parse_args()
    with connect() as con:
        game_id = args.game_id
        if game_id is None:
            game_id = int(con.execute('SELECT game_id FROM matches ORDER BY game_id LIMIT 1').fetchone()[0])
    payload = build_payload(game_id, args.max_time, args.map, display_step=args.display_step,
                            forecast_checkpoint=args.checkpoint,
                            position_fusion_candidate=args.position_fusion_candidate,
                            llm_rows=args.llm_rows)
    raw_data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    packed_data = gzip.compress(raw_data, compresslevel=6, mtime=0)
    encoded_data = base64.b64encode(packed_data).decode("ascii")
    html = HTML_PREFIX.replace("__DATA_GZIP__", encoded_data)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")
    print(json.dumps({"out": str(args.out), "game_id": game_id,
                       "frames": len(payload["frames"]),
                       "llm_decision_rows_loaded": payload["llm_decision_rows_loaded"],
                      "payload_bytes": len(raw_data),
                      "compressed_payload_bytes": len(packed_data),
                      "bytes": args.out.stat().st_size}, ensure_ascii=False))


if __name__ == "__main__":
    main()
