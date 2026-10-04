# -*- coding: utf-8 -*-
"""真分层决策执行器（模块级无 torch）。

热路径: Tier0 规则 / Tier1 mode 学生(+可选 unit_student 钩子) / Tier2 LoRA。
无 CUDA 时 Tier2 后端为 Null → 规则回退，只写 meta。

闭环评测（Physics/M2Env 对比规则 vs 分层）接入点预留，本阶段不实现大战。
未来 TemporalModeStudent 可替换 ModeSmoother（见 rl.mode_student.ModeSmoother）。
"""
from __future__ import annotations

from schema import layered as sl
from worldstate import worldstate as wsmod
from worldstate import semantic as semmod
from tree import global_mode as gm
from labels import opp_intent as oi
from labels import router as rt
from rl import mode_features as mf
from rl.mode_student import ModeSmoother
from runtime.tier2 import NullTier2
from runtime.decision_guard import audit_tier2_decision
from worldstate.forecast_context import build_forecast_context
from tree.sentry_state_machine import SentryStateMachine
from tree.priority_behavior_tree import PriorityBehaviorTree


_UNSET_TEMPORAL_FORECAST = object()


def _project_tier2_to_visible_state(decision, visible_input):
    """Enforce the visible-slot contract without changing live-unit intent."""
    if not isinstance(decision, dict) or not isinstance(visible_input, dict):
        return decision, []
    from eval.llm_visible_input import project_decision_to_observable_contract
    projected = project_decision_to_observable_contract(decision, visible_input)
    before = decision.get('units') or {}
    after = (projected or {}).get('units') or {}
    overridden = [unit_type for unit_type in sl.UNIT_CLASSES
                  if before.get(unit_type) != after.get(unit_type)]
    return projected, overridden


class TieredDecider:
    """单局可复用的分层决策器。"""

    def __init__(self, mode_student=None, tier2_backend=None, unit_student=None,
                 team='红', smooth_window=3, thresholds=None, as_decision_request=False,
                 temporal_forecaster=None, temporal_mode_threshold=0.7,
                 tier2_repair_rate_limit=0.05, tier2_repair_min_samples=20,
                 sentry_state_machine=None, sentry_behavior_tree=None,
                 executor_arbiter=None):
        self.student = mode_student
        self.tier2 = tier2_backend if tier2_backend is not None else NullTier2()
        # P1 钩子：未加载时单位仍走规则+意图偏置
        self.unit_student = unit_student
        self.team = team
        self.thresholds = thresholds
        self.as_decision_request = bool(as_decision_request)
        self.temporal_forecaster = temporal_forecaster
        self.temporal_mode_threshold = float(temporal_mode_threshold)
        self.tier2_repair_rate_limit = float(tier2_repair_rate_limit)
        self.tier2_repair_min_samples = max(1, int(tier2_repair_min_samples))
        self.sentry_state_machine = sentry_state_machine or SentryStateMachine()
        self.sentry_behavior_tree = sentry_behavior_tree or PriorityBehaviorTree(
            rules=self.sentry_state_machine.rules)
        self.executor_arbiter = executor_arbiter
        self._tier2_repair_circuit_open = False
        self.smoother = ModeSmoother(window=smooth_window) if mode_student is not None else None
        self.tracker = oi.GameIntentTracker()
        self.prev_w = None
        self.mode_hist = []
        self.stats = {
            'n': 0,
            'n_need_llm': 0,
            'n_mode_student': 0,
            'n_rule_fallback': 0,
            'n_tier2_ok': 0,
            'n_tier2_fallback': 0,
            'n_tier2_guard_reject': 0,
            'n_tier2_contract_repaired': 0,
            'n_tier2_repair_observed': 0,
            'n_tier2_repair_budget_fallback': 0,
            'n_unit_student': 0,
            'n_temporal_forecast': 0,
            'n_temporal_mode_hint': 0,
        }

    def reset_episode(self):
        self.tracker = oi.GameIntentTracker()
        self.prev_w = None
        self.mode_hist = []
        if self.smoother is not None:
            self.smoother.reset()
        if self.temporal_forecaster is not None and hasattr(self.temporal_forecaster, 'reset'):
            self.temporal_forecaster.reset()
        if hasattr(self.sentry_state_machine, 'reset'):
            self.sentry_state_machine.reset()

    def _record_tier2_generation(self, generation):
        """Track narrow parser repairs and open a conservative circuit breaker."""
        if not isinstance(generation, dict):
            return False
        self.stats['n_tier2_repair_observed'] += 1
        repairs = list(generation.get('contract_repairs') or [])
        if repairs:
            self.stats['n_tier2_contract_repaired'] += 1
        observed = self.stats['n_tier2_repair_observed']
        repaired = self.stats['n_tier2_contract_repaired']
        rate = repaired / max(1, observed)
        if (observed >= self.tier2_repair_min_samples and
                rate > self.tier2_repair_rate_limit):
            self._tier2_repair_circuit_open = True
        return self._tier2_repair_circuit_open

    def _tier2_repair_monitor(self):
        observed = self.stats['n_tier2_repair_observed']
        repaired = self.stats['n_tier2_contract_repaired']
        return {
            'observed': int(observed),
            'repaired': int(repaired),
            'repair_rate': repaired / max(1, observed),
            'repair_rate_limit': self.tier2_repair_rate_limit,
            'min_samples': self.tier2_repair_min_samples,
            'circuit_open': bool(self._tier2_repair_circuit_open),
        }

    @staticmethod
    def _copy_decision_core(decision):
        """Copy contract fields for arbitration evidence without mutable aliases."""
        return {
            't': decision.get('t'),
            'global_mode': decision.get('global_mode'),
            'units': {name: dict(action or {})
                      for name, action in (decision.get('units') or {}).items()},
        }

    def _apply_behavior_priority(self, decision, worldstate, sentry_behavior, opponent_intent):
        """Make T0/T1 safety and task branches authoritative over T2 output."""
        priority = sentry_behavior.get('priority_level')
        if priority not in ('T0_system_safety', 'T1_task_coordination'):
            return decision, False
        before = self._copy_decision_core(decision)
        enforced = mf.refill_units_for_mode(
            decision, worldstate, '防守', team=self.team,
            opp_intent=opponent_intent)
        sentry_action = dict(enforced.get('units', {}).get('哨兵') or {})
        combat_target = sentry_behavior.get('combat_action', {}).get('target')
        if combat_target in sl.TARGETS:
            sentry_action['target'] = combat_target
        sentry_action['move'] = '回防'
        sentry_action['fire'] = (False if priority == 'T0_system_safety'
                                  else bool(sentry_action.get('fire', False)))
        sentry_action['risk'] = ('高' if priority == 'T0_system_safety'
                                 else sentry_action.get('risk', '中'))
        enforced.setdefault('units', {})['哨兵'] = sentry_action
        changed = before != self._copy_decision_core(enforced)
        return enforced, changed

    def decide(self, dp, bounds, base_decision=None, conflict_rate=0.0,
               observable_snapshot=None,
               temporal_forecast_override=_UNSET_TEMPORAL_FORECAST):
        """对单决策点产出带 sidecar 的分层决策。"""
        w = wsmod.build_worldstate(dp, bounds)
        opp = oi.infer_opponent_intent(w, self.prev_w, our_team=self.team, tracker=self.tracker)
        temporal_forecast = None
        forecast_context = None
        if temporal_forecast_override is not _UNSET_TEMPORAL_FORECAST:
            # A caller may prewarm the forecaster with every 1-second causal
            # snapshot and provide only the forecast belonging to this
            # decision timestamp.  This keeps Tier2 deterministic while
            # preserving the complete issue-time history.
            temporal_forecast = temporal_forecast_override
            if temporal_forecast is not None:
                self.stats['n_temporal_forecast'] += 1
                forecast_context = build_forecast_context(temporal_forecast)
        elif self.temporal_forecaster is not None and observable_snapshot is not None:
            try:
                temporal_forecast = self.temporal_forecaster.predict(observable_snapshot)
                if temporal_forecast is not None:
                    self.stats['n_temporal_forecast'] += 1
                    forecast_context = build_forecast_context(temporal_forecast)
            except Exception:
                # Forecasting is an enhancement; decision legality and safety
                # must not depend on a failed optional model.
                temporal_forecast = None
        stu_conf = None
        raw_mode = None
        if self.student is not None:
            try:
                obs = mf.pack_obs(dp, bounds, team=self.team)
                raw_mode, stu_conf, _probs = self.student.predict_proba(obs)
            except Exception:
                raw_mode, stu_conf = None, None

        situ = semmod.derive_semantic_situation(w, team=self.team, prev_w=self.prev_w)
        proposed_sentry = gm.derive_sentry_state(w, team=self.team)
        sentry_transition = self.sentry_state_machine.transition(
            w, team=self.team, semantic_situation=situ, proposed_state=proposed_sentry)
        sentry_behavior = self.sentry_behavior_tree.tick(
            w, team=self.team, semantic_situation=situ,
            strategic_state=sentry_transition.state)

        route = rt.route_tier(
            w, mode_history=self.mode_hist, our_team=self.team,
            conflict_rate=conflict_rate, student_conf=stu_conf, thresholds=self.thresholds)

        base = base_decision if base_decision is not None else sl.build_empty_decision(dp.get('t', 0.0))
        d_out = base
        mode_source = 'fusion'
        tier2_attempted = False
        tier2_ok = False
        tier2_generation = None
        tier2_observable_projection = []
        tier2_repair_budget_blocked = False

        self.stats['n'] += 1
        if route.get('need_llm'):
            self.stats['n_need_llm'] += 1
            tier2_attempted = not self._tier2_repair_circuit_open
            gen = None
            tier2_visible_input = None
            try:
                # A Tier2 model must not receive the complete replay decision
                # point.  The observable snapshot is the deployable current
                # view; if it is absent or fails validation, fail closed to
                # the deterministic rule path below.
                if (tier2_attempted and observable_snapshot is not None and
                        hasattr(self.tier2, 'generate_visible_one')):
                    from eval.llm_visible_input import build_visible_llm_input
                    tier2_visible_input = build_visible_llm_input(
                        [observable_snapshot], game_id=int(dp.get('game_id', 0)),
                        side=self.team, issued_at_s=float(dp.get('t', 0.0)),
                        history_frames=1)
                    gen = self.tier2.generate_visible_one(
                        tier2_visible_input, forecast_context=forecast_context)
                elif tier2_attempted and not hasattr(self.tier2, 'generate_visible_one'):
                    # Compatibility only for external legacy backends.  The
                    # bundled LazyLoraTier2 always implements the visible
                    # interface, so a deployable adapter cannot enter here.
                    # Keep this narrow branch for pre-existing integrations
                    # and mark it non-deployable in decision metadata below.
                    gen = self.tier2.generate_one(
                        dp, bounds, forecast_context=forecast_context)
            except Exception:
                gen = None
            if isinstance(gen, dict):
                candidate_generation = gen.get('_tier2_generation')
                if isinstance(candidate_generation, dict):
                    tier2_generation = dict(candidate_generation)
                    gen = rt.strip_sidecar_for_validate(gen)
                    tier2_repair_budget_blocked = self._record_tier2_generation(
                        tier2_generation)
                    if tier2_repair_budget_blocked:
                        gen = None
                if gen is not None and tier2_visible_input is not None:
                    gen, tier2_observable_projection = _project_tier2_to_visible_state(
                        gen, tier2_visible_input)
            guard = None
            if gen is not None and sl.validate(gen)[0]:
                guard = audit_tier2_decision(
                    gen, (tier2_visible_input or {}).get('issue_state') or w,
                    forecast_context, our_team=self.team)
            if gen is not None and sl.validate(gen)[0] and guard and guard.get('accepted'):
                d_out = gen
                mode_source = 'tier2_llm'
                tier2_ok = True
                self.stats['n_tier2_ok'] += 1
            else:
                if guard is not None and not guard.get('accepted'):
                    self.stats['n_tier2_guard_reject'] += 1
                mode = gm.derive_global_mode(w)
                d_out = mf.refill_units_for_mode(base, w, mode, team=self.team, opp_intent=opp)
                if tier2_repair_budget_blocked or self._tier2_repair_circuit_open:
                    mode_source = 'tier2_repair_budget_fallback'
                    self.stats['n_tier2_repair_budget_fallback'] += 1
                else:
                    mode_source = 'tier2_fallback_rule'
                self.stats['n_tier2_fallback'] += 1
        elif self.student is not None and raw_mode is not None:
            if route.get('prefer_rule'):
                mode = gm.derive_global_mode(w)
                mode_source = 'tier0_rule_fallback'
                self.stats['n_rule_fallback'] += 1
            else:
                mode = self.smoother.push(raw_mode)
                mode_source = 'tier1_student_smooth'
                self.stats['n_mode_student'] += 1
            d_out = mf.refill_units_for_mode(base, w, mode, team=self.team, opp_intent=opp)
            # P1: 可选单位学生覆盖（未加载则跳过）
            if self.unit_student is not None:
                try:
                    d_out = self.unit_student.apply(d_out, dp, bounds, w, opp_intent=opp)
                    self.stats['n_unit_student'] += 1
                    mode_source = mode_source + '+unit_student'
                except Exception:
                    pass
        else:
            # 无学生：规则 mode + 意图偏置填单位
            mode = base.get('global_mode') or gm.derive_global_mode(w)
            hint = ((temporal_forecast or {}).get('horizons') or {}).get('10', {})
            hint_mode = ((hint.get('global_mode') or {}).get('top'))
            hint_conf = float((hint.get('global_mode') or {}).get('confidence') or 0.0)
            if hint_mode in sl.GLOBAL_MODES and hint_conf >= self.temporal_mode_threshold:
                mode = hint_mode
                mode_source = 'tier0_temporal_forecast'
                self.stats['n_temporal_mode_hint'] += 1
            d_out = mf.refill_units_for_mode(base, w, mode, team=self.team, opp_intent=opp)
            if mode_source == 'fusion':
                mode_source = 'tier0_rule'

        # 最终合法性兜底
        if not sl.validate(rt.strip_sidecar_for_validate(d_out))[0]:
            mode = gm.derive_global_mode(w)
            d_out = mf.refill_units_for_mode(
                sl.build_empty_decision(dp.get('t', 0.0)), w, mode,
                team=self.team, opp_intent=opp)
            mode_source = 'illegal_fallback_rule'

        pre_behavior_arbitration = self._copy_decision_core(d_out)
        d_out, behavior_arbitrated = self._apply_behavior_priority(
            d_out, w, sentry_behavior, opp)
        behavior_arbitration = {
            'applied': bool(behavior_arbitrated),
            'priority_level': sentry_behavior['priority_level'],
            'reason': sentry_behavior['reason'],
            'before': pre_behavior_arbitration,
            'after': self._copy_decision_core(d_out),
        }
        sentry = sentry_behavior['state']
        strategic = gm.sentry_to_strategic(sentry)
        global_analysis = semmod.derive_global_analysis(
            w, team=self.team, prev_w=self.prev_w,
            mode=d_out.get('global_mode'), opponent_intent=opp,
            sentry_state=sentry)
        if temporal_forecast is not None:
            if forecast_context is not None:
                global_analysis['forecast_context'] = forecast_context
            global_analysis['temporal_forecast_summary'] = {
                str(horizon): {
                    'mode': (((temporal_forecast.get('horizons') or {}).get(str(horizon), {})
                              .get('global_mode') or {}).get('top')),
                    'confidence': (((temporal_forecast.get('horizons') or {}).get(str(horizon), {})
                                    .get('global_mode') or {}).get('confidence')),
                }
                for horizon in (10, 30, 60, 120)
            }
            global_analysis['temporal_outlook'] = {
                str(horizon): (((temporal_forecast.get('horizons') or {})
                                .get(str(horizon), {})).get('global_outlook'))
                for horizon in (10, 30, 60, 120)
            }
            # Rebuild the narrative after adding the temporal outlook so the
            # prose can explain likely control and threats without printing
            # every coordinate or probability table.
            from worldstate.narrative import build_global_narrative
            global_analysis['narrative'] = build_global_narrative(global_analysis)
        attached = rt.attach_decision_meta(
            d_out, route, opponent_intent=opp, semantic_situation=situ,
            sentry_state=sentry, strategic_intent=strategic,
            global_analysis=global_analysis, temporal_forecast=temporal_forecast)
        meta = dict(attached.get('decision_meta') or {})
        meta['mode_source'] = mode_source
        meta['tier2_attempted'] = bool(tier2_attempted)
        meta['tier2_ok'] = bool(tier2_ok)
        meta['sentry_state'] = sentry
        meta['strategic_intent'] = strategic
        meta['sentry_state_machine'] = sentry_transition.as_dict()
        meta['sentry_behavior_tree'] = {
            'priority_level': sentry_behavior['priority_level'],
            'reason': sentry_behavior['reason'],
            'combat_action': dict(sentry_behavior['combat_action']),
        }
        meta['behavior_arbitration'] = behavior_arbitration
        meta['global_analysis_schema'] = global_analysis.get('schema_version')
        meta['temporal_forecast_attached'] = temporal_forecast is not None
        meta['forecast_context_attached'] = forecast_context is not None
        if tier2_attempted:
            meta['tier2_input_contract'] = (
                'rmuc_2026_referee_radar_protocol_v1'
                if tier2_visible_input is not None else
                ('legacy_backend_compatibility_not_deployable'
                 if gen is not None else 'missing_visible_input_rule_fallback'))
            meta['tier2_guard'] = guard or {
                'accepted': False,
                'reasons': ['generation_or_schema_failure'],
                'forecast_disagreements': [],
            }
            if tier2_generation is not None:
                meta['tier2_generation'] = {
                    'strict_parse_valid': bool(tier2_generation.get('strict_parse_valid')),
                    'contract_repairs': list(tier2_generation.get('contract_repairs') or []),
                    'parse_error': tier2_generation.get('parse_error'),
                    'contract_error': tier2_generation.get('contract_error'),
                }
            if tier2_observable_projection:
                meta['tier2_observable_projection'] = list(tier2_observable_projection)
            meta['tier2_repair_monitor'] = self._tier2_repair_monitor()
        elif route.get('need_llm') and self._tier2_repair_circuit_open:
            meta['tier2_repair_monitor'] = self._tier2_repair_monitor()
        if stu_conf is not None:
            meta['student_conf'] = round(float(stu_conf), 4)
        attached['decision_meta'] = meta

        self.mode_hist.append(attached.get('global_mode', '均衡'))
        if len(self.mode_hist) > 5:
            self.mode_hist = self.mode_hist[-5:]
        self.prev_w = w
        if self.as_decision_request:
            from runtime.decision_request import to_decision_request
            request = to_decision_request(attached)
            if self.executor_arbiter is not None:
                receipt = self.executor_arbiter.dispatch(request)
                request_meta = dict(request.get('decision_meta') or {})
                request_meta['executor_dispatch'] = receipt
                request['decision_meta'] = request_meta
            return request
        return attached

    def run_series(self, dps, bounds, base_decisions=None, conflict_rate=0.0,
                   observable_snapshots=None, temporal_forecasts=None):
        """整局序列。base_decisions 与 dps 等长或 None。"""
        self.reset_episode()
        out = []
        n = len(dps)
        for i, dp in enumerate(dps):
            base = None
            if base_decisions is not None and i < len(base_decisions):
                base = base_decisions[i]
            snapshot = None
            if observable_snapshots is not None and i < len(observable_snapshots):
                snapshot = observable_snapshots[i]
            precomputed = _UNSET_TEMPORAL_FORECAST
            if temporal_forecasts is not None and i < len(temporal_forecasts):
                if temporal_forecasts[i] is not None:
                    precomputed = temporal_forecasts[i]
            out.append(self.decide(dp, bounds, base_decision=base,
                                   conflict_rate=conflict_rate,
                                   observable_snapshot=snapshot,
                                   temporal_forecast_override=precomputed))
        return out
