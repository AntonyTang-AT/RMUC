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


class TieredDecider:
    """单局可复用的分层决策器。"""

    def __init__(self, mode_student=None, tier2_backend=None, unit_student=None,
                 team='红', smooth_window=3, thresholds=None, as_decision_request=False):
        self.student = mode_student
        self.tier2 = tier2_backend if tier2_backend is not None else NullTier2()
        # P1 钩子：未加载时单位仍走规则+意图偏置
        self.unit_student = unit_student
        self.team = team
        self.thresholds = thresholds
        self.as_decision_request = bool(as_decision_request)
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
            'n_unit_student': 0,
        }

    def reset_episode(self):
        self.tracker = oi.GameIntentTracker()
        self.prev_w = None
        self.mode_hist = []
        if self.smoother is not None:
            self.smoother.reset()

    def decide(self, dp, bounds, base_decision=None, conflict_rate=0.0):
        """对单决策点产出带 sidecar 的分层决策。"""
        w = wsmod.build_worldstate(dp, bounds)
        opp = oi.infer_opponent_intent(w, self.prev_w, our_team=self.team, tracker=self.tracker)
        stu_conf = None
        raw_mode = None
        if self.student is not None:
            try:
                obs = mf.pack_obs(dp, bounds, team=self.team)
                raw_mode, stu_conf, _probs = self.student.predict_proba(obs)
            except Exception:
                raw_mode, stu_conf = None, None

        route = rt.route_tier(
            w, mode_history=self.mode_hist, our_team=self.team,
            conflict_rate=conflict_rate, student_conf=stu_conf, thresholds=self.thresholds)

        base = base_decision if base_decision is not None else sl.build_empty_decision(dp.get('t', 0.0))
        d_out = base
        mode_source = 'fusion'
        tier2_attempted = False
        tier2_ok = False

        self.stats['n'] += 1
        if route.get('need_llm'):
            self.stats['n_need_llm'] += 1
            tier2_attempted = True
            gen = None
            try:
                gen = self.tier2.generate_one(dp, bounds)
            except Exception:
                gen = None
            if gen is not None and sl.validate(gen)[0]:
                d_out = gen
                mode_source = 'tier2_llm'
                tier2_ok = True
                self.stats['n_tier2_ok'] += 1
            else:
                mode = gm.derive_global_mode(w)
                d_out = mf.refill_units_for_mode(base, w, mode, team=self.team, opp_intent=opp)
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
            d_out = mf.refill_units_for_mode(base, w, mode, team=self.team, opp_intent=opp)
            mode_source = 'tier0_rule'

        # 最终合法性兜底
        if not sl.validate(rt.strip_sidecar_for_validate(d_out))[0]:
            mode = gm.derive_global_mode(w)
            d_out = mf.refill_units_for_mode(
                sl.build_empty_decision(dp.get('t', 0.0)), w, mode,
                team=self.team, opp_intent=opp)
            mode_source = 'illegal_fallback_rule'

        situ = semmod.derive_semantic_situation(w, team=self.team, prev_w=self.prev_w)
        sentry = gm.derive_sentry_state(w, team=self.team)
        strategic = gm.sentry_to_strategic(sentry)
        global_analysis = semmod.derive_global_analysis(
            w, team=self.team, prev_w=self.prev_w,
            mode=d_out.get('global_mode'), opponent_intent=opp,
            sentry_state=sentry)
        attached = rt.attach_decision_meta(
            d_out, route, opponent_intent=opp, semantic_situation=situ,
            sentry_state=sentry, strategic_intent=strategic,
            global_analysis=global_analysis)
        meta = dict(attached.get('decision_meta') or {})
        meta['mode_source'] = mode_source
        meta['tier2_attempted'] = bool(tier2_attempted)
        meta['tier2_ok'] = bool(tier2_ok)
        meta['sentry_state'] = sentry
        meta['strategic_intent'] = strategic
        meta['global_analysis_schema'] = global_analysis.get('schema_version')
        if stu_conf is not None:
            meta['student_conf'] = round(float(stu_conf), 4)
        attached['decision_meta'] = meta

        self.mode_hist.append(attached.get('global_mode', '均衡'))
        if len(self.mode_hist) > 5:
            self.mode_hist = self.mode_hist[-5:]
        self.prev_w = w
        if self.as_decision_request:
            from runtime.decision_request import to_decision_request
            return to_decision_request(attached)
        return attached

    def run_series(self, dps, bounds, base_decisions=None, conflict_rate=0.0):
        """整局序列。base_decisions 与 dps 等长或 None。"""
        self.reset_episode()
        out = []
        n = len(dps)
        for i, dp in enumerate(dps):
            base = None
            if base_decisions is not None and i < len(base_decisions):
                base = base_decisions[i]
            out.append(self.decide(dp, bounds, base_decision=base, conflict_rate=conflict_rate))
        return out
