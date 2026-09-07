# fastloop/behavior.py
from dataclasses import dataclass

SENTRY_STATES = ('def', 'hitbuilding', 'cruise')
SAFE_FALLBACK = 'def'

@dataclass
class Rules:
    min_hp_to_attack: float = 0.2
    allow_overheating_attack: bool = False

def _sentry(snapshot):
    return snapshot.get('robots', {}).get(7)

def feasible(proposed, snapshot, rules):
    s = _sentry(snapshot)
    if s is None:
        return proposed in SENTRY_STATES
    hp = s.get('hp_ratio', 1.0)
    vuln = s.get('vuln', 0)
    if proposed == 'hitbuilding':
        if hp < rules.min_hp_to_attack:
            return False
        if vuln:
            return False
    return proposed in SENTRY_STATES

def decide_fast(snapshot, rules):
    proposed = snapshot.get('bot_state', 'cruise')
    if not feasible(proposed, snapshot, rules):
        proposed = SAFE_FALLBACK
    s = _sentry(snapshot) or {}
    return {
        'state': proposed,
        'zone': s.get('zone', 'Z--'),
        'target': '敌方前哨站' if proposed == 'hitbuilding' else ('无' if proposed == 'cruise' else '敌方单位'),
    }
