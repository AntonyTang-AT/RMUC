from tree.priority_behavior_tree import PriorityBehaviorTree


def _worldstate(*, hp=1.0, vuln=0):
    return {'robots': {'s': {'type': '哨兵', 'team': '红', 'hp_ratio': hp, 'vuln': vuln}}}


def test_t0_safety_overrides_t2_attack():
    result = PriorityBehaviorTree().tick(_worldstate(hp=0.1), strategic_state='hitbuilding')
    assert result['state'] == 'def'
    assert result['priority_level'] == 'T0_system_safety'


def test_t1_task_overrides_t2_strategy():
    result = PriorityBehaviorTree().tick(
        _worldstate(), semantic_situation={'base_threat': True}, strategic_state='hitbuilding')
    assert result['state'] == 'def'
    assert result['priority_level'] == 'T1_task_coordination'


def test_t2_strategy_reaches_t3_combat_without_override():
    result = PriorityBehaviorTree().tick(_worldstate(), strategic_state='hitbuilding')
    assert result['state'] == 'hitbuilding'
    assert result['priority_level'] == 'T2_strategic_goal'
    assert result['combat_action']['state'] == 'hitbuilding'
    assert result['blackboard']['t2.sentry_intent'] == 'hitbuilding'
