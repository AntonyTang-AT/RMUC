# model/prompt.py
import json
VALID_STATES = ('def', 'hitbuilding', 'cruise')

SYSTEM = (
    "你是 RoboMaster 比赛局势分析专家。给定全场 WorldState(JSON)，请判断己方哨兵的宏观决策——只输出一个状态："
    "def(守基地) / hitbuilding(攻击敌方建筑) / cruise(巡逻)。"
    "严格输出 JSON：{\"state\": \"<def|hitbuilding|cruise>\"}。仅输出 JSON，不要多余文字。"
)

def build_prompt(worldstate, few_shot=None):
    parts = [SYSTEM]
    if few_shot:
        parts.append("示例：")
        for ws, lbl in few_shot:
            parts.append(f"输入: {json.dumps(ws, ensure_ascii=False)}\n输出: {{\"state\": \"{lbl}\"}}")
    parts.append(f"当前: {json.dumps(worldstate, ensure_ascii=False)}")
    return "\n\n".join(parts)

def parse_decision(text):
    try:
        data = json.loads(text)
        s = str(data.get('state', '')).strip().lower()
        if s in VALID_STATES:
            return {'state': s}
    except Exception:
        pass
    return {'state': 'cruise'}
