import json

from eval.build_visible_temporal_context_dataset import _forecast_context, _group_source_rows


def _row(t):
    return {
        'instruction': ('局势: t=' + str(t) + ' | 预测参考信息：' + json.dumps({
            'schema_version': 'forecast-context-v1', 'predictions': []}, ensure_ascii=False)),
        'completion': '{"global_mode":"均衡"}',
    }


def test_source_rows_partition_on_monotonic_timestamp_reset():
    groups = _group_source_rows([_row(1), _row(4), _row(1), _row(4)], [11, 12])
    assert [len(groups[game]) for game in (11, 12)] == [2, 2]


def test_forecast_context_extracts_only_the_embedded_json_object():
    context = _forecast_context(_row(1)['instruction'] + ' trailing text')
    assert context['schema_version'] == 'forecast-context-v1'
