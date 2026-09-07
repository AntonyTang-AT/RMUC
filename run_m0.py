# -*- coding: utf-8 -*-
import os, argparse, json
import config
from data import db, features
from worldstate import zone, worldstate as ws
from labels import sentry_labels as sl
from eval import viz

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--game', type=int, default=config.GAME_REF)
    ap.add_argument('--out', type=str, default='out_m0')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    con = db.connect()
    rows = db.timeseries_for(con, args.game)
    meta = db.match_meta(con, args.game)
    con.close()
    dps = features.build_decision_points(rows, args.game)
    b = features.field_bounds(dps)
    series = sl.label_sentry_series(dps, b, team='红', sentry_rid=7)
    strip = sl.smooth_sequence(series, min_dur=5.0)
    viz.plot_state_strip([t for t, _ in strip], [l for _, l in strip],
                         os.path.join(args.out, 'sentry_strip.png'))
    xs = [r['x'] for dp in dps for r in dp['robots'].values() if r['type'] not in features.BUILDING_TYPES]
    ys = [r['y'] for dp in dps for r in dp['robots'].values() if r['type'] not in features.BUILDING_TYPES]
    viz.plot_heatmap(xs, ys, os.path.join(args.out, 'heatmap.png'))
    # 统计
    counts = {k: 0 for k in sl.SENTRY_STATES}
    for _, l in strip: counts[l] += 1
    w0 = ws.build_worldstate(dps[0], b)
    report = {
        'game': args.game, 'meta': dict(meta) if meta else None,
        'decision_points': len(dps), 'zone_bounds': b,
        'sentry_label_counts': counts,
        'sample_worldstate': w0,
    }
    with open(os.path.join(args.out, 'report.txt'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print(f"[M0] wrote {args.out}/sentry_strip.png, heatmap.png, report.txt")
    print('sentry label counts:', counts)

if __name__ == '__main__':
    main()
