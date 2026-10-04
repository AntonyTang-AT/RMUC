# Detailed posture coverage audit

- Validation unit: fixed first-eight-game exploratory sample, red and blue radar view.
- Sampling: every 15 seconds.
- Interpretation: output coverage only; no ground-truth strategy label exists.

## Rule coverage

- Non-fallback: 447/448 (99.8%)
- Fallback: 1/448 (0.2%)

## Counts and coarse-mode cross-tab

| Detailed posture code | Count | Defensive | Balanced | Offensive |
|---|---:|---:|---:|---:|
| deep_interception | 245 | 168 | 59 | 18 |
| contested_zone | 131 | 0 | 45 | 86 |
| outpost_defense | 47 | 0 | 8 | 39 |
| frontline_delay | 9 | 0 | 5 | 4 |
| protect_vulnerable_units | 7 | 0 | 0 | 7 |
| focused_pressure | 5 | 0 | 0 | 5 |
| controlled_pursuit | 3 | 0 | 0 | 3 |
| formation_and_patrol | 1 | 0 | 1 | 0 |

The cross-tab is descriptive consistency information, not a confusion matrix against truth. It does not establish which posture is tactically correct.
