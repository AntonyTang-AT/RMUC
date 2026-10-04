# Decision Quality Metric

## Primary Metric

The project now uses **end-to-end Hindsight Decision Quality (HDQ)** as the
primary offline metric for comparing LLM decision candidates:

`end-to-end HDQ = 0` for an unparseable/schema-invalid output; otherwise
`0.4 * global-mode accuracy + 0.6 * mean unit-field accuracy`.

The global-mode label is generated at issue time only for evaluation by the
existing side-specific `forward_strategy` look-ahead contract. The unit-field
score averages exact matches for `target`, `move`, `fire`, and `risk` over
uniquely attributable friendly unit classes. A schema-invalid output scores
zero on the structural gate and is excluded from semantic denominators.

HDQ is an operational-label score, not a causal claim. It answers: *did the
candidate reproduce the declared short-horizon strategic label and the unit
intent derived from it?* A separate conditional HDQ on parsed rows diagnoses
semantic fidelity without allowing parse failures to disappear. Both must be
reported beside parse rate, legal rate, and per-scenario coverage.

The acceptance comparison is the **paired end-to-end HDQ delta versus the
current-state Tier 0 rule** on identical issue records. The report groups rows
by match, averages red/blue within a match when both are present, and uses a
complete-match bootstrap interval. A candidate must have all four structural
scenarios valid before a positive semantic delta can be used for promotion.

The `0.4 / 0.6` split is the declared primary measure because a decision must
both choose the strategic mode and turn it into per-unit intent. Each report
also emits a non-gating sensitivity panel with global-mode weights of `0.2`,
`0.4`, `0.6`, and `0.8`. A promotion claim is therefore accompanied by the
direction of the candidate-vs-rule delta across plausible emphasis choices,
instead of relying on one unexamined weighting convention.

## Secondary Metric

**Observable Response Proxy (ORP)** uses a fixed 10-second window. It scores
movement compatibility with the emitted `move` field and logged firing events
when `fire=true`. Missing telemetry is unknown. ORP is an observational
compatibility diagnostic, not proof that a controller received or executed the
recommendation.

Both metrics use complete-game bootstrap intervals and average red/blue within
each game before averaging games. This prevents dense games or one perspective
from dominating the result.

## Label-Stability Diagnostic

The scorecard also reports a stable-label HDQ slice: an issue row is stable
when its offline look-ahead global-mode label remains unchanged for the next
five decision frames. The full end-to-end HDQ remains the promotion metric;
the stable-label slice is diagnostic evidence for whether errors concentrate
around short-lived teacher-label transitions. It must never replace full HDQ,
because excluding transitions can make a candidate appear better than it is.

## Runtime Reliability Separation

The evaluator records two deliberately different structural quantities:

- **Strict generation rate**: the raw model completion already satisfies the
  closed layered schema without alteration.
- **Runtime delivery rate**: the decision that the Tier2 runtime can actually
  pass to its factual guard. The only permitted repair is the deterministic
  contract normalizer: a documented unit alias may be normalized and a missing
  unit may receive the conservative idle action. Every repaired row remains
  visible in the artifact and is scored using its repaired action, so an idle
  fill cannot improve HDQ by disappearing from the denominator.

For a scenario to satisfy the structural reliability gate it must have zero
unrecoverable rows, full schema legality and visibility validation, and a
repair rate no higher than 5%. Strict generation is reported as a separate
non-hidden gate. The distinction lets the operational path fail safely while
still making the raw model's output quality measurable and improvable.

## Required Causal Upgrade

To promote from offline HDQ/ORP to actual decision value, the runtime must log
`decision_id`, controller receipt/acknowledgement, unit identity, executed
command, and fixed-horizon unit/target/facility outcomes. The next evaluation
stage should use matched-state or randomized/shadow assignment with a declared
baseline. Inverse-propensity or doubly robust off-policy estimates are only
valid after action propensities and support are logged; they cannot be
reconstructed from this replay alone.

This follows the standard logged-feedback requirement that off-policy estimates
need observed actions and the policy that selected them; doubly robust methods
then combine an outcome model with policy information to reduce bias and
variance. See Dudik, Langford, and Li (2011), Swaminathan and Joachims (2015),
and Jiang and Li (2016). Jiang and Li's sequential doubly robust estimator
also makes clear why a replay without behavior propensities cannot supply a
valid counterfactual value estimate. Voloshin et al. (2019) further shows that
OPE reliability depends on stress-testing estimator assumptions, rather than
selecting a method from a point estimate alone. The implementation-ready record
builders are in `runtime/decision_evidence.py`.

## Acceptance Gates

- Structural: all four red/blue x forecast/no-forecast scenarios parse and pass
  schema validation.
- Semantic: HDQ is reported against the legacy rule baseline and the
  side-specific look-ahead teacher, with no missing scenario hidden by pooling.
  Each full reliability scenario must contain at least 12 labels for each of
  the three global modes before semantic promotion is allowed. Each mode slice
  must also cover at least 12 complete games and have a strictly positive 95%
  bootstrap lower bound for its HDQ delta versus the legacy rule.
- Response: ORP and its coverage are reported separately; unknown is never
  treated as failure.
- Causal: remains `unverifiable` until controller and counterfactual evidence
  exists.
