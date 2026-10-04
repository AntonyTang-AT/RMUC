# LLM Runtime Reliability Baseline

## Evaluation Contract

The baseline replays saved raw completions from the v4 QV smoke matrix through
the current narrow contract parser. It does not regenerate text or change the
candidate model. Each scenario is evaluated on four held-out games with ten
uniformly sampled decision points per game.

Two rates remain separate throughout the report:

- Strict generation: raw completion already passes the closed layered schema.
- Runtime delivery: the runtime can return a schema-valid decision after only
  documented alias normalization or conservative idle filling for a missing
  unit. Repairs are retained and score as their delivered actions in HDQ.

The source artifacts are under
`out_m2/llm_v4_qv_run2_smoke_retry_runtime_reaudit_v1/`.

## Prompt And Generation Length Contract

The completion-only run uses the local `Qwen2.5-3B-Instruct` base, whose
`max_position_embeddings` is 32,768. The visible v4 training dataset has a
maximum combined prompt-and-completion length of 903 tokens, below the 1,024
training limit. Runtime evaluation bounds the prompt at 896 tokens and
generation at 200 new tokens, for a 1,096-token worst case. This is safely
within the base model context window, so output reliability is evaluated as a
generation-and-schema problem rather than an avoidable context truncation
artifact.

## Baseline Findings

| Scenario | Strict generation | Runtime delivery | Repair rate | End-to-end HDQ delta vs rule |
| --- | ---: | ---: | ---: | ---: |
| Red with forecast | 100% | 100% | 0% | -6.08pp |
| Red without forecast | 100% | 100% | 0% | -10.19pp |
| Blue with forecast | 95% | 100% | 5% | +36.31pp |
| Blue without forecast | 90% | 100% | 10% | +41.96pp |

The runtime parser avoids unrecoverable structural failure on this small smoke
set, but the blue/no-forecast repair rate exceeds the 5% delivery budget. The
candidate therefore fails the runtime structural gate. The semantic gate also
fails because both red scenarios have negative paired HDQ deltas.

The per-field breakdown identifies two different failure modes:

1. **Blue structural closure**: all six strict failures are recoverable
   omissions or a documented `工兵团 -> 工程` alias. Completion-only supervision
   should be judged first on reducing these raw omissions, not merely on the
   repaired delivery rate.
2. **Red semantic regression**: red mode accuracy is 67.5% without forecasts
   and 70.0% with forecasts. Across unit fields, the old candidate is worse
   than the rule on 260/315 (no forecast) and 229/299 (forecast) matched field
   opportunities. The weakest move agreement is concentrated in sentry,
   engineer, infantry-4, and hero actions; air actions are materially stronger.

The stable-label diagnostic confirms that the red deficit is not an artifact
of short-lived teacher-label transitions. On the four-game smoke rows whose
look-ahead label remains unchanged for the next five decision frames, red
without forecast has an HDQ delta of -10.07pp (95% CI [-18.50pp, -3.43pp])
and red with forecast has -9.85pp (95% CI [-23.13pp, -0.78pp]). The next
candidate must therefore improve red unit-action semantics, not merely avoid
transition-adjacent rows.

## Promotion Criteria For The Completion-only Candidate

The queued candidate evaluation has two stages:

1. **Four-game smoke**
   - Zero unrecoverable rows in every red/blue and forecast/no-forecast
     scenario.
   - At most 5% repaired rows in every scenario.
   - Strict generation rate is reported separately; a repair cannot count as a
     strict success.
   - Report end-to-end HDQ and its paired game-bootstrap interval versus the
     same Tier 0 rule on identical issue records.
2. **Twenty-game training-disjoint reliability run**
   - Run only after the smoke runtime-structural gate passes.
   - Require the lower 95% bound of the paired end-to-end HDQ delta to be
     positive in every scenario before setting `shadow_ready=true`.
   - Require at least 12 labels and 12 complete games for each of the
     attack/defense/balanced mode slices; each slice must also have a positive
     95% paired HDQ lower bound versus the rule. The planned holdout covers
     the least-common balanced mode in 14 games per side.
   - Keep causal policy value explicitly unverified until decision issue,
     controller acknowledgement, assigned baseline, and fixed-horizon outcome
     evidence exist.

The companion code paths are `eval/evaluate_llm_candidate_matrix.py`,
`eval/audit_llm_candidate_acceptance.py`,
`eval/reconcile_llm_runtime_rows.py`, and
`eval/audit_llm_hdq_breakdown.py`.
