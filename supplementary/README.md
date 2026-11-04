# Supplementary data and analysis

## Independent-test protocol

- Existing best policies only; no training or parameter updates.
- Test seeds: 20000–20009; selection seeds 10000–10009 are excluded.
- Greedy actions; PBRS disabled during evaluation.
- All arms, A0, and references share the same episodes within each scenario.
- The evaluation procedure matches the original stream: one environment realization is constructed from the first stream seed, followed by episode-specific resets.
- Best policy weights are loaded directly without restoring optimizer state for inference.

Representative checkpoints were also re-evaluated on the original stream and reproduced every logged score exactly (difference 0.0).

## Files

- `independent_test_per_seed.csv`: 376 per-run evaluations with original best/last and independent-test scores.
- `score_summary.csv`: independent and original-stream scenario/arm means.
- `paired_28_independent.csv`: all 28 paired arm-versus-A1 comparisons, CI, Cohen's dz, raw paired-t and Wilcoxon p-values, plus `holm_p` (paired-t) and `wilcoxon_holm_p` (Wilcoxon), each adjusted separately within its own 28-test family.
- `factor_contrasts_a5_independent.csv`: PBRS, pretraining, joint freezing configuration, A5–A1b, and A5–A0 contrasts.
- `a5_vs_ospf_independent.csv`: A5–OSPF mean differences and intervals.
- `a5_budget_zero_independent.csv`: A5 budget and zero-shot comparisons.
- `selection_stream_metrics.json`: original-stream time-to-threshold, censoring, AUC, and asymptote metrics.
- `run_independent_test.py`: best-policy evaluation script.
- `analyze_independent_test.py`: general paired-statistics script.
- `analyze_a5_supplement.py`: A5-specific supplementary analysis.

Cohen's dz is the mean paired difference divided by the sample standard deviation of per-training-seed differences. Intervals use the t distribution with nine degrees of freedom. The Holm family contains 28 paired t-tests; component contrasts are exploratory.
