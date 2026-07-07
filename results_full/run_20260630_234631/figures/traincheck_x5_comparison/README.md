# MARS vs TrainCheck x5 Comparison

- mars_run_dir: results_full\run_20260630_234631
- traincheck_dir: runs_task_level_bbh_traincheck_x5
- overlapping_tasks: 6
- traincheck_accuracy_metrics: vote_acc_mean, oracle_acc_mean
- delta_definition: each non-MARS series accuracy minus MARS accuracy

## Mean Accuracy

| series | mean_accuracy | mean_num_samples |
|---|---:|---:|
| CoT-FS | 0.7813 | 250.0 |
| CoT-ZS | 0.6767 | 250.0 |
| MARS | 0.8267 | 250.0 |
| Origin | 0.7213 | 250.0 |
| TrainCheck baseline oracle | 0.8533 | 100.0 |
| TrainCheck baseline vote | 0.7900 | 100.0 |
| TrainCheck guarded beam oracle | 0.9150 | 100.0 |
| TrainCheck guarded beam vote | 0.8350 | 100.0 |

## Comparability Notes

- MARS and reproduction baselines are loaded from `results_full`.
- TrainCheck x5 results are loaded from `accuracy_summary.csv`; both `vote_acc_mean` and `oracle_acc_mean` are plotted.
- The overlapping TrainCheck rows use 100 test samples, while the MARS BBH rows use 250 samples in this run.
- The TrainCheck detail file marks `leakage_warning=True`; interpret direct accuracy deltas as diagnostic rather than final paper-style claims.

## Files

- `mars_vs_traincheck_x5_accuracy.png`
- `mars_vs_traincheck_x5_delta.png`
- `mars_vs_traincheck_x5_comparison.csv`
- `mars_vs_traincheck_x5_delta.csv`
