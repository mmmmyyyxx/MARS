# MARS Visualization Report

Run directory: `D:\myx\grade_one\experiments\MARS\results_mars\run_20260629_234236`
Output directory: `D:\myx\grade_one\experiments\MARS\results_mars\run_20260629_234236\visualizations`

## Run metadata
- model: deepseek-chat
- temperature: 0.6
- max_iterations: 10
- early_stop_delta: 0.01
- concurrency: 8
- dry_run: False

## Overall summary
- number of tasks: 6
- number of successful tasks: 6
- number of failed tasks: 0
- average best accuracy: 38.69%
- average final accuracy: 26.17%
- average delta vs paper: -40.90%
- total runtime: 9494.95s

## Caveats
This visualization compares the local MARS-only run against paper-reported MARS results. It does not reproduce the paper's baseline comparisons, ablation studies, efficiency analysis, or cross-model transfer experiments.
The local run used deepseek-chat, while the paper reports main results with deepseek-v2.5-1210. Therefore, numerical differences may reflect model-version differences, API behavior, sampling variance, runnable-task filtering, and implementation differences.

## Required note
This run uses deepseek-chat, while the paper reports results with deepseek-v2.5-1210. The comparison is for reference only and should not be interpreted as an exact reproduction of the paper's numerical results.

## Generated figures
- visualizations\summary_overview.png
- visualizations\task_accuracy_vs_paper.png
- visualizations\task_delta_vs_paper.png
- visualizations\group_average_vs_paper.png
- visualizations\status_overview.png
- visualizations\best_iteration_distribution.png
- visualizations\convergence_curves_all.png
- visualizations\convergence_curves_bbh.png
- visualizations\convergence_curves_mmlu.png
- visualizations\convergence_curves_domain.png

## Comparison table
- `visualizations\paper_comparison_table.csv`

## Missing inputs
- none
