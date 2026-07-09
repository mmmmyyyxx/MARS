# Full MARS Reproduction Report

This repository supports full local paper-matrix reproduction.
Exact numerical reproduction is only claimed for methods marked exact_official or faithful_reimplementation.
APE, ProTeGi, OPRO, PE2 are currently best-effort unless their original algorithms are implemented.
Paper reference values are used only for comparison and are never substituted for local runs.

## Run Configuration

- suite: main
- model: deepseek-chat
- source_model: deepseek-chat
- target_models: deepseek-r1,gpt-3.5,gpt-4,gpt-4o
- temperature: 0.6
- max_samples: all
- max_iterations: 10
- eval_protocol: paper_mode
- dry_run: False
- cache_enabled: True

## Matrix Coverage

- registered_tasks: 17
- main_methods: origin, cot_zs, cot_fs, mars_official
- expected_main_task_method_pairs: 24
- completed_main_task_method_pairs: 24
- main_coverage_fraction: 1.0000
- partial_pairs: 0
- failed_pairs: 0
- skipped_complete_pairs: 0

## Suite Rows

- main: 24

## Status

- completed: 24

## Error Summary

- api_errors: 20
- parse_errors: 0

## Resume / Run-State Summary

- completed: 24
- skipped_complete: 0
- partial: 0
- failed: 0

## Output Validation Summary

- validation_errors: not_run
- validation_warnings: not_run

## Cost and API Summary

- total_api_call_records: 16179
- total_tokens: 6837096
- total_estimated_tokens: 8602072
- cache_hit_rate: 0.1030
- total_estimated_cost: 0.000000
- mean_latency_seconds: 2.4557

## Paper Reference Comparison

- comparable_rows: 18
- mars_average_delta_percentage_points: unavailable

## Exactness Status

- prompt_dependent: 6
- best_effort_reimplementation: 12
- legacy_aligned_partial: 6

|Method|Exactness|Notes|
|---|---|---|
|Origin|prompt_dependent|Prompt-dependent; exact only when the local origin prompt matches the original paper prompt.|
|CoT(ZS)|best_effort_reimplementation|Local CoT baseline; paper reports baseline numbers but the original repository does not provide a complete executable baseline protocol.|
|CoT(FS)|best_effort_reimplementation|Local CoT baseline; paper reports baseline numbers but the original repository does not provide a complete executable baseline protocol.|
|APE|best_effort_reimplementation|Generic candidate-generation search; pending original APE search details.|
|ProTeGi|best_effort_reimplementation|Generic textual-gradient-inspired search; pending full ProTeGi beam protocol.|
|OPRO|best_effort_reimplementation|Generic optimizer-history search; pending full OPRO optimizer protocol.|
|PE2|best_effort_reimplementation|Generic meta-prompt search; pending exact PE2 prompt-engineer protocol.|
|MARS-official|legacy_aligned_partial|Legacy-aligned partial reproduction: uses original UserProxy/Planner task blocks, agent system prompts, legacy Target prompt format, p0 evaluation, and target call-count limit; remaining differences may include framework wrappers, local baseline protocol, and model/API behavior.|
|MARS|legacy_aligned_partial|Legacy-aligned partial reproduction: uses original UserProxy/Planner task blocks, agent system prompts, legacy Target prompt format, p0 evaluation, and target call-count limit; remaining differences may include framework wrappers, local baseline protocol, and model/API behavior.|
|MARS-light|best_effort_reimplementation|Simplified local MARS variant retained for diagnostics.|
|w/oPlan|legacy_aligned_partial|Legacy-aligned partial reproduction: uses original UserProxy/Planner task blocks, agent system prompts, legacy Target prompt format, p0 evaluation, and target call-count limit; remaining differences may include framework wrappers, local baseline protocol, and model/API behavior.|
|w/oSoc|legacy_aligned_partial|Legacy-aligned partial reproduction: uses original UserProxy/Planner task blocks, agent system prompts, legacy Target prompt format, p0 evaluation, and target call-count limit; remaining differences may include framework wrappers, local baseline protocol, and model/API behavior.|
|w/oCri|legacy_aligned_partial|Legacy-aligned partial reproduction: uses original UserProxy/Planner task blocks, agent system prompts, legacy Target prompt format, p0 evaluation, and target call-count limit; remaining differences may include framework wrappers, local baseline protocol, and model/API behavior.|

## Evaluation Protocol

- protocol: paper_mode
- Warning: paper_mode opt_hash == val_hash == test_hash. paper_mode is intended for paper comparability; use strict_mode for leakage-controlled evaluation.

## Output Files

- `summary.csv`, `summary.json`, `paper_comparison.csv`, `coverage.json`
- `api_calls.csv`, `token_summary.csv`, `latency_summary.csv`, `cost_summary.csv`
- method directories under `methods/<method>/<task>/`
