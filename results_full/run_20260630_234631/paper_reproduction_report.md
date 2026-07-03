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
- main_methods: mars_official
- expected_main_task_method_pairs: 12
- completed_main_task_method_pairs: 12
- main_coverage_fraction: 1.0000
- partial_pairs: 0
- failed_pairs: 0
- skipped_complete_pairs: 0

## Suite Rows

- main: 12

## Status

- completed: 12

## Error Summary

- api_errors: 0
- parse_errors: 0

## Resume / Run-State Summary

- completed: 12
- skipped_complete: 0
- partial: 0
- failed: 0

## Output Validation Summary

- validation_errors: 102
- validation_warnings: 0

## Cost and API Summary

- total_api_call_records: 18476
- total_tokens: 5069636
- total_estimated_tokens: 7400072
- cache_hit_rate: 0.1771
- total_estimated_cost: 0.000000
- mean_latency_seconds: 3.4038

## Paper Reference Comparison

- comparable_rows: 6
- mars_average_delta_percentage_points: -3.79

## Exactness Status

- faithful_reimplementation: 12

|Method|Exactness|Notes|
|---|---|---|
|Origin|prompt_dependent|Prompt-dependent; exact only when the local origin prompt matches the original paper prompt.|
|CoT(ZS)|faithful_reimplementation|Local zero-shot chain-of-thought template.|
|CoT(FS)|faithful_reimplementation|Local few-shot construction.|
|APE|best_effort_reimplementation|Generic candidate-generation search; pending original APE search details.|
|ProTeGi|best_effort_reimplementation|Generic textual-gradient-inspired search; pending full ProTeGi beam protocol.|
|OPRO|best_effort_reimplementation|Generic optimizer-history search; pending full OPRO optimizer protocol.|
|PE2|best_effort_reimplementation|Generic meta-prompt search; pending exact PE2 prompt-engineer protocol.|
|MARS-official|faithful_reimplementation|Official-compatible Manager/UserProxy/Planner/Teacher/Critic/Student/Target flow without Prompt/ mutation.|
|MARS|faithful_reimplementation|Alias for the official-compatible MARS runner.|
|MARS-light|best_effort_reimplementation|Simplified local MARS variant retained for diagnostics.|
|w/oPlan|faithful_reimplementation|Official-compatible ablation with Planner disabled.|
|w/oSoc|faithful_reimplementation|Official-compatible ablation with Socratic Teacher/Critic loop disabled.|
|w/oCri|faithful_reimplementation|Official-compatible ablation with Critic disabled.|

## Evaluation Protocol

- protocol: paper_mode
- Warning: paper_mode opt_hash == val_hash == test_hash. paper_mode is intended for paper comparability; use strict_mode for leakage-controlled evaluation.

## Output Files

- `summary.csv`, `summary.json`, `paper_comparison.csv`, `coverage.json`
- `api_calls.csv`, `token_summary.csv`, `latency_summary.csv`, `cost_summary.csv`
- method directories under `methods/<method>/<task>/`
