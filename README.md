# MARS Full Paper Reproduction

This repository is organized as a one-command reproduction framework for:

**MARS: Multi-Agent Adaptive Reasoning with Socratic Guidance for Automated Prompt Optimization**

arXiv: https://arxiv.org/abs/2503.16874

The active workflow is the full-paper reproduction pipeline. It covers the registered
17 tasks, main baselines, MARS variants, ablations, efficiency logging, convergence
curves, and cross-model transfer runs.

## Environment

```bash
conda activate MARS
```

The runner uses OpenAI-compatible environment variables by default:

```bash
set OPENAI_API_KEY=...
set OPENAI_BASE_URL=...
```

PowerShell users can set them with:

```powershell
$env:OPENAI_API_KEY="..."
$env:OPENAI_BASE_URL="..."
```

## Run

Use `reproduce_paper.py` as the canonical entrypoint:

```bash
# Validate all registered tasks, prompts, parsers, and output schemas.
python reproduce_paper.py --preset smoke --dry-run

# Run MARS on all 17 registered tasks.
python reproduce_paper.py --preset mars_full

# Run the full local paper matrix.
python reproduce_paper.py --preset paper_full --cache-enabled --resume --skip-existing
```

Useful scoped runs:

```bash
python reproduce_paper.py --preset paper_full --tasks boolean_expressions,gsm8k
python reproduce_paper.py --preset paper_full --target-models deepseek-r1,gpt-4o
python reproduce_paper.py --preset paper_full --max-samples 20 --dry-run
```

## Outputs

Runs are written under `results_full/run_<timestamp>/`.

Important files:

- `summary.csv` and `summary.json`: all local result rows.
- `paper_comparison.csv`: local results compared with paper reference values.
- `coverage.json`: expected/completed main-suite task-method coverage.
- `paper_reproduction_report.md`: matrix coverage, errors, and exactness notes.
- `tables/`: paper-style CSV/Markdown tables.
- `efficiency/`: accuracy, runtime, token, and cost points.
- `convergence/`: per-task convergence curves.
- `figures/`: generated visualizations.
- `methods/<method>/<task>/api_calls.csv`: per-call token, latency, cache, and error traces.

## Visualize

```bash
python visualize_full_results.py --latest
python visualize_full_results.py --run-dir results_full/latest
```

`visualize_full_results.py` is the only visualization entrypoint.

## Configuration

- `configs/reproduction_matrix.yaml`: `smoke`, `mars_full`, and `paper_full` presets.
- `configs/tasks.yaml`: registered paper tasks and datasets.
- `configs/methods.yaml`: main methods and MARS ablation variants.
- `configs/suites.yaml`: main, ablation, efficiency, convergence, and transfer suites.
- `configs/models.yaml`: API defaults and pricing metadata.
- `configs/paper_results.yaml`: paper reference numbers for comparison only.

Paper reference values are never substituted for failed or missing local runs.

## Entrypoints

Use:

```bash
python reproduce_paper.py --preset mars_full
python reproduce_paper.py --preset paper_full
```

All full-reproduction run logic now lives in `reproduce_paper.py`.
The original manual scripts (`run.sh`, `main_MARS.py`, `Config.py`) are retained as
legacy upstream code. They are not the active reproduction path.

## Citation

```bibtex
@article{zhang2025mars,
  title={Mars: A multi-agent framework incorporating socratic guidance for automated prompt optimization},
  author={Zhang, Jian and Wang, Zhangqi and Zhu, Haiping and Liu, Jun and Lin, Qika and Cambria, Erik},
  journal={arXiv preprint arXiv:2503.16874},
  year={2025}
}
```
