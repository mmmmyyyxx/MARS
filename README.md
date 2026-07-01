# MARS Full Paper Reproduction

This repository provides a one-command framework for running the full local MARS paper reproduction matrix.
Some baselines are faithful reimplementations; others are marked as best-effort until their original algorithms are fully matched.
Paper reference values are used only for comparison and are never substituted for local runs.

Paper: **MARS: Multi-Agent Adaptive Reasoning with Socratic Guidance for Automated Prompt Optimization**

arXiv: https://arxiv.org/abs/2503.16874

## Environment

```bash
conda activate MARS
```

The runner uses OpenAI-compatible environment variables by default:

```bash
set OPENAI_API_KEY=...
set OPENAI_BASE_URL=...
```

PowerShell:

```powershell
$env:OPENAI_API_KEY="..."
$env:OPENAI_BASE_URL="..."
```

## Run

Use `reproduce_paper.py` as the canonical entrypoint:

```bash
python reproduce_paper.py --preset smoke --dry-run
python reproduce_paper.py --preset mars_full --cache-enabled --resume --skip-existing
python reproduce_paper.py --preset paper_full --cache-enabled --resume --skip-existing
python validate_reproduction_outputs.py --latest
pytest -q
```

Useful scoped runs:

```bash
python reproduce_paper.py --preset smoke --methods mars_official --dry-run
python reproduce_paper.py --preset smoke --max-samples 2 --dry-run
python reproduce_paper.py --preset mars_full --eval-protocol strict_mode
python validate_reproduction_outputs.py --latest
```

## Exactness Status

| Method | Exactness | Notes |
|---|---|---|
| Origin | prompt-dependent | exact if original prompt matches |
| CoT-ZS | faithful | standard local implementation |
| CoT-FS | faithful | records examples |
| APE | best-effort or faithful | depends on implementation status |
| ProTeGi | best-effort or faithful | depends on textual-gradient/beam implementation |
| OPRO | best-effort or faithful | depends on optimizer-history implementation |
| PE2 | best-effort or faithful | depends on PE2 meta-prompt implementation |
| MARS | faithful | mars_official runner |
| MARS-light | best-effort | simplified local variant |

The final report includes an `Exactness Status` section. Exact numerical reproduction is only claimed for methods marked `exact_official` or `faithful_reimplementation`.

## Outputs

Runs are written under `results_full/run_<timestamp>/`.

Important files:

- `summary.csv` and `summary.json`: all local result rows with split hashes and exactness metadata.
- `paper_comparison.csv`: local results compared with paper reference values.
- `coverage.json`: completed, partial, failed, skipped, and missing task-method pairs.
- `paper_reproduction_report.md` and `final_report.md`: coverage, exactness, API usage, costs, latency, and protocol warnings.
- `api_calls.csv`, `token_summary.csv`, `latency_summary.csv`, `cost_summary.csv`: run-level API accounting.
- `methods/<method>/<task>/`: per-method outputs, including `metrics.json`, `run_state.json`, prompts, predictions, diagnostics, and local API logs.
- `tables/`, `figures/`, `efficiency/`, `convergence/`: paper-style derived artifacts.

## Visualize

```bash
python visualize_full_results.py --latest
python visualize_full_results.py --run-dir results_full/run_<timestamp>
```

`visualize_full_results.py` is the only visualization entrypoint.

## Configuration

- `configs/reproduction_matrix.yaml`: presets and evaluation protocols.
- `configs/exactness.yaml`: exactness categories and per-method annotations.
- `configs/tasks.yaml`: registered paper tasks and datasets.
- `configs/methods.yaml`: main methods, MARS official-compatible runner, MARS-light, and ablations.
- `configs/suites.yaml`: main, ablation, efficiency, convergence, and transfer suites.
- `configs/models.yaml`: API defaults and pricing metadata.
- `configs/paper_results.yaml`: paper reference numbers for comparison only.

`paper_mode` follows paper-style evaluation and may use identical opt/val/test formatted rows. `strict_mode` splits rows into opt/val/test to avoid prompt-search leakage.

## Legacy Code

The original manual scripts (`run.sh`, `main_MARS.py`, `Config.py`) are retained as upstream reference code. The active reproduction workflow is `reproduce_paper.py`.

## Citation

```bibtex
@article{zhang2025mars,
  title={Mars: A multi-agent framework incorporating socratic guidance for automated prompt optimization},
  author={Zhang, Jian and Wang, Zhangqi and Zhu, Haiping and Liu, Jun and Lin, Qika and Cambria, Erik},
  journal={arXiv preprint arXiv:2503.16874},
  year={2025}
}
```
