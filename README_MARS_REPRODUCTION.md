# MARS Reproduction Runner

This runner automates the official MARS method only. It does not reproduce baseline comparisons, ablation studies, efficiency comparisons, or cross-model transfer experiments.

## Environment

Use the existing environment:

```bash
conda activate MARS
```

Install dependencies if needed:

```bash
pip install -r requirements.txt
```

## API Configuration

Do not write API keys into the code. Configure them with environment variables:

```bash
export OPENAI_API_KEY="..."
export OPENAI_BASE_URL="..."
```

On PowerShell:

```powershell
$env:OPENAI_API_KEY="..."
$env:OPENAI_BASE_URL="..."
```

The variable names can be overridden:

```bash
python reproduce_mars.py --api-key-env OPENAI_API_KEY --base-url-env OPENAI_BASE_URL
```

## Dry Run

Dry-run checks task registration, prompt loading, output directory creation, logs, and summary/report generation without calling the API:

```bash
python reproduce_mars.py --tasks geometric_shapes --dry-run
```

## Single Task

```bash
python reproduce_mars.py --tasks geometric_shapes --model deepseek-v2.5-1210 --temperature 0.6
```

Use sample-level concurrency during each prompt evaluation:

```bash
python reproduce_mars.py --tasks geometric_shapes --model deepseek-v2.5-1210 --temperature 0.6 --concurrency 8
```

Some BBH datasets use label answers instead of option letters. The runner now records `answer_format` in `configs/mars_tasks.yaml` and evaluates these formats directly:

```text
boolean_expressions: True / False
formal_fallacies: valid / invalid
sports_understanding: yes / no
option tasks: (A), (B), ...
```

If an older run shows 0 accuracy for these label-answer tasks, rerun them after this fix:

```bash
python reproduce_mars.py --tasks "boolean_expressions,formal_fallacies,sports_understanding" --model deepseek-chat --temperature 0.6 --concurrency 8
```

For quick debugging, avoid the full 249-sample evaluation loop:

```bash
python reproduce_mars.py --tasks "boolean_expressions,formal_fallacies,sports_understanding" --model deepseek-chat --temperature 0.6 --concurrency 8 --max-samples 50 --max-answer-retries 2 --request-timeout 30
```

Use the full dataset only when you are ready to record final numbers:

```bash
python reproduce_mars.py --tasks runnable --model deepseek-chat --temperature 0.6 --concurrency 8
```

## Task Groups

```bash
python reproduce_mars.py --tasks bbh
python reproduce_mars.py --tasks mmlu
python reproduce_mars.py --tasks domain
python reproduce_mars.py --tasks all --model deepseek-v2.5-1210
```

`domain` currently selects C-EVAL, GSM8K, and AGIEval tasks registered in `configs/mars_tasks.yaml`.

To run only tasks that are currently runnable from the checked-in datasets and prompt sections:

```bash
python reproduce_mars.py --tasks runnable --model deepseek-v2.5-1210 --temperature 0.6 --concurrency 8
```

Equivalent filtering can be applied to another selector:

```bash
python reproduce_mars.py --tasks all --runnable-only --model deepseek-v2.5-1210 --temperature 0.6 --concurrency 8
```

## Output Files

Each run creates a timestamped directory:

```text
results_mars/run_YYYYMMDD_HHMMSS/
  config.yaml
  summary.csv
  summary.json
  report.md
  tasks/<task_id>/
    config.yaml
    user_prompt.txt
    planner_prompt.txt
    final_prompt.txt
    prompt_accuracy_history.csv
    predictions.csv
    raw_logs.txt
    errors.jsonl
```

If a dataset or prompt is missing, that task is marked failed in `summary.csv` and the details are written to `errors.jsonl`. Other tasks continue running.

## Adding A Task

Add an entry to `configs/mars_tasks.yaml`:

```yaml
- task_id: new_task
  group: BBH
  dataset_path: Dataset_format/BBH/new_task.csv
  question_type: choice
  user_prompt_key: new_task
  planner_prompt_key: new_task
  metric: accuracy
```

Then add matching sections to:

```text
Prompt/ALL_userproxy_task_input.md
Prompt/ALL_prompt_planner_template.md
```

Section titles are normalized, so `## New Task` maps to `new_task`.

## Legacy Entry

The original entry points are preserved:

```bash
bash run.sh choice
bash run.sh short_answer
```

## Troubleshooting

- `missing_api_key`: set the environment variable named by `api_key_env`.
- `missing_dataset`: check `dataset_path` in `configs/mars_tasks.yaml`.
- `missing_prompt`: add the corresponding prompt sections or adjust `user_prompt_key` / `planner_prompt_key`.
- `task_failed`: inspect `tasks/<task_id>/raw_logs.txt` and `errors.jsonl`.
