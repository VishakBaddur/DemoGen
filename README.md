# DemoGen (CS 517)

DemoGen audits demographic bias in LLM-generated content and applies mitigation strategies.

## Project Structure (high level)

- `data/names/`: demographic name associations
- `data/prompts/`: prompt templates and prompt suite
- `data/responses/`: saved LLM responses per condition
- `src/generate/`: prompt-suite and response scripts
- `src/audit/`: metric computation and audit pipeline
- `src/visualize/`: plots for the report
- `results/`: audit CSVs and figures
- `report/`: NeurIPS LaTeX drafts

## Setup

1. `pip install -r requirements.txt`
2. Copy `.env.example` to `.env` and set `GROQ_API_KEY` (and optionally `OPENAI_API_KEY` as fallback).

## Final report pipeline (500+ responses, 3 conditions)

Run from the `demogen/` directory.

### 1. Prompt suite (540 prompts: 5 task types × 3 races × 36)

Task types: `cover_letter`, `advice_giving`, `concept_explanation`, `recommendation_letter`, `problem_solving`.

```bash
python src/generate/build_prompt_suite.py --balanced_per_cell 36 --output_path data/prompts/prompt_suite.csv
```

### 2. Baseline generation (all prompts)

```bash
python src/generate/generate_baseline.py --all_tasks
```

### 3. Persona-blind (Algorithm 1)

```bash
python src/generate/generate_blind.py
```

### 4. Self-consistency reranking (Algorithm 2)

Requires `results/baseline_audit.csv` with LLM-as-judge scores (step 5 for baseline first), so **after** baseline audit:

```bash
python src/generate/generate_reranked.py --k 5 --temperature 0.7
```

### 5. Audit each condition (primary metric: LLM-as-judge; length/readability secondary; VADER optional)

```bash
python src/audit/audit_pipeline.py --condition baseline --parity_by_task
python src/audit/audit_pipeline.py --condition persona_blind --parity_by_task
python src/audit/audit_pipeline.py --condition reranked --parity_by_task
```

Optional VADER (not used as a primary fairness metric): add `--include_sentiment` and include `sentiment` in `--parity_metrics`.

### 6. Statistical tests (per condition)

```bash
python src/audit/statistical_tests.py --condition baseline
python src/audit/statistical_tests.py --condition persona_blind
python src/audit/statistical_tests.py --condition reranked
```

### 7. Figures

Bar charts (mean metric by race):

```bash
python src/visualize/plot_metrics.py --condition baseline --group_col race
```

Fairness–quality tradeoff (mean quality vs max racial parity gap on `llm_quality`):

```bash
python src/visualize/plot_tradeoffs.py \
  --audit_paths results/baseline_audit.csv results/persona_blind_audit.csv results/reranked_audit.csv \
  --labels baseline persona_blind reranked
```

Task-type × metric heatmap (needs `*_parity_gaps_by_task.csv` from step 5):

```bash
python src/visualize/plot_task_parity_heatmap.py --parity_by_task_csv results/baseline_parity_gaps_by_task.csv
```

## Intermediate milestone (smaller subset)

For the original intermediate checkpoint (subset / cover-letter focus), you can still use `--target_prompts` without `--balanced_per_cell`, and `generate_baseline.py --task_type cover_letter` instead of `--all_tasks`.

## Notes

- Generation is **resumable**: existing `data/responses/<condition>/<prompt_id>.json` files are skipped.
- Primary evaluation metric for the final report is **LLM-as-judge** (`llm_quality`); VADER is optional and not recommended as a primary disparity metric for formal text.
