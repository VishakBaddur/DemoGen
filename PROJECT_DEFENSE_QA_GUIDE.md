# DemoGen Project Defense Q&A Guide

This document is a conversational defense guide for the DemoGen project. It is written so you can answer professor questions in plain English. It explains jargon in simple terms and connects every claim to the actual pipeline, files, algorithms, results, plots, and limitations.

Use this as your "if the professor asks anything" prep sheet.

Important current facts:

- Project: DemoGen, CS 517 Socially Responsible AI, UIC.
- Main question: Do LLM-generated responses change in quality when prompts include names associated with different perceived racial groups?
- Conditions: baseline, persona-blind, reranked.
- Full prompt suite: 540 prompts.
- Full saved responses: 1,620 JSON files, 540 per condition.
- Main metric: `llm_quality`, an LLM-as-judge score from 1 to 5.
- Baseline max race gap on `llm_quality`: 0.110.
- Persona-blind max race gap on `llm_quality`: 0.017.
- Reranked max race gap on `llm_quality`: 0.063.
- Baseline mean `llm_quality`: 4.737.
- Persona-blind mean `llm_quality`: 4.981.
- Reranked mean `llm_quality`: 4.815 over 514 valid scored rows.
- Reranked had 26 missing/non-numeric `llm_quality` rows in `results/reranked_audit.csv`.
- Reranked used mixed `k`: 205 prompts with `k=5`, 335 prompts with `k=3`, total 2,030 candidate generations.

---

## 1. One-Minute Summary

### Q: What is DemoGen?

DemoGen is a pipeline for auditing demographic bias in LLM-generated text. In simple terms, we give a language model similar writing tasks, but change the name in the prompt. Then we check whether the output quality, length, or readability changes across perceived racial groups.

Example:

- Prompt A: "Write a cover letter for Emily..."
- Prompt B: "Write a cover letter for Jamal..."
- Prompt C: "Write a cover letter for Arjun..."

The task is similar, but the name changes. We compare the model's outputs across groups.

### Q: What is the core research question?

Do LLM-generated responses differ in measured quality across demographic groups when prompts contain racially associated names, and can simple mitigation strategies reduce those differences?

Simpler version: If only the name changes, does the model write better or worse answers for some groups?

### Q: What did we find?

In the baseline condition, there was a measurable race-linked quality gap. Persona-blind prompting reduced that gap strongly, and reranking reduced it partially. Persona-blind also had the highest measured average `llm_quality`.

The simplest finding:

- Baseline quality gap: 0.110.
- Persona-blind quality gap: 0.017.
- Reranked quality gap: 0.063.

So persona-blind was the strongest mitigation for the primary quality metric.

---

## 2. Motivation and Problem Framing

### Q: Why did we pick this problem?

We picked it because LLMs are already used for practical writing tasks: cover letters, recommendation letters, advice, explanations, and problem solving. Those outputs can matter in real life. If one person's prompt gets a more polished or helpful response than another person's prompt only because of a name, that is a fairness issue.

This is not about the model saying something openly racist. It is about subtle quality differences.

Example:

- Both people ask for a cover letter.
- One output is detailed and professional.
- The other is shorter or less polished.

That difference may affect how useful the tool is.

### Q: What made LLM bias interesting here?

LLM bias in generation is harder to see than bias in classification. In classification, a model might output "yes" or "no." In generation, the difference might be tone, length, detail, helpfulness, or professionalism. That makes it harder to detect, but still important.

### Q: What did we expect before running the experiments?

We expected some group differences, but we were not sure where they would appear. We thought they might show up more in surface metrics like length or readability. The surprising result was that `llm_quality` showed a statistically significant baseline difference, and the largest quality gaps were concentrated in certain task types, especially cover letters.

### Q: What was surprising?

Three things:

1. Persona-blind prompting reduced the `llm_quality` gap much more than reranking.
2. Persona-blind also increased measured average quality compared with baseline.
3. Bias was not equally spread across tasks. Cover letters had much larger quality disparity than advice giving or recommendation letters.

---

## 3. Key Jargon in Simple Terms

### Q: What is an LLM?

LLM means large language model. It is a system trained on a lot of text that can generate text when given a prompt.

Example: You ask it, "Write a recommendation letter," and it writes one.

### Q: What is a prompt?

A prompt is the instruction we send to the model.

Example: "Write a cover letter for Arjun applying for a software engineer role."

### Q: What is a prompt suite?

A prompt suite is the full table of prompts we use for the experiment. It is like a spreadsheet of all test cases.

In our project, `data/prompts/prompt_suite.csv` has 540 rows.

### Q: What is a demographic cue?

A demographic cue is a hint in the text that may signal race, gender, class, or another identity category.

Example: A first name like "Jamal," "Emily," or "Arjun" may be associated by readers, and possibly by the model, with different demographic groups.

### Q: Why do we say "perceived race" instead of just "race"?

Because the labels are not claims about real people. They are research labels for names. We are studying how the model may react to names that are commonly perceived as associated with a group.

### Q: What is an audit?

An audit is a structured test. We hold most things constant and change one thing to see whether the system behaves differently.

Example: Same cover letter task, different name.

### Q: What is mitigation?

Mitigation means a method we try to reduce a measured problem.

In our project:

- Persona-blind prompting is a mitigation.
- Reranking is a mitigation.
- Baseline is not a mitigation; it is the reference condition.

### Q: What is a parity gap?

A parity gap is the difference between group averages.

Example:

- Asian mean quality = 4.80.
- Black mean quality = 4.69.
- Gap = 4.80 - 4.69 = 0.11.

We use absolute gaps, so we care about size of difference, not direction.

### Q: What is LLM-as-judge?

LLM-as-judge means we ask another model call to rate the output. In our code, the judge reads a generated response and returns a score from 1 to 5.

Simple example:

- 1 = poor.
- 3 = okay.
- 5 = excellent.

In the code, this lives in `src/audit/metrics.py` as `measure_llm_quality`.

### Q: What is readability?

Readability estimates how complex the writing is. We use Flesch-Kincaid grade level style scoring.

Example:

- Grade level 8 means roughly eighth-grade reading complexity.
- Grade level 12 means more complex text.

Readability is not the same as fairness. It is a supporting metric.

### Q: What is ANOVA?

ANOVA is a statistical test that asks whether several group means are different enough that the difference is unlikely to be random noise.

Simple version: Are White, Black, and Asian group averages different enough to flag?

### Q: What is a p-value?

A p-value is a number used in statistics. Smaller values mean the observed group difference would be more surprising if there were no real difference.

Common threshold: p < 0.05.

Important: p < 0.05 does not prove discrimination. It means the metric showed a statistically detectable group difference in this dataset.

### Q: What is Cohen's d?

Cohen's d is an effect size. It tells how large a difference is in standard deviation units.

Simple version: p-value asks "is there evidence of a difference?" Cohen's d asks "how big is the difference?"

---

## 4. Project Architecture

### Q: What is the project pipeline?

The pipeline has five main stages:

1. Build prompt suite.
2. Generate model responses.
3. Compute metrics.
4. Compute fairness gaps and statistical tests.
5. Generate plots.

Simple flow:

`prompt_templates + names -> prompt_suite.csv -> response JSON files -> audit CSVs -> parity/statistical CSVs -> plots`

### Q: What folders matter?

- `data/names/`: name lists and demographic labels.
- `data/prompts/`: prompt templates and final prompt suite.
- `data/responses/`: generated model outputs, one JSON per prompt per condition.
- `src/generate/`: scripts that build prompts and call the model.
- `src/audit/`: scripts that compute metrics, parity gaps, and statistical tests.
- `src/visualize/`: scripts that make plots.
- `results/`: audit CSVs, parity CSVs, statistical test CSVs, logs, and figures.
- `report/`: report drafts.

### Q: What is the most important join key?

`prompt_id`.

It links:

- one row in `data/prompts/prompt_suite.csv`,
- one response file such as `data/responses/baseline/0.json`,
- one audit row in `results/baseline_audit.csv`.

If the professor asks how to prove the data is real, show one `prompt_id` through all three files.

---

## 5. Data and Prompt Suite

### Q: How many prompts did we use?

540 prompts.

Formula:

5 task types x 3 perceived racial groups x 36 prompts per cell = 540.

### Q: What are the five task types?

1. `cover_letter`
2. `advice_giving`
3. `concept_explanation`
4. `recommendation_letter`
5. `problem_solving`

### Q: What racial groups did we analyze?

We analyzed three perceived racial groups:

- White
- Black
- Asian

These are based on name labels, not claims about real people.

### Q: Why did we use names?

Names are a standard audit method. They act as demographic signals while allowing us to keep the task mostly the same.

Example:

- "Write a cover letter for Emily..."
- "Write a cover letter for Jamal..."

The content can stay similar while the perceived demographic cue changes.

### Q: Why use Bertrand and Mullainathan name logic?

Bertrand and Mullainathan is a classic name-based audit study. Using established name lists is more defensible than inventing our own names, because the names have precedent in prior research.

### Q: Why did Caliskan et al. matter?

Caliskan et al. showed that language representations can encode social associations. That influenced our design because we treated names as signals that may activate learned associations in the model.

### Q: Did we use WinoBias or BBQ?

No, not for the final completed run. We discussed adding a second dataset, but the completed final pipeline is the custom DemoGen prompt suite. The reason is that DemoGen targets long-form practical generation tasks like cover letters and advice, while many benchmark datasets focus on classification, short answers, or fixed bias tests.

If asked, say: a second dataset would strengthen external validation, but our completed study focuses on a custom long-form generation suite.

---

## 6. Generation Conditions

### Q: What are the three conditions?

1. Baseline.
2. Persona-blind.
3. Reranked.

### Q: What is baseline?

Baseline means the model sees the normal prompt with the name included.

Example:

"Write a cover letter for Arjun applying for a software engineer role..."

Purpose: measure what happens when demographic cues are present.

### Q: What is persona-blind prompting?

Persona-blind means we remove the name cue and replace it with a neutral phrase like "the applicant."

Example:

Original:

"Write a cover letter for Arjun..."

Persona-blind:

"Write a cover letter for the applicant..."

Purpose: test whether removing the name reduces group differences.

### Q: Why is persona-blind a mitigation?

Because it tries to reduce bias by preventing the model from seeing the demographic cue.

Simple analogy: If the model cannot see the name, it has less chance to react differently because of the name.

### Q: What is reranking?

Reranking means generating multiple candidate answers and selecting one using a rule.

In our project:

1. Generate several candidate responses for the same prompt.
2. Score each candidate with LLM-as-judge.
3. Pick the candidate whose score is closest to the baseline mean quality for that task type.

### Q: Why did reranking need the baseline audit first?

Because reranking needs a target quality score per task type. Those targets come from `results/baseline_audit.csv`.

Example:

If baseline cover letters have mean quality 4.26, reranking tries to pick a candidate near 4.26 for cover letter prompts.

### Q: Did reranking use `k=5` or `k=3`?

It used a mixed `k`.

Exact current artifact counts:

- 205 prompts used `k=5`.
- 335 prompts used `k=3`.
- Total candidate generations: 2,030.

Why mixed? We started with `k=5`, but due to API rate limits and runtime constraints, later prompts used `k=3` to complete the full pipeline.

### Q: Is the mixed `k` a weakness?

Yes, it is a limitation. It makes the reranked condition less clean than a fully fixed-`k` experiment. We should state it honestly.

Good wording:

"The reranked condition used a mixed candidate budget because of API/runtime constraints: 205 prompts used `k=5` and 335 used `k=3`, yielding 2,030 candidate generations. This should be considered when interpreting reranking results."

---

## 7. API, Models, and Reliability

### Q: What API/model did we use?

The code uses a unified API client in `src/utils/api_client.py`. The project used Groq-based generation during runs, and the code has fallback/configuration options depending on keys and model availability.

When answering in class, keep it simple:

"We generated and judged responses through an API client, logged every call, and saved every response locally as JSON."

### Q: What is `api_calls.log`?

It is a log of API calls. A log is a record of what happened over time.

Why it matters: It shows the project was actually run and helps with reproducibility/debugging.

### Q: What implementation struggles happened?

The long runs hit API limits and transient network errors. Some calls failed or returned null judge scores. We added retry/backoff behavior and resume support so the pipeline could continue instead of restarting everything.

Simple version:

"The research was not just statistics. We also had to make the system reliable enough to finish a long experiment."

### Q: What does resumable mean?

Resumable means if a run stops, the script can continue from already-saved files.

Example:

If `data/responses/baseline/200.json` already exists, the generator skips it instead of calling the API again.

### Q: Why is resume behavior important?

Without resume behavior, one API failure late in the run could waste hundreds of completed calls. Resume behavior makes the experiment practical and reproducible.

---

## 8. Audit Metrics

### Q: What metrics did we compute?

Main metrics:

- `llm_quality`: 1 to 5 quality score from LLM-as-judge.
- `length`: word count.
- `readability`: Flesch-Kincaid grade level.

Optional/older metric:

- `sentiment`: VADER sentiment. This is not the primary final metric.

### Q: Why is `llm_quality` the primary metric?

Because the project is about generated writing quality. Length and readability are useful, but they do not directly tell whether a response is helpful, professional, or complete.

Example:

A response can be long but bad. A response can be readable but unhelpful. `llm_quality` tries to capture usefulness and clarity.

### Q: What is a weakness of LLM-as-judge?

The judge model can also be biased. It is not perfect ground truth. We should always say it is an automated proxy, not a human evaluation.

### Q: Why include length?

Length is a simple surface feature. If one group consistently receives shorter responses, that may indicate unequal treatment, even if quality scores look similar.

### Q: Why include readability?

Readability checks whether the complexity of writing differs across groups. If the model writes simpler or more complex text for one group, that may matter.

### Q: How does the code handle missing metric values in parity gaps?

`compute_demographic_parity_gap` in `src/audit/metrics.py` uses `dropna()`. That means missing values are dropped, not treated as zero.

Important example:

The reranked `llm_quality` gap of 0.063 was computed from 514 valid rows, not from 540 rows with nulls as zero.

---

## 9. Exact Data Completeness

### Q: How many response files exist?

- Baseline: 540 JSON files.
- Persona-blind: 540 JSON files.
- Reranked: 540 JSON files.
- Total: 1,620 JSON files.

### Q: How many audit rows exist?

Each audit CSV has 540 rows:

- `results/baseline_audit.csv`: 540 rows.
- `results/persona_blind_audit.csv`: 540 rows.
- `results/reranked_audit.csv`: 540 rows.

### Q: Are all `llm_quality` values present?

No.

- Baseline: 540 valid `llm_quality` scores.
- Persona-blind: 540 valid `llm_quality` scores.
- Reranked: 514 valid `llm_quality` scores and 26 missing/non-numeric values.

### Q: Where are the missing reranked quality scores?

Breakdown of the 26 missing reranked `llm_quality` rows:

By race:

- Asian: 10.
- Black: 9.
- White: 7.

By task:

- `problem_solving`: 16.
- `advice_giving`: 7.
- `concept_explanation`: 3.

### Q: Does this invalidate the reranked results?

No, but it is a limitation. The reported reranked quality means and quality gaps use the 514 valid scored rows. We should disclose this clearly.

Good answer:

"The reranked condition has 540 generated responses, but 26 judge scores are missing. Our quality-based reranked results are computed over the 514 valid judge scores, with missing values dropped by the metric code."

---

## 10. Main Results

### Q: What are the overall mean metric values?

Overall means:

| Condition | Mean LLM quality | Mean length | Mean readability |
|---|---:|---:|---:|
| Baseline | 4.737 | 389.241 | 11.918 |
| Persona-blind | 4.981 | 409.417 | 11.585 |
| Reranked | 4.815 | 385.469 | 11.534 |

Note: reranked `llm_quality` mean uses 514 valid rows.

### Q: Which condition had the highest average quality?

Persona-blind had the highest measured average `llm_quality`: 4.981.

This is important because older report wording said persona-blind had "no meaningful sacrifice." The stronger accurate claim is that persona-blind reduced the gap while increasing measured average quality compared with baseline.

### Q: What were the mean quality scores by race?

Baseline:

- Asian: 4.8045.
- Black: 4.6944.
- White: 4.7127.

Persona-blind:

- Asian: 4.9722.
- Black: 4.9889.
- White: 4.9833.

Reranked:

- Asian: 4.8471.
- Black: 4.7836.
- White: 4.8150.

### Q: Which group had highest and lowest baseline quality?

In baseline:

- Highest: Asian, 4.8045.
- Lowest: Black, 4.6944.

Be careful: some earlier report drafts incorrectly said Asian was lower than White. The actual CSV means show Asian highest and Black lowest in baseline.

### Q: What are the max parity gaps by metric?

Baseline:

- `llm_quality`: 0.110.
- `length`: 5.422.
- `readability`: 0.767.

Persona-blind:

- `llm_quality`: 0.017.
- `length`: 7.644.
- `readability`: 0.416.

Reranked:

- `llm_quality`: 0.063.
- `length`: 7.856.
- `readability`: 0.462.

### Q: How much did persona-blind reduce the quality gap?

Baseline quality gap = 0.110.

Persona-blind quality gap = 0.017.

Absolute reduction = 0.110 - 0.017 = 0.093.

Relative reduction = about 85%.

### Q: How much did reranking reduce the quality gap?

Baseline quality gap = 0.110.

Reranked quality gap = 0.063.

Absolute reduction = 0.047.

Relative reduction = about 42%.

### Q: Which mitigation performed better?

For the primary metric, `llm_quality`, persona-blind performed better.

Why:

- Lower quality gap: 0.017 vs reranked 0.063.
- Higher mean quality: 4.981 vs reranked 4.815.

### Q: Did any metric get worse under mitigation?

Yes. Length parity got worse.

Max length gap:

- Baseline: 5.422 words.
- Persona-blind: 7.644 words.
- Reranked: 7.856 words.

This means the mitigation improved quality parity but did not improve every metric. That is a fairness tradeoff.

### Q: What is the correct conclusion?

The clean conclusion is:

"Persona-blind prompting was the strongest mitigation for the primary quality metric, reducing the race-linked quality gap from 0.110 to 0.017 while increasing average measured quality. However, length parity did not improve, so the intervention should not be described as improving every fairness metric."

---

## 11. Statistical Tests

### Q: What statistical tests did we run?

We ran one-way ANOVA across perceived racial groups for each metric and condition.

We also ran pairwise Welch t-tests for:

- White vs Black.
- White vs Asian.

The pairwise p-values use Bonferroni correction.

### Q: Which ANOVA results were significant?

Only baseline `llm_quality` was significant at p < 0.05.

ANOVA p-values:

Baseline:

- `llm_quality`: 0.039957.
- `length`: 0.796518.
- `readability`: 0.352993.

Persona-blind:

- `llm_quality`: 0.491554.
- `length`: 0.535045.
- `readability`: 0.448698.

Reranked:

- `llm_quality`: 0.321663.
- `length`: 0.486599.
- `readability`: 0.418376.

### Q: What does baseline `llm_quality` p = 0.040 mean?

It means that, for baseline responses, the quality scores differed across perceived race groups enough to pass the p < 0.05 threshold.

Plain English:

"The baseline quality differences were statistically detectable."

### Q: What does it not mean?

It does not prove the model is legally discriminatory. It does not prove the exact cause. It means our measured automated quality scores differed across groups in this controlled setup.

### Q: Why do persona-blind and reranked p-values matter?

Their `llm_quality` ANOVA p-values were not significant:

- Persona-blind: 0.492.
- Reranked: 0.322.

This supports the interpretation that the measured group quality differences were reduced under mitigation.

### Q: What should we say about pairwise tests?

Be careful. Baseline White vs Asian had an uncorrected p-value below 0.05, but after Bonferroni correction it was 0.084, which is not below 0.05.

Safe wording:

"The overall baseline ANOVA for `llm_quality` was significant. Pairwise comparisons suggested the largest contrast involved White and Asian names, but the Bonferroni-corrected pairwise result did not cross 0.05."

---

## 12. Task-Level Results

### Q: Why did we do task-level analysis?

Overall averages can hide where the problem is. Task-level analysis asks: which type of writing shows the biggest gap?

Example:

Bias might show up strongly in cover letters but not in advice giving.

### Q: Which task had the highest quality gap in baseline?

`cover_letter`.

Baseline `llm_quality` task gaps:

- `cover_letter`: 0.269.
- `concept_explanation`: 0.164.
- `advice_giving`: 0.065.
- `problem_solving`: 0.031.
- `recommendation_letter`: 0.013.

### Q: What happened to task quality gaps under persona-blind?

Persona-blind task quality gaps:

- `cover_letter`: 0.083.
- `advice_giving`: 0.000.
- `concept_explanation`: 0.000.
- `problem_solving`: 0.000.
- `recommendation_letter`: 0.000.

### Q: What happened under reranking?

Reranked task quality gaps:

- `cover_letter`: 0.194.
- `concept_explanation`: 0.115.
- `advice_giving`: 0.000.
- `problem_solving`: 0.000.
- `recommendation_letter`: 0.000.

### Q: Why might cover letters show the biggest disparity?

Cover letters are identity-centered professional writing. The name is attached to the person being described as qualified, professional, and employable. That may give the model more room to activate social associations.

Simple example:

In a math explanation, the name might just be an addressee. In a cover letter, the name is the subject of the professional story.

### Q: What task had the biggest gap overall across all metrics?

For baseline, the largest task-level single metric gap was `problem_solving` on `length`: 22.506 words.

Do not confuse this with the largest `llm_quality` task gap, which was `cover_letter`.

---

## 13. Plots and Figures

### Q: What plots did we generate?

Figures in `results/figures/` include:

Mean-by-race plots:

- `baseline_mean_llm_quality_by_race.png`
- `persona_blind_mean_llm_quality_by_race.png`
- `reranked_mean_llm_quality_by_race.png`
- and length/readability versions for each condition.

Tight-axis quality plots:

- `baseline_mean_llm_quality_by_race_tight.png`
- `persona_blind_mean_llm_quality_by_race_tight.png`
- `reranked_mean_llm_quality_by_race_tight.png`
- `llm_quality_by_race_across_conditions_tight.png`

Heatmaps:

- `baseline_parity_gaps_by_task_heatmap.png`
- `persona_blind_parity_gaps_by_task_heatmap.png`
- `reranked_parity_gaps_by_task_heatmap.png`

Tradeoff plot:

- `fairness_quality_tradeoff.png`

Older/extra:

- `baseline_mean_sentiment_by_race.png`

### Q: What does the quality-by-race plot show?

It shows average `llm_quality` for each race group under each condition.

The tight-axis version makes small differences visible. The normal 0-to-5 y-axis makes the bars look almost identical because the scores are all high.

### Q: Is it okay to use a tight y-axis?

Yes, if the axis is clearly labeled. A tight y-axis helps show small differences, but it can visually exaggerate them if not disclosed.

Good explanation:

"We use a zoomed y-axis to make small group differences visible; the actual score range remains 1 to 5."

### Q: What does the fairness-quality tradeoff plot show?

It compares conditions by:

- x-axis: max `llm_quality` parity gap. Lower is fairer.
- y-axis: mean `llm_quality`. Higher is better.

The ideal point is upper-left: high quality, low gap.

### Q: What should the tradeoff conclusion be?

Persona-blind is best on the primary quality fairness tradeoff:

- It has the lowest quality gap.
- It has the highest mean measured quality.

Reranking improves over baseline on quality gap, but not as much as persona-blind.

### Q: What do heatmaps show?

Heatmaps show task-by-metric parity gaps. Darker cells mean bigger gaps.

Plain example:

If the `cover_letter` row is darker under `llm_quality`, that means cover letters have larger quality gaps than other tasks.

---

## 14. Algorithms in Simple Terms

### Q: What is Algorithm 1, persona-blind prompting?

Input: a prompt with a name.

Process:

1. Find names from our name list in the prompt.
2. Replace the name with "the applicant" or a neutral phrase.
3. Send the modified prompt to the model.
4. Save the response.

Example:

"Write a cover letter for Arjun..." becomes "Write a cover letter for the applicant..."

### Q: Why replace names longest-first?

To avoid partial replacement bugs.

Example:

If one name is "Ann" and another is "Annie", replacing "Ann" first could accidentally damage "Annie." Longest-first replacement avoids this problem.

### Q: What is Algorithm 2, reranking?

Input: a prompt and a target quality score for that task type.

Process:

1. Generate multiple candidate responses.
2. Score each candidate with LLM-as-judge.
3. Compare candidate scores to the baseline target for that task.
4. Pick the candidate closest to the target.
5. Save the selected response plus metadata like `k`, `candidate_scores`, and `selected_index`.

### Q: Why choose closest to target instead of highest quality?

The idea was not simply to maximize quality. It was to reduce extreme quality variation by selecting responses near a task-specific baseline quality target.

Limitation: This does not explicitly optimize fairness. That is probably why reranking was less effective than persona-blind.

### Q: Why did reranking underperform persona-blind?

Because reranking still starts from prompts with names. Every candidate can already be influenced by the demographic cue. Selecting among biased candidates cannot fully remove the upstream cue.

Plain analogy:

If all candidates are already affected by the same input signal, picking the "best" among them does not erase the signal.

---

## 15. Exact Commands

### Q: What command builds the prompt suite?

```bash
python src/generate/build_prompt_suite.py --balanced_per_cell 36 --output_path data/prompts/prompt_suite.csv
```

### Q: What command generates baseline responses?

```bash
python src/generate/generate_baseline.py --all_tasks
```

### Q: What command generates persona-blind responses?

```bash
python src/generate/generate_blind.py
```

### Q: What command generates reranked responses?

Original intended command:

```bash
python src/generate/generate_reranked.py --k 5 --temperature 0.7
```

Actual final artifacts used mixed `k` because the run was resumed with different `k` values:

- 205 prompts with `k=5`.
- 335 prompts with `k=3`.

### Q: What commands audit responses?

```bash
python src/audit/audit_pipeline.py --condition baseline --parity_by_task
python src/audit/audit_pipeline.py --condition persona_blind --parity_by_task
python src/audit/audit_pipeline.py --condition reranked --parity_by_task
```

### Q: What command resumes an interrupted audit?

```bash
python src/audit/audit_pipeline.py --condition reranked --parity_by_task --resume_audit
```

`--resume_audit` reuses existing non-null `llm_quality` values and only scores missing rows.

### Q: What commands run statistical tests?

```bash
python src/audit/statistical_tests.py --condition baseline --metrics llm_quality length readability
python src/audit/statistical_tests.py --condition persona_blind --metrics llm_quality length readability
python src/audit/statistical_tests.py --condition reranked --metrics llm_quality length readability
```

### Q: What commands make plots?

Mean-by-race plots:

```bash
python src/visualize/plot_metrics.py --condition baseline --group_col race
python src/visualize/plot_metrics.py --condition persona_blind --group_col race
python src/visualize/plot_metrics.py --condition reranked --group_col race
```

Tradeoff plot:

```bash
python src/visualize/plot_tradeoffs.py \
  --audit_paths results/baseline_audit.csv results/persona_blind_audit.csv results/reranked_audit.csv \
  --labels baseline persona_blind reranked
```

Task heatmaps:

```bash
python src/visualize/plot_task_parity_heatmap.py --parity_by_task_csv results/baseline_parity_gaps_by_task.csv
python src/visualize/plot_task_parity_heatmap.py --parity_by_task_csv results/persona_blind_parity_gaps_by_task.csv
python src/visualize/plot_task_parity_heatmap.py --parity_by_task_csv results/reranked_parity_gaps_by_task.csv
```

---

## 16. Files to Show the Professor

### Q: If the professor asks "show me the data," what should I open?

Open:

- `data/prompts/prompt_suite.csv`
- `data/responses/baseline/0.json`
- `results/baseline_audit.csv`
- `results/baseline_parity_gaps.csv`
- `results/statistical_tests_baseline.csv`

### Q: If the professor asks "show me the mitigation," what should I open?

Open:

- `src/generate/generate_blind.py` for persona-blind.
- `src/generate/generate_reranked.py` for reranking.
- `data/responses/persona_blind/0.json`
- `data/responses/reranked/0.json`

### Q: If the professor asks "show me the metrics," what should I open?

Open:

- `src/audit/metrics.py`
- `src/audit/audit_pipeline.py`
- `results/baseline_audit.csv`

### Q: If the professor asks "show me the statistics," what should I open?

Open:

- `src/audit/statistical_tests.py`
- `results/statistical_tests_baseline.csv`
- `results/statistical_tests_persona_blind.csv`
- `results/statistical_tests_reranked.csv`

### Q: If the professor asks "show me the figures," what should I open?

Open:

- `results/figures/llm_quality_by_race_across_conditions_tight.png`
- `results/figures/fairness_quality_tradeoff.png`
- `results/figures/baseline_parity_gaps_by_task_heatmap.png`
- `results/figures/persona_blind_parity_gaps_by_task_heatmap.png`
- `results/figures/reranked_parity_gaps_by_task_heatmap.png`

---

## 17. Reproducibility

### Q: How can we prove the project is reproducible?

We can point to:

1. The prompt suite CSV.
2. Saved JSON response files for every condition.
3. Audit CSVs with computed metrics.
4. Parity and statistical test CSVs.
5. Figure files.
6. API call logs.
7. GitHub commits.

### Q: Why save one JSON file per prompt?

It makes debugging and verification easy.

Example:

If someone asks about prompt 0, open:

- `data/prompts/prompt_suite.csv`, row `prompt_id=0`.
- `data/responses/baseline/0.json`.
- `results/baseline_audit.csv`, row `prompt_id=0`.

### Q: What is the current GitHub status?

The repo has code, response JSONs, CSV results, plots, and docs pushed. API keys are not pushed because GitHub blocked secret push protection.

### Q: Why are API keys not in GitHub?

Because they are secrets. Pushing them can expose paid accounts or private access. GitHub also blocks detected secrets.

---

## 18. Report Accuracy Traps

### Q: What numbers must not be mixed up?

Do not say baseline mean quality is 4.97. That is wrong for the current final CSVs.

Correct:

- Baseline mean quality: 4.737.
- Persona-blind mean quality: 4.981.
- Reranked mean quality: 4.815 over 514 valid rows.

### Q: What should the abstract say about persona-blind quality?

Say:

"Persona-blind prompting reduces the maximum quality gap to 0.017 while increasing average LLM-as-judge quality from 4.74 to 4.98."

Do not only say "no quality sacrifice" if you want the stronger accurate claim.

### Q: What should the report say about reranking `k`?

Do not say all reranked prompts used `k=5`.

Correct:

"The reranked condition used a mixed candidate budget: 205 prompts with `k=5` and 335 prompts with `k=3`, for 2,030 total candidate generations."

### Q: What should the report say about reranked missing scores?

Say:

"Reranked quality analyses use 514 valid LLM-as-judge scores; 26 reranked rows had missing/non-numeric quality scores and were dropped by the metric code for quality-based means and gaps."

### Q: What group direction should we report?

Baseline mean quality:

- Asian highest: 4.8045.
- Black lowest: 4.6944.
- White middle: 4.7127.

Do not say Asian was lower than White in baseline; the actual CSV does not support that.

---

## 19. Limitations

### Q: What are the main limitations?

1. LLM-as-judge can be biased.
2. Name labels are only perceived demographic proxies.
3. We analyze three race groups, not all identities.
4. We do not have human evaluation.
5. Reranking used mixed `k`, not one fixed candidate budget.
6. Reranked has 26 missing/non-numeric quality scores.
7. This is one custom prompt suite, not every possible writing task.

### Q: How do we explain name labels as a limitation?

Names are imperfect signals. A name does not determine a real person's race or identity. We use perceived associations for auditing, not to claim truth about individuals.

### Q: How do we explain LLM-as-judge as a limitation?

The judge is another model. It may share some biases with the generation model. So the quality score is useful but not perfect ground truth.

### Q: How do we explain no human evaluation?

Human ratings would be stronger, but they are expensive and time-consuming. Our automated judge makes the full 1,620-response study feasible, but future work should include human evaluation.

### Q: How do we explain mixed `k`?

API limits forced a practical tradeoff. We completed the full reranked condition with a mixed candidate budget. Future work should rerun reranking with fixed `k` and compare sensitivity.

---

## 20. Related Work Questions

### Q: What did Bertrand and Mullainathan contribute to our project?

They provide the classic name-based audit idea. They showed that names can be used to test differential treatment while holding other content constant.

### Q: Why not invent our own names?

Inventing names would be less defensible. Established audit names connect our project to prior research.

### Q: What did Caliskan et al. contribute?

They showed that word embeddings can encode social bias. That supports our assumption that names may activate learned associations inside language models.

### Q: What did Borkan et al. contribute?

They influenced our use of group-based parity gap metrics. Parity gaps are easy to interpret because they compare group averages directly.

### Q: What did Wan and Chang / Guan et al. contribute?

They show that broader bias benchmarks and bias evaluation pipelines exist. Our project is complementary because we focus on custom long-form writing prompts rather than only fixed benchmark tasks.

---

## 21. Professor-Style Hard Questions

### Q: Did you prove the model is biased?

No. We measured statistically significant group differences in automated quality scores under a controlled prompt suite. That is evidence of demographic disparity in this setup, not proof of universal model bias or legal discrimination.

### Q: Why is this not just random noise?

The baseline `llm_quality` ANOVA p-value was 0.039957, below 0.05. Also, the disparity pattern reduced under persona-blind prompting, which supports the idea that names were part of the mechanism.

### Q: If Asian had the highest baseline score, why call it bias?

Bias here means systematic group difference, not only harm to one specific group. A fairness audit asks whether outputs differ by demographic cue. The direction matters, but the existence of a group-linked difference is the issue being measured.

### Q: Why use race labels if race is socially constructed and names are imperfect?

We use perceived race labels as audit categories, following prior name-based audit methods. We are not claiming names reveal true identities. We are testing whether the model reacts differently to names associated with groups.

### Q: Why did persona-blind improve average quality?

One possible explanation is that removing names made prompts more generic and less likely to trigger name-specific associations or template variations. The model may have produced more standardized, high-scoring responses. This is an interpretation, not a proven mechanism.

### Q: Why did length gaps get worse under persona-blind?

Persona-blind optimized input cues, not output length. Removing names reduced quality disparity but did not force equal verbosity. Different tasks and generated structures can still produce length variation.

### Q: Why did reranking not work as well?

Reranking still used prompts with names, so every candidate could already be affected by the demographic cue. Also, the selection rule targeted quality closeness, not fairness directly.

### Q: Why did you not use human judges?

Human judges would be stronger but expensive and time-consuming for 1,620 outputs. We used LLM-as-judge for scale and paired it with simpler metrics and statistical tests.

### Q: Why use a custom dataset instead of an existing benchmark?

We wanted long-form practical writing tasks. Many benchmarks focus on short classification-style bias. Our prompt suite tests tasks like cover letters and advice, where output quality matters.

### Q: Is the project reproducible if API outputs can change?

The exact generation process may vary if rerun because APIs/models can change. But we saved all generated responses as JSON, so the reported analysis is reproducible from saved artifacts.

### Q: What would you do next?

1. Add a second validation dataset such as BBQ or WinoBias adapted for generation.
2. Use human evaluators for quality.
3. Rerun reranking with fixed `k`.
4. Analyze intersectional groups such as race plus gender.
5. Try fairness-aware reranking that explicitly penalizes group variation.

---

## 22. If Asked to Explain One Full Example

### Q: Can you walk through one prompt from start to finish?

Yes.

Prompt 0 in `data/prompts/prompt_suite.csv` is a cover letter prompt using the name Arjun, labeled Asian, male, low SES. The baseline generator sends that exact prompt to the model and saves the result as `data/responses/baseline/0.json`.

Then `audit_pipeline.py` reads that JSON and computes:

- word count,
- readability,
- optional sentiment,
- LLM-as-judge quality.

The row appears in `results/baseline_audit.csv`. Then the fairness code compares average metrics across perceived race groups and writes parity gaps to `results/baseline_parity_gaps.csv`.

Simple version:

"One row becomes one prompt, one saved response, one audit row, and then contributes to group averages."

---

## 23. How to Talk About Findings Safely

### Q: What is the safest one-sentence result?

"In our 540-prompt audit, baseline prompting with names showed a statistically significant `llm_quality` difference across perceived race groups, and persona-blind prompting reduced the maximum quality gap from 0.110 to 0.017 while increasing measured average quality."

### Q: What should we avoid saying?

Avoid:

- "The model is racist."
- "We proved fairness."
- "Persona-blind solves bias."
- "All metrics improved."
- "Reranking used `k=5` for all prompts."
- "Baseline quality was 4.97."

Say instead:

- "We measured group differences."
- "Persona-blind reduced the quality gap in this setup."
- "Length parity did not improve."
- "Reranking used mixed `k` due to API constraints."
- "Baseline quality was 4.737."

---

## 24. Final Mental Model

### Q: What is the entire project in one analogy?

Imagine giving the same writing assignment to an assistant many times, changing only the name of the person in the assignment. Then we grade every answer and ask:

- Did some names get better answers?
- Does hiding the name fix that?
- Does generating multiple answers and picking one fix that?

In our results:

- Yes, baseline had a measurable quality gap.
- Hiding the name helped the most.
- Reranking helped, but less.
- Some metrics, like length, did not improve.

That is DemoGen.

