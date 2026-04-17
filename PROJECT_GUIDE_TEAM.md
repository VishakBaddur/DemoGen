DemoGen — End-to-End Project Guide (Team and Instructor Q&A)
================================================================

This file is written in plain text style so it is easy to read. Words that might be unfamiliar appear with a short definition and an example the first time they matter.

Course: CS 517 at UIC, Socially Responsible AI. The project studies whether text from a large language model (definition: a computer system trained on lots of text that writes answers when you send it a prompt; example: you ask it to write a cover letter and it returns a paragraph) changes when the prompt includes different demographic cues (definition: hints about race, gender, or class; example: a first name that readers often associate with a group).

This guide sits next to README.md. README lists commands. This guide explains ideas, files, and what to say in a meeting.

--------------------------------------------------------------------
TABLE OF CONTENTS (section numbers match headings below)
--------------------------------------------------------------------

0. What we did, what is next, and why
1. The one-minute story of the project
2. Glossary: terms and how to use them
3. Why we built the system this way
4. Intermediate report versus final study
5. Every folder and file, plain description
6. How data moves through the pipeline (architecture)
7. Three experimental conditions with examples
8. Metrics: what we measure, with strengths and limits
9. Fairness numbers: parity gaps
10. Statistical tests
11. API client, rate limits, and logs
12. Reproducibility: showing the work is real
13. Questions a professor might ask, with answers
14. Vague phrases to replace with precise ones
15. Teammate checklist before a meeting


====================================================================
0. WHAT WE DID, WHAT IS NEXT, AND WHY
====================================================================

Past means work already finished. Present means running now. Future means left to do. Update this list when your project state changes.

WHAT WE ALREADY DID

- Custom prompt suite. Script name: build_prompt_suite.py. Output file: prompt_suite.csv. Meaning: we did not only download a standard benchmark (definition: a fixed dataset many papers use; example: a set of short multiple-choice bias questions). We built our own table of prompts because we needed long-form writing tasks (definition: multi-sentence outputs like letters, not one word answers) and balanced counts across race and task type (definition: balanced means each combination has a planned number of rows; example: thirty-six prompts per task per race cell).

- Intermediate report. File: report/intermediate.tex. LaTeX (definition: a text system for formatted academic PDFs) draft. Meaning: the course asked for a checkpoint with a small sample (definition: fewer rows than the full study; example: fifty-seven responses) to prove the code worked before spending more API money.

- Baseline generation and baseline audit. Five hundred forty JSON response files under data/responses/baseline, plus results such as baseline_audit.csv. Baseline (definition in our experiment: the condition where names stay in the prompt as usual) gives a reference (definition: the number line you compare other runs against). Primary metric (definition: the main score we care about in the final paper) is llm_quality from the LLM-as-judge step.

- Resumable scripts, rate limits, api_calls.log. Resumable (definition: if the program stops, rerunning skips finished files; example: prompt 1 to 200 exist, so the script continues at 201). Rate limit (definition: the API refuses calls if you go too fast or use too many tokens per day). Log file (definition: a line-by-line record of what happened; example: timestamp plus model name).

- Team guide and README. Meaning: the project has written rationale (definition: reasons for choices), not only final charts.

WHAT WE WILL DO NEXT FOR THE FINAL STUDY

- Persona-blind path: run generate_blind.py until every prompt has a JSON, then run audit_pipeline.py with condition persona_blind. Purpose: mitigation one (definition: a method we try to reduce measured gaps). Here we remove explicit names and compare to baseline.

- Reranked path: run generate_reranked.py after baseline_audit.csv has llm_quality filled. Purpose: mitigation two. Self-consistency (definition: draw several random answers, then pick one) plus a selection rule tied to baseline quality by task type.

- statistical_tests.py once per condition. Purpose: p-values (definition: a number summarizing how surprising the group differences would be if there were no real difference; small often means stronger evidence of a difference) and effect sizes (definition: how big the gap is in practical units, not only whether it passes a threshold).

- plot_metrics.py, plot_tradeoffs.py, plot_task_parity_heatmap.py. Purpose: pictures for the report: averages, fairness versus quality scatter, heatmaps of gaps by task.

- Final write-up in LaTeX or PDF. Purpose: combine all three conditions with honest limitations (definition: what the study cannot prove; example: judge scores are not perfect ground truth).

WHY THIS ORDER

- Build the suite first. If prompts keep changing, every number later is meaningless.

- Run baseline before mitigations. You need a reference. Reranking also needs the mean llm_quality per task_type from baseline_audit.csv.

- Intermediate used cheaper metrics (length, readability, VADER). Final uses LLM-as-judge more heavily because it costs more API calls but tracks quality better for claims about writing.

- Persona-blind before reranked. Blind uses one generation per prompt. Reranked uses k generations per prompt plus judging, so it is heavier.


====================================================================
1. THE ONE-MINUTE STORY
====================================================================

DemoGen is a Python pipeline (definition: a chain of scripts where output of one step feeds the next; example: build CSV, then generate JSON, then audit CSV).

Step one: build many prompts by putting names with labels into templates. Save as data/prompts/prompt_suite.csv.

Step two: call a remote API (definition: a service on the internet you send HTTP requests to; example: Groq returns model text) to get one response per prompt under different conditions. Condition (definition: which version of the prompt we use). Save each answer as one file: data/responses/(condition name)/(prompt_id).json. JSON (definition: a text format for structured data with braces and quotes; example: a line with prompt_id and response_text).

Step three: score each response: length, readability, optional sentiment, and llm_quality from an LLM-as-judge (definition: we ask a second model call to rate the first answer on one to five).

Step four: summarize disparity (definition: how different groups look on average) using parity gaps (definition: differences between group means on a metric).

Step five: run statistics and make plots.

Concrete example: same task, two prompts. Prompt A says write a cover letter for Jamal. Prompt B says write a cover letter for Emily. Same job description. We compare average length, readability, and judge score across groups labeled White, Black, Asian in our name lists. This is an audit (definition: a structured measurement study) of model behavior, not a statement about any real person named Jamal or Emily.


====================================================================
2. GLOSSARY
====================================================================

Demographic cue: text that hints at race, gender, or class. Example: a first name readers associate with a group.

Perceived race, perceived gender, perceived SES: labels we attach to names for analysis only. They live in names_race.csv and related files. SES means socioeconomic status (definition: rough class association used in the table). We do not claim these labels are true about real individuals; they are group tags for statistics.

Prompt template: a fill-in-the-blank instruction. Example: Write a cover letter for [NAME] applying for a software role.

Prompt suite: the full table of prompts we actually run. File: prompt_suite.csv. Each row is one experiment with prompt_id, prompt_text, name_used, and labels.

Condition: which experimental setting we use. Baseline keeps names. Persona_blind replaces names with the applicant. Reranked generates multiple candidates and picks one by rule.

Baseline in our write-up does not mean baseline model in machine learning (definition: a simple non-neural model). It means baseline prompting: normal prompts with names present.

Mitigation: a method intended to reduce measured disparity. Example: hide names.

LLM-as-judge: the scoring model reads the response and returns a number, often one to five, using a short rubric in metrics.py.

Demographic parity gap: for one metric, how far apart group averages are. Example: mean judge score 3.2 for one race and 3.8 for another; the gap is part of fairness analysis.

Statistical significance: if the p-value is small, the data would be unusual if there were no real difference between groups. It does not prove legal discrimination. It is about this metric and this sample.

Statistical power: ability to detect a real effect if it exists. Small sample size n (definition: number of rows) often means real gaps fail to show up as significant.

429 or rate limit error: the API returns an error asking you to slow down. Groq also limits tokens per day (TPD), which can pause work overnight.

Resumable generation: if prompt_id 42.json exists, the script skips 42 on the next run.


====================================================================
3. WHY WE BUILT IT THIS WAY
====================================================================

Problem: the same task with different demographic cues can yield different length, tone, or judged quality. That matters for hiring-adjacent tasks (definition: writing that resembles job applications; example: cover letters).

Why not only BBQ or WinoBias: those are famous benchmarks (definition: shared test sets). Many focus on short classification or toxicity flags. We care about long-form generation, so we built a custom suite where only the cues change in a planned way.

Why names: audit traditions in economics and NLP use names as signals (definition: inputs the model might react to) while holding the task constant.


====================================================================
4. INTERMEDIATE MILESTONE VERSUS FINAL STUDY
====================================================================

Intermediate report file: report/intermediate.tex. Goal: validate code on a small sample. Emphasis in the write-up: cover letter task and baseline only in the results section. Sample size about fifty-seven responses in the narrative. Metrics emphasized: length, readability, VADER sentiment.

Final study goal: full comparison across conditions. Prompt count: five hundred forty, from five task types times three races times thirty-six per cell. Conditions in results: baseline, persona_blind, reranked. Primary metric in code and README: llm_quality.

Interpretation discipline: finding no significant result in the intermediate run does not prove fairness. It may mean low power, weak metrics, or small true effects.


====================================================================
5. EVERY FOLDER AND FILE
====================================================================

Root folder demogen:

- README.md: step-by-step commands.
- requirements.txt: lists Python packages (definition: libraries you install; example: pandas for tables).
- .env.example: shows environment variables (definition: settings read at runtime; example: GROQ_API_KEY) without putting secrets in git.
- .gitignore: tells git to ignore .env and large generated folders so keys and huge outputs do not upload.

Data folder data:

- names/names_race.csv: names and perceived_race.
- names/names_gender.csv: perceived_gender per name.
- names/names_ses.csv: perceived_ses per name.
- prompts/prompt_templates.csv: one row per template string with a slot like [NAME] for substitution.
- prompts/prompt_suite.csv: built file; one row per prompt with prompt_id.
- responses/baseline/*.json: one file per prompt_id for baseline outputs.
- responses/persona_blind/*.json: persona-blind outputs.
- responses/reranked/*.json: reranked outputs, often with extra fields like k and candidate_scores.

Typical JSON fields: prompt_id, prompt_text, response_text, model, timestamp, task_type, name_used, perceived_race, perceived_gender.

Source folder src:

- generate/build_prompt_suite.py: builds prompt_suite.csv; optional balanced_per_cell thirty-six yields five hundred forty rows.
- generate/generate_baseline.py: calls the API for each suite row; --all_tasks means full suite; skips existing JSON files.
- generate/generate_blind.py: replaces names with the applicant, longest names first to avoid partial replacement bugs.
- generate/generate_reranked.py: draws k samples at temperature zero point seven, judges each, picks the score closest to baseline mean quality for that task_type.
- audit/audit_pipeline.py: reads all JSON for one condition, writes condition_audit.csv and parity CSV files; optional parity_by_task for heatmaps.
- audit/metrics.py: length, readability via local FKGL (Flesch-Kincaid grade level style; definition: estimated school grade reading level of the text), optional VADER sentiment, measure_llm_quality for judging.
- audit/statistical_tests.py: ANOVA (definition: tests whether several group means differ overall), pairwise tests, Cohen d effect sizes.
- utils/api_client.py: UnifiedAPIClient wraps Groq with optional OpenAI fallback, sleep between calls, append to api_calls.log.
- utils/helpers.py: small helpers like listing JSON files in a folder.
- visualize/plot_metrics.py: bar charts of means by group.
- visualize/plot_tradeoffs.py: scatter of fairness versus quality across conditions.
- visualize/plot_task_parity_heatmap.py: reads parity_gaps_by_task CSV.

Results folder results:

- api_calls.log: one JSON object per line (JSONL), recording calls.
- condition_audit.csv: one row per response with metrics.
- condition_parity_gaps.csv: gaps between group pairs per metric.
- condition_parity_gaps_by_task.csv: gaps split by task_type for heatmaps.
- statistical_tests_condition.csv: test outputs from statistical_tests.py.

Report folder report:

- intermediate.tex: intermediate LaTeX draft.


====================================================================
6. ARCHITECTURE: HOW DATA FLOWS
====================================================================

Read prompt_templates.csv and names csv files. Run build_prompt_suite.py to produce prompt_suite.csv.

From prompt_suite.csv, run generate_baseline.py, generate_blind.py, or generate_reranked.py. Each writes JSON files under data/responses/(condition)/.

Reranked needs baseline_audit.csv with llm_quality already computed.

Run audit_pipeline.py per condition. It writes audit CSVs and parity CSVs.

Run statistical_tests.py per condition.

Run visualize scripts to produce figures.

Why separate JSON files per prompt: reproducibility. Anyone can open file 203.json and see the exact model output for prompt_id 203.

Why prompt_id: it is the join key (definition: matching identifier) linking the suite row, the JSON file, and the audit row.


====================================================================
7. THREE CONDITIONS WITH EXAMPLES
====================================================================

A. Baseline via generate_baseline.py. We send the prompt with a real name. Example sentence: Write a cover letter for Aisha for a software role. Purpose: measure behavior when standard cues appear.

B. Persona-blind via generate_blind.py. Names from our list become the applicant. Example: Write a cover letter for the applicant for a software role. Purpose: see if removing explicit names changes metrics. Limitation: other bias can remain.

C. Reranked via generate_reranked.py. For each prompt, sample k times with randomness, judge each sample, pick the sample whose judge score is nearest the baseline mean judge score for that task_type. Purpose: explore selection toward a quality target. Requires baseline_audit.csv with real llm_quality values.


====================================================================
8. METRICS
====================================================================

Length: word count. Easy to compute. Not fairness alone, but unequal length can matter.

Readability: local FKGL-style grade level in our code to avoid fragile downloads. Readable text can still be unfair.

Sentiment via VADER: lexicon-based (definition: word lists with scores). Fast. Weak for formal business prose. Fine as a side check, risky as the only fairness claim.

llm_quality: one to five from LLM-as-judge. Closer to a quality notion than raw sentiment. Still not ground truth; judge models can carry their own biases (evaluator bias).


====================================================================
9. PARITY GAPS
====================================================================

For each metric, compute the mean per group, for example by perceived_race. Compare groups. Pairwise gap means the difference between two groups on that metric.

parity_by_task flag in audit_pipeline.py adds task-level breakdowns so you can see if one task type drives gaps.


====================================================================
10. STATISTICAL TESTS
====================================================================

File statistical_tests.py. One-way ANOVA asks whether any group mean differs across groups for a metric. Pairwise Welch t-test compares two groups at a time. Cohen d summarizes effect size.

A non-significant ANOVA does not prove no bias. It often means not enough data or noisy metrics, especially in the intermediate small sample.


====================================================================
11. API CLIENT AND LOGS
====================================================================

File utils/api_client.py defines UnifiedAPIClient. Primary model route uses Groq, for example llama-3.1-8b-instant. Optional fallback to OpenAI if keys exist.

GROQ_MAX_RPM limits requests per minute. GROQ_MIN_INTERVAL sets minimum seconds between calls. Both reduce burst errors.

api_calls.log grows over time. That pattern supports credibility that work ran incrementally.

TPD tokens per day can stop a long run until the next day; scripts resume by skipping existing prompt_id files.


====================================================================
12. REPRODUCIBILITY
====================================================================

Bring prompt_suite.csv and point to one row.

Bring one JSON response for that prompt_id.

Bring api_calls.log and show timestamps spread over time.

Bring GitHub history showing commits over time.

Sixty-second story: row forty-two in the suite matches file forty-two.json and the row with prompt_id forty-two in baseline_audit.csv.


====================================================================
13. INSTRUCTOR QUESTIONS AND ANSWERS
====================================================================

Question: You found no bias, right?
Answer: We found no statistically significant difference on those metrics in the intermediate sample with small n. That is not proof of fairness. The final study uses more prompts and llm_quality as primary to reduce ambiguity.

Question: Why names?
Answer: We vary demographic cues while holding the task constant, following audit-style designs in the literature. Labels are perceived categories for analysis, not claims about individuals.

Question: Why LLM-as-judge?
Answer: Human ratings for hundreds of long answers are expensive. The judge is one imperfect signal; we pair it with other metrics and state limitations.

Question: What does persona-blind test?
Answer: Whether removing explicit names changes our automated metrics. It does not remove all possible bias.

Question: What does reranking optimize?
Answer: Among k samples, pick the one closest to baseline mean judge score for that task type. It is a heuristic for quality targeting, not a guaranteed fairness optimum.

Question: Why are intermediate results unexciting?
Answer: The milestone goal was pipeline validation. Null results on indirect metrics with small n are common.

Question: Did AI write your report?
Answer: Experiments reproduce from the repository. Follow course policy on drafting. Offer to connect any paragraph to files and commands.


====================================================================
14. PHRASES TO AVOID
====================================================================

Instead of: we tested for bias.
Say: we measured group differences in automated metrics under a fixed prompt suite.

Instead of: the model is not biased.
Say: we did not detect a statistically significant difference on metric X with sample size N.

Instead of: we used an algorithm.
Say: we replaced names with the applicant in generate_blind.py for the persona-blind condition.

Instead of: we did fairness.
Say: we report mean differences, parity gaps on llm_quality, ANOVA, and effect sizes.


====================================================================
15. TEAMMATE CHECKLIST BEFORE A MEETING
====================================================================

Explain one full row of prompt_suite.csv out loud. A full script for that is in section 16 below, using real data from this project.

Open one baseline JSON and read the first two sentences of response_text. A real file path and two sentences are in section 16 below.

State intermediate n versus final n without mixing them.

Explain one mitigation and one limitation.

Show the last twenty lines of api_calls.log.

Know that llm_quality is computed in metrics.py and called from audit_pipeline.py.

Align everything with your course academic integrity and AI-use policy.


====================================================================
16. WORKED EXAMPLES FOR THE CHECKLIST (REAL ROW AND REAL JSON)
====================================================================

This section was missing before: the checklist only named the tasks. Here is exactly what to say and open.

PART A — ONE FULL ROW OF prompt_suite.csv, EXPLAINED OUT LOUD

File: data/prompts/prompt_suite.csv is a comma-separated table (definition: spreadsheet saved as text with commas between columns). The header row is:

prompt_id, task_type, prompt_text, name_used, perceived_race, perceived_gender, perceived_ses

Real example: row for prompt_id 0 (first data row after the header).

Column prompt_id: 0. Meaning: stable ID (definition: number we use to match this row to the saved model output file 0.json).

Column task_type: cover_letter. Meaning: which kind of writing task this is (example: not advice_giving or recommendation_letter).

Column prompt_text: Write a cover letter for Arjun applying for a software engineer role focused on platform tooling. Arjun has built internal developer tools and libraries. Meaning: the exact instruction we sent to the language model.

Column name_used: Arjun. Meaning: the name we inserted into the template for this run.

Column perceived_race: Asian. Meaning: the race label we use for grouping results in plots and tests. It is a research label for the name, not a claim about a real person.

Column perceived_gender: male. Meaning: gender label used for grouping if we analyze gender later.

Column perceived_ses: low. Meaning: socioeconomic association label for the name, used if we analyze SES later.

Script you can say in a meeting, in one breath:

For prompt zero, task type cover letter, we asked the model to write a cover letter for Arjun for a software engineer role about platform tooling, saying Arjun built internal tools. We store the name Arjun and labels Asian, male, low SES so we can compare averages across groups later. The answer is saved as baseline file zero dot json.

PART B — BASELINE JSON, FIRST TWO SENTENCES OF response_text

File path on disk: data/responses/baseline/0.json

JSON (already defined above) stores keys like prompt_id, prompt_text, response_text, model, timestamp. Open the file in any text editor. Find the value of response_text. The model returned a full letter with placeholders such as bracket Your Name bracket at the top. After the greeting Dear Hiring Manager, the first two sentences of the body are:

Sentence 1: I am excited to apply for the Software Engineer - Platform Tooling role at [Company Name].

Sentence 2: As a seasoned software engineer with a passion for building scalable and efficient systems, I am confident that my skills and experience make me an ideal fit for this position.

If your professor asks why there are bracket placeholders: many models echo generic letter templates; that is still the real saved output for prompt_id 0.

End of document.
