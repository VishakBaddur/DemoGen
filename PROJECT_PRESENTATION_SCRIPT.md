# DemoGen Poster Presentation Script

Use this as a word-for-word speaking script. Vishak covers the setup, research questions, pipeline, conditions, limitations, and takeaways. Kushank covers the main central result line and the graphs.

The goal is to answer the professor's easy and hard questions inside the talk itself: what the problem is, what exactly changed, how the pipeline works, what each condition means, what the numbers mean, and what the limitations are.

---

## Speaker 1: Vishak

### Opening: Background and Motivation

"Hi professor, our project is DemoGen. The problem we study is whether large language models produce different quality outputs when the same writing task includes different demographic name cues.

The motivation is that LLMs are increasingly used for high-stakes writing tasks, like cover letters, recommendation letters, advice, concept explanations, and problem solving. In those settings, output quality can affect real outcomes. A more polished cover letter or a more complete recommendation can be useful, while a weaker one can hurt the user.

The important point is that this kind of bias is not always obvious. The model does not have to say anything directly discriminatory. The bias can be invisible. For example, one name might get a more detailed and professional output, while another name gets a shorter or less polished output, even when the task is basically the same.

So our question is simple: if the task stays the same, but the name in the prompt changes, does the model's response quality shift systematically? In plain words, does a prompt for Jamal get a worse cover letter than a prompt for Emily or Greg from the same model, on the same kind of task?"

### Research Questions

"We organized the project around three research questions.

First, do LLM quality scores differ by perceived race when the prompt includes racially associated names?

Second, do mitigation strategies reduce those gaps?

Third, if fairness improves, does that require sacrificing quality?

That third question matters because a mitigation is not very useful if it only makes the model fairer by making every output worse. So we measured both fairness and quality together."

### Pipeline Overview

"Our pipeline has five steps.

First, we build a prompt suite. The final prompt suite has 540 prompts. That comes from 5 task types, 3 perceived racial groups, and 36 prompts per task-race cell.

The five tasks are cover letters, recommendation letters, advice giving, concept explanations, and problem solving. We picked these because they are realistic long-form writing tasks, not just short classification questions.

Second, we generate model responses. We used Groq with Llama-3.1-8b-instant. Each generated response is saved as its own JSON file, keyed by prompt ID. That was important because later we could verify exact counts without rerunning the model.

Third, we audit the responses. For each response, we compute length, readability, and LLM quality. Length is just word count. Readability is a Flesch-Kincaid-style grade level. LLM quality is a 1 to 5 judge score for clarity, helpfulness, professionalism, and completeness.

Fourth, we compute fairness gaps. The main fairness number is the maximum pairwise gap across the three perceived race groups. For example, if Asian mean quality is 4.80 and Black mean quality is 4.69, the gap is about 0.11.

Fifth, we run statistical tests and generate figures. We use one-way ANOVA to test whether group means differ overall, Welch t-tests for pairwise comparisons, Cohen's d for effect size, and Bonferroni correction because we compare multiple pairs."

### Experimental Conditions

"We tested four conditions.

The first condition is baseline. In baseline, the name remains in the prompt. This is the reference condition. It tells us what happens when the model sees demographic name cues normally.

The second condition is persona-blind. In persona-blind prompting, we remove the name and replace it with a neutral phrase like 'the applicant.' This tests whether removing the demographic cue reduces the measured gap.

The third condition is reranked. In reranking, we generate multiple candidate outputs for a prompt, score the candidates, and select one using a quality-based rule. This is an output-level mitigation. It still leaves the name in the prompt, but tries to choose a better candidate after generation.

The fourth condition is system prompt. In this condition, the name stays in the user prompt, but we add a system instruction telling the model to disregard demographic information, including names. This tests whether instruction alone can fix the issue without removing the name.

That distinction is important. Persona-blind removes the signal. System prompting leaves the signal visible and asks the model to ignore it. Reranking also leaves the signal visible and tries to fix the output afterward."

### Why This Design Is Defensible

"The reason this design is defensible is that we are doing an audit. We are not claiming that a name proves a real person's race. We are using perceived racial associations as controlled demographic cues, following the logic of name-based audit studies.

Also, we do not rely on one metric only. We report quality, length, readability, parity gaps, statistical tests, and robustness checks. The main claim is about measured group differences in this controlled setup, not a universal claim that the model is racist in every possible use case."

### Limitations and Robustness

"There are several limitations, and we want to be direct about them.

First, LLM-as-judge is not perfect ground truth. The judge model can have its own biases. That is why we treat LLM quality as an automated proxy, not as human evaluation.

Second, names are imperfect racial proxies. We are measuring how the model responds to name cues, not making claims about real people's identities.

Third, the reranked audit had some missing or non-numeric quality scores. For reranked quality results, the analysis uses 514 valid quality scores, not all 540. Missing values are dropped, not treated as zero.

Fourth, the reranked condition used a mixed candidate budget: 205 prompts used k equals 5 and 335 prompts used k equals 3, for 2,030 candidate generations. That is a practical limitation caused by API and runtime constraints.

Fifth, the name variance analysis showed that within-race name variance exceeded between-race variance in baseline. That means the results should be described carefully as name-based demographic cue auditing, not as proof that every name in a race group behaves the same way.

Sixth, our Mistral/Ollama robustness check used 180 responses. It supported the broader direction, but the exact absolute gaps changed. That means judge choice affects the exact numbers, so we should not overclaim precision."

### Key Takeaways

"The main takeaway is that persona-blind preprocessing was the strongest mitigation in this project.

It reduced the LLM quality gap from 0.110 in baseline to 0.017. That is about an 85 percent reduction. It also increased average measured quality from 4.737 to 4.981.

That is important because it means fairness improvement did not require sacrificing quality here. In our results, the best fairness condition also had the best average quality.

The strongest statistical result was the baseline Asian versus Black quality comparison. The baseline gap was about 0.110, the p-value was about 0.016, and Cohen's d was 0.255. With three race-pair comparisons, the Bonferroni threshold is p less than 0.0167, so this narrowly survives correction.

The policy implication is simple: if a system does not need names to complete the task, removing the name cue before generation can be a zero-cost intervention. It requires no model retraining, no access to model weights, and no extra inference calls.

The novel part is that input-level signal removal beat both output-level candidate selection and instruction-only mitigation in our setup. Reranking helped somewhat. System prompting helped somewhat. But neither matched persona-blind, because both still left the demographic cue visible to the model.

So the safest final conclusion is: in this 540-prompt audit, baseline prompting showed a measurable race-linked quality gap, and persona-blind prompting reduced that gap the most while also improving average measured quality."

### Handoff to Teammate

"Now Kushank will walk through the central result line and the graphs, which show the main numbers visually."

---

## Speaker 2: Kushank

### Central Result Line

"The central result of the poster is this line: baseline gap 0.110, persona-blind gap 0.017, which is an 85 percent reduction.

That means that in the baseline condition, the largest quality difference between perceived race groups was 0.110 on the 1 to 5 LLM quality scale. After persona-blind preprocessing, the largest quality difference dropped to 0.017.

This is the main result because it combines fairness and practicality. Persona-blind did not require retraining the model, changing model weights, or making extra model calls. It just removed the name cue from the prompt."

### Graph 1: Fairness-Quality Tradeoff

"The first graph is the fairness-quality tradeoff plot.

The x-axis is the maximum LLM quality parity gap. Lower is better because it means the group averages are closer together.

The y-axis is mean LLM quality. Higher is better because the responses are judged as clearer, more helpful, more professional, and more complete.

So the best region of the graph is the upper-left: low gap and high quality.

Baseline has a larger gap and lower mean quality. Reranked and system prompt move in the right direction, meaning they reduce the gap compared with baseline. But persona-blind is the best point because it has the smallest gap and the highest mean quality.

That is why we say persona-blind gave the best fairness-quality tradeoff. It did not create the usual tradeoff where fairness improves but quality drops. In our results, fairness improved and quality improved."

### Graph 2: LLM Quality by Perceived Race

"The second graph shows LLM quality by perceived race across the conditions.

In baseline, the group means are not equal. Asian names had the highest average quality at about 4.8045, Black names had the lowest at about 4.6944, and White names were in the middle at about 4.7127. The biggest baseline gap is Asian versus Black, about 0.110.

In persona-blind, those bars become much closer. The quality scores are all near 4.98, and the maximum gap is only 0.017.

In reranked, the gap improves compared with baseline, but it is still 0.063, so it does not beat persona-blind.

In system prompt, the gap is 0.072. That also improves over baseline, but again it does not beat persona-blind.

The interpretation is that asking the model to ignore demographic cues helps less than removing the cue from the prompt."

### Graph 3: Task-Level Parity Gaps

"The third graph shows where the baseline quality gap is concentrated by task type.

The biggest gap is in cover letters, with a gap of 0.269. Concept explanations are next at 0.164. Advice giving is 0.065, problem solving is 0.031, and recommendation letters are only 0.013.

This matters because the bias is not evenly spread across tasks. Cover letters are much more sensitive than recommendation letters in our audit.

One possible reason is that cover letters directly describe a person's professionalism, qualifications, and employability. That gives the model more room to activate social associations attached to the name. In a recommendation letter or a problem-solving prompt, the name may matter less to the structure of the generated answer."

### Graph 4: Mitigation Logic Diagram

"The mitigation diagram explains why the conditions behave differently.

Baseline keeps the name in the prompt and does not apply any mitigation.

System prompt keeps the name in the prompt but adds an instruction to ignore demographic information. The problem is that the model still sees the name.

Reranked also keeps the name in the prompt. It generates multiple outputs and selects one, but all candidates may already be influenced by the same name cue.

Persona-blind removes the name cue before generation. That is why it is the cleanest intervention in this project. It prevents the model from conditioning on the name in the first place."

### Robustness and Caveats While Discussing Graphs

"There are also caveats behind the graphs.

The quality score is an LLM-as-judge score, so it should be treated as an automated proxy. We ran a Mistral/Ollama independent judge check on 180 responses, and it showed that absolute gap sizes can change when the judge changes. So we do not claim the exact numbers are universal.

But the main pattern is still useful: baseline shows a measurable gap, persona-blind strongly reduces it, and leaving the name visible through reranking or system prompting does not perform as well."

### Teammate Closing

"So the graph-level story is: the model produced measurable quality differences when names were visible; persona-blind preprocessing moved the result to the best fairness-quality point; and task-level analysis showed that the largest disparity was concentrated in cover letters.

That is the main evidence behind the poster's claim: demographic conditioning is not fully fixed by telling the model to ignore it. In our setup, the signal had to be removed at the input."

---

## If the Professor Interrupts: Short Answers

### If asked: "Did you prove the model is biased?"

"No. We measured statistically detectable group differences in automated quality scores under a controlled prompt suite. It is evidence of disparity in this setup, not proof of universal model bias."

### If asked: "Why use names?"

"Names are standard demographic cues in audit studies. We are not claiming names reveal true identity. We are testing whether the model reacts differently to perceived name associations."

### If asked: "Why trust LLM-as-judge?"

"We do not treat it as perfect ground truth. It is a scalable proxy for quality. That is why we also report length, readability, statistical tests, and an independent Mistral/Ollama robustness check."

### If asked: "Why did persona-blind improve quality?"

"The safest answer is that removing names may have made prompts more standardized and reduced name-specific variation. We do not claim to prove the internal mechanism."

### If asked: "Why did system prompting not work as well?"

"Because the name cue was still visible. The model was instructed to ignore it, but instruction following does not guarantee that learned associations disappear."

### If asked: "Why did reranking not work as well?"

"Because reranking selects from candidates generated from the same name-containing prompt. If the demographic cue affects the candidate pool, selecting afterward cannot fully remove the upstream signal."

### If asked: "What is the exact strongest result?"

"Baseline Asian versus Black LLM quality gap was about 0.110, p equals 0.0161, Cohen's d equals 0.255, narrowly passing the Bonferroni threshold of p less than 0.0167."

### If asked: "What is the limitation you are most honest about?"

"The biggest limitation is that LLM-as-judge is not human ground truth, and the Mistral robustness check showed that exact gap sizes depend on the judge. So we present the result as automated evidence, not final proof."

### If asked: "How is this different from advanced papers like Wan and Chang or SAGED?"

"Wan and Chang use selective rewriting with an agency classifier, and SAGED uses a broader calibrated benchmarking pipeline. DemoGen is simpler but practical: persona-blind preprocessing works with a black-box API, requires no model weights, no fine-tuning, and no extra inference calls."

---

## Ultra-Short Version If Time Is Limited

### Vishak Short Version

"DemoGen audits whether LLM-generated writing quality changes when only the name cue changes. We built 540 prompts across 5 tasks, 3 perceived race groups, and 36 prompts per cell. We generated responses under four conditions: baseline, persona-blind, reranked, and system prompt. We measured LLM quality, length, and readability, then computed parity gaps and statistical tests.

The main methodological point is that persona-blind removes the demographic cue, while reranking and system prompting leave the cue visible. The limitations are that names are imperfect proxies, LLM-as-judge is not human ground truth, reranking had mixed k and some missing quality scores, and exact robustness numbers vary with judge model.

The main takeaway is that removing the name cue was more effective than asking the model to ignore it or selecting among outputs afterward."

### Kushank Short Version

"The central result is baseline gap 0.110 to persona-blind gap 0.017, an 85 percent reduction. The tradeoff graph shows persona-blind in the best region: lowest gap and highest mean quality. The quality-by-race graph shows baseline group differences almost disappear under persona-blind. The task graph shows cover letters drive the biggest gap, while recommendation letters show almost none. Overall, persona-blind was the strongest fairness-quality intervention in our experiment."
