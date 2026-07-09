# MARS Prompt vs Best Baseline Prompt

This report compares each task's MARS final prompt with the strongest local baseline prompt among `origin`, `cot_zs`, and `cot_fs`.

## Summary Table

| task | best baseline | best acc | MARS acc | MARS-best | baseline chars | MARS chars | key difference |
|---|---|---:|---:|---:|---:|---:|---|
| boolean_expressions | origin | 0.916 | 0.888 | -0.028 | 188 | 1756 | MARS much longer; MARS adds domain rules |
| disambiguation_qa | cot_fs | 0.728 | 0.648 | -0.080 | 1269 | 448 | baseline has examples; MARS adds elimination |
| formal_fallacies | cot_zs | 1.000 | 0.996 | -0.004 | 156 | 3419 | MARS much longer; MARS adds domain rules |
| geometric_shapes | cot_fs | 0.736 | 0.736 | +0.000 | 1105 | 3896 | MARS much longer; MARS adds elimination; MARS adds domain rules |
| ruin_names | origin | 0.692 | 0.860 | +0.168 | 207 | 2196 | MARS much longer |
| sports_understanding | cot_fs | 0.916 | 0.832 | -0.084 | 601 | 1860 | MARS much longer; MARS adds domain rules |
| college_biology | origin | 0.958 | 0.868 | -0.090 | 217 | 1187 | MARS much longer; MARS adds elimination; MARS adds domain rules |
| college_medicine | origin | 0.861 | 0.850 | -0.012 | 218 | 499 | MARS much longer; MARS adds domain rules |
| electrical_engineering | cot_fs | 0.869 | 0.855 | -0.014 | 1025 | 1043 | baseline has examples; MARS adds domain rules |
| high_school_world_history | origin | 0.916 | 0.886 | -0.030 | 227 | 750 | MARS much longer; MARS adds elimination |
| human_aging | origin | 0.830 | 0.807 | -0.022 | 213 | 1176 | MARS much longer; MARS adds domain rules |
| marketing | origin | 0.949 | 0.944 | -0.004 | 211 | 423 | MARS much longer; MARS adds elimination; MARS adds domain rules |

## Main Pattern

- Best baselines are often short answer-format prompts, or few-shot prompts when examples matter.
- MARS prompts usually add task-specific solving policy: domain principles, elimination rules, internal reasoning, and stricter answer constraints.
- The added structure helps `ruin_names`, but often underperforms strong simple/few-shot baselines in this run.

## Task-Level Notes

### boolean_expressions

- best baseline: `origin` (0.916); MARS: 0.888; delta: -0.028
- baseline prompt: 188 chars; MARS prompt: 1756 chars
- baseline excerpt: Answer BIG-Bench Hard questions for Boolean Expressions. Read the question carefully. Return only the final answer as true or false. Do not include extra commentary after the final answer.
- MARS excerpt: Evaluate the Boolean expression by strictly adhering to the following rules: 1. **Operator Precedence**: Apply operators in this exact order: parentheses first, then NOT (¬), followed by AND (∧), then OR (∨), and finally XOR (⊕) and equivalence (↔), which share equal precedence and are evaluated strictly left-to-right.

### disambiguation_qa

- best baseline: `cot_fs` (0.728); MARS: 0.648; delta: -0.080
- baseline prompt: 1269 chars; MARS prompt: 448 chars
- baseline excerpt: Answer BIG-Bench Hard questions for Disambiguation QA. Use the examples as guidance for reasoning style and answer format. End with Final answer: a parenthesized option letter such as (A). Example question: In the following sentences, explain the antecedent of the pronoun (which thing the pronoun refers to), or state t
- MARS excerpt: Read the question and identify the key ambiguity. For each option, determine which one most directly resolves the ambiguity by providing a clear, contextually appropriate interpretation that eliminates confusion or multiple meanings. Eliminate options that do not clearly address the ambiguity or create inconsistencies.

### formal_fallacies

- best baseline: `cot_zs` (1.000); MARS: 0.996; delta: -0.004
- baseline prompt: 156 chars; MARS prompt: 3419 chars
- baseline excerpt: Answer BIG-Bench Hard questions for Formal Fallacies. Think step by step internally, then provide the final answer. End with Final answer: valid or invalid.
- MARS excerpt: **Revised Prompt:** You will be presented with logical arguments. Your task is to determine whether each argument is **valid** or **invalid** based solely on its logical structure, completely ignoring the factual truth or falsity of the premises. Additionally, when an argument is **invalid**, you must identify the **sp

### geometric_shapes

- best baseline: `cot_fs` (0.736); MARS: 0.736; delta: +0.000
- baseline prompt: 1105 chars; MARS prompt: 3896 chars
- baseline excerpt: Answer BIG-Bench Hard questions for Geometric Shapes. Use the examples as guidance for reasoning style and answer format. End with Final answer: a parenthesized option letter such as (A). Example question: This SVG path element <path d="M 25.00,38.00 L 89.00,58.00"/> draws a Options: (A) circle (B) heptagon (C) hexagon
- MARS excerpt: Answer BIG-Bench Hard questions for Geometric Shapes. Read the question carefully. Before finalizing your answer, reason step-by-step by first identifying the specific geometric principle or transformation type governing the pattern—such as rotation, reflection, translation, congruence, similarity, or symmetry. Then, a

### ruin_names

- best baseline: `origin` (0.692); MARS: 0.860; delta: +0.168
- baseline prompt: 207 chars; MARS prompt: 2196 chars
- baseline excerpt: Answer BIG-Bench Hard questions for Ruin Names. Read the question carefully. Return only the final answer as a parenthesized option letter such as (A). Do not include extra commentary after the final answer.
- MARS excerpt: Read the question carefully and first identify the original word in the context provided. Then, determine the altered version by identifying which letter has been changed to create a new, valid word. Compare the original and altered names side-by-side, focusing on distinguishing between intentional phonetic alterations

### sports_understanding

- best baseline: `cot_fs` (0.916); MARS: 0.832; delta: -0.084
- baseline prompt: 601 chars; MARS prompt: 1860 chars
- baseline excerpt: Answer BIG-Bench Hard questions for Sports Understanding. Use the examples as guidance for reasoning style and answer format. End with Final answer: yes or no. Example question: Is the following sentence plausible? "Neymar did a maradona on the defender in the Champions Leage Semifinal." Example answer: Final answer: y
- MARS excerpt: Read the sports scenario carefully and determine if the described situation is possible according to established sports rules and physical/logical constraints. If the scenario is possible, answer "yes"; if it contains any factual contradictions, physically impossible actions, or logically impossible sequences of events

### college_biology

- best baseline: `origin` (0.958); MARS: 0.868; delta: -0.090
- baseline prompt: 217 chars; MARS prompt: 1187 chars
- baseline excerpt: Answer MMLU multiple-choice questions in College Biology. Read the question carefully. Return only the final answer as a parenthesized option letter such as (A). Do not include extra commentary after the final answer.
- MARS excerpt: For each College Biology multiple-choice question, first analyze the scientific principles involved—such as cellular respiration, central dogma, enzyme function, or membrane transport—to systematically eliminate incorrect options based on contradictions to biological rules. Then, output only your final answer as a pare

### college_medicine

- best baseline: `origin` (0.861); MARS: 0.850; delta: -0.012
- baseline prompt: 218 chars; MARS prompt: 499 chars
- baseline excerpt: Answer MMLU multiple-choice questions in College Medicine. Read the question carefully. Return only the final answer as a parenthesized option letter such as (A). Do not include extra commentary after the final answer.
- MARS excerpt: Read the question and options carefully. Before selecting an answer, internally evaluate each option using step-by-step reasoning based on established medical knowledge, clinical guidelines, and evidence-based practices. In your reasoning, explicitly prioritize the most current clinical guidelines and foundational medi

### electrical_engineering

- best baseline: `cot_fs` (0.869); MARS: 0.855; delta: -0.014
- baseline prompt: 1025 chars; MARS prompt: 1043 chars
- baseline excerpt: Answer MMLU multiple-choice questions in Electrical Engineering. Use the examples as guidance for reasoning style and answer format. End with Final answer: a parenthesized option letter such as (A). Example question: The Barkhausen criterion for an oscillator Options: (A)Loop gain should be unity (B)Loop gain should be
- MARS excerpt: You are an expert in Electrical Engineering. Analyze the following multiple-choice question carefully, applying key Electrical Engineering principles such as Ohm's Law, Kirchhoff's Laws, circuit analysis techniques (e.g., nodal/mesh analysis), signal processing concepts, electromagnetic theory, and semiconductor fundam

### high_school_world_history

- best baseline: `origin` (0.916); MARS: 0.886; delta: -0.030
- baseline prompt: 227 chars; MARS prompt: 750 chars
- baseline excerpt: Answer MMLU multiple-choice questions in High School World History. Read the question carefully. Return only the final answer as a parenthesized option letter such as (A). Do not include extra commentary after the final answer.
- MARS excerpt: Read the question and all options carefully. Identify the historical period, geographic region, and key themes. First, eliminate clearly incorrect options by identifying and ruling out distractors based on common historical misconceptions, focusing on chronological inaccuracies, geographic implausibilities, and themati

### human_aging

- best baseline: `origin` (0.830); MARS: 0.807; delta: -0.022
- baseline prompt: 213 chars; MARS prompt: 1176 chars
- baseline excerpt: Answer MMLU multiple-choice questions in Human Aging. Read the question carefully. Return only the final answer as a parenthesized option letter such as (A). Do not include extra commentary after the final answer.
- MARS excerpt: Read the question carefully and identify the core concept related to human aging. Recall established aging principles, such as cellular senescence, telomere shortening, oxidative stress, genetic factors, and lifestyle influences on longevity. Critically evaluate each option against these principles, considering potenti

### marketing

- best baseline: `origin` (0.949); MARS: 0.944; delta: -0.004
- baseline prompt: 211 chars; MARS prompt: 423 chars
- baseline excerpt: Answer MMLU multiple-choice questions in Marketing. Read the question carefully. Return only the final answer as a parenthesized option letter such as (A). Do not include extra commentary after the final answer.
- MARS excerpt: Read the question carefully and identify the core marketing principle involved. For each option, evaluate whether it directly applies this principle to the specific scenario described. Eliminate options that are irrelevant or misaligned, then select the one that most accurately addresses the question based on establish

