# ALL_prompt_planner_template

The following content is part of the prompt used by the PlannerAgent during the training of the BBH task in this program. When using it, please copy the corresponding prompt into the `MARS/Prompt/EDIT_2_prompt_planner_template.txt` file.




## Boolean  Expressions

Split the task '{task_description}' into detailed steps and details. 
For example, for the Boolean Expressions task, the task is planned as follows:
Total steps: 5  
Step 1: Clearly define the task as evaluating the truth value of a Boolean expression using Boolean constants (True, False) and operators (and, or, not).  
Step 2: Specify the input format, ensuring it includes a Boolean expression followed by a delimiter (e.g., "is").  
Step 3: Instruct the model to parse the input, identify the Boolean expression, and evaluate it step by step according to Boolean logic rules.  
Step 4: Emphasize the importance of following operator precedence (not, and, or) and handling parentheses correctly.  
Step 5: Direct the model to output only the final truth value (True or False) without additional explanation or commentary.



## Disambiguation QA

Split the task '{task_description}' into detailed steps and details. 
For example, for the Disambiguation QA task, the task is planned as follows:
Total steps: 4  
Step 1: Identify the pronoun in the sentence and the possible antecedents (nouns it could refer to).  
Step 2: Analyze the context and grammatical structure to determine if the pronoun's antecedent can be inferred.  
Step 3: If the antecedent can be deduced, state the correct noun it refers to; otherwise, declare the sentence as ambiguous.  
Step 4: Match the conclusion with the provided options and select the correct answer.



## Formal Fallacies Syllogisms  Negation

Split the task '{task_description}' into detailed steps and details. 
For example, for the Formal Fallacies Syllogisms Negation task, the task is planned as follows:
Total steps: 4  
Step 1: Clearly define the task and its focus on formal fallacies, syllogisms, and negations.  
Step 2: Structure the prompt to include the context, the argument, and the question about deductive validity.  
Step 3: Ensure the prompt explicitly asks the model to evaluate whether the argument is deductively valid or invalid based on the provided premises.  
Step 4: Format the prompt to include clear instructions and options (e.g., "valid" or "invalid") for the model to choose from.



## Geometric Shapes

Split the task '{task_description}' into detailed steps and details. 
For example, for the Geometric Shapes task, the task is planned as follows:
Total steps: 5  
Step 1: Analyze the requirements of the geometric shape identification task to understand the desired output and the types of shapes to be distinguished.  
Step 2: Examine the example SVG path element to identify patterns, key features, and commands that define different geometric shapes.  
Step 3: Create a clear and structured prompt that instructs the language model to analyze the SVG path element and determine the geometric shape it represents.  
Step 4: Include relevant examples of SVG path elements and their corresponding shapes in the prompt to guide the language model's understanding and reasoning.  
Step 5: Test and refine the prompt iteratively by providing sample SVG paths and adjusting the prompt for clarity, accuracy, and alignment with the correctness goal.  



## Ruin Names

Split the task '{task_description}' into detailed steps and details. 
For example, for the ruin names task, the task is planned as follows:
Total steps: 4  
Step 1: Analyze the task requirements to understand the goal of creating a humorous one-character edit for an artist, band, or movie name.  
Step 2: Identify the key components of the prompt, including the input format, the requirement for humor, and the need for a single-character edit.  
Step 3: Draft a clear and concise prompt that instructs the language model to generate a humorous one-character edit for a given name, ensuring it aligns with the task's objectives.  
Step 4: Review and refine the prompt to maximize clarity and effectiveness in guiding the language model to produce the desired output.



## Sports Understanding

Split the task '{task_description}' into detailed steps and details. 
For example, for the Sports Understanding task, the task is planned as follows:
Total steps: 4  
Step 1: Define the task clearly, specifying that the goal is to evaluate the plausibility of a factitious sentence related to sports based on real-world knowledge of sports events, players, and terminology.  
Step 2: Create a structured prompt that includes an example input and output to guide the model on how to assess plausibility, ensuring it understands the context of sports and common terminology.  
Step 3: Ensure the prompt explicitly instructs the model to analyze the sentence for consistency with known sports facts, player capabilities, and event details, and to output "yes" or "no" based on plausibility.  
Step 4: Test the prompt with multiple examples to verify that the model consistently produces accurate and correct assessments of plausibility for sports-related sentences.



## Marketing

Split the task '{task_description}' into detailed steps and details.
For example, for the Marketing task, the task is planned as follows:
Total steps: 4
Step 1: Identify the marketing concept, scenario, or definition being tested by the question.
Step 2: Compare every option against standard marketing principles and eliminate distractors.
Step 3: Select the single best option supported by the question.
Step 4: Instruct the model to output only the final option letter in the required format.



## Human Aging

Split the task '{task_description}' into detailed steps and details.
For example, for the Human Aging task, the task is planned as follows:
Total steps: 4
Step 1: Identify the aging-related biological, medical, psychological, or social concept in the question.
Step 2: Analyze the facts and definitions needed to answer the question accurately.
Step 3: Compare all answer options and choose the best-supported one.
Step 4: Instruct the model to output only the final option letter in the required format.



## High School World History

Split the task '{task_description}' into detailed steps and details.
For example, for the High School World History task, the task is planned as follows:
Total steps: 4
Step 1: Identify the time period, region, event, person, or historical process in the question.
Step 2: Recall the relevant historical facts and causal relationships.
Step 3: Evaluate each option and eliminate historically inconsistent choices.
Step 4: Instruct the model to output only the final option letter in the required format.



## Electrical Engineering

Split the task '{task_description}' into detailed steps and details.
For example, for the Electrical Engineering task, the task is planned as follows:
Total steps: 4
Step 1: Identify the electrical engineering principle or calculation required by the question.
Step 2: Parse circuit, signal, device, or system information carefully.
Step 3: Compare the options using the relevant formulas, definitions, or design constraints.
Step 4: Instruct the model to output only the final option letter in the required format.



## College Medicine

Split the task '{task_description}' into detailed steps and details.
For example, for the College Medicine task, the task is planned as follows:
Total steps: 4
Step 1: Identify the medical topic, symptom pattern, mechanism, or treatment decision being tested.
Step 2: Apply clinical and biomedical knowledge to interpret the question.
Step 3: Compare the answer options and select the most medically appropriate answer.
Step 4: Instruct the model to output only the final option letter in the required format.



## College Biology

Split the task '{task_description}' into detailed steps and details.
For example, for the College Biology task, the task is planned as follows:
Total steps: 4
Step 1: Identify the biological concept, process, organism, or experiment described in the question.
Step 2: Apply the relevant biological principles and definitions.
Step 3: Compare the options and choose the single best answer.
Step 4: Instruct the model to output only the final option letter in the required format.



## Art Studies

Split the task '{task_description}' into detailed steps and details.
For example, for the Art Studies task, the task is planned as follows:
Total steps: 4
Step 1: Identify the art historical period, concept, artwork, artist, technique, or aesthetic idea being tested.
Step 2: Recall the relevant art studies knowledge and cultural context.
Step 3: Compare all answer options and eliminate implausible choices.
Step 4: Instruct the model to output only the final option letter in the required format.



## Clinical Medicine

Split the task '{task_description}' into detailed steps and details.
For example, for the Clinical Medicine task, the task is planned as follows:
Total steps: 4
Step 1: Identify the clinical condition, diagnostic clue, treatment decision, or medical principle.
Step 2: Reason from the provided symptoms, findings, and context.
Step 3: Compare options and choose the most clinically appropriate answer.
Step 4: Instruct the model to output only the final option letter in the required format.



## Urban and Rural Planner

Split the task '{task_description}' into detailed steps and details.
For example, for the Urban and Rural Planner task, the task is planned as follows:
Total steps: 4
Step 1: Identify the urban or rural planning topic, policy, regulation, or design principle in the question.
Step 2: Analyze the planning context and constraints.
Step 3: Compare the options using planning knowledge and choose the best answer.
Step 4: Instruct the model to output only the final option letter in the required format.



## GSM8K

Split the task '{task_description}' into detailed steps and details.
For example, for the GSM8K task, the task is planned as follows:
Total steps: 4
Step 1: Parse the word problem and identify the quantities, relationships, and target value.
Step 2: Work through the arithmetic carefully in a concise internal reasoning process.
Step 3: Verify that the computed value answers the question being asked.
Step 4: Instruct the model to output only the final numeric answer without explanation.



## LSAT AR

Split the task '{task_description}' into detailed steps and details.
For example, for the LSAT Analytical Reasoning task, the task is planned as follows:
Total steps: 5
Step 1: Identify the entities, positions, groups, and constraints in the setup.
Step 2: Translate each rule into explicit logical constraints.
Step 3: Apply the constraints to the specific question being asked.
Step 4: Compare every answer option against the valid arrangements or deductions.
Step 5: Instruct the model to output only the final option letter in the required format.
