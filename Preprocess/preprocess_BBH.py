import json
import os
import random

import pandas as pd

# Define the path to the input JSON file
input_file_path = "../Dataset/BBH/causal_judgement.json"

# Open and load the JSON file containing the dataset
with open(input_file_path, "r", encoding="utf-8") as file:
    data = json.load(file)

# Extract the examples from the loaded data
examples = data["examples"]

# Determine the number of questions available
num_questions = len(examples)

# Sample all available questions instead of a fixed number
sampled_examples = random.sample(examples, num_questions)

# Convert the sampled examples to a DataFrame
df = pd.DataFrame(
    {
        "question": [example["input"] for example in sampled_examples],
        "answer": [example["target"] for example in sampled_examples],
    }
)

# Extract the base name of the input file (without extension)
base_name = os.path.splitext(os.path.basename(input_file_path))[0]

# Construct the output file path using the base name
output_file_path = f"../Dataset_format/BBH/{base_name}.csv"

# Save the DataFrame to a CSV file
df.to_csv(output_file_path, index=False, encoding="utf-8")

# Print a confirmation message with the output file path
print(f"Data has been successfully saved to {output_file_path}.")
