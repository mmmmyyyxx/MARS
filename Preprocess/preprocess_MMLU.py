import csv

input_file = "../Dataset/MMLU/test/human_aging_test.csv"
output_file = "../Dataset_format/MMLU/human_aging_test.csv"

with open(input_file, mode="r", encoding="utf-8") as infile:
    reader = csv.reader(infile)
    rows = list(reader)


formatted_data = []
for row in rows:
    # Extract questions and options
    question = row[0]
    option_a = f"(A){row[1]}"
    option_b = f"(B){row[2]}"
    option_c = f"(C){row[3]}"
    option_d = f"(D){row[4]}"

    # Portfolio issues and options
    formatted_question = (
        f"{question}\n\nOptions:\n{option_a}\n{option_b}\n{option_c}\n{option_d}\n"
    )

    answer = row[5]
    formatted_answer = f"({answer})"

    formatted_data.append([formatted_question, formatted_answer])


with open(output_file, mode="w", encoding="utf-8", newline="") as outfile:
    writer = csv.writer(outfile)

    writer.writerow(["question", "answer"])

    writer.writerows(formatted_data)

print(f"Save to {output_file}")
