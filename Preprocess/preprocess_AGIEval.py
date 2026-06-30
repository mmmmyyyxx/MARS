import csv
import json

input_jsonl_file = "../Dataset/AGIEval/lsat-ar.jsonl"
output_csv_file = "../Dataset_format/AGIEval/lsat-ar_format.csv"

with (
    open(input_jsonl_file, "r", encoding="utf-8") as jsonl_file,
    open(output_csv_file, "w", newline="", encoding="utf-8") as csv_file,
):

    csv_writer = csv.writer(csv_file)

    csv_writer.writerow(["question", "answer"])

    for line in jsonl_file:

        data = json.loads(line.strip())

        # Combining passages, questions, and options
        options_text = "\n".join(data["options"])
        question = f"{data['passage']}\nquestions:\n{data['question']}\nOptions:\n{options_text}"

        # get label
        answer = f"({data['label']})"

        csv_writer.writerow([question, answer])

print(f"The conversion is complete and the result has been saved to {output_csv_file}")
