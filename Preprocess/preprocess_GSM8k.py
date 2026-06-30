import csv
import re

import pandas as pd

##
test_dataset = pd.read_parquet("../Dataset/GSM8K/test-00000-of-00001.parquet")


# Extract the contents of #### followed by
def extract_answer(answer):
    match = re.search(r"####\s*(\S+)", answer)
    if match:
        return match.group(1)
    return None


test_dataset["new_answer"] = test_dataset["answer"].apply(extract_answer)

new_dataset = test_dataset[["question", "new_answer"]]
new_dataset.columns = ["question", "answer"]


new_dataset.to_csv(
    "../Dataset_format/GSM8K/gsm8k_test.csv",
    index=False,
    encoding="utf-8",
    quoting=csv.QUOTE_ALL,
)

print("Save as gsm8k_test.csv")
