import pandas as pd

# input csv
df = pd.read_csv("../Dataset/C-EVAL/clinical_medicine_val.csv")

df["question"] = (
    df["question"]
    + "\nOptions:\n"
    + "(A) "
    + df["A"].astype(str)
    + "\n"
    + "(B) "
    + df["B"].astype(str)
    + "\n"
    + "(C) "
    + df["C"].astype(str)
    + "\n"
    + "(D) "
    + df["D"].astype(str)
)

df["answer"] = "(" + df["answer"].astype(str) + ")"

new_df = df[["question", "answer"]]

new_df.to_csv("../Dataset_format/C-EVAL/clinical_medicine_val.csv", index=False)

print(
    "The conversion is complete and the new CSV file has been saved as 'clinical_medicine_val.csv'"
)
