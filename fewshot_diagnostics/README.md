# Few-shot Diagnostic Tables

- paper run: `D:\myx\grade_one\experiments\MARS\paper_full\run_20260709_142043`
- strict run: `D:\myx\grade_one\experiments\MARS\paper_full_strict\run_20260709_212917`
- fixed run: `missing`
- meaningful delta: `0.02`

## Table 1. Protocols

|Protocol|Few-shot source|Test overlap|Role|
|---|---|---|---|
|paper_mode CoT-FS|splits["opt"]|yes|Paper-aligned main result; CoT-FS is strongest but opt/val/test share rows.|
|strict_mode CoT-FS|splits["opt"]|no|Removes exact opt/test overlap while preserving split-selected demonstrations.|
|strict_mode CoT-FS-fixed|few_shot.jsonl filtered by test sample_ids|no|Fixed demonstrations; removes split-specific demonstration selection.|

## Table 2. Task Accuracy

|Task|paper CoT-FS|paper MARS|paper MARS-CoT|strict CoT-FS|strict MARS|strict MARS-CoT|strict CoT-FS-fixed|fixed-strict CoT|MARS-fixed|
|---|---|---|---|---|---|---|---|---|---|
|Boolean Expressions|0.9799|0.9157|-0.0643|0.9680|0.9200|-0.0480||||
|Disambiguation QA|0.7831|0.6867|-0.0964|0.7280|0.7040|-0.0240||||
|Formal Fallacies|0.8112|0.7550|-0.0562|0.8000|0.7600|-0.0400||||
|Geometric Shapes|0.4980|0.4458|-0.0522|0.4960|0.5040|0.0080||||
|Ruin Names|0.8755|0.9076|0.0321|0.8960|0.9360|0.0400||||
|Sports Understanding|0.8474|0.7871|-0.0602|0.8400|0.8240|-0.0160||||
|Average|0.7992|0.7497|-0.0495|0.7880|0.7747|-0.0133||||

## Table 3. Interpretation

|Case|Condition|Status|Evidence|Interpretation|
|---|---|---|---|---|
|A|cot_fs_fixed clearly below strict CoT-FS|pending|cot_fs_fixed real results are not available yet.|Run cot_fs_fixed to test whether split-selected opt examples drive CoT-FS strength.|
|B|cot_fs_fixed close to strict CoT-FS|pending|cot_fs_fixed real results are not available yet.|Run cot_fs_fixed to test whether few-shot prompting itself is the main driver.|
|C|cot_fs_fixed below strict MARS|pending|cot_fs_fixed real results are not available yet.|Run cot_fs_fixed to test whether MARS beats fixed non-overlapping demonstrations.|

