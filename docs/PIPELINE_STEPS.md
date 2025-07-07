# Detailed Pipeline Steps

## Step 1: Preprocessing (Data Cleaning)

**What it does:** The preprocessing step cleans and normalizes the raw data, and removes records that duplicate the same company or domain. It will:

* **Normalize text fields** such as company names, website domains, and phone numbers into a standard format (lowercasing, removing accents, stripping URLs, etc.). For example, "Acme Inc" and "acme inc." would be normalized to the same representation.
* **Ensure required columns exist:** The script will check that a unique ID and a company name column are present. If the input has `sys_id` instead of `record_id`, it will rename it automatically. If the company column is named differently (e.g. "Name"), it will be detected and used. Missing critical columns will result in an error instructing you to add or rename that column.
* **Optionally translate company names to English:** If you have company names in multiple languages, you can use the `--use-openai` flag to call OpenAI GPT for translation. By default this uses a model called `"gpt-4o-mini"` (you can specify a different model with `--openai-model`). This will send each company name to the API and replace it with an English version using Latin characters. (Ensure you installed `openai` and set your API key as described above, otherwise an error will be raised.)
* **Create a combined key for each row:** A `combined_id` is built from the normalised company name, domain and phone number to give every record a unique signature.
* **Track merged record IDs:** Another column `merged_ids` lists all `record_id` values that share the same normalised company or domain, joined with semicolons so you can see which rows were grouped together.
* **Remove obvious duplicate records:** Rows that share the same normalised company name **or** the same normalised domain are considered duplicates. Only the first occurrence is kept. Dropped rows are saved with a note for review.
* **Save cleaned output:** The cleaned dataset (after dropping these duplicates) is written to **`data/outputs/cleaned.csv`**, and any removed duplicate rows are saved to **`data/outputs/removed_rows.csv`**. The removed rows file includes a column (e.g., `reason`) explaining why those records were filtered out (in this case, `"duplicate company or domain"`).

**After running this step:** Check the `data/outputs` folder. You should see:

* `cleaned.csv` – the processed data ready for the next step.
* `removed_rows.csv` – any dropped duplicates (if no rows were dropped, this file may have only headers or be empty).
* `run_history.log` (in `data/`) – a log file where each pipeline step appends the stage name, start/end times, number of rows processed, and duration.

## Step 2: Blocking (Candidate Generation)

**What it does:** The blocking step generates candidate record pairs for comparison by grouping records on certain keys. The idea is to avoid comparing every record with every other (which can be very slow for large datasets) by **"blocking"** on one or more fields that likely match for true duplicates. In this pipeline, blocking is intended to be based on the normalized **phone number** and **company name**: only records sharing the same phone or company block will be considered as potential duplicates.

Concretely, this stage uses the `recordlinkage` library to create an index of candidate pairs. Records with identical values in the chosen blocking fields are paired up for further comparison. For example, all records with the same normalized phone number might form one block of candidates, and similarly for company names. Any record that doesn't share a block with another will not be considered a duplicate (reducing false comparisons).

**Output:** Running this step now writes a CSV `pairs.csv` in the `data/outputs` directory containing the record ID pairs produced by blocking. Each row lists `record_id_1` and `record_id_2` so you can inspect which records will be compared in later stages.

**What to watch out for:** If your dataset has no values in one of the blocking fields (e.g., if `phone_clean` is empty for all records), the blocking step will automatically adapt and only use available fields for blocking. The pipeline has been updated to handle minimal datasets gracefully:

- **Minimal Data Handling:** If you only have company names and IDs, the blocking will only use company name similarity for generating candidate pairs
- **Missing Optional Fields:** The pipeline automatically creates empty columns for missing fields (`domain_clean`, `phone_clean`, `address_clean`) so subsequent steps work correctly
- **Reduced Accuracy Warning:** With fewer fields available, duplicate detection accuracy may be lower. The pipeline will still work but may miss some duplicates or generate more false positives

The default blocking on phone, domain, and company works best when those fields are populated. For minimal datasets with only company names, consider:
- Ensuring company names are as clean and consistent as possible in your input data
- Using the OpenAI translation feature (`--use-openai`) if company names are in multiple languages
- Manually reviewing more of the results since confidence scores may be less reliable with limited features

## Step 3: Similarity Features

**What it does:** This step takes all candidate pairs (likely produced by the blocking step) and computes a set of **similarity features** for each pair. These features quantify how alike two records are in various dimensions, and they will be used by the model in the next step to decide if a pair is a duplicate. Typical similarity metrics include:

* **Exact matches** (e.g., whether two records have the exact same phone number or domain),
* **String similarity scores** for names or addresses (to catch slight differences or typos), and
* Other custom comparisons as needed.

In this pipeline, we leverage the `recordlinkage.Compare` class and the `rapidfuzz` library for string matching. For example, the code plans to calculate token set ratio similarity for address fields (if an address or similar free-text field is present). A token set ratio is a fuzzy matching score that is insensitive to word order, which is helpful for comparing addresses like "123 Main St." vs "Main St 123". Company or domain fields might be compared for exact equality or through a similar text similarity metric. Each comparison produces a numeric feature (often between 0 and 1) indicating how similar or different the two records are in that aspect.

**Output:** After running, you should get a file **`data/outputs/features.csv`** containing both the similarity features and the cleaned company name, domain, phone, address, state and country code for each record in a pair. This single file is the input to the machine learning model in the next stage. Inspecting `features.csv` can be useful for debugging – for example, to check the cleaned values side by side with the numeric similarity scores.

**What to watch out for:** The number of candidate pairs can grow quickly with large datasets. If your dataset is very large, consider using more aggressive blocking or filtering out obviously non-matching pairs prior to this step to keep the number of comparisons manageable. Also, ensure that the data has been sufficiently cleaned in the preprocessing step; inconsistent formatting that wasn't addressed (e.g., one address including an apartment number and the other not) can lower the similarity scores.

**Minimal Data Considerations:** When working with minimal datasets (just company name and ID):
- The pipeline will only compute `company_sim` (company name similarity) features
- Optional features like `domain_sim`, `phone_exact`, and `address_sim` will be skipped if those columns don't exist or are empty
- The resulting `features.csv` will have fewer columns but the model training will still work
- Consider the trade-off: fewer features mean the model has less information to distinguish duplicates accurately

The features computed are quite basic in this minimal pipeline – developers can enhance this step by adding more domain-specific similarity functions (for example, phonetic comparison for names, numeric difference for numeric fields like age or price, etc.).

## Step 4: Model Training (Duplicate Scoring)

**What it does:** In this step, a simple **machine learning model** is trained to distinguish duplicates from non-duplicates using the similarity features from Step 3. The pipeline uses a **logistic regression** classifier (from scikit-learn) as the model. Logistic regression will assign weights to each similarity feature to best separate true duplicate pairs from false pairs, and output a probability (between 0 and 1) that each candidate pair is a duplicate.

For training this model, you ideally need a set of example pairs labeled as "duplicate" or "not duplicate" – this is your training data (`labels.csv`). If you already have some known duplicate pairs in your dataset (or a subset that was manually labeled), you can prepare a CSV of labels for those pairs. The pipeline expects labels in **`data/outputs/labels.csv`** by default. The exact format may be for example: a row with the two record IDs and a label (`1` for duplicate, `0` for not).

If the `labels.csv` file is missing, the script now falls back to a simple **unsupervised mode**. In this mode it automatically labels pairs that look like exact duplicates (very high similarity across all features) as positives and pairs that look obviously different as negatives. These heuristic labels allow the logistic regression to train even without manual input, though providing real labels is still recommended for best results.

**Note:** This fallback only works if your `features.csv` contains at least one pair that is almost certainly a duplicate and one pair that is clearly not. If no such extremes exist, the model cannot create the heuristic labels and will fail with an error like `"Labels file not found and insufficient heuristic examples for training"`. In that case, create a small `labels.csv` with at least one positive and one negative row.

Once trained (or even without explicit training data), the logistic model will **score all candidate pairs**. It will add a new column (let's say `prob` or `score`) to the features table indicating the predicted probability that each pair is a duplicate. The pipeline will then identify **high-confidence duplicates**, e.g. pairs with a probability above a certain threshold, and save them for reporting.

**Output:** After running, you should see:

* **`data/outputs/high_confidence.csv`** – a list of the candidate pairs that the model flagged as likely duplicates with high confidence. This is essentially the subset of the features table with an added probability column and filtered by a probability threshold. You might also see:
* An updated **`data/outputs/features.csv`** – possibly now including the new `prob` column for all pairs (depending on implementation of saving the results).
* **`data/outputs/model.joblib`** – the trained logistic regression model saved for reuse.

By inspecting `high_confidence.csv`, you can get a quick sense of which records the pipeline thinks are duplicates. Each row should reference two record IDs (from the original `record_id` field) and might include their similarity features and the predicted probability. High-confidence duplicates might be those with probability > 0.9 (90%) or a similar threshold that was chosen in the code. You can adjust this threshold by modifying the code in `src/model.py` if needed.

**What to watch out for:** The effectiveness of this step depends heavily on the quality of your training data (labels) and features. If you provide no labels, the model isn't truly learning what a duplicate looks like in your context – it might default to some baseline or require you to manually set a threshold on features instead. If the model's output doesn't seem accurate (e.g., it's flagging many false positives or missing obvious duplicates), consider improving the training labels:

* Add more confirmed duplicate examples to `labels.csv` (and some confirmed non-duplicates) to better train the model.
* Check if the features are informative enough – you might need to add additional features (e.g., numeric differences, geographic distance if you have location data, etc.) for the model to use.
* **For Minimal Datasets:** If you only have company name features, the model has limited information to work with. Consider:
  - Being more conservative with confidence thresholds (maybe 0.95+ instead of 0.9)
  - Manually reviewing more pairs in the medium confidence range
  - Adding more company name variations to your labels if available
  - Using the clustering approach (Step 6) as an alternative to the supervised model
* As a simplistic alternative, you could bypass model training and just use one of the similarity scores with a threshold (for instance, mark any pair with company name similarity > 0.95 as duplicate). The provided framework is meant to be a starting point that you can tune or extend.

## Step 5: Reporting (Review and Export)

**What it does:** The final core step is to produce a human-readable **report of duplicate suggestions**. After the model identifies likely duplicates, this step will gather those pairs and present them side-by-side in an Excel workbook for you to review manually. The idea is to make it easy for a user (or data steward) to verify the suggested merges before applying them to the original dataset.

The reporting module will take the high-confidence duplicate pairs (`high_confidence.csv`) and the cleaned data, and for each duplicate pair, retrieve the full original records from the cleaned dataset. It then creates a spreadsheet (Excel `.xlsx` file) where each row contains two records that are deemed duplicates, with their details in adjacent columns. This way, you can scroll through the file and visually compare each pair – checking if they truly look like duplicates or if the algorithm made a mistake.

**Output:** The main output is **`manual_review.xlsx`** in the project root or in the `data/outputs/` directory (check the README or code for the exact location; typically it would be placed with the outputs). In this Excel workbook, you'll find all the pairs of records that were identified as likely duplicates. Each pair is presented with original field values for side-by-side comparison. There may also be an indication of the model's confidence score or key matching fields to help you judge.

After this stage, **your involvement as a human reviewer is important** – go through `manual_review.xlsx` and decide for each pair if they are indeed duplicates. If they are, you will want to merge or consolidate those records in your source data. If not, you can ignore that suggestion. The pipeline does **not** automatically modify your original dataset; it just provides recommendations. This gives you full control to avoid incorrect merges.

**What to watch out for:** Ensure that the `high_confidence.csv` exists and is not empty before running this step; if the model didn't flag any duplicates, the report might be empty or the script might simply produce an empty spreadsheet. If the reporting step fails, it might be due to unexpected data formats in the CSV or missing files. Also note that the Excel is generated using `pandas` with OpenPyXL/XlsxWriter, so very large numbers of duplicate pairs might lead to a large file – if you had hundreds of thousands of suggestions (unlikely in practice after blocking and scoring), you might instead want to output a CSV or handle the review in a database. But for most use cases, the Excel output is convenient and easy to filter or annotate as you review the results.

## Step 6: Clustering (Optional Grouping)

**What it does:** Rather than evaluating pairs with a model, you can cluster records directly based on their similarity features. The `src/clustering.py` module uses DBSCAN to assign each record to a cluster using the mean of its pairwise similarity scores. 

**Note:** In the current implementation, the `domain_sim` feature is given double weight before clustering. This means that domain similarity will have a stronger influence on the clustering results compared to other features. If you want to adjust the weighting, you can modify the code in `src/clustering.py`.

Enabling `--scale` standardises feature ranges so each metric contributes equally (after weighting).

Running this step creates **`clusters.csv`** with the cluster label for every record and **`agg_features.csv`** with the aggregated feature matrix that includes that label.

**How to run:**

```bash
python -m src.clustering --eps 0.5 --min-samples 2 --scale
```

Tune the parameters for your dataset. The resulting clusters can be inspected directly or reviewed with GPT as described in [GPT_INTEGRATION.md](GPT_INTEGRATION.md).
