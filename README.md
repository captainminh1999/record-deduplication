# Spreadsheet Deduplication Pipeline

This repository provides a **minimal record deduplication workflow** in Python, designed to identify and merge duplicate entries in spreadsheet data. Duplicate records (e.g. the same company or customer appearing twice with slight variations) can clutter databases and analyses. This pipeline offers a step-by-step solution: it cleans and normalizes raw data, generates candidate record pairs to compare, computes similarity features, trains a simple model to score duplicates, and finally produces an easy-to-review report of high-confidence duplicate pairs. The goal is to help both non-technical users deduplicate their data with straightforward commands and to give technical users a clear, extensible framework for record linkage.

## Installation and Setup

**Prerequisites:** Ensure you have **Python 3** installed (the code uses modern Python features) and `pip` for package management. We recommend setting up a virtual environment for this project.

**1. Clone the repository:** If you haven't already, download or clone this repository to your local machine.

```bash
git clone https://github.com/captainminh1999/record-deduplication.git
cd record-deduplication
```

**2. Install dependencies:** Use pip to install the required libraries from the provided `requirements.txt` file:

```bash
python -m pip install -r requirements.txt
```

This will install pandas, NumPy, scikit-learn, `recordlinkage` (for pairing records), `rapidfuzz` (for string similarity), and Excel output libraries like OpenPyXL. These are the core dependencies needed for the pipeline to run.

**3. (Optional) Enable GPT integration:** If you plan to use the optional OpenAI GPT-powered features, you need to install the OpenAI package and have an API key. The `openai` library is commented out in requirements (not installed by default). To include it, run:

```bash
pip install openai
```

Also, set your OpenAI API key as an environment variable (`OPENAI_API_KEY`) or configure it in your code before use. *Skip this step if you don’t intend to use the GPT features.* (More on this in the [Optional: GPT Integration](#optional-gpt-integration) section.)

## Usage Guide: Step-by-Step Workflow

Using the pipeline involves running a series of modular Python scripts in order. Each stage reads the output of the previous stage and writes its results to the **`data/outputs/`** directory. By following the steps below, even users with little Python experience can deduplicate their dataset. For illustration, a sample dataset (`data/sample_input.csv`) is provided — you can use it to test the pipeline or as a template for your own data.

**Before you start:** Prepare your input spreadsheet as a CSV file. By default, the pipeline expects **`data/your_spreadsheet.csv`** as the input. Make sure your dataset has a unique **record identifier** column named `record_id` (if your file uses a different name like `sys_id`, the code will handle it by mapping that to `record_id`). Also ensure there is a **company name** column (or a similarly purposed name column) in your data. Other fields like domain (website) and phone number are optional but recommended for better deduplication results. The sample data includes columns `company`, `domain`, `phone`, and `address` along with `record_id`.

Now, execute the following pipeline steps **in order**:

### Step 1: Preprocessing (Data Cleaning)

**What it does:** The preprocessing step cleans and normalizes the raw data, and removes records that duplicate the same company or domain. It will:

* **Normalize text fields** such as company names, website domains, and phone numbers into a standard format (lowercasing, removing accents, stripping URLs, etc.). For example, "Acme Inc" and "acme inc." would be normalized to the same representation.
* **Ensure required columns exist:** The script will check that a unique ID and a company name column are present. If the input has `sys_id` instead of `record_id`, it will rename it automatically. If the company column is named differently (e.g. "Name"), it will be detected and used. Missing critical columns will result in an error instructing you to add or rename that column.
* **Optionally translate company names to English:** If you have company names in multiple languages, you can use the `--use-openai` flag to call OpenAI GPT for translation. By default this uses a model called `"gpt-4o-mini"` (you can specify a different model with `--openai-model`). This will send each company name to the API and replace it with an English version using Latin characters. (Ensure you installed `openai` and set your API key as described above, otherwise an error will be raised.)
* **Create a combined key for each row:** A `combined_id` is built from the normalised company name, domain and phone number to give every record a unique signature.
* **Track merged record IDs:** Another column `merged_ids` lists all `record_id` values that share the same normalised company or domain, joined with semicolons so you can see which rows were grouped together.
* **Remove obvious duplicate records:** Rows that share the same normalised company name **or** the same normalised domain are considered duplicates. Only the first occurrence is kept. Dropped rows are saved with a note for review.
* **Save cleaned output:** The cleaned dataset (after dropping these duplicates) is written to **`data/outputs/cleaned.csv`**, and any removed duplicate rows are saved to **`data/outputs/removed_rows.csv`**. The removed rows file includes a column (e.g., `reason`) explaining why those records were filtered out (in this case, `"duplicate company or domain"`).

**How to run:** Execute the preprocessing module with Python. From the repository root directory, run:

```bash
python -m src.preprocess --input-path data/your_spreadsheet.csv --use-openai
```

Replace the `--input-path` value if your input file is named differently or located elsewhere. Omit `--use-openai` if you do not want to use the GPT translation (for example, if all company names are already consistently formatted). You can also add `--clear` to this (or any) step to **clear previous outputs** before running, which is useful if you want to reset the pipeline and start fresh. By default, running with `--clear` will delete old output files in `data/outputs` so you get a clean slate for the new run.

**After running this step:** Check the `data/outputs` folder. You should see:

* `cleaned.csv` – the processed data ready for the next step.
* `removed_rows.csv` – any dropped duplicates (if no rows were dropped, this file may have only headers or be empty).
* `run_history.log` – a log file where each pipeline step appends the stage name, start/end times, number of rows processed, and duration.

> **Example:** Using the provided `sample_input.csv` (already in the `data/` folder) as input will result in a `cleaned.csv` with 2 records and a `removed_rows.csv` containing the two rows that shared the same company or domain. The log file will note the preprocessing step and how many rows remained.

### Step 2: Blocking (Candidate Generation)

**What it does:** The blocking step generates candidate record pairs for comparison by grouping records on certain keys. The idea is to avoid comparing every record with every other (which can be very slow for large datasets) by **“blocking”** on one or more fields that likely match for true duplicates. In this pipeline, blocking is intended to be based on the normalized **phone number** and **company name**: only records sharing the same phone or company block will be considered as potential duplicates.

Concretely, this stage uses the `recordlinkage` library to create an index of candidate pairs. Records with identical values in the chosen blocking fields are paired up for further comparison. For example, all records with the same normalized phone number might form one block of candidates, and similarly for company names. Any record that doesn’t share a block with another will not be considered a duplicate (reducing false comparisons).

**How to run:** Execute the blocking module:

```bash
python -m src.blocking
```

By default, it will read the cleaned data from **`data/outputs/cleaned.csv`**. Ensure that the preprocessing step has been run and that `cleaned.csv` exists; otherwise, this step will raise an error if it cannot find the input file. The script will also validate that the necessary normalized columns (`company_clean` and `phone_clean`) are present in the cleaned data. (These columns are created during preprocessing. If either is missing, it means the preprocessing step was not run correctly or did not produce the expected output.)

**Output:** Running this step now writes a CSV `pairs.csv` in the `data/outputs` directory containing the record ID pairs produced by blocking. Each row lists `record_id_1` and `record_id_2` so you can inspect which records will be compared in later stages.

**What to watch out for:** If your dataset has no values in one of the blocking fields (e.g., if `phone_clean` is empty for all records), the blocking step might not effectively reduce candidate pairs (worst-case, every record could be paired with every other). In such cases, you might consider adding another blocking key or ensuring at least one of the chosen fields has useful data for grouping. The default blocking on phone and company works well if those fields are generally populated; if not, you can modify `src/blocking.py` to use a different blocking strategy (for example, blocking on email domain or zip code, depending on your data).

### Step 3: Similarity Features

**What it does:** This step takes all candidate pairs (likely produced by the blocking step) and computes a set of **similarity features** for each pair. These features quantify how alike two records are in various dimensions, and they will be used by the model in the next step to decide if a pair is a duplicate. Typical similarity metrics include:

* **Exact matches** (e.g., whether two records have the exact same phone number or domain),
* **String similarity scores** for names or addresses (to catch slight differences or typos), and
* Other custom comparisons as needed.

In this pipeline, we leverage the `recordlinkage.Compare` class and the `rapidfuzz` library for string matching. For example, the code plans to calculate token set ratio similarity for address fields (if an address or similar free-text field is present). A token set ratio is a fuzzy matching score that is insensitive to word order, which is helpful for comparing addresses like "123 Main St." vs "Main St 123". Company or domain fields might be compared for exact equality or through a similar text similarity metric. Each comparison produces a numeric feature (often between 0 and 1) indicating how similar or different the two records are in that aspect.

**How to run:** Execute the similarity module:

```bash
python -m src.similarity
```

This will read the cleaned data (and potentially reuse the blocking logic internally to form candidate pairs). Make sure you have run the blocking step prior, or that you run similarity soon after preprocessing so that it can generate pairs. The script expects the cleaned data at **`data/outputs/cleaned.csv`** (and possibly the blocking index from the previous step, although in this implementation it might generate it on the fly). If `cleaned.csv` is missing, or if for some reason required columns for comparison are missing, this step will not run. For instance, if you removed or renamed columns like `address` or `company_clean`, you should adjust the code accordingly or restore those columns.

**Output:** After running, you should get a file **`data/outputs/features.csv`** containing both the similarity features and the cleaned company name, domain, phone, address, state and country code for each record in a pair. This single file is the input to the machine learning model in the next stage. Inspecting `features.csv` can be useful for debugging – for example, to check the cleaned values side by side with the numeric similarity scores.


**What to watch out for:** The number of candidate pairs can grow quickly with large datasets. If your dataset is very large, consider using more aggressive blocking or filtering out obviously non-matching pairs prior to this step to keep the number of comparisons manageable. Also, ensure that the data has been sufficiently cleaned in the preprocessing step; inconsistent formatting that wasn’t addressed (e.g., one address including an apartment number and the other not) can lower the similarity scores. The features computed are quite basic in this minimal pipeline – developers can enhance this step by adding more domain-specific similarity functions (for example, phonetic comparison for names, numeric difference for numeric fields like age or price, etc.).

### Step 4: Model Training (Duplicate Scoring)

**What it does:** In this step, a simple **machine learning model** is trained to distinguish duplicates from non-duplicates using the similarity features from Step 3. The pipeline uses a **logistic regression** classifier (from scikit-learn) as the model. Logistic regression will assign weights to each similarity feature to best separate true duplicate pairs from false pairs, and output a probability (between 0 and 1) that each candidate pair is a duplicate.

For training this model, you ideally need a set of example pairs labeled as "duplicate" or "not duplicate" – this is your training data (`labels.csv`). If you already have some known duplicate pairs in your dataset (or a subset that was manually labeled), you can prepare a CSV of labels for those pairs. The pipeline expects labels in **`data/outputs/labels.csv`** by default. The exact format may be for example: a row with the two record IDs and a label (`1` for duplicate, `0` for not).

If the `labels.csv` file is missing, the script now falls back to a simple **unsupervised mode**. In this mode it automatically labels pairs that look like exact duplicates (very high similarity across all features) as positives and pairs that look obviously different as negatives. These heuristic labels allow the logistic regression to train even without manual input, though providing real labels is still recommended for best results.

**Note:** This fallback only works if your `features.csv` contains at least one pair that is almost certainly a duplicate and one pair that is clearly not. If no such extremes exist, the model cannot create the heuristic labels and will fail with an error like `"Labels file not found and insufficient heuristic examples for training"`. In that case, create a small `labels.csv` with at least one positive and one negative row.

Once trained (or even without explicit training data), the logistic model will **score all candidate pairs**. It will add a new column (let's say `prob` or `score`) to the features table indicating the predicted probability that each pair is a duplicate. The pipeline will then identify **high-confidence duplicates**, e.g. pairs with a probability above a certain threshold, and save them for reporting.

**How to run:** Execute the model training module:

```bash
python -m src.model
```

This assumes that `features.csv` (from the previous step) and `labels.csv` are present in the `data/outputs` directory. The script will load the features and labels, train the `LogisticRegression` model, and then produce scores for all pairs. If `labels.csv` is missing, it will automatically generate heuristic labels and train in unsupervised mode instead. Providing real labels is still preferred when possible.

**Output:** After running, you should see:

* **`data/outputs/dupes_high_conf.csv`** – a list of the candidate pairs that the model flagged as likely duplicates with high confidence. This is essentially the subset of the features table with an added probability column and filtered by a probability threshold. You might also see:
* An updated **`data/outputs/features.csv`** – possibly now including the new `prob` column for all pairs (depending on implementation of saving the results).
* The logistic model itself is not saved in this pipeline (training is fast enough to run as needed), but developers could easily modify the code to save the model to disk if desired.

By inspecting `dupes_high_conf.csv`, you can get a quick sense of which records the pipeline thinks are duplicates. Each row should reference two record IDs (from the original `record_id` field) and might include their similarity features and the predicted probability. High-confidence duplicates might be those with probability > 0.9 (90%) or a similar threshold that was chosen in the code. You can adjust this threshold by modifying the code in `src/model.py` if needed.

**What to watch out for:** The effectiveness of this step depends heavily on the quality of your training data (labels) and features. If you provide no labels, the model isn’t truly learning what a duplicate looks like in your context – it might default to some baseline or require you to manually set a threshold on features instead. If the model’s output doesn’t seem accurate (e.g., it’s flagging many false positives or missing obvious duplicates), consider improving the training labels:

* Add more confirmed duplicate examples to `labels.csv` (and some confirmed non-duplicates) to better train the model.
* Check if the features are informative enough – you might need to add additional features (e.g., numeric differences, geographic distance if you have location data, etc.) for the model to use.
* As a simplistic alternative, you could bypass model training and just use one of the similarity scores with a threshold (for instance, mark any pair with company name similarity > 0.95 and phone match as duplicate). The provided framework is meant to be a starting point that you can tune or extend.

### Step 5: Reporting (Review and Export)

**What it does:** The final core step is to produce a human-readable **report of duplicate suggestions**. After the model identifies likely duplicates, this step will gather those pairs and present them side-by-side in an Excel workbook for you to review manually. The idea is to make it easy for a user (or data steward) to verify the suggested merges before applying them to the original dataset.

The reporting module will take the high-confidence duplicate pairs (`dupes_high_conf.csv`) and the cleaned data, and for each duplicate pair, retrieve the full original records from the cleaned dataset. It then creates a spreadsheet (Excel `.xlsx` file) where each row contains two records that are deemed duplicates, with their details in adjacent columns. This way, you can scroll through the file and visually compare each pair – checking if they truly look like duplicates or if the algorithm made a mistake.

**How to run:** Execute the reporting module:

```bash
python -m src.reporting
```

This will read **`data/outputs/dupes_high_conf.csv`** and **`data/outputs/cleaned.csv`** by default. It will then create the Excel file with merge suggestions.

**Output:** The main output is **`merge_suggestions.xlsx`** in the project root or in the `data/outputs/` directory (check the README or code for the exact location; typically it would be placed with the outputs). In this Excel workbook, you'll find all the pairs of records that were identified as likely duplicates. Each pair is presented with original field values for side-by-side comparison. There may also be an indication of the model’s confidence score or key matching fields to help you judge.

After this stage, **your involvement as a human reviewer is important** – go through `merge_suggestions.xlsx` and decide for each pair if they are indeed duplicates. If they are, you will want to merge or consolidate those records in your source data. If not, you can ignore that suggestion. The pipeline does **not** automatically modify your original dataset; it just provides recommendations. This gives you full control to avoid incorrect merges.

**What to watch out for:** Ensure that the `dupes_high_conf.csv` exists and is not empty before running this step; if the model didn’t flag any duplicates, the report might be empty or the script might simply produce an empty spreadsheet. If the reporting step fails, it might be due to unexpected data formats in the CSV or missing files. Also note that the Excel is generated using `pandas` with OpenPyXL/XlsxWriter, so very large numbers of duplicate pairs might lead to a large file – if you had hundreds of thousands of suggestions (unlikely in practice after blocking and scoring), you might instead want to output a CSV or handle the review in a database. But for most use cases, the Excel output is convenient and easy to filter or annotate as you review the results.

### Optional: GPT Integration

*(This step is optional and not required for the core deduplication pipeline. Non-technical users can skip this unless they are interested in AI-assisted data cleaning.)*

The project includes an **optional integration with OpenAI's GPT** for enhanced data cleaning and merging assistance. There are two places GPT can help:

1. **Data Normalization:** As mentioned in preprocessing, GPT can translate or normalize company names (or other text fields) to a consistent language and character set. This can be very useful if your dataset contains entries in multiple languages or scripts (e.g., "株式会社ABC" to "ABC Corp"). By using the `--use-openai` flag in Step 1, the `translate_to_english` function will be called to handle this. Keep in mind this will call an external API for each entry, which may be slower and will require API credits.
2. **Merge Suggestion Prompting:** In the future (or if you extend the code), GPT could be used to directly suggest how to merge two records or to justify why they are duplicates. The module `src/openai_integration.py` contains a placeholder for how one might formulate a prompt describing two potentially duplicate records and ask GPT for a decision or suggestion. For example, you could feed the details of two records to GPT and ask "Are these two entries likely the same entity? If so, suggest a consolidated record." This is not fully implemented in the current pipeline (the function is a stub to demonstrate where such logic would go), but it provides a blueprint for those interested in experimenting with AI in record deduplication.

**How to use GPT features:** To use the translation feature, simply include `--use-openai` when running the preprocess step (and ensure `openai` library is installed and API key configured). If you wanted to experiment with the merge suggestion via GPT, you would run:

```bash
python -m src.openai_integration
```

This module as provided does a minimal check and uses the `translate_to_english` on a hardcoded example. Developers can expand this to actually read in the duplicate suggestions and call the API. Remember to handle costs and privacy – sending record details to the OpenAI API means that data is leaving your environment, so avoid this for sensitive data unless you have the right agreements in place.

**Caution:** Using GPT is powerful but comes with considerations:

* **Cost:** OpenAI API calls are not free. Translating every company name or querying for every pair can accumulate costs.
* **Speed:** API calls introduce latency. Preprocessing with GPT will be slower than without, especially for large datasets.
* **Accuracy:** GPT can sometimes produce inconsistent results, especially for very ambiguous cases. Always verify AI-generated suggestions.
* **Setup:** As noted, you must have an API key set (the code will error if it's not found) and the `openai` package installed.

If unsure, you can achieve a lot with the regular, non-AI pipeline steps. The GPT integration is there for advanced use cases and to inspire further enhancements.

## Project Structure

Understanding the repository structure will help you navigate the code and data:

* **`data/`** – Contains input and output data files.

  * **`your_spreadsheet.csv`** – This is the default input file path for your raw data. You should place your own dataset here (or specify a different path when running the commands). It can be a CSV exported from Excel or any source. *(If your data is in Excel format (.xlsx), you may convert it to CSV first for use with this pipeline. The code uses `pandas.read_csv` by default to load data.)*
  * **`sample_input.csv`** – A provided sample dataset for testing and understanding the pipeline. You can run the entire pipeline on this sample to see how it works end-to-end.
  * **`outputs/`** – This sub-directory is where all intermediate and final outputs will be written. Important files include:

    * `cleaned.csv` – Cleaned data after preprocessing (duplicate entries removed, new normalized columns added).
    * `removed_rows.csv` – Records that were dropped because another row shared the same normalised company or domain. The `merged_ids` column lists the related record IDs.
    * `pairs.csv` – Record ID pairs generated by the blocking step.
    * `features.csv` – Similarity feature matrix for candidate pairs (output of the similarity step).
    * `labels.csv` – *(Optional)* Label data for training the model (you might prepare this manually if you have known duplicates for supervised learning).
    * `dupes_high_conf.csv` – High-confidence duplicate pairs identified by the model, to be reviewed.
    * `merge_suggestions.xlsx` – Excel report generated in the reporting step, listing likely duplicates side by side for human review.
    * `run_history.log` – A log file appending a line each time you run a pipeline step with the `--log-path` option or by default. It records the step name, start/end time, number of records processed, and duration. This is useful for tracking the pipeline execution over time or debugging performance.

* **`src/`** – The source code for the pipeline, organized by stage:

  * `preprocess.py` – Step 1: Data cleaning and normalization script (adds `record_id` field, normalizes company/domain/phone, removes duplicates, writes `cleaned.csv`).
  * `blocking.py` – Step 2: Blocking script (reads `cleaned.csv`, forms candidate pairs by blocking on keys like phone and company, using the `recordlinkage` library).
  * `similarity.py` – Step 3: Similarity computation script (takes candidate pairs and computes features such as string similarity scores, writing out `features.csv`).
  * `model.py` – Step 4: Model training and scoring script (loads features and labels, trains logistic regression, scores all pairs, outputs `dupes_high_conf.csv` with a probability for each potential duplicate pair).
  * `reporting.py` – Step 5: Reporting script (loads the high-confidence duplicates and the cleaned data, then creates the `merge_suggestions.xlsx` Excel file for review).
  * `openai_integration.py` – Optional GPT integration module (contains functions to use OpenAI API, e.g., `translate_to_english` for company names, and a placeholder for prompting GPT to evaluate duplicates). This is not required for core functionality but is provided for extension.
  * `utils.py` – Utility functions used across the pipeline (for example, file cleanup and logging). Key utilities:

    * `clear_all_data()` – Deletes all files in the outputs directory (to reset the pipeline).
    * `clear_files()` – Remove specific files by path.
    * `log_run()` – Appends an entry to the run history log with step name and timing info.
  * `__init__.py` – Makes the `src` directory a package; not much content relevant to usage.

* **`tests/`** – Unit tests for some of the core functionality:

  * `test_preprocess.py` – Tests the preprocessing logic on small sample data, ensuring duplicates are removed and columns normalized as expected.
  * `test_utils.py` – Tests the utility functions like logging and file clearing.

* **`notebooks/`** – (If present) Jupyter notebooks for interactive exploration or development. For example, `deduplication.ipynb` might be intended to demonstrate the pipeline in an interactive manner. You can open it to walk through the process step by step. *(Note: This might be empty or a work-in-progress in the repository.)*

This modular structure makes it easy to understand and modify each part of the pipeline in isolation. For instance, if you want to improve the similarity computation, you can focus on `src/similarity.py` without touching other parts, as long as you keep the input/output interface consistent (reading `cleaned.csv` and writing `features.csv`).

## Debugging, Troubleshooting, and Tips

Even with a straightforward pipeline, you might run into issues or have questions. Here are some common scenarios, problems, and solutions:

* **Starting Over / Clearing Outputs:** If you need to **reset the pipeline** and run it fresh (maybe after changing your input data or adjusting the code), make sure to clean out old outputs. Leftover files can confuse the scripts or lead to mixing old and new results. Easiest way: use the `--clear` flag on the preprocessing step (or any step) to wipe the `data/outputs/` directory before generating new outputs. You can also manually delete files in `data/outputs` or use the utility functions (`utils.clear_all_data()` in a Python shell) to do this. If you only want to remove specific intermediate files, you can delete them or use `utils.clear_files()` with those paths.

* **Missing Column Errors:** The preprocessing step may raise a `KeyError` if a required column is missing (e.g., it will complain about missing `record_id/sys_id` or `company` column). If you see such an error:

  * Check your input CSV’s header row. Do you have a column that uniquely identifies each record? If named differently (not `record_id` or `sys_id`), you may rename it in the file or adjust the code to accept that name.
  * Ensure there is a company/name column. If your dataset doesn’t have a company (for example, you are deduplicating people and have first name, last name, etc.), you might need to adjust the code to use a different field as the primary entity name. The current code expects something like a company or organization name to exist.
  * If you have the columns but named with spaces or different casing (e.g., "Record ID" or "Company Name"), the code’s normalization should handle it (it lowercases and strips spaces in column names to find matches). But if it still fails, you can manually rename columns in the CSV to match expected names.

* **File Not Found / Path Issues:** If a step complains that it cannot find a file (for example, `Cleaned data not found: data/outputs/cleaned.csv` in the blocking step), it likely means a previous step was not run or its output is not where expected. Double-check that:

  * You have run the steps in sequence. Each script expects the prior stage’s output to be ready.
  * You didn’t change the default paths without telling the next step. If you use a custom `--output-path` in one step, you might need to provide the matching `--input-path` to the next step. Consistency is key if you override any paths.
  * The working directory is correct. Run the commands from the root of the repository (where the `data` folder is accessible). If you run the script from a different folder, the relative paths might not resolve correctly.

* **Output Seems Incorrect or Empty:** If the pipeline runs but the results don't look right (e.g., `dupes_high_conf.csv` is empty or the Excel report has no entries):

  * It could be that no duplicate candidates met the "high confidence" threshold. This might happen if your data truly has no duplicates (good news!) or if the model was not effective. Try lowering the confidence threshold in the code or checking some pairs with slightly lower scores.
  * It’s also possible that the similarity features weren't generated properly. Peek into `features.csv` to see if it contains reasonable values. If all similarity scores are 0 for example, something might have gone wrong in feature computation.
  * Make sure the blocking step didn't eliminate true matches. If two records are duplicates but didn't share a block (e.g., phone was different and company was spelled differently so they ended up in different blocks), they would never be compared. In such a case, you might need to refine the blocking step to be less strict or use a two-pass approach (blocking on multiple keys or more loosely).
  * If using OpenAI for translation, ensure that step actually returned translated names. If the API failed or quota exhausted, you might have empty or partial data which can affect later steps.

* **Performance Issues:** For moderate data sizes (hundreds or thousands of records), this pipeline should run quickly (especially without the OpenAI calls). If you attempt to run it on a very large dataset (millions of records), you could run into performance bottlenecks:

  * The blocking step might still generate a lot of candidate pairs if the blocking keys are too common. You might need to introduce more refined blocking (like blocking on combinations of fields or using multiple passes).
  * The similarity computation is pairwise and could be slow if there are millions of pairs. Consider sampling or more advanced indexing techniques in the `recordlinkage` library (like Sorted Neighborhood or other algorithms).
  * The logistic model training on a huge number of pairs could be slow or memory-intensive. Ensure you have enough memory, or consider doing a preliminary blocking and labeling on a smaller sample to train the model.
  * Writing a very large Excel file in the reporting step could also be slow. For extremely large numbers of duplicate pairs, you might opt to output to a CSV or database for review rather than Excel.

* **Common Mistakes:**

  * Not normalizing data before deduplication. (Our pipeline handles basic normalization, but always sanity-check things like consistent formats for phone numbers, or remove trailing spaces in names that might throw off comparisons.)
  * Relying on a single field to deduplicate can be dangerous, as different entities can share names or domains. The preprocessing step therefore removes rows that duplicate **either** the normalised company name or the domain to catch obvious repeats before modelling.
  * Forgetting to review suggestions manually. Even a high-confidence model can make mistakes. Always review the `merge_suggestions.xlsx` file; do not assume it’s 100% accurate. This pipeline is a tool to assist you, not a fully automatic solution.

* **Limitations:** This is a **minimal demo pipeline**, not a one-size-fits-all solution. It provides a foundation, but you may need to adapt it to your specific data:

  * The current feature set is basic – more complex duplicates (with typos, or missing fields) might require additional logic or external reference data to resolve.
  * The model uses logistic regression; for more complex scenarios, more advanced algorithms or even clustering approaches might be needed. However, logistic regression is interpretable and works well when you have good features.
  * The OpenAI integration is experimental and not integrated into the full pipeline except for translation. Use it as a starting point for AI integration, but it’s not extensively tested.
  * The pipeline assumes the data can fit in memory (since pandas is used for dataframes). If you have extremely large datasets, a scalable approach using databases or distributed processing would be more appropriate.

If you encounter a problem not covered here, consider running the pipeline on the sample data to ensure the code is functioning as expected. You can then compare step by step where your data might be causing issues. Reading the error messages carefully, and referring to the code (which is heavily commented with `TODO` and explanations for each step), can also guide you to the root cause of any issue.

## Developer Guide

For developers or advanced users, this section provides guidance on extending or integrating the deduplication pipeline, as well as information on testing and architecture.

**Running Tests:** The repository includes a suite of unit tests under the `tests/` directory to verify that key components work as expected. To run all tests, use Python's built-in test discovery:

```bash
python -m unittest discover
```

Run this command from the root of the repository. It will automatically find tests (like `tests/test_preprocess.py`, `tests/test_utils.py`, etc.) and execute them. All tests should pass. These tests cover scenarios like basic duplicate removal, handling of missing columns, and the logging/clearing utilities. If you plan to modify the code or add new features, it's a good idea to add corresponding tests and ensure existing tests continue to pass.

**Understanding the Pipeline Architecture:** The pipeline is designed in a modular fashion. Each stage is a separate module/script that can be run independently. They communicate **through the filesystem** (one stage writes a CSV that the next stage reads). This design was chosen for simplicity and transparency:

* It makes debugging easier (you can open intermediate CSV files to inspect what’s happening at each step).
* You can rerun or tweak one stage without re-running the entire pipeline (for example, you might re-run the model training with different parameters without re-running preprocessing, as long as `features.csv` is unchanged).
* It’s easy to swap out components. If a developer wants to try a different blocking technique or a different set of similarity features, they can modify or replace the corresponding module.

**Adding New Features or Improvements:** Developers can extend the pipeline in various ways:

* **Custom Preprocessing:** You might want to add more data cleaning (e.g., standardizing address formats, handling missing values in a smarter way, etc.). You can edit `src/preprocess.py` to add those transformations. Just ensure that the output still contains the needed columns for later steps (record\_id, company\_clean, domain\_clean, phone\_clean, etc.).
* **Better Blocking:** The current blocking is minimal. For larger data, consider multi-step blocking (first on something broad like country or first letter of name, then on finer keys) to manage candidate volume. The `recordlinkage` library supports blocking on multiple keys (even combinations or sorted neighborhood indexing). You can implement those in `src/blocking.py`.
* **Additional Similarity Metrics:** The more features the model has, the better it can distinguish duplicates. You could incorporate things like edit distance for names, Jaccard similarity for sets of words (useful for addresses or lists), or even domain-specific comparisons (if deduplicating products, compare categories; if people, compare birthdates, etc.). Use the `rapidfuzz` library or other text similarity libraries to compute these in `src/similarity.py`. Keep an eye on performance; you might not want to compute every possible metric if not needed.
* **Different Models:** Logistic regression is straightforward, but you could try other algorithms (random forests, gradient boosting, neural nets) for the classification step. These might capture non-linear patterns in the features. However, more complex models may require more data to train effectively and are harder to interpret. If you do swap out the model, update `src/model.py` accordingly and ensure you still output a probability or score for each pair (for reporting).
* **Active Learning/Feedback Loop:** In a practical setting, you might use the results from reporting to further train the model. For example, if the user reviews `merge_suggestions.xlsx` and confirms some pairs as true duplicates and rejects others, those could be fed back into the labels dataset to retrain and improve the model. A developer could script this feedback loop externally or extend the pipeline to incorporate a manual review step where the user’s input is collected.
* **Integration:** You may want to integrate this pipeline into a larger system (for instance, running it as part of an ETL job or a web service). Each module can be imported and called from other Python code. For example, you could `import src.preprocess` and call `src.preprocess.main(input_path="...")` from a larger application. The modular design supports such reuse. If doing so, make sure to handle the working directory or path issues (you might want to pass absolute paths for inputs/outputs in that case).

**Code Style and Conventions:** The code is written with clarity in mind. There are docstrings at the top of each module explaining its purpose. Functions and variables have intuitive names (e.g., `_normalize_name`, `_normalize_domain` for cleaning tasks). If you contribute or modify, try to maintain this clarity. Small, pure functions for normalization make the code easier to test and reuse (notice how `_normalize_name`, `_normalize_domain`, `_normalize_phone` are separate helpers in preprocessing).

**Logging and Monitoring:** The `run_history.log` can be useful for developers to see how long each step takes and how many records are processed each run. This can help identify bottlenecks or anomalies (e.g., if suddenly a run processes far fewer records, maybe many were filtered out unexpectedly). You can extend logging or add more detailed logging within each step if needed.

**Working with the Codebase:** If using an interactive environment like Jupyter notebooks (as hinted by the `notebooks/` directory), you can run parts of the pipeline step by step and inspect variables at each stage, which is great for development and debugging. For instance, you could load `cleaned.csv` into a DataFrame and manually experiment with recordlinkage indexing and comparing before finalizing the code in `blocking.py` or `similarity.py`.

**Contributing:** If this were an open-source project inviting contributions, you'd want to follow typical practice: fork the repo, create a feature branch, write tests for any new functionality, and open a pull request. Ensure that `unittest discover` passes before merging changes. Given that this seems to be a personal or demo project, the contribution process might be informal, but it's always good to test your changes against the provided tests and perhaps add new ones.

**Understanding Test Coverage:** The tests included show expected behaviors:

* After preprocessing a small sample, the number of remaining records and removed records are checked.
* They also confirm that certain columns exist or are all filled with NaN when input columns are missing (e.g., if no domain column in input, the `domain_clean` in output should exist but be empty/NaN for all records).
* Utility tests ensure the logging appends correctly and the clearing functions do not accidentally remove the log if specified to keep it.

Running and understanding these tests can give insight into edge cases the developer of this pipeline had in mind.

---

By following this README, you should be able to get the deduplication pipeline running on your data, understand how each part works, and know how to troubleshoot common issues. Both non-technical users (who just want to clean their spreadsheet of duplicates) and technical users (who may want to extend the pipeline) are encouraged to experiment with the tool. Deduplicating records can significantly improve data quality, and this project provides a transparent, extensible way to achieve that. Good luck with your data cleaning! If you get stuck, refer back to the steps above, and happy deduplicating!
