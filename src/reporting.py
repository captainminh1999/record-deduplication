"""Generate a spreadsheet for human review of potential duplicates."""

import pandas as pd


def main(dupes_path: str = "data/dupes_high_conf.csv", cleaned_path: str = "data/cleaned.csv"):
    """Create an Excel file with side-by-side records for review."""
    dupes = pd.read_csv(dupes_path, index_col=[0, 1])
    df = pd.read_csv(cleaned_path)

    report = dupes.merge(
        df.reset_index(),
        left_index=True,
        right_on="ID",
        suffixes=("_A", "_B"),
    )

    report.to_excel("merge_suggestions.xlsx", index=False)


if __name__ == "__main__":
    main()
