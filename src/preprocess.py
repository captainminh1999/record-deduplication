"""Preprocessing step for cleaning raw records prior to deduplication."""

import pandas as pd


def main(input_path: str = "data/your_spreadsheet.csv", output_path: str = "data/cleaned.csv") -> None:
    """Load raw data and write a cleaned CSV for downstream steps."""
    df = pd.read_csv(input_path)  # or use pd.read_excel for Excel files

    # normalize names
    df["name_clean"] = df["Name"].str.lower().str.replace(r"[^\w\s]", "", regex=True)

    # normalize phones
    df["phone_clean"] = df["Phone"].str.replace(r"\D", "", regex=True)

    # drop exact duplicates
    df = df.drop_duplicates()

    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
