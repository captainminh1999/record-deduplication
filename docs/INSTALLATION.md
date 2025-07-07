# Installation and Setup

**Prerequisites:** Ensure you have **Python 3** installed (the code uses modern Python features) and `pip` for package management. We recommend setting up a virtual environment for this project.

## Step 1: Clone the Repository

If you haven't already, download or clone this repository to your local machine.

```bash
git clone https://github.com/captainminh1999/record-deduplication.git
cd record-deduplication
```

## Step 2: Install Dependencies

Use pip to install the required libraries from the provided `requirements.txt` file:

```bash
python -m pip install -r requirements.txt
```

This will install pandas, NumPy, scikit-learn, `recordlinkage` (for pairing records), `rapidfuzz` (for string similarity), and Excel output libraries like OpenPyXL. These are the core dependencies needed for the pipeline to run.

## Step 3: (Optional) Enable GPT Integration

If you plan to use the optional OpenAI GPT-powered features, you need to install the OpenAI package and have an API key. The `openai` library is commented out in requirements (not installed by default). To include it, run:

```bash
pip install openai
```

Also, set your OpenAI API key as an environment variable (`OPENAI_KEY`) or configure it in your code before use. *Skip this step if you don't intend to use the GPT features.* (More on this in the [GPT Integration guide](GPT_INTEGRATION.md).)

## Virtual Environment Setup (Recommended)

For better dependency management, consider using a virtual environment:

```bash
# Create virtual environment
python -m venv dedup-env

# Activate it (Windows)
dedup-env\Scripts\activate

# Activate it (macOS/Linux)
source dedup-env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Verification

To verify your installation works, you can run the tests:

```bash
python -m unittest discover
```

All tests should pass if the installation was successful.
