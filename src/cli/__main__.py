"""
Make the CLI module executable as a package.

This allows running the CLI with:
  python -m src.cli preprocess data/sample_input.csv
"""

from .main import main

if __name__ == "__main__":
    exit(main())
