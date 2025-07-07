"""
File I/O utilities for reading and writing data files.
"""

from __future__ import annotations

import os
import glob
from typing import Optional, Iterable

import pandas as pd

from ..tracking.iteration_tracker import ITERATION_FILE


def clear_files(paths: Iterable[str]) -> None:
    """Remove files if they exist."""
    for path in paths:
        if os.path.exists(path):
            os.remove(path)


def clear_all_data(data_dir: str = "data/outputs", exclude: Iterable[str] | None = None) -> None:
    """Delete all files in ``data_dir`` except those in ``exclude``."""
    exclude = set(exclude or [])
    # Add iteration tracker to excluded files
    exclude.add(os.path.basename(ITERATION_FILE))
    
    for file_path in glob.glob(os.path.join(data_dir, "*")):
        if os.path.basename(file_path) in exclude:
            continue
        if os.path.isfile(file_path):
            os.remove(file_path)


class FileReader:
    """Handles reading various file formats."""
    
    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        """
        Read data from CSV or Excel file.
        
        Returns
        -------
        pd.DataFrame
            The loaded data
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Input file not found: {file_path}")
        
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == ".csv":
            return pd.read_csv(file_path, encoding="utf-8")
        elif file_ext in [".xlsx", ".xls"]:
            return pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")


class FileWriter:
    """Handles writing data to various file formats."""
    
    @staticmethod
    def write_csv(df: pd.DataFrame, output_path: str, create_dirs: bool = True) -> None:
        """
        Write DataFrame to CSV file.
        
        Parameters
        ----------
        df : pd.DataFrame
            Data to write
        output_path : str
            Path to output file
        create_dirs : bool, default True
            Whether to create parent directories if they don't exist
        """
        if create_dirs:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        df.to_csv(output_path, index=True)
    
    @staticmethod
    def write_csv_no_index(df: pd.DataFrame, output_path: str, create_dirs: bool = True) -> None:
        """
        Write DataFrame to CSV file without index.
        
        Parameters
        ----------
        df : pd.DataFrame
            Data to write
        output_path : str
            Path to output file
        create_dirs : bool, default True
            Whether to create parent directories if they don't exist
        """
        if create_dirs:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        df.to_csv(output_path, index=False)
