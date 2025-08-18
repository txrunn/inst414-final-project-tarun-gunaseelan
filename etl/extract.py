"""
Data extraction logic for the diabetes project.

This module contains a single function, `extract_data`, which
downloads the diabetes dataset from scikit‑learn and writes it to disk as
a raw CSV file. The extract stage is intentionally simple because the
dataset is packaged with scikit‑learn, but the function structure
mirrors what you might build for an API call or web scraper in a
production pipeline.
"""

from typing import Optional
import os
import pandas as pd
from sklearn.datasets import load_diabetes
import logging


def extract_data(output_path: str = "data/extracted/diabetes_raw.csv") -> str:
    """
    Extract the diabetes dataset and save as a raw CSV file.

    The diabetes dataset originates from the UCI Machine Learning Repository
    and is included with scikit‑learn. This function loads the data
    into a pandas DataFrame and writes it to the specified location.

    Parameters
    ----------
    output_path : str, optional
        Relative path where the raw CSV will be saved. Defaults to
        ``"data/extracted/diabetes_raw.csv"``.

    Returns
    -------
    str
        The path to the saved raw CSV file.

    Raises
    ------
    Exception
        If the dataset cannot be loaded or saved.
    """
    try:
        diabetes = load_diabetes(as_frame=True)
        df = diabetes.frame
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logging.info(f"Extracted raw data to {output_path}")
        return output_path
    except Exception as e:
        logging.exception("Error during data extraction.")
        raise e