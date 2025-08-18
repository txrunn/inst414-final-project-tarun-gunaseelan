"""
Data transformation logic for the diabetes project.

This module defines the `transform_data` function used to clean
and prepare the raw dataset for analysis. The key transformation
implemented here converts the numeric disease progression target into a
binary class indicator (above or below the median). An exploratory
summary is also written to a CSV for quick inspection.
"""

import os
import pandas as pd
import logging


def transform_data(
    input_path: str = "data/extracted/diabetes_raw.csv",
    output_path: str = "data/processed/diabetes_processed.csv",
) -> str:
    """
    Transform the raw diabetes dataset into a cleaned and analysis‑ready format.

    The transformation currently creates a binary classification target
    named ``class`` based on whether the original ``target`` value is above
    its median. It also writes a summary statistics table to
    ``data/outputs/eda_summary.csv``.

    Parameters
    ----------
    input_path : str, optional
        Path to the raw CSV produced by :func:`extract_data`. Defaults to
        ``"data/extracted/diabetes_raw.csv"``.
    output_path : str, optional
        Path where the cleaned CSV will be saved. Defaults to
        ``"data/processed/diabetes_processed.csv"``.

    Returns
    -------
    str
        The path to the saved cleaned CSV.

    Raises
    ------
    Exception
        If the transformation fails.
    """
    try:
        df = pd.read_csv(input_path)
        # Create classification target based on median of target.
        median_value = df["target"].median()
        df["class"] = (df["target"] > median_value).astype(int)
        # Save cleaned data
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logging.info(f"Processed data saved to {output_path}")
        # Save simple EDA summary
        os.makedirs("data/outputs", exist_ok=True)
        summary = df.describe(include="all")
        summary.to_csv("data/outputs/eda_summary.csv")
        logging.info("Saved EDA summary to data/outputs/eda_summary.csv")
        return output_path
    except Exception as e:
        logging.exception("Error during data transformation.")
        raise e