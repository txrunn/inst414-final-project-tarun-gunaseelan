"""
Data loading logic for the diabetes project.

This module defines a helper function to load the processed dataset
for downstream analysis and modelling. Separating loading logic into
its own function helps to decouple file I/O from business logic.
"""

import pandas as pd
import logging


def load_data(path: str = "data/processed/diabetes_processed.csv") -> pd.DataFrame:
    """
    Load the processed dataset for analysis.

    Parameters
    ----------
    path : str, optional
        Path to the processed CSV. Defaults to
        ``"data/processed/diabetes_processed.csv"``.

    Returns
    -------
    pandas.DataFrame
        The processed dataset.

    Raises
    ------
    Exception
        If the file cannot be loaded.
    """
    try:
        df = pd.read_csv(path)
        logging.info(f"Loaded processed data from {path}")
        return df
    except Exception as e:
        logging.exception("Error loading processed data.")
        raise e