"""
Model training logic for the diabetes project.

This module defines a function that fits a logistic regression classifier
to the processed diabetes dataset and persists the trained model to disk.
Training and splitting logic lives here to isolate modelling concerns from
other aspects of the pipeline.
"""

import os
import pickle
from typing import Dict, Any

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import logging


def train_model(
    df: pd.DataFrame,
    target_col: str = "class",
    model_path: str = "data/outputs/model.pkl",
) -> Dict[str, Any]:
    """
    Train a logistic regression model on the processed dataset.

    Parameters
    ----------
    df : pandas.DataFrame
        The processed dataset returned by :func:`etl.load.load_data`.
    target_col : str, optional
        Name of the column to use as the target for classification. Defaults
        to ``"class"``.
    model_path : str, optional
        Path where the trained model will be written as a pickle file. Defaults
        to ``"data/outputs/model.pkl"``.

    Returns
    -------
    dict
        A dictionary containing the model, test features and labels, and
        predictions/probabilities for the test set.

    Raises
    ------
    Exception
        If model training fails.
    """
    try:
        X = df.drop(columns=[target_col])
        y = df[target_col]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        # Save model
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        logging.info(f"Model trained and saved to {model_path}")
        return {
            "model": model,
            "X_test": X_test,
            "y_test": y_test,
            "y_pred": y_pred,
            "y_prob": y_prob,
        }
    except Exception as e:
        logging.exception("Error training model.")
        raise e