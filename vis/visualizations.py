"""
Additional visualizations for the diabetes project.

While the evaluation module produces standard performance plots, this module
provides functions for other visual analyses, such as feature importance
charts. Having separate visualization functions keeps the evaluation code
focused on metrics and allows further plots to be added easily.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import logging


def create_visualizations(results: dict, output_dir: str = "data/outputs") -> None:
    """
    Create additional visualizations such as feature importance.

    Parameters
    ----------
    results : dict
        Dictionary returned by :func:`analysis.model.train_model` containing
        the trained model and test feature matrix.
    output_dir : str, optional
        Directory where plots will be written. Defaults to ``"data/outputs"``.

    Raises
    ------
    Exception
        If visualization fails.
    """
    try:
        model = results["model"]
        X_test = results["X_test"]
        # Compute absolute value of coefficients for feature importance
        importance = np.abs(model.coef_[0])
        feature_names = X_test.columns
        os.makedirs(output_dir, exist_ok=True)
        fig_imp = plt.figure()
        plt.barh(feature_names, importance)
        plt.title("Feature Importance (Absolute Coefficients)")
        plt.xlabel("Importance")
        fig_imp.savefig(os.path.join(output_dir, "feature_importance.png"))
        plt.close(fig_imp)
        logging.info("Saved feature importance plot.")
    except Exception as e:
        logging.exception("Error creating visualizations.")
        raise e