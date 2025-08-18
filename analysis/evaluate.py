"""
Model evaluation logic for the diabetes project.

This module computes common classification metrics and generates plots to
assess the performance of a trained classifier. Results are written to
files in the `data/outputs/` directory.
"""

import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def evaluate_model(results: dict, output_dir: str = "data/outputs") -> pd.DataFrame:
    """
    Evaluate the trained model and save metrics and plots.

    Parameters
    ----------
    results : dict
        Dictionary returned by :func:`analysis.model.train_model` containing
        predictions and probabilities.
    output_dir : str, optional
        Directory where evaluation outputs will be saved. Defaults to
        ``"data/outputs"``.

    Returns
    -------
    pandas.DataFrame
        A DataFrame of evaluation metrics.

    Raises
    ------
    Exception
        If evaluation fails.
    """
    try:
        y_test = results["y_test"]
        y_pred = results["y_pred"]
        y_prob = results["y_prob"]
        # Compute metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        metrics_df = pd.DataFrame(
            {
                "metric": ["accuracy", "precision", "recall", "roc_auc"],
                "value": [acc, prec, rec, auc],
            }
        )
        os.makedirs(output_dir, exist_ok=True)
        metrics_df.to_csv(os.path.join(output_dir, "evaluation_metrics.csv"), index=False)
        logging.info("Saved evaluation metrics to evaluation_metrics.csv")
        # Confusion matrix plot
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = plt.figure()
        plt.imshow(cm, interpolation="nearest")
        plt.title("Confusion Matrix")
        plt.colorbar()
        tick_marks = range(len(set(y_test)))
        plt.xticks(tick_marks)
        plt.yticks(tick_marks)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], "d"), horizontalalignment="center")
        fig_cm.savefig(os.path.join(output_dir, "confusion_matrix.png"))
        plt.close(fig_cm)
        # ROC curve plot
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fig_roc = plt.figure()
        plt.plot(fpr, tpr)
        plt.title("ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        fig_roc.savefig(os.path.join(output_dir, "roc_curve.png"))
        plt.close(fig_roc)
        logging.info("Saved confusion matrix and ROC curve plots.")
        return metrics_df
    except Exception as e:
        logging.exception("Error evaluating model.")
        raise e