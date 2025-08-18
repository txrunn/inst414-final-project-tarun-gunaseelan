"""
Entry point for the diabetes prediction pipeline.

This script orchestrates the extract, transform, load (ETL) stages,
trains a machine‑learning model, evaluates it, and generates visualizations.
Logging is configured to write minimal messages to a file and to the console.

Run this file from the root of the project to execute the entire workflow.
"""

import logging

from etl.extract import extract_data
from etl.transform import transform_data
from etl.load import load_data
from analysis.model import train_model
from analysis.evaluate import evaluate_model
from vis.visualizations import create_visualizations


def main() -> None:
    """
    Orchestrate the data pipeline.

    Each stage of the pipeline is wrapped in try/except blocks so that failures
    in one stage do not necessarily halt execution of subsequent stages.
    Minimal logging messages are written to both a log file (`pipeline.log`)
    and the console so you can trace the progress of the pipeline.
    """
    # Configure logging to file and console.
    logging.basicConfig(
        filename="pipeline.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)

    # Execute pipeline stages.
    try:
        raw_path = extract_data()
    except Exception as exc:
        logging.error("Data extraction failed.", exc_info=exc)
        raw_path = None

    try:
        processed_path = transform_data()
    except Exception as exc:
        logging.error("Data transformation failed.", exc_info=exc)
        processed_path = None

    try:
        df = load_data()
    except Exception as exc:
        logging.error("Data loading failed.", exc_info=exc)
        df = None

    # Only proceed with modelling if data has been loaded.
    if df is not None:
        try:
            results = train_model(df)
        except Exception as exc:
            logging.error("Model training failed.", exc_info=exc)
            results = None

        if results is not None:
            try:
                evaluate_model(results)
            except Exception as exc:
                logging.error("Model evaluation failed.", exc_info=exc)

            try:
                create_visualizations(results)
            except Exception as exc:
                logging.error("Visualization creation failed.", exc_info=exc)


if __name__ == "__main__":
    main()