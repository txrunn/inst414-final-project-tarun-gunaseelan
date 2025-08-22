# Diabetes Prediction Pipeline

This repository contains the source code for a data science pipeline that
predicts the progression of diabetes based on the UCI diabetes dataset. The
project was developed as part of an academic course and demonstrates how to
build and test a simple extract–transform–load (ETL) pipeline, train a
machine‑learning model, evaluate it, and visualize the results.

## Project Overview

The goal of this project is to explore factors associated with diabetes
progression and to build a model that distinguishes between individuals with
higher and lower disease progression. The pipeline uses the diabetes dataset
available from `scikit‑learn`, which originates from the UCI Machine Learning
Repository. This dataset contains 442 observations of 10 baseline variables
(such as age, body mass index and various serum measurements) along with a
numerical target representing a quantitative measure of disease progression one
year after baseline.

In the transformation step the numerical target is converted into a binary
classification variable indicating whether the target value is above the
median. A logistic regression model is then trained to predict this class.
    The pipeline reports evaluation metrics (accuracy, precision, recall,
    F1‑score and ROC AUC) and produces several plots, including a confusion
    matrix, ROC curve and feature‑importance chart. A detailed text
    classification report is also generated.

## Setup Instructions

1. **Clone the repository**

   ```bash
   git clone https://github.com/txrunn/inst414-final-project-tarun-gunaseelan.git
   cd inst414-final-project-tarun-gunaseelan
   ```

2. **Create and activate a virtual environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

## Running the Project

Execute the entire pipeline from the project root:

```bash
python main.py
```

When run, the pipeline will:

1. Extract the diabetes dataset and write it to a raw CSV in a newly created `data/extracted/` directory.
2. Clean the raw data and generate summary statistics. A processed CSV is written to `data/processed/` and an EDA summary is written to `data/outputs/`.
3. Train a logistic regression classifier on the processed data.
4. Evaluate the model and write a CSV of metrics plus plots (confusion matrix, ROC curve and feature‑importance chart) to `data/outputs/`.

The first time you run the pipeline these directories (`data/extracted/`, `data/processed/` and `data/outputs/`) will be created automatically. They are not included in the repository to keep the code package lightweight.  A `pipeline.log` file is produced at the project root to trace progress and any errors.

## Code Package Structure

```
inst414-final-project-tarun-gunaseelan/
├── data/
│   └── reference-tables/   # Data dictionaries and other static reference tables
├── etl/
│   ├── __init__.py
│   ├── extract.py          # Extract the raw data
│   ├── transform.py        # Transform and clean the raw data
│   └── load.py             # Load the processed data
├── analysis/
│   ├── __init__.py
│   ├── model.py            # Train the model
│   └── evaluate.py         # Evaluate the trained model
├── vis/
│   ├── __init__.py
│   └── visualizations.py    # Create additional plots
├── main.py                 # Orchestrate the entire pipeline
├── requirements.txt        # Project dependencies
├── .gitignore              # Excludes runtime data and logs from version control
└── README.md               # This file
```

## Data

The project uses the diabetes dataset included in `scikit‑learn`. A data
dictionary (`data_dictionary_diabetes.csv`) is included in
`data/reference-tables/` to describe each variable in the dataset. No
external APIs or additional datasets are required.

## License

This project is for educational purposes.