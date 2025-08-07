# INST414 Final Project

## Project Overview
This project predicts 30-day hospital readmissions for diabetic patients using historical inpatient data.

**Business Problem**: Reduce unplanned 30-day readmissions in diabetic patients.

**Data**: UCI Diabetes 130-US Hospitals Dataset, CMS Readmissions Benchmarking Data

**Techniques**: Classification models (logistic regression, random forest), data wrangling, EDA, predictive modeling

## Setup Instructions
```bash
git clone https://github.com/txrunn/inst414-final-project-tarun-gunaseelan.git
cd inst414-final-project-tarun-gunaseelan
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Running the Project
```bash
python main.py
```

## Code Package Structure
- `data/`: Raw, processed, and output data
- `etl/`: Scripts for extract, transform, load
- `analysis/`: Scripts for modeling and evaluation
- `vis/`: Scripts for visualizations
