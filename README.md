## Introduction
This repository contains the files for thesis experiments on Early Time Series Classification (ETSC). The case study is performed on the S&P500 index using the VIX, MACRO indicators and 4 technical indicators in a variety of configurations. This classification problem is a binary UP/DOWN (1.0/2.0) task and is performed using the CALIMERA ETSC method. The experiment uses the MiniROCKET feature extractor and the Ridge classifier respectively, as used by Bilski et al. in their paper "CALIMERA: A new early time series classification method" (2023, Information Processing and Management, vol. 60, issue 5, DOI: 10.1016/j.ipm.2023.103465, Link: https://doi.org/10.1016/j.ipm.2023.103465).
(Not a real citation just helpful information for everyone interested) 

This experiment will likely not run in the WSL environment due to high RAM usage. However, reducing the amount of kernels should not reduce accuracy in a major way (it could actually increase it by reducing overfitting) but should reduce RAM use by some amount. The simpler experiments, such as S&P500 only, are also quicker to run due to their lower time complexity.

The data was acquired from Kaggle and the VIX was from the CBOE website.

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Python 3 is required for these experiments. CALIMERA will not work with 

## File Descriptions

- `run_experiment.py` - Main experiment runner with CLI. Handles data loading, model training, testing, and results visualization.
- `calimera.py` - Standard CALIMERA early time series classification implementation.
- `calimera_weighted.py` - Weighted version of CALIMERA with modified cost function (RidgeClassifierCV class_weight ='balanced'. The weighted option exists as there are more UP moves than DOWN moves, and weights 'balanced' (approximately) would close this gap.
- `thesis_data_generator.py` - This Python file generates .ts files, which are used as data for testing the algorithm.
- `batch_experiments.py` - Batch processing utility for experiments.
- `financial_indicators.py` - Script for calculating technical indicators (RSI, MACD, etc.), RUN THIS BEFORE THESIS DATA GENERATOR.
- `usage_example.py` - Simple example showing how to use the CALIMERA models. This is left in the repository from the original CALIMERA repository as a backup in case other options do not work.
- `original_datasets/` - Directory containing original time series datasets in .csv format.
- `thesis_datasets/` - Directory containing preprocessed time series datasets in .ts format. Please download the dataset linked.a 
- `results/` - Directory where experiment results, plots, and metrics are saved. Names include experiment name and timestamp.
- Raw data files: `1_min_SPY_2008-2021.csv`, `VIX_History.csv`, `macro_monthly.csv`, etc.

## Usage

Help to run experiments:
```bash
python run_experiment.py
```

Run programmatically, for example:
```bash
python run_experiment.py TECH_ONLY standard 0.6
```

Disclaimer: Some of the code was originally written using the help of AI tools for the code logic and Python. This was done to prototype the experiment quickly. However, the code has been thoroughly reviewed and partially rewritten by a human.
