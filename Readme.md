# Climate ML Experiments Repository

## Comparative Evaluation of Machine Learning Models for Climate Prediction

---

## Overview

This repository contains the experimental work for my MSc dissertation, which focuses on evaluating how different machine learning models perform when predicting key climate variables using ERA5 data.

The aim of this project is not only to identify high-performing models, but to understand how model performance varies across different types of climate data. All experiments are carried out using a consistent dataset and feature engineering pipeline to ensure fair comparison.

---

## Tasks

The experiments cover the following prediction tasks:

* Temperature
* Precipitation
* Soil Moisture

Additional work also includes anomaly detection.

---

## Models

The following machine learning models are evaluated:

* Linear Regression
* Support Vector Regression (SVR)
* Decision Tree
* Random Forest
* XGBoost
* K-Nearest Neighbours (KNN)

---

## Approach

All models are trained under the same setup:

* Time-based train/test split to preserve temporal structure
* Feature engineering including lag variables, rolling averages, and seasonal indicators
* Consistent preprocessing and evaluation across all tasks

This allows direct and fair comparison between models.

---

## Evaluation Metrics

Model performance is evaluated using a combination of regression and task-specific metrics:

Regression Metrics (used across temperature, precipitation, and soil moisture):

RMSE (Root Mean Squared Error)
MAE (Mean Absolute Error)
R² Score

Additional Analysis Metrics:

Residual analysis (distribution and temporal patterns)
Cross-validation scores (mean and standard deviation across folds)
Seasonal evaluation metrics (performance across Winter, Spring, Summer, Autumn)

Anomaly Detection Metrics (where applicable):

Precision
Recall
F1 Score

---

## Repository Structure

data/               Input dataset (ERA5 features)  
code/               Scripts for training, evaluation, and plotting  
results/            Model performance outputs (CSV files)  
visualisations/     Generated figures and plots  
README.md  

---

## Key Findings

* Tree-based models such as Random Forest and XGBoost perform best overall
* Model performance varies depending on the target variable
* Precipitation is more difficult to predict compared to temperature and soil moisture
* Feature engineering has a significant impact on model performance

---

## How to Use

1. Clone the repository
2. Install required libraries (pandas, numpy, scikit-learn, xgboost)
3. Run scripts from the `code/` folder
4. View outputs in the `results/` and `visualisations/` folders

