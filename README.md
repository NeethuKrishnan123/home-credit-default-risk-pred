# Home Credit Default Risk Prediction

## Project Overview

This project predicts whether a customer will default on a loan using the Home Credit Default Risk dataset.

The project follows a complete Machine Learning and MLOps workflow including:

* Data Preprocessing
* Feature Engineering
* Feature Selection
* Model Training
* Hyperparameter Tuning
* MLflow Experiment Tracking
* Pipeline Automation
* Git Version Control
* Model Deployment

---


## Dataset

**Dataset:** Home Credit Default Risk Dataset

**Files Used:**

* application_train.csv
* bureau.csv

**Source:** Kaggle Home Credit Default Risk Dataset

---

## Project Structure

```text
home-credit-default-risk-pred/

├── data/
│   ├── application_train.csv
│   ├── bureau.csv
│   └── preprocessed_data.csv
│
├── notebooks/
│
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── feature_selection.py
│   ├── train_model.py
│   ├── main.py
│   └── api.py
│
├── models/
│   └── model.pkl
│
├── mlruns/
│
├── README.md
├── .gitignore
└── requirements.txt
```

---

## Workflow

### 1. Data Preprocessing

* Load `application_train.csv`
* Load `bureau.csv`
* Bureau Aggregation
* Merge Datasets
* Missing Value Handling
* Target Encoding
* Drop Low Importance Features

### 2. Feature Engineering

Created Features:

* AGE
* YEARS_EMPLOYED
* EMPLOYED_BIRTH_RATIO
* GOODS_INCOME_RATIO
* EXT_SOURCE_RANGE
* CREDIT_INCOME_RATIO
* ANNUITY_INCOME_RATIO
* PAYMENT_RATE
* GOODS_CREDIT_RATIO
* EXT_MEAN
* EXT_STD
* EXT_MIN
* EXT_MAX
* EXT_SUM
* BUREAU_DEBT_RATIO
* BUREAU_OVERDUE_RATIO

### 3. Feature Selection

* Remove Weak Features
* Remove Highly Correlated Features

### 4. Model Training

**Algorithm Used**

* LightGBM Classifier

**Cross Validation**

* 5-Fold Cross Validation

**Evaluation Metrics**

* ROC-AUC Score
* Precision
* Recall
* F1 Score

### 5. MLflow Tracking

MLflow is used for:

* Experiment Tracking
* Metric Logging
* Model Logging

Tracked Metrics:

* ROC-AUC
* Precision
* Recall
* F1 Score

---

## Running the Pipeline

### Activate Virtual Environment

```bash
venv\Scripts\activate
```

### Run Pipeline

```bash
python src/main.py
```

### Output

* Preprocessed Data
* Trained Model
* MLflow Logs
* Saved Model File

---

## Model Deployment

**Framework:** FastAPI


---

## Technologies Used

* Python
* Pandas
* NumPy
* LightGBM
* MLflow
* FastAPI
* Joblib
* Scikit-Learn
* Git
* GitHub




