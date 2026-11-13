
# Credit Card Fraud Detection System

An end-to-end Machine Learning project that detects potentially fraudulent credit card transactions using Random Forest and an interactive Streamlit dashboard.

## Project Overview

This project uses Machine Learning classification techniques to identify fraudulent and genuine credit card transactions. It includes data preprocessing, exploratory data analysis, SMOTE-based class balancing, model training, evaluation, and deployment.

## Features

- Credit card fraud detection using Random Forest
- Exploratory Data Analysis (EDA)
- Data preprocessing and feature scaling
- SMOTE for handling imbalanced datasets
- Single transaction prediction
- Bulk CSV transaction screening
- Fraud probability and risk-level display
- Interactive Streamlit dashboard
- Separate PhonePe transaction analysis module

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Matplotlib and Seaborn
- Plotly
- Streamlit
- Joblib

## Model Performance

| Metric | Score |
|---|---:|
| Accuracy | 99.95% |
| Precision | 91.03% |
| Recall | 74.74% |
| F1-Score | 82.08% |
| PR-AUC | 81.84% |

Model: Random Forest Classifier

Note: These metrics are from the held-out test dataset.

## Project Structure

```text
Credit-Card-Fraud-Detection/
├── data/
├── models/
│   └── credit_fraud_model.pkl
├── notebook/
│   └── 01_EDA.ipynb
├── src/
├── app.py
├── requirements.txt
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/Rishi321145/Credit-Card-Fraud-Detection.git
```

Navigate to the project:

```bash
cd Credit-Card-Fraud-Detection
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## PhonePe Test Mode

The dashboard also includes a separate PhonePe transaction analysis module for transaction summaries and unusual amount review.

This module is exploratory and does not use the credit card fraud classifier to predict UPI fraud.

## Disclaimer

This project is developed for educational and research purposes. Predictions are not guaranteed to identify every fraudulent transaction.

## Author

Rishi321145


