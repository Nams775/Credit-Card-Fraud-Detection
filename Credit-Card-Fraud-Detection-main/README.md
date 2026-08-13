# Credit Card Fraud Detection Using Machine Learning

## Project Overview

This project develops a **Machine Learning-based Credit Card Fraud Detection System** that classifies credit card transactions as genuine or fraudulent. The project focuses on the main challenge of fraud detection: fraudulent transactions are extremely rare compared with genuine transactions.

Two classification algorithms are implemented and compared:

- **Logistic Regression**
- **Random Forest Classifier**

The models are evaluated using metrics that are more meaningful for imbalanced fraud data, including Precision, Recall, F1 Score, ROC-AUC, and PR-AUC.

## Problem Statement

Credit card fraud can cause significant financial losses for customers and financial institutions. Because fraudulent transactions form only a small fraction of all transactions, a model that simply predicts every transaction as genuine can still achieve high accuracy while failing to detect fraud. Therefore, the system must identify fraudulent transactions while keeping false alarms as low as possible.

## Objectives

- Detect fraudulent credit card transactions using Machine Learning.
- Handle the highly imbalanced nature of the dataset using class weighting.
- Preprocess transaction features before model training.
- Compare Logistic Regression and Random Forest performance.
- Evaluate the models using fraud-focused classification metrics.
- Identify the most influential transaction features using Random Forest feature importance.

## Dataset

The project uses the commonly used **Credit Card Fraud Detection dataset**, which contains **284,807 transactions**, including **492 fraudulent transactions**.

The target column is `Class`:

| Class | Meaning |
|---|---|
| `0` | Genuine transaction |
| `1` | Fraudulent transaction |

The dataset is highly imbalanced, making Precision, Recall, F1 Score, and PR-AUC especially important when evaluating the model.

> **Dataset note:** The `creditcard.csv` file is required to run the program but is not included in this repository archive. Place it in the same folder as `fraud_detection.py` before running the project.

## Technologies Used

- **Python** – programming language
- **Pandas** – data loading and manipulation
- **NumPy** – numerical operations
- **Matplotlib** – visualization
- **Seaborn** – statistical visualization
- **Scikit-learn** – preprocessing, model training, and evaluation

## Machine Learning Workflow

```text
Credit Card Transaction Dataset
            ↓
      Data Loading
            ↓
   Data Quality Checking
            ↓
 Fraud/Genuine Distribution Analysis
            ↓
   Amount Feature Scaling
            ↓
       Remove Time
            ↓
   Train-Test Split (80:20)
            ↓
      Class Weighting
            ↓
 ┌───────────────────────┐
 │                       │
 ▼                       ▼
Logistic Regression   Random Forest
 │                       │
 └───────────┬───────────┘
             ▼
      Model Evaluation
             ↓
 Confusion Matrix & Feature Importance
```

## Methodology

1. Load the transaction data from `creditcard.csv`.
2. Display the first few records and dataset shape.
3. Check for missing values.
4. Analyze the distribution of genuine and fraudulent transactions.
5. Standardize the `Amount` feature using `StandardScaler`.
6. Remove the `Time` feature.
7. Separate the input features (`X`) and target (`y`).
8. Split the data into training and testing sets using an 80:20 stratified split.
9. Train Logistic Regression using `class_weight="balanced"`.
10. Train Random Forest using `class_weight="balanced"`.
11. Calculate Accuracy, Precision, Recall, F1 Score, ROC-AUC, and PR-AUC.
12. Generate a Random Forest confusion matrix.
13. Identify the top 10 features using Random Forest feature importance.
14. Display the final classification report.

## Model Results

The following results are recorded from the Random Forest run included with this project:

| Metric | Random Forest |
|---|---:|
| Accuracy | 99.96% |
| Precision | 91.95% |
| Recall | 81.63% |
| F1 Score | 86.48% |
| ROC-AUC | 96.73% |
| PR-AUC | 86.64% |

### Confusion Matrix

The recorded Random Forest confusion matrix is:

| | Predicted Genuine | Predicted Fraud |
|---|---:|---:|
| **Actual Genuine** | 56,857 | 7 |
| **Actual Fraud** | 18 | 80 |

This means the model correctly identified most fraudulent transactions while producing only a small number of false positives.

> **Important:** These figures are reference results from the existing project run. Exact results can vary with the dataset version, library versions, hardware, or changes to the code.

## Important Features

The top 10 features identified by Random Forest in the recorded run were:

1. `V10`
2. `V14`
3. `V4`
4. `V12`
5. `V11`
6. `V17`
7. `V3`
8. `V7`
9. `V16`
10. `V2`

These features have the highest relative importance for the Random Forest model in the recorded experiment. Feature importance should be interpreted as model behavior, not as proof that a feature independently causes fraud.

## Why Accuracy Alone Is Not Enough

Fraud datasets are highly imbalanced. If almost every transaction is genuine, a model could obtain very high accuracy by predicting `0` for nearly every transaction while missing fraudulent transactions.

For this reason:

- **Precision** indicates how many transactions predicted as fraud were actually fraudulent.
- **Recall** indicates how many actual fraudulent transactions were detected.
- **F1 Score** balances Precision and Recall.
- **ROC-AUC** measures the model's ability to rank the two classes across thresholds.
- **PR-AUC** is particularly useful for evaluating performance on imbalanced classification problems.

## Business Use Case

A financial institution can use a fraud detection model as an initial screening layer for incoming transactions. Transactions predicted as suspicious can be sent for additional verification, while normal transactions can continue through the payment process.

In a real production system, the model would need additional controls such as threshold tuning, continuous monitoring, model retraining, data drift detection, explainability, and human review for high-risk transactions.

## Project Structure

```text
Credit-Card-Fraud-Detection-main/
│
├── fraud_detection.py   # Data preprocessing, model training and evaluation
├── requirements.txt      # Required Python libraries
├── .gitignore            # Git ignore rules
└── README.md             # Project documentation
```

## Installation

### 1. Clone or download the project

Place all project files in the same directory.

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add the dataset

Download the Credit Card Fraud Detection dataset and place the file named `creditcard.csv` in the project root directory:

```text
Credit-Card-Fraud-Detection-main/
├── creditcard.csv
├── fraud_detection.py
├── requirements.txt
├── .gitignore
└── README.md
```

### 4. Run the program

```bash
python fraud_detection.py
```

The program prints model metrics and displays visualizations for class distribution, the Random Forest confusion matrix, and the top 10 important features.

## Limitations

- The project uses a historical dataset and is not a complete production fraud prevention system.
- Fraud patterns can change over time, so a deployed model requires regular monitoring and retraining.
- The current implementation uses a fixed prediction threshold and does not perform threshold optimization.
- Model performance depends on the dataset and preprocessing choices.

## Conclusion

This project demonstrates how Machine Learning can be applied to credit card fraud detection. Random Forest achieved strong performance in the recorded experiment, with **99.96% accuracy, 91.95% precision, 81.63% recall, and an 86.48% F1 Score**. More importantly, the project evaluates the model using fraud-sensitive metrics rather than relying on accuracy alone, providing a more appropriate approach for an imbalanced classification problem.
