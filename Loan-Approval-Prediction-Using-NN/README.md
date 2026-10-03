# 💰 Loan Approval Prediction using Neural Network

A Machine Learning project that uses a **Feedforward Neural Network (FNN)** to predict whether a loan application should be **approved or rejected** based on an applicant's financial and employment information.

---

## 📌 Project Overview

Loan approval prediction is a binary classification problem where a machine learning model analyzes an applicant's financial profile and predicts the outcome of a loan application.

In this project, a **Feedforward Neural Network (FNN)** is trained using Scikit-learn's `MLPClassifier`.

The model predicts:

- `0` → Loan Rejected
- `1` → Loan Approved

---

## 🎯 Objectives

- Create a neural network for loan approval prediction.
- Preprocess the dataset.
- Handle the employment-status feature.
- Apply feature scaling using `StandardScaler`.
- Train an FNN classification model.
- Evaluate the model.
- Predict loan approval for a new applicant.

---

## 🧾 Features

| Feature | Description |
|---|---|
| Applicant Income | Monthly/annual income of the applicant |
| Credit Score | Applicant's credit score |
| Loan Amount | Amount of loan requested |
| Existing EMI | Existing monthly EMI obligation |
| Employment Status | Stability of applicant's employment |

### Employment Status

| Value | Meaning |
|---|---|
| `0` | Not Stable |
| `1` | Stable |

### Target Variable

| Value | Meaning |
|---|---|
| `0` | Loan Rejected |
| `1` | Loan Approved |

---

## 🗂️ Dataset

The project uses a small manually created dataset for educational purposes.

```python
X = [
    [25000, 600, 200000, 10000, 0],
    [40000, 700, 300000, 8000, 1],
    [60000, 750, 500000, 12000, 1],
    [20000, 550, 150000, 15000, 0],
    [80000, 800, 700000, 10000, 1],
    [35000, 650, 250000, 9000, 1],
    [18000, 500, 100000, 12000, 0],
    [90000, 850, 800000, 15000, 1],
    [30000, 580, 200000, 14000, 0],
    [70000, 780, 600000, 10000, 1]
]
```

Target values:

```python
y = [
    0, 1, 1, 0, 1,
    1, 0, 1, 0, 1
]
```

---

## 🧠 Model Architecture

The project uses `MLPClassifier` as the Feedforward Neural Network.

```text
Input Layer
     │
     ├── Applicant Income
     ├── Credit Score
     ├── Loan Amount
     ├── Existing EMI
     └── Employment Status
     │
     ▼
Hidden Layer 1
   8 Neurons
     │
     ▼
Hidden Layer 2
   4 Neurons
     │
     ▼
Output Layer
 Binary Classification
     │
     ▼
Approved / Rejected
```

### Model Configuration

```python
MLPClassifier(
    hidden_layer_sizes=(8, 4),
    activation='relu',
    solver='lbfgs',
    max_iter=5000,
    random_state=42
)
```

---

## ⚙️ Technologies Used

- 🐍 Python
- 🧠 Scikit-learn
- 🔢 NumPy
- 🤖 Feedforward Neural Network
- 📏 StandardScaler
- 📊 Machine Learning
- 📈 Classification Metrics

---

## 🔄 Machine Learning Workflow

```text
Dataset
   ↓
Data Preprocessing
   ↓
Train-Test Split
   ↓
Feature Scaling
   ↓
FNN / MLP Model
   ↓
Model Training
   ↓
Model Evaluation
   ↓
New Applicant
   ↓
Loan Approval Prediction
```

---

## 🛠️ Steps Performed

### 1. Create Dataset

The dataset contains applicant financial information and employment status.

### 2. Split Dataset

The data is divided into training and testing sets using:

```python
train_test_split()
```

### 3. Apply Feature Scaling

Financial features have very different numerical ranges. Therefore, `StandardScaler` is used.

```python
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

### 4. Train FNN

The neural network is trained using:

```python
model.fit(X_train, y_train)
```

### 5. Evaluate Model

The model generates predictions for the test dataset.

```python
y_pred = model.predict(X_test)
```

Accuracy is calculated using:

```python
accuracy_score(y_test, y_pred)
```

### 6. Predict New Applicant

The test applicant is:

```python
new_applicant = [
    [55000, 720, 400000, 10000, 1]
]
```

The model predicts:

```text
Prediction: Loan Approved
```

---

## 📊 Model Evaluation

The project evaluates the model using:

- Accuracy
- Precision
- Recall
- F1-Score
- Classification Report

Example:

```python
print("Model Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```

> **Note:** The dataset contains only 10 records and is designed for educational/practical demonstration. Its evaluation metrics should not be considered representative of real-world loan approval performance.

---

## 📁 Project Structure

```text
Loan-Approval-Prediction/
│
├── loan_approval_prediction.py
├── README.md
└── requirements.txt
```

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/loan-approval-prediction.git
```

Navigate to the project:

```bash
cd loan-approval-prediction
```

Install the required libraries:

```bash
pip install numpy scikit-learn
```

Or:

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

Run the Python program:

```bash
python loan_approval_prediction.py
```

Example output:

```text
Model Accuracy: ...

Classification Report:
...

Prediction: Loan Approved
```

---

## 🧪 Test Applicant

The model is tested with:

```text
Applicant Income : 55,000
Credit Score     : 720
Loan Amount      : 400,000
Existing EMI     : 10,000
Employment       : Stable
```

### Expected Prediction

```text
Prediction: Loan Approved
```

---

## 🚀 Future Improvements

- Use a larger real-world loan dataset.
- Add more applicant features.
- Handle missing values.
- Add categorical encoding for employment types.
- Perform hyperparameter tuning.
- Compare FNN with Logistic Regression, Random Forest and XGBoost.
- Add confusion matrix visualization.
- Add ROC-AUC evaluation.
- Build an interactive Streamlit dashboard.
- Deploy the model using FastAPI.

---

## 🎓 Learning Outcomes

Through this project, you can learn:

- Feedforward Neural Networks
- Binary Classification
- Loan approval prediction
- Data preprocessing
- Feature scaling
- Neural network training
- `MLPClassifier`
- Model evaluation
- Classification metrics
- Making predictions for new data

---

## ⚠️ Disclaimer

This project is created **for educational and academic purposes only**. It should not be used to make real-world lending or financial decisions. Real loan approval systems require substantially larger datasets, rigorous validation, appropriate risk controls, and consideration of legal and fairness requirements.

---

## 👨‍💻 Author

**Ashutosh Kadu**

Engineering Student | Python | Machine Learning | Neural Network

---

## ⭐ Project

If you found this project useful for learning **Machine Learning and Neural Networks**, consider giving the repository a ⭐ on GitHub.
