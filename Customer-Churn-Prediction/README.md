# 📊 Customer Churn Prediction using Neural Network

A Machine Learning project that uses a **Feedforward Neural Network (FNN)** to predict whether a customer is likely to **stay with or leave a service** based on customer behavior and service-related features.

---

## 📌 Project Overview

Customer churn prediction helps businesses identify customers who may stop using their services.

In this project, a **Feedforward Neural Network (FNN)** is trained to classify customers into two categories:

- `0` → Customer will stay
- `1` → Customer will leave

The model uses customer information such as age, monthly charges, tenure, complaints, and support calls to make the prediction.

---

## 🎯 Objectives

- Create a neural network model for customer churn prediction.
- Preprocess and scale the dataset.
- Train an FNN using `MLPClassifier`.
- Evaluate the model using accuracy and classification metrics.
- Predict churn for a new customer.

---

## 🧾 Features

| Feature | Description |
|---|---|
| Age | Age of the customer |
| Monthly Charges | Monthly service charges |
| Tenure | Number of months the customer has stayed |
| Complaints | Number of complaints made by the customer |
| Support Calls | Number of customer support calls |

### Target Variable

| Value | Meaning |
|---|---|
| `0` | Customer will stay |
| `1` | Customer will leave |

---

## 🗂️ Dataset

The project uses a small manually created dataset for demonstration and academic purposes.

```python
X = [
    [25, 500, 12, 1, 2],
    [30, 700, 24, 0, 1],
    [45, 1200, 6, 5, 8],
    [50, 1500, 5, 6, 10],
    [28, 600, 18, 1, 1],
    [35, 800, 30, 0, 0],
    [48, 1400, 4, 7, 9],
    [52, 1600, 3, 8, 12],
    [27, 550, 20, 0, 1],
    [42, 1300, 8, 4, 7]
]
```

Target values:

```python
y = [
    0, 0, 1, 1, 0,
    0, 1, 1, 0, 1
]
```

---

## 🧠 Model Architecture

The project uses `MLPClassifier` from Scikit-learn as the Feedforward Neural Network.

```text
Input Layer
     │
     ├── Age
     ├── Monthly Charges
     ├── Tenure
     ├── Complaints
     └── Support Calls
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
   Binary Output
     │
     ▼
Stay / Leave
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
- 📊 Machine Learning
- 🤖 Feedforward Neural Network
- 📏 StandardScaler
- 📈 Classification Metrics

---

## 🔄 Machine Learning Workflow

```text
Dataset
   ↓
Data Preparation
   ↓
Train-Test Split
   ↓
StandardScaler
   ↓
FNN / MLP Model
   ↓
Model Training
   ↓
Prediction
   ↓
Accuracy Evaluation
```

---

## 🛠️ Steps Performed

### 1. Load/Create Dataset

Customer information is stored in NumPy arrays.

### 2. Split Dataset

The dataset is divided into training and testing sets using:

```python
train_test_split()
```

### 3. Feature Scaling

Since the features have very different numerical ranges, `StandardScaler` is applied.

```python
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

### 4. Train FNN Model

The neural network is trained using:

```python
model.fit(X_train, y_train)
```

### 5. Evaluate Model

Predictions are compared with actual values.

```python
accuracy_score(y_test, y_pred)
```

### 6. Predict New Customer

Example customer:

```python
new_customer = [
    [46, 1450, 5, 6, 9]
]
```

The model predicts:

```text
Prediction: Customer may leave
```

---

## 📊 Evaluation

The model is evaluated using:

- Accuracy
- Classification Report
- Precision
- Recall
- F1-Score

Example:

```python
print("Model Accuracy:", accuracy)
print(classification_report(y_test, y_pred))
```

> **Note:** The dataset contains only 10 records and is intended for educational demonstration. Therefore, the evaluation accuracy should not be interpreted as real-world model performance.

---

## 📁 Project Structure

```text
Customer-Churn-Prediction/
│
├── customer_churn_prediction.py
├── README.md
└── requirements.txt
```

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/customer-churn-prediction.git
```

Navigate into the project:

```bash
cd customer-churn-prediction
```

Install dependencies:

```bash
pip install numpy scikit-learn
```

Or use:

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

Run the Python program:

```bash
python customer_churn_prediction.py
```

Example output:

```text
Model Accuracy: ...

Classification Report:
...

Prediction: Customer may leave
```

---

## 💡 Example Prediction

### New Customer

```text
Age              : 46
Monthly Charges  : 1450
Tenure           : 5 months
Complaints       : 6
Support Calls    : 9
```

### Prediction

```text
Customer may leave
```

This prediction demonstrates how the neural network can identify a potentially high-churn customer based on the given features.

---

## 🚀 Future Improvements

- Use a larger real-world customer churn dataset.
- Add more customer-related features.
- Handle missing values and categorical variables.
- Perform hyperparameter tuning.
- Compare FNN with Logistic Regression, Random Forest and XGBoost.
- Add confusion matrix visualization.
- Build a Streamlit web application.
- Deploy the model as an API using FastAPI.

---

## 🎓 Learning Outcomes

Through this project, you can learn:

- Basics of Artificial Neural Networks
- Feedforward Neural Networks
- Binary Classification
- Data preprocessing
- Feature scaling
- Train/Test splitting
- `MLPClassifier`
- Model evaluation
- Customer churn prediction

---

## 👨‍💻 Author

**Ashutosh Kadu**

Engineering Student | Machine Learning | Python | Full Stack Development

---

## ⭐ Project

If you found this project useful for learning **Machine Learning and Neural Networks**, consider giving the repository a ⭐ on GitHub.
