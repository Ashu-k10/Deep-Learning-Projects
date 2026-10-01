# Employee Attrition Prediction using MLPClassifier

A Deep Learning project that predicts **employee attrition** using a **Multi-Layer Perceptron (MLP)** neural network implemented with `scikit-learn`.

The project covers data preprocessing, categorical encoding, feature scaling, train/test splitting, MLP model training, accuracy evaluation, confusion matrix, classification report, loss-curve visualization, employee-level prediction, and overfitting/underfitting analysis.

---

## 📌 Project Overview

Employee attrition refers to an employee leaving an organization.

This project uses employee-related features such as:

- Age
- Monthly Income
- Years at Company
- Total Working Years
- Distance From Home
- Job Satisfaction
- Work-Life Balance
- Overtime
- Number of Companies Worked
- Training Times Last Year

The target variable is:

```text
Attrition
```

It is converted into:

| Value | Meaning |
|---|---|
| `0` | Likely to Stay |
| `1` | Likely to Leave |

---

## 🎯 Objectives

The main objectives of this assignment are:

1. Load the employee attrition dataset.
2. Explore the dataset.
3. Check for missing values.
4. Identify numerical and categorical features.
5. Convert categorical values into numerical values.
6. Separate independent and dependent variables.
7. Split the dataset into training and testing sets.
8. Apply feature scaling.
9. Build an MLP neural network with at least two hidden layers.
10. Train the neural network.
11. Calculate training and testing accuracy.
12. Generate a confusion matrix.
13. Generate a classification report.
14. Plot the MLP training loss curve.
15. Create a reusable `PredictAttrition()` function.
16. Predict attrition for five new employees.
17. Analyze the model for overfitting or underfitting.

---

## 🛠️ Technologies Used

- **Python**
- **Pandas**
- **Matplotlib**
- **Scikit-learn**
- **MLPClassifier**
- **StandardScaler**

---

## 📂 Project Structure

```text
Employee-Attrition-Prediction/
│
├── Assignment_62.py
├── Fraudulent_Attribution.csv
└── README.md
```

> **Note:** The Python code currently loads the dataset using:
>
> ```python
> df = pd.read_csv("Fraudulent_Attribution.csv")
> ```
>
> Therefore, the CSV filename in the project folder must match this name exactly.

---

## 📊 Dataset Features

### Numerical Features

The following numerical features are used:

```python
numerical_features = [
    "Age",
    "MonthlyIncome",
    "YearsAtCompany",
    "TotalWorkingYears",
    "DistanceFromHome",
    "JobSatisfaction",
    "WorkLifeBalance",
    "NumCompaniesWorked",
    "TrainingTimesLastYear"
]
```

### Categorical Feature

```python
categorical_features = [
    "OverTime"
]
```

---

## 🔄 Data Preprocessing

### 1. Overtime Encoding

The `OverTime` feature is converted into numerical values:

```text
Yes → 1
No  → 0
```

Using:

```python
df["OverTime"] = df["OverTime"].map({
    "Yes": 1,
    "No": 0
})
```

### 2. Attrition Encoding

The target variable is converted as:

```text
Yes → 1
No  → 0
```

Therefore:

```text
0 → Likely to Stay
1 → Likely to Leave
```

---

## 📐 Feature Selection

The independent variables `X` are:

```text
Age
MonthlyIncome
YearsAtCompany
TotalWorkingYears
DistanceFromHome
JobSatisfaction
WorkLifeBalance
OverTime
NumCompaniesWorked
TrainingTimesLastYear
```

The dependent variable `y` is:

```text
Attrition
```

---

## ✂️ Train-Test Split

The dataset is divided into:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)
```

### Split Configuration

| Parameter | Value |
|---|---|
| Test Size | 20% |
| Training Size | 80% |
| Random State | 42 |

The code also displays:

```python
print("Dataset Size:", len(df))
print("Target Distribution:")
print(y.value_counts())
```

This helps verify the number of records and the distribution of the two attrition classes.

---

## ⚖️ Feature Scaling

Since the input features have different numerical ranges, `StandardScaler` is used.

```python
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)
```

The scaler is fitted only on the training data and then applied to the test data.

---

# 🧠 MLP Neural Network

The project uses:

```python
MLPClassifier
```

with two hidden layers.

### Architecture

```text
Input Layer
    ↓
Hidden Layer 1
16 Neurons
    ↓
Hidden Layer 2
8 Neurons
    ↓
Output Layer
    ↓
Attrition Prediction
```

The model is configured as:

```python
model = MLPClassifier(
    hidden_layer_sizes=(16, 8),
    activation="relu",
    solver="adam",
    learning_rate_init=0.001,
    max_iter=1000,
    random_state=42
)
```

### Model Parameters

| Parameter | Value |
|---|---|
| Hidden Layers | 2 |
| Layer 1 | 16 neurons |
| Layer 2 | 8 neurons |
| Activation | ReLU |
| Optimizer/Solver | Adam |
| Learning Rate | 0.001 |
| Maximum Iterations | 1000 |
| Random State | 42 |

---

## 🚀 Model Training

The neural network is trained using:

```python
model.fit(
    X_train_scaled,
    y_train
)
```

The number of iterations required for training is obtained using:

```python
model.n_iter_
```

---

# 📈 Model Evaluation

## Training Accuracy

```python
y_train_pred = model.predict(X_train_scaled)

train_accuracy = accuracy_score(
    y_train,
    y_train_pred
)
```

The training accuracy is displayed as a percentage.

---

## Testing Accuracy

```python
y_test_pred = model.predict(X_test_scaled)

test_accuracy = accuracy_score(
    y_test,
    y_test_pred
)
```

Testing accuracy is used to evaluate how the model performs on unseen data.

---

# 📋 Classification Report

The project generates a classification report containing:

- Precision
- Recall
- F1-score
- Support

The two classes are explicitly specified:

```python
classification_report(
    y_test,
    y_test_pred,
    labels=[0, 1],
    target_names=[
        "Likely to Stay",
        "Likely to Leave"
    ],
    zero_division=0
)
```

Using `labels=[0, 1]` ensures that the report maintains both expected classes even when a small test set happens to contain only one class.

---

# 📊 Confusion Matrix

The confusion matrix is generated using:

```python
cm = confusion_matrix(
    y_test,
    y_test_pred,
    labels=[0, 1]
)
```

The matrix is displayed graphically using:

```python
ConfusionMatrixDisplay
```

The labels are:

```text
Stay
Leave
```

This allows the model's predictions to be visually compared against the actual classes.

---

# 📉 Loss Curve

The training loss curve is plotted using:

```python
plt.plot(
    model.loss_curve_,
    label="Training Loss"
)
```

The graph contains:

- X-axis → Iterations
- Y-axis → Loss

The loss curve helps visualize how the model's training loss changes during learning.

---

# 🔮 PredictAttrition Function

The project includes a reusable:

```python
PredictAttrition(employee_data)
```

function.

The function accepts employee information in dictionary format.

Example:

```python
employee = {
    "Age": 30,
    "MonthlyIncome": 5000,
    "YearsAtCompany": 3,
    "TotalWorkingYears": 6,
    "DistanceFromHome": 10,
    "JobSatisfaction": 3,
    "WorkLifeBalance": 3,
    "OverTime": "Yes",
    "NumCompaniesWorked": 2,
    "TrainingTimesLastYear": 3
}

PredictAttrition(employee)
```

The function:

1. Converts the dictionary into a DataFrame.
2. Converts `OverTime` into numerical form.
3. Arranges the features in the correct order.
4. Applies the trained scaler.
5. Uses the trained MLP model for prediction.
6. Displays the predicted attrition class.
7. Displays the probability of staying.
8. Displays the probability of leaving.

---

# 👥 Five New Employee Predictions

The program tests the trained model using five new employee records.

The employee records contain:

```text
Age
MonthlyIncome
YearsAtCompany
TotalWorkingYears
DistanceFromHome
JobSatisfaction
WorkLifeBalance
OverTime
NumCompaniesWorked
TrainingTimesLastYear
```

The program loops through all five records:

```python
for i, employee in enumerate(
    employees,
    start=1
):
    print("\nEmployee", i)
    PredictAttrition(employee)
```

The output indicates whether each employee is predicted to:

```text
STAY
```

or

```text
LEAVE
```

along with the corresponding probabilities.

---

# 🔍 Overfitting and Underfitting Analysis

The project calculates the difference between training and testing accuracy:

```python
difference = (
    train_accuracy -
    test_accuracy
)
```

### Overfitting condition

The code checks:

```python
if (
    train_accuracy > 0.95
    and difference > 0.10
):
```

If this condition is satisfied, the program reports:

```text
Model may be suffering from OVERFITTING.
```

### Underfitting condition

The code checks:

```python
elif (
    train_accuracy < 0.70
    and test_accuracy < 0.70
):
```

If this condition is satisfied:

```text
Model may be suffering from UNDERFITTING.
```

Otherwise:

```text
Model does not show strong evidence
of overfitting or underfitting.
```

---

# 📦 Installation

Install the required Python libraries:

```bash
pip install pandas matplotlib scikit-learn
```

Or:

```bash
pip3 install pandas matplotlib scikit-learn
```

---

# ▶️ How to Run

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

### 2. Open the project directory

```bash
cd Employee-Attrition-Prediction
```

### 3. Install dependencies

```bash
pip install pandas matplotlib scikit-learn
```

### 4. Make sure the CSV is present

The code expects:

```text
Fraudulent_Attribution.csv
```

in the same directory as the Python file.

### 5. Run the program

```bash
python Sourcecode.py
```

---

# ⚠️ Important Dataset Requirement

The MLP model requires a dataset containing enough records for both classes:

```text
0 → Stay
1 → Leave
```

A very small dataset can cause problems during train/test splitting and evaluation.

For example, if the test set contains only one record while there are two classes, `stratify` cannot create a test set containing both classes.

For a meaningful machine-learning experiment, use a sufficiently large dataset containing examples of both attrition classes.

---

# 📁 Expected Dataset Columns

Your CSV should contain these columns:

```text
Age
MonthlyIncome
YearsAtCompany
TotalWorkingYears
DistanceFromHome
JobSatisfaction
WorkLifeBalance
OverTime
NumCompaniesWorked
TrainingTimesLastYear
Attrition
```

Example:

```text
Age,MonthlyIncome,YearsAtCompany,TotalWorkingYears,DistanceFromHome,JobSatisfaction,WorkLifeBalance,OverTime,NumCompaniesWorked,TrainingTimesLastYear,Attrition
```

---

# 📌 Project Workflow

```text
              Employee Dataset
                     │
                     ▼
              Load CSV File
                     │
                     ▼
            Explore Dataset
                     │
                     ▼
             Check Missing Data
                     │
                     ▼
             Encode Features
                     │
                     ▼
          Separate X and y
                     │
                     ▼
             Train/Test Split
                     │
                     ▼
              StandardScaler
                     │
                     ▼
              MLPClassifier
              ┌──────────────┐
              │ Hidden Layer │
              │  16 Neurons  │
              └──────┬───────┘
                     │
              ┌──────▼───────┐
              │ Hidden Layer │
              │   8 Neurons  │
              └──────┬───────┘
                     │
                     ▼
               Train Model
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
   Training Accuracy      Testing Accuracy
          │                     │
          └──────────┬──────────┘
                     ▼
             Model Evaluation
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
   Confusion     Classification   Loss
    Matrix          Report        Curve
                     │
                     ▼
            PredictAttrition()
                     │
                     ▼
           Five New Employees
                     │
                     ▼
          Stay / Leave Prediction
```

---

# 🎓 Concepts Covered

This project demonstrates the following Machine Learning and Deep Learning concepts:

- Data loading
- Data exploration
- Missing-value checking
- Feature identification
- Categorical encoding
- Feature selection
- Train-test splitting
- Feature normalization/standardization
- Multi-Layer Perceptron
- Artificial Neural Networks
- ReLU activation
- Adam optimization
- Model training
- Classification accuracy
- Confusion matrix
- Precision
- Recall
- F1-score
- Training loss
- Prediction probability
- Overfitting
- Underfitting

---

# 👨‍💻 Author

**Ashutosh Kadu**

Deep Learning Assignment  
Employee Attrition Prediction using MLPClassifier

---

## 📜 License

This project is created for **educational and academic purposes**.
