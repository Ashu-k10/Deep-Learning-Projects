# ============================================================
# Deep Learning Assignment
# Employee Attrition Prediction using MLPClassifier
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

# ============================================================
# 1. Load Dataset
# ============================================================

df = pd.read_csv("Fraudulent_Attribution.csv")

print("1. Load Dataset ")
print(df)

# ============================================================
# 2. Display Shape, Columns and First Five Records
# ============================================================

print("2. Display Shape, Columns and First Five Records")
print("\n========== DATASET SHAPE ==========")
print(df.shape)

print("\n========== COLUMNS ==========")
print(df.columns)

print("\n========== FIRST FIVE RECORDS ==========")
print(df.head())

# ============================================================
# 3. Check Missing Values
# ============================================================

print("3. Check Missing Values")
print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

# ============================================================
# 4. Identify Numerical and Categorical Features
# ============================================================

print("4. Identify Numerical and Categorical Features")

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

categorical_features = [
    "OverTime"
]

print("\n========== NUMERICAL FEATURES ==========")
print(numerical_features)

print("\n========== CATEGORICAL FEATURES ==========")
print(categorical_features)

# ============================================================
# 5. Convert OverTime into Numerical Representation
# Yes -> 1
# No  -> 0
# ============================================================

print("5. Convert OverTime into Numerical Representation")

df["OverTime"] = df["OverTime"].map({
    "Yes": 1,
    "No": 0
})

# ============================================================
# 6. Convert Attrition into 0 and 1
# Yes -> 1 (Likely to Leave)
# No  -> 0 (Likely to Stay)
# ============================================================
print("6. Convert Attrition into 0 and 1")

df["Attrition"] = df["Attrition"].map({
    "Yes": 1,
    "No": 0
})

print("\n========== AFTER ENCODING ==========")
print(df.head())

# ============================================================
# 7. Separate Independent and Dependent Variables
# ============================================================

X = df[
    [
        "Age",
        "MonthlyIncome",
        "YearsAtCompany",
        "TotalWorkingYears",
        "DistanceFromHome",
        "JobSatisfaction",
        "WorkLifeBalance",
        "OverTime",
        "NumCompaniesWorked",
        "TrainingTimesLastYear"
    ]
]

y = df["Attrition"]

print("\n========== X ==========")
print(X.head())

print("\n========== y ==========")
print(y.head())

# ============================================================
# 8. Split Dataset into Training and Testing Data
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,

)
print("Dataset Size:", len(df))
print("Target Distribution:")
print(y.value_counts())

print("\n========== DATA SPLIT ==========")
print("Training Data :", X_train.shape)
print("Testing Data  :", X_test.shape)

# ============================================================
# 9. Feature Scaling
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# 10. Design MLP with At Least Two Hidden Layers
#
# Hidden Layer 1 = 16 neurons
# Hidden Layer 2 = 8 neurons
# ============================================================

model = MLPClassifier(
    hidden_layer_sizes=(16, 8),
    activation="relu",
    solver="adam",
    learning_rate_init=0.001,
    max_iter=1000,
    random_state=42
)

print("\n========== MLP MODEL ==========")
print(model)

# ============================================================
# 11. Train the Network
# ============================================================

model.fit(X_train_scaled, y_train)

print("\n========== MODEL TRAINED SUCCESSFULLY ==========")

# ============================================================
# 12. Display Number of Iterations Required for Training
# ============================================================

print("\n========== TRAINING INFORMATION ==========")
print("Number of Iterations :", model.n_iter_)

# ============================================================
# 13. Calculate Training Accuracy
# ============================================================

y_train_pred = model.predict(X_train_scaled)

train_accuracy = accuracy_score(
    y_train,
    y_train_pred
)

print("\n========== TRAINING ACCURACY ==========")
print("Training Accuracy :", train_accuracy * 100, "%")

# ============================================================
# 14. Calculate Testing Accuracy
# ============================================================

y_test_pred = model.predict(X_test_scaled)

test_accuracy = accuracy_score(
    y_test,
    y_test_pred
)

print("\n========== TESTING ACCURACY ==========")
print("Testing Accuracy :", test_accuracy * 100, "%")

# ============================================================
# 15. Generate Confusion Matrix
# ============================================================

cm = confusion_matrix(
    y_test,
    y_test_pred
)

print("\n========== CONFUSION MATRIX ==========")
print(cm)

# ============================================================
# 15. Classification Report
# ============================================================

print("\n========== CLASSIFICATION REPORT ==========")

print(
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
)


# ============================================================
# Confusion Matrix
# ============================================================

cm = confusion_matrix(
    y_test,
    y_test_pred,
    labels=[0, 1]
)

print("\n========== CONFUSION MATRIX ==========")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Stay",
        "Leave"
    ]
)

disp.plot()

plt.title("Employee Attrition - Confusion Matrix")

plt.show()


# ============================================================
# 16. Plot Loss Curve
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    model.loss_curve_,
    label="Training Loss"
)

plt.title("MLP Training Loss Curve")
plt.xlabel("Iterations")
plt.ylabel("Loss")

plt.legend()
plt.grid()

plt.show()


# ============================================================
# 17. Create PredictAttrition(employee_data) Function
# ============================================================

def PredictAttrition(employee_data):

    # --------------------------------------------------------
    # Convert dictionary into DataFrame
    # --------------------------------------------------------

    employee_df = pd.DataFrame(
        [employee_data]
    )

    # --------------------------------------------------------
    # Convert OverTime into numerical value
    # Yes = 1
    # No  = 0
    # --------------------------------------------------------

    employee_df["OverTime"] = employee_df[
        "OverTime"
    ].map({
        "Yes": 1,
        "No": 0
    })

    # --------------------------------------------------------
    # Arrange features in the same order used during training
    # --------------------------------------------------------

    employee_df = employee_df[
        [
            "Age",
            "MonthlyIncome",
            "YearsAtCompany",
            "TotalWorkingYears",
            "DistanceFromHome",
            "JobSatisfaction",
            "WorkLifeBalance",
            "OverTime",
            "NumCompaniesWorked",
            "TrainingTimesLastYear"
        ]
    ]

    # --------------------------------------------------------
    # Scale the employee data
    # --------------------------------------------------------

    employee_scaled = scaler.transform(
        employee_df
    )

    # --------------------------------------------------------
    # Make prediction
    # --------------------------------------------------------

    prediction = model.predict(
        employee_scaled
    )[0]

    # --------------------------------------------------------
    # Get prediction probability
    # --------------------------------------------------------

    probability = model.predict_proba(
        employee_scaled
    )[0]

    # --------------------------------------------------------
    # Display Prediction
    # --------------------------------------------------------

    print("\n========== ATTRITION PREDICTION ==========")

    if prediction == 0:

        print(
            "Prediction : Employee is likely to STAY"
        )

    else:

        print(
            "Prediction : Employee is likely to LEAVE"
        )

    # --------------------------------------------------------
    # Display probabilities
    # --------------------------------------------------------

    print(
        "Probability of Staying :",
        round(probability[0] * 100, 2),
        "%"
    )

    print(
        "Probability of Leaving :",
        round(probability[1] * 100, 2),
        "%"
    )

    return prediction


# ============================================================
# 18. Test System Using Five New Employee Records
# ============================================================

employees = [

    {
        "Age": 25,
        "MonthlyIncome": 3000,
        "YearsAtCompany": 1,
        "TotalWorkingYears": 2,
        "DistanceFromHome": 15,
        "JobSatisfaction": 2,
        "WorkLifeBalance": 2,
        "OverTime": "Yes",
        "NumCompaniesWorked": 2,
        "TrainingTimesLastYear": 2
    },

    {
        "Age": 35,
        "MonthlyIncome": 7000,
        "YearsAtCompany": 7,
        "TotalWorkingYears": 10,
        "DistanceFromHome": 5,
        "JobSatisfaction": 4,
        "WorkLifeBalance": 4,
        "OverTime": "No",
        "NumCompaniesWorked": 1,
        "TrainingTimesLastYear": 4
    },

    {
        "Age": 29,
        "MonthlyIncome": 4000,
        "YearsAtCompany": 3,
        "TotalWorkingYears": 5,
        "DistanceFromHome": 20,
        "JobSatisfaction": 2,
        "WorkLifeBalance": 2,
        "OverTime": "Yes",
        "NumCompaniesWorked": 3,
        "TrainingTimesLastYear": 2
    },

    {
        "Age": 42,
        "MonthlyIncome": 9000,
        "YearsAtCompany": 12,
        "TotalWorkingYears": 18,
        "DistanceFromHome": 4,
        "JobSatisfaction": 4,
        "WorkLifeBalance": 4,
        "OverTime": "No",
        "NumCompaniesWorked": 2,
        "TrainingTimesLastYear": 5
    },

    {
        "Age": 31,
        "MonthlyIncome": 5000,
        "YearsAtCompany": 2,
        "TotalWorkingYears": 7,
        "DistanceFromHome": 25,
        "JobSatisfaction": 1,
        "WorkLifeBalance": 2,
        "OverTime": "Yes",
        "NumCompaniesWorked": 4,
        "TrainingTimesLastYear": 1
    }
]


print("\n\n================================================")
print("PREDICTIONS FOR FIVE NEW EMPLOYEES")
print("================================================")


for i, employee in enumerate(
    employees,
    start=1
):

    print("\nEmployee", i)

    PredictAttrition(employee)


# ============================================================
# 19. Check for Overfitting / Underfitting
# ============================================================

print("\n========== MODEL ANALYSIS ==========")

difference = (
    train_accuracy -
    test_accuracy
)


print(
    "Training Accuracy :",
    round(
        train_accuracy * 100,
        2
    ),
    "%"
)


print(
    "Testing Accuracy  :",
    round(
        test_accuracy * 100,
        2
    ),
    "%"
)


print(
    "Accuracy Difference :",
    round(
        difference * 100,
        2
    ),
    "%"
)


# ------------------------------------------------------------
# Overfitting / Underfitting Analysis
# ------------------------------------------------------------

if (
    train_accuracy > 0.95
    and difference > 0.10
):

    print(
        "\nModel may be suffering from OVERFITTING."
    )


elif (
    train_accuracy < 0.70
    and test_accuracy < 0.70
):

    print(
        "\nModel may be suffering from UNDERFITTING."
    )


else:

    print(
        "\nModel does not show strong evidence "
        "of overfitting or underfitting."
    )


# ============================================================
# END OF PROGRAM
# ============================================================
