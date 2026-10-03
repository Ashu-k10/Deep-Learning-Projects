# =================================================================================
# Loan Approval Prediction Using Neural Network 
# =================================================================================

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

# ============================================================
# Step 1 : Find Dataset
# ============================================================

X = np.array([
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
])

# 0 = Loan rejected
# 1 = Loan approved

y = np.array([
    0, 1, 1, 0, 1,
    1, 0, 1, 0, 1
])

# ============================================================
# Step 2 : Split Dataset
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ============================================================
# Feature Scaling
# ============================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ============================================================
# Step 3 : Create FNN Model
# ============================================================

model = MLPClassifier(
    hidden_layer_sizes=(8, 4),
    activation='relu',
    solver='lbfgs',
    max_iter=5000,
    random_state=42
)

# ============================================================
# Step 4 :Train Model
# ============================================================

model.fit(X_train, y_train)

# ============================================================
# Step 5 : Evaluate Model
# ============================================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ============================================================
# Step 6 : Test Input
# ============================================================

new_applicant = np.array([
    [55000, 720, 400000, 10000, 1]
])

new_applicant_scaled = scaler.transform(new_applicant)

prediction = model.predict(new_applicant_scaled)

# ============================================================
# Step 7 :Final Prediction
# ============================================================

if prediction[0] == 1:
    print("\nPrediction: Loan Approved")
else:
    print("\nPrediction: Loan Rejected")
