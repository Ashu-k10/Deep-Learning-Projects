#==================================================================================
# Customer Chrun Prediction :
# By Ashutosh Kadu 
#==================================================================================

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score,classification_report

#====================================
# Step 1 : Import Dataset 
#====================================

X = np.array([
    [25,500,12,1,2],
    [30,700,24,0,1],
    [45,1200,6,5,8],
    [50, 1500, 5, 6, 10],
    [28, 600, 18, 1, 1],
    [35, 800, 30, 0, 0],
    [48, 1400, 4, 7, 9],
    [52, 1600, 3, 8, 12],
    [27, 550, 20, 0, 1],
    [42, 1300, 8, 4, 7]

])

# 0 = Customer will stay
# 1 = Customer will leave

Y = np.array([
    0,0,1,1,0,
    0,1,1,0,1
])

#====================================
# Step 2 : Split Dataset 
#====================================

X_train,X_test,Y_train,Y_Test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42,
    stratify=Y
    
)

#====================================
# Step 3 : Feature Scaling 
#====================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#====================================
# Step 4 : Create FNN Model
#====================================

model = MLPClassifier(
    hidden_layer_sizes=(8,4),
    activation='relu',
    solver = 'lbfgs',
    max_iter=5000,
    random_state=42
)

#====================================
# Step 5 : Train Model
#====================================

model.fit(X_train,Y_train)

#====================================
# Step 6 : Evaluate Model
#====================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(Y_Test, y_pred)

print("Model Accuracy :",accuracy)
print("\nClassification Report:")
print(classification_report(Y_Test, y_pred))

# ============================================================
# Step 7 : Test Input
# ============================================================

new_customer = np.array([
    [46, 1450, 5, 6, 9]
])

new_customer_scaled = scaler.transform(new_customer)

prediction = model.predict(new_customer_scaled)

# ============================================================
# Step 8 : Final Prediction
# ============================================================

if prediction[0] == 1:
    print("\nPrediction: Customer may leave")
else:
    print("\nPrediction: Customer will stay")
