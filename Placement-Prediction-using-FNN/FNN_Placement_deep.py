###########################################################
#----------------------------------------------------
# Deep Learning Pipeline 
#----------------------------------------------------
# 1. Read the Data from CSV
# 2. Data Analysis (EDA)
# 3. Preprocessing
# 4. Train Test Split
# 5. Feature Engineering 
# 6. FNN Model Training 
# 7. Model Evaluation
# 8. Graphical Representation
# 9. Model preserve
# 10. Model loading and preserve
# 11. Test Unseen Data
###########################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score,confusion_matrix

#######################################
# 1. Read the Data from CSV           
#######################################

print("1. Read the Data from CSV")

data = pd.read_csv('Placement_data.csv')

print("Complete Dataset :")
print(data)

####################################
# 2. Data Analysis (EDA)
####################################

print("2. Data Analysis (EDA)")

print("First 5 rows :")
print(data.head())

print("Columns names :")
print(data.columns)

print("Shape of Dataset : ")
print(data.shape)

print("Statistical Summary :")
print(data.describe())

####################################
# 3. Preprocessing 
####################################

print("3.Preprocessing")

X = data[['Aptitude', 'Coding', 'Communication', 'Academics', 'Internship']]
Y = data['Placed']

print("Input Features :")
print(X.head())

print("Target :")
print(Y.head())

###################################
# 4. Train Test Split
###################################

print("4. Train Test Split")

X_train,X_Test,Y_train,Y_test = train_test_split(X,Y,test_size=0.30,random_state=42)

print("Training Input Shape : ",X_train.shape)
print("Testing Input Shape : ",X_Test.shape)
print("Training Output Shape : ",Y_train.shape)
print("Testing Input Shape : ",Y_train.shape)

######################################
# 5. Feature Engineering 
#######################################

print("5. Feature Engineering")

scalar = StandardScaler()

X_train_scaled = scalar.fit_transform(X_train)
X_Test_scaled = scalar.fit_transform(X_Test)

print("Scaled training Data :")
print(X_train_scaled[:5])   # first 5 scaled values

###################################
# 6. FNN Model Training 
###################################

print("6. FNN Model Training")

model = MLPClassifier(
    hidden_layer_sizes=(8,4),    # first hidden layer contains 8 and second contains 4
    activation='relu',          
    solver="adam",
    max_iter=1000,
    random_state=42
)

print(model)

print("Train the Model :")
model.fit(X_train_scaled,Y_train)

print("Model Training Completed : ")

##############################
# 7. Model Evaluation
##############################

print("7. Model Evaluation")

y_pred = model.predict(X_Test_scaled)

accuracy = accuracy_score(Y_test,y_pred)
print("Accuracy is : ",accuracy)

cm = confusion_matrix(Y_test,y_pred)
print("Confusion Matrix : ",cm)

print("Predict the probablity :")
Y_prob = model.predict_proba(X_Test_scaled)

print(Y_prob[:5])

######################################
# 8. Graphical Representation
######################################

# Actual vs Predicted Placement
plt.figure(figsize=(8,5))

plt.plot(range(len(Y_test)), Y_test.values,
         marker='o', label='Actual')

plt.plot(range(len(y_pred)), y_pred,
         marker='x', label='Predicted')

plt.xlabel("Student Index")
plt.ylabel("Placement (0 = Not Placed, 1 = Placed)")
plt.title("Actual vs Predicted Placement")
plt.legend()
plt.grid(True)
plt.show()


# Confusion Matrix Graph
plt.figure(figsize=(6,5))

plt.imshow(cm, interpolation='nearest')
plt.title("Confusion Matrix")
plt.colorbar()

plt.xticks([0, 1], ['Not Placed', 'Placed'])
plt.yticks([0, 1], ['Not Placed', 'Placed'])

plt.xlabel("Predicted")
plt.ylabel("Actual")

# Display values inside the matrix
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j],
                 ha="center",
                 va="center")

plt.show()


# Prediction Probability Graph
plt.figure(figsize=(8,5))

plt.plot(range(len(Y_prob)),
         Y_prob[:, 1],
         marker='o')

plt.xlabel("Student Index")
plt.ylabel("Probability of Placement")
plt.title("Placement Prediction Probability")
plt.grid(True)
plt.show()

#############################
# 9. Model preserve
#############################

print("9. Model Loading and preserve : ")

joblib.dump(model,"placement_fnn_model.pkl")
joblib.dump(scalar,"placement_scalar.pkl")

print("Model and Scalar gets dump Sucessfully")

#####################################
# 10. Model Loading and preserve
#####################################

print("10. Model Loading and preserve")

loaded_model = joblib.load("placement_fnn_model.pkl")
loaded_scalar = joblib.load("placement_scalar.pkl")

print("Model gets Loaded Sucessfully")

#################################
#
# 11. Test Unseen Data
#
# Aptitude :       70
# Coding :         75
# Communication :  80
# Acedemics :      85
# Internship :     1
#################################

new_student = pd.DataFrame([[70,75,80,85,1]], columns=['Aptitude', 'Coding', 'Communication', 'Academics', 'Internship'])

new_student_scaled = loaded_scalar.transform(new_student)

new_prediction = loaded_model.predict(new_student_scaled)

new_probablity = loaded_model.predict_proba(new_student_scaled)

print("New students Data : ")
print(new_student)

print("Prediction Probablity : ",new_probablity)

if new_prediction[0] == 1:
    print("Prediction : Placed ")
else :
    print("Prediction : Not Placed")
