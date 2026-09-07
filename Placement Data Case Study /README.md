# 🎓 Placement Prediction using FNN

A simple **Deep Learning project** that predicts whether a student will be placed based on their academic and skill-related attributes.

## 📌 Features

* Exploratory Data Analysis (EDA)
* Data preprocessing
* Train-test split
* Feature scaling using `StandardScaler`
* Feed Forward Neural Network (FNN) using `MLPClassifier`
* Model evaluation using Accuracy and Confusion Matrix
* Prediction probability
* Model saving and loading using Joblib
* Prediction on unseen student data

## 🧠 Input Features

* Aptitude
* Coding
* Communication
* Academics
* Internship

**Target:** `Placed`

## ⚙️ Model

The project uses a Feed Forward Neural Network with:

* Hidden Layers: `(8, 4)`
* Activation: `ReLU`
* Optimizer: `Adam`
* Maximum Iterations: `1000`

## 🛠️ Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Joblib

## 🚀 How to Run

```bash
pip install pandas numpy matplotlib scikit-learn joblib
python FNN_Placement_deep.py
```

Make sure `Placement_data.csv` is in the same directory as the Python file.

## 📂 Project Structure

```text
├── FNN_Placement_deep.py
├── Placement_data.csv
├── placement_fnn_model.pkl
├── placement_scalar.pkl
└── README.md
```

## 🎯 Output

The trained model predicts:

```text
Placed
```

or

```text
Not Placed
```

along with the prediction probability.

---

**Deep Learning | FNN | Machine Learning | Placement Prediction**
