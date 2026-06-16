#importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk
import os

#loading the data
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "..", "data", "creditcard_data.csv")
data = pd.read_csv(data_path)

print(data.head())
print(data.describe())

#understanding class distribution of dataset
fraud = data[data['Class'] == 1]
valid = data[data['Class'] == 0]
classRatio = len(fraud)/len(valid)
print(f"Ratio of classes: {classRatio} i.e. {classRatio*100:.4f}%" )
print("Fraud transactions: {}".format(len(fraud)))
print("Valid transactions: {}".format(len(valid)))

#understanding amount details of transactions
print("Amount details of the fraudulent transactions", fraud.Amount.describe())
print("Amount details of the valid transactions", valid.Amount.describe())  

#preparing data
X = data.drop(['Class'], axis=1)
Y = data['Class']
print(X.shape)
print(Y.shape)

xData = X.values
yData = Y.values


#preparing the data for training and testing using different algorithms
# 1. importing necessary libraries
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

# 2. Split original data
xTrain, xTest, yTrain, yTest = train_test_split(
    xData, yData, test_size=0.2, random_state=42, stratify=Y) 

# 3. Prepare the SMOTE training data (For Random Forest variations)
sm = SMOTE(sampling_strategy=0.1, random_state=42)
xTrain_res, yTrain_res = sm.fit_resample(xTrain, yTrain)


#training the data with all the algorithms
print("--- Training Started (This will take a few minutes) ---")

# --- MODEL 1: Standard Random Forest ---
print("Training Model 1: Standard Random Forest...")
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
rfc.fit(xTrain_res, yTrain_res)
pred_m1 = rfc.predict(xTest)

# --- MODEL 2: Optimized Random Forest ---
print("Training Model 2: Optimized Random Forest...")
# A GridSearchCV was performed offline to find the best parameters, which are now used to train the optimized model
# The GridSearchCV code and results are not included here to keep the script focused on the final model training and evaluation
rfc_opt = RandomForestClassifier(n_estimators=150, max_depth=None, min_samples_split=2, 
                                 class_weight='balanced', random_state=42, n_jobs=-1)
rfc_opt.fit(xTrain_res, yTrain_res)
pred_m2 = rfc_opt.predict(xTest)

# --- MODEL 3: XGBoost Classifier ---
print("Training Model 3: XGBoost Classifier...")
from xgboost import XGBClassifier
scale_factor = sum(yTrain == 0) / sum(yTrain == 1)
xgb_model = XGBClassifier(n_estimators=150, learning_rate=0.05, max_depth=5, scale_pos_weight=scale_factor, 
                          random_state=42, eval_metric='logloss', n_jobs=-1)
xgb_model.fit(xTrain, yTrain) # Note: XGBoost can handle imbalance with scale_pos_weight, so we use the original training data
pred_m3 = xgb_model.predict(xTest)

# --- MODEL 4: Balanced Random Forest ---
print("Training Model 4: Balanced Random Forest...")
from imblearn.ensemble import BalancedRandomForestClassifier
brfc = BalancedRandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
brfc.fit(xTrain, yTrain) 
pred_m4 = brfc.predict(xTest)


# Importing model evaluation metrics and compiling the results
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef

# Creating a function to generate clean lists containing the results of each model 
def get_metrics(y_true, y_pred):
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred) 
    missed = sum((y_true == 1) & (y_pred == 0))
    return prec, rec, f1, mcc, missed

# Calculating scores for all predictions
m1_metrics = get_metrics(yTest, pred_m1)
m2_metrics = get_metrics(yTest, pred_m2)
m3_metrics = get_metrics(yTest, pred_m3)
m4_metrics = get_metrics(yTest, pred_m4)

# Creating the final comparison summary
model_data = {
    'Model Strategy': [
        'RandomForestClassifier (with SMOTE)', 
        'RandomForestClassifier (Optimized using GridSearchCV)', 
        'XGBoost Classifier (with scale_pos_weight)', 
        'BalancedRandomForestClassifier (with class balancing)'
    ],
    'Precision': [m1_metrics[0], m2_metrics[0], m3_metrics[0], m4_metrics[0]],
    'Recall': [m1_metrics[1], m2_metrics[1], m3_metrics[1], m4_metrics[1]],
    'F1-Score': [m1_metrics[2], m2_metrics[2], m3_metrics[2], m4_metrics[2]],
    'Matthews Correlation Coefficient': [m1_metrics[3], m2_metrics[3], m3_metrics[3], m4_metrics[3]], 
    'Total Frauds Missed': [m1_metrics[4], m2_metrics[4], m3_metrics[4], m4_metrics[4]]
}

comparison_df = pd.DataFrame(model_data)

# Printing a clean report grid to terminal
print("\n" + "="*100)
print("                       FINAL MODEL COMPARISON REPORT")
print("="*100)
print(comparison_df.to_string(index=False, formatters={
    'Precision': '{:,.2%}'.format,
    'Recall': '{:,.2%}'.format,
    'F1-Score': '{:,.2%}'.format,
    'Matthews Correlation Coefficient': '{:,.4f}'.format,
    'Total Frauds Missed': '{:,.0f}'.format 
}))
print("="*100)

