#importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk
import os

#loading the data
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(scripts_dir, "..", "data", "creditcard_data.csv")
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

#plotting correlation matrix
corrmat = data.corr()
fig = plt.figure(figsize=(12,9))
sns.heatmap(corrmat, vmax=0.8, square=True)
plt.title("Correlation Matrix", fontsize=16, fontweight='bold', pad=20)
plt.savefig(os.path.join(script_dir, "..", "plots", "correlation_matrix.png"))
plt.close()
print("Correlation matrix plotted and saved to repository 'plots' successfully.")

#preparing data
X = data.drop(['Class'], axis=1)
Y = data['Class']
print(X.shape)
print(Y.shape)

xData = X.values
yData = Y.values

#splitting data into train and test sets
from sklearn.model_selection import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(
    xData, yData, test_size=0.2, random_state=42, stratify=Y) 

#adjusting for imbalance in data
from imblearn.over_sampling import SMOTE
sm = SMOTE(sampling_strategy=0.1, random_state=42)
xTrain_res, yTrain_res = sm.fit_resample(xTrain, yTrain )

#training the data with all the algorithms
print("--- Training Started (This will take a few minutes) ---")

#fitting a random forest classifier
print("step 1: Fitting a Random Forest Classifier...")
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rfc.fit(xTrain_res, yTrain_res)
yPred = rfc.predict(xTest)

#evaluating the model metrics
print("step 2: Evaluating the model...")
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, matthews_corrcoef, confusion_matrix, classification_report)
accuracy = accuracy_score(yTest, yPred)
precision = precision_score(yTest, yPred)
recall = recall_score(yTest, yPred)
f1 = f1_score(yTest, yPred)
mcc = matthews_corrcoef(yTest, yPred)
print("Model evaluation completed.")

print("Model evaluation metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"Mathew's Correlation Coefficient: {mcc:.4f}")

# Printing a detailed classification report onto the terminal
print("\n" + "="*100)
print("\n         Detailed Report:")
print("\n" + "="*100)
print(classification_report(yTest, yPred))
print("\n" + "="*100)

#creating a confusion matrix to visualise the predictions of the model
conf_mat = confusion_matrix(yTest, yPred)
plt.figure(figsize=(8,6))
sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues",
            xticklabels=['Normal', 'Fraud'], yticklabels=['Normal', 'Fraud'])
plt.title("Confusion matrix")
plt.xlabel("Predicted class")
plt.ylabel("True class")
plt.savefig(os.path.join(script_dir, "..", "plots", "confusion_matrix.png"))
print("Confusion matrix plotted and saved to repository 'plots' successfully.")

#creating a dataframe for the prediction results
results = pd.DataFrame(xTest, columns=X.columns)
results['Actual'] = yTest
results['Predicted'] = yPred

#isolating false negetives
false_neg = results[(results['Actual'] == 1) & (results['Predicted'] == 0)]

print(f"Total frauds missed: {len(false_neg)}")
print("\nAmount of missed transactions:")
print(false_neg.sort_values(by='Amount', ascending=True))

