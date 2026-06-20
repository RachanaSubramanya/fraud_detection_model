# importing necessary libraries
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef

# --- LOCAL PACKAGED MODULE IMPORTS ---
from scripts.algorithms import (
    train_standard_rfc, train_optimized_rfc, train_xgboost, 
    train_balanced_rfc, train_logistic_regression, train_linear_svc, train_knn
)

from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

def smote_resampled(x_train, y_train):
    # Rebalanceing the dataset rows using SMOTE.
    sm = SMOTE(sampling_strategy=0.1, random_state=42)
    x_train_res, y_train_res = sm.fit_resample(x_train, y_train)
    return x_train_res, y_train_res


def scale_data(x_train_res, x_test):    
    # Standardizing column scales for distance-based models.
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train_res)       
    x_test_scaled = scaler.transform(x_test)                 
    return x_train_scaled, x_test_scaled


def get_metrics(y_true, y_pred):
    # Calculating standardized metrics grid for transaction evaluation.
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred) 
    missed_frauds = sum((y_true == 1) & (y_pred == 0))
    return prec, rec, f1, mcc, missed_frauds


def model_evaluation():
    # Loading dataset
    print("\n--- Starting Evaluation ---\n")
    print("Loading dataset...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "..", "data", "creditcard_data.csv")
    data = pd.read_csv(data_path)

    # Setting up training and testing datasets
    print("Undergoing pre-processing steps...)")
    X = data.drop(['Class'], axis=1).values
    Y = data['Class'].values
    xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y) 

    # applying SMOTE resampling to balance the training dataset for tree-based models 
    print("Applying SMOTE resampling...")
    xTrain_res, yTrain_res = smote_resampled(xTrain, yTrain)

    # applying feature scaling to the resampled training data and original test data for distance-based models
    print("Applying Feature Scaling to resampled data...")
    xTrain_scaled, xTest_scaled = scale_data(xTrain_res, xTest)


    print("\n--- Training Benchmark Grid Started ---")
    print("This will take a few moments as we are training 7 different models for benchmarking purposes\n")

    print("Training Model 1: Standard Random Forest Classifier...")
    pred_m1 = train_standard_rfc(xTrain_res, yTrain_res, xTest)

    print("Training Model 2: Optimized Random Forest Classifier...")
    pred_m2 = train_optimized_rfc(xTrain_res, yTrain_res, xTest)

    print("Training Model 3: XGBoost Classifier...")
    pred_m3 = train_xgboost(xTrain, yTrain, xTest)

    print("Training Model 4: Balanced Random Forest Classifier...")
    pred_m4 = train_balanced_rfc(xTrain, yTrain, xTest)

    print("Training Model 5: Logistic Regression...")
    pred_m5 = train_logistic_regression(xTrain_scaled, yTrain_res, xTest_scaled)

    print("Training Model 6: Linear Support Vector Classifier...")
    pred_m6 = train_linear_svc(xTrain_scaled, yTrain_res, xTest_scaled)

    print("Training Model 7: K-Nearest Neighbors Classifier...")
    pred_m7 = train_knn(xTrain_scaled, yTrain_res, xTest_scaled)

    print("\nAll models trained successfully. Compiling metrics for each model...\n")                                              

    # Compiling metrics for all models into a comprehensive comparison report
    all_metrics = [get_metrics(yTest, p) for p in [pred_m1, pred_m2, pred_m3, pred_m4, pred_m5, pred_m6, pred_m7]]
    
    model_data = {
        'Model Strategy': [
            'RandomForestClassifier (with SMOTE)', 'RandomForestClassifier (Optimized using GridSearchCV)', 
            'XGBoost Classifier (with scale_pos_weight)', 'BalancedRandomForestClassifier (with internal class balancing)',
            'Logistic Regression (Baseline on Scaled SMOTE Data)', 'Linear Support Vector Classifier (LinearSVC)', 'K-Nearest Neighbors (KNN Classifier)'
        ],
        'Precision': [m[0] for m in all_metrics],
        'Recall': [m[1] for m in all_metrics],
        'F1-Score': [m[2] for m in all_metrics],
        'Matthews Correlation Coefficient': [m[3] for m in all_metrics], 
        'Total Frauds Missed': [m[4] for m in all_metrics]
    }
    
    comparison_df = pd.DataFrame(model_data).sort_values(by='Matthews Correlation Coefficient', ascending=False)

    print("\n" + "="*120 + "\n                       FINAL MODEL COMPARISON REPORT\n" + "="*120)
    print(comparison_df.to_string(index=False, formatters={
        'Precision': '{:,.2%}'.format, 'Recall': '{:,.2%}'.format, 'F1-Score': '{:,.2%}'.format,
        'Matthews Correlation Coefficient': '{:,.4f}'.format, 'Total Frauds Missed': '{:,.0f}'.format 
    }))
    print("="*120)
    print("\nConclusion: Based on the benchmarking data, the standard RandomForestClassifier (with SMOTE) is selected as the best-fit production model."\
          "While models like Logistic Regression missed fewer total frauds (10 vs 17), they suffered from a very low precision rate of ~13%,"\
          "which would result in thousands of innocent customer accounts being frozen by mistake."\
          "The RandomForestClassifier (with SMOTE) strikes a better balance between precision and recall, "\
          "making it the most suitable choice for deployment in a real-world fraud detection scenario.\n")
    print("\n--- Evaluation Completed ---\n")

if __name__ == "__main__":
    model_evaluation()
