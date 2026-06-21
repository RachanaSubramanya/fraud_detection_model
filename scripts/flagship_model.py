# scripts/fraud_detection_model.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from scripts.algorithms import train_standard_rfc
from scripts.compiled_model_results import smote_resampled, get_metrics

def fraud_detection_model():
    print("\n--- Starting Fraud Detection Model Execution ---\n")
    print("Loading dataset...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "..", "data", "creditcard_data.csv")
    data = pd.read_csv(data_path)

    # --- 1. Generating EDA Correlation Matrix plot ---
    print("Generating Correlation matrix plot...")
    plt.figure(figsize=(12, 10))
    sns.heatmap(data.corr(), vmax=0.8, square=True)
    plt.title("Correlation Matrix - Credit Card Fraud Dataset")
    corr_path = os.path.join(script_dir, "..", "plots", "correlation_matrix.png")
    plt.savefig(corr_path, bbox_inches='tight')
    plt.close()
    print("Correlation matrix successfully generated and saved to plots/ folder as 'correlation_matrix.png'.")

    # --- 2. Preparing Pipeline ---
    X = data.drop(['Class'], axis=1).values
    Y = data['Class'].values
    xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y) 
    xTrain_res, yTrain_res = smote_resampled(xTrain, yTrain)

    # --- 3. Executing best fit model ---
    print("Deploying Flagship Production Model (Standard Random Forest). This will take a few moments...")
    production_predictions = train_standard_rfc(xTrain_res, yTrain_res, xTest)
    prec, rec, f1, mcc, missed = get_metrics(yTest, production_predictions)
    print("Production Model successfully executed. Compiling metrics summary...\n")

    print("\n" + "="*80 + "\n          PRODUCTION MODEL METRICS SUMMARY\n" + "="*80)
    print(f"Isolated Strategy: Standard Random Forest (with SMOTE)")
    print(f"Precision:         {prec:.2%}\nRecall (Catch):    {rec:.2%}\nF1-Score:          {f1:.2%}")
    print(f"MCC Rank:     {mcc:.4f}")
    print(f"Risk Assessment: {'High' if rec < 0.80 else 'Moderate' if rec < 0.90 else 'Low'} based on Recall metric")
    print("\nFraud transactions breakdown:")
    print(f"Total Fraudulent Transactions in Test Set: {sum(yTest == 1)}")
    print(f"Fraudulent Transactions Detected: {sum((yTest == 1) & (production_predictions == 1))}")
    print(f"Fraudulent Transactions Missed: {missed} out of {sum(yTest == 1)}")
    print("="*80 + "\n")

    print("\nExtracting and auditing false negative transactions...")
    results = pd.DataFrame(xTest, columns=data.drop(['Class'], axis=1).columns)
    results['Actual'] = yTest
    results['Predicted'] = production_predictions

    # Isolating the exact rows where actual fraud (1) slipped past as normal (0)
    false_neg = results[(results['Actual'] == 1) & (results['Predicted'] == 0)]

    print(f"Total frauds missed: {len(false_neg)}")
    print("Amount details of missed transactions (Sorted Low to High):")
    print(false_neg[['Time', 'Amount']].sort_values(by='Amount', ascending=True).to_string(index=False))


    # --- 4. Generating Confusion Matrix Visual ---
    print("\nPlotting Production Confusion Matrix...")
    cm = confusion_matrix(yTest, production_predictions)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Valid', 'Fraud'], yticklabels=['Valid', 'Fraud'])
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix - Flagship Model (Standard Random Forest)')
    cm_path = os.path.join(script_dir, "..", "plots", "confusion_matrix.png")
    plt.savefig(cm_path, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix successfully generated and saved to plots/ folder as 'confusion_matrix.png'.")
    print("\n--- Flagship Model Execution Completed ---\n")


if __name__ == "__main__":
    fraud_detection_model()
