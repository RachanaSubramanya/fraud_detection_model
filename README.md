# Fraud detection model

_This repository contains a machine learning pipeline engineered to detect fraudulent credit card transactions. The project addresses extreme class imbalance using advanced resampling techniques and evaluates performance using metrics suited for highly skewed datasets.._

---

## Table of contents
- <a href= "#overview">Overview</a>
- <a href= "#business-problem">Business problem</a>
- <a href= "#dataset">Dataset</a>
- <a href= "#tools--technologies">Tools and Technologies</a>
- <a href= "#project-structure">Project Structure</a>
- <a href= "#data-cleaning-preparation">Data Cleaning and Preparation</a>
- <a href= "#exploratory-data-analysis">Exploatory Data Analysis</a>
- <a href= "#research-questions-key-findings">Research questions and Key findings</a>
- <a href= "#dashboard">Dashboard</a>
- <a href= "#how-to-run-this-project">How to run this project</a>
- <a href= "#final-recommendations">Final Recommendations</a>
- <a href= "#author-contact">Author & Contact</a>

---

<h2><a class="anchor" id="overview"></a>Overview</h2>

The project analyzes transactional data to identify anomalous patterns indicative of fraud. Because fraud data is notoriously sparse, the codebase implements a robust preprocessing and modeling workflow:
* **Exploratory Data Analysis:** Visualises dataset correlations and distributions.
* **Imbalance Handling:** Utilises Synthetic Minority Over-sampling Technique (SMOTE) to synthetically balance training classes safely.
* **Classification Algorithm:** Trains a Random Forest Classifier with balanced class weights to maximize fraud isolation.
* **Granular Evaluation:** Looks beyond standard accuracy to focus heavily on false-negative mitigation and Matthews Correlation Coefficient (MCC) tracking.

---

<h2><a class="anchor" id="business-problem"></a>Business Problem</h2>

Train ML classifiers on transaction datasets with anomaly detection.
This model focuses on training with labelled data to make predictions about the nature of the financial transactions, i.e. being valid or fraudulent transactions. The data has been fitted with a target column, which will act as the basis of the training. The target column corresponds to either of the 2 values: 0 or 1, with 0 representing valid transactions, and 1 representing fraud transactions. 

---

<h2><a class="anchor" id="dataset"></a>Dataset</h2>

The pipeline expects a structured format to locate files dynamically across different operating systems.
* **Source File:** CSV file stored in folder `/data/`(fraud_detection_model)
* **Features:** Contains numeric features resulting from a PCA transformation (V1 through V28), alongside `Time` and `Amount`.
* **Target Vector:** `Class`, where `0` represents a valid transaction and `1` represents a fraudulent transaction.

---

<h2><a class="anchor" id="tools--technologies"></a>Tools and Technologies</h2>

The infrastructure of this pipeline is built entirely in Python, utilizing industry-standard libraries for data science and statistical modeling:

* **Core Engine:** [Python 3.8+](https://python.org)
* **Data Manipulation:** [Pandas](https://pydata.org) & [NumPy](https://numpy.org)
* **Machine Learning Pipeline:** [Scikit-Learn](https://scikit-learn.org)
* **Imbalance & Resampling:** [Imbalanced-Learn (SMOTE)](https://imbalanced-learn.org)
* **Data Visualization:** [Matplotlib](https://matplotlib.org) & [Seaborn](https://pydata.org)
* **Environment Logic:** Built-in `os` system path utilities

---

<h2><a class="anchor" id="project-structure"></a>Project Structures</h2>

```text
fraud_detection_model/
│   .gitignore
│   README.md
│   requirements.txt
│
├───data
│       creditcard_data.csv
│
└───scripts
        fraud_detection_model.py
```

---

<h2><a class="anchor" id="data-cleaning-preparation"></a>Data Cleaning and Preparation</h2>

Your codebase executes explicit processing actions to safeguard model integrity:
* **Path Standardization:** Uses `os.path.abspath(__file__)` to eliminate manual pathing errors regardless of whether the script is run locally or in a remote production container.
* **Deterministic Splitting:** Utilizes a `train_test_split` with a fixed `random_state=42` to guarantee reproducible experimental runs.
* **Stratified Sampling:** Enforces `stratify=Y` during splitting. This preserves the exact minority-to-majority ratio in both training and test subsets, preventing data distribution shifts.
* **Synthetic Synthetic Over-sampling (SMOTE):** Dynamically generates artificial minority instances on the training subset using `sampling_strategy=0.1`. This steps the fraud class up to exactly $10\%$ of the majority class size before the data hits the classifier, preventing the model from ignoring fraud patterns.

---

<h2><a class="anchor" id="exploratory-data-analysis"></a>Exploatory Data Analysis</h2>

The script triggers automatic exploratory operations immediately upon data ingestion:
* **Statistical Descriptives:** Automatically prints distribution summaries (`.describe()`) and structural records (`.head()`) to map monetary spread.
* **Class Distribution Mapping:** Calculates the exact baseline dataset imbalance ratio ($ClassRatio = \frac{\text{Fraud}}{\text{Valid}}$) and logs total transaction volumes to the console terminal.
* **Feature Relationship Mapping:** Generates an interactive **Correlation Matrix Heatmap** scaled to a window size of $12 \times 9$ using Seaborn. This highlights hidden multicollinearity patterns among the anonymized vector fields ($V1 - V28$) and transaction values.

---

<h2><a class="anchor" id="research-questions-key-findings"></a>Research questions and Key findings</h2>

Upon termination, the model bypasses baseline accuracy scores (which are often misleadingly high in imbalanced data contexts) and reports premium tracking metrics:

1. **Matthews Correlation Coefficient (MCC):** Evaluates all four quadrants of the confusion matrix to measure the true stability of the binary classifier.
2. **Precision vs. Recall Balance:** Tracks the critical trade-off between false alarms (Precision) and true fraud catch rates (Recall).
3. **Confusion Matrix Visualization:** Renders a clean, blue-gradient matrix labeled explicitly with `Normal` and `Fraud` fields for fast visual auditing.
4. **Targeted Forensic Output:** Automatically isolates missed fraudulent attempts ($False\ Negatives$) where `Actual == 1` and `Predicted == 0`. It groups these records and prints them out sorted by their transaction `Amount` in ascending order, exposing where the system is financially vulnerable.

---

<h2><a class="anchor" id="how-to-run-this-project"></a>How to run this project</h2>

## Getting Started

Follow these steps to set up the environment, position the data file correctly, and execute the fraud detection model.

### Prerequisites
Make sure you have [Python 3.8+](https://python.org) installed on your machine.

### Setup & Installation

1. **Clone the project repository:**
   ```bash
   git clone https://github.com
   cd fraud_detection_model
   ```

2. **Create and activate a virtual environment (Recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install the required packages:**
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn
   ```

### Data Preparation
The script expects a specific folder structure to locate the dataset. Make sure your project directory looks exactly like this:

```text
fraud_detection_model/
├── data/
│   └── creditcard_data.csv   <-- Place your dataset here
└── src/
    └── fraud_detection_model.py   <-- Your python script file
```

### Running the Program
Navigate to the directory containing your script and run it using Python:

```bash
cd src
python fraud_detection_model.py
```

*Note: Running the program will output model performance metrics (Accuracy, Precision, Recall, F1-Score, MCC) directly to your terminal and display two visual plots (Correlation Heatmap and Confusion Matrix).*

---

<h2><a class="anchor" id="author-contact"></a>Author & Contact</h2>

- *Rachana Subramanya*
- email: rachanasubramanya50@gmail.com
- LinkedIn: https://www.linkedin.com/in/rachana-subramanya-4ab0b3303/
- Github: https://github.com/RachanaSubramanya

---