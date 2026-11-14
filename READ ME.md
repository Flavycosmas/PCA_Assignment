# BAN6420 – Milestone Assignment 2
## Principal Component Analysis (PCA) Using PySpark

### 1. Project Overview

This project applies Principal Component Analysis (PCA) using PySpark to analyze breast cancer referral data from Anderson Cancer Center.

The objective is to reduce the dimensionality of the dataset while preserving as much of its original information as possible. The analysis uses 30 numerical features and reduces them to two principal components.

An optional Logistic Regression model is also implemented to explore whether the two principal components can be used to predict the diagnosis category.

### 2. Project Objectives

The main objectives are to:

- Load the breast cancer referral dataset into a PySpark DataFrame using an explicit schema.
- Identify and remove duplicate records.
- Detect and handle missing numerical values using median imputation.
- Assemble numerical features into a single feature vector.
- Standardize numerical features before PCA.
- Reduce the 30 numerical features to two principal components.
- Calculate and report the explained variance of the principal components.
- Visualize the PCA results using a scatter plot.
- Apply optional Logistic Regression and evaluate its classification performance.

### 3. Dataset Description

**Dataset:** `breast_cancer_referrals.csv`

The dataset contains patient referral records with 32 columns:

| Column category | Description |
|---|---|
| Patient ID | Unique patient identifier |
| Numerical features | 30 numerical measurements |
| Diagnosis | Diagnosis category: B (Benign) or M (Malignant) |

The 30 numerical features are grouped into:

- Mean measurements
- Standard error measurements
- Worst measurements

The `patient_id` column is excluded from PCA because it is an identifier rather than a numerical measurement. The `diagnosis` column is retained for optional classification.

### 4. Technologies Used

- Python 3.13.5
- Apache Spark / PySpark 4.3.0
- Java 17
- Matplotlib
- PyCharm
- CSV dataset

### 5. Project Structure

```text
PCA_Assignment/
│
├── data/
│   └── breast_cancer_referrals.csv
│
├── outputs/
│   └── pca_scatter_plot.png
│
├── pca_analysis.py
├── README.md
└── .venv/
```

**Note:** The `.venv` directory contains the local Python virtual environment and does not need to be included in the submission ZIP.

### 6. Installation and Setup

#### Prerequisites

Ensure that Python, Java, and pip are installed.

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the required libraries:

```bash
python -m pip install pyspark==4.3.0 matplotlib
```

Verify the PySpark installation:

```bash
python -c "import pyspark; print(pyspark.__version__)"
```

### 7. How to Run the Project

1. Open the `PCA_Assignment` folder in PyCharm.
2. Ensure the virtual environment is activated.
3. Confirm that the CSV dataset is located in the `data` directory.
4. Open the terminal in the project root.
5. Execute:

```bash
python pca_analysis.py
```

The program performs data preparation, PCA, visualization, and optional Logistic Regression.

### 8. Data Preparation and Cleaning

The dataset initially contained 572 records and 32 columns.

The following cleaning operations were performed:

- Duplicate removal using PySpark's `dropDuplicates()` method.
- Missing-value detection across the numerical features.
- Median imputation to handle missing numerical values.
- Removal of records without a valid diagnosis category.

The verified cleaning results were:

| Processing step | Result |
|---|---:|
| Original records | 572 |
| Duplicate records removed | 3 |
| Records after duplicate removal | 569 |
| Missing numerical values before imputation | 18 |
| Missing numerical values after imputation | 0 |

### 9. Principal Component Analysis

PCA was performed using the 30 numerical features.

Before applying PCA, the numerical features were:

1. Combined using `VectorAssembler`.
2. Standardized using PySpark's `StandardScaler`.
3. Reduced to two principal components using PySpark's `PCA`.

#### Explained Variance Results

| Principal component | Explained variance |
|---|---:|
| PC1 | 44.23% |
| PC2 | 18.97% |
| **Total** | **63.20%** |

The two principal components retain 63.20% of the variance in the original standardized numerical features.

This demonstrates that the dataset can be represented in two dimensions while retaining a substantial proportion of its original variation.

### 10. PCA Visualization

The project generates a scatter plot using:

- Principal Component 1 on the X-axis.
- Principal Component 2 on the Y-axis.
- Diagnosis categories to distinguish benign and malignant observations.

The generated visualization is saved to:

```text
outputs/pca_scatter_plot.png
```

### 11. Optional Logistic Regression

An optional Logistic Regression model uses the two PCA components as input features to predict the diagnosis category.

The dataset is divided into:

- 80% training data.
- 20% testing data.

The model is evaluated using:

- Accuracy
- Weighted precision
- Weighted recall
- F1-score
- Confusion matrix

The actual evaluation metrics are generated when the Python script runs.

### 12. Conclusion

This project demonstrates how PySpark can be used to prepare, clean, standardize, and reduce the dimensionality of a dataset through Principal Component Analysis.

The analysis reduced 30 numerical features to two principal components, retaining 63.20% of the original variance.

The PCA visualization provides a two-dimensional representation of the observations, while the optional Logistic Regression exercise explores the predictive usefulness of the reduced features.

PCA is a dimensionality-reduction technique and should not, by itself, be treated as a clinical diagnosis or a donor-funding allocation decision.

### 13. Author and Submission

**Course:** BAN6420  
**Assignment:** Milestone Assignment 2  
**Project:** Principal Component Analysis Using PySpark

