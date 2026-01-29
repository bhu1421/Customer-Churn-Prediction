## Customer-Churn-Prediction

### 📌 Project Overview

This project focuses on **predicting customer churn** using machine learning and simple deep learning models. Customer churn refers to customers who stop using a company’s service or product. Predicting churn helps businesses take proactive steps to retain customers.

The project includes:

* Data preprocessing and exploratory data analysis (EDA)
* Multiple ML classification models
* A simple Artificial Neural Network (ANN) model
* Model evaluation and comparison

---

### 📂 Project Structure

```
Customer-Churn-Prediction-main/
│
├── customer_churn (1).xls                # Dataset file
├── customer_churn_classfication.ipynb   # Jupyter Notebook with full workflow
└── README.md                            # Project documentation (this file)
```

---

### 📊 Dataset Description

The dataset contains customer-related features such as:

* Demographic details (e.g., gender, age)
* Service usage details
* Billing information
* Target variable: **Churn** (Yes/No)

> File format: `.xls` (Excel)

---

### 🧠 Models Used

The notebook implements and compares multiple models:

#### Machine Learning Models

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier
* Naive Bayes (GaussianNB)

#### Deep Learning Model

* Artificial Neural Network (ANN) using Keras Sequential API

---

### ⚙️ Workflow Steps

1. **Data Loading**

   * Load dataset using Pandas

2. **Data Preprocessing**

   * Handle missing values
   * Encode categorical variables
   * Feature scaling using MinMaxScaler

3. **Exploratory Data Analysis (EDA)**

   * Visualization using Matplotlib and Seaborn
   * Correlation analysis

4. **Train-Test Split**

   * Split dataset into training and testing sets

5. **Model Training**

   * Train ML models and ANN

6. **Evaluation**

   * Accuracy score
   * Confusion matrix
   * Model comparison

---

### 🧪 Evaluation Metrics

* Accuracy Score
* Confusion Matrix
* Model performance comparison

---

### 🛠️ Technologies & Libraries Used

* Python 3.x
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* TensorFlow / Keras

---

### ▶️ How to Run the Project

1. Clone the repository:

   ```bash
   git clone <repo-url>
   ```
2. Install dependencies:

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
   ```
3. Open the notebook:

   ```bash
   jupyter notebook customer_churn_classfication.ipynb
   ```
4. Run all cells sequentially.

---

### 📈 Results

* The Random Forest and ANN models generally provide better accuracy compared to basic classifiers.
* Visualization helps identify key churn-driving factors.

---

### 🚀 Future Improvements

* Hyperparameter tuning (GridSearchCV)
* Cross-validation
* Feature engineering
* Deployment using Streamlit or Flask
* Model interpretability (SHAP/LIME)

---

### Author:  **Bhuvan Patil** 

---

###
