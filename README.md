# Santander Customer Satisfaction Project
* This project attempts to predict customer satisfaction from the [Santander Customer Satisfaction](https://www.kaggle.com/competitions/santander-customer-satisfaction) Kaggle Challenge.

## Overview  
* The goal is to predict the probability that a customer is unsatisfied, using the area under the ROC curve (AUC) as the evaluation metric.
* **My Approach:**
  * This project formulates the problem as a binary classification task. I started by applying Logistic Regression as a baseline model.To improve performance, more advanced models such as XGBoost and Random Forest.  
* The best model was able to predict Customer di  

## Summary of Workdone
**Data**
* A CSV file containing anonymized features.
* **Target**: A binary classification indicating customer satisfaction. TARGET column (1 for dissatisfied, 0 for satisfied).    
* **Size**  
  * **Data:** 76020 rows, 371 columns  
* **Instances (Train,Test Split):** 
  * 80% of the data (60723 samples) for training), 20% of data (15181)used for testing.
  * `GridSearchCV` was for hyperparameter tuning.  

## Preprocessing / Clean up  
**1. Handling Missing Data**   
* Columns with missing values were replaced with the mode, since missing values were categorized as -99999.

**2. Feature Selection/Scaling**  
* Dropped irrelavant ID column.    
* Used `StandardScaler` to scale numeric features.
  
**3. Encoding**
* Converted categorical features into numeric using One-hot Encoding.

## Data Visualization
**Class Balance**
