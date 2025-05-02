# Santander Customer Satisfaction Project
* This project aims to predict customer satisfaction from the [Santander Customer Satisfaction](https://www.kaggle.com/competitions/santander-customer-satisfaction) Kaggle Challenge.

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
![Class Imbalance](Images%20(Data%20Visualization)/class%20imbalance.png)  
* **Observation:** The dataset is imbalanced — most customers are satisfied (TARGET = 0), and only a small portion are dissatisfied (TARGET = 1).  
**XGBoost Top Features**  
* The top features XG Boost used for interpreting the dataset.    
![Top 20 Features - XGBoost](Images%20(Data%20Visualization)/XGBoost_top_20_features.png)  

## Problem Formulation  
* Predicting Customer disatisfaction using binary target variable.  
* **Models:**  
  * **Logistic Regression**:  
    *  A baseline model for binary classification. Used because it is quick to train and simple. 
  * **Random Forest**:  
    * Used to handle many features, most of which are anonymized and possibly irrelevant.  
  * **XGBoosting**:  
    * Handles large data sets well and does not need outliers removed to perform well.   

## Optimizers and HyperParameters  
**Optimizer**  
* **XGBoost:** `binary:logistic` objective
  
**Hyperparameters**  
* **Logistic Regression:** `class_weight='balanced'` 
* **Random Forest:** `class_weight=None`,`max_depth=20`,`n_estimator=200`
* **XGBoosting:** `learning_rate=0.1`, `max_depth=3`, `n_estimator=50`

## Training
**Software & Hardware**
* **Software:** Python 3.x
* **Hardware:** Trained locally on a personal laptop
* **Training Duration:** Training did not take long for the models
* **Train Stopping:**
     * Logistic Regression: maximum of 3000 iterations.
     * XGBoost: 50 trees.
     * Random Forest: 200 trees.
     * None of the models had early stopping.
       
* **Problems**:
     * Over 90% of customers in the dataset were labeled as satisfied, making it difficult to accurately predict the minority class (unsatisfied).
     * I attempted hyperparameter tuning using GridSearchCV to improve model performance, but the imbalance still posed a major challenge.

## Performance Comparison
* **Logistic Regression**:
     * Resulted in moderate performance based on AUC score, but struggled the most with class imbalance.
*  **Random Forest**:
     * Resulted in a better performance than logistic regression, but was slower training.
     * AUC improved slightly, but sensitivity to the minority class (unsatisfied customers) was poor.
* **XGBoosting**:
     * Best AUC score among models.
     * Class imbalance still affected performance; most predictions leaned toward the majority class imbalance.
* AUC Score Comparison before Hyperparameter tuning:

