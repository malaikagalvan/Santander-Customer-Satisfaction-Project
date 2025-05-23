# Santander Customer Satisfaction Project
* This project aims to predict customer satisfaction from the [Santander Customer Satisfaction](https://www.kaggle.com/competitions/santander-customer-satisfaction) Kaggle Challenge.

## Overview  
* The goal is to predict the probability that a customer is unsatisfied, using the area under the ROC curve (AUC) as the evaluation metric. This project formulates the problem as a binary classification task. A Logistic Regression was applied as a baseline model. To improve performance, more advanced models such as XGBoost and Random Forest were used. The best model was able to predict customer dissatisfaction 67% of the time. Currently the best performance on Kaggle is  82%.

## Summary of Work Done
**Data**
* A train CSV file containing anonymized features.
* **Target**: A binary classification indicating customer satisfaction. TARGET column (1 for dissatisfied, 0 for satisfied).    
* **Size**  
  * **Data:** 76020 rows, 371 columns  
* **Train, Test Split & Cross Validation:** 
  * 80% of the data (60723 samples) for training), 20% of data (15181 samples) used for testing.
  * `GridSearchCV` was used for hyperparameter tuning to find the best model.  

## Preprocessing / Clean up  
**1. Handling Missing Data**   
* Columns with missing values were replaced with the mode, since missing values were categorized as -99999.

**2. Feature Selection/Scaling**  
* Dropped irrelevant ID column.    
* Used `StandardScaler` to scale numeric features.
  
**3. Encoding**
* Converted categorical features into numeric using One-Hot Encoding.

## Data Visualization
**Class Balance**   
* The dataset is imbalanced — most customers are satisfied (TARGET = 0), and only a small portion are dissatisfied (TARGET = 1).

**Categorical Features**
* Table to better represent categorical features:  
![Categorical Features Table](https://github.com/malaikagalvan/Santander-Customer-Satisfaction-Project/blob/main/Images/Categorical-columns-table.png)  
* **Observation:**  
  * Certain features such as num_var1 are dominated by a single values (0.0 appears in over 98% of cases). This suggests that some features have low variability and may not carry strong predictive power.
  * Although certain columns include the name 'num' this does not mean the data is truly numeric in nature. For this project, categorical features where selected based on number of unique values (3-9).     

**XGBoost Top Features**  
* The top features XGBoost used for interpreting the dataset.    
![Top Features XGBoost](https://github.com/malaikagalvan/Santander-Customer-Satisfaction-Project/blob/main/Images/XGBoost%20-top-20-features.png)  
* **Observation:** var15 was also a top feature for Logistic Regression and Random Forest.

**var15 Histogram**  
* Distribution for var 15 (age):  
![var15 Histogram](https://github.com/malaikagalvan/Santander-Customer-Satisfaction-Project/blob/main/Images/var15-hist.png)     
* **Observation:** The majority of satisfied customers have lower var15 values, especially clustered around 20-30. Unsatisfied customers are also concentrated in the same area but show more spread across higher values. 


## Problem Formulation  
* Predicting Customer dissatisfaction using binary target variable.  
* **Models:**  
  * **Logistic Regression**:  
    *  A baseline model for binary classification. Used because it is quick to train and simple. 
  * **Random Forest**:  
    * Used to handle many features, most of which are anonymized and possibly irrelevant.  
  * **XGBoosting**:  
    * Handles large data sets well and does not need outliers removed to perform well.   

## Optimizers and Hyperparameters  
**Optimizer**  
* **XGBoost:** `binary:logistic` objective
  
**Hyperparameters**  
* **Logistic Regression:** `class_weight='balanced'` 
* **Random Forest:** `class_weight=None`,`max_depth=20`,`n_estimator=200`
* **XGBoosting:** `learning_rate=0.1`, `max_depth=3`, `n_estimator=50`

## Training  
* **Software:** Python 3.x
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
* AUC Score Comparison **before Hyperparameter tuning**:  
 ![AUC Comparison Before](https://github.com/malaikagalvan/Santander-Customer-Satisfaction-Project/blob/main/Images/ROC-before-tuning.png)
* AUC Score **after Hyperparameter tuning** (XGBoost & Random Forest)  
![AUC Comparison After](https://github.com/malaikagalvan/Santander-Customer-Satisfaction-Project/blob/main/Images/ROC-after-tuning.png)

## Conclusion
* **XGBoost outperformed** both Logistic Regression and Random Forest in terms of AUC score, making it the most effective model for identifying dissatisfied customers.

## Future Work
* **Handle Class Imbalance More Effectively:**
    * Although using GridSearch helped somewhat, class imbalance remains the key challenge. In the future, exploring techniques like SMOTE to create a more balanced dataset may help with this.

## Using this Project for Your Own Study
1. **Prepare the data**
   * Format your data as a pandas DataFrame. Ensure that your dataset has features (input variables) and a target column (output variable) that the model will predict.
2. **Modify Configuration**
    * Adjust the code, hyperparameters, (e.g., in train_model.py or model_training.ipynb) to reflect your dataset's structure and target variable.
3. **Train the Model**
    * Use the provided training csv to train models on your data.
4. **Evaluate the Model**
    * Use the provided evaluation metrics (e.g., AUC, accuracy, confusion matrix) to assess your model’s performance.
5. **Resources**
    * For running and training the models Google Colab was used, a free cloud-based environment.
  

## Overview of Files in Repository:
* **Data-Visualization-Cleaning.ipynb**: Data analysis and visualization. Cleaning train.csv file from Kaggle project
* **Models_Before_Tuning.ipynb**: Contains all models used on a clean the cleaned trained csv file. Includes GridSearchCV.
* **Updated_Models_with_Tuning.ipynb**: Includes retrained XGBoosting and Random Forest models based on best parameters from GridSearchCV.
*  **Images**: Contains all images used in this README.md file.

## Required Packages
* **pandas, numpy, xgboost, scikit-learn, matplotlib**

## Kaggle Data
* The Santander Customer Satisfaction datasets can be downloaded on Kaggle [right here](https://www.kaggle.com/competitions/santander-customer-satisfaction/data).

 ## Training/Evaluation Performance Guide
 * Training the model involves selecting a model (e.g., Logistic Regression, Random Forest, XGBoost), training it on the training data, and optionally tuning hyperparameters for better performance.
 * Evaluating performance includes using AUC to assess how well the model predicts the target variable.
