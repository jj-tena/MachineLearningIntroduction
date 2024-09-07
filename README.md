# MachineLearningSupervisedIntroduction
This notebook explores several supervised machine learning techniques for training artificial intelligence models.

## Supervised Learning Notebook: Cardiovascular Disease Risk Prediction

### Use Case:
The use case for this notebook is to predict the risk of cardiovascular diseases in patients based on various health metrics. The goal is to create a predictive model that can estimate the likelihood of cardiovascular issues from a dataset, providing a preliminary assessment that can help reduce the need for more resource-intensive tests.

### Objective:
The objective is to develop and evaluate a predictive model for cardiovascular disease risk using the "Cardiovascular Diseases Risk Prediction Dataset." This involves data ingestion, exploratory analysis, preprocessing, feature selection, model training, and evaluation. The final goal is to identify the most effective model for predicting cardiovascular risk and provide a streamlined tool for preliminary risk assessment.

### Steps:

#### Data Ingestion and Exploration:
Download the dataset from Google Drive and load it into a Pandas DataFrame.
Use a custom module, EDAModule, for exploratory data analysis (EDA). This includes examining the dataset's dimensions, variable types, statistical summaries, missing values, outliers, unique values, histograms, correlation matrices, and scatter plots.

#### Data Cleaning and Preprocessing:
Load and preprocess the dataset using the DSWorkflows library. This involves separating features and the target variable, addressing class imbalance using undersampling, and encoding the target variable for model compatibility.
Assess the balance of the target variable and apply undersampling to address class imbalance if necessary.

#### Feature Selection:
Apply feature selection techniques using SelectKBest with f_classif for numerical features and chi2 for categorical features. This helps identify the most important features for predicting cardiovascular disease risk.
Visualize the importance scores of features to understand which variables are most influential.

#### Algorithm Selection and Training:
Define and implement various machine learning pipelines with different algorithms: GaussianNB, KNeighborsClassifier, LinearSVC, RandomForestClassifier, and LogisticRegression.
Train and evaluate these models using cross-validation to determine their effectiveness.
Optimize the RandomForestClassifier using GridSearchCV to find the best hyperparameters and re-evaluate the model with these parameters.

#### Model Evaluation:
Compare the performance of different classifiers using metrics such as F1 score, accuracy, precision, recall, and the area under the ROC curve (AUC).
Visualize the ROC curves for each classifier to compare their performance.

#### Results Presentation and Deployment:
Save the best-performing model using joblib for future use.
Implement a front-end form in VueJS that allows users to input data and receive risk predictions from the model.
Provide links to the front-end and back-end code repositories for further review and deployment.

### Conclusion:
The RandomForestClassifier, with optimized hyperparameters, achieved the best performance for predicting cardiovascular disease risk. Although improvements were modest after optimization, the RandomForest model consistently outperformed other algorithms based on F1 score and AUC. Future enhancements could involve using ensemble methods or deep learning techniques for potentially better performance. The project provided valuable insights into model building, feature selection, and the application of machine learning techniques in healthcare.
