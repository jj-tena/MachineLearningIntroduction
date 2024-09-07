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

## Supervised Learning Notebook: Unsupervised Learning Techniques for Clustering and Dimensionality Reduction

### Use Case:
This notebook focuses on the application of unsupervised learning techniques, specifically clustering and dimensionality reduction, to the “Forest Cover Type Dataset.” The goal is to explore and analyze data without predefined labels, uncovering intrinsic patterns and structures within the dataset.

### Objective:
The primary objective is to delve into unsupervised learning methods to achieve the following:
Data Ingestion: Load and prepare the dataset for analysis.
Exploratory Data Analysis (EDA): Understand the dataset’s structure and statistics.
Data Cleaning and Preprocessing: Prepare data for analysis by handling outliers, missing values, feature selection, and standardization.
Clustering: Apply various clustering algorithms and evaluate the clustering results using internal validation metrics.
Dimensionality Reduction: Reduce the dataset’s dimensions for visualization purposes.

### Steps:

#### Data Ingestion:
Load the “Forest Cover Type Dataset” from the CSV file using Pandas.

#### Exploratory Data Analysis (EDA):
Investigate the dataset by analyzing its dimensions, variables, data types, unique values, descriptive statistics, missing values, outliers, and histograms.
Utilize a custom EDAModule to facilitate the exploratory analysis.

#### Data Cleaning and Preprocessing:
Outlier Detection and Removal:
Use Local Outlier Factor (LOF) and Isolation Forest algorithms to detect outliers.
Handling Missing Values:
Check for and address any missing values in the dataset.
Feature Selection:
Remove non-informative features, such as Soil_Type15 which contains only zeros.
Standardization:
Apply StandardScaler to standardize numerical variables.
Categorical Encoding:
Note that no additional encoding is necessary, as there are no categorical variables.

#### Clustering:
Partition-Based Methods:
Apply K-Means clustering and evaluate with Davies-Bouldin and Silhouette indices.
Hierarchical Methods:
Use Agglomerative Hierarchical Clustering and assess with Davies-Bouldin and Silhouette indices.
Density-Based Methods:
Apply DBSCAN, adjusting the epsilon parameter for clustering, although it often results in a single cluster for this dataset.

#### Dimensionality Reduction:
Principal Components Analysis (PCA):
Reduce data to 2D and 3D for visualization.
2D Visualization:
Create scatter plots to visualize clustering results from K-Means and Agglomerative Clustering.
3D Visualization:
Generate 3D scatter plots for the same clustering results using Plotly.

### Conclusion:
The notebook provides a comprehensive approach to exploring and analyzing the “Forest Cover Type Dataset” using unsupervised learning techniques. It includes data ingestion, exploratory analysis, data cleaning, clustering, and dimensionality reduction. The results from various clustering algorithms and their visualizations help in understanding the intrinsic structure of the data and evaluating the effectiveness of different clustering methods.
