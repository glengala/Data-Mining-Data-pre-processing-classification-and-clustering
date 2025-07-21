# -*- coding: utf-8 -*-

"""
###Part 1: Classification

####Subtask 1A. Data pre-processing
"""

import pandas as pd
import numpy as np
df = pd.read_csv('winequality-RED.csv')
print(df.shape)
#checking for missing value & replace with mean:

for column in df.columns:
  if df[column].isna().sum()>0:
    mean_value=df[column].mean()
    print(f'Column: {column}')
    print(f'_____Numer of missing values:{df[column].isna().sum()}')
    print(f'_____Mean value: {mean_value:.3f}')
    df[column]=np.where(df[column].isna(),mean_value,df[column])

#verify no missing values left
print(f'\nMissing value after replacement: \n{df.isna().any()}')

df.to_csv('newwinequality.csv', index=False)

"""####Subtask 1B. Wine Quality Prediction:

Using the data exported in subtask 1A, create a decision tree and neural network (NN) learner and perform 10-fold cross-validation to evaluate the performance of the classifier with these data.

* Subtask 1B.1. Building Decision Tree classifier:



"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, accuracy_score, roc_auc_score, confusion_matrix
import pandas as pd
import numpy as np

# Import new dataframe
new_df = pd.read_csv('newwinequality.csv')
print(f'\nNew Dataframe shape: {new_df.shape}')

# Extract features and labels
features = new_df.drop("quality", axis=1)
labels = new_df["quality"]
print(f'\nFeatures Dataframe shape: {features.shape}')
print(f'\nLabel Data shape: {labels.shape}')

# Initialize classifier
treeLearner = DecisionTreeClassifier(random_state=0)

# Custom scorer functions for confusion matrix components
def tn_scorer(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    return cm[0, 0]

def fp_scorer(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    return cm[0, 1]

def fn_scorer(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    return cm[1, 0]

def tp_scorer(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    return cm[1, 1]

# Create scorers using make_scorer
tn = make_scorer(tn_scorer)
fp = make_scorer(fp_scorer)
fn = make_scorer(fn_scorer)
tp = make_scorer(tp_scorer)

# Scoring dictionary
scoring = {
    "accuracy": "accuracy",
    "roc_auc": "roc_auc",
    "tn": tn,
    "fp": fp,
    "fn": fn,
    "tp": tp
}

# Perform cross-validation with custom scorers
evalResults = cross_validate(treeLearner, features, labels, cv=10, scoring=scoring, return_estimator=True)

# Calculate averages
average_accuracy = np.mean(evalResults['test_accuracy'])
average_roc_auc = np.mean(evalResults['test_roc_auc'])

# Show the confusion matrix for the estimator with the highest accuracy:
best_est_index = evalResults['test_accuracy'].argmax()

# Calculate confusion matrix for the best estimator
best_est_tn = evalResults['test_tn'][best_est_index]
best_est_fp = evalResults['test_fp'][best_est_index]
best_est_fn = evalResults['test_fn'][best_est_index]
best_est_tp = evalResults['test_tp'][best_est_index]
conf_matrix_best_est = (best_est_tn, best_est_fp, best_est_fn, best_est_tp)


# Output results
print(f'\n Cross Validation result:{evalResults}')
print(f'\nAverage Test Accuracy: {average_accuracy:.3f}')
print(f'\nAverage Test ROC AUC: {average_roc_auc:.3f}')
print(f'\nConfusion Matrix for Best Estimator: \nTN: {best_est_tn}, FP: {best_est_fp}, FN: {best_est_fn}, TP: {best_est_tp}')

"""* Subtask 1B.2. Building neural network (NN) classifier:"""

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score,make_scorer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd

# Initiate an empty list to capture all the results
results_ANN = []

# Custom scorer functions for confusion matrix components
def tn_scorer(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    return cm[0, 0]

def fp_scorer(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    return cm[0, 1]

def fn_scorer(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    return cm[1, 0]

def tp_scorer(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    return cm[1, 1]

# Define a function to pass different hyper-parameters to build and train model with 10-fold cross-validation:
def evaluate_ANN(features_normalized, label, hidden_layer_sizes=(100,), solver='adam', learning_rate_init=0.001, max_iter=200, random_state=0):
    print(f'______Neural Network training with hidden_layer_sizes= {hidden_layer_sizes}, solver={solver}, learning_rate_init= {learning_rate_init}, max_iter= {max_iter}, random_state={random_state}:_______')

    # Initialize MLP Classifier with different hyper-parameters
    mlp = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        solver=solver,
        learning_rate_init=learning_rate_init,
        max_iter=max_iter,
        random_state=random_state
    )
    # Create scorers using make_scorer
    tn = make_scorer(tn_scorer)
    fp = make_scorer(fp_scorer)
    fn = make_scorer(fn_scorer)
    tp = make_scorer(tp_scorer)

    # Scoring dictionary
    scoring = {
      "accuracy": "accuracy",
      "roc_auc": "roc_auc",
      "tn": tn,
      "fp": fp,
      "fn": fn,
      "tp": tp
    }
    eval_Results = cross_validate(mlp, X=features_normalized, y=label, cv=10, scoring=scoring, return_estimator=True)
    print(f'Cross Validation results: \n{eval_Results}')
    print(f'\nAverage Test Accuracy: {eval_Results["test_accuracy"].mean():.3f}')
    print(f'\nAverage Test ROC AUC: {eval_Results["test_roc_auc"].mean():.3f}')

    # Select the best estimator based on test accuracy
    best_estimator_index = eval_Results['test_accuracy'].argmax()
    #best_estimator = eval_Results['estimator'][best_estimator_index]

    # Calculate confusion matrix for the best estimator
    best_estimator_tn = eval_Results['test_tn'][best_estimator_index]
    best_estimator_fp = eval_Results['test_fp'][best_estimator_index]
    best_estimator_fn = eval_Results['test_fn'][best_estimator_index]
    best_estimator_tp = eval_Results['test_tp'][best_estimator_index]
    conf_matrix_best_estimator = (best_estimator_tn, best_estimator_fp, best_estimator_fn, best_estimator_tp)
    print(f'\nConfusion Matrix for Best Estimator: \nTN: {best_estimator_tn}, FP: {best_estimator_fp}, FN: {best_estimator_fn}, TP: {best_estimator_tp}')
    # Helper function to capture all the results for comparison
    record_results_ANN(eval_Results, hidden_layer_sizes, solver, learning_rate_init, max_iter, random_state, best_estimator_tp,best_estimator_fp,best_estimator_tn,best_estimator_fn)

# Define an function to add results to the list:
def record_results_ANN(evalResults, hidden_layer_sizes, solver, learning_rate_init, max_iter, random_state, tp,fp,tn,fn):
    results_ANN.append({
        'Hidden Layer Sizes': hidden_layer_sizes,
        'Solver': solver,
        'Learning Rate': learning_rate_init,
        'Max Iter': max_iter,
        'Random State': random_state,
        'Average Test Accuracy': evalResults["test_accuracy"].mean(),
        'Average Test ROC AUC': evalResults["test_roc_auc"].mean(),
        'TP': tp,
        'FP': fp,
        'TN': tn,
        'FN': fn
    })


# Define an ANN Evaluation function in order to pass different dataframe in:
def runANNEvaluation(df, label_column, random_state=0, **mlp_params):
    feature = df.drop(label_column, axis=1)
    label = df[label_column]
    print(f'\nFeatures Dataframe shape: {feature.shape}')
    print(f'\nLabel Data shape: {label.shape}')

    # Normalize features
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(feature)

    # Call the evaluation function with the normalized data
    evaluate_ANN(features_normalized, label, **mlp_params)


#. Import new dataframe
dfnew = pd.read_csv('newwinequality.csv')
print(f'\nNew Dataframe shape: {dfnew.shape}')

# Call the function with different hyper-parameters
# Model represented on the report
print("***Default hyper-parameters Model***")
runANNEvaluation(dfnew, label_column='quality')
print("***Different hyper-parameters Model***")
runANNEvaluation(dfnew,label_column='quality', hidden_layer_sizes=(500,50), solver='adam', learning_rate_init =0.001, max_iter = 200)
runANNEvaluation(dfnew,label_column='quality', hidden_layer_sizes=(50,), solver='adam', learning_rate_init =0.001, max_iter = 200)
runANNEvaluation(dfnew,label_column='quality', hidden_layer_sizes=(100,), solver='sgd', learning_rate_init =0.001, max_iter = 200)
runANNEvaluation(dfnew,label_column='quality', hidden_layer_sizes=(100,), solver='adam', learning_rate_init =0.01, max_iter = 200)
runANNEvaluation(dfnew,label_column='quality', hidden_layer_sizes=(100,), solver='adam', learning_rate_init =0.1, max_iter = 200)
runANNEvaluation(dfnew,label_column='quality', hidden_layer_sizes=(100,), solver='adam', learning_rate_init =0.001, max_iter = 600)
runANNEvaluation(dfnew,label_column='quality', hidden_layer_sizes=(100,), solver='adam', learning_rate_init =0.001, max_iter = 100)
runANNEvaluation(dfnew,label_column='quality', hidden_layer_sizes=(100,), solver='adam', learning_rate_init =0.01, max_iter = 300)

# Extra experience not included in report
runANNEvaluation(dfnew,label_column='quality', hidden_layer_sizes=(50,50), solver='adam', learning_rate_init =0.001, max_iter = 200)
runANNEvaluation(dfnew,label_column='quality', hidden_layer_sizes=(500,), solver='adam', learning_rate_init =0.001, max_iter = 200)
runANNEvaluation(dfnew,label_column='quality', hidden_layer_sizes=(50,5), solver='sgd', learning_rate_init =0.001, max_iter = 200)
runANNEvaluation(dfnew,label_column='quality', hidden_layer_sizes=(100,), solver='sgd', learning_rate_init =0.01, max_iter = 300)


# Print all the results to csv for comparison & download the result file:
df_results_ANN = pd.DataFrame(results_ANN)
df_results_ANN.to_csv('results_ANN.csv', index=False)
from google.colab import files

files.download('results_ANN.csv')

"""###Part 2: Clustering

####Subtask 2A & 2B: K-means Clustering and counting table
"""

from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Data Loading
df = pd.read_csv('ionoshpere.csv')
print(df.shape)
print(f'Check for missing value:\n{df.isna().any()}')

# Data splitting
features = df.drop("y", axis=1)
labels = df["y"]
print(f'\nFeatures Dataframe shape: {features.shape}')
print(f'\nLabel Data shape: {labels.shape}')


# Initialising the algorithm
kmeans_estimator = KMeans(n_clusters=2, random_state=0)
k_cluster = kmeans_estimator.fit(features)
cluster_labels = k_cluster.labels_
cluster_centers = k_cluster.cluster_centers_
print(cluster_labels)

# Sub task 2B: count the number of class label 'y' for each Cluster

results_df = df.copy()
results_df = pd.DataFrame ({'x axis':df['a33'], 'y axis': df['a34'], 'Label':labels, 'Cluster': cluster_labels})
results_df['Label'] = results_df['Label'].map({'g': 1, 'b': 0})
print(results_df.head())

x_axis = results_df['x axis'].to_numpy()
y_axis = results_df['y axis'].to_numpy()

# Define markers and colors for clusters and labels
markers = {0: '^', 1: 'o'}  # Cluster 0: triangle, Cluster 1: circle
colors = {0: 'green', 1: 'yellow'}  # Label 0: blue, Label 1: red

# Plot the data using different colors for each label and different shapes for clusters
plt.figure(figsize=(10, 6))

# Scatter plot for each cluster and label with outlines
for cluster in results_df['Cluster'].unique():
    for label in results_df['Label'].unique():
        subset = results_df[(results_df['Cluster'] == cluster) & (results_df['Label'] == label)]
        plt.scatter(subset['x axis'], subset['y axis'],
                    label=f'Cluster {cluster}, Label {"g" if label == 1 else "b"}',
                    marker=markers[cluster], color=colors[label], edgecolor='black',alpha =0.6)

# Plot cluster centers
plt.scatter(cluster_centers[:, 32], cluster_centers[:, 33], marker='x', color='blue', s=100, label='Centroids')

plt.xlabel('a33')
plt.ylabel('a34')
plt.title('Scatter Plot of Clusters and Labels')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# Count the instances
results_df['Label'] = results_df['Label'].map({1: 'g', 0: 'b'})
category_counts = results_df.groupby(['Cluster', 'Label']).size().unstack(fill_value=0)
print(category_counts)