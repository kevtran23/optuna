import pandas as pd
import numpy as np
import torch
import random

from pathlib import Path
from tqdm import tqdm
from shutil import copyfile
from collections import defaultdict
from sklearn.metrics.ranking import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from statistics import mean 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier





val_data = pd.read_csv(Path('/deep/group/RareXpert/kevin/all_cam_val.csv'))
val_data = torch.tensor(val_data.values)
val_features = val_data[:,1:val_data.shape[1] -1]
val_labels = pd.read_csv(Path('/deep/group/RareXpert/ground_truth_val.csv'))
val_labels = torch.tensor(val_labels.values)[:,1]


test_data = pd.read_csv(Path('/deep/group/RareXpert/kevin/all_cam_test.csv'))
test_data = torch.tensor(test_data.values)
test_features = test_data[:,1:test_data.shape[1] -1]
test_labels = pd.read_csv(Path('/deep/group/RareXpert/ground_truth_test.csv'))
test_labels = torch.tensor(test_labels.values)[:,1]


# Logistic Regression
clf = LogisticRegression(random_state=0)
clf = clf.fit(val_features, val_labels)
val_pred = clf.predict(val_features)
val_auc = roc_auc_score(val_labels, val_pred)
print(val_auc)

test_pred = clf.predict(test_features)
test_auc = roc_auc_score(test_labels, test_pred)
print(test_auc)

# Decision Tree 
clf = DecisionTreeClassifier(max_depth = 2)

cv_results = cross_validate(clf, val_features , y=val_labels, cv=3, scoring='roc_auc',return_estimator=True)
estimators = list(cv_results['estimator'])

val_predictions = []
for i in range(len(estimators)):
  val_predictions.append(torch.from_numpy(estimators[i].predict(val_features)))
val_pred = torch.mean(torch.stack(val_predictions).float(),axis=0)
val_auc = roc_auc_score(val_labels, val_pred)
print(val_auc)

test_predictions = []
for i in range(len(estimators)):
  test_predictions.append(torch.from_numpy(estimators[i].predict(test_features)))
test_pred = torch.mean(torch.stack(test_predictions).float(),axis=0)
test_auc = roc_auc_score(test_labels, test_pred)
print(test_auc)


# Random Forest 
clf = RandomForestClassifier(n_estimators = 100, max_depth=2)
cv_results = cross_validate(clf, val_features , y=val_labels, cv=3, scoring='roc_auc',return_estimator=True)
estimators = list(cv_results['estimator'])

val_predictions = []
for i in range(len(estimators)):
  val_predictions.append(torch.from_numpy(estimators[i].predict(val_features)))
val_pred = torch.mean(torch.stack(val_predictions).float(),axis=0)
val_auc = roc_auc_score(val_labels, val_pred)
print(val_auc)

test_predictions = []
for i in range(len(estimators)):
  test_predictions.append(torch.from_numpy(estimators[i].predict(test_features)))
test_pred = torch.mean(torch.stack(test_predictions).float(),axis=0)
test_auc = roc_auc_score(test_labels, test_pred)
print(test_auc)

# Neural Network 
classifier = MLPClassifier(hidden_layer_sizes=(5),max_iter=10000,activation = 'relu',solver='adam',random_state=1, learning_rate='adaptive')
classifier.fit(val_features, val_labels)
y_pred = classifier.predict(val_features)
test_auc = roc_auc_score(val_labels, y_pred)
print(test_auc)
y_pred = classifier.predict(test_features)
test_auc = roc_auc_score(test_labels, y_pred)
test_auc = roc_auc_score(test_labels, y_pred)
print(test_auc)




