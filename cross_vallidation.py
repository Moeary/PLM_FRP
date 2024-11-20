import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, confusion_matrix, matthews_corrcoef
from xgboost import XGBClassifier

def calculate_specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)

# Load training data
data_train = pd.read_csv('./best_features/X_train_216.csv', header=None).to_numpy()

# The first 919 rows are positive samples, the last 919 rows are negative samples
X_train = data_train
y_train = np.concatenate((np.ones(919), np.zeros(919)))

# Load test data
data_test = pd.read_csv('./best_features/X_test_216.csv', header=None).to_numpy()

# The first 230 rows are positive samples, the last 230 rows are negative samples
X_test = data_test
y_test = np.concatenate((np.ones(230), np.zeros(230)))

# Define model
model = XGBClassifier(max_depth=3, learning_rate=0.2, n_estimators=500, 
min_child_weight=1, max_delta_step=0, subsample=0.8,
colsample_bytree=0.8, reg_alpha=0, reg_lambda=0.4,
scale_pos_weight=0.8, objective='binary:logistic',
eval_metric='auc', seed=1440, gamma=0)

# 10-fold cross-validation
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='accuracy')
print("Cross-validation scores:", scores)
print("Mean cross-validation score:", scores.mean())

# Evaluate on test set
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)
sensitivity = recall_score(y_test, y_pred)  # sensitivity = recall
specificity = calculate_specificity(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)

print(f"Test Accuracy: {accuracy}")
print(f"Test AUC: {auc}")
print(f"Test Sensitivity (Recall): {sensitivity}")
print(f"Test Specificity: {specificity}")
print(f"Test MCC: {mcc}")