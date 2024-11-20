import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ndcg_score, roc_auc_score, matthews_corrcoef
from xgboost import XGBClassifier
import argparse
import xgboost as xgb

# --- Data Loading and Preprocessing ---
def load_data(file_paths):
    dfs = [pd.read_csv(file_path, header=None) for file_path in file_paths]
    return pd.concat(dfs, ignore_index=True).to_numpy()

def normalize_features(features):
    return (features - np.mean(features, axis=0)) / np.std(features, axis=0)

def calculate_specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) != 0 else 0

def main(args):
    feature_files = {
        'dde': {
            'positive_train': ["feature/DDE/positive_train.csv"],
            'negative_train': ["feature/DDE/negative_train.csv"],
            'positive_test': ["feature/DDE/positive_test.csv"],
            'negative_test': ["feature/DDE/negative_test.csv"]
        },
        'esm1b': {
            'positive_train': ["feature/ESM1b/positive_train.csv"],
            'negative_train': ["feature/ESM1b/negative_train.csv"],
            'positive_test': ["feature/ESM1b/positive_test.csv"],
            'negative_test': ["feature/ESM1b/negative_test.csv"]
        },
    }

    # Load DDE and ESM1b features
    X_train_pos_dde = load_data(feature_files['dde']['positive_train'])
    X_train_neg_dde = load_data(feature_files['dde']['negative_train'])
    X_test_pos_dde = load_data(feature_files['dde']['positive_test'])
    X_test_neg_dde = load_data(feature_files['dde']['negative_test'])

    X_train_pos_esm1b = load_data(feature_files['esm1b']['positive_train'])
    X_train_neg_esm1b = load_data(feature_files['esm1b']['negative_train'])
    X_test_pos_esm1b = load_data(feature_files['esm1b']['positive_test'])
    X_test_neg_esm1b = load_data(feature_files['esm1b']['negative_test'])

    # Merge features and labels
    X_train = np.concatenate((np.hstack([X_train_pos_dde, X_train_pos_esm1b]), 
                             np.hstack([X_train_neg_dde, X_train_neg_esm1b])), axis=0)
    X_test = np.concatenate((np.hstack([X_test_pos_dde, X_test_pos_esm1b]),
                            np.hstack([X_test_neg_dde, X_test_neg_esm1b])), axis=0)

    # Create group IDs for ranking
    n_train_samples = X_train.shape[0]
    n_test_samples = X_test.shape[0]
    train_group_size = 10
    test_group_size = 5
    train_groups = [train_group_size] * (n_train_samples // train_group_size)
    test_groups = [test_group_size] * (n_test_samples // test_group_size)
    if n_train_samples % train_group_size != 0:
        train_groups.append(n_train_samples % train_group_size)
    if n_test_samples % test_group_size != 0:
        test_groups.append(n_test_samples % test_group_size)

    # Create ranking labels
    y_train_ranked = np.array([i for size in train_groups for i in range(size, 0, -1)])
    y_test_ranked = np.array([i for size in test_groups for i in range(size, 0, -1)])

    # Normalize features
    X_train = normalize_features(X_train)
    X_test = normalize_features(X_test)

    # XGBoost ranking model
    dtrain = xgb.DMatrix(X_train, label=y_train_ranked)
    dtrain.set_group(train_groups)
    dtest = xgb.DMatrix(X_test, label=y_test_ranked)
    dtest.set_group(test_groups)

    params = {
        'objective': 'rank:pairwise',
        'eta': 0.1,
        'max_depth': 5,
        'eval_metric': 'ndcg@5'
    }

    model = xgb.train(params, dtrain, num_boost_round=100)
    y_pred_ranked = model.predict(dtest)

    # Create binary labels
    y_train_binary = np.concatenate((np.ones(X_train_pos_dde.shape[0]), np.zeros(X_train_neg_dde.shape[0])))
    y_test_binary = np.concatenate((np.ones(X_test_pos_dde.shape[0]), np.zeros(X_test_neg_dde.shape[0])))

    # Feature importance
    feature_importances = model.get_score(importance_type='weight')
    sorted_feature_indices = sorted(feature_importances, key=feature_importances.get, reverse=True)

    best_accuracy = 0
    best_num_features = 0
    best_feature_indices = []

    for num_features in range(1, min(X_train.shape[1], 1680) + 1):
        selected_features = sorted_feature_indices[:num_features]
        X_train_selected = X_train[:, selected_features]
        X_test_selected = X_test[:, selected_features]

        model_binary = XGBClassifier(max_depth=3, learning_rate=0.2, n_estimators=500,
                                     min_child_weight=1, max_delta_step=0, subsample=0.8,
                                     colsample_bytree=0.8, reg_alpha=0, reg_lambda=0.4,
                                     scale_pos_weight=0.8, objective='binary:logistic',
                                     eval_metric='auc', seed=1440, gamma=0)

        model_binary.fit(X_train_selected, y_train_binary)
        y_pred_binary = model_binary.predict(X_test_selected)
        y_pred_proba = model_binary.predict_proba(X_test_selected)[:, 1]

        accuracy = accuracy_score(y_test_binary, y_pred_binary)
        auc = roc_auc_score(y_test_binary, y_pred_proba)
        sensitivity = recall_score(y_test_binary, y_pred_binary)
        specificity = calculate_specificity(y_test_binary, y_pred_binary)
        mcc = matthews_corrcoef(y_test_binary, y_pred_binary)

        print(f"Num Features: {num_features}, Acc: {accuracy:.4f}, AUC: {auc:.4f}, Sn: {sensitivity:.4f}, Sp: {specificity:.4f}, MCC: {mcc:.4f}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_num_features = num_features
            best_feature_indices = selected_features
            print(f"NEW !! Best Num Features: {best_num_features}, Best Acc: {best_accuracy:.4f}")

    # Evaluate best model
    X_train_best = X_train[:, best_feature_indices]
    X_test_best = X_test[:, best_feature_indices]

    model_best = XGBClassifier(max_depth=3, learning_rate=0.2, n_estimators=500,
                               min_child_weight=1, max_delta_step=0, subsample=0.8,
                               colsample_bytree=0.8, reg_alpha=0, reg_lambda=0.4,
                               scale_pos_weight=0.8, objective='binary:logistic',
                               eval_metric='auc', seed=1440, gamma=0)

    model_best.fit(X_train_best, y_train_binary)
    y_pred_binary = model_best.predict(X_test_best)
    y_pred_proba = model_best.predict_proba(X_test_best)[:, 1]

    accuracy = accuracy_score(y_test_binary, y_pred_binary)
    auc = roc_auc_score(y_test_binary, y_pred_proba)
    sensitivity = recall_score(y_test_binary, y_pred_binary)
    specificity = calculate_specificity(y_test_binary, y_pred_binary)
    mcc = matthews_corrcoef(y_test_binary, y_pred_binary)

    print(f"Best Num Features: {best_num_features}, Best Acc: {accuracy:.4f}, Best AUC: {auc:.4f}, Best Sn: {sensitivity:.4f}, Best Sp: {specificity:.4f}, Best MCC: {mcc:.4f}")

    # Evaluate using NDCG
    ndcg = ndcg_score(y_test_ranked.reshape(1, -1), y_pred_ranked.reshape(1, -1), k=5)
    print(f"NDCG@5: {ndcg}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature selection and classifier choice")
    args = parser.parse_args()
    main(args)