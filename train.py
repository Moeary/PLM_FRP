import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, matthews_corrcoef, roc_auc_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Bidirectional, LSTM
from tensorflow.keras.callbacks import ModelCheckpoint
import argparse
from tensorflow.keras.utils import plot_model
import csv

# --- Data Loading and Preprocessing ---
def load_data(file_paths):
    """Load data from multiple CSV files and merge into one DataFrame."""
    dfs = [pd.read_csv(file_path, header=None) for file_path in file_paths]
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df.to_numpy()

def concatenate_features(feature1, feature2):
    """Concatenate two feature vectors."""
    return np.concatenate((feature1, feature2), axis=1)

def normalize_features(features):
    """Normalize feature vectors."""
    return (features - np.mean(features, axis=0)) / np.std(features, axis=0)

def calculate_specificity(y_true, y_pred):
    """Calculate specificity."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)

# --- Evaluation Metrics Calculation ---
def evaluate_model(y_true, y_pred):
    """Calculate evaluation metrics."""
    y_pred_classes = (y_pred > 0.5).astype(int)  # Convert predicted probabilities to class labels
    accuracy = accuracy_score(y_true, y_pred_classes)
    precision = precision_score(y_true, y_pred_classes)
    recall = recall_score(y_true, y_pred_classes)
    f1 = f1_score(y_true, y_pred_classes)
    confusion = confusion_matrix(y_true, y_pred_classes)
    mcc = matthews_corrcoef(y_true, y_pred_classes)
    specificity = calculate_specificity(y_true, y_pred_classes)
    auc = roc_auc_score(y_true, y_pred)
    return accuracy, auc, recall, specificity, mcc

# --- Build CNN Model ---
def build_cnn_model(input_shape):
    """Build CNN model."""
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(units=128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=1, activation='sigmoid'))  # Binary classification, use sigmoid activation function in output layer
    return model

# --- Build Other Classifiers ---
def build_svm_model():
    """Build SVM model."""
    return SVC(C=10, gamma='scale', decision_function_shape='ovr', kernel='linear', probability=True)

def build_rf_model():
    """Build Random Forest model."""
    return RandomForestClassifier(n_estimators=1000, max_depth=3, random_state=0, n_jobs=-1)

def build_nb_model():
    """Build Naive Bayes model."""
    return GaussianNB(priors=None, var_smoothing=1e-9)

def build_xgb_model():
    """Build XGBoost model."""
    return XGBClassifier(max_depth=3, learning_rate=0.2, n_estimators=500)

def build_bilstm_model(input_shape):
    """Build BiLSTM model."""
    model = Sequential()
    model.add(Bidirectional(LSTM(units=64, return_sequences=True), input_shape=input_shape))
    model.add(Bidirectional(LSTM(units=32)))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=1, activation='sigmoid'))  # Binary classification, use sigmoid activation function in output layer
    return model

# --- Main Function ---
def main(args):
    # Pre-read args parameters
    selected_features = args.features.split('+')

    # Define feature file paths
    feature_files = {
        'aac': {
            'positive_train': ["feature/AAC/positive_train.csv"],
            'negative_train': ["feature/AAC/negative_train.csv"],
            'positive_test': ["feature/AAC/positive_test.csv"],
            'negative_test': ["feature/AAC/negative_test.csv"]
        },
        'aadp': { 
            'positive_train': ["feature/AADP-PSSM/positive_train.csv"],
            'negative_train': ["feature/AADP-PSSM/negative_train.csv"],
            'positive_test': ["feature/AADP-PSSM/positive_test.csv"],
            'negative_test': ["feature/AADP-PSSM/negative_test.csv"]
        },
        'cksaap':{
            'positive_train': ["feature/CKSAAP/positive_train.csv"],
            'negative_train': ["feature/CKSAAP/negative_train.csv"],
            'positive_test': ["feature/CKSAAP/positive_test.csv"],
            'negative_test': ["feature/CKSAAP/negative_test.csv"]
        },
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
        'esm2':{
            'positive_train': ["feature/ESM2/positive_train.csv"],
            'negative_train': ["feature/ESM2/negative_train.csv"],
            'positive_test': ["feature/ESM2/positive_test.csv"],
            'negative_test': ["feature/ESM2/negative_test.csv"],
        },
        'gtpc': {
            'positive_train': ["feature/GTPC/positive_train.csv"],
            'negative_train': ["feature/GTPC/negative_train.csv"],
            'positive_test': ["feature/GTPC/positive_test.csv"],
            'negative_test': ["feature/GTPC/negative_test.csv"]
        },
        'kbigram': {
            'positive_train': ["feature/k-separated-bigram-PSSM/positive_train.csv"],
            'negative_train': ["feature/k-separated-bigram-PSSM/negative_train.csv"],
            'positive_test': ["feature/k-separated-bigram-PSSM/positive_test.csv"],
            'negative_test': ["feature/k-separated-bigram-PSSM/negative_test.csv"]
        },
        'sfpssm': {
            'positive_train': ["feature/S-FPSSM/positive_train.csv"],
            'negative_train': ["feature/S-FPSSM/negative_train.csv"],
            'positive_test': ["feature/S-FPSSM/positive_test.csv"],
            'negative_test': ["feature/S-FPSSM/negative_test.csv"]
        },
        'protbert': {
            'positive_train': ["feature/ProtBERT/positive_train.csv"],
            'negative_train': ["feature/ProtBERT/negative_train.csv"],
            'positive_test': ["feature/ProtBERT/positive_test.csv"],
            'negative_test': ["feature/ProtBERT/negative_test.csv"]
        },
    }

    # Load data
    X_train_positive = []
    X_train_negative = []
    X_test_positive = []
    X_test_negative = []

    for feature in selected_features:
        X_train_positive.append(load_data(feature_files[feature]['positive_train']))
        X_train_negative.append(load_data(feature_files[feature]['negative_train']))
        X_test_positive.append(load_data(feature_files[feature]['positive_test']))
        X_test_negative.append(load_data(feature_files[feature]['negative_test']))

    X_train_positive = np.hstack(X_train_positive)
    X_train_negative = np.hstack(X_train_negative)
    X_test_positive = np.hstack(X_test_positive)
    X_test_negative = np.hstack(X_test_negative)

    # Merge positive and negative samples
    X_train = np.concatenate((X_train_positive, X_train_negative), axis=0)
    X_test = np.concatenate((X_test_positive, X_test_negative), axis=0)

    # Create labels
    y_train = np.concatenate((np.ones(X_train_positive.shape[0]), np.zeros(X_train_negative.shape[0])))
    y_test = np.concatenate((np.ones(X_test_positive.shape[0]), np.zeros(X_test_negative.shape[0])))

    # Normalize feature vectors
    X_train = normalize_features(X_train)
    X_test = normalize_features(X_test)

    print(X_train.shape)
    print(X_test.shape)
    
    # --- Split training and validation sets ---
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # --- Select classifier ---
    if args.classifier == 'cnn':
        input_shape = (X_train.shape[1], 1)
        model = build_cnn_model(input_shape)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Save model
        checkpoint = ModelCheckpoint('cnn_model.h5', monitor='val_loss', save_best_only=True, mode='min')

        # Train model
        model.fit(X_train.reshape(-1, X_train.shape[1], 1), y_train, epochs=16, batch_size=32, validation_data=(X_val.reshape(-1, X_val.shape[1], 1), y_val), callbacks=[checkpoint])
        plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

        # Evaluate model
        y_pred = model.predict(X_test.reshape(-1, X_test.shape[1], 1))
    elif args.classifier == 'svm':
        model = build_svm_model()

        # Train model
        model.fit(X_train, y_train)

        # Evaluate model
        y_pred = model.predict_proba(X_test)[:, 1]
    elif args.classifier == 'rf':
        model = build_rf_model()

        # Train model
        model.fit(X_train, y_train)

        # Evaluate model
        y_pred = model.predict_proba(X_test)[:, 1]
    elif args.classifier == 'nb':
        model = build_nb_model()

        # Train model
        model.fit(X_train, y_train)

        # Evaluate model
        y_pred = model.predict_proba(X_test)[:, 1]
    elif args.classifier == 'xgb':
        model = build_xgb_model()

        # Train model
        model.fit(X_train, y_train)

        # Evaluate model
        y_pred = model.predict_proba(X_test)[:, 1]

        # Train model
        model.fit(X_train, y_train)

        # Evaluate model
        y_pred = model.predict_proba(X_test)[:, 1]
    elif args.classifier == 'bilstm':  # Add BiLSTM classifier
        input_shape = (X_train.shape[1], 1)
        model = build_bilstm_model(input_shape)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Save model
        checkpoint = ModelCheckpoint('bilstm_model.h5', monitor='val_loss', save_best_only=True, mode='min')

        # Train model
        model.fit(X_train.reshape(-1, X_train.shape[1], 1), y_train, epochs=8, batch_size=32, validation_data=(X_val.reshape(-1, X_val.shape[1], 1), y_val), callbacks=[checkpoint])

        # Evaluate model
        y_pred = model.predict(X_test.reshape(-1, X_test.shape[1], 1))

    # Calculate evaluation metrics
    accuracy, auc, recall, specificity, mcc = evaluate_model(y_test, y_pred)

    # Print evaluation results
    print(f"Accuracy: {accuracy}")
    print(f"AUC: {auc}")
    print(f"Sensitivity (Recall): {recall}")
    print(f"Specificity: {specificity}")
    print(f"MCC: {mcc}")

    # save results to CSV
    results = {
        'Features': args.features,
        'Classifier': args.classifier,
        'Accuracy': accuracy,
        'Precision': specificity,
        'Recall': recall,
        'MCC': mcc,
    }

    with open('evaluation_results.csv', 'a', newline='') as csvfile:
        fieldnames = ['Features', 'Classifier', 'Accuracy', 'Precision', 'Recall', 'MCC']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if csvfile.tell() == 0:
            writer.writeheader()

        writer.writerow(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature selection and classifier choice")
    parser.add_argument('--features', type=str, required=True, help="Select feature combination, separated by '+': 'aac','aadp','cksaap','dde','esm1b','esm2','gtpc','kbigram','sfpssm','protbert'")
    parser.add_argument('--classifier', type=str, choices=['cnn', 'svm', 'rf', 'nb', 'xgb', 'bilstm'], required=True, help="Select classifier: 'cnn', 'svm', 'rf', 'nb', 'xgb', 'bilstm'")
    args = parser.parse_args()
    main(args)
    # Example usage: python train.py --features dde --classifier svm