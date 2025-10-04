# exoplanet_classifier.py

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

def resolve_file_path(relative_path):
    """
    Construct and verify the absolute path of a file relative to this script.
    Raises FileNotFoundError if the file does not exist.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    abs_path = os.path.join(base_dir, relative_path)
    abs_path = os.path.normpath(abs_path)  # normalize ../ etc.

    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"[ERROR] File not found: {abs_path}")
    
    print(f"[INFO] Verified file path: {abs_path}")
    return abs_path

def load_koi_data(file_path):
    """
    Load KOI CSV file from NASA archive and return a pandas DataFrame.
    """
    print(f"[INFO] Loading KOI data from: {file_path}")
    df = pd.read_csv(file_path, comment='#')
    print(f"[INFO] Loaded {len(df)} rows and {len(df.columns)} columns")
    return df

def preprocess_data(df, label_mapping={'CONFIRMED': 1, 'CANDIDATE': 1, 'FALSE POSITIVE': 0}):
    """
    Preprocess the KOI data:
    - Map dispositions to binary labels
    - Fill missing values
    - Select relevant features
    """
    print("[INFO] Preprocessing data...")
    df['label'] = df['koi_disposition'].map(label_mapping)
    df = df.dropna(subset=['label'])

    features = [
        'koi_period', 'koi_duration', 'koi_depth', 'koi_prad',
        'koi_steff', 'koi_slogg', 'koi_srad',
        'koi_teq', 'koi_insol'
    ]
    
    X = df[features].fillna(df[features].median())
    y = df['label']
    
    print(f"[INFO] Selected {len(features)} features")
    print(f"[INFO] Dataset shape: X={X.shape}, y={y.shape}")
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split features and labels into training and test sets
    """
    print(f"[INFO] Splitting data: test_size={test_size}")
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_random_forest(X_train, y_train, n_estimators=100, random_state=42):
    """
    Train a Random Forest classifier on the training data
    """
    print(f"[INFO] Training Random Forest with {n_estimators} estimators...")
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        class_weight='balanced',
        verbose=1
    )
    clf.fit(X_train, y_train)
    print("[INFO] Training complete")
    return clf

def evaluate_model(clf, X_test, y_test):
    """
    Evaluate classifier performance and print metrics
    """
    print("[INFO] Evaluating model...")
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    print("Classification Report:\n", report)
    print("Confusion Matrix:\n", cm)
    return report, cm

def scale_features(X_train, X_test):
    """
    Standardize features (optional, useful for SVM/Logistic Regression)
    """
    print("[INFO] Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

# Main pipeline
def run_pipeline(relative_file_path):
    """
    Full pipeline: resolve path, load data, preprocess, train, evaluate
    """
    file_path = resolve_file_path(relative_file_path)
    df = load_koi_data(file_path)
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    clf = train_random_forest(X_train, y_train)
    report, cm = evaluate_model(clf, X_test, y_test)
    print("[INFO] Pipeline finished successfully")
    return clf, report, cm

if __name__ == "__main__":
    # Example usage:
    relative_file_path = "./data/cumulative_2025.10.04_01.59.40.csv"
    run_pipeline(relative_file_path)
