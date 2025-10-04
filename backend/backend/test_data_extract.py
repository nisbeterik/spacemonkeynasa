# exoplanet_classifier.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

def load_koi_data(file_path):
    """
    Load KOI CSV file from NASA archive and return a pandas DataFrame.
    """
    df = pd.read_csv(file_path, comment='#')
    return df

def preprocess_data(df, label_mapping={'CONFIRMED': 1, 'CANDIDATE': 1, 'FALSE POSITIVE': 0}):
    """
    Preprocess the KOI data:
    - Map dispositions to binary labels
    - Fill missing values
    - Select relevant features
    """
    # Create binary label
    df['label'] = df['koi_disposition'].map(label_mapping)
    df = df.dropna(subset=['label'])
    
    # Select features
    features = [
        'koi_period', 'koi_duration', 'koi_depth', 'koi_prad',
        'koi_steff', 'koi_slogg', 'koi_srad',
        'koi_teq', 'koi_insol'
    ]
    
    X = df[features].fillna(df[features].median())
    y = df['label']
    
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split features and labels into training and test sets
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_random_forest(X_train, y_train, n_estimators=100, random_state=42):
    """
    Train a Random Forest classifier on the training data
    """
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, class_weight='balanced')
    clf.fit(X_train, y_train)
    return clf

def evaluate_model(clf, X_test, y_test):
    """
    Evaluate classifier performance and print metrics
    """
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
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

# Example main workflow (can be called from backend)
def run_pipeline(file_path):
    df = load_koi_data(file_path)
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    clf = train_random_forest(X_train, y_train)
    report, cm = evaluate_model(clf, X_test, y_test)
    return clf, report, cm

if __name__ == "__main__":
    # Example usage:
    file_path = "../data/cumulative_2025.10.04_01.59.40.csv"  # replace with your actual file path
    run_pipeline(file_path)
