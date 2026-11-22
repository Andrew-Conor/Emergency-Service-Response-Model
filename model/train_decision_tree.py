import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = (
    PROJECT_ROOT / "csvs" / "Response-Time-Noise-Removed-Fire-and-EMS-Response-Data.csv"
)

def load_data():
    balanced_data = pd.read_csv(INPUT_PATH)

    return balanced_data

def splitdataset(balanced_data):

    X = balanced_data.drop(columns=["outcome", "incident_response_time"])
    Y = balanced_data["outcome"].astype(int).values

    cat_cols = [
        "call_type_group",
        "call_type",
        "original_priority",
        "unit_type",
        "station_area",
        "battalion",
        "neighborhoods_analysis_boundaries",
    ]

    num_cols = ["distance_km"]

    for col in cat_cols:
        X[col] = X[col].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

    onehot_encoder = OneHotEncoder(sparse_output=False, drop=None, handle_unknown="ignore")

    X_train_cat = onehot_encoder.fit_transform(X_train[cat_cols])
    X_test_cat = onehot_encoder.transform(X_test[cat_cols])

    cat_feature_names = onehot_encoder.get_feature_names_out(cat_cols)

    #Recombine encoded categorical columns with numeric 'distance_km'
    X_train_final = pd.concat([X_train[num_cols].reset_index(drop=True), pd.DataFrame(X_train_cat, columns=cat_feature_names)], axis=1,)
    X_test_final = pd.concat([X_test[num_cols].reset_index(drop=True), pd.DataFrame(X_test_cat, columns=cat_feature_names)], axis=1)

    return X_train_final, X_test_final, y_train, y_test, list(X_train_final.columns)


def train_using_gini(X_train, y_train):
    clf_gini = DecisionTreeClassifier(criterion="gini", random_state=42, max_depth=10, min_samples_leaf=50)
    clf_gini.fit(X_train, y_train)
    return clf_gini

def train_using_entropy(X_train, y_train):
    clf_entropy = DecisionTreeClassifier(criterion="entropy", random_state=42, max_depth=3, min_samples_leaf=5)
    clf_entropy.fit(X_train, y_train)
    return clf_entropy

def prediction(X_test, clf_object):
    y_pred = clf_object.predict(X_test)
    print("Predicted Values:\n", y_pred)
    return y_pred


def cal_accuracy(y_test, y_pred):
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nAccuracy:", accuracy_score(y_test, y_pred) * 100)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

def plot_decision_tree(clf_object, feature_names, class_names):
    plt.figure(figsize=(15, 10))
    plot_tree(clf_object, filled=True, feature_names=feature_names, class_names=class_names, rounded=True)
    plt.show()


def main():
 
    data = load_data()
    X_train, X_test, y_train, y_test, feature_names = splitdataset(data)

  
    print("\n----- Training Using Gini -----")
    clf_gini = train_using_gini(X_train, y_train)
    y_pred_gini = prediction(X_test, clf_gini)
    cal_accuracy(y_test, y_pred_gini)

    
    print("\n----- Training Using Entropy -----")
    clf_entropy = train_using_entropy(X_train, y_train)
    y_pred_entropy = prediction(X_test, clf_entropy)
    cal_accuracy(y_test, y_pred_entropy)

if __name__ == "__main__":
    main()