import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB, GaussianNB
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = (
    PROJECT_ROOT / "csvs" / "Response-Time-Noise-Removed-Fire-and-EMS-Response-Data.csv"
)

def load_data():
    balanced_data = pd.read_csv(INPUT_PATH)

    return balanced_data

def train_naive_bayes(balanced_data):
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

    # Identical 80/20 split so test rows match the Decision Trees exactly
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42, stratify=Y
    )

    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_train_cat = (enc.fit_transform(X_train[cat_cols]) + 1).astype(int)
    X_test_cat = (enc.transform(X_test[cat_cols]) + 1).astype(int)

    cnb = CategoricalNB().fit(X_train_cat, y_train)
    gnb = GaussianNB().fit(X_train[num_cols], y_train)

    p_cat = cnb.predict_proba(X_test_cat)
    p_num = gnb.predict_proba(X_test[num_cols])
    joint_prob = (p_cat * p_num) / gnb.class_prior_

    y_pred = [1 if prob_1 > prob_0 else 0 for prob_0, prob_1 in joint_prob]

    print("Predicted Values:\n", y_pred[:20])
    cal_accuracy(y_test, y_pred)
    return y_pred

def prediction(X_test, clf_object):
    y_pred = clf_object.predict(X_test)
    print("Predicted Values:\n", y_pred)
    return y_pred


def cal_accuracy(y_test, y_pred):
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nAccuracy:", accuracy_score(y_test, y_pred) * 100)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

def main():
 
    data = load_data()

  
    print("\n----- Training Using Naive Bayes -----")
    train_naive_bayes(data)

if __name__ == "__main__":
    main()