# based on https://www.datacamp.com/tutorial/random-forests-classifier-python
from pathlib import Path

# Data Processing
import pandas as pd
import numpy as np

# Modelling
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint

# Tree Visualisation
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import export_graphviz
from IPython.display import Image
import graphviz

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = (
    PROJECT_ROOT / "csvs" / "Response-Time-Noise-Removed-Fire-and-EMS-Response-Data.csv"
)


CAT_COLS = [
    "call_type_group",
    "call_type",
    "original_priority",
    "unit_type",
    "station_area",
    "battalion",
    "neighborhoods_analysis_boundaries",
]
NUM_COLS = ["distance_km"]


def load_data():
    balanced_data = pd.read_csv(INPUT_PATH)

    return balanced_data


# reused from decision tree model, as the data processing is the same
def splitdataset(balanced_data):

    X = balanced_data.drop(columns=["outcome", "incident_response_time"])
    Y = balanced_data["outcome"].astype(int).values

    for col in CAT_COLS:
        X[col] = X[col].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42, stratify=Y
    )

    onehot_encoder = OneHotEncoder(
        sparse_output=False, drop=None, handle_unknown="ignore"
    )

    X_train_cat = onehot_encoder.fit_transform(X_train[CAT_COLS])
    X_test_cat = onehot_encoder.transform(X_test[CAT_COLS])

    cat_feature_names = onehot_encoder.get_feature_names_out(CAT_COLS)

    # Recombine encoded categorical columns with numeric 'distance_km'
    X_train_final = pd.concat(
        [
            X_train[NUM_COLS].reset_index(drop=True),
            pd.DataFrame(X_train_cat, columns=cat_feature_names),
        ],
        axis=1,
    )
    X_test_final = pd.concat(
        [
            X_test[NUM_COLS].reset_index(drop=True),
            pd.DataFrame(X_test_cat, columns=cat_feature_names),
        ],
        axis=1,
    )

    return X_train_final, X_test_final, y_train, y_test, list(X_train_final.columns)


# from decision tree model
def cal_accuracy(y_test, y_pred):
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nAccuracy:", accuracy_score(y_test, y_pred) * 100)
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))


# from link above, with some modifications to hyperparameter ranges
def train_rf(X_train, y_train):
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    return rf


# based on https://www.datacamp.com/tutorial/random-forests-classifier-python
def tune_rf(X_train, y_train):
    param_dist = {
        "n_estimators": randint(100, 200),
        "max_depth": randint(3, 15),
        "min_samples_split": randint(2, 10),
        "min_samples_leaf": randint(1, 5),
    }

    # Create a random forest classifier
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)

    # Use random search to find the best hyperparameters
    rand_search = RandomizedSearchCV(
        rf,
        param_distributions=param_dist,
        n_iter=10,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        random_state=42,
    )

    # Fit the random search object to the data
    rand_search.fit(X_train, y_train)

    # Print the best hyperparameters
    print("Best hyperparameters:", rand_search.best_params_)

    # Create a variable for the best model
    return rand_search.best_estimator_


def export_trees(rf, feature_names):
    # Export the first three decision trees from the forest
    for i in range(3):
        tree = rf.estimators_[i]
        dot_data = export_graphviz(
            tree,
            feature_names=feature_names,
            filled=True,
            max_depth=2,
            impurity=False,
            proportion=True,
        )
        graph = graphviz.Source(dot_data)
        graph.render(f"tree_{i}", format="png", cleanup=True)


# based on https://www.datacamp.com/tutorial/random-forests-classifier-python
def plot_importances(best_rf, feature_names):
    # Create a series containing feature importances from the model and feature names from the training data
    importances = pd.Series(best_rf.feature_importances_, index=feature_names)

    # Add the one-hot columns back together so each original column has one score
    original = {
        name: next((c for c in CAT_COLS if name.startswith(c + "_")), name)
        for name in feature_names
    }
    importances = importances.groupby(original).sum()

    importances.sort_values(ascending=False).plot.bar()
    plt.tight_layout()
    plt.show()


def main():
    data = load_data()
    X_train, X_test, y_train, y_test, feature_names = splitdataset(data)

    print("\n----- Random Forest -----")
    rf = train_rf(X_train, y_train)
    y_pred = rf.predict(X_test)
    cal_accuracy(y_test, y_pred)
    export_trees(rf, feature_names)
    plot_importances(rf, feature_names)

    y_pred_train = rf.predict(X_train)
    print(f"Train accuracy: {accuracy_score(y_train, y_pred_train) * 100:.4f}")

    print("\n----- Tuned Random Forest -----")
    best_rf = tune_rf(X_train, y_train)
    y_pred = best_rf.predict(X_test)
    cal_accuracy(y_test, y_pred)
    plot_importances(best_rf, feature_names)

    y_pred_train = best_rf.predict(X_train)
    print(f"Train accuracy: {accuracy_score(y_train, y_pred_train) * 100:.4f}")


if __name__ == "__main__":
    main()
