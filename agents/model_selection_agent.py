import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


class CustomPCA:
    """User-defined PCA implementation."""
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit_transform(self, X):
        X_arr = np.array(X)
        self.mean = np.mean(X_arr, axis=0)
        X_centered = X_arr - self.mean
        cov_matrix = np.cov(X_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        self.components = eigenvectors[:, sorted_indices[:self.n_components]]
        return np.dot(X_centered, self.components)


class CustomLogisticRegression:
    """User-defined Logistic Regression using One-vs-Rest for multi-class support."""
    def __init__(self, learning_rate=0.01, max_iter=1000):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.classes_ = None
        self.models = {}

    def _sigmoid(self, x):
        x = np.clip(x, -250, 250)
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        X_arr = np.array(X)
        y_arr = np.array(y)

        for c in self.classes_:
            y_binary = np.where(y_arr == c, 1, 0)
            n_samples, n_features = X_arr.shape
            weights = np.zeros(n_features)
            bias = 0

            for _ in range(self.max_iter):
                linear_model = np.dot(X_arr, weights) + bias
                y_predicted = self._sigmoid(linear_model)

                dw = (1 / n_samples) * np.dot(X_arr.T, (y_predicted - y_binary))
                db = (1 / n_samples) * np.sum(y_predicted - y_binary)

                weights -= self.learning_rate * dw
                bias -= self.learning_rate * db

            self.models[c] = (weights, bias)
        return self

    def predict_proba(self, X):
        X_arr = np.array(X)
        probas = []
        for c in self.classes_:
            weights, bias = self.models[c]
            linear_model = np.dot(X_arr, weights) + bias
            prob = self._sigmoid(linear_model)
            probas.append(prob)
        probas = np.array(probas).T
        # Normalize probabilities so they sum to 1 across classes
        probas = probas / probas.sum(axis=1, keepdims=True)
        return probas

    def predict(self, X):
        probas = self.predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)]


class ModelSelectionAgent:

    def __init__(self, file_path):
        self.df = pd.read_csv(file_path)

    def preprocess(self):

        # region_name is human-readable text kept for reporting only -
        # it must not go into the model's feature matrix, only the
        # already-encoded "region" column should.
        drop_cols = [c for c in ["outbreak_risk", "region_name"] if c in self.df.columns]

        X = self.df.drop(columns=drop_cols)
        y = self.df["outbreak_risk"]

        if "region" in X.columns and X["region"].dtype == object:
            encoder = LabelEncoder()
            X["region"] = encoder.fit_transform(X["region"])

        return train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

    def train(self):

        X_train, X_test, y_train, y_test = self.preprocess()

        models = {

            "Random Forest": RandomForestClassifier(random_state=42),

            "Decision Tree": DecisionTreeClassifier(random_state=42),

            # Using the custom class instead of sklearn's built-in function
            "Logistic Regression": CustomLogisticRegression(max_iter=1000),

            "KNN": KNeighborsClassifier(),

            # SVM probability set to True to prevent LIME Explainer crash
            "SVM": SVC(probability=True),

            "XGBoost": XGBClassifier(eval_metric="mlogloss"),

            "LightGBM": LGBMClassifier(),

            "CatBoost": CatBoostClassifier(verbose=0)

        }

        best_model = None
        best_name = ""
        best_accuracy = 0
        scores = {}

        print("\n===== Model Accuracy =====")

        for name, model in models.items():

            model.fit(X_train, y_train)

            prediction = model.predict(X_test)

            accuracy = accuracy_score(y_test, prediction)

            print(f"{name}: {accuracy:.4f}")

            scores[name] = round(float(accuracy), 4)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
                best_name = name

        print("\nBest Model:", best_name)
        print("Accuracy:", round(best_accuracy, 4))

        os.makedirs("models", exist_ok=True)
        joblib.dump(best_model, "models/best_model.pkl")

        print("\nModel saved in models/best_model.pkl")

        return {
            "scores": scores,
            "best_name": best_name,
            "best_accuracy": round(float(best_accuracy), 4),
            "model_path": "models/best_model.pkl",
        }


if __name__ == "__main__":

    agent = ModelSelectionAgent("datasets/final_dataset.csv")

    agent.train()
