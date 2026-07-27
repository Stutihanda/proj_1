import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV

from xgboost import XGBClassifier


class AutoMLAgent:

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

    def optimize(self):

        X_train, X_test, y_train, y_test = self.preprocess()

        model = XGBClassifier(
            eval_metric="mlogloss",
            random_state=42
        )

        parameters = {
            "max_depth": [3, 5],
            "learning_rate": [0.01, 0.1],
            "n_estimators": [50, 100]
        }

        grid = GridSearchCV(
            estimator=model,
            param_grid=parameters,
            cv=2,
            scoring="accuracy"
        )

        grid.fit(X_train, y_train)

        print("\n===== Best Parameters =====")
        print(grid.best_params_)

        print("\nBest Accuracy:", round(grid.best_score_, 4))

        os.makedirs("models", exist_ok=True)
        joblib.dump(
            grid.best_estimator_,
            "models/best_automl_model.pkl"
        )

        print("\nOptimized model saved successfully.")

        return {
            "best_params": grid.best_params_,
            "best_accuracy": round(float(grid.best_score_), 4),
            "model_path": "models/best_automl_model.pkl",
        }


if __name__ == "__main__":

    agent = AutoMLAgent("datasets/final_dataset.csv")

    agent.optimize()
