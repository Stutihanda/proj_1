import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


class ModelSelectionAgent:

    def __init__(self, file_path):
        self.df = pd.read_csv(file_path)

    def preprocess(self):

        X = self.df.drop("outbreak_risk", axis=1)
        y = self.df["outbreak_risk"]

        if "region" in X.columns:
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

            "Logistic Regression": LogisticRegression(max_iter=1000),

            "KNN": KNeighborsClassifier(),

            "SVM": SVC(),

            "XGBoost": XGBClassifier(eval_metric="mlogloss"),

            "LightGBM": LGBMClassifier(),

            "CatBoost": CatBoostClassifier(verbose=0)

        }

        best_model = None
        best_name = ""
        best_accuracy = 0

        print("\n===== Model Accuracy =====")

        for name, model in models.items():

            model.fit(X_train, y_train)

            prediction = model.predict(X_test)

            accuracy = accuracy_score(y_test, prediction)

            print(f"{name}: {accuracy:.4f}")

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
                best_name = name

        print("\nBest Model:", best_name)
        print("Accuracy:", round(best_accuracy, 4))

        joblib.dump(best_model, "models/best_model.pkl")

        print("\nModel saved in models/best_model.pkl")


if __name__ == "__main__":

    agent = ModelSelectionAgent("datasets/final_dataset.csv")

    agent.train()