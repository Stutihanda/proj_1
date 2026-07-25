import joblib
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt

class ExplainabilityAgent:

    def __init__(self, model_path, X_train, X_test):
        self.model = joblib.load(model_path)
        self.X_train = X_train
        self.X_test = X_test

    def shap_explanation(self):

        explainer = shap.Explainer(self.model)

        shap_values = explainer(self.X_test)

        shap.summary_plot(shap_values, self.X_test)

        print("SHAP Explanation Generated")

    def lime_explanation(self):

        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=self.X_train.values,
            feature_names=self.X_train.columns.tolist(),
            class_names=["Low", "Medium", "High"],
            mode="classification"
        )

        explanation = explainer.explain_instance(
            self.X_test.iloc[0].values,
            self.model.predict_proba
        )

        explanation.show_in_notebook()

        print("LIME Explanation Generated")

    def run(self):

        self.shap_explanation()

        self.lime_explanation()

        print("Explainability Completed")