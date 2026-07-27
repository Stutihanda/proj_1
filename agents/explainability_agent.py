import joblib
import shap
import lime
import lime.lime_tabular
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, required for Streamlit/servers
import matplotlib.pyplot as plt


class ExplainabilityAgent:

    def __init__(self, model_path, X_train, X_test):
        self.model = joblib.load(model_path)
        self.X_train = X_train
        self.X_test = X_test

    def _build_explainer(self):
        """Pick the right SHAP explainer for whichever model actually won.

        ModelSelectionAgent compares 8 different model types, so we can't
        assume a tree model here. For tree-based winners (Random Forest,
        Decision Tree, XGBoost, LightGBM, CatBoost), TreeExplainer with
        feature_perturbation="tree_path_dependent" is used - this reads the
        tree structure directly and doesn't need background data, which
        avoids the "Categorical split is not yet supported" error some
        tree models (notably CatBoost) throw under SHAP's default
        interventional mode. For non-tree winners (Logistic Regression,
        KNN, SVM), we fall back to the generic auto-dispatching Explainer.
        """

        tree_model_names = ("RandomForest", "DecisionTree", "XGB", "LGBM", "CatBoost")
        model_name = type(self.model).__name__

        if any(name in model_name for name in tree_model_names):
            return shap.TreeExplainer(
                self.model,
                feature_perturbation="tree_path_dependent",
            )

        return shap.Explainer(self.model, self.X_train)

    def shap_explanation(self):
        """Returns a matplotlib Figure (does not open a GUI window)."""

        explainer = self._build_explainer()
        shap_values = explainer(self.X_test)

        plt.figure()
        shap.summary_plot(shap_values, self.X_test, show=False)
        fig = plt.gcf()
        fig.tight_layout()

        print("SHAP Explanation Generated")

        return fig

    def lime_explanation(self, row_index=0):
        """Returns LIME explanation as an HTML string (no notebook required)."""

        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=self.X_train.values,
            feature_names=self.X_train.columns.tolist(),
            class_names=["Low", "Medium", "High"],
            mode="classification",
        )

        explanation = explainer.explain_instance(
            self.X_test.iloc[row_index].values,
            self.model.predict_proba,
        )

        print("LIME Explanation Generated")

        return explanation.as_html()

    def feature_importance_summary(self, top_n=10):
        """Returns a plain-language ranked list of the top features driving predictions.
        This is what the AI Q&A layer reads from when a user asks 'what are the important features?'
        """

        explainer = self._build_explainer()
        shap_values = explainer(self.X_test)

        importances = abs(shap_values.values).mean(axis=0)

        # Multi-class SHAP returns one importance per class; average across classes if needed
        if importances.ndim > 1:
            importances = importances.mean(axis=1)

        ranked = sorted(
            zip(self.X_test.columns.tolist(), importances),
            key=lambda pair: pair[1],
            reverse=True,
        )[:top_n]

        return [{"feature": name, "importance": round(float(score), 4)} for name, score in ranked]

    def run(self):
        shap_fig = self.shap_explanation()
        lime_html = self.lime_explanation()

        print("Explainability Completed")

        return {"shap_figure": shap_fig, "lime_html": lime_html}
