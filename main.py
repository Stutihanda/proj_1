import os
import json

import pandas as pd
from sklearn.model_selection import train_test_split

from agents.climate_agent import ClimateAgent
from agents.health_agent import HealthAgent
from agents.social_agent import SocialAgent
from agents.fusion_agent import FusionAgent
from agents.data_validation_agent import DataValidationAgent
from agents.model_selection_agent import ModelSelectionAgent
from agents.automl_agent import AutoMLAgent
from agents.explainability_agent import ExplainabilityAgent
from agents.decision_agent import DecisionAgent

import joblib


def main(
    climate_path="datasets/climate.csv",
    health_path="datasets/health.csv",
    social_path="datasets/social.csv",
):

    print("=" * 60)
    print("CLIMATE GUARDIAN AI")
    print("=" * 60)

    os.makedirs("datasets", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # -----------------------------
    # Step 1-3: Domain Agents
    # -----------------------------
    climate_df = ClimateAgent(climate_path).run()
    health_df = HealthAgent(health_path).run()
    social_df = SocialAgent(social_path).run()

    # -----------------------------
    # Step 4: Fusion Agent
    # -----------------------------
    fusion = FusionAgent(climate_df, health_df, social_df)
    final_df = fusion.run()

    final_df.to_csv("datasets/final_dataset.csv", index=False)
    print("\nFinal dataset created.")

    # -----------------------------
    # Step 5: Data Validation
    # -----------------------------
    validator = DataValidationAgent(final_df)
    validation_results = validator.run("outbreak_risk")

    # -----------------------------
    # Step 6: Model Selection
    # -----------------------------
    model_agent = ModelSelectionAgent("datasets/final_dataset.csv")
    model_results = model_agent.train()

    # -----------------------------
    # Step 7: AutoML
    # -----------------------------
    automl = AutoMLAgent("datasets/final_dataset.csv")
    automl_results = automl.optimize()

    # -----------------------------
    # Step 8: Predict risk per region (fixes the old hardcoded DecisionAgent(2))
    # -----------------------------
    df = pd.read_csv("datasets/final_dataset.csv")

    drop_cols = [c for c in ["outbreak_risk", "region_name"] if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df["outbreak_risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    best_model = joblib.load(model_results["model_path"])

    # Predict for every row so we get one risk level per region, not one
    # global guess. This replaces the old hardcoded DecisionAgent(2).
    all_predictions = best_model.predict(X)

    region_risk_df = df[["region_name"]].copy() if "region_name" in df.columns else pd.DataFrame()
    region_risk_df["predicted_risk"] = all_predictions

    # If a region appears multiple times, keep its highest observed risk
    if not region_risk_df.empty:
        region_risk_table = (
            region_risk_df.groupby("region_name")["predicted_risk"]
            .max()
            .sort_values(ascending=False)
            .to_dict()
        )
    else:
        region_risk_table = {}

    # -----------------------------
    # Step 9: Explainability
    # -----------------------------
    explain = ExplainabilityAgent(model_results["model_path"], X_train, X_test)

    shap_fig = explain.shap_explanation()
    lime_html = explain.lime_explanation()
    top_features = explain.feature_importance_summary()

    os.makedirs("results", exist_ok=True)
    shap_path = "results/shap_summary.png"
    shap_fig.savefig(shap_path, bbox_inches="tight")

    lime_path = "results/lime_explanation.html"
    with open(lime_path, "w", encoding="utf-8") as f:
        f.write(lime_html)

    # -----------------------------
    # Step 10: Decision Agent - one decision per region, driven by the
    # model's actual prediction, not a hardcoded number
    # -----------------------------
    decisions = {}
    for region, risk in region_risk_table.items():
        decisions[region] = DecisionAgent(risk).recommend()

    print("\nPROJECT EXECUTED SUCCESSFULLY")

    results = {
        "validation": validation_results,
        "model_comparison": model_results,
        "automl": automl_results,
        "region_risk_table": region_risk_table,
        "top_features": top_features,
        "decisions": decisions,
        "shap_image_path": shap_path,
        "lime_html_path": lime_path,
    }

    with open("results/latest_run.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    return results


if __name__ == "__main__":
    main()
