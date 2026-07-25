from agents.climate_agent import ClimateAgent
from agents.health_agent import HealthAgent
from agents.social_agent import SocialAgent
from agents.fusion_agent import FusionAgent
from agents.data_validation_agent import DataValidationAgent
from agents.model_selection_agent import ModelSelectionAgent
from agents.automl_agent import AutoMLAgent
from agents.explainability_agent import ExplainabilityAgent
from agents.decision_agent import DecisionAgent

import pandas as pd
from sklearn.model_selection import train_test_split


def main():

    print("=" * 60)
    print("CLIMATE GUARDIAN AI")
    print("=" * 60)

    # -----------------------------
    # Step 1: Climate Agent
    # -----------------------------
    climate = ClimateAgent("datasets/climate.csv")
    climate_df = climate.run()

    # -----------------------------
    # Step 2: Health Agent
    # -----------------------------
    health = HealthAgent("datasets/health.csv")
    health_df = health.run()

    # -----------------------------
    # Step 3: Social Agent
    # -----------------------------
    social = SocialAgent("datasets/social.csv")
    social_df = social.run()

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
    validator.run("outbreak_risk")

    # -----------------------------
    # Step 6: Model Selection
    # -----------------------------
    model_agent = ModelSelectionAgent("datasets/final_dataset.csv")
    model_agent.train()

    # -----------------------------
    # Step 7: AutoML
    # -----------------------------
    automl = AutoMLAgent("datasets/final_dataset.csv")
    automl.optimize()

    # -----------------------------
    # Step 8: Explainability
    # -----------------------------
    df = pd.read_csv("datasets/final_dataset.csv")

    X = df.drop("outbreak_risk", axis=1)
    y = df["outbreak_risk"]
    
    X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

    explain = ExplainabilityAgent(
        "models/best_automl_model.pkl",
        X_train,
        X_test
    )

    explain.run()

    # -----------------------------
    # Step 9: Decision Agent
    # -----------------------------
    decision = DecisionAgent(2)
    decision.recommend()

    print("\nPROJECT EXECUTED SUCCESSFULLY")


if __name__ == "__main__":
    main()