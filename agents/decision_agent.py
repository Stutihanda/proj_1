class DecisionAgent:

    LABELS = {0: "LOW OUTBREAK RISK", 1: "MEDIUM OUTBREAK RISK", 2: "HIGH OUTBREAK RISK"}

    RECOMMENDATIONS = {
        0: [
            "Continue routine monitoring.",
            "Maintain sanitation.",
            "Regular health checkups.",
        ],
        1: [
            "Increase disease surveillance.",
            "Conduct awareness campaigns.",
            "Prepare nearby hospitals.",
        ],
        2: [
            "Issue public health alert.",
            "Increase mosquito control.",
            "Deploy emergency medical teams.",
            "Increase hospital resources.",
        ],
    }

    def __init__(self, prediction):
        self.prediction = int(prediction)

    def recommend(self):
        """Returns a dict so callers (Streamlit, API, CLI) can use the result directly."""

        label = self.LABELS.get(self.prediction, "UNKNOWN RISK")
        actions = self.RECOMMENDATIONS.get(self.prediction, [])

        print("\n===== DECISION AGENT =====")
        print(f"Prediction: {label}")
        for action in actions:
            print(f"- {action}")

        return {
            "risk_level": self.prediction,
            "risk_label": label,
            "recommendations": actions,
        }


if __name__ == "__main__":

    # Example prediction
    prediction = 2

    agent = DecisionAgent(prediction)

    agent.recommend()
