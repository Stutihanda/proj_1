class DecisionAgent:

    def __init__(self, prediction):
        self.prediction = prediction

    def recommend(self):

        print("\n===== DECISION AGENT =====")

        if self.prediction == 0:
            print("Prediction: LOW OUTBREAK RISK")
            print("- Continue routine monitoring.")
            print("- Maintain sanitation.")
            print("- Regular health checkups.")

        elif self.prediction == 1:
            print("Prediction: MEDIUM OUTBREAK RISK")
            print("- Increase disease surveillance.")
            print("- Conduct awareness campaigns.")
            print("- Prepare nearby hospitals.")

        else:
            print("Prediction: HIGH OUTBREAK RISK")
            print("- Issue public health alert.")
            print("- Increase mosquito control.")
            print("- Deploy emergency medical teams.")
            print("- Increase hospital resources.")


if __name__ == "__main__":

    # Example prediction
    prediction = 2

    agent = DecisionAgent(prediction)

    agent.recommend()