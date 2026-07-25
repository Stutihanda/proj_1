import pandas as pd


class DataValidationAgent:

    def __init__(self, df):
        self.df = df

    def check_missing_values(self):
        print("\n===== Missing Values =====")
        print(self.df.isnull().sum())

    def check_duplicates(self):
        duplicates = self.df.duplicated().sum()
        print(f"\nDuplicate Rows: {duplicates}")

    def check_datatypes(self):
        print("\n===== Data Types =====")
        print(self.df.dtypes)

    def check_outliers(self):

        numeric = self.df.select_dtypes(include=["int64", "float64"])

        print("\n===== Outliers =====")

        for col in numeric.columns:

            Q1 = numeric[col].quantile(0.25)
            Q3 = numeric[col].quantile(0.75)

            IQR = Q3 - Q1

            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

            outliers = numeric[
                (numeric[col] < lower) |
                (numeric[col] > upper)
            ]

            print(f"{col}: {len(outliers)}")

    def check_class_balance(self, target):

        print("\n===== Target Distribution =====")

        if target in self.df.columns:
            print(self.df[target].value_counts())
        else:
            print(f"Column '{target}' not found!")

    def run(self, target):

        print("=" * 50)
        print("DATA VALIDATION AGENT")
        print("=" * 50)

        self.check_missing_values()
        self.check_duplicates()
        self.check_datatypes()
        self.check_outliers()
        self.check_class_balance(target)

        print("\nData Validation Completed Successfully")


if __name__ == "__main__":

    # Read the dataset
    df = pd.read_csv("datasets/final_dataset.csv")

    # Create the agent
    agent = DataValidationAgent(df)

    # Run validation
    agent.run("outbreak_risk")