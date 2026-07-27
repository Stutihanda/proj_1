import pandas as pd


class DataValidationAgent:

    def __init__(self, df):
        self.df = df

    def check_missing_values(self):
        result = self.df.isnull().sum()
        print("\n===== Missing Values =====")
        print(result)
        return result.to_dict()

    def check_duplicates(self):
        duplicates = int(self.df.duplicated().sum())
        print(f"\nDuplicate Rows: {duplicates}")
        return duplicates

    def check_datatypes(self):
        result = self.df.dtypes.astype(str)
        print("\n===== Data Types =====")
        print(result)
        return result.to_dict()

    def check_outliers(self):

        numeric = self.df.select_dtypes(include=["int64", "float64"])

        print("\n===== Outliers =====")

        outlier_counts = {}

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

            outlier_counts[col] = int(len(outliers))

        return outlier_counts

    def check_class_balance(self, target):

        print("\n===== Target Distribution =====")

        if target in self.df.columns:
            counts = self.df[target].value_counts()
            print(counts)
            return counts.to_dict()
        else:
            print(f"Column '{target}' not found!")
            return {}

    def run(self, target):

        print("=" * 50)
        print("DATA VALIDATION AGENT")
        print("=" * 50)

        results = {
            "missing_values": self.check_missing_values(),
            "duplicate_rows": self.check_duplicates(),
            "datatypes": self.check_datatypes(),
            "outliers": self.check_outliers(),
            "class_balance": self.check_class_balance(target),
        }

        print("\nData Validation Completed Successfully")

        return results


if __name__ == "__main__":

    # Read the dataset
    df = pd.read_csv("datasets/final_dataset.csv")

    # Create the agent
    agent = DataValidationAgent(df)

    # Run validation
    agent.run("outbreak_risk")
