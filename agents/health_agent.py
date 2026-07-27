import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

class HealthAgent:

    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None

    def load_data(self):
        self.df = pd.read_csv(self.file_path)

        # Keep a readable copy of region BEFORE it gets label-encoded below,
        # so downstream reporting / AI Q&A can say "Chennai" instead of "3"
        if "region" in self.df.columns:
            self.df["region_name"] = self.df["region"].astype(str)

        print("Health Dataset Loaded")
        return self.df

    def clean_data(self):

        # Fill missing numeric values
        numeric_cols = self.df.select_dtypes(include=['int64','float64']).columns
        self.df[numeric_cols] = self.df[numeric_cols].fillna(
            self.df[numeric_cols].median()
        )

        # Fill missing categorical values
        cat_cols = self.df.select_dtypes(include=['object']).columns
        self.df[cat_cols] = self.df[cat_cols].fillna("Unknown")

        return self.df

    def encode_data(self):

        encoder = LabelEncoder()

        cat_cols = self.df.select_dtypes(include=['object']).columns

        # region_name is kept as readable text on purpose - never encode it
        cat_cols = [c for c in cat_cols if c != "region_name"]

        for col in cat_cols:
            self.df[col] = encoder.fit_transform(self.df[col])

        return self.df

    def scale_data(self):

        scaler = StandardScaler()

        numeric_cols = self.df.select_dtypes(include=['int64','float64']).columns

        self.df[numeric_cols] = scaler.fit_transform(self.df[numeric_cols])

        return self.df

    def feature_engineering(self):

        # Example Feature
        if "cases" in self.df.columns and "population" in self.df.columns:

            self.df["infection_rate"] = (
                self.df["cases"] /
                self.df["population"]
            ) * 1000

        return self.df

    def run(self):

        self.load_data()
        self.clean_data()
        self.encode_data()
        self.feature_engineering()
        self.scale_data()

        print("Health Agent Completed")

        return self.df


if __name__ == "__main__":

    agent = HealthAgent("datasets/health.csv")

    health_data = agent.run()

    print(health_data.head())
