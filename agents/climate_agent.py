import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

class ClimateAgent:

    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None

    def load_data(self):
        self.df = pd.read_csv(self.file_path)

        # Keep a readable copy of region BEFORE it gets label-encoded below,
        # so downstream reporting / AI Q&A can say "Chennai" instead of "3"
        if "region" in self.df.columns:
            self.df["region_name"] = self.df["region"].astype(str)

        print("Climate Dataset Loaded")
        return self.df

    def clean_data(self):
        # Fill missing numerical values
        numeric_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
        self.df[numeric_cols] = self.df[numeric_cols].fillna(
            self.df[numeric_cols].mean()
        )

        # Fill missing categorical values
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        self.df[categorical_cols] = self.df[categorical_cols].fillna("Unknown")

        print("Missing Values Handled")
        return self.df

    def encode_data(self):
        encoder = LabelEncoder()

        categorical_cols = self.df.select_dtypes(include=['object']).columns

        # region_name is kept as readable text on purpose - never encode it
        categorical_cols = [c for c in categorical_cols if c != "region_name"]

        for col in categorical_cols:
            self.df[col] = encoder.fit_transform(self.df[col])

        print("Categorical Columns Encoded")
        return self.df

    def scale_data(self):
        scaler = StandardScaler()

        numeric_cols = self.df.select_dtypes(include=['int64', 'float64']).columns

        self.df[numeric_cols] = scaler.fit_transform(self.df[numeric_cols])

        print("Features Scaled")
        return self.df

    def run(self):
        self.load_data()
        self.clean_data()
        self.encode_data()
        self.scale_data()

        print("Climate Agent Completed")

        return self.df


if __name__ == "__main__":
    agent = ClimateAgent("datasets/climate.csv")
    climate_data = agent.run()

    print(climate_data.head())
