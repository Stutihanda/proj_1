import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

class SocialAgent:

    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None

    def load_data(self):
        self.df = pd.read_csv(self.file_path)
        print("Social Dataset Loaded")
        return self.df

    def clean_data(self):

        # Fill numeric missing values
        numeric_cols = self.df.select_dtypes(include=['int64','float64']).columns
        self.df[numeric_cols] = self.df[numeric_cols].fillna(
            self.df[numeric_cols].mean()
        )

        # Fill categorical missing values
        cat_cols = self.df.select_dtypes(include=['object']).columns
        self.df[cat_cols] = self.df[cat_cols].fillna("Unknown")

        return self.df

    def encode_data(self):

        encoder = LabelEncoder()

        cat_cols = self.df.select_dtypes(include=['object']).columns

        for col in cat_cols:
            self.df[col] = encoder.fit_transform(self.df[col])

        return self.df

    def feature_engineering(self):

        # Example feature
        if "population" in self.df.columns and "area" in self.df.columns:

            self.df["population_density"] = (
                self.df["population"] /
                self.df["area"]
            )

        return self.df

    def scale_data(self):

        scaler = StandardScaler()

        numeric_cols = self.df.select_dtypes(include=['int64','float64']).columns

        self.df[numeric_cols] = scaler.fit_transform(
            self.df[numeric_cols]
        )

        return self.df

    def run(self):

        self.load_data()
        self.clean_data()
        self.encode_data()
        self.feature_engineering()
        self.scale_data()

        print("Social Agent Completed")

        return self.df


if __name__ == "__main__":

    agent = SocialAgent("social.csv")

    social_data = agent.run()

    print(social_data.head())