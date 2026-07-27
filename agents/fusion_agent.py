import pandas as pd
from sklearn.preprocessing import LabelEncoder


class FusionAgent:

    def __init__(self, climate_df, health_df, social_df):
        self.climate_df = climate_df
        self.health_df = health_df
        self.social_df = social_df

    def merge_data(self):

        # IMPORTANT: merge on "region_name", not "region".
        # Each domain agent (Climate/Health/Social) label-encodes "region"
        # with its OWN independently-fit LabelEncoder, so the same city can
        # end up as a different integer in each dataset. Merging on those
        # encoded ints would silently join the wrong rows (or drop
        # everything) whenever the sets of regions aren't identical.
        # "region_name" is the untouched string version, so it's a safe key.

        # Merge Climate + Health
        merged_df = pd.merge(
            self.climate_df,
            self.health_df,
            on="region_name",
            how="inner"
        )

        # Merge Social
        merged_df = pd.merge(
            merged_df,
            self.social_df,
            on="region_name",
            how="inner"
        )

        # The merge above leaves behind the old per-dataset encoded "region"
        # columns (suffixed region_x / region_y / region since each source
        # had its own). They're inconsistent with each other, so drop them
        # and build ONE clean, consistent encoded region column from
        # region_name instead.
        stale_region_cols = [
            c for c in ["region", "region_x", "region_y"] if c in merged_df.columns
        ]
        merged_df = merged_df.drop(columns=stale_region_cols)

        encoder = LabelEncoder()
        merged_df["region"] = encoder.fit_transform(merged_df["region_name"])

        print("Datasets Successfully Merged")

        return merged_df

    def remove_duplicates(self, df):

        df = df.drop_duplicates()

        print("Duplicate Records Removed")

        return df

    def final_check(self, df):

        print("\nFinal Dataset Shape:", df.shape)

        print("\nMissing Values:\n")
        print(df.isnull().sum())

        return df

    def create_target(self, df):

        if "dengue_cases" not in df.columns:
            raise ValueError(
                "Expected a 'dengue_cases' column (from the Health dataset) "
                "to build the outbreak_risk target, but it wasn't found after "
                f"merging. Columns available: {list(df.columns)}"
            )

        # Create 3 balanced classes using dengue_cases
        df["outbreak_risk"] = pd.qcut(
            df["dengue_cases"],
            q=3,
            labels=[0, 1, 2]
        )

        # Convert to integer
        df["outbreak_risk"] = df["outbreak_risk"].astype(int)

        print("\nTarget column 'outbreak_risk' created.")

        print("\nTarget Distribution:")
        print(df["outbreak_risk"].value_counts())

        return df

    def run(self):

        merged_df = self.merge_data()

        merged_df = self.remove_duplicates(merged_df)

        merged_df = self.final_check(merged_df)

        merged_df = self.create_target(merged_df)

        return merged_df


if __name__ == "__main__":

    climate = pd.read_csv("datasets/climate.csv")
    health = pd.read_csv("datasets/health.csv")
    social = pd.read_csv("datasets/social.csv")

    agent = FusionAgent(climate, health, social)

    final_dataset = agent.run()

    final_dataset.to_csv(
        "datasets/final_dataset.csv",
        index=False
    )

    print("\nFinal Dataset Saved Successfully!")

    print(final_dataset.head())
