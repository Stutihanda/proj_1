import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


class DynamicFusionAgent:

    def __init__(self, dataframes: list):
        """
        Accepts any list of pandas DataFrames (e.g., [df1, df2, df3, ...])
        """
        self.dfs = [df.copy() for df in dataframes if df is not None and not df.empty]

    def _find_best_join_key(self):
        """
        Scans all DataFrames to automatically discover common join keys.
        """
        if not self.dfs:
            return None

        # Gather column sets from all DataFrames (normalized)
        col_sets = []
        for df in self.dfs:
            cleaned_cols = {str(col).strip().lower(): str(col).strip() for col in df.columns}
            col_sets.append(cleaned_cols)

        # Find intersection of lowercase column names across ALL DataFrames
        common_lower_cols = set(col_sets[0].keys())
        for c_set in col_sets[1:]:
            common_lower_cols = common_lower_cols.intersection(set(c_set.keys()))

        if common_lower_cols:
            # Priority join keys if multiple common columns exist
            priority_keys = ["country", "city", "region_name", "region", "state", "id", "date", "zipcode"]
            for pk in priority_keys:
                if pk in common_lower_cols:
                    return col_sets[0][pk]
            
            # Return the first matching common column
            first_common = list(common_lower_cols)[0]
            return col_sets[0][first_common]

        return None

    def _clean_and_suffix(self, df, join_key, ds_index):
        """
        1. Cleans string formatting and fixes data type mismatches (object vs float).
        2. Uniquely suffixes overlapping non-key columns to prevent duplicate column names.
        """
        df = df.copy()
        
        # Clean column spaces
        df.columns = [str(col).strip() for col in df.columns]

        # Fix object vs float64 type mismatch on the join key
        if join_key and join_key in df.columns:
            df[join_key] = (
                df[join_key]
                .fillna("Unknown")
                .astype(str)
                .str.strip()
                .str.title()
            )

        # Rename overlapping non-key columns
        new_columns = {}
        for col in df.columns:
            if col != join_key:
                new_columns[col] = f"{col}_ds{ds_index}"

        return df.rename(columns=new_columns)

    def merge_data(self):
        if not self.dfs:
            raise ValueError("No valid DataFrames were provided to the Fusion Agent.")

        if len(self.dfs) == 1:
            print("Single dataset provided. Skipping merge.")
            return self.dfs[0]

        join_key = self._find_best_join_key()

        if join_key:
            print(f"🔗 Dynamically detected join key: '{join_key}' across all datasets.")
            
            # Clean and apply unique column suffixes to each dataframe
            cleaned_dfs = [
                self._clean_and_suffix(df, join_key, idx + 1) 
                for idx, df in enumerate(self.dfs)
            ]

            # Sequential outer merge
            merged_df = cleaned_dfs[0]
            for next_df in cleaned_dfs[1:]:
                merged_df = pd.merge(merged_df, next_df, on=join_key, how="outer")

        else:
            print("⚠️ No common join key found. Performing dynamic horizontal concatenation...")
            cleaned_dfs = []
            for idx, df in enumerate(self.dfs):
                df_c = df.copy()
                df_c.columns = [f"{c}_ds{idx+1}" for c in df_c.columns]
                cleaned_dfs.append(df_c.reset_index(drop=True))

            merged_df = pd.concat(cleaned_dfs, axis=1)
            join_key = "region_name"
            merged_df[join_key] = "Zone_A"

        # Encode categorical join key into integer 'region'
        if join_key in merged_df.columns:
            le = LabelEncoder()
            merged_df["region_code"] = le.fit_transform(merged_df[join_key].astype(str))

        # Absolute safety check against duplicate column names
        merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]

        print("Data Fusion Complete!")
        return merged_df

    def remove_duplicates(self, df):
        before = len(df)
        df = df.drop_duplicates()
        print(f"Removed {before - len(df)} duplicate records.")
        return df

    def create_target(self, df):
        """
        Dynamically generates an 'outbreak_risk' target from ANY continuous numeric feature available.
        """
        # Exclude administrative ID/Code columns from target selection
        exclude_kw = ["id", "code", "zip", "latitude", "longitude", "epoch", "region_code"]
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        candidate_cols = [
            c for c in numeric_cols 
            if not any(k in c.lower() for k in exclude_kw) and df[c].nunique() > 1
        ]

        target_col = candidate_cols[0] if candidate_cols else (numeric_cols[0] if numeric_cols else None)

        if target_col:
            print(f"🎯 Dynamically selected driver attribute '{target_col}' for target creation.")
            clean_series = pd.to_numeric(df[target_col], errors="coerce").fillna(0)

            try:
                # Quantile binning into 3 Risk Levels (0: Low, 1: Medium, 2: High)
                df["outbreak_risk"] = pd.qcut(
                    clean_series, q=3, labels=[0, 1, 2], duplicates="drop"
                ).astype(int)
            except Exception:
                # Fallback equal-width binning
                df["outbreak_risk"] = pd.cut(
                    clean_series, bins=3, labels=[0, 1, 2]
                ).fillna(0).astype(int)
        else:
            print("⚠️ No numeric column found. Creating default risk target.")
            df["outbreak_risk"] = 0

        print("Target Distribution:\n", df["outbreak_risk"].value_counts())
        return df

    def run(self):
        merged_df = self.merge_data()
        merged_df = self.remove_duplicates(merged_df)
        merged_df = self.create_target(merged_df)
        
        print("\nFinal Merged Dataset Shape:", merged_df.shape)
        return merged_df


# ==========================================
# EXAMPLE USAGE WITH ARBITRARY CSVs
# ==========================================
if __name__ == "__main__":
    import os
    import glob

    # Dynamically pick up whatever CSV files exist in datasets folder
    csv_files = glob.glob("datasets/*.csv")
    csv_files = [f for f in csv_files if "final_dataset" not in f]

    if csv_files:
        print(f"Found {len(csv_files)} CSV files: {csv_files}")
        loaded_dfs = [pd.read_csv(f) for f in csv_files]

        # Run Dynamic Fusion Agent
        agent = DynamicFusionAgent(loaded_dfs)
        final_df = agent.run()

        os.makedirs("datasets", exist_ok=True)
        final_df.to_csv("datasets/final_dataset.csv", index=False)
        print("\nSaved output to 'datasets/final_dataset.csv'")
        print(final_df.head())
    else:
        print("No CSV files found in 'datasets/' folder to process.")