import pandas as pd

from src.console import configure_console


class DataProfiler:

    def profile(self, df: pd.DataFrame):

        configure_console()

        print("\n========== PROFIL DES DONNÉES ==========\n")

        print("Colonnes :")
        print(df.columns.tolist())

        print("\nTypes :")
        print(df.dtypes)

        print("\nValeurs manquantes :")
        print(df.isnull().sum())

        print("\nDoublons :", df.duplicated().sum())

        print("\nStatistiques :")
        print(df.describe(include="all"))

        return {
            "rows": df.shape[0],
            "columns": df.shape[1],
            "missing_values": int(df.isnull().sum().sum()),
            "duplicates": int(df.duplicated().sum())
        }
