import pandas as pd
import numpy as np


class IntelligentCleaner:

    def __init__(
        self,
        missing_threshold=0.80,
        remove_duplicates=True,
        remove_constant=True
    ):
        self.missing_threshold = missing_threshold
        self.remove_duplicates = remove_duplicates
        self.remove_constant = remove_constant

        self.report = {}

    def clean(self, df):

        print("\n" + "=" * 50)
        print("        INTELLIGENT DATA CLEANER")
        print("=" * 50)

        df = df.copy()

        initial_shape = df.shape

        # -------------------------------------------------
        # Normalisation des noms
        # -------------------------------------------------

        df.columns = [
            str(col).strip()
            for col in df.columns
        ]

        # -------------------------------------------------
        # Suppression doublons
        # -------------------------------------------------

        duplicates = int(df.duplicated().sum())

        if self.remove_duplicates and duplicates > 0:
            df = df.drop_duplicates()

        print(
            f"\n🗑️ Doublons supprimés : {duplicates}"
        )

        # -------------------------------------------------
        # Colonnes complètement vides
        # -------------------------------------------------

        empty_columns = [
            col
            for col in df.columns
            if df[col].isna().all()
        ]

        if empty_columns:
            df = df.drop(
                columns=empty_columns
            )

        print(
            f"🗑️ Colonnes entièrement vides : "
            f"{empty_columns}"
        )

        # -------------------------------------------------
        # Colonnes avec trop de valeurs manquantes
        # -------------------------------------------------

        missing_ratio = df.isna().mean()

        high_missing = list(
            missing_ratio[
                missing_ratio > self.missing_threshold
            ].index
        )

        if high_missing:
            df = df.drop(
                columns=high_missing
            )

        print(
            f"🗑️ Colonnes avec trop de valeurs "
            f"manquantes : {high_missing}"
        )

        # -------------------------------------------------
        # Colonnes constantes
        # -------------------------------------------------

        constant_columns = [
            col
            for col in df.columns
            if df[col].nunique(
                dropna=False
            ) <= 1
        ]

        if self.remove_constant and constant_columns:
            df = df.drop(
                columns=constant_columns
            )

        print(
            f"🗑️ Colonnes constantes : "
            f"{constant_columns}"
        )

        # -------------------------------------------------
        # Traitement valeurs manquantes
        # -------------------------------------------------

        for col in df.columns:

            if df[col].isna().sum() == 0:
                continue

            if pd.api.types.is_numeric_dtype(
                df[col]
            ):

                median = df[col].median()

                df[col] = df[col].fillna(
                    median
                )

            else:

                mode = df[col].mode()

                if len(mode) > 0:

                    df[col] = df[col].fillna(
                        mode.iloc[0]
                    )

                else:

                    df[col] = df[col].fillna(
                        "Unknown"
                    )

        # -------------------------------------------------
        # Nettoyage texte
        # -------------------------------------------------

        for col in df.select_dtypes(
            include=["object", "string"]
        ).columns:

            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
            )

        # -------------------------------------------------
        # Rapport
        # -------------------------------------------------

        self.report = {

            "initial_rows": initial_shape[0],
            "initial_columns": initial_shape[1],

            "final_rows": df.shape[0],
            "final_columns": df.shape[1],

            "duplicates_removed": duplicates,

            "empty_columns": empty_columns,

            "high_missing_columns":
                high_missing,

            "constant_columns":
                constant_columns,

            "remaining_missing":
                int(df.isna().sum().sum())
        }

        print("\n" + "-" * 50)
        print("RÉSULTAT")
        print("-" * 50)

        print(
            f"📊 Dimensions initiales : "
            f"{initial_shape[0]} × "
            f"{initial_shape[1]}"
        )

        print(
            f"📊 Dimensions finales : "
            f"{df.shape[0]} × "
            f"{df.shape[1]}"
        )

        print(
            f"❗ Valeurs manquantes restantes : "
            f"{df.isna().sum().sum()}"
        )

        print("\n✅ Nettoyage terminé.")

        return df