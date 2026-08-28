import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class AutomaticEDA:

    def __init__(self, output_dir="reports/eda"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def analyze(self, df):

        print("\n" + "=" * 60)
        print("              AUTOMATIC EDA")
        print("=" * 60)

        print("\n📊 STRUCTURE")
        print(f"   Lignes : {df.shape[0]}")
        print(f"   Colonnes : {df.shape[1]}")

        # ------------------------------------------------
        # MISSING VALUES
        # ------------------------------------------------

        print("\n❗ VALEURS MANQUANTES")

        missing = df.isnull().sum()
        missing = missing[missing > 0]

        if len(missing) == 0:
            print("   Aucune valeur manquante.")
        else:
            for col, value in missing.items():
                print(f"   {col} : {value}")

        # ------------------------------------------------
        # NUMERIC
        # ------------------------------------------------

        print("\n🔢 VARIABLES NUMÉRIQUES")

        numeric = df.select_dtypes(include=np.number)

        for col in numeric.columns:

            print(f"\n   {col}")

            print(f"      Moyenne : {numeric[col].mean():.3f}")
            print(f"      Médiane : {numeric[col].median():.3f}")
            print(f"      Min : {numeric[col].min()}")
            print(f"      Max : {numeric[col].max()}")

        # ------------------------------------------------
        # CATEGORICAL
        # ------------------------------------------------

        print("\n🔤 VARIABLES CATÉGORIELLES")

        categorical = df.select_dtypes(
            include=["object", "category", "string"]
        )

        for col in categorical.columns:

            print(
                f"   {col} : "
                f"{df[col].nunique()} catégories"
            )

        # ------------------------------------------------
        # OUTLIERS
        # ------------------------------------------------

        print("\n⚠️ OUTLIERS")

        for col in numeric.columns:

            q1 = numeric[col].quantile(0.25)
            q3 = numeric[col].quantile(0.75)

            iqr = q3 - q1

            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            count = (
                (numeric[col] < lower)
                | (numeric[col] > upper)
            ).sum()

            if count > 0:

                print(
                    f"   • {col} : "
                    f"{count} valeurs"
                )

        # ------------------------------------------------
        # CORRELATIONS
        # ------------------------------------------------

        print("\n🔗 CORRÉLATIONS")

        if len(numeric.columns) >= 2:

            corr = numeric.corr()

            for i in range(len(corr.columns)):

                for j in range(i + 1, len(corr.columns)):

                    value = corr.iloc[i, j]

                    if abs(value) >= 0.7:

                        print(
                            f"   ⚡ "
                            f"{corr.columns[i]} ↔ "
                            f"{corr.columns[j]} : "
                            f"{value:.3f}"
                        )

        # ------------------------------------------------
        # GRAPHS
        # ------------------------------------------------

        print("\n📈 GÉNÉRATION DES GRAPHIQUES")

        for col in numeric.columns:

            try:

                plt.figure(figsize=(8, 5))

                plt.hist(
                    df[col].dropna(),
                    bins=30
                )

                plt.title(f"Distribution - {col}")
                plt.xlabel(col)
                plt.ylabel("Fréquence")

                path = os.path.join(
                    self.output_dir,
                    f"{col}_distribution.png"
                )

                plt.savefig(path)
                plt.close()

            except Exception as e:

                print(
                    f"⚠️ Graphique impossible "
                    f"pour {col}: {e}"
                )

        print(
            f"   📁 Graphiques : "
            f"{self.output_dir}"
        )

        print("\n✅ Analyse exploratoire terminée.")

        return {
            "rows": len(df),
            "columns": len(df.columns),
            "numeric_columns": list(numeric.columns),
            "categorical_columns": list(categorical.columns),
            "missing_values": missing.to_dict()
        }