import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Au-delà de ce nombre de lignes, les graphiques sont générés à
# partir d'un échantillon représentatif plutôt que du dataset
# complet. Les statistiques (moyenne, médiane, corrélations, ...)
# restent, elles, toujours calculées sur le dataset complet.
SEUIL_ECHANTILLONNAGE_GRAPHIQUES = 10000

TAILLE_ECHANTILLON_GRAPHIQUES = 5000


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
        # ECHANTILLON REPRESENTATIF POUR LES GRAPHIQUES
        #
        # Dataset complet -> statistiques globales -> échantillon
        # représentatif pour les graphiques lourds ->
        # visualisations. Les statistiques ci-dessous sont TOUJOURS
        # calculées sur df (dataset complet) ; seule la génération
        # des graphiques utilise df_graphiques.
        # ------------------------------------------------

        if len(df) > SEUIL_ECHANTILLONNAGE_GRAPHIQUES:

            df_graphiques = df.sample(
                n=TAILLE_ECHANTILLON_GRAPHIQUES,
                random_state=42
            )

            print(
                f"\nℹ️ Dataset volumineux ({len(df)} lignes) : "
                f"les graphiques utilisent un échantillon "
                f"représentatif de {len(df_graphiques)} lignes. "
                f"Les statistiques restent calculées sur "
                f"l'ensemble du dataset."
            )

        else:

            df_graphiques = df

        # ------------------------------------------------
        # MISSING VALUES (dataset complet)
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
        # NUMERIC (dataset complet)
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
        # CATEGORICAL (dataset complet)
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
        # OUTLIERS (dataset complet)
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
        # CORRELATIONS (dataset complet)
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
        # GRAPHS (échantillon représentatif si gros dataset)
        # ------------------------------------------------

        print("\n📈 GÉNÉRATION DES GRAPHIQUES")

        n_colonnes_numeriques = len(numeric.columns)

        for index, col in enumerate(numeric.columns, start=1):

            try:

                plt.figure(figsize=(8, 5))

                plt.hist(
                    df_graphiques[col].dropna(),
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

            # Progression : évite un silence total pendant les
            # traitements longs sur les datasets avec beaucoup de
            # colonnes numériques.
            if n_colonnes_numeriques >= 20 and (
                index % 10 == 0 or index == n_colonnes_numeriques
            ):

                print(
                    f"   ... {index}/{n_colonnes_numeriques} "
                    f"graphiques générés"
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
            "missing_values": missing.to_dict(),
            "echantillon_graphiques": (
                len(df_graphiques) if len(df_graphiques) != len(df)
                else None
            )
        }