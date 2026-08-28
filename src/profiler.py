import pandas as pd


class DataProfiler:

    def profile(self, df):

        print("\n" + "=" * 50)
        print("          DATA PROFILER")
        print("=" * 50)

        print("\n📊 Dimensions")
        print(
            f"   Lignes : {df.shape[0]}"
        )
        print(
            f"   Colonnes : {df.shape[1]}"
        )

        print("\n📋 Colonnes")
        print(list(df.columns))

        print("\n🔤 Types")
        print(df.dtypes)

        print("\n❗ Valeurs manquantes")

        missing = df.isna().sum()

        print(missing)

        print(
            f"\n🗑️ Doublons : "
            f"{df.duplicated().sum()}"
        )

        print("\n📈 Statistiques")

        try:

            print(
                df.describe(
                    include="all"
                )
            )

        except Exception:

            print(
                df.describe()
            )

        return {
            "shape": df.shape,
            "columns": list(df.columns),
            "missing": missing.to_dict(),
            "duplicates": int(
                df.duplicated().sum()
            )
        }