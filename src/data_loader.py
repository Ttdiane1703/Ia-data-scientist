import pandas as pd
from pathlib import Path

from src.console import configure_console


class DataLoader:

    def load(self, file_path: str) -> pd.DataFrame:

        configure_console()

        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Fichier introuvable : {file_path}"
            )

        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)

        elif path.suffix.lower() in [".xlsx", ".xls"]:
            df = pd.read_excel(path)

        else:
            raise ValueError(
                "Format non supporté. Utilisez CSV ou Excel."
            )

        print("✅ Données chargées")
        print(f"📊 Lignes : {df.shape[0]}")
        print(f"📋 Colonnes : {df.shape[1]}")

        return df
