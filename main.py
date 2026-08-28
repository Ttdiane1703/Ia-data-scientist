#!/usr/bin/env python3
"""
IA DATA SCIENTIST — point d'entrée principal.

Interface terminal permettant de charger un dataset, le profiler, le
nettoyer, l'explorer (EDA), choisir la ou les targets (toujours par
l'utilisateur), lancer l'AutoML/Optuna, comparer les modèles, générer
l'explicabilité, l'analyse des erreurs, faire des prédictions, et
produire un rapport final.
"""

from __future__ import annotations

import sys
import traceback
from typing import Any, Dict, List, Optional

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console()
    _HAS_RICH = True
except ImportError:
    console = None
    _HAS_RICH = False

from config.logging_config import get_logger, setup_logging
from config.settings import APP_NAME, DEBUG, OPTUNA_TRIALS, RAW_DATA_DIR, VERSION
from src.automl import AutoMLEngine
from src.cleaner import DataCleaner
from src.eda import EDAEngine
from src.error_analysis import analyze_classification_errors, analyze_regression_errors
from src.explainability import explain
from src.feature_engineering import FeatureEngineer
from src.loader import DataLoader, DataLoadError
from src.model_manager import ModelManager
from src.predictor import Predictor
from src.preprocessing import build_preprocessor
from src.problem_detector import ProblemDetector
from src.profiler import DataProfiler
from src.report_generator import ReportGenerator
from src.splitter import split_data
from src.target_selector import InvalidTargetError, TargetSelector

setup_logging()
logger = get_logger("main")


def _print(msg: str = "", style: Optional[str] = None) -> None:
    if _HAS_RICH:
        console.print(msg, style=style)
    else:
        print(msg)


def _print_title(title: str) -> None:
    if _HAS_RICH:
        console.print(Panel(title, style="bold cyan"))
    else:
        bar = "=" * 70
        print(f"\n{bar}\n{title.center(70)}\n{bar}")


def _print_table(headers: List[str], rows: List[List[str]]) -> None:
    if _HAS_RICH:
        table = Table(show_header=True, header_style="bold cyan")
        for h in headers:
            table.add_column(h)
        for row in rows:
            table.add_row(*[str(c) for c in row])
        console.print(table)
    else:
        print(" | ".join(headers))
        print("-" * 70)
        for row in rows:
            print(" | ".join(str(c) for c in row))


class AIDataScientistApp:
    """Orchestre le workflow complet de bout en bout."""

    def __init__(self) -> None:
        self.loader = DataLoader()
        self.profiler = DataProfiler()
        self.cleaner = DataCleaner()
        self.eda_engine = EDAEngine()
        self.feature_engineer = FeatureEngineer()
        self.automl_engine = AutoMLEngine()
        self.model_manager = ModelManager()
        self.report_generator = ReportGenerator()

        self.df = None
        self.dataset_summary: Dict[str, Any] = {}
        self.profile_report: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Étapes du workflow
    # ------------------------------------------------------------------
    def load_dataset(self, filepath: str) -> None:
        self.df = self.loader.load(filepath)
        self.dataset_summary = self.loader.summary(self.df)
        _print_title("CHARGEMENT")
        _print(f"✅ {self.dataset_summary['lignes']} lignes")
        _print(f"✅ {self.dataset_summary['colonnes']} colonnes")
        _print(f"Taille mémoire : {self.dataset_summary['taille_memoire']}")
        _print(f"Encodage : {self.dataset_summary['encodage']}")

    def profile_dataset(self) -> None:
        _print_title("PROFILAGE")
        self.profile_report = self.profiler.profile(self.df)
        _print(f"Numériques      : {self.profile_report['numeriques']}")
        _print(f"Catégorielles   : {self.profile_report['categorielles']}")
        _print(f"Dates           : {self.profile_report['dates']}")
        _print(f"Texte libre     : {self.profile_report['texte_libre']}")
        _print(f"Identifiants    : {self.profile_report['identifiants_potentiels']}")
        _print(f"Doublons        : {self.profile_report['doublons']}")

    def clean_dataset(self) -> None:
        _print_title("NETTOYAGE")
        self.df = self.cleaner.clean(self.df, id_like_cols=self.profile_report["identifiants_potentiels"])
        rows = [[a["colonne"], a["raison"], a["action"]] for a in self.cleaner.actions]
        if rows:
            _print_table(["Colonne", "Raison", "Action"], rows)
        else:
            _print("Aucune action de nettoyage nécessaire.")

    def run_eda(self, target: Optional[str] = None) -> Dict[str, Any]:
        _print_title("EDA AUTOMATIQUE")
        result = self.eda_engine.run(
            self.df,
            numeric_cols=self.profile_report["numeriques"],
            categorical_cols=self.profile_report["categorielles"],
            date_cols=self.profile_report["dates"],
            target=target,
        )
        _print(f"{len(result['fichiers'])} graphiques générés dans reports/eda/")
        return result

    def select_targets(self, raw_input: str) -> List[str]:
        selector = TargetSelector(self.df, id_like_cols=self.profile_report["identifiants_potentiels"])
        targets = selector.parse_selection(raw_input)
        return targets

    def suggest_targets(self) -> List[str]:
        selector = TargetSelector(self.df, id_like_cols=self.profile_report["identifiants_potentiels"])
        return selector.suggest()

    def process_target(self, target: str, all_targets: List[str], n_trials: int = OPTUNA_TRIALS) -> Dict[str, Any]:
        """Exécute le pipeline complet (ML) pour UNE target donnée."""
        _print_title(f"TARGET : {target}")

        detector = ProblemDetector()
        problem_info = detector.detect(self.df, target)
        _print(f"🎯 Target : {target}")
        _print(f"🧠 Problème : {problem_info.problem_type}")

        fe_result = self.feature_engineer.transform(
            self.df,
            targets=all_targets,
            date_cols=self.profile_report["dates"],
            id_like_cols=self.profile_report["identifiants_potentiels"],
            text_cols=self.profile_report["texte_libre"],
            numeric_cols=self.profile_report["numeriques"],
            categorical_cols=self.profile_report["categorielles"],
        )

        df_fe = fe_result.df.dropna(subset=[target])
        y = df_fe[target]
        feature_cols = fe_result.numeric_features + fe_result.categorical_features
        X = df_fe[feature_cols]

        X_train, X_test, y_train, y_test = split_data(X, y, problem_info.problem_type)

        preprocessor = build_preprocessor(fe_result.numeric_features, fe_result.categorical_features)

        automl_result = self.automl_engine.run(
            preprocessor,
            problem_info.problem_type,
            X_train,
            y_train,
            X_test,
            y_test,
            n_trials=n_trials,
        )

        rows = []
        for r in automl_result.results:
            status = "✅" if r.status == "success" else "❌"
            metric_str = ", ".join(f"{k}={v:.4f}" for k, v in r.metrics.items()) if r.metrics else (r.error or "")
            rows.append([r.name, status, metric_str])
        _print_table(["Modèle", "Statut", "Métriques"], rows)

        target_report: Dict[str, Any] = {
            "target": target,
            "problem_type": problem_info.problem_type,
            "modeles": [
                {"name": r.name, "status": r.status, "metrics": r.metrics, "error": r.error}
                for r in automl_result.results
            ],
        }

        if automl_result.champion is None:
            _print("❌ Aucun modèle n'a réussi pour cette target.", style="bold red")
            target_report["champion"] = None
            return target_report

        champion = automl_result.champion
        _print(f"\n🏆 CHAMPION : {champion.name}", style="bold green")
        _print(f"Métriques : {champion.metrics}")

        target_report["champion"] = {"name": champion.name, "metrics": champion.metrics, "params": champion.best_params}

        # Explicabilité
        explain_result = explain(champion.pipeline, X_test, target)
        target_report["explicabilite"] = explain_result

        # Analyse des erreurs
        y_pred = champion.pipeline.predict(X_test)
        if problem_info.problem_type in ("classification_binaire", "classification_multiclasse"):
            error_result = analyze_classification_errors(y_test, y_pred, target)
        else:
            error_result = analyze_regression_errors(y_test, y_pred, target)
        target_report["erreurs"] = error_result

        # Sauvegarde (avec les infos nécessaires pour rejouer le feature
        # engineering — ex. extraction de dates — sur de futures données brutes)
        raw_input_columns = [
            c
            for c in self.df.columns
            if c not in all_targets
            and c not in self.profile_report["identifiants_potentiels"]
            and c not in self.profile_report["texte_libre"]
        ]
        fe_info = {
            "all_targets": all_targets,
            "date_cols": self.profile_report["dates"],
            "id_like_cols": self.profile_report["identifiants_potentiels"],
            "text_cols": self.profile_report["texte_libre"],
            "numeric_cols": self.profile_report["numeriques"],
            "categorical_cols": self.profile_report["categorielles"],
            "raw_input_columns": raw_input_columns,
        }
        self.model_manager.save(
            target=target,
            problem_type=problem_info.problem_type,
            model_name=champion.name,
            pipeline=champion.pipeline,
            metrics=champion.metrics,
            features=feature_cols,
            dataset_shape=self.df.shape,
            feature_engineering_info=fe_info,
        )
        _print(f"💾 Modèle sauvegardé sous models/{target}/")

        return target_report

    def generate_final_report(self, target_reports: List[Dict[str, Any]]) -> Dict[str, str]:
        _print_title("RAPPORT FINAL")
        dataset_summary = {
            "fichier": self.dataset_summary.get("fichier"),
            "lignes": self.dataset_summary.get("lignes"),
            "colonnes": self.dataset_summary.get("colonnes"),
            "valeurs_manquantes": len(self.profile_report.get("valeurs_manquantes", {})),
            "doublons": self.profile_report.get("doublons"),
        }
        report = self.report_generator.build(dataset_summary, target_reports)
        paths = self.report_generator.save(report)
        _print(f"Rapports générés : {paths}")
        return paths


# --------------------------------------------------------------------- #
# CLI interactif
# --------------------------------------------------------------------- #
def main_menu() -> None:
    app = AIDataScientistApp()

    _print_title(f"{APP_NAME} v{VERSION}")
    _print("Bienvenue.\n")

    while True:
        _print("[1] Nouvelle analyse")
        _print("[2] Charger un projet (modèles existants)")
        _print("[3] Faire une prédiction")
        _print("[4] Voir les modèles sauvegardés")
        _print("[5] Quitter")
        choice = input("\nChoix : ").strip()

        try:
            if choice == "1":
                run_new_analysis(app)
            elif choice == "2":
                list_models(app)
            elif choice == "3":
                run_prediction(app)
            elif choice == "4":
                list_models(app)
            elif choice == "5":
                _print("Au revoir !")
                break
            else:
                _print("Choix invalide.")
        except (DataLoadError, InvalidTargetError) as exc:
            _print(f"❌ {exc}", style="bold red")
        except Exception as exc:  # noqa: BLE001
            _print(f"❌ Erreur inattendue : {exc}", style="bold red")
            if DEBUG:
                traceback.print_exc()


def run_new_analysis(app: AIDataScientistApp) -> None:
    default_path = RAW_DATA_DIR / "sales_data.csv"
    filepath = input(f"Chemin du fichier [{default_path}] : ").strip() or str(default_path)
    app.load_dataset(filepath)
    app.profile_dataset()
    app.clean_dataset()
    app.run_eda()

    _print_title("CHOIX DES TARGETS")
    for i, col in enumerate(app.df.columns, start=1):
        _print(f"[{i}] {col}")

    suggestions = app.suggest_targets()
    if suggestions:
        _print("\n💡 Suggestions :")
        for s in suggestions:
            _print(f"  - {s}")

    raw = input("\nQuelles variables voulez-vous prédire ? > ").strip()
    targets = app.select_targets(raw)

    n_trials_raw = input(f"Nombre de trials Optuna [{OPTUNA_TRIALS}] : ").strip()
    n_trials = int(n_trials_raw) if n_trials_raw.isdigit() else OPTUNA_TRIALS

    target_reports = [app.process_target(t, targets, n_trials=n_trials) for t in targets]
    app.generate_final_report(target_reports)


def list_models(app: AIDataScientistApp) -> None:
    models = app.model_manager.list_models()
    if not models:
        _print("Aucun modèle sauvegardé pour le moment.")
        return
    _print_title("MODÈLES SAUVEGARDÉS")
    for m in models:
        _print(f"- {m}")


def run_prediction(app: AIDataScientistApp) -> None:
    models = app.model_manager.list_models()
    if not models:
        _print("Aucun modèle disponible. Lancez d'abord une analyse.")
        return
    _print("Modèles disponibles : " + ", ".join(models))
    target = input("Nom de la target à utiliser : ").strip()
    pipeline, metadata = app.model_manager.load(target)
    predictor = Predictor(
        pipeline,
        metadata.get("problem_type", "regression"),
        target,
        feature_engineering_info=metadata.get("feature_engineering", {}),
    )

    mode = input("[1] Prédire depuis un CSV  [2] Saisie manuelle : ").strip()
    if mode == "1":
        path = input("Chemin du CSV : ").strip()
        new_df = app.loader.load(path)
        result = predictor.predict_dataframe(new_df)
        out_path = ReportGenerator().output_dir.parent / "predictions" / f"predictions_{target}.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(out_path, index=False)
        _print(f"🔮 Prédictions sauvegardées dans {out_path}")
    else:
        values: Dict[str, Any] = {}
        raw_columns = metadata.get("feature_engineering", {}).get("raw_input_columns") or metadata.get("features", [])
        for feature in raw_columns:
            values[feature] = input(f"{feature} : ").strip()
        output = predictor.predict_single(values)
        _print(f"🔮 PRÉDICTION : {output}")


if __name__ == "__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\nInterrompu par l'utilisateur.")
        sys.exit(0)
