import os
import numpy as np
import pandas as pd


class ErrorAnalyzer:

    def __init__(self, output_dir="reports/errors"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def analyze(
        self,
        X_test,
        y_test,
        predictions,
        probabilities=None,
        original_data=None
    ):
        print("\n")
        print("=" * 70)
        print("              ERROR ANALYZER")
        print("=" * 70)

        y_true = np.asarray(y_test)
        y_pred = np.asarray(predictions)

        results = pd.DataFrame({
            "actual": y_true,
            "predicted": y_pred
        })

        results["correct"] = (
            results["actual"] == results["predicted"]
        )

        results["error"] = ~results["correct"]

        # ---------------------------------------------------------
        # PROBABILITÉS
        # ---------------------------------------------------------

        if probabilities is not None:

            probabilities = np.asarray(probabilities)

            if probabilities.ndim == 2:

                if probabilities.shape[1] == 2:
                    results["probability"] = probabilities[:, 1]
                else:
                    results["probability"] = probabilities.max(axis=1)

            else:
                results["probability"] = probabilities

        # ---------------------------------------------------------
        # STATISTIQUES GÉNÉRALES
        # ---------------------------------------------------------

        total = len(results)
        errors = int(results["error"].sum())
        correct = int(results["correct"].sum())

        accuracy = correct / total if total > 0 else 0
        error_rate = errors / total if total > 0 else 0

        print(f"\n📊 Nombre total de prédictions : {total}")
        print(f"✅ Prédictions correctes : {correct}")
        print(f"❌ Erreurs : {errors}")
        print(f"🎯 Accuracy : {accuracy:.4f}")
        print(f"📉 Taux d'erreur : {error_rate:.4f}")

        # ---------------------------------------------------------
        # MATRICE DES ERREURS
        # ---------------------------------------------------------

        print("\n📋 RÉPARTITION DES ERREURS")

        confusion = pd.crosstab(
            results["actual"],
            results["predicted"],
            rownames=["Réel"],
            colnames=["Prédit"]
        )

        print(confusion)

        confusion.to_csv(
            os.path.join(
                self.output_dir,
                "confusion_details.csv"
            )
        )

        # ---------------------------------------------------------
        # ERREURS PAR CLASSE
        # ---------------------------------------------------------

        print("\n🎯 ERREURS PAR CLASSE")

        class_errors = (
            results.groupby("actual")["error"]
            .agg(
                total="count",
                errors="sum"
            )
        )

        class_errors["error_rate"] = (
            class_errors["errors"] /
            class_errors["total"]
        )

        print(class_errors)

        class_errors.to_csv(
            os.path.join(
                self.output_dir,
                "errors_by_class.csv"
            )
        )

        # ---------------------------------------------------------
        # CAS LES PLUS DIFFICILES
        # ---------------------------------------------------------

        difficult_cases = None

        if "probability" in results.columns:

            difficult_cases = results.copy()

            difficult_cases["confidence"] = np.maximum(
                difficult_cases["probability"],
                1 - difficult_cases["probability"]
            )

            difficult_cases = difficult_cases.sort_values(
                "confidence"
            )

            difficult_cases.head(50).to_csv(
                os.path.join(
                    self.output_dir,
                    "most_uncertain_predictions.csv"
                ),
                index=False
            )

            print("\n⚠️ PRÉDICTIONS LES PLUS INCERTAINES")

            print(
                difficult_cases[
                    [
                        "actual",
                        "predicted",
                        "probability",
                        "confidence"
                    ]
                ].head(10)
            )

        # ---------------------------------------------------------
        # DATASET ORIGINAL
        # ---------------------------------------------------------

        if original_data is not None:

            original = original_data.reset_index(drop=True)
            results_reset = results.reset_index(drop=True)

            min_length = min(
                len(original),
                len(results_reset)
            )

            original = original.iloc[:min_length].copy()
            results_reset = results_reset.iloc[:min_length].copy()

            error_dataset = original[
                results_reset["error"].values
            ].copy()

            error_dataset.to_csv(
                os.path.join(
                    self.output_dir,
                    "error_dataset.csv"
                ),
                index=False
            )

            print(
                f"\n📁 Dataset des erreurs sauvegardé : "
                f"{len(error_dataset)} lignes"
            )

        # ---------------------------------------------------------
        # RAPPORT GLOBAL
        # ---------------------------------------------------------

        report = {
            "total_predictions": total,
            "correct_predictions": correct,
            "errors": errors,
            "accuracy": accuracy,
            "error_rate": error_rate,
            "class_errors": class_errors.to_dict()
        }

        print("\n💡 DIAGNOSTIC AUTOMATIQUE")

        if error_rate > 0.40:
            print(
                "❌ Le modèle produit beaucoup d'erreurs."
            )
            print(
                "👉 Les données ou la cible peuvent être "
                "peu prédictibles."
            )

        elif error_rate > 0.25:
            print(
                "⚠️ Le taux d'erreur est relativement élevé."
            )
            print(
                "👉 Une amélioration du Feature Engineering "
                "peut être nécessaire."
            )

        else:
            print(
                "✅ Le taux d'erreur est relativement faible."
            )

        if "probability" in results.columns:

            avg_confidence = results["confidence"].mean()

            print(
                f"🧠 Confiance moyenne : "
                f"{avg_confidence:.4f}"
            )

            if avg_confidence < 0.60:
                print(
                    "⚠️ Le modèle manque de confiance."
                )

        print("\n" + "=" * 70)
        print("              ANALYSE DES ERREURS TERMINÉE")
        print("=" * 70)

        return {
            "results": results,
            "class_errors": class_errors,
            "accuracy": accuracy,
            "error_rate": error_rate,
            "difficult_cases": difficult_cases
        }