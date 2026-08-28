import numpy as np
import pandas as pd

from sklearn.model_selection import cross_validate
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


class ModelEvaluator:

    def __init__(
        self,
        cv=5,
        random_state=42
    ):

        self.cv = cv
        self.random_state = random_state

        self.evaluation = {}

    # ==================================================
    # CLASSIFICATION
    # ==================================================

    def evaluate_classification(
        self,
        model,
        X_train,
        y_train,
        X_test,
        y_test
    ):

        print("\n" + "=" * 60)
        print("          ÉVALUATION CLASSIFICATION")
        print("=" * 60)

        # ------------------------------------------------
        # Prédictions
        # ------------------------------------------------

        y_pred = model.predict(
            X_test
        )

        # ------------------------------------------------
        # Rapport de classification
        # ------------------------------------------------

        report = classification_report(
            y_test,
            y_pred,
            output_dict=True,
            zero_division=0
        )

        report_df = pd.DataFrame(
            report
        ).transpose()

        # ------------------------------------------------
        # Matrice de confusion
        # ------------------------------------------------

        matrix = confusion_matrix(
            y_test,
            y_pred
        )

        # ------------------------------------------------
        # Cross Validation
        # ------------------------------------------------

        cv_results = cross_validate(
            model,
            X_train,
            y_train,
            cv=self.cv,
            scoring=[
                "accuracy",
                "precision_weighted",
                "recall_weighted",
                "f1_weighted"
            ],
            return_train_score=True,
            n_jobs=-1
        )

        train_f1 = (
            cv_results[
                "train_f1_weighted"
            ].mean()
        )

        validation_f1 = (
            cv_results[
                "test_f1_weighted"
            ].mean()
        )

        validation_std = (
            cv_results[
                "test_f1_weighted"
            ].std()
        )

        # ------------------------------------------------
        # Détection Overfitting
        # ------------------------------------------------

        gap = (
            train_f1
            - validation_f1
        )

        if gap > 0.15:

            overfitting = "élevé"

        elif gap > 0.08:

            overfitting = "modéré"

        else:

            overfitting = "faible"

        # ------------------------------------------------
        # Stabilité
        # ------------------------------------------------

        if validation_std < 0.03:

            stability = "excellente"

        elif validation_std < 0.07:

            stability = "bonne"

        elif validation_std < 0.12:

            stability = "moyenne"

        else:

            stability = "faible"

        # ------------------------------------------------
        # Score final
        # ------------------------------------------------

        test_f1 = (
            report.get(
                "weighted avg",
                {}
            ).get(
                "f1-score",
                0
            )
        )

        # ------------------------------------------------
        # Diagnostic
        # ------------------------------------------------

        if (
            test_f1 >= 0.90
            and overfitting == "faible"
            and stability in [
                "excellente",
                "bonne"
            ]
        ):

            quality = "excellente"

            decision = (
                "MODÈLE TRÈS PERFORMANT"
            )

        elif (
            test_f1 >= 0.80
            and overfitting != "élevé"
        ):

            quality = "bonne"

            decision = (
                "MODÈLE UTILISABLE"
            )

        elif test_f1 >= 0.70:

            quality = "moyenne"

            decision = (
                "MODÈLE À AMÉLIORER"
            )

        else:

            quality = "faible"

            decision = (
                "MODÈLE NON SATISFAISANT"
            )

        self.evaluation = {

            "type":
                "classification",

            "test_f1":
                test_f1,

            "train_f1":
                train_f1,

            "validation_f1":
                validation_f1,

            "validation_std":
                validation_std,

            "overfitting":
                overfitting,

            "stability":
                stability,

            "quality":
                quality,

            "decision":
                decision,

            "confusion_matrix":
                matrix,

            "classification_report":
                report_df
        }

        # ------------------------------------------------
        # Affichage
        # ------------------------------------------------

        self.print_classification_results()

        return self.evaluation

    # ==================================================
    # RÉGRESSION
    # ==================================================

    def evaluate_regression(
        self,
        model,
        X_train,
        y_train,
        X_test,
        y_test
    ):

        print("\n" + "=" * 60)
        print("            ÉVALUATION RÉGRESSION")
        print("=" * 60)

        # ------------------------------------------------
        # Prédictions
        # ------------------------------------------------

        y_pred = model.predict(
            X_test
        )

        # ------------------------------------------------
        # Métriques
        # ------------------------------------------------

        mae = mean_absolute_error(
            y_test,
            y_pred
        )

        mse = mean_squared_error(
            y_test,
            y_pred
        )

        rmse = np.sqrt(
            mse
        )

        r2 = r2_score(
            y_test,
            y_pred
        )

        # ------------------------------------------------
        # Cross Validation
        # ------------------------------------------------

        cv_results = cross_validate(
            model,
            X_train,
            y_train,
            cv=self.cv,
            scoring="r2",
            return_train_score=True,
            n_jobs=-1
        )

        train_r2 = (
            cv_results[
                "train_score"
            ].mean()
        )

        validation_r2 = (
            cv_results[
                "test_score"
            ].mean()
        )

        validation_std = (
            cv_results[
                "test_score"
            ].std()
        )

        # ------------------------------------------------
        # Overfitting
        # ------------------------------------------------

        gap = (
            train_r2
            - validation_r2
        )

        if gap > 0.15:

            overfitting = "élevé"

        elif gap > 0.08:

            overfitting = "modéré"

        else:

            overfitting = "faible"

        # ------------------------------------------------
        # Stabilité
        # ------------------------------------------------

        if validation_std < 0.03:

            stability = "excellente"

        elif validation_std < 0.07:

            stability = "bonne"

        elif validation_std < 0.12:

            stability = "moyenne"

        else:

            stability = "faible"

        # ------------------------------------------------
        # Qualité
        # ------------------------------------------------

        if (
            r2 >= 0.90
            and overfitting == "faible"
        ):

            quality = "excellente"

            decision = (
                "MODÈLE TRÈS PERFORMANT"
            )

        elif (
            r2 >= 0.75
            and overfitting != "élevé"
        ):

            quality = "bonne"

            decision = (
                "MODÈLE UTILISABLE"
            )

        elif r2 >= 0.50:

            quality = "moyenne"

            decision = (
                "MODÈLE À AMÉLIORER"
            )

        else:

            quality = "faible"

            decision = (
                "MODÈLE NON SATISFAISANT"
            )

        self.evaluation = {

            "type":
                "regression",

            "r2":
                r2,

            "mae":
                mae,

            "mse":
                mse,

            "rmse":
                rmse,

            "train_r2":
                train_r2,

            "validation_r2":
                validation_r2,

            "validation_std":
                validation_std,

            "overfitting":
                overfitting,

            "stability":
                stability,

            "quality":
                quality,

            "decision":
                decision
        }

        self.print_regression_results()

        return self.evaluation

    # ==================================================
    # MÉTHODE PRINCIPALE
    # ==================================================

    def evaluate(
        self,
        model,
        X_train,
        y_train,
        X_test,
        y_test,
        problem_type
    ):

        if problem_type in [
            "classification_binaire",
            "classification_multiclasse"
        ]:

            return self.evaluate_classification(
                model,
                X_train,
                y_train,
                X_test,
                y_test
            )

        elif problem_type == "regression":

            return self.evaluate_regression(
                model,
                X_train,
                y_train,
                X_test,
                y_test
            )

        else:

            raise ValueError(
                f"Type non supporté : "
                f"{problem_type}"
            )

    # ==================================================
    # AFFICHAGE CLASSIFICATION
    # ==================================================

    def print_classification_results(
        self
    ):

        e = self.evaluation

        print(
            f"\n🎯 F1 Test : "
            f"{e['test_f1']:.4f}"
        )

        print(
            f"📚 F1 Train : "
            f"{e['train_f1']:.4f}"
        )

        print(
            f"🔄 F1 Validation : "
            f"{e['validation_f1']:.4f}"
        )

        print(
            f"📉 Écart Train/Validation : "
            f"{e['train_f1'] - e['validation_f1']:.4f}"
        )

        print(
            f"⚠️ Overfitting : "
            f"{e['overfitting']}"
        )

        print(
            f"📊 Stabilité : "
            f"{e['stability']}"
        )

        print(
            f"🏅 Qualité : "
            f"{e['quality']}"
        )

        print(
            f"👉 Décision : "
            f"{e['decision']}"
        )

        print(
            "\n📊 Matrice de confusion :"
        )

        print(
            e["confusion_matrix"]
        )

    # ==================================================
    # AFFICHAGE RÉGRESSION
    # ==================================================

    def print_regression_results(
        self
    ):

        e = self.evaluation

        print(
            f"\n🎯 R² Test : "
            f"{e['r2']:.4f}"
        )

        print(
            f"📚 R² Train : "
            f"{e['train_r2']:.4f}"
        )

        print(
            f"🔄 R² Validation : "
            f"{e['validation_r2']:.4f}"
        )

        print(
            f"📉 MAE : "
            f"{e['mae']:.4f}"
        )

        print(
            f"📉 RMSE : "
            f"{e['rmse']:.4f}"
        )

        print(
            f"⚠️ Overfitting : "
            f"{e['overfitting']}"
        )

        print(
            f"📊 Stabilité : "
            f"{e['stability']}"
        )

        print(
            f"🏅 Qualité : "
            f"{e['quality']}"
        )

        print(
            f"👉 Décision : "
            f"{e['decision']}"
        )