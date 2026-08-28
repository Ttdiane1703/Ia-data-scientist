import shap
import numpy as np


class ModelExplainer:

    def __init__(self):

        self.explainer = None
        self.shap_values = None
        self.feature_names = None

    # ==================================================
    # EXPLICATION
    # ==================================================

    def explain(
        self,
        model,
        X
    ):

        print("\n" + "=" * 60)
        print("              SHAP EXPLAINABILITY")
        print("=" * 60)

        self.feature_names = list(
            X.columns
        )

        # ------------------------------------------------
        # Tree models
        # ------------------------------------------------

        try:

            self.explainer = (
                shap.TreeExplainer(
                    model
                )
            )

            self.shap_values = (
                self.explainer.shap_values(
                    X
                )
            )

        except Exception:

            print(
                "\n⚠️ TreeExplainer impossible."
            )

            print(
                "Tentative avec Explainer générique..."
            )

            self.explainer = (
                shap.Explainer(
                    model,
                    X
                )
            )

            self.shap_values = (
                self.explainer(
                    X
                )
            )

        self.print_importance()

        return self.shap_values

    # ==================================================
    # IMPORTANCE
    # ==================================================

    def print_importance(
        self
    ):

        values = self.shap_values

        # ----------------------------------------------
        # Classification multiclasses
        # ----------------------------------------------

        if isinstance(
            values,
            list
        ):

            values = np.array(
                values
            )

            values = np.abs(
                values
            ).mean(
                axis=0
            )

        # ----------------------------------------------
        # SHAP moderne
        # ----------------------------------------------

        elif hasattr(
            values,
            "values"
        ):

            values = values.values

            if values.ndim == 3:

                values = np.abs(
                    values
                ).mean(
                    axis=2
                )

        # ----------------------------------------------
        # Importance moyenne
        # ----------------------------------------------

        importance = np.abs(
            values
        ).mean(
            axis=0
        )

        ranking = sorted(
            zip(
                self.feature_names,
                importance
            ),
            key=lambda x: x[1],
            reverse=True
        )

        print(
            "\n📊 Variables les plus importantes :"
        )

        for index, (
            feature,
            value
        ) in enumerate(
            ranking[:15],
            start=1
        ):

            print(
                f"{index:2}. "
                f"{feature:<30} "
                f"{value:.6f}"
            )

        return ranking

    # ==================================================
    # EXPLICATION D'UNE LIGNE
    # ==================================================

    def explain_prediction(
        self,
        X_row
    ):

        if self.explainer is None:

            raise RuntimeError(
                "L'explainer n'est pas initialisé."
            )

        explanation = (
            self.explainer(
                X_row
            )
        )

        values = explanation.values

        if values.ndim == 3:

            values = values[0].mean(
                axis=1
            )

        elif values.ndim == 2:

            values = values[0]

        else:

            values = values.flatten()

        ranking = sorted(
            zip(
                self.feature_names,
                values
            ),
            key=lambda x: abs(x[1]),
            reverse=True
        )

        print(
            "\n🔎 EXPLICATION DE LA PRÉDICTION"
        )

        for feature, value in ranking[:10]:

            direction = (
                "↑"
                if value > 0
                else "↓"
            )

            print(
                f"{direction} "
                f"{feature:<30} "
                f"{value:+.6f}"
            )

        return ranking