"""
================================================================
VALIDATION AUTOMATIQUE DU DATASET AVANT ENTRAÎNEMENT
================================================================

Ce module vérifie, AVANT le train/test split, que le dataset est
suffisamment grand et suffisamment équilibré pour permettre une
séparation entraînement/test fiable.

Objectif : ne jamais laisser le pipeline planter (crash serveur,
exception sklearn non gérée) sur un dataset trop petit. À la place,
on renvoie un diagnostic clair, avec une recommandation, et on
permet à l'appelant (man.py) d'arrêter proprement le pipeline en
conservant tout ce qui a déjà été calculé.
================================================================
"""

import pandas as pd


# Seuils par défaut. Volontairement prudents : en dessous, une
# séparation train/test n'a statistiquement aucun sens.
MIN_TOTAL_ROWS = 10
MIN_OBS_PAR_CLASSE_BLOQUANT = 2
MIN_OBS_PAR_CLASSE_RECOMMANDE = 5


class DatasetValidator:
    """
    Valide qu'un dataset peut raisonnablement être séparé en
    train/test avant de lancer le reste du pipeline.
    """

    def __init__(
        self,
        min_total_rows=MIN_TOTAL_ROWS,
        min_obs_par_classe_bloquant=MIN_OBS_PAR_CLASSE_BLOQUANT,
        min_obs_par_classe_recommande=MIN_OBS_PAR_CLASSE_RECOMMANDE,
        test_size=0.20,
    ):
        self.min_total_rows = min_total_rows
        self.min_obs_par_classe_bloquant = min_obs_par_classe_bloquant
        self.min_obs_par_classe_recommande = min_obs_par_classe_recommande
        self.test_size = test_size

    def validate(self, df, target, problem_type):
        """
        Retourne un dictionnaire de diagnostic complet. Ne lève
        jamais d'exception : toute erreur interne est convertie en
        diagnostic bloquant avec un message explicite, pour ne
        jamais faire planter l'appelant.
        """

        try:
            return self._validate(df, target, problem_type)
        except Exception as e:
            return {
                "ok": False,
                "bloquant": True,
                "titre": "⚠️ Impossible de valider le dataset",
                "n_rows": int(len(df)) if df is not None else 0,
                "n_classes": None,
                "class_counts": None,
                "min_class_count": None,
                "can_stratify": False,
                "recommended_test_size": self.test_size,
                "messages": [
                    f"La validation automatique a rencontré une erreur "
                    f"inattendue : {e}"
                ],
                "recommandation": (
                    "Vérifiez le format de la colonne cible et "
                    "réessayez."
                ),
            }

    def _validate(self, df, target, problem_type):

        n_rows = int(len(df))

        est_classification = str(problem_type).startswith(
            "classification"
        )

        resultat = {
            "ok": True,
            "bloquant": False,
            "titre": "✅ Dataset validé",
            "n_rows": n_rows,
            "n_classes": None,
            "class_counts": None,
            "min_class_count": None,
            "can_stratify": False,
            "recommended_test_size": self.test_size,
            "messages": [],
            "recommandation": None,
        }

        # ------------------------------------------------------------
        # CAS 1 : dataset globalement trop petit, peu importe le type
        # ------------------------------------------------------------

        if n_rows < self.min_total_rows:

            resultat.update(
                {
                    "ok": False,
                    "bloquant": True,
                    "titre": "⚠️ Dataset trop petit",
                    "messages": [
                        f"Le dataset contient seulement {n_rows} "
                        f"observation(s), ce qui est insuffisant "
                        f"pour toute séparation entraînement/test "
                        f"fiable (minimum recommandé : "
                        f"{self.min_total_rows})."
                    ],
                    "recommandation": (
                        "Ajoutez davantage d'observations avant de "
                        "relancer l'analyse."
                    ),
                }
            )

            return resultat

        # ------------------------------------------------------------
        # CAS 2 : classification -> vérifier les classes
        # ------------------------------------------------------------

        if est_classification:

            y = df[target]

            class_counts = y.value_counts(dropna=False)

            n_classes = int(class_counts.shape[0])

            min_class_count = int(class_counts.min())

            resultat["n_classes"] = n_classes
            resultat["class_counts"] = {
                str(k): int(v) for k, v in class_counts.items()
            }
            resultat["min_class_count"] = min_class_count

            if n_classes < 2:

                resultat.update(
                    {
                        "ok": False,
                        "bloquant": True,
                        "titre": "⚠️ Une seule classe détectée",
                        "messages": [
                            f"La colonne cible '{target}' ne "
                            f"contient qu'une seule classe unique. "
                            f"Un modèle de classification a besoin "
                            f"d'au moins deux classes distinctes."
                        ],
                        "recommandation": (
                            "Vérifiez la colonne cible choisie, ou "
                            "ajoutez des observations couvrant "
                            "d'autres classes."
                        ),
                    }
                )

                return resultat

            if min_class_count < self.min_obs_par_classe_bloquant:

                resultat.update(
                    {
                        "ok": False,
                        "bloquant": True,
                        "titre": "⚠️ Dataset trop petit",
                        "messages": [
                            f"Le dataset contient seulement "
                            f"{n_rows} observations pour "
                            f"{n_classes} classes.",
                            f"La classe la moins représentée ne "
                            f"contient que {min_class_count} "
                            f"observation(s) : une séparation "
                            f"entraînement/test fiable n'est pas "
                            f"possible.",
                        ],
                        "recommandation": (
                            "Ajoutez davantage d'observations pour "
                            "la ou les classes minoritaires."
                        ),
                        "can_stratify": False,
                    }
                )

                return resultat

            # Peut être séparé, mais pas nécessairement de façon
            # confortable.

            resultat["can_stratify"] = True

            if min_class_count < self.min_obs_par_classe_recommande:

                resultat.update(
                    {
                        "ok": True,
                        "bloquant": False,
                        "titre": "⚠️ Dataset validé avec réserves",
                        "messages": [
                            f"La classe la moins représentée ne "
                            f"contient que {min_class_count} "
                            f"observations (recommandé : au moins "
                            f"{self.min_obs_par_classe_recommande}). "
                            f"Les métriques calculées sur cette "
                            f"classe seront peu fiables.",
                        ],
                        "recommandation": (
                            "Les résultats seront calculés, mais "
                            "restent à interpréter avec prudence "
                            "pour les classes minoritaires."
                        ),
                    }
                )

            # Vérifie que le test_size par défaut laisse au moins
            # une observation de chaque classe côté test.

            attendu_test = min_class_count * self.test_size

            if attendu_test < 1:

                test_size_ajuste = max(
                    round(1 / min_class_count, 2),
                    self.test_size,
                )

                test_size_ajuste = min(test_size_ajuste, 0.5)

                resultat["recommended_test_size"] = test_size_ajuste

                resultat["messages"].append(
                    f"Le test_size par défaut ({self.test_size}) "
                    f"ne garantirait pas au moins une observation "
                    f"de chaque classe dans le jeu de test. "
                    f"Ajustement automatique à "
                    f"{test_size_ajuste}."
                )

            return resultat

        # ------------------------------------------------------------
        # CAS 3 : régression -> juste une taille minimale raisonnable
        # ------------------------------------------------------------

        min_total_regression = max(self.min_total_rows, 10)

        min_train_reg = max(int(n_rows * (1 - self.test_size)), 1)
        min_test_reg = max(n_rows - min_train_reg, 0)

        if n_rows < min_total_regression or min_test_reg < 2:

            resultat.update(
                {
                    "ok": False,
                    "bloquant": True,
                    "titre": "⚠️ Dataset trop petit",
                    "messages": [
                        f"Le dataset contient seulement {n_rows} "
                        f"observations, ce qui ne permet pas une "
                        f"séparation entraînement/test fiable pour "
                        f"une régression."
                    ],
                    "recommandation": (
                        "Ajoutez davantage d'observations avant de "
                        "relancer l'analyse."
                    ),
                }
            )

            return resultat

        if n_rows < 30:

            resultat.update(
                {
                    "titre": "⚠️ Dataset validé avec réserves",
                    "messages": [
                        f"Le dataset ne contient que {n_rows} "
                        f"observations. Les métriques de "
                        f"régression (R², RMSE, MAE) seront "
                        f"instables sur un jeu de test aussi "
                        f"petit.",
                    ],
                    "recommandation": (
                        "Interprétez les résultats avec prudence "
                        "et privilégiez la validation croisée."
                    ),
                }
            )

        return resultat


def formater_panneau_validation(resultat):
    """
    Construit le texte affiché à l'utilisateur pour un résultat de
    validation, dans le format demandé par le cahier des charges.
    """

    lignes = []

    lignes.append(resultat["titre"])
    lignes.append("")

    for message in resultat.get("messages", []):
        lignes.append(message)

    if resultat.get("recommandation"):
        lignes.append("")
        lignes.append("Recommandation :")
        lignes.append(resultat["recommandation"])

    return "\n".join(lignes)
