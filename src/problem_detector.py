# ==============================================================
# DÉTECTEUR AUTOMATIQUE DU PROBLÈME
# IA DATA SCIENTIST
# ==============================================================

import pandas as pd
import numpy as np


class ProblemDetector:

    def __init__(self, verbose=True):
        self.verbose = verbose
        self.target = None
        self.problem_type = None
        self.features = []

    def _print(self, message=""):
        if self.verbose:
            print(message)

    # ==========================================================
    # DÉTECTION DU TYPE DE PROBLÈME
    # ==========================================================

    def _detect_problem_type(self, serie):

        nombre_unique = serie.nunique(dropna=True)

        # Catégorielle binaire
        if (
            pd.api.types.is_object_dtype(serie)
            or pd.api.types.is_string_dtype(serie)
            or pd.api.types.is_categorical_dtype(serie)
        ):

            if nombre_unique == 2:
                return "classification_binaire"

            if 2 < nombre_unique <= 20:
                return "classification_multiclasse"

            return None

        # Numérique
        if pd.api.types.is_numeric_dtype(serie):

            # Peu de valeurs distinctes :
            # peut être une classification
            if nombre_unique == 2:
                return "classification_binaire"

            if nombre_unique <= 10:
                return "classification_multiclasse"

            return "regression"

        return None

    # ==========================================================
    # SCORE DE CIBLE
    # ==========================================================

    def _score_target(self, df, colonne):

        serie = df[colonne]

        unique = serie.nunique(dropna=True)

        if unique <= 1:
            return -999

        score = 0

        nom = colonne.lower()

        # ------------------------------------------------------
        # Priorité aux noms typiques de cible
        # ------------------------------------------------------

        mots_cibles = [
            "target",
            "label",
            "class",
            "classe",
            "customer_type",
            "churn",
            "default",
            "fraud",
            "outcome",
            "status",
            "category",
            "segment"
        ]

        for mot in mots_cibles:
            if mot in nom:
                score += 10

        # ------------------------------------------------------
        # Classification binaire
        # ------------------------------------------------------

        if unique == 2:
            score += 8

        # ------------------------------------------------------
        # Classification petite
        # ------------------------------------------------------

        elif 3 <= unique <= 10:
            score += 5

        # ------------------------------------------------------
        # Régression
        # ------------------------------------------------------

        elif unique > 10:
            score += 1

        # ------------------------------------------------------
        # Colonnes qui sont presque certainement des IDs
        # ------------------------------------------------------

        if (
            "id" in nom
            or "code" in nom
            or "identifier" in nom
        ):
            score -= 15

        # ------------------------------------------------------
        # Dates : très rarement une cible
        # ------------------------------------------------------

        if "date" in nom or "time" in nom:
            score -= 10

        # ------------------------------------------------------
        # Quantité : généralement feature, pas target
        # ------------------------------------------------------

        if (
            "quantity" in nom
            or "quantite" in nom
            or "qty" in nom
        ):
            score -= 4

        # ------------------------------------------------------
        # Prix / montant / coût : généralement features
        # ------------------------------------------------------

        mots_feature = [
            "price",
            "prix",
            "amount",
            "montant",
            "cost",
            "cout",
            "discount",
            "remise"
        ]

        if any(mot in nom for mot in mots_feature):
            score -= 3

        return score

    # ==========================================================
    # DÉTECTION PRINCIPALE
    # ==========================================================

    def detect(self, df):

        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                "Le dataset doit être un pandas DataFrame."
            )

        if df.empty:
            raise ValueError(
                "Le dataset est vide."
            )

        self._print("")
        self._print("=" * 55)
        self._print("          DÉTECTEUR AUTOMATIQUE")
        self._print("=" * 55)

        candidats = []

        # ------------------------------------------------------
        # Analyse des colonnes
        # ------------------------------------------------------

        for colonne in df.columns:

            serie = df[colonne]

            type_probleme = self._detect_problem_type(
                serie
            )

            if type_probleme is None:
                continue

            score = self._score_target(
                df,
                colonne
            )

            candidats.append(
                {
                    "colonne": colonne,
                    "score": score,
                    "uniques": serie.nunique(dropna=True),
                    "type": type_probleme
                }
            )

        # ------------------------------------------------------
        # Tri
        # ------------------------------------------------------

        candidats.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        self._print("")
        self._print("🎯 CIBLES CANDIDATES")

        for candidat in candidats:

            self._print(
                f"   • {candidat['colonne']} "
                f"| score={candidat['score']} "
                f"| uniques={candidat['uniques']} "
                f"| type={candidat['type']}"
            )

        if not candidats:
            raise ValueError(
                "Aucune cible potentielle détectée."
            )

        # ------------------------------------------------------
        # Meilleure cible
        # ------------------------------------------------------

        meilleure = candidats[0]

        self.target = meilleure["colonne"]
        self.problem_type = meilleure["type"]

        # ------------------------------------------------------
        # Features
        # ------------------------------------------------------

        self.features = [
            colonne
            for colonne in df.columns
            if colonne != self.target
        ]

        self._print("")
        self._print(
            f"🎯 Cible détectée : {self.target}"
        )

        self._print(
            f"🧠 Type de problème : "
            f"{self.problem_type}"
        )

        self._print(
            f"📊 Features potentielles : "
            f"{len(self.features)}"
        )

        return {
            "target": self.target,
            "problem_type": self.problem_type,
            "features": self.features,
            "candidates": candidats
        }

    # ==========================================================
    # ALIAS
    # ==========================================================

    def detect_problem(self, df):
        return self.detect(df)

    # ==========================================================
    # GETTERS
    # ==========================================================

    def get_target(self):
        return self.target

    def get_problem_type(self):
        return self.problem_type

    def get_features(self):
        return self.features