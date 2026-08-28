# ==============================================================
# FEATURE ENGINEERING AUTOMATIQUE
# IA DATA SCIENTIST
# ==============================================================

import re
import warnings
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    FunctionTransformer
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Préparation automatique des données pour le Machine Learning.

    Fonctionnalités :
    - détection des variables numériques
    - détection des variables catégorielles
    - détection des dates
    - détection des identifiants
    - création automatique de variables temporelles
    - encodage OneHot des variables catégorielles
    - standardisation des variables numériques
    - gestion des valeurs manquantes
    - suppression sécurisée des identifiants
    """

    def __init__(
        self,
        scale_numeric=True,
        encode_categorical=True,
        create_date_features=True,
        remove_identifiers=True,
        verbose=True
    ):
        self.scale_numeric = scale_numeric
        self.encode_categorical = encode_categorical
        self.create_date_features = create_date_features
        self.remove_identifiers = remove_identifiers
        self.verbose = verbose

        self.numeric_columns_ = []
        self.categorical_columns_ = []
        self.date_columns_ = []
        self.text_columns_ = []
        self.identifier_columns_ = []

        self.original_columns_ = []
        self.final_columns_ = []

        self.preprocessor = None
        self.feature_names_ = []

    # ==========================================================
    # AFFICHAGE
    # ==========================================================

    def _print(self, message=""):
        if self.verbose:
            print(message)

    # ==========================================================
    # DÉTECTION DES DATES
    # ==========================================================

    def _is_date_column(self, series):
        """
        Détermine si une colonne texte semble être une date.
        """

        if not (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
        ):
            return False

        if len(series) == 0:
            return False

        sample = series.dropna().astype(str).head(100)

        if len(sample) == 0:
            return False

        # Vérification par mots-clés
        nom = str(series.name).lower()

        mots_date = [
            "date",
            "datetime",
            "timestamp",
            "jour",
            "day",
            "month",
            "year"
        ]

        if any(mot in nom for mot in mots_date):
            return True

        # Tentative de conversion
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            converted = pd.to_datetime(
                sample,
                errors="coerce",
                format="mixed"
            )

        taux_conversion = converted.notna().mean()

        return taux_conversion >= 0.80

    # ==========================================================
    # DÉTECTION DES IDENTIFIANTS
    # ==========================================================

    def _is_identifier(self, series):
        """
        Détecte les colonnes pouvant être des identifiants.
        """

        nom = str(series.name).lower()

        mots_id = [
            "id",
            "code",
            "identifier",
            "identifiant",
            "uuid"
        ]

        # Cas évident : Product_ID
        if any(
            re.search(rf"(^|_){re.escape(mot)}($|_)", nom)
            for mot in mots_id
        ):
            return True

        # Une colonne avec presque autant de valeurs uniques
        # que de lignes peut être un identifiant.
        nombre_unique = series.nunique(dropna=True)
        nombre_lignes = len(series)

        if nombre_lignes > 0:
            ratio = nombre_unique / nombre_lignes

            if ratio >= 0.95:
                return True

        return False

    # ==========================================================
    # CRÉATION DES VARIABLES DE DATE
    # ==========================================================

    def _create_date_features(self, X):
        """
        Transforme les dates en variables numériques utiles.
        """

        X = X.copy()

        for colonne in self.date_columns_:

            if colonne not in X.columns:
                continue

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                date = pd.to_datetime(
                    X[colonne],
                    errors="coerce",
                    format="mixed"
                )

            X[f"{colonne}_annee"] = date.dt.year
            X[f"{colonne}_mois"] = date.dt.month
            X[f"{colonne}_jour"] = date.dt.day
            X[f"{colonne}_jour_semaine"] = date.dt.dayofweek
            X[f"{colonne}_semaine"] = date.dt.isocalendar().week.astype(float)

            X[f"{colonne}_trimestre"] = date.dt.quarter
            X[f"{colonne}_fin_mois"] = date.dt.is_month_end.astype(float)
            X[f"{colonne}_debut_mois"] = date.dt.is_month_start.astype(float)

            # On retire la date originale
            X.drop(columns=[colonne], inplace=True)

        return X

    # ==========================================================
    # FIT
    # ==========================================================

    def fit(self, X, y=None):

        X = X.copy()

        self.original_columns_ = list(X.columns)

        self.numeric_columns_ = []
        self.categorical_columns_ = []
        self.date_columns_ = []
        self.text_columns_ = []
        self.identifier_columns_ = []

        # ------------------------------------------------------
        # Détection
        # ------------------------------------------------------

        for colonne in X.columns:

            serie = X[colonne]

            # Date
            if self.create_date_features and self._is_date_column(serie):
                self.date_columns_.append(colonne)
                continue

            # Identifiant
            if self.remove_identifiers and self._is_identifier(serie):
                self.identifier_columns_.append(colonne)
                continue

            # Numérique
            if pd.api.types.is_numeric_dtype(serie):
                self.numeric_columns_.append(colonne)
                continue

            # Catégorielle
            if (
                pd.api.types.is_object_dtype(serie)
                or pd.api.types.is_string_dtype(serie)
                or pd.api.types.is_categorical_dtype(serie)
                or pd.api.types.is_bool_dtype(serie)
            ):

                # Texte libre
                nombre_unique = serie.nunique(dropna=True)

                if (
                    nombre_unique > 50
                    and len(X) > 0
                    and nombre_unique / len(X) > 0.50
                ):
                    self.text_columns_.append(colonne)
                else:
                    self.categorical_columns_.append(colonne)

        # ------------------------------------------------------
        # IMPORTANT :
        # Les dates et IDs sont retirés AVANT la construction
        # du ColumnTransformer.
        # ------------------------------------------------------

        X_work = X.copy()

        # Suppression des IDs
        if self.identifier_columns_:
            X_work = X_work.drop(
                columns=[
                    c for c in self.identifier_columns_
                    if c in X_work.columns
                ],
                errors="ignore"
            )

        # Création des variables de dates
        if self.create_date_features:
            X_work = self._create_date_features(X_work)

        # Les dates originales n'existent plus
        # après _create_date_features.
        #
        # On reconstruit les listes de colonnes réellement
        # présentes dans X_work.

        self.numeric_columns_ = [
            c for c in self.numeric_columns_
            if c in X_work.columns
        ]

        self.categorical_columns_ = [
            c for c in self.categorical_columns_
            if c in X_work.columns
        ]

        self.text_columns_ = [
            c for c in self.text_columns_
            if c in X_work.columns
        ]

        # ------------------------------------------------------
        # Variables créées par les dates
        # ------------------------------------------------------

        for colonne in X_work.columns:

            if colonne not in self.numeric_columns_:
                if colonne not in self.categorical_columns_:
                    if colonne not in self.text_columns_:
                        if pd.api.types.is_numeric_dtype(
                            X_work[colonne]
                        ):
                            self.numeric_columns_.append(colonne)

        # ------------------------------------------------------
        # Pipelines
        # ------------------------------------------------------

        transformers = []

        # Numériques
        if self.numeric_columns_:

            numeric_steps = [
                (
                    "imputation",
                    SimpleImputer(strategy="median")
                )
            ]

            if self.scale_numeric:
                numeric_steps.append(
                    (
                        "standardisation",
                        StandardScaler()
                    )
                )

            numeric_pipeline = Pipeline(
                steps=numeric_steps
            )

            transformers.append(
                (
                    "numeric",
                    numeric_pipeline,
                    self.numeric_columns_
                )
            )

        # Catégorielles
        if self.categorical_columns_:

            categorical_pipeline = Pipeline(
                steps=[
                    (
                        "imputation",
                        SimpleImputer(
                            strategy="most_frequent"
                        )
                    ),
                    (
                        "encodage",
                        OneHotEncoder(
                            handle_unknown="ignore",
                            sparse_output=False
                        )
                    )
                ]
            )

            transformers.append(
                (
                    "categorical",
                    categorical_pipeline,
                    self.categorical_columns_
                )
            )

        # Texte
        # Pour le moment, les colonnes de texte libre sont
        # supprimées afin d'éviter de créer des milliers de
        # variables inutiles.
        #
        # Elles pourront être traitées plus tard avec TF-IDF.

        self.final_columns_ = (
            self.numeric_columns_
            + self.categorical_columns_
        )

        self.preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
            verbose_feature_names_out=True
        )

        self.preprocessor.fit(X_work)

        try:
            self.feature_names_ = list(
                self.preprocessor.get_feature_names_out()
            )
        except Exception:
            self.feature_names_ = []

        # ------------------------------------------------------
        # Affichage
        # ------------------------------------------------------

        self._print("")
        self._print("=" * 70)
        self._print("             FEATURE ENGINEERING")
        self._print("=" * 70)

        self._print(
            f"🔢 Numériques : {len(self.numeric_columns_)}"
        )

        self._print(
            f"🔤 Catégorielles : {len(self.categorical_columns_)}"
        )

        self._print(
            f"📅 Dates : {len(self.date_columns_)}"
        )

        self._print(
            f"📝 Texte : {len(self.text_columns_)}"
        )

        self._print(
            f"🆔 Identifiants : {len(self.identifier_columns_)}"
        )

        if self.identifier_columns_:
            self._print(
                f"🗑️ Identifiants ignorés : "
                f"{self.identifier_columns_}"
            )

        self._print("")

        return self

    # ==========================================================
    # TRANSFORM
    # ==========================================================

    def transform(self, X):

        X = X.copy()

        # ------------------------------------------------------
        # Supprimer les identifiants
        # ------------------------------------------------------

        if self.identifier_columns_:

            X = X.drop(
                columns=[
                    c
                    for c in self.identifier_columns_
                    if c in X.columns
                ],
                errors="ignore"
            )

        # ------------------------------------------------------
        # Dates
        # ------------------------------------------------------

        if self.create_date_features:
            X = self._create_date_features(X)

        # ------------------------------------------------------
        # IMPORTANT :
        # On s'assure que toutes les colonnes attendues
        # existent avant ColumnTransformer.transform().
        # ------------------------------------------------------

        for colonne in self.final_columns_:

            if colonne not in X.columns:

                # Pour une variable numérique
                if colonne in self.numeric_columns_:
                    X[colonne] = np.nan

                # Pour une variable catégorielle
                elif colonne in self.categorical_columns_:
                    X[colonne] = np.nan

        # ------------------------------------------------------
        # On garde uniquement les colonnes nécessaires.
        #
        # Cela évite notamment :
        # ValueError: columns are missing: {'Product_ID'}
        # ------------------------------------------------------

        colonnes_utiles = [
            c
            for c in self.final_columns_
            if c in X.columns
        ]

        X = X[colonnes_utiles]

        # ------------------------------------------------------
        # Transformation
        # ------------------------------------------------------

        resultat = self.preprocessor.transform(X)

        # DataFrame final
        try:
            resultat = pd.DataFrame(
                resultat,
                columns=self.feature_names_,
                index=X.index
            )
        except Exception:

            resultat = pd.DataFrame(
                resultat,
                index=X.index
            )

        return resultat

    # ==========================================================
    # FIT TRANSFORM
    # ==========================================================

    def fit_transform(self, X, y=None, **fit_params):

        self.fit(X, y)

        resultat = self.transform(X)

        self._print(
            f"🔢 Features après transformation : "
            f"{resultat.shape[1]}"
        )

        self._print("")
        self._print("✅ Feature Engineering préparé.")

        return resultat

    # ==========================================================
    # INFORMATIONS
    # ==========================================================

    def get_feature_names_out(self, input_features=None):

        if self.feature_names_:
            return np.asarray(
                self.feature_names_,
                dtype=object
            )

        if self.preprocessor is not None:

            try:
                return self.preprocessor.get_feature_names_out()
            except Exception:
                pass

        return np.asarray([], dtype=object)

    # ==========================================================
    # RÉSUMÉ
    # ==========================================================

    def summary(self):

        return {
            "colonnes_numeriques": self.numeric_columns_,
            "colonnes_categorielles": self.categorical_columns_,
            "colonnes_dates": self.date_columns_,
            "colonnes_texte": self.text_columns_,
            "colonnes_identifiants": self.identifier_columns_,
            "nombre_features_finales": len(self.feature_names_)
        }