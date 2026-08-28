# src/trainer.py

import pandas as pd

from sklearn.model_selection import train_test_split


class ModelTrainer:

    def __init__(
        self,
        feature_engineer=None,
        test_size=0.2,
        random_state=42
    ):

        self.feature_engineer = feature_engineer

        self.test_size = test_size
        self.random_state = random_state

        self.X_train = None
        self.X_test = None

        self.y_train = None
        self.y_test = None

    # ==================================================
    # PREPARATION COMPLETE
    # ==================================================

    def prepare_data(
        self,
        X,
        y,
        problem_type
    ):

        print("\n")
        print("=" * 70)
        print("                 MODEL TRAINER")
        print("=" * 70)

        # ------------------------------------------------
        # Vérification
        # ------------------------------------------------

        if X is None or len(X) == 0:

            raise ValueError(
                "X est vide."
            )

        if y is None or len(y) == 0:

            raise ValueError(
                "y est vide."
            )

        if len(X) != len(y):

            raise ValueError(
                "X et y n'ont pas le même nombre "
                "d'observations."
            )

        # ------------------------------------------------
        # Copie
        # ------------------------------------------------

        X = X.copy()
        y = y.copy()

        print(
            f"\n📊 Dataset : "
            f"{X.shape[0]} lignes × "
            f"{X.shape[1]} features"
        )

        print(
            f"🎯 Target : "
            f"{len(y)} observations"
        )

        print(
            f"🧠 Problème : "
            f"{problem_type}"
        )

        # =================================================
        # TRAIN / TEST SPLIT
        # =================================================

        print(
            "\n✂️ Séparation Train / Test..."
        )

        # ------------------------------------------------
        # Stratification classification
        # ------------------------------------------------

        stratify = None

        if problem_type in [
            "classification_binaire",
            "classification_multiclasse"
        ]:

            # Stratification uniquement si
            # chaque classe possède assez d'observations

            try:

                value_counts = (
                    y.value_counts()
                )

                if (
                    len(value_counts) > 1
                    and value_counts.min() >= 2
                ):

                    stratify = y

            except Exception:

                stratify = None

        # ------------------------------------------------
        # Split
        # ------------------------------------------------

        self.X_train, self.X_test, \
        self.y_train, self.y_test = (
            train_test_split(
                X,
                y,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=stratify
            )
        )

        print(
            f"\n📚 Train : "
            f"{self.X_train.shape[0]} lignes"
        )

        print(
            f"🧪 Test : "
            f"{self.X_test.shape[0]} lignes"
        )

        # =================================================
        # FEATURE ENGINEERING
        # =================================================

        if self.feature_engineer is not None:

            print(
                "\n⚙️ Application du "
                "Feature Engineering..."
            )

            # ------------------------------------------------
            # FIT UNIQUEMENT SUR TRAIN
            # ------------------------------------------------

            self.feature_engineer.fit(
                self.X_train,
                self.y_train,
                problem_type
            )

            # ------------------------------------------------
            # TRANSFORMATION TRAIN
            # ------------------------------------------------

            self.X_train = (
                self.feature_engineer.transform(
                    self.X_train
                )
            )

            # ------------------------------------------------
            # TRANSFORMATION TEST
            # ------------------------------------------------

            self.X_test = (
                self.feature_engineer.transform(
                    self.X_test
                )
            )

            print(
                f"\n🔧 Features après transformation : "
                f"{self.X_train.shape[1]}"
            )

        # =================================================
        # NETTOYAGE FINAL
        # =================================================

        self._validate_data()

        # =================================================
        # RÉSUMÉ
        # =================================================

        print("\n")
        print("-" * 70)

        print(
            "✅ DONNÉES PRÊTES POUR L'AUTO ML"
        )

        print(
            f"📚 X_train : "
            f"{self.X_train.shape}"
        )

        print(
            f"🧪 X_test : "
            f"{self.X_test.shape}"
        )

        print(
            f"🎯 y_train : "
            f"{self.y_train.shape}"
        )

        print(
            f"🎯 y_test : "
            f"{self.y_test.shape}"
        )

        print("-" * 70)

        return (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test
        )

    # ==================================================
    # VALIDATION
    # ==================================================

    def _validate_data(self):

        # ------------------------------------------------
        # Vérifier NaN
        # ------------------------------------------------

        if hasattr(
            self.X_train,
            "isna"
        ):

            train_nan = (
                self.X_train
                .isna()
                .sum()
                .sum()
            )

            test_nan = (
                self.X_test
                .isna()
                .sum()
                .sum()
            )

            if train_nan > 0:

                raise ValueError(
                    f"X_train contient "
                    f"{train_nan} NaN."
                )

            if test_nan > 0:

                raise ValueError(
                    f"X_test contient "
                    f"{test_nan} NaN."
                )

        # ------------------------------------------------
        # Vérifier infinis
        # ------------------------------------------------

        try:

            import numpy as np

            train_inf = np.isinf(
                self.X_train
                .select_dtypes(
                    include="number"
                )
            ).sum().sum()

            test_inf = np.isinf(
                self.X_test
                .select_dtypes(
                    include="number"
                )
            ).sum().sum()

            if train_inf > 0:

                raise ValueError(
                    "X_train contient des "
                    "valeurs infinies."
                )

            if test_inf > 0:

                raise ValueError(
                    "X_test contient des "
                    "valeurs infinies."
                )

        except TypeError:

            pass

    # ==================================================
    # GETTERS
    # ==================================================

    def get_train_data(self):

        return (
            self.X_train,
            self.y_train
        )

    def get_test_data(self):

        return (
            self.X_test,
            self.y_test
        )

    # ==================================================
    # RESUME
    # ==================================================

    def summary(self):

        print("\n")
        print("=" * 60)
        print("                  TRAINER SUMMARY")
        print("=" * 60)

        if self.X_train is not None:

            print(
                f"\n📚 Train : "
                f"{self.X_train.shape}"
            )

        if self.X_test is not None:

            print(
                f"🧪 Test : "
                f"{self.X_test.shape}"
            )

        if self.y_train is not None:

            print(
                f"🎯 y_train : "
                f"{self.y_train.shape}"
            )

        if self.y_test is not None:

            print(
                f"🎯 y_test : "
                f"{self.y_test.shape}"
            )

        print(
            "\n" + "=" * 60
        )