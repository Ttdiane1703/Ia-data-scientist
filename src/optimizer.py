import numpy as np
import optuna

from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import make_scorer, f1_score

from src.model_factory import ModelFactory


class ModelOptimizer:

    def __init__(
        self,
        problem_type,
        n_trials=10,
        cv=3,
        random_state=42
    ):

        self.problem_type = problem_type
        self.n_trials = n_trials
        self.cv = cv
        self.random_state = random_state

        self.factory = ModelFactory()

        self.label_encoder = None

    # =========================================================
    # ENCODAGE DE LA TARGET
    # =========================================================

    def prepare_target(self, y):

        # Copie pour éviter de modifier la target originale
        y = y.copy()

        # -----------------------------------------------------
        # CLASSIFICATION
        # -----------------------------------------------------

        if self.problem_type.startswith("classification"):

            # Toujours encoder la target.
            #
            # Exemple :
            #
            # New       -> 0
            # Returning -> 1
            #
            # Cela évite les problèmes avec :
            #
            # XGBoost
            # LightGBM
            # F1
            # -------------------------------------------------

            self.label_encoder = LabelEncoder()

            y = self.label_encoder.fit_transform(
                y.astype(str)
            )

            # Conversion explicite en entier
            y = np.asarray(y, dtype=np.int64)

        else:

            # -------------------------------------------------
            # REGRESSION
            # -------------------------------------------------

            y = np.asarray(y, dtype=float)

        return y

    # =========================================================
    # ESPACE DES HYPERPARAMETRES
    #
    # IMPORTANT :
    #
    # Cette méthode ne doit renvoyer QUE les hyperparamètres
    # réellement optimisés par Optuna (n_estimators, max_depth,
    # learning_rate, etc.).
    #
    # Les paramètres techniques fixes (random_state, random_seed,
    # n_jobs, eval_metric, verbosity, verbose, ...) sont déjà
    # gérés de façon centralisée dans model_factory.py.
    #
    # Si on les renvoie ici aussi, ils se retrouvent dupliqués
    # lors de l'appel à create_model(**params), ce qui provoque :
    #
    # TypeError: ... got multiple values for keyword argument
    # 'random_state'
    #
    # et fait échouer TOUS les modèles.
    # =========================================================

    def suggest_parameters(
        self,
        trial,
        model_name
    ):

        # -----------------------------------------------------
        # RANDOM FOREST
        # -----------------------------------------------------

        if model_name == "RandomForest":

            return {

                "n_estimators":
                    trial.suggest_int(
                        "n_estimators",
                        100,
                        500,
                        step=50
                    ),

                "max_depth":
                    trial.suggest_int(
                        "max_depth",
                        3,
                        30
                    ),

                "min_samples_split":
                    trial.suggest_int(
                        "min_samples_split",
                        2,
                        10
                    ),

                "min_samples_leaf":
                    trial.suggest_int(
                        "min_samples_leaf",
                        1,
                        5
                    ),

                "max_features":
                    trial.suggest_categorical(
                        "max_features",
                        ["sqrt", "log2"]
                    )

                # random_state et n_jobs : gérés par model_factory.py
            }

        # -----------------------------------------------------
        # EXTRA TREES
        # -----------------------------------------------------

        if model_name == "ExtraTrees":

            return {

                "n_estimators":
                    trial.suggest_int(
                        "n_estimators",
                        100,
                        500,
                        step=50
                    ),

                "max_depth":
                    trial.suggest_int(
                        "max_depth",
                        3,
                        30
                    ),

                "min_samples_split":
                    trial.suggest_int(
                        "min_samples_split",
                        2,
                        10
                    ),

                "min_samples_leaf":
                    trial.suggest_int(
                        "min_samples_leaf",
                        1,
                        5
                    ),

                "max_features":
                    trial.suggest_categorical(
                        "max_features",
                        ["sqrt", "log2"]
                    )

                # random_state et n_jobs : gérés par model_factory.py
            }

        # -----------------------------------------------------
        # GRADIENT BOOSTING
        # -----------------------------------------------------

        if model_name == "GradientBoosting":

            return {

                "n_estimators":
                    trial.suggest_int(
                        "n_estimators",
                        50,
                        400,
                        step=50
                    ),

                "learning_rate":
                    trial.suggest_float(
                        "learning_rate",
                        0.01,
                        0.3,
                        log=True
                    ),

                "max_depth":
                    trial.suggest_int(
                        "max_depth",
                        2,
                        8
                    ),

                "subsample":
                    trial.suggest_float(
                        "subsample",
                        0.6,
                        1.0
                    )

                # random_state : géré par model_factory.py
            }

        # -----------------------------------------------------
        # XGBOOST
        # -----------------------------------------------------

        if model_name == "XGBoost":

            return {

                "n_estimators":
                    trial.suggest_int(
                        "n_estimators",
                        100,
                        500,
                        step=50
                    ),

                "max_depth":
                    trial.suggest_int(
                        "max_depth",
                        2,
                        10
                    ),

                "learning_rate":
                    trial.suggest_float(
                        "learning_rate",
                        0.01,
                        0.3,
                        log=True
                    ),

                "subsample":
                    trial.suggest_float(
                        "subsample",
                        0.6,
                        1.0
                    ),

                "colsample_bytree":
                    trial.suggest_float(
                        "colsample_bytree",
                        0.6,
                        1.0
                    )

                # random_state, n_jobs, eval_metric :
                # gérés par model_factory.py
            }

        # -----------------------------------------------------
        # LIGHTGBM
        # -----------------------------------------------------

        if model_name == "LightGBM":

            return {

                "n_estimators":
                    trial.suggest_int(
                        "n_estimators",
                        100,
                        500,
                        step=50
                    ),

                "learning_rate":
                    trial.suggest_float(
                        "learning_rate",
                        0.01,
                        0.2,
                        log=True
                    ),

                "num_leaves":
                    trial.suggest_int(
                        "num_leaves",
                        15,
                        100
                    ),

                "max_depth":
                    trial.suggest_int(
                        "max_depth",
                        3,
                        15
                    ),

                "subsample":
                    trial.suggest_float(
                        "subsample",
                        0.6,
                        1.0
                    )

                # random_state et verbosity : gérés par model_factory.py
            }

        # -----------------------------------------------------
        # CATBOOST
        # -----------------------------------------------------

        if model_name == "CatBoost":

            return {

                "iterations":
                    trial.suggest_int(
                        "iterations",
                        100,
                        500,
                        step=50
                    ),

                "depth":
                    trial.suggest_int(
                        "depth",
                        4,
                        10
                    ),

                "learning_rate":
                    trial.suggest_float(
                        "learning_rate",
                        0.01,
                        0.3,
                        log=True
                    ),

                "l2_leaf_reg":
                    trial.suggest_float(
                        "l2_leaf_reg",
                        1,
                        10
                    )

                # random_seed et verbose : gérés par model_factory.py
            }

        raise ValueError(
            f"Modèle inconnu : {model_name}"
        )

    # =========================================================
    # METRIQUE
    # =========================================================

    def get_scoring(self):

        # -----------------------------------------------------
        # REGRESSION
        # -----------------------------------------------------

        if self.problem_type == "regression":

            return "neg_root_mean_squared_error"

        # -----------------------------------------------------
        # CLASSIFICATION BINAIRE
        # -----------------------------------------------------

        if self.problem_type == "classification_binaire":

            # IMPORTANT :
            #
            # sklearn.f1 par défaut utilise :
            #
            # pos_label=1
            #
            # Ici notre LabelEncoder garantit :
            #
            # classe 0
            # classe 1
            #
            # On définit explicitement la métrique.

            return make_scorer(
                f1_score,
                average="binary",
                pos_label=1,
                zero_division=0
            )

        # -----------------------------------------------------
        # CLASSIFICATION MULTICLASSE
        # -----------------------------------------------------

        if self.problem_type == "classification_multiclasse":

            return make_scorer(
                f1_score,
                average="weighted",
                zero_division=0
            )

        raise ValueError(
            f"Type de problème inconnu : "
            f"{self.problem_type}"
        )

    # =========================================================
    # OPTIMISATION
    # =========================================================

    def optimize(
        self,
        model_name,
        X,
        y
    ):

        print("\n")
        print("=" * 60)
        print("                MODEL OPTIMIZER")
        print("=" * 60)

        print(
            f"\n🤖 Modèle : {model_name}"
        )

        print(
            f"🧠 Type : {self.problem_type}"
        )

        print(
            f"🔄 Trials : {self.n_trials}"
        )

        # -----------------------------------------------------
        # TARGET
        # -----------------------------------------------------

        y_encoded = self.prepare_target(y)

        print(
            "\n🎯 Target préparée"
        )

        print(
            f"Classes : "
            f"{np.unique(y_encoded).tolist()}"
        )

        if self.label_encoder is not None:

            print(
                "Mapping :"
            )

            for index, label in enumerate(
                self.label_encoder.classes_
            ):

                print(
                    f"   {label} → {index}"
                )

        # -----------------------------------------------------
        # VALIDATION DE LA TARGET
        # -----------------------------------------------------

        if self.problem_type.startswith(
            "classification"
        ):

            unique_classes = np.unique(
                y_encoded
            )

            print(
                f"\nNombre de classes : "
                f"{len(unique_classes)}"
            )

            if len(unique_classes) < 2:

                raise ValueError(
                    "La target contient moins de "
                    "deux classes."
                )

        # -----------------------------------------------------
        # CROSS VALIDATION
        # -----------------------------------------------------

        if self.problem_type.startswith(
            "classification"
        ):

            cv = StratifiedKFold(
                n_splits=self.cv,
                shuffle=True,
                random_state=self.random_state
            )

        else:

            cv = KFold(
                n_splits=self.cv,
                shuffle=True,
                random_state=self.random_state
            )

        scoring = self.get_scoring()

        # -----------------------------------------------------
        # COMPTEUR DE TRIALS VALIDES
        # -----------------------------------------------------

        valid_trials = 0

        # -----------------------------------------------------
        # OBJECTIVE
        # -----------------------------------------------------

        def objective(trial):

            nonlocal valid_trials

            # -------------------------------------------------
            # PARAMETRES
            # -------------------------------------------------

            params = self.suggest_parameters(
                trial,
                model_name
            )

            # -------------------------------------------------
            # MODELE
            # -------------------------------------------------

            model = self.factory.create_model(
                model_name,
                self.problem_type,
                params
            )

            try:

                # ---------------------------------------------
                # VALIDATION
                # ---------------------------------------------

                scores = cross_val_score(
                    model,
                    X,
                    y_encoded,
                    cv=cv,
                    scoring=scoring,
                    n_jobs=-1,
                    error_score="raise"
                )

                # ---------------------------------------------
                # SCORE MOYEN
                # ---------------------------------------------

                score = float(
                    np.mean(scores)
                )

                # ---------------------------------------------
                # REGRESSION
                #
                # sklearn retourne :
                #
                # -RMSE
                #
                # donc on inverse.
                # ---------------------------------------------

                if self.problem_type == "regression":

                    score = -score

                # ---------------------------------------------
                # VALIDATION SCORE
                # ---------------------------------------------

                if not np.isfinite(score):

                    raise ValueError(
                        f"Score non valide : {score}"
                    )

                valid_trials += 1

                print(
                    f"\n✅ Trial {trial.number}"
                    f" | Score = {score:.4f}"
                )

                return score

            except Exception as e:

                print(
                    f"\n❌ Trial {trial.number} échoué"
                )

                print(
                    f"   Erreur : {e}"
                )

                # On demande à Optuna d'abandonner
                # proprement ce trial.

                raise optuna.exceptions.TrialPruned(
                    str(e)
                )

        # -----------------------------------------------------
        # OPTUNA
        # -----------------------------------------------------

        study = optuna.create_study(
            direction="maximize"
        )

        study.optimize(
            objective,
            n_trials=self.n_trials,
            show_progress_bar=False
        )

        # -----------------------------------------------------
        # VERIFICATION
        # -----------------------------------------------------

        if valid_trials == 0:

            raise RuntimeError(
                f"Aucun trial valide pour "
                f"{model_name}."
            )

        # -----------------------------------------------------
        # MEILLEUR TRIAL
        # -----------------------------------------------------

        best_trial = study.best_trial

        best_params = best_trial.params

        best_score = float(
            best_trial.value
        )

        # -----------------------------------------------------
        # RESULTATS
        # -----------------------------------------------------

        print(
            "\n🏆 OPTIMISATION TERMINÉE"
        )

        print(
            f"Modèle : {model_name}"
        )

        print(
            f"Score : {best_score:.4f}"
        )

        print(
            f"Paramètres : {best_params}"
        )

        # -----------------------------------------------------
        # CREATION MODELE FINAL
        # -----------------------------------------------------

        best_model = self.factory.create_model(
            model_name,
            self.problem_type,
            best_params
        )

        # -----------------------------------------------------
        # RESULTAT
        # -----------------------------------------------------

        return {

            "model":
                model_name,

            "score":
                best_score,

            "params":
                best_params,

            "model_object":
                best_model,

            "label_encoder":
                self.label_encoder
        }