class ModelFactory:

    # =========================================================
    # NETTOYAGE DES PARAMETRES
    #
    # Garde-fou centralisé : quel que soit l'appelant, on retire
    # ici toute clé technique (random_state, random_seed, n_jobs,
    # eval_metric, verbosity, verbose) qui serait déjà présente
    # dans "params", AVANT de la fixer explicitement plus bas.
    #
    # Cela évite définitivement :
    #
    # TypeError: ... got multiple values for keyword argument
    # 'random_state'
    #
    # même si un futur appelant (optimizer.py, un script de test,
    # etc.) venait à réintroduire une de ces clés par erreur.
    # =========================================================

    RESERVED_KEYS = (
        "random_state",
        "random_seed",
        "n_jobs",
        "eval_metric",
        "verbosity",
        "verbose",
        "allow_writing_files",
    )

    def clean_model_params(self, params):

        params = params or {}

        return {
            key: value
            for key, value in params.items()
            if key not in self.RESERVED_KEYS
        }

    def get_models(self, problem_type):

        if problem_type == "regression":

            return [
                "RandomForest",
                "ExtraTrees",
                "GradientBoosting",
                "XGBoost",
                "LightGBM",
                "CatBoost"
            ]

        return [
            "RandomForest",
            "ExtraTrees",
            "GradientBoosting",
            "XGBoost",
            "LightGBM",
            "CatBoost"
        ]

    def create_model(
        self,
        name,
        problem_type,
        params=None
    ):

        params = self.clean_model_params(params)

        if problem_type == "regression":

            if name == "RandomForest":

                from sklearn.ensemble import RandomForestRegressor

                return RandomForestRegressor(
                    random_state=42,
                    n_jobs=-1,
                    **params
                )

            if name == "ExtraTrees":

                from sklearn.ensemble import ExtraTreesRegressor

                return ExtraTreesRegressor(
                    random_state=42,
                    n_jobs=-1,
                    **params
                )

            if name == "GradientBoosting":

                from sklearn.ensemble import GradientBoostingRegressor

                return GradientBoostingRegressor(
                    random_state=42,
                    **params
                )

            if name == "XGBoost":

                from xgboost import XGBRegressor

                return XGBRegressor(
                    random_state=42,
                    n_jobs=-1,
                    verbosity=0,
                    **params
                )

            if name == "LightGBM":

                from lightgbm import LGBMRegressor

                return LGBMRegressor(
                    random_state=42,
                    verbosity=-1,
                    n_jobs=-1,
                    **params
                )

            if name == "CatBoost":

                from catboost import CatBoostRegressor

                return CatBoostRegressor(
                    random_seed=42,
                    verbose=False,
                    allow_writing_files=False,
                    **params
                )

        else:

            if name == "RandomForest":

                from sklearn.ensemble import RandomForestClassifier

                return RandomForestClassifier(
                    random_state=42,
                    n_jobs=-1,
                    **params
                )

            if name == "ExtraTrees":

                from sklearn.ensemble import ExtraTreesClassifier

                return ExtraTreesClassifier(
                    random_state=42,
                    n_jobs=-1,
                    **params
                )

            if name == "GradientBoosting":

                from sklearn.ensemble import GradientBoostingClassifier

                return GradientBoostingClassifier(
                    random_state=42,
                    **params
                )

            if name == "XGBoost":

                from xgboost import XGBClassifier

                return XGBClassifier(
                    random_state=42,
                    n_jobs=-1,
                    eval_metric="logloss",
                    verbosity=0,
                    **params
                )

            if name == "LightGBM":

                from lightgbm import LGBMClassifier

                return LGBMClassifier(
                    random_state=42,
                    verbosity=-1,
                    n_jobs=-1,
                    **params
                )

            if name == "CatBoost":

                from catboost import CatBoostClassifier

                return CatBoostClassifier(
                    random_seed=42,
                    verbose=False,
                    allow_writing_files=False,
                    **params
                )

        raise ValueError(
            f"Modèle inconnu : {name}"
        )