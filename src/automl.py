from src.optimizer import ModelOptimizer
from src.model_factory import ModelFactory


class AutoML:

    def __init__(
        self,
        problem_type,
        n_trials=10,
        cv=3,
        timeout_par_modele=None,
        modeles_exclus=None
    ):
        """
        Initialise le système AutoML.

        Parameters
        ----------
        problem_type : str
            Type de problème :
            - regression
            - classification_binaire
            - classification_multiclasse

        n_trials : int
            Nombre d'essais Optuna par modèle.

        cv : int
            Nombre de plis de validation croisée pendant la
            recherche d'hyperparamètres.

        timeout_par_modele : float | None
            Budget de temps (en secondes) accordé à l'optimisation
            de CHAQUE modèle. Permet de garder un temps de réponse
            maîtrisé sur les gros datasets, quel que soit n_trials.

        modeles_exclus : list[str] | None
            Modèles à ne pas tester (ex. les plus lents, sur de
            très gros volumes).
        """

        self.problem_type = problem_type
        self.n_trials = n_trials
        self.cv = cv
        self.timeout_par_modele = timeout_par_modele
        self.modeles_exclus = set(modeles_exclus or [])

        self.factory = ModelFactory()

        self.results = []
        self.champion = None

    # =========================================================
    # EXECUTION AUTOMATIQUE
    # =========================================================

    def run(
        self,
        X_train,
        y_train,
        X_test=None,
        y_test=None,
        X_recherche=None,
        y_recherche=None
    ):
        """
        Lance automatiquement tous les modèles disponibles,
        optimise leurs hyperparamètres et sélectionne le champion.

        X_recherche / y_recherche : sous-échantillon optionnel
        utilisé UNIQUEMENT pour la recherche d'hyperparamètres
        (plus rapide sur les gros datasets). Le modèle final est
        toujours entraîné sur X_train complet, en dehors de cette
        classe (voir man.py).
        """

        X_recherche = (
            X_recherche if X_recherche is not None else X_train
        )

        y_recherche = (
            y_recherche if y_recherche is not None else y_train
        )

        print("\n")
        print("=" * 70)
        print("                           AUTO ML")
        print("=" * 70)

        print(
            f"\nType de problème : {self.problem_type}"
        )

        print(
            f"Nombre de trials par modèle : {self.n_trials}"
        )

        if len(X_recherche) != len(X_train):

            print(
                f"🎯 Recherche d'hyperparamètres sur un "
                f"échantillon de {len(X_recherche)} lignes "
                f"(dataset complet : {len(X_train)} lignes)."
            )

        # =====================================================
        # RECUPERATION DES MODELES
        # =====================================================

        models = [
            nom
            for nom in self.factory.get_models(self.problem_type)
            if nom not in self.modeles_exclus
        ]

        print(
            f"Nombre de modèles disponibles : {len(models)}"
        )

        if self.modeles_exclus:

            print(
                f"Modèles exclus pour ce volume de données : "
                f"{sorted(self.modeles_exclus)}"
            )

        if not models:

            raise RuntimeError(
                f"Aucun modèle disponible pour "
                f"le problème : {self.problem_type}"
            )

        self.results = []

        # =====================================================
        # OPTIMISATION DE CHAQUE MODELE
        # =====================================================

        for index, model_name in enumerate(
            models,
            start=1
        ):

            print("\n")
            print("-" * 70)

            print(
                f"[{index}/{len(models)}] "
                f"Modèle : {model_name}"
            )

            print("-" * 70)

            try:

                optimizer = ModelOptimizer(
                    problem_type=self.problem_type,
                    n_trials=self.n_trials,
                    cv=self.cv,
                    timeout=self.timeout_par_modele
                )

                result = optimizer.optimize(
                    model_name,
                    X_recherche,
                    y_recherche
                )

                if result is None:
                    print(
                        f"Modèle ignoré : {model_name}"
                    )
                    continue

                self.results.append(result)

                print(
                    f"\n{model_name} terminé."
                )

                print(
                    f"Score CV : {result['score']:.4f}"
                )

            except Exception as e:

                print("\n")
                print("ATTENTION : modèle ignoré")
                print(
                    f"Modèle : {model_name}"
                )
                print(
                    f"Raison : {e}"
                )

        # =====================================================
        # VERIFICATION
        # =====================================================

        if not self.results:

            raise RuntimeError(
                "AutoML : aucun modèle n'a pu être entraîné."
            )

        # =====================================================
        # SELECTION DU CHAMPION
        # =====================================================

        self.champion = max(
            self.results,
            key=lambda result: result["score"]
        )

        champion = self.champion

        # =====================================================
        # AFFICHAGE DU CHAMPION
        # =====================================================

        print("\n")
        print("=" * 70)
        print("                         RESULTAT AUTO ML")
        print("=" * 70)

        print(
            f"\nMODELE CHAMPION : "
            f"{champion['model']}"
        )

        print(
            f"SCORE CV : "
            f"{champion['score']:.4f}"
        )

        # =====================================================
        # CLASSEMENT
        # =====================================================

        print("\nCLASSEMENT DES MODELES")

        print("-" * 70)

        # Trier du meilleur au moins bon

        classement = sorted(
            self.results,
            key=lambda result: result["score"],
            reverse=True
        )

        for index, result in enumerate(
            classement,
            start=1
        ):

            print(
                f"{index}. "
                f"{result['model']:<25}"
                f"Score = {result['score']:.4f}"
            )

        # =====================================================
        # PARAMETRES DU CHAMPION
        # =====================================================

        print("\nPARAMETRES DU MODELE CHAMPION")

        print("-" * 70)

        params = champion.get(
            "params",
            {}
        )

        if params:

            for key, value in params.items():

                print(
                    f"   {key} : {value}"
                )

        else:

            print(
                "   Aucun paramètre disponible."
            )

        print("\n" + "=" * 70)

        return self.results

    # =========================================================
    # RECUPERER LE CHAMPION
    # =========================================================

    def get_champion(self):

        """
        Retourne le meilleur modèle trouvé par AutoML.
        """

        if self.champion is not None:

            return self.champion

        if not self.results:

            return None

        self.champion = max(
            self.results,
            key=lambda result: result["score"]
        )

        return self.champion

    # =========================================================
    # RECUPERER LE MODELE
    # =========================================================

    def get_champion_model(self):

        """
        Retourne directement l'objet modèle champion.
        """

        champion = self.get_champion()

        if champion is None:

            return None

        return champion.get(
            "model_object"
        )

    # =========================================================
    # RECUPERER LE SCORE
    # =========================================================

    def get_champion_score(self):

        """
        Retourne le score du champion.
        """

        champion = self.get_champion()

        if champion is None:

            return None

        return champion.get(
            "score"
        )

    # =========================================================
    # RECUPERER LES PARAMETRES
    # =========================================================

    def get_champion_params(self):

        """
        Retourne les meilleurs hyperparamètres.
        """

        champion = self.get_champion()

        if champion is None:

            return {}

        return champion.get(
            "params",
            {}
        )

    # =========================================================
    # RESUME
    # =========================================================

    def summary(self):

        """
        Affiche un résumé de l'AutoML.
        """

        champion = self.get_champion()

        if champion is None:

            print(
                "\nAucun résultat AutoML disponible."
            )

            return

        print("\n")
        print("=" * 70)
        print("                         RESUME AUTO ML")
        print("=" * 70)

        print(
            f"\nType de problème : "
            f"{self.problem_type}"
        )

        print(
            f"Nombre de modèles testés : "
            f"{len(self.results)}"
        )

        print(
            f"Modèle champion : "
            f"{champion['model']}"
        )

        print(
            f"Score champion : "
            f"{champion['score']:.4f}"
        )

        print("\n" + "=" * 70)