class ModelSelector:

    def __init__(self):

        self.selected_model = None
        self.selected_model_name = None
        self.ranking = []

    def select(
        self,
        candidates
    ):
        """
        candidates doit être une liste de dictionnaires :

        {
            "name": "...",
            "model": ...,
            "score": 0.85,
            "stability": 0.04,
            "overfitting": "faible"
        }
        """

        if not candidates:

            raise ValueError(
                "Aucun modèle disponible."
            )

        ranking = []

        for candidate in candidates:

            score = candidate.get(
                "score",
                0
            )

            stability = candidate.get(
                "stability",
                1
            )

            overfitting = candidate.get(
                "overfitting",
                "élevé"
            )

            # ------------------------------------------
            # Pénalité stabilité
            # ------------------------------------------

            stability_penalty = (
                stability * 0.5
            )

            # ------------------------------------------
            # Pénalité overfitting
            # ------------------------------------------

            if overfitting == "faible":

                overfitting_penalty = 0

            elif overfitting == "modéré":

                overfitting_penalty = 0.05

            else:

                overfitting_penalty = 0.15

            # ------------------------------------------
            # Score final
            # ------------------------------------------

            final_score = (
                score
                - stability_penalty
                - overfitting_penalty
            )

            candidate_copy = candidate.copy()

            candidate_copy[
                "final_score"
            ] = final_score

            ranking.append(
                candidate_copy
            )

        # ----------------------------------------------
        # Classement
        # ----------------------------------------------

        ranking.sort(
            key=lambda x: x["final_score"],
            reverse=True
        )

        self.ranking = ranking

        best = ranking[0]

        self.selected_model = (
            best["model"]
        )

        self.selected_model_name = (
            best["name"]
        )

        print("\n" + "=" * 60)
        print("             MODEL SELECTOR")
        print("=" * 60)

        print("\n🏆 Classement des modèles :")

        for index, model in enumerate(
            ranking,
            start=1
        ):

            print(
                f"\n{index}. "
                f"{model['name']}"
            )

            print(
                f"   Score       : "
                f"{model['score']:.4f}"
            )

            print(
                f"   Stabilité   : "
                f"{model['stability']:.4f}"
            )

            print(
                f"   Overfitting : "
                f"{model['overfitting']}"
            )

            print(
                f"   Score final : "
                f"{model['final_score']:.4f}"
            )

        print(
            f"\n🥇 Modèle sélectionné : "
            f"{self.selected_model_name}"
        )

        return self.selected_model