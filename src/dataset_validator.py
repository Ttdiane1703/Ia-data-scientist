"""
================================================================
VALIDATION AUTOMATIQUE DU DATASET AVANT ENTRAÎNEMENT
================================================================

Ce module vérifie, AVANT le train/test split, que le dataset est
suffisamment grand et suffisamment équilibré pour permettre une
séparation entraînement/test fiable.

Il gère intelligemment deux problèmes fréquents plutôt que de
tout bloquer au moindre souci :

  1. Lignes où la cible est manquante (NaN) : elles sont toujours
     retirées (impossible d'entraîner un modèle supervisé sans
     cible connue), quelle que soit la taille du dataset.

  2. Classes ultra-rares (souvent des erreurs de saisie, ex. une
     valeur de durée "74 min" qui se retrouve dans une colonne de
     catégorie) : si le dataset reste suffisamment grand une fois
     ces quelques lignes retirées, on les exclut automatiquement
     et on continue, au lieu de rejeter tout le dataset à cause de
     quelques lignes.

Le pipeline n'est arrêté ("bloquant") que lorsque, même après ce
nettoyage, il ne reste vraiment pas assez de données pour une
séparation entraînement/test fiable.
================================================================
"""

import pandas as pd


# Au-delà de ce nombre de catégories distinctes, une classification
# n'a plus vraiment de sens statistique (trop peu d'exemples par
# classe, quel que soit la taille du dataset). Une colonne cible
# choisie manuellement avec, par exemple, 4 500 valeurs différentes
# (un nom de réalisateur, un titre, un identifiant...) doit être
# refusée avec un message clair plutôt que de produire des milliers
# de classes à une seule observation.
MAX_CLASSES_CLASSIFICATION = 50

# Au-delà de ce ratio (nombre de classes / nombre de lignes), la
# colonne ressemble à un identifiant ou à du texte libre plutôt
# qu'à une catégorie à prédire.
RATIO_IDENTIFIANT = 0.5


# Seuils par défaut. Volontairement prudents : en dessous, une
# séparation train/test n'a statistiquement aucun sens.
MIN_TOTAL_ROWS = 10
MIN_OBS_PAR_CLASSE_BLOQUANT = 2
MIN_OBS_PAR_CLASSE_RECOMMANDE = 5


class DatasetValidator:
    """
    Valide qu'un dataset peut raisonnablement être séparé en
    train/test avant de lancer le reste du pipeline, et renvoie
    au passage une version filtrée du dataset (cible manquante et
    classes ultra-rares retirées) prête à être utilisée par la
    suite du pipeline.
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

    def validate(self, df, target, problem_type, df_original=None):
        """
        Retourne un dictionnaire de diagnostic complet, avec en
        plus une clé "dataset_filtre" contenant le DataFrame à
        utiliser pour la suite du pipeline (identique à `df` si
        rien n'a dû être retiré). Ne lève jamais d'exception :
        toute erreur interne est convertie en diagnostic bloquant
        avec un message explicite.

        df_original : le dataset AVANT nettoyage/imputation, s'il
        est disponible. Le nettoyage automatique du pipeline
        (imputation par médiane/mode) intervient AVANT que la
        cible ne soit choisie, et peut donc avoir déjà comblé des
        valeurs de cible manquantes avec de fausses valeurs. Quand
        df_original est fourni, c'est LUI qui sert de référence
        pour détecter les véritables lignes à cible manquante,
        pas `df` (potentiellement déjà imputé).
        """

        try:
            return self._validate(
                df, target, problem_type, df_original
            )
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
                "lignes_cible_manquante": 0,
                "classes_exclues": [],
                "dataset_filtre": None,
                "messages": [
                    f"La validation automatique a rencontré une erreur "
                    f"inattendue : {e}"
                ],
                "recommandation": (
                    "Vérifiez le format de la colonne cible et "
                    "réessayez."
                ),
            }

    def _resultat_de_base(self, n_rows):

        return {
            "ok": True,
            "bloquant": False,
            "titre": "✅ Dataset validé",
            "n_rows": n_rows,
            "n_classes": None,
            "class_counts": None,
            "min_class_count": None,
            "can_stratify": False,
            "recommended_test_size": self.test_size,
            "lignes_cible_manquante": 0,
            "classes_exclues": [],
            "dataset_filtre": None,
            "messages": [],
            "recommandation": None,
        }

    def _validate(self, df, target, problem_type, df_original=None):

        est_classification = str(problem_type).startswith(
            "classification"
        )

        resultat = self._resultat_de_base(int(len(df)))

        # ------------------------------------------------------------
        # ETAPE 0 : retirer les lignes sans valeur cible connue.
        #
        # Un modèle supervisé ne peut de toute façon rien apprendre
        # d'une ligne dont la cible est inconnue : ces lignes sont
        # toujours retirées, quelle que soit la taille du dataset.
        #
        # IMPORTANT : on vérifie la cible manquante sur df_original
        # quand il est fourni, car le nettoyage automatique du
        # pipeline impute déjà les valeurs manquantes (médiane /
        # mode) AVANT que la cible ne soit choisie — `df` peut donc
        # déjà contenir de fausses valeurs à la place des vrais NaN.
        # ------------------------------------------------------------

        if (
            df_original is not None
            and target in df_original.columns
        ):

            cible_originale = df_original.reindex(df.index)[target]

        else:

            cible_originale = df[target]

        masque_cible_connue = cible_originale.notna()

        n_cible_manquante = int((~masque_cible_connue).sum())

        df_travail = df

        if n_cible_manquante > 0:

            df_travail = df.loc[masque_cible_connue].copy()

            resultat["lignes_cible_manquante"] = n_cible_manquante

            resultat["messages"].append(
                f"{n_cible_manquante} ligne(s) sans valeur pour la "
                f"cible '{target}' ont été retirée(s) : impossible "
                f"d'entraîner un modèle supervisé sans cible connue."
            )

        n_rows = int(len(df_travail))

        resultat["n_rows"] = n_rows

        # ------------------------------------------------------------
        # CAS 1 : dataset globalement trop petit, peu importe le type
        # ------------------------------------------------------------

        if n_rows < self.min_total_rows:

            resultat.update(
                {
                    "ok": False,
                    "bloquant": True,
                    "titre": "⚠️ Dataset trop petit",
                    "dataset_filtre": df_travail,
                }
            )

            resultat["messages"].append(
                f"Il ne reste que {n_rows} observation(s) exploitable"
                f"(s), ce qui est insuffisant pour toute séparation "
                f"entraînement/test fiable (minimum recommandé : "
                f"{self.min_total_rows})."
            )

            resultat["recommandation"] = (
                "Ajoutez davantage d'observations avant de relancer "
                "l'analyse."
            )

            return resultat

        # ------------------------------------------------------------
        # CAS 2 : classification -> vérifier les classes
        # ------------------------------------------------------------

        if est_classification:

            return self._valider_classification(
                df_travail, target, resultat
            )

        # ------------------------------------------------------------
        # CAS 3 : régression -> juste une taille minimale raisonnable
        # ------------------------------------------------------------

        return self._valider_regression(df_travail, resultat)

    # ----------------------------------------------------------------
    # CLASSIFICATION
    # ----------------------------------------------------------------

    def _valider_classification(self, df_travail, target, resultat):

        n_rows = int(len(df_travail))

        y = df_travail[target]

        class_counts = y.value_counts(dropna=True)

        n_classes = int(class_counts.shape[0])

        if n_classes < 2:

            resultat.update(
                {
                    "ok": False,
                    "bloquant": True,
                    "titre": "⚠️ Une seule classe détectée",
                    "dataset_filtre": df_travail,
                }
            )

            resultat["messages"].append(
                f"La colonne cible '{target}' ne contient qu'une "
                f"seule classe unique. Un modèle de classification a "
                f"besoin d'au moins deux classes distinctes."
            )

            resultat["recommandation"] = (
                "Vérifiez la colonne cible choisie, ou ajoutez des "
                "observations couvrant d'autres classes."
            )

            return resultat

        # ------------------------------------------------------------
        # Cible avec beaucoup trop de catégories pour une
        # classification fiable (identifiant, texte libre, nom
        # propre à très forte cardinalité...). Ce n'est PAS un
        # problème de taille de dataset : même un dataset énorme ne
        # rendrait pas cette colonne exploitable telle quelle.
        # ------------------------------------------------------------

        if n_classes > MAX_CLASSES_CLASSIFICATION:

            ratio_unique = n_classes / n_rows if n_rows else 1

            resultat.update(
                {
                    "ok": False,
                    "bloquant": True,
                    "n_classes": n_classes,
                    "dataset_filtre": df_travail,
                }
            )

            if ratio_unique >= RATIO_IDENTIFIANT:

                resultat["titre"] = (
                    "⚠️ Cette colonne ressemble à un identifiant, "
                    "pas à une catégorie"
                )

                resultat["messages"].append(
                    f"La colonne cible '{target}' contient "
                    f"{n_classes} valeurs quasiment toutes "
                    f"différentes pour {n_rows} lignes. Cela "
                    f"ressemble à un identifiant, un titre ou un "
                    f"texte libre, pas à une catégorie à prédire."
                )

                resultat["recommandation"] = (
                    "Choisissez une colonne avec un nombre limité de "
                    "valeurs qui se répètent, ou une colonne "
                    "numérique pour une régression."
                )

            else:

                resultat["titre"] = (
                    "⚠️ Trop de catégories pour une classification "
                    "fiable"
                )

                resultat["messages"].append(
                    f"La colonne cible '{target}' contient "
                    f"{n_classes} catégories différentes, ce qui est "
                    f"trop élevé pour une classification fiable "
                    f"(maximum recommandé : "
                    f"{MAX_CLASSES_CLASSIFICATION})."
                )

                resultat["recommandation"] = (
                    "Regroupez les catégories les plus rares avant "
                    "l'entraînement, ou choisissez une autre colonne "
                    "cible."
                )

            return resultat

        # ------------------------------------------------------------
        # Classes trop rares (souvent des erreurs de saisie) : on
        # essaie de les exclure automatiquement plutôt que de tout
        # bloquer, SI le dataset reste exploitable une fois ces
        # quelques lignes retirées.
        # ------------------------------------------------------------

        classes_rares = class_counts[
            class_counts < self.min_obs_par_classe_bloquant
        ]

        if len(classes_rares) > 0:

            noms_classes_rares = list(classes_rares.index)

            masque_rare = y.isin(noms_classes_rares)

            n_lignes_rares = int(masque_rare.sum())

            df_sans_rares = df_travail.loc[~masque_rare].copy()

            y_sans_rares = df_sans_rares[target]

            n_classes_restantes = int(y_sans_rares.nunique())

            n_rows_restantes = int(len(df_sans_rares))

            exclusion_viable = (
                n_rows_restantes >= self.min_total_rows
                and n_classes_restantes >= 2
            )

            if not exclusion_viable:

                resultat.update(
                    {
                        "ok": False,
                        "bloquant": True,
                        "titre": "⚠️ Dataset trop petit",
                        "n_classes": n_classes,
                        "class_counts": {
                            str(k): int(v)
                            for k, v in class_counts.items()
                        },
                        "min_class_count": int(class_counts.min()),
                        "can_stratify": False,
                        "dataset_filtre": df_travail,
                    }
                )

                resultat["messages"].append(
                    f"Le dataset contient {n_rows} observations pour "
                    f"{n_classes} classes, mais la ou les classes "
                    f"les moins représentées ({', '.join(str(c) for c in noms_classes_rares)}) "
                    f"n'ont que {int(classes_rares.min())} "
                    f"observation(s) chacune. Les exclure ne "
                    f"laisserait plus assez de données "
                    f"({n_rows_restantes} lignes, "
                    f"{n_classes_restantes} classes) pour une "
                    f"séparation fiable."
                )

                resultat["recommandation"] = (
                    "Ajoutez davantage d'observations pour la ou les "
                    "classes minoritaires."
                )

                return resultat

            # Exclusion automatique viable : on continue avec le
            # dataset filtré.

            resultat["classes_exclues"] = [
                {
                    "classe": str(classe),
                    "observations": int(classes_rares[classe]),
                }
                for classe in noms_classes_rares
            ]

            resultat["titre"] = "⚠️ Dataset validé avec exclusions"

            details_classes = ", ".join(
                f"'{classe}' ({int(classes_rares[classe])} obs.)"
                for classe in noms_classes_rares
            )

            resultat["messages"].append(
                f"{len(noms_classes_rares)} classe(s) trop rare(s) "
                f"(moins de {self.min_obs_par_classe_bloquant} "
                f"observations, souvent des erreurs de saisie) ont "
                f"été exclues automatiquement : {details_classes}. "
                f"{n_lignes_rares} ligne(s) au total retirée(s) sur "
                f"{n_rows}."
            )

            resultat["recommandation"] = (
                "Vérifiez si ces valeurs rares correspondent à des "
                "erreurs de saisie dans votre fichier source."
            )

            df_travail = df_sans_rares
            y = y_sans_rares
            class_counts = class_counts.drop(
                index=noms_classes_rares
            )
            n_classes = n_classes_restantes
            n_rows = n_rows_restantes

        # ------------------------------------------------------------
        # A ce stade, toutes les classes restantes ont au moins
        # min_obs_par_classe_bloquant observations : la séparation
        # stratifiée est possible.
        # ------------------------------------------------------------

        min_class_count = int(class_counts.min())

        resultat["n_rows"] = n_rows
        resultat["n_classes"] = n_classes
        resultat["class_counts"] = {
            str(k): int(v) for k, v in class_counts.items()
        }
        resultat["min_class_count"] = min_class_count
        resultat["can_stratify"] = True
        resultat["dataset_filtre"] = df_travail

        if min_class_count < self.min_obs_par_classe_recommande:

            resultat["messages"].append(
                f"La classe la moins représentée ne contient que "
                f"{min_class_count} observations (recommandé : au "
                f"moins {self.min_obs_par_classe_recommande}). Les "
                f"métriques calculées sur cette classe seront peu "
                f"fiables."
            )

            if resultat["titre"] == "✅ Dataset validé":

                resultat["titre"] = "⚠️ Dataset validé avec réserves"

            if not resultat["recommandation"]:

                resultat["recommandation"] = (
                    "Les résultats seront calculés, mais restent à "
                    "interpréter avec prudence pour les classes "
                    "minoritaires."
                )

        # Vérifie que le test_size par défaut laisse au moins une
        # observation de chaque classe côté test.

        attendu_test = min_class_count * self.test_size

        if attendu_test < 1:

            test_size_ajuste = max(
                round(1 / min_class_count, 2),
                self.test_size,
            )

            test_size_ajuste = min(test_size_ajuste, 0.5)

            resultat["recommended_test_size"] = test_size_ajuste

            resultat["messages"].append(
                f"Le test_size par défaut ({self.test_size}) ne "
                f"garantirait pas au moins une observation de chaque "
                f"classe dans le jeu de test. Ajustement automatique "
                f"à {test_size_ajuste}."
            )

        return resultat

    # ----------------------------------------------------------------
    # REGRESSION
    # ----------------------------------------------------------------

    def _valider_regression(self, df_travail, resultat):

        n_rows = int(len(df_travail))

        min_total_regression = max(self.min_total_rows, 10)

        min_train_reg = max(int(n_rows * (1 - self.test_size)), 1)
        min_test_reg = max(n_rows - min_train_reg, 0)

        resultat["dataset_filtre"] = df_travail

        if n_rows < min_total_regression or min_test_reg < 2:

            resultat.update(
                {
                    "ok": False,
                    "bloquant": True,
                    "titre": "⚠️ Dataset trop petit",
                }
            )

            resultat["messages"].append(
                f"Le dataset contient seulement {n_rows} "
                f"observations, ce qui ne permet pas une séparation "
                f"entraînement/test fiable pour une régression."
            )

            resultat["recommandation"] = (
                "Ajoutez davantage d'observations avant de relancer "
                "l'analyse."
            )

            return resultat

        if n_rows < 30:

            resultat["titre"] = "⚠️ Dataset validé avec réserves"

            resultat["messages"].append(
                f"Le dataset ne contient que {n_rows} observations. "
                f"Les métriques de régression (R², RMSE, MAE) seront "
                f"instables sur un jeu de test aussi petit."
            )

            resultat["recommandation"] = (
                "Interprétez les résultats avec prudence et "
                "privilégiez la validation croisée."
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