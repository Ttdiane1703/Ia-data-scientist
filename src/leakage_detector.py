"""
================================================================
DÉTECTION AUTOMATIQUE DU DATA LEAKAGE
================================================================

Ce module détecte, avant l'entraînement, les variables qui
constituent une fuite de données vis-à-vis de la cible :

- variables identiques ou quasi identiques à la cible ;
- variables directement dérivées de la cible ;
- corrélations anormalement élevées ;
- variables permettant de reconstruire la cible (mapping quasi
  déterministe, même sans lien linéaire) ;
- fuite temporelle potentielle (une colonne de date sépare
  parfaitement les classes) ;
- performance globale anormalement élevée (signal global qu'une
  fuite existe quelque part dans les features).

Volontairement, la détection ne se base JAMAIS uniquement sur le
nom des colonnes : chaque signal est calculé statistiquement à
partir des valeurs réelles.
================================================================
"""

import numpy as np
import pandas as pd


NIVEAU_CRITIQUE = "CRITIQUE"
NIVEAU_ELEVE = "ÉLEVÉ"
NIVEAU_MODERE = "MODÉRÉ"

# Taille d'échantillon utilisée pour les vérifications rapides sur
# de gros datasets (les checks de fuite n'ont pas besoin de la
# totalité des lignes pour être fiables).
TAILLE_ECHANTILLON_MAX = 5000


class DataLeakageDetector:
    """
    Détecte les colonnes de X qui constituent une fuite de données
    vis-à-vis de y, et propose une exclusion automatique des
    colonnes les plus à risque.
    """

    def __init__(
        self,
        correlation_critique=0.98,
        correlation_elevee=0.90,
        purete_critique=0.995,
        purete_elevee=0.95,
        auto_exclude_niveaux=(NIVEAU_CRITIQUE, NIVEAU_ELEVE),
        random_state=42,
    ):
        self.correlation_critique = correlation_critique
        self.correlation_elevee = correlation_elevee
        self.purete_critique = purete_critique
        self.purete_elevee = purete_elevee
        self.auto_exclude_niveaux = set(auto_exclude_niveaux)
        self.random_state = random_state

    # ----------------------------------------------------------
    # POINT D'ENTREE
    # ----------------------------------------------------------

    def detect(self, X, y, target_name="target", date_cols=None):
        """
        Analyse chaque colonne de X vis-à-vis de y.

        Retourne un dictionnaire :
            {
                "colonnes_a_risque": [ {...}, ... ],
                "colonnes_exclues": [...],
                "performance_anormale": {...} | None,
                "resume": str,
            }

        Ne lève jamais d'exception : une colonne qui pose problème
        est simplement ignorée pour ce check précis, plutôt que de
        faire échouer toute la détection.
        """

        try:
            return self._detect(X, y, target_name, date_cols)
        except Exception as e:
            return {
                "colonnes_a_risque": [],
                "colonnes_exclues": [],
                "performance_anormale": None,
                "resume": (
                    f"La détection de data leakage a rencontré une "
                    f"erreur inattendue et a été ignorée : {e}"
                ),
                "erreur": str(e),
            }

    def _detect(self, X, y, target_name, date_cols):

        est_classification = not pd.api.types.is_numeric_dtype(y) or (
            pd.api.types.is_numeric_dtype(y) and y.nunique() <= 20
        )

        colonnes_a_risque = []

        # Échantillonnage pour rester rapide sur de gros datasets.
        X_ech, y_ech = self._echantillonner(X, y)

        for colonne in X.columns:

            try:
                diagnostic = self._analyser_colonne(
                    X_ech[colonne],
                    y_ech,
                    colonne,
                    est_classification,
                )
            except Exception:
                continue

            if diagnostic is not None:
                colonnes_a_risque.append(diagnostic)

        # Fuite temporelle potentielle
        if date_cols:

            for colonne in date_cols:

                if colonne not in X.columns:
                    continue

                try:
                    diagnostic = self._analyser_fuite_temporelle(
                        X_ech[colonne],
                        y_ech,
                        colonne,
                        est_classification,
                    )
                except Exception:
                    continue

                if diagnostic is not None:
                    colonnes_a_risque.append(diagnostic)

        # Signal global : performance anormalement élevée avec un
        # modèle volontairement simple et rapide.
        performance_anormale = self._verifier_performance_globale(
            X_ech, y_ech, est_classification
        )

        # Détermine les colonnes à exclure automatiquement
        colonnes_exclues = [
            d["colonne"]
            for d in colonnes_a_risque
            if d["niveau_risque"] in self.auto_exclude_niveaux
        ]

        resume = self._construire_resume(
            colonnes_a_risque, performance_anormale
        )

        return {
            "colonnes_a_risque": colonnes_a_risque,
            "colonnes_exclues": colonnes_exclues,
            "performance_anormale": performance_anormale,
            "resume": resume,
        }

    def filtrer(self, X, resultat_detection):
        """
        Retourne une copie de X sans les colonnes marquées comme
        étant à exclure automatiquement.
        """

        colonnes_exclues = resultat_detection.get(
            "colonnes_exclues", []
        )

        if not colonnes_exclues:
            return X

        return X.drop(
            columns=[c for c in colonnes_exclues if c in X.columns]
        )

    # ----------------------------------------------------------
    # ANALYSE PAR COLONNE
    # ----------------------------------------------------------

    def _analyser_colonne(self, serie, y, nom_colonne, est_classification):

        serie_valide = serie.notna() & y.notna()

        if serie_valide.sum() < 5:
            return None

        s = serie[serie_valide]
        cible = y[serie_valide]

        n_valeurs_uniques = s.nunique(dropna=True)
        n_lignes = len(s)

        # Une colonne quasi unique par ligne ressemble à un
        # identifiant, pas à une fuite : on la laisse à la
        # détection des colonnes inutiles (autre chantier).
        ressemble_a_un_id = n_valeurs_uniques >= 0.98 * n_lignes

        est_numerique = pd.api.types.is_numeric_dtype(s)

        raisons = []
        niveau = None
        valeur_reference = None

        # --------------------------------------------------
        # 1) Identité exacte / quasi exacte avec la cible
        # --------------------------------------------------

        if self._colonnes_identiques(s, cible):

            raisons.append(
                "Cette colonne contient des valeurs identiques "
                "(ou quasi identiques) à la cible."
            )

            niveau = NIVEAU_CRITIQUE
            valeur_reference = 1.0

        # --------------------------------------------------
        # 2) Cas numérique : corrélation + relation arithmétique
        # --------------------------------------------------

        if est_numerique and pd.api.types.is_numeric_dtype(cible):

            try:
                correlation = abs(
                    np.corrcoef(
                        s.astype(float), cible.astype(float)
                    )[0, 1]
                )
            except Exception:
                correlation = 0.0

            if not np.isnan(correlation):

                if correlation >= self.correlation_critique:

                    raisons.append(
                        f"Corrélation quasi parfaite avec la "
                        f"cible (r = {correlation:.3f})."
                    )

                    niveau = NIVEAU_CRITIQUE
                    valeur_reference = correlation

                elif correlation >= self.correlation_elevee:

                    raisons.append(
                        f"Corrélation anormalement élevée avec "
                        f"la cible (r = {correlation:.3f})."
                    )

                    niveau = niveau or NIVEAU_ELEVE
                    valeur_reference = valeur_reference or correlation

            # Relation arithmétique directe (colonne dérivée)
            if self._relation_arithmetique_directe(s, cible):

                raisons.append(
                    "Cette colonne semble directement dérivée de "
                    "la cible par une opération arithmétique "
                    "simple (différence ou ratio constant)."
                )

                niveau = NIVEAU_CRITIQUE
                valeur_reference = 1.0

        # --------------------------------------------------
        # 3) Purté (mapping quasi déterministe vers la cible),
        #    utile pour capter les relations non linéaires et
        #    les colonnes catégorielles.
        # --------------------------------------------------

        if not ressemble_a_un_id:

            purete = self._purete(s, cible, est_numerique)

            if purete is not None:

                if purete >= self.purete_critique:

                    raisons.append(
                        f"Connaître la valeur de cette colonne "
                        f"permet de reconstruire la cible dans "
                        f"{purete * 100:.1f}% des cas."
                    )

                    niveau = NIVEAU_CRITIQUE
                    valeur_reference = purete

                elif purete >= self.purete_elevee:

                    raisons.append(
                        f"Cette colonne permet de reconstruire la "
                        f"cible dans {purete * 100:.1f}% des cas, "
                        f"ce qui est anormalement élevé."
                    )

                    niveau = niveau or NIVEAU_ELEVE
                    valeur_reference = valeur_reference or purete

        if niveau is None:
            return None

        return {
            "colonne": nom_colonne,
            "raisons": raisons,
            "niveau_risque": niveau,
            "valeur_reference": (
                round(float(valeur_reference), 4)
                if valeur_reference is not None
                else None
            ),
            "recommandation": self._recommandation(niveau),
            "exclue_automatiquement": (
                niveau in self.auto_exclude_niveaux
            ),
        }

    def _analyser_fuite_temporelle(
        self, serie, y, nom_colonne, est_classification
    ):

        try:
            dates = pd.to_datetime(serie, errors="coerce")
        except Exception:
            return None

        masque = dates.notna() & y.notna()

        if masque.sum() < 10:
            return None

        dates_valides = dates[masque]
        cible_valide = y[masque]

        ordre = dates_valides.sort_values().index

        cible_ordonnee = cible_valide.loc[ordre].reset_index(
            drop=True
        )

        if est_classification:

            # Cherche un seuil temporel qui sépare quasi
            # parfaitement les classes (fuite d'information
            # future : "on sait déjà comment ça finit").
            meilleure_purete = 0.0

            n = len(cible_ordonnee)

            pas = max(n // 20, 1)

            for i in range(pas, n - pas, pas):

                avant = cible_ordonnee.iloc[:i]
                apres = cible_ordonnee.iloc[i:]

                purete_avant = avant.value_counts(
                    normalize=True
                ).max()

                purete_apres = apres.value_counts(
                    normalize=True
                ).max()

                purete_globale = (
                    purete_avant * len(avant)
                    + purete_apres * len(apres)
                ) / n

                meilleure_purete = max(
                    meilleure_purete, purete_globale
                )

            if meilleure_purete >= self.purete_critique:

                return {
                    "colonne": nom_colonne,
                    "raisons": [
                        "Cette colonne temporelle sépare presque "
                        "parfaitement les classes de la cible, ce "
                        "qui suggère une information connue "
                        "uniquement après coup (fuite temporelle)."
                    ],
                    "niveau_risque": NIVEAU_CRITIQUE,
                    "valeur_reference": round(
                        float(meilleure_purete), 4
                    ),
                    "recommandation": self._recommandation(
                        NIVEAU_CRITIQUE
                    ),
                    "exclue_automatiquement": (
                        NIVEAU_CRITIQUE in self.auto_exclude_niveaux
                    ),
                }

        return None

    # ----------------------------------------------------------
    # SIGNAL GLOBAL : PERFORMANCE ANORMALEMENT ELEVEE
    # ----------------------------------------------------------

    def _verifier_performance_globale(
        self, X, y, est_classification
    ):

        try:

            from sklearn.tree import (
                DecisionTreeClassifier,
                DecisionTreeRegressor,
            )
            from sklearn.model_selection import cross_val_score
            from sklearn.preprocessing import LabelEncoder

            X_num = X.select_dtypes(include=[np.number]).copy()

            for colonne in X.select_dtypes(
                exclude=[np.number]
            ).columns:

                try:
                    X_num[colonne] = LabelEncoder().fit_transform(
                        X[colonne].astype(str)
                    )
                except Exception:
                    continue

            X_num = X_num.fillna(X_num.median(numeric_only=True))
            X_num = X_num.fillna(0)

            if X_num.shape[1] == 0 or X_num.shape[0] < 20:
                return None

            if est_classification:

                y_enc = LabelEncoder().fit_transform(
                    y.astype(str)
                )

                modele = DecisionTreeClassifier(
                    max_depth=4, random_state=self.random_state
                )

                scores = cross_val_score(
                    modele, X_num, y_enc, cv=3, scoring="accuracy"
                )

                seuil = 0.99

            else:

                modele = DecisionTreeRegressor(
                    max_depth=4, random_state=self.random_state
                )

                scores = cross_val_score(
                    modele, X_num, y, cv=3, scoring="r2"
                )

                seuil = 0.995

            score_moyen = float(np.mean(scores))

            if score_moyen >= seuil:

                modele.fit(X_num, y_enc if est_classification else y)

                importances = pd.Series(
                    modele.feature_importances_,
                    index=X_num.columns,
                ).sort_values(ascending=False)

                top_features = list(
                    importances.head(3).index
                )

                return {
                    "detectee": True,
                    "score_moyen": round(score_moyen, 4),
                    "message": (
                        f"Un modèle volontairement simple (arbre "
                        f"de décision peu profond) atteint déjà un "
                        f"score de {score_moyen:.4f} en validation "
                        f"croisée. C'est anormalement élevé et "
                        f"suggère qu'une ou plusieurs variables "
                        f"contiennent une fuite d'information."
                    ),
                    "features_suspectes": top_features,
                }

            return None

        except Exception:
            return None

    # ----------------------------------------------------------
    # OUTILS STATISTIQUES
    # ----------------------------------------------------------

    def _colonnes_identiques(self, s, cible):

        try:

            if pd.api.types.is_numeric_dtype(
                s
            ) and pd.api.types.is_numeric_dtype(cible):

                return np.allclose(
                    s.astype(float),
                    cible.astype(float),
                    equal_nan=True,
                    rtol=1e-6,
                    atol=1e-9,
                )

            return (
                s.astype(str).reset_index(drop=True)
                == cible.astype(str).reset_index(drop=True)
            ).mean() >= 0.999

        except Exception:
            return False

    def _relation_arithmetique_directe(self, s, cible):

        try:

            s = s.astype(float)
            cible = cible.astype(float)

            diff = s - cible

            if diff.std() < 1e-9 and abs(diff.mean()) > 0:
                return True

            cible_non_nulle = cible.replace(0, np.nan)

            ratio = (s / cible_non_nulle).dropna()

            if len(ratio) > 0 and ratio.std() < 1e-9:
                return True

            return False

        except Exception:
            return False

    def _purete(self, s, cible, est_numerique):
        """
        Mesure à quel point connaître la valeur de `s` permet de
        deviner `cible` : pour chaque valeur (ou bin) de `s`, on
        regarde la proportion de la valeur de cible majoritaire,
        pondérée par la taille du groupe. Proche de 1 = fuite.
        """

        try:

            if est_numerique:

                n_bins = min(20, max(2, s.nunique() // 2))

                groupes = pd.qcut(
                    s, q=n_bins, duplicates="drop"
                )

            else:

                groupes = s

            df_tmp = pd.DataFrame(
                {"groupe": groupes, "cible": cible}
            )

            n_groupes = df_tmp["groupe"].nunique()

            if n_groupes <= 1 or n_groupes >= 0.9 * len(df_tmp):
                return None

            purete_par_groupe = df_tmp.groupby(
                "groupe", observed=True
            )["cible"].apply(
                lambda g: g.value_counts(normalize=True).max()
            )

            tailles = df_tmp.groupby(
                "groupe", observed=True
            ).size()

            purete_ponderee = (
                purete_par_groupe * tailles
            ).sum() / tailles.sum()

            return float(purete_ponderee)

        except Exception:
            return None

    def _echantillonner(self, X, y):

        n = len(X)

        if n <= TAILLE_ECHANTILLON_MAX:
            return X, y

        index_echantillon = X.sample(
            n=TAILLE_ECHANTILLON_MAX,
            random_state=self.random_state,
        ).index

        return X.loc[index_echantillon], y.loc[index_echantillon]

    def _recommandation(self, niveau):

        if niveau == NIVEAU_CRITIQUE:

            return (
                "Exclure cette variable avant l'entraînement : "
                "elle constitue très probablement une fuite de "
                "données directe."
            )

        if niveau == NIVEAU_ELEVE:

            return (
                "Vérifier manuellement l'origine de cette variable "
                "avant de l'utiliser. Une exclusion est recommandée "
                "par prudence."
            )

        return (
            "Surveiller cette variable : sa relation avec la cible "
            "est plus forte que la moyenne."
        )

    def _construire_resume(self, colonnes_a_risque, performance_anormale):

        if not colonnes_a_risque and not performance_anormale:
            return "✅ Aucune fuite de données détectée."

        n_critique = sum(
            1
            for d in colonnes_a_risque
            if d["niveau_risque"] == NIVEAU_CRITIQUE
        )

        n_eleve = sum(
            1
            for d in colonnes_a_risque
            if d["niveau_risque"] == NIVEAU_ELEVE
        )

        n_modere = len(colonnes_a_risque) - n_critique - n_eleve

        morceaux = []

        if n_critique:
            morceaux.append(f"{n_critique} critique(s)")

        if n_eleve:
            morceaux.append(f"{n_eleve} élevé(s)")

        if n_modere:
            morceaux.append(f"{n_modere} modéré(s)")

        resume = "🚨 Data leakage détecté"

        if morceaux:
            resume += " — risque(s) : " + ", ".join(morceaux)

        if performance_anormale:
            resume += (
                " ; performance globale anormalement élevée "
                "détectée."
            )

        return resume
