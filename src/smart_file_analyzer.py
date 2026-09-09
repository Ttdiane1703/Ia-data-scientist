"""
================================================================
ANALYSE DE STRUCTURE ET IMPORT INTELLIGENT (CSV / EXCEL)
================================================================

Ce module prépare intelligemment un fichier CSV ou Excel — même
très mal structuré — AVANT toute analyse statistique ou Machine
Learning :

- lecture de toutes les feuilles d'un classeur Excel ;
- détection de la vraie ligne d'en-tête (même après des lignes de
  titre, de notes, ou des en-têtes fusionnés) ;
- détection et nettoyage des lignes/colonnes de bruit en haut, en
  bas, à gauche et à droite (titres, notes, totaux, colonnes
  "Unnamed" réellement vides) ;
- décision de transposition (manuelle ou automatique) ;
- détection de relations probables entre les feuilles d'un même
  classeur (clés/identifiants communs) ;
- résumé complet des transformations effectuées, avec conservation
  systématique d'une copie du fichier original pour permettre de
  tout annuler.

RÈGLE D'OR : ne jamais supprimer une information seulement parce
qu'elle se trouve en haut, en bas, à gauche ou à droite du fichier.
Chaque suppression est motivée par un signal concret (ligne vide,
ligne de titre/notes détectée, colonne entièrement vide, ligne
d'en-tête dupliquée...), jamais par sa seule position.
================================================================
"""

import re
import os
import pandas as pd
import numpy as np


MOTS_CLES_BRUIT = (
    "total", "sous-total", "sous total", "subtotal", "grand total",
    "source", "note", "notes", "commentaire", "commentaires",
    "généré le", "genere le", "généré par", "extraction du",
    "page ", "confidentiel", "copyright", "©",
)

MAX_LIGNES_CANDIDATES_ENTETE = 15


# ================================================================
# CHARGEMENT (TOUTES FEUILLES POUR EXCEL)
# ================================================================

def charger_fichier_brut(chemin):
    """
    Charge un fichier CSV ou Excel SANS aucun nettoyage. Retourne
    un dictionnaire {nom_de_feuille: DataFrame brut}. Pour un CSV,
    le dictionnaire contient une seule entrée ("Feuille1").

    Le chargement se fait toujours avec header=None : la détection
    de la vraie ligne d'en-tête est un problème statistique traité
    séparément (voir detecter_ligne_entete), pas une hypothèse de
    lecture.
    """

    extension = os.path.splitext(str(chemin).lower())[1]

    if extension == ".csv":

        df_brut = pd.read_csv(
            chemin, header=None, dtype=object
        )

        return {"Feuille1": df_brut}

    if extension in (".xlsx", ".xls"):

        classeur = pd.read_excel(
            chemin, sheet_name=None, header=None, dtype=object
        )

        return dict(classeur)

    raise ValueError(
        f"Extension non supportée pour l'import intelligent : "
        f"{extension}. Formats acceptés : .csv, .xlsx, .xls"
    )


# ================================================================
# DETECTION DE LA VRAIE LIGNE D'EN-TETE
# ================================================================

def _score_ligne_entete(df_brut, index_ligne):
    """
    Estime à quel point la ligne `index_ligne` ressemble à un
    véritable en-tête de tableau : cellules non vides, valeurs
    textuelles distinctes, et lignes suivantes qui ressemblent
    davantage à des données (types plus variés / plus numériques).
    """

    ligne = df_brut.iloc[index_ligne]

    n_cellules = len(ligne)

    non_vides = ligne.notna().sum()

    if non_vides == 0:
        return -1.0

    taux_remplissage = non_vides / n_cellules

    valeurs = ligne.dropna().astype(str).str.strip()

    if len(valeurs) == 0:
        return -1.0

    taux_texte = valeurs.apply(
        lambda v: not _ressemble_a_un_nombre(v)
    ).mean()

    taux_unicite = valeurs.nunique() / len(valeurs)

    # Les lignes suivantes doivent ressembler davantage à des
    # données qu'à des en-têtes (plus de valeurs numériques, moins
    # de répétitions).
    lignes_suivantes = df_brut.iloc[
        index_ligne + 1: index_ligne + 6
    ]

    if len(lignes_suivantes) == 0:
        score_donnees_suivantes = 0.0
    else:
        valeurs_suivantes = (
            lignes_suivantes.stack().astype(str).str.strip()
        )

        if len(valeurs_suivantes) == 0:
            score_donnees_suivantes = 0.0
        else:
            score_donnees_suivantes = valeurs_suivantes.apply(
                _ressemble_a_un_nombre
            ).mean()

    score = (
        taux_remplissage * 0.35
        + taux_texte * 0.30
        + taux_unicite * 0.15
        + score_donnees_suivantes * 0.20
    )

    return score


def _ressemble_a_un_nombre(valeur):

    valeur = str(valeur).strip().replace(",", ".")

    if valeur == "" or valeur.lower() in ("nan", "none"):
        return False

    try:
        float(valeur)
        return True
    except ValueError:
        return False


def detecter_ligne_entete(df_brut):
    """
    Cherche, parmi les MAX_LIGNES_CANDIDATES_ENTETE premières
    lignes, celle qui ressemble le plus à une véritable ligne
    d'en-tête de tableau.

    Retourne l'index de cette ligne (0 par défaut si le fichier
    est déjà propre).
    """

    n_candidates = min(
        MAX_LIGNES_CANDIDATES_ENTETE, len(df_brut)
    )

    if n_candidates == 0:
        return 0

    scores = [
        _score_ligne_entete(df_brut, i)
        for i in range(n_candidates)
    ]

    meilleur_index = int(np.argmax(scores))

    return meilleur_index


# ================================================================
# DETECTION DES LIGNES DE BRUIT (HAUT / BAS)
# ================================================================

def _ligne_est_du_bruit(ligne, n_colonnes_attendues):
    """
    Une ligne est considérée comme du bruit (titre, note, total,
    ligne de génération de rapport...) si :
    - elle est presque entièrement vide, OU
    - une grande partie de son texte contient un mot-clé de bruit
      connu, OU
    - une seule cellule est remplie alors que le tableau attend
      plusieurs colonnes (typique d'un titre fusionné en haut d'un
      fichier Excel).
    """

    non_vides = ligne.notna().sum()

    if non_vides == 0:
        return True

    texte_ligne = " ".join(
        str(v).strip().lower() for v in ligne.dropna()
    )

    if any(mot in texte_ligne for mot in MOTS_CLES_BRUIT):
        return True

    if (
        n_colonnes_attendues >= 3
        and non_vides == 1
    ):
        return True

    return False


def nettoyer_haut_bas(df_brut, index_entete):
    """
    Retire les lignes de bruit AVANT la ligne d'en-tête détectée
    (titres, notes) et APRES la fin réelle des données (totaux,
    notes de bas de page, lignes vides). Ne retire jamais une ligne
    de données valide : chaque ligne retirée en bas doit d'abord
    être identifiée comme du bruit par _ligne_est_du_bruit, et on
    s'arrête dès qu'une ligne "normale" est rencontrée.
    """

    n_colonnes = df_brut.shape[1]

    lignes_titre_supprimees = index_entete

    corps = df_brut.iloc[index_entete + 1:].reset_index(drop=True)

    # Recherche du bruit en bas, en partant de la fin.
    derniere_ligne_valide = len(corps) - 1

    while derniere_ligne_valide >= 0:

        ligne = corps.iloc[derniere_ligne_valide]

        if _ligne_est_du_bruit(ligne, n_colonnes):

            derniere_ligne_valide -= 1

        else:

            break

    lignes_bas_supprimees = (
        len(corps) - 1 - derniere_ligne_valide
    )

    corps = corps.iloc[: derniere_ligne_valide + 1]

    return corps, lignes_titre_supprimees, lignes_bas_supprimees


# ================================================================
# APPLICATION DE L'EN-TETE + NETTOYAGE DES COLONNES
# ================================================================

def _nom_colonne_propre(valeur, position):

    if valeur is None or (
        isinstance(valeur, float) and np.isnan(valeur)
    ):
        return f"colonne_{position}"

    nom = str(valeur).strip()

    if nom == "" or nom.lower().startswith("unnamed"):
        return f"colonne_{position}"

    return nom


def appliquer_entete(df_brut, index_entete, corps):

    ligne_entete = df_brut.iloc[index_entete]

    noms_colonnes = [
        _nom_colonne_propre(valeur, position)
        for position, valeur in enumerate(ligne_entete)
    ]

    # Dédoublonne les noms de colonnes identiques (en-têtes
    # fusionnés/répétés donnant deux fois "Ventes" par exemple).
    compteur = {}

    noms_finaux = []

    for nom in noms_colonnes:

        if nom not in compteur:

            compteur[nom] = 0
            noms_finaux.append(nom)

        else:

            compteur[nom] += 1
            noms_finaux.append(f"{nom}_{compteur[nom]}")

    corps = corps.copy()
    corps.columns = noms_finaux

    return corps, noms_finaux


def retirer_lignes_entete_dupliquees(df, noms_colonnes):
    """
    Certains exports Excel répètent la ligne d'en-tête au milieu
    des données (ex. un en-tête réimprimé toutes les 50 lignes).
    On retire les lignes dont les valeurs correspondent presque
    exactement aux noms de colonnes.
    """

    if len(df) == 0:
        return df, 0

    noms_normalises = [
        str(n).strip().lower() for n in noms_colonnes
    ]

    def _est_repetition_entete(ligne):

        valeurs = [
            str(v).strip().lower() if pd.notna(v) else ""
            for v in ligne
        ]

        correspondances = sum(
            1
            for v, n in zip(valeurs, noms_normalises)
            if v == n and v != ""
        )

        return correspondances >= max(2, int(0.6 * len(noms_normalises)))

    masque_repetition = df.apply(_est_repetition_entete, axis=1)

    n_supprimees = int(masque_repetition.sum())

    if n_supprimees > 0:
        df = df.loc[~masque_repetition].reset_index(drop=True)

    return df, n_supprimees


def retirer_lignes_colonnes_vides(df):

    n_lignes_avant = len(df)
    n_colonnes_avant = df.shape[1]

    df = df.dropna(axis=0, how="all")
    df = df.dropna(axis=1, how="all")

    df = df.reset_index(drop=True)

    return (
        df,
        n_lignes_avant - len(df),
        n_colonnes_avant - df.shape[1],
    )


# ================================================================
# CONVERSION DES TYPES
# ================================================================

def convertir_types(df):
    """
    Tente de convertir chaque colonne vers le type le plus adapté
    (numérique, date, ou texte/catégoriel laissé tel quel), sans
    jamais faire planter le pipeline sur une colonne récalcitrante.
    """

    conversions = {}

    for colonne in df.columns:

        serie = df[colonne]

        if serie.dropna().empty:
            continue

        # Tentative numérique
        serie_numerique = pd.to_numeric(
            serie.astype(str).str.replace(",", ".", regex=False).str.strip(),
            errors="coerce",
        )

        taux_conversion_numerique = (
            serie_numerique.notna().sum()
            / max(serie.notna().sum(), 1)
        )

        if taux_conversion_numerique >= 0.95:

            df[colonne] = serie_numerique

            conversions[colonne] = "numérique"

            continue

        # Tentative date, uniquement si le nom ou le contenu
        # suggère une date (évite de transformer des codes/ID en
        # dates par coïncidence).
        if _ressemble_a_une_colonne_date(colonne, serie):

            utiliser_jour_en_premier = _format_jour_en_premier(
                serie
            )

            serie_date = pd.to_datetime(
                serie,
                errors="coerce",
                dayfirst=utiliser_jour_en_premier,
            )

            taux_conversion_date = (
                serie_date.notna().sum()
                / max(serie.notna().sum(), 1)
            )

            if taux_conversion_date >= 0.90:

                df[colonne] = serie_date

                conversions[colonne] = "date"

                continue

    return df, conversions


def _format_jour_en_premier(serie):
    """
    Détermine si le format de date observé commence par l'année
    (ISO, ex. 2024-01-31 -> sans ambiguïté, dayfirst=False) ou par
    le jour/mois (ex. 31/01/2024 ou 01/02/2024 -> ambiguïté,
    dayfirst=True par convention française/européenne).

    Appliquer systématiquement dayfirst=True casserait les dates
    déjà au format ISO (ex. "2024-01-10" serait lu comme le 10ème
    mois plutôt que le 10 janvier).
    """

    echantillon = serie.dropna().astype(str).str.strip().head(20)

    motif_annee_en_premier = re.compile(r"^\d{4}[/\-.]")

    if len(echantillon) == 0:
        return False

    taux_annee_en_premier = echantillon.apply(
        lambda v: bool(motif_annee_en_premier.match(v))
    ).mean()

    return taux_annee_en_premier < 0.5


def _ressemble_a_une_colonne_date(nom_colonne, serie):

    nom = str(nom_colonne).lower()

    if any(
        mot in nom
        for mot in ("date", "jour", "mois", "annee", "année", "time")
    ):
        return True

    echantillon = serie.dropna().astype(str).head(20)

    motif_date = re.compile(
        r"^\d{1,4}[/\-.]\d{1,2}[/\-.]\d{1,4}$"
    )

    if len(echantillon) == 0:
        return False

    taux = echantillon.apply(
        lambda v: bool(motif_date.match(v.strip()))
    ).mean()

    return taux >= 0.7


# ================================================================
# TRANSPOSITION
# ================================================================

def _score_tabularite(df):
    """
    Estime à quel point un DataFrame ressemble à un tableau de
    données Data Science bien orienté (une observation par ligne,
    une variable par colonne) :
    - des colonnes majoritairement homogènes en type ;
    - des colonnes non constantes (variance d'information) ;
    - davantage de lignes que de colonnes (typique en Data
      Science : peu de variables, beaucoup d'observations) ;
    - des en-têtes qui ressemblent à des noms de variables (texte),
      pas à des valeurs.
    """

    if df.shape[0] == 0 or df.shape[1] == 0:
        return -1.0

    # Ratio lignes/colonnes : trop de colonnes par rapport aux
    # lignes est un signal fréquent de mauvaise orientation.
    ratio_lignes_colonnes = df.shape[0] / df.shape[1]

    score_ratio = min(ratio_lignes_colonnes / 5, 1.0)

    # Colonnes non constantes
    taux_non_constant = np.mean(
        [df[c].nunique(dropna=True) > 1 for c in df.columns]
    )

    # En-têtes qui ressemblent à du texte plutôt qu'à des valeurs
    # numériques (ex. des dates ou des nombres utilisés comme
    # colonnes, signe que le tableau est à l'envers).
    taux_entetes_textuels = np.mean(
        [not _ressemble_a_un_nombre(c) for c in df.columns]
    )

    score = (
        score_ratio * 0.4
        + taux_non_constant * 0.3
        + taux_entetes_textuels * 0.3
    )

    return score


def decider_transposition(df, mode="automatique"):
    """
    mode : "non" | "oui" | "automatique"

    Retourne (df_final, transposition_effectuee: bool, score_normal,
    score_transpose).
    """

    if mode == "non":
        return df, False, None, None

    df_transpose = df.set_index(df.columns[0]).T.reset_index()

    df_transpose = df_transpose.rename(
        columns={"index": df.columns[0]}
    )

    df_transpose.columns.name = None

    if mode == "oui":
        return df_transpose, True, None, None

    score_normal = _score_tabularite(df)
    score_transpose = _score_tabularite(df_transpose)

    if score_transpose > score_normal * 1.10:

        return df_transpose, True, score_normal, score_transpose

    return df, False, score_normal, score_transpose


# ================================================================
# DETECTION DE RELATIONS ENTRE FEUILLES
# ================================================================

def detecter_relations_entre_feuilles(
    dict_dataframes, seuil_similarite=0.3
):
    """
    Cherche des colonnes candidates à être des clés communes entre
    plusieurs feuilles d'un même classeur (mêmes noms de colonnes
    et/ou fort recouvrement de valeurs).

    Ne fusionne jamais automatiquement les feuilles : signale
    seulement les relations probables, à titre informatif.
    """

    relations = []

    noms_feuilles = list(dict_dataframes.keys())

    for i in range(len(noms_feuilles)):

        for j in range(i + 1, len(noms_feuilles)):

            feuille_a = noms_feuilles[i]
            feuille_b = noms_feuilles[j]

            df_a = dict_dataframes[feuille_a]
            df_b = dict_dataframes[feuille_b]

            for colonne_a in df_a.columns:

                for colonne_b in df_b.columns:

                    similarite_nom = (
                        str(colonne_a).strip().lower()
                        == str(colonne_b).strip().lower()
                    )

                    try:

                        valeurs_a = set(
                            df_a[colonne_a].dropna().astype(str)
                        )

                        valeurs_b = set(
                            df_b[colonne_b].dropna().astype(str)
                        )

                    except Exception:

                        continue

                    if not valeurs_a or not valeurs_b:
                        continue

                    intersection = valeurs_a & valeurs_b

                    union = valeurs_a | valeurs_b

                    jaccard = (
                        len(intersection) / len(union)
                        if union
                        else 0
                    )

                    if similarite_nom or jaccard >= seuil_similarite:

                        relations.append(
                            {
                                "feuille_a": feuille_a,
                                "colonne_a": str(colonne_a),
                                "feuille_b": feuille_b,
                                "colonne_b": str(colonne_b),
                                "correspondance_nom": similarite_nom,
                                "recouvrement_valeurs": round(
                                    jaccard, 3
                                ),
                            }
                        )

    return relations


# ================================================================
# PIPELINE COMPLET D'UNE FEUILLE
# ================================================================

def analyser_et_nettoyer_feuille(df_brut, mode_transposition="automatique"):
    """
    Applique l'intégralité du nettoyage intelligent à UNE feuille
    brute (header=None) : détection d'en-tête, nettoyage haut/bas,
    colonnes vides, en-têtes dupliqués, conversion de types, et
    décision de transposition.

    Retourne (df_propre, rapport) où `rapport` détaille chaque
    transformation effectuée (pour l'aperçu et pour permettre
    d'annuler).
    """

    rapport = {
        "dimensions_brutes": list(df_brut.shape),
        "transformations": [],
    }

    if df_brut.empty:

        rapport["transformations"].append(
            "Feuille vide : aucune transformation possible."
        )

        return df_brut, rapport

    index_entete = detecter_ligne_entete(df_brut)

    corps, lignes_titre, lignes_bas = nettoyer_haut_bas(
        df_brut, index_entete
    )

    if lignes_titre > 0:

        rapport["transformations"].append(
            f"{lignes_titre} ligne(s) de titre/notes retirée(s) "
            f"avant l'en-tête (ligne d'en-tête détectée : "
            f"{index_entete})."
        )

    if lignes_bas > 0:

        rapport["transformations"].append(
            f"{lignes_bas} ligne(s) de bruit retirée(s) en bas du "
            f"fichier (totaux, notes, lignes vides)."
        )

    df, noms_colonnes = appliquer_entete(df_brut, index_entete, corps)

    df, n_entetes_dupliques = retirer_lignes_entete_dupliquees(
        df, noms_colonnes
    )

    if n_entetes_dupliques > 0:

        rapport["transformations"].append(
            f"{n_entetes_dupliques} ligne(s) d'en-tête répétée(s) "
            f"au milieu des données ont été retirée(s)."
        )

    df, n_lignes_vides, n_colonnes_vides = (
        retirer_lignes_colonnes_vides(df)
    )

    if n_lignes_vides > 0:

        rapport["transformations"].append(
            f"{n_lignes_vides} ligne(s) entièrement vide(s) "
            f"retirée(s)."
        )

    if n_colonnes_vides > 0:

        rapport["transformations"].append(
            f"{n_colonnes_vides} colonne(s) entièrement vide(s) "
            f"retirée(s)."
        )

    if df.shape[1] == 0 or df.shape[0] == 0:

        rapport["transformations"].append(
            "⚠️ Plus aucune donnée exploitable après nettoyage."
        )

        return df, rapport

    df, transposition_effectuee, score_normal, score_transpose = (
        decider_transposition(df, mode=mode_transposition)
    )

    rapport["transposition_effectuee"] = transposition_effectuee

    if mode_transposition == "automatique":

        rapport["transformations"].append(
            f"Transposition automatique : "
            f"{'appliquée' if transposition_effectuee else 'non nécessaire'} "
            f"(score orientation normale = {score_normal:.2f}, "
            f"score orientation transposée = {score_transpose:.2f})."
        )

    elif transposition_effectuee:

        rapport["transformations"].append(
            "Transposition appliquée (demande explicite)."
        )

    df, conversions = convertir_types(df)

    if conversions:

        rapport["transformations"].append(
            f"{len(conversions)} colonne(s) converties : "
            + ", ".join(
                f"{col} → {type_}"
                for col, type_ in conversions.items()
            )
        )

    rapport["dimensions_finales"] = list(df.shape)

    rapport["colonnes"] = list(df.columns)

    return df, rapport


# ================================================================
# POINT D'ENTREE PRINCIPAL (FICHIER COMPLET, TOUTES FEUILLES)
# ================================================================

def importer_fichier_intelligent(chemin, mode_transposition="automatique"):
    """
    Charge et nettoie intelligemment un fichier CSV ou Excel
    (toutes feuilles). Retourne un dictionnaire :

        {
            "feuilles": {
                nom_feuille: {
                    "df_original": DataFrame brut (jamais modifié),
                    "df_propre": DataFrame nettoyé,
                    "rapport": {...},
                }
            },
            "relations_entre_feuilles": [...],
            "feuille_principale": nom_de_la_feuille_retenue,
        }

    La "feuille principale" est celle qui contient le plus de
    données exploitables après nettoyage : c'est celle utilisée
    par défaut pour la suite du pipeline Machine Learning tant que
    l'utilisateur n'en choisit pas une autre explicitement.
    """

    feuilles_brutes = charger_fichier_brut(chemin)

    resultat = {
        "feuilles": {},
        "relations_entre_feuilles": [],
        "feuille_principale": None,
    }

    dataframes_propres = {}

    for nom_feuille, df_brut in feuilles_brutes.items():

        df_propre, rapport = analyser_et_nettoyer_feuille(
            df_brut, mode_transposition=mode_transposition
        )

        resultat["feuilles"][nom_feuille] = {
            "df_original": df_brut,
            "df_propre": df_propre,
            "rapport": rapport,
        }

        dataframes_propres[nom_feuille] = df_propre

    if len(dataframes_propres) > 1:

        resultat["relations_entre_feuilles"] = (
            detecter_relations_entre_feuilles(dataframes_propres)
        )

    # Sélection de la feuille principale : la plus grande en
    # nombre de cellules exploitables (lignes × colonnes) après
    # nettoyage.
    meilleure_feuille = max(
        dataframes_propres,
        key=lambda nom: (
            dataframes_propres[nom].shape[0]
            * dataframes_propres[nom].shape[1]
        ),
    )

    resultat["feuille_principale"] = meilleure_feuille

    return resultat


def generer_apercu(df_original, df_propre, rapport, transposition_effectuee):
    """
    Construit le résumé affiché avant analyse (cahier des charges,
    section 5) : dimensions, types de variables, valeurs
    manquantes, doublons, transformations effectuées.
    """

    numeriques = list(
        df_propre.select_dtypes(include=np.number).columns
    )

    dates = list(
        df_propre.select_dtypes(
            include=["datetime", "datetimetz"]
        ).columns
    )

    categorielles = [
        c
        for c in df_propre.columns
        if c not in numeriques and c not in dates
    ]

    return {
        "lignes": int(df_propre.shape[0]),
        "colonnes": int(df_propre.shape[1]),
        "lignes_originales": int(df_original.shape[0]),
        "colonnes_originales": int(df_original.shape[1]),
        "valeurs_manquantes": int(df_propre.isna().sum().sum()),
        "doublons": int(df_propre.duplicated().sum()),
        "variables_numeriques": numeriques,
        "variables_categorielles": categorielles,
        "variables_temporelles": dates,
        "transformations": rapport.get("transformations", []),
        "transposition_effectuee": bool(transposition_effectuee),
        "apercu": df_propre.head(10).to_dict(orient="records"),
    }
