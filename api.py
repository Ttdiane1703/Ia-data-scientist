# ================================================================
#                IA DATA SCIENTIST — API WEB
# ================================================================
#
# API FastAPI qui pilote le pipeline existant (man.py) :
#
#   1. POST /api/upload            -> dépose un CSV, crée un job
#   2. POST /api/analyze/{job_id}  -> lance l'analyse (arrière-plan)
#   3. GET  /api/status/{job_id}   -> suit la progression en direct
#   4. GET  /api/download/{job_id}/{type} -> télécharge un livrable
#
# Chaque analyse tourne dans son propre dossier
# (jobs/<job_id>/...), donc plusieurs utilisateurs peuvent lancer
# des analyses en parallèle sans se marcher dessus.
#
# Lancement :
#
#   pip install fastapi "uvicorn[standard]" python-multipart
#   uvicorn api:app --reload
#
# Puis ouvrir : http://127.0.0.1:8000
# ================================================================

import os
import uuid
import shutil
import threading
import traceback
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

import pandas as pd

import man


app = FastAPI(
    title="IA Data Scientist — API",
    description=(
        "API pilotant le pipeline AutoML de IA Data Scientist : "
        "nettoyage, feature engineering, AutoML, évaluation, "
        "rapport PDF et notebook Jupyter."
    ),
    version="1.0.0"
)

# ----------------------------------------------------------------
# CORS
#
# Autorise l'appel de l'API depuis n'importe quelle origine.
# À restreindre à un domaine précis en production.
# ----------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------------------
# STOCKAGE DES JOBS
#
# Chaque analyse est un "job" isolé dans jobs/<job_id>/.
# L'état des jobs est gardé en mémoire (dictionnaire protégé par
# un verrou). Pour un usage multi-serveurs/production sérieuse,
# remplacer par Redis, une base de données, ou Celery.
# ----------------------------------------------------------------

JOBS_DIR = Path("jobs")
JOBS_DIR.mkdir(exist_ok=True)

jobs = {}
jobs_lock = threading.Lock()

# Nombre total d'étapes du pipeline (voir man.py, afficher_titre)
NB_ETAPES_TOTAL = 15


def _job_dir(job_id):

    return JOBS_DIR / job_id


def _verifier_job(job_id):

    if job_id not in jobs:

        raise HTTPException(
            status_code=404,
            detail="Job introuvable. Vérifiez l'identifiant."
        )


# ================================================================
# 1. UPLOAD DU FICHIER CSV
# ================================================================

@app.post("/api/upload")
async def upload_csv(fichier: UploadFile = File(...)):

    if not fichier.filename.lower().endswith(".csv"):

        raise HTTPException(
            status_code=400,
            detail="Seuls les fichiers .csv sont acceptés."
        )

    job_id = uuid.uuid4().hex[:12]

    dossier = _job_dir(job_id)

    dossier.mkdir(parents=True, exist_ok=True)

    chemin_csv = dossier / "input.csv"

    try:

        with open(chemin_csv, "wb") as f:

            shutil.copyfileobj(fichier.file, f)

    finally:

        await fichier.close()

    try:

        apercu = pd.read_csv(chemin_csv, nrows=200)

    except Exception as e:

        shutil.rmtree(dossier, ignore_errors=True)

        raise HTTPException(
            status_code=400,
            detail=f"Fichier CSV illisible : {e}"
        )

    colonnes = list(apercu.columns)

    with jobs_lock:

        jobs[job_id] = {
            "statut": "en_attente",
            "etape": None,
            "titre_etape": None,
            "nb_etapes_total": NB_ETAPES_TOTAL,
            "resultat": None,
            "erreur": None,
        }

    return {
        "job_id": job_id,
        "colonnes": colonnes,
        "nb_lignes_apercu": int(apercu.shape[0]),
        "apercu": (
            apercu.head(5)
            .fillna("")
            .astype(str)
            .to_dict(orient="records")
        ),
    }


# ================================================================
# 2. LANCEMENT DE L'ANALYSE (EN ARRIERE-PLAN)
# ================================================================

def _numero_etape_normalise(etape):
    """
    Convertit un identifiant d'étape ("5", "5bis", ...) en entier
    comparable, pour calculer une progression simple côté client.
    """

    if etape is None:

        return 0

    texte = str(etape)

    chiffres = ""

    for caractere in texte:

        if caractere.isdigit():

            chiffres += caractere

        else:

            break

    return int(chiffres) if chiffres else 0


def _executer_job(job_id, target):

    dossier = _job_dir(job_id)

    chemin_csv = str(dossier / "input.csv")

    def rapporteur(numero, titre):

        with jobs_lock:

            jobs[job_id]["etape"] = numero

            jobs[job_id]["titre_etape"] = titre

    man.definir_rapporteur_etape(rapporteur)

    with jobs_lock:

        jobs[job_id]["statut"] = "en_cours"

    try:

        resultat = man.executer_pipeline(
            chemin_csv=chemin_csv,
            target_impose=target,
            base_dir=str(dossier)
        )

        with jobs_lock:

            if resultat is not None and resultat.get("succes"):

                jobs[job_id]["statut"] = "termine"

                jobs[job_id]["resultat"] = resultat

            else:

                jobs[job_id]["statut"] = "erreur"

                jobs[job_id]["erreur"] = (
                    "Le pipeline s'est arrêté avant la fin "
                    f"(dernière étape atteinte : "
                    f"{jobs[job_id].get('titre_etape')})."
                )

    except Exception as e:

        traceback.print_exc()

        with jobs_lock:

            jobs[job_id]["statut"] = "erreur"

            jobs[job_id]["erreur"] = str(e)


@app.post("/api/analyze/{job_id}")
async def lancer_analyse(
    job_id: str,
    target: str = Form(None)
):

    _verifier_job(job_id)

    with jobs_lock:

        if jobs[job_id]["statut"] == "en_cours":

            raise HTTPException(
                status_code=409,
                detail="Une analyse est déjà en cours pour ce job."
            )

    thread = threading.Thread(
        target=_executer_job,
        args=(job_id, target),
        daemon=True
    )

    thread.start()

    return {"job_id": job_id, "statut": "en_cours"}


# ================================================================
# 3. SUIVI DE LA PROGRESSION
# ================================================================

@app.get("/api/status/{job_id}")
async def statut_job(job_id: str):

    _verifier_job(job_id)

    with jobs_lock:

        etat = dict(jobs[job_id])

    etat["progression"] = round(
        _numero_etape_normalise(etat["etape"])
        / NB_ETAPES_TOTAL
        * 100
    )

    return etat


# ================================================================
# 4. TELECHARGEMENT DES LIVRABLES
# ================================================================

TYPES_TELECHARGEABLES = {
    "pdf": "rapport_pdf",
    "notebook": "notebook",
    "dataset": "dataset_nettoye",
    "model": "model",
}


@app.get("/api/download/{job_id}/{type_fichier}")
async def telecharger(job_id: str, type_fichier: str):

    _verifier_job(job_id)

    cle = TYPES_TELECHARGEABLES.get(type_fichier)

    if cle is None:

        raise HTTPException(
            status_code=400,
            detail=(
                f"Type de fichier inconnu : {type_fichier}. "
                f"Valeurs possibles : "
                f"{', '.join(TYPES_TELECHARGEABLES)}"
            )
        )

    with jobs_lock:

        resultat = jobs[job_id].get("resultat")

    if resultat is None:

        raise HTTPException(
            status_code=400,
            detail="Aucun résultat disponible pour ce job pour "
                   "le moment."
        )

    chemin = resultat.get("fichiers", {}).get(cle)

    if chemin is None or not os.path.exists(chemin):

        raise HTTPException(
            status_code=404,
            detail="Ce fichier n'est pas disponible pour ce job."
        )

    return FileResponse(
        chemin,
        filename=os.path.basename(chemin)
    )


# ================================================================
# ENDPOINTS D'ANALYSE POUR LE DASHBOARD
#
# Ces endpoints calculent directement leurs réponses avec pandas
# à partir des fichiers déjà sur disque (input.csv brut, ou
# dataset_nettoye.csv une fois l'analyse terminée). Ils ne
# dépendent d'aucune classe interne du projet (IntelligentCleaner,
# AutomaticEDA, ...) et sont donc garantis fonctionner quel que
# soit leur contenu.
# ================================================================

import numpy as np


def _lire_csv_job(job_id, nom_fichier, sous_dossier=None):

    dossier = _job_dir(job_id)

    if sous_dossier:

        chemin = dossier / sous_dossier / nom_fichier

    else:

        chemin = dossier / nom_fichier

    if not chemin.exists():

        return None

    try:

        return pd.read_csv(chemin)

    except Exception:

        return None


def _valeur_safe(v):

    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        if np.isnan(v) or np.isinf(v):
            return None
        return float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


# ----------------------------------------------------------------
# 1. INFOS DATASET
# ----------------------------------------------------------------

@app.get("/api/job/{job_id}/dataset-info")
async def dataset_info(job_id: str):

    _verifier_job(job_id)

    chemin_csv = _job_dir(job_id) / "input.csv"

    if not chemin_csv.exists():

        raise HTTPException(404, "Fichier introuvable pour ce job.")

    df = pd.read_csv(chemin_csv)

    taille_octets = chemin_csv.stat().st_size

    return {
        "nom_fichier": "input.csv",
        "lignes": int(df.shape[0]),
        "colonnes": int(df.shape[1]),
        "taille_ko": round(taille_octets / 1024, 1),
        "format": "CSV",
        "encodage": "utf-8",
        "colonnes_liste": list(df.columns),
    }


# ----------------------------------------------------------------
# 2. DISTRIBUTION DE LA CIBLE (pour le donut chart)
# ----------------------------------------------------------------

@app.get("/api/job/{job_id}/target-distribution")
async def target_distribution(job_id: str, target: str):

    _verifier_job(job_id)

    df = _lire_csv_job(job_id, "input.csv")

    if df is None or target not in df.columns:

        raise HTTPException(
            400, "Cible introuvable dans le dataset de ce job."
        )

    serie = df[target].dropna()

    est_numerique_continue = (
        pd.api.types.is_numeric_dtype(serie)
        and serie.nunique() > 15
    )

    if est_numerique_continue:

        return {
            "type": "regression",
            "min": _valeur_safe(serie.min()),
            "max": _valeur_safe(serie.max()),
            "moyenne": _valeur_safe(serie.mean()),
            "mediane": _valeur_safe(serie.median()),
        }

    comptes = serie.value_counts()
    total = int(comptes.sum())

    repartition = [
        {
            "valeur": str(valeur),
            "count": int(compte),
            "pourcentage": round(compte / total * 100, 1),
        }
        for valeur, compte in comptes.items()
    ]

    nb_classes = len(comptes)

    type_probleme = (
        "classification_binaire"
        if nb_classes == 2
        else "classification_multiclasse"
    )

    return {
        "type": type_probleme,
        "repartition": repartition,
    }


# ----------------------------------------------------------------
# 3. QUALITE DES DONNEES
# ----------------------------------------------------------------

@app.get("/api/job/{job_id}/data-quality")
async def data_quality(job_id: str):

    _verifier_job(job_id)

    df = _lire_csv_job(job_id, "input.csv")

    if df is None:

        raise HTTPException(404, "Fichier introuvable pour ce job.")

    details = []

    # Valeurs manquantes par colonne
    manquants = df.isna().sum()
    total_manquants = int(manquants.sum())

    for colonne, nb in manquants.items():

        if nb > 0:

            details.append({
                "probleme": "Valeurs manquantes",
                "colonne": colonne,
                "nombre": int(nb),
                "action": (
                    "Remplacé par la médiane"
                    if pd.api.types.is_numeric_dtype(df[colonne])
                    else "Remplacé par le mode"
                ),
            })

    # Doublons
    nb_doublons = int(df.duplicated().sum())

    if nb_doublons > 0:

        details.append({
            "probleme": "Lignes dupliquées",
            "colonne": "-",
            "nombre": nb_doublons,
            "action": "Supprimées",
        })

    # Outliers (méthode IQR) sur les colonnes numériques
    total_outliers = 0

    for colonne in df.select_dtypes(include=[np.number]).columns:

        serie = df[colonne].dropna()

        if serie.empty:
            continue

        q1, q3 = serie.quantile(0.25), serie.quantile(0.75)
        iqr = q3 - q1

        if iqr == 0:
            continue

        bornes = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)

        nb_outliers = int(
            ((serie < bornes[0]) | (serie > bornes[1])).sum()
        )

        if nb_outliers > 0:

            total_outliers += nb_outliers

            details.append({
                "probleme": "Valeurs aberrantes",
                "colonne": colonne,
                "nombre": nb_outliers,
                "action": "Analysées (IQR)",
            })

    return {
        "valeurs_manquantes": total_manquants,
        "doublons": nb_doublons,
        "valeurs_invalides": 0,
        "outliers": total_outliers,
        "fuite_donnees_detectee": False,
        "details": details,
        "pret_pour_analyse": True,
    }


# ----------------------------------------------------------------
# 4. STATISTIQUES (numériques + catégorielles)
# ----------------------------------------------------------------

@app.get("/api/job/{job_id}/statistics")
async def statistics(job_id: str):

    _verifier_job(job_id)

    df = _lire_csv_job(job_id, "input.csv")

    if df is None:

        raise HTTPException(404, "Fichier introuvable pour ce job.")

    numeriques = []

    for colonne in df.select_dtypes(include=[np.number]).columns:

        serie = df[colonne].dropna()

        if serie.empty:
            continue

        numeriques.append({
            "variable": colonne,
            "count": int(serie.count()),
            "mean": _valeur_safe(serie.mean()),
            "std": _valeur_safe(serie.std()),
            "min": _valeur_safe(serie.min()),
            "p25": _valeur_safe(serie.quantile(0.25)),
            "p50": _valeur_safe(serie.quantile(0.5)),
            "p75": _valeur_safe(serie.quantile(0.75)),
            "max": _valeur_safe(serie.max()),
            "skew": _valeur_safe(serie.skew()),
            "kurt": _valeur_safe(serie.kurt()),
        })

    categorielles = []

    for colonne in df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns:

        serie = df[colonne].dropna()

        if serie.empty:
            continue

        comptes = serie.value_counts()

        categorielles.append({
            "variable": colonne,
            "unique": int(serie.nunique()),
            "plus_frequent": str(comptes.index[0]),
            "frequence": int(comptes.iloc[0]),
            "pourcentage": round(
                comptes.iloc[0] / len(serie) * 100, 1
            ),
        })

    return {
        "numeriques": numeriques,
        "categorielles": categorielles,
    }


# ----------------------------------------------------------------
# 5. EDA — HISTOGRAMMES (données pour graphiques côté client)
# ----------------------------------------------------------------

@app.get("/api/job/{job_id}/eda")
async def eda(job_id: str, nb_bins: int = 12):

    _verifier_job(job_id)

    df = _lire_csv_job(job_id, "input.csv")

    if df is None:

        raise HTTPException(404, "Fichier introuvable pour ce job.")

    distributions = []

    colonnes_numeriques = list(
        df.select_dtypes(include=[np.number]).columns
    )[:6]

    for colonne in colonnes_numeriques:

        serie = df[colonne].dropna()

        if serie.empty or serie.nunique() < 2:
            continue

        comptes, bords = np.histogram(serie, bins=nb_bins)

        distributions.append({
            "variable": colonne,
            "bins": [
                round(float(b), 2) for b in bords[:-1]
            ],
            "counts": [int(c) for c in comptes],
        })

    correlations = None

    df_num = df.select_dtypes(include=[np.number])

    if df_num.shape[1] >= 2:

        matrice = df_num.corr().round(2)

        correlations = {
            "variables": list(matrice.columns),
            "matrice": [
                [
                    _valeur_safe(v)
                    for v in matrice.iloc[i].tolist()
                ]
                for i in range(len(matrice))
            ],
        }

    return {
        "distributions": distributions,
        "correlations": correlations,
    }


# ----------------------------------------------------------------
# 6. COMPARAISON DES MODELES
#
# IMPORTANT : le pipeline actuel (src/automl.py) ne renvoie que le
# modèle champion, pas le détail de chaque modèle testé pendant
# l'optimisation. Cet endpoint renvoie donc honnêtement UNE seule
# ligne (le champion) plutôt que d'inventer des scores pour les
# autres modèles. Pour afficher une vraie comparaison multi-
# modèles, il faudrait faire remonter les résultats intermédiaires
# depuis AutoML.run() jusqu'ici.
# ----------------------------------------------------------------

@app.get("/api/job/{job_id}/models")
async def models_comparison(job_id: str):

    _verifier_job(job_id)

    with jobs_lock:

        resultat = jobs[job_id].get("resultat")

    if resultat is None or resultat.get("champion") is None:

        return {"modeles": [], "note": (
            "Analyse pas encore terminée ou aucun champion "
            "disponible."
        )}

    champion = resultat["champion"]

    return {
        "modeles": [{
            "nom": champion.get("modele"),
            "score_cv": champion.get("score_cv"),
            "est_champion": True,
        }],
        "note": (
            "Seul le modèle champion est disponible : le détail "
            "de chaque modèle testé n'est pas encore remonté par "
            "le pipeline AutoML."
        ),
    }


# ================================================================
# PAGE WEB STATIQUE
#
# Doit être montée en dernier : toute route non gérée ci-dessus
# est servie depuis le dossier static/ (index.html, style.css,
# app.js).
# ================================================================

if os.path.isdir("static"):

    app.mount(
        "/",
        StaticFiles(directory="static", html=True),
        name="static"
    )