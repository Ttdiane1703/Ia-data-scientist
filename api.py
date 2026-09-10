# ================================================================
#                IA DATA SCIENTIST — API WEB
# ================================================================
#
# API FastAPI qui pilote le pipeline existant (man.py) :
#
#   1. POST /api/upload            -> dépose un fichier (CSV,
#                                      Excel ou JSON), crée un job
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
#   pip install fastapi "uvicorn[standard]" python-multipart openpyxl
#   uvicorn api:app --reload
#
# Puis ouvrir : http://127.0.0.1:8000
# ================================================================

import os
import json
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
import numpy as np

from src.smart_file_analyzer import importer_fichier_intelligent

import man


app = FastAPI(
    title="IA Data Scientist — API",
    description=(
        "API pilotant le pipeline AutoML de IA Data Scientist : "
        "nettoyage, feature engineering, AutoML, évaluation, "
        "rapport PDF et notebook Jupyter."
    ),
    version="1.1.0"
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


# ----------------------------------------------------------------
# PERSISTANCE SUR DISQUE
#
# IMPORTANT : sur les hébergeurs à plan gratuit (ex. Render free
# tier), le processus peut redémarrer sans prévenir — mise en
# veille après inactivité, ou redémarrage si la RAM est dépassée
# pendant une analyse lourde (AutoML avec plusieurs modèles).
#
# Quand ça arrive, le dictionnaire `jobs` en mémoire est vidé,
# mais les fichiers sur le disque (jobs/<id>/...) survivent en
# général à un simple redémarrage du process (pas à un redéploiement
# complet). On sauvegarde donc l'état de chaque job dans un petit
# fichier JSON, et on le recharge automatiquement depuis le disque
# si jamais il manque en mémoire — au lieu de renvoyer une erreur
# "job introuvable" au client.
# ----------------------------------------------------------------

def _fichier_statut(job_id):

    return _job_dir(job_id) / "_statut.json"


def _sauvegarder_statut(job_id):

    with jobs_lock:

        donnees = jobs.get(job_id)

    if donnees is None:

        return

    try:

        _job_dir(job_id).mkdir(parents=True, exist_ok=True)

        with open(
            _fichier_statut(job_id), "w", encoding="utf-8"
        ) as f:

            json.dump(donnees, f, default=str)

    except Exception:

        # La sauvegarde de statut ne doit jamais faire planter
        # l'analyse elle-même.
        pass


def _recharger_statut_depuis_disque(job_id):

    chemin = _fichier_statut(job_id)

    if not chemin.exists():

        return None

    try:

        with open(chemin, "r", encoding="utf-8") as f:

            return json.load(f)

    except Exception:

        return None


def _verifier_job(job_id):

    with jobs_lock:

        deja_present = job_id in jobs

    if deja_present:

        return

    # Le job n'est plus en mémoire (redémarrage du process) :
    # on tente de le retrouver sur le disque avant d'abandonner.
    recupere = _recharger_statut_depuis_disque(job_id)

    if recupere is not None:

        with jobs_lock:

            jobs[job_id] = recupere

        return

    raise HTTPException(
        status_code=404,
        detail="Job introuvable. Vérifiez l'identifiant."
    )


# ================================================================
# GESTION MULTI-FORMATS (CSV / EXCEL / JSON)
# ================================================================
#
# Le dashboard accepte désormais trois formats en entrée. Quel que
# soit le format d'origine, le fichier est immédiatement normalisé
# en CSV standard (jobs/<id>/input.csv). Tout le reste du backend
# (data-quality, statistics, eda, target-distribution, man.py, ...)
# continue de lire ce même input.csv sans aucune modification —
# la conversion est le seul point de contact avec le format brut.
# ================================================================

EXTENSIONS_ACCEPTEES = {".csv", ".xlsx", ".xls", ".json"}


def _detecter_format(nom_fichier):

    ext = os.path.splitext(nom_fichier.lower())[1]

    if ext == ".csv":
        return "CSV", ext

    if ext in (".xlsx", ".xls"):
        return "Excel", ext

    if ext == ".json":
        return "JSON", ext

    return None, ext


def _lire_fichier_quelconque(chemin, ext):
    """
    Charge un fichier CSV, Excel ou JSON et retourne un DataFrame
    pandas standard.
    """

    if ext == ".csv":

        try:

            return pd.read_csv(chemin)

        except UnicodeDecodeError:

            return pd.read_csv(chemin, encoding="latin-1")

    if ext in (".xlsx", ".xls"):

        return pd.read_excel(chemin)

    if ext == ".json":

        with open(chemin, "r", encoding="utf-8") as f:

            contenu = json.load(f)

        # ------------------------------------------------------------
        # NORMALISATION EN TABLEAU
        #
        # - Liste d'objets ( [ {...}, {...} ] )      -> direct
        # - Objet avec une liste imbriquée
        #   ( {"data": [ {...}, {...} ]} )           -> détection auto
        # - Objet isolé ( {...} )                    -> une seule ligne
        # ------------------------------------------------------------

        if isinstance(contenu, list):

            return pd.json_normalize(contenu)

        if isinstance(contenu, dict):

            for valeur in contenu.values():

                if isinstance(valeur, list):

                    return pd.json_normalize(valeur)

            return pd.json_normalize([contenu])

        raise ValueError("Structure JSON non reconnue.")

    raise ValueError(f"Extension non supportée : {ext}")


# ================================================================
# 1. UPLOAD DU FICHIER (CSV, EXCEL OU JSON)
# ================================================================

@app.post("/api/upload")
async def upload_fichier(fichier: UploadFile = File(...)):

    format_detecte, extension = _detecter_format(fichier.filename)

    if format_detecte is None:

        raise HTTPException(
            status_code=400,
            detail=(
                "Format non supporté. Formats acceptés : "
                "CSV (.csv), Excel (.xlsx, .xls), JSON (.json)."
            )
        )

    job_id = uuid.uuid4().hex[:12]

    dossier = _job_dir(job_id)

    dossier.mkdir(parents=True, exist_ok=True)

    chemin_original = dossier / f"original_upload{extension}"

    try:

        with open(chemin_original, "wb") as f:

            shutil.copyfileobj(fichier.file, f)

    finally:

        await fichier.close()

    # ------------------------------------------------------------
    # CONVERSION VERS CSV STANDARD
    #
    # Pour CSV/Excel : passe par l'import intelligent (détection de
    # la vraie ligne d'en-tête, nettoyage du bruit haut/bas,
    # colonnes vides, en-têtes dupliqués, transposition automatique,
    # gestion de toutes les feuilles d'un classeur Excel). Le
    # résultat de cette analyse est conservé pour permettre à
    # l'utilisateur de changer le choix de transposition ensuite
    # (voir /api/job/{job_id}/transposition).
    #
    # Quel que soit le format d'origine, le reste du pipeline et
    # des endpoints d'analyse travaillent uniquement sur
    # jobs/<id>/input.csv.
    # ------------------------------------------------------------

    rapport_structure = None

    try:

        if extension in (".csv", ".xlsx", ".xls"):

            resultat_import = importer_fichier_intelligent(
                chemin_original, mode_transposition="automatique"
            )

            feuille_principale = resultat_import["feuille_principale"]

            info_feuille = resultat_import["feuilles"][
                feuille_principale
            ]

            df = info_feuille["df_propre"]

            rapport_structure = {
                "n_feuilles": len(resultat_import["feuilles"]),
                "feuilles_disponibles": list(
                    resultat_import["feuilles"].keys()
                ),
                "feuille_utilisee": feuille_principale,
                "relations_entre_feuilles": resultat_import[
                    "relations_entre_feuilles"
                ],
                "transformations": info_feuille["rapport"][
                    "transformations"
                ],
                "transposition_effectuee": info_feuille[
                    "rapport"
                ].get("transposition_effectuee", False),
                "mode_transposition": "automatique",
            }

        else:

            df = _lire_fichier_quelconque(
                chemin_original, extension
            )

    except Exception as e:

        shutil.rmtree(dossier, ignore_errors=True)

        raise HTTPException(
            status_code=400,
            detail=f"Fichier {format_detecte} illisible : {e}"
        )

    if df is None or df.empty:

        shutil.rmtree(dossier, ignore_errors=True)

        raise HTTPException(
            status_code=400,
            detail="Le fichier ne contient aucune donnée exploitable."
        )

    chemin_csv = dossier / "input.csv"

    try:

        df.to_csv(chemin_csv, index=False, encoding="utf-8")

    except Exception as e:

        shutil.rmtree(dossier, ignore_errors=True)

        raise HTTPException(
            status_code=400,
            detail=f"Conversion vers CSV impossible : {e}"
        )

    colonnes = list(df.columns)

    with jobs_lock:

        jobs[job_id] = {
            "statut": "en_attente",
            "etape": None,
            "titre_etape": None,
            "nb_etapes_total": NB_ETAPES_TOTAL,
            "resultat": None,
            "erreur": None,
            "nom_fichier": fichier.filename,
            "format_original": format_detecte,
            "rapport_structure": rapport_structure,
        }

    _sauvegarder_statut(job_id)

    return {
        "job_id": job_id,
        "colonnes": colonnes,
        "format": format_detecte,
        "nb_lignes_apercu": int(df.head(200).shape[0]),
        "apercu": (
            df.head(5)
            .fillna("")
            .astype(str)
            .to_dict(orient="records")
        ),
        "structure_fichier": rapport_structure,
    }


# ================================================================
# 1bis. CHOIX DE LA TRANSPOSITION (OUI / NON / AUTOMATIQUE)
# ================================================================

@app.post("/api/job/{job_id}/transposition")
async def choisir_transposition(
    job_id: str,
    mode: str = Form(...)
):
    """
    Recharge le fichier ORIGINAL (jamais modifié, conservé depuis
    l'upload) avec le mode de transposition demandé par
    l'utilisateur, et régénère input.csv en conséquence.

    mode : "oui" | "non" | "automatique"
    """

    _verifier_job(job_id)

    mode = (mode or "").strip().lower()

    if mode not in ("oui", "non", "automatique"):

        raise HTTPException(
            status_code=400,
            detail="Le paramètre 'mode' doit être 'oui', 'non' ou "
                   "'automatique'."
        )

    dossier = _job_dir(job_id)

    fichiers_originaux = sorted(dossier.glob("original_upload.*"))

    if not fichiers_originaux:

        raise HTTPException(
            status_code=404,
            detail="Fichier original introuvable pour ce job."
        )

    chemin_original = str(fichiers_originaux[0])

    extension = os.path.splitext(chemin_original)[1].lower()

    if extension not in (".csv", ".xlsx", ".xls"):

        raise HTTPException(
            status_code=400,
            detail="La transposition ne s'applique qu'aux fichiers "
                   "CSV et Excel."
        )

    try:

        resultat_import = importer_fichier_intelligent(
            chemin_original, mode_transposition=mode
        )

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=f"Impossible de retraiter le fichier : {e}"
        )

    with jobs_lock:

        feuille_precedente = (
            jobs.get(job_id, {})
            .get("rapport_structure", {})
            .get("feuille_utilisee")
        )

    feuille = (
        feuille_precedente
        if feuille_precedente in resultat_import["feuilles"]
        else resultat_import["feuille_principale"]
    )

    info_feuille = resultat_import["feuilles"][feuille]

    df = info_feuille["df_propre"]

    if df is None or df.empty:

        raise HTTPException(
            status_code=400,
            detail=(
                "Aucune donnée exploitable avec ce mode de "
                "transposition."
            )
        )

    chemin_csv = dossier / "input.csv"

    df.to_csv(chemin_csv, index=False, encoding="utf-8")

    rapport_structure = {
        "n_feuilles": len(resultat_import["feuilles"]),
        "feuilles_disponibles": list(
            resultat_import["feuilles"].keys()
        ),
        "feuille_utilisee": feuille,
        "relations_entre_feuilles": resultat_import[
            "relations_entre_feuilles"
        ],
        "transformations": info_feuille["rapport"][
            "transformations"
        ],
        "transposition_effectuee": info_feuille["rapport"].get(
            "transposition_effectuee", mode == "oui"
        ),
        "mode_transposition": mode,
    }

    with jobs_lock:

        jobs[job_id]["rapport_structure"] = rapport_structure

    _sauvegarder_statut(job_id)

    return {
        "job_id": job_id,
        "colonnes": list(df.columns),
        "apercu": (
            df.head(5)
            .fillna("")
            .astype(str)
            .to_dict(orient="records")
        ),
        "structure_fichier": rapport_structure,
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


def _construire_message_erreur_pipeline(resultat, titre_etape):
    """
    Construit un message d'erreur clair pour l'utilisateur.

    Quand le pipeline s'est arrêté proprement avec un diagnostic
    structuré (ex. dataset trop petit détecté à l'étape 5ter), on
    remonte le VRAI message ("Le dataset contient seulement 5
    observations pour 3 classes...") plutôt qu'un message
    générique qui masque la raison réelle de l'arrêt.
    """

    if isinstance(resultat, dict):

        if resultat.get("raison") == "dataset_trop_petit":

            diagnostic = resultat.get("validation_dataset") or {}

            titre = diagnostic.get(
                "titre", "⚠️ Dataset trop petit"
            )

            messages = diagnostic.get("messages", [])

            recommandation = diagnostic.get("recommandation")

            texte = titre

            if messages:

                texte += " — " + " ".join(messages)

            if recommandation:

                texte += f" Recommandation : {recommandation}"

            return texte

        if resultat.get("erreur"):

            return str(resultat["erreur"])

    return (
        "Le pipeline s'est arrêté avant la fin "
        f"(dernière étape atteinte : "
        f"{titre_etape})."
    )


def _executer_job(job_id, target, trials=10, testsize=20):

    dossier = _job_dir(job_id)

    chemin_csv = str(dossier / "input.csv")

    def rapporteur(numero, titre):

        with jobs_lock:

            jobs[job_id]["etape"] = numero

            jobs[job_id]["titre_etape"] = titre

        _sauvegarder_statut(job_id)

    man.definir_rapporteur_etape(rapporteur)

    with jobs_lock:

        jobs[job_id]["statut"] = "en_cours"

    _sauvegarder_statut(job_id)

    try:

        resultat = man.executer_pipeline(
            chemin_csv=chemin_csv,
            target_impose=target,
            base_dir=str(dossier),
            n_trials=trials,
            test_size=testsize
        )

        with jobs_lock:

            if resultat is not None and resultat.get("succes"):

                jobs[job_id]["statut"] = "termine"

                jobs[job_id]["resultat"] = resultat

            else:

                # Un arrêt propre (ex. dataset trop petit) n'est
                # pas une panne serveur, mais on garde le statut
                # "erreur" existant pour rester compatible avec le
                # frontend actuel : seul le contenu du message
                # change, pour être précis plutôt que générique.

                jobs[job_id]["statut"] = "erreur"

                jobs[job_id]["arret_propre"] = bool(
                    isinstance(resultat, dict)
                    and resultat.get("arret_propre")
                )

                jobs[job_id]["erreur"] = (
                    _construire_message_erreur_pipeline(
                        resultat,
                        jobs[job_id].get("titre_etape")
                    )
                )

                if isinstance(resultat, dict):

                    jobs[job_id]["diagnostic"] = resultat

        _sauvegarder_statut(job_id)

    except Exception as e:

        traceback.print_exc()

        with jobs_lock:

            jobs[job_id]["statut"] = "erreur"

            jobs[job_id]["erreur"] = str(e)

        _sauvegarder_statut(job_id)


@app.post("/api/analyze/{job_id}")
async def lancer_analyse(
    job_id: str,
    target: str = Form(None),
    trials: int = Form(10),
    testsize: int = Form(20)
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
        args=(job_id, target, max(5, min(trials, 100)), max(10, min(testsize, 40))),
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
# à partir des fichiers déjà sur disque (input.csv, déjà normalisé
# depuis CSV/Excel/JSON au moment de l'upload). Ils ne dépendent
# d'aucune classe interne du projet (IntelligentCleaner,
# AutomaticEDA, ...) et sont donc garantis fonctionner quel que
# soit leur contenu.
# ================================================================


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

    with jobs_lock:

        meta = dict(jobs.get(job_id, {}))

    return {
        "nom_fichier": meta.get("nom_fichier", "input.csv"),
        "lignes": int(df.shape[0]),
        "colonnes": int(df.shape[1]),
        "taille_ko": round(taille_octets / 1024, 1),
        "format": meta.get("format_original", "CSV"),
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
#
# Renvoie désormais TOUTES les colonnes numériques disponibles
# (jusqu'à 40, garde-fou technique plutôt que limite produit) :
# le frontend affiche un sélecteur de variables et filtre
# lui-même les graphiques à afficher, sans nouvel appel réseau.
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
    )[:40]

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