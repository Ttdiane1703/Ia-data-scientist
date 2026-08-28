# ================================================================
#                     IA DATA SCIENTIST
# ================================================================
#
# Pipeline automatique :
#
# 1. Chargement des données
# 2. Profilage
# 3. Nettoyage
# 4. EDA
# 5. Détection du problème
# 6. Préparation des données
# 7. Feature Engineering
# 8. Train / Test
# 9. Préparation des modèles
# 10. AutoML + optimisation
# 11. Évaluation
# 12. Explicabilité
# 13. Analyse des erreurs
# 14. Prédiction
# 15. Rapport
#
# ================================================================

import os
import traceback
import warnings

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")


# ================================================================
# IMPORTS DU PROJET
# ================================================================

from src.data_loader import DataLoader
from src.data_profiler import DataProfiler
from src.data_cleaner import IntelligentCleaner
from src.eda import AutomaticEDA
from src.problem_detector import ProblemDetector
from src.feature_engineering import FeatureEngineer
from src.trainer import ModelTrainer
from src.model_factory import ModelFactory
from src.automl import AutoML


# Imports optionnels
try:
    from src.evaluator import ModelEvaluator
except Exception:
    ModelEvaluator = None

try:
    from src.explainability import ModelExplainer
except Exception:
    ModelExplainer = None

try:
    from src.analyzer import ErrorAnalyzer
except Exception:
    ErrorAnalyzer = None

try:
    from src.predictor import Predictor
except Exception:
    Predictor = None

try:
    from src.auto_report import AutoReport
except Exception:
    AutoReport = None


# ================================================================
# CONFIGURATION
# ================================================================

N_TRIALS = 10

TEST_SIZE = 0.20

RANDOM_STATE = 42

REPORTS_DIR = "reports"


# ================================================================
# OUTILS
# ================================================================

def afficher_titre(numero, titre):

    print("\n")
    print("=" * 70)

    print(
        f"[{numero}/15] {titre}"
    )

    print("=" * 70)


def afficher_erreur(message, exception):

    print("\n")
    print("!" * 70)
    print("ERREUR")
    print("!" * 70)

    print(message)

    print(
        f"\nDétail : {exception}"
    )

    print("\nTraceback :")

    traceback.print_exc()


def executer_methode(
    objet,
    nom_methode,
    *args,
    **kwargs
):

    methode = getattr(
        objet,
        nom_methode
    )

    return methode(
        *args,
        **kwargs
    )


def trouver_methode(
    objet,
    noms
):

    for nom in noms:

        if hasattr(
            objet,
            nom
        ):

            return getattr(
                objet,
                nom
            )

    return None


def afficher_intro():

    print("\n")
    print("=" * 70)
    print("                         IA DATA SCIENTIST")
    print("=" * 70)

    print("""
        Analyse automatique de données

        Chargement • Nettoyage • EDA • Machine Learning
        AutoML • Optimisation • Explicabilité
        Analyse des erreurs • Prédiction • Rapport
    """)

    print("=" * 70)


# ================================================================
# CREATION DES DOSSIERS
# ================================================================

def creer_dossiers():

    dossiers = [
        "reports",
        "reports/eda",
        "reports/errors",
        "reports/models",
        "reports/predictions",
        "reports/explainability",
        "reports/final",
        "data/processed"
    ]

    for dossier in dossiers:

        os.makedirs(
            dossier,
            exist_ok=True
        )


# ================================================================
# ETAPE 1 — CHARGEMENT
# ================================================================

def charger_donnees():

    afficher_titre(
        1,
        "📂 CHARGEMENT DES DONNÉES"
    )

    chemin = input(
        "Entrez le chemin du fichier CSV : "
    ).strip()

    if not chemin:

        raise ValueError(
            "Aucun fichier CSV n'a été fourni."
        )

    if not os.path.exists(chemin):

        raise FileNotFoundError(
            f"Fichier introuvable : {chemin}"
        )

    loader = DataLoader()

    # Recherche automatique de la bonne méthode

    methode = trouver_methode(
        loader,
        [
            "load_csv",
            "charger_csv",
            "load",
            "charger",
            "read_csv"
        ]
    )

    if methode is None:

        raise AttributeError(
            "Aucune méthode de chargement compatible "
            "n'a été trouvée dans DataLoader."
        )

    df = methode(
        chemin
    )

    print("✅ Données chargées")

    print(
        f"📊 Lignes : {df.shape[0]}"
    )

    print(
        f"📋 Colonnes : {df.shape[1]}"
    )

    return df


# ================================================================
# ETAPE 2 — PROFILAGE
# ================================================================

def profiler_donnees(df):

    afficher_titre(
        2,
        "🔍 PROFILAGE AUTOMATIQUE"
    )

    profiler = DataProfiler()

    methode = trouver_methode(
        profiler,
        [
            "profile",
            "profil",
            "profiler",
            "analyze",
            "analyser"
        ]
    )

    if methode is None:

        raise AttributeError(
            "Aucune méthode de profilage compatible."
        )

    resultat = methode(
        df
    )

    print(
        "\n✅ Profilage terminé."
    )

    return resultat


# ================================================================
# ETAPE 3 — NETTOYAGE
# ================================================================

def nettoyer_donnees(df):

    afficher_titre(
        3,
        "🧹 NETTOYAGE INTELLIGENT"
    )

    cleaner = IntelligentCleaner()

    # Recherche des méthodes disponibles

    methode = trouver_methode(
        cleaner,
        [
            "fit_transform",
            "clean",
            "nettoyer",
            "transform"
        ]
    )

    if methode is None:

        raise AttributeError(
            "Aucune méthode de nettoyage compatible."
        )

    resultat = methode(
        df
    )

    # Certains cleaners retournent
    # directement le DataFrame.

    if isinstance(
        resultat,
        tuple
    ):

        df_clean = resultat[0]

    else:

        df_clean = resultat

    print(
        f"\n✅ Dataset nettoyé : "
        f"{df_clean.shape[0]} lignes × "
        f"{df_clean.shape[1]} colonnes"
    )

    return df_clean


# ================================================================
# ETAPE 3bis — EXPORT DU DATASET NETTOYE
# ================================================================

def exporter_dataset_nettoye(df_clean):

    chemin = os.path.join(
        "data",
        "processed",
        "dataset_nettoye.csv"
    )

    try:

        df_clean.to_csv(
            chemin,
            index=False,
            encoding="utf-8"
        )

        print(
            f"✅ Dataset nettoyé exporté : {chemin}"
        )

    except Exception as e:

        print(
            f"⚠️ Export du dataset nettoyé impossible : {e}"
        )

    return chemin


# ================================================================
# ETAPE 4 — EDA
# ================================================================

def effectuer_eda(df):

    afficher_titre(
        4,
        "📊 ANALYSE EXPLORATOIRE AUTOMATIQUE"
    )

    eda = AutomaticEDA()

    methode = trouver_methode(
        eda,
        [
            "fit_transform",
            "analyze",
            "analyse",
            "run",
            "execute"
        ]
    )

    if methode is None:

        raise AttributeError(
            "Aucune méthode EDA compatible."
        )

    try:

        resultat = methode(
            df
        )

    except TypeError:

        resultat = methode()

    print(
        "\n✅ Analyse exploratoire terminée."
    )

    return resultat


# ================================================================
# ETAPE 5 — DETECTION DU PROBLEME
# ================================================================

def detecter_probleme(df):

    afficher_titre(
        5,
        "🧠 DÉTECTION AUTOMATIQUE DU PROBLÈME"
    )

    detector = ProblemDetector()

    methode = trouver_methode(
        detector,
        [
            "detect",
            "detect_problem",
            "detect_problem_type",
            "analyze",
            "run"
        ]
    )

    if methode is None:

        raise AttributeError(
            "Aucune méthode de détection compatible."
        )

    resultat = methode(
        df
    )

    # ------------------------------------------------------------
    # SUPPORT DES DIFFERENTS FORMATS DE RETOUR
    # ------------------------------------------------------------

    target = None

    problem_type = None

    features = None

    if isinstance(
        resultat,
        dict
    ):

        target = (
            resultat.get("target")
            or resultat.get("target_column")
            or resultat.get("cible")
        )

        problem_type = (
            resultat.get("problem_type")
            or resultat.get("type")
            or resultat.get("problem")
        )

        features = (
            resultat.get("features")
            or resultat.get("feature_columns")
        )

    elif isinstance(
        resultat,
        tuple
    ):

        if len(resultat) >= 1:
            target = resultat[0]

        if len(resultat) >= 2:
            problem_type = resultat[1]

        if len(resultat) >= 3:
            features = resultat[2]

    # ------------------------------------------------------------
    # SI LE DETECTEUR STOCKE LES INFORMATIONS
    # ------------------------------------------------------------

    if target is None:

        for attribut in [
            "target",
            "target_column",
            "cible",
            "target_name"
        ]:

            if hasattr(
                detector,
                attribut
            ):

                target = getattr(
                    detector,
                    attribut
                )

                if target is not None:
                    break

    if problem_type is None:

        for attribut in [
            "problem_type",
            "type_probleme",
            "problem"
        ]:

            if hasattr(
                detector,
                attribut
            ):

                problem_type = getattr(
                    detector,
                    attribut
                )

                if problem_type is not None:
                    break

    if target is None:

        raise RuntimeError(
            "Impossible de déterminer automatiquement "
            "la colonne cible."
        )

    if problem_type is None:

        raise RuntimeError(
            "Impossible de déterminer automatiquement "
            "le type de problème."
        )

    print(
        f"\n🎯 Cible : {target}"
    )

    print(
        f"🧠 Type de problème : {problem_type}"
    )

    if features is not None:

        print(
            f"📊 Features potentielles : "
            f"{len(features)}"
        )

    return (
        target,
        problem_type,
        features
    )


# ================================================================
# ETAPE 5bis — CHOIX MANUEL DE LA CIBLE (OPTIONNEL)
# ================================================================
#
# Permet à l'utilisateur de choisir lui-même la colonne cible
# parmi les colonnes du dataset nettoyé, au lieu de dépendre
# uniquement de la détection automatique.
#
# Le type de problème (classification / régression) est ensuite
# déterminé à partir de cette cible choisie.
# ================================================================

def choisir_cible_manuellement(
    df,
    target_auto=None,
    problem_type_auto=None
):

    afficher_titre(
        "5bis",
        "🎯 CHOIX DE LA VARIABLE À PRÉDIRE"
    )

    colonnes = list(df.columns)

    print(
        "\nColonnes disponibles :"
    )

    for index, colonne in enumerate(colonnes):

        marqueur = (
            "  (suggestion automatique)"
            if colonne == target_auto
            else ""
        )

        print(
            f"   [{index}] {colonne}{marqueur}"
        )

    reponse = input(
        "\nEntrez le numéro ou le nom de la colonne à prédire "
        "(laisser vide pour garder la suggestion automatique) : "
    ).strip()

    if not reponse:

        return target_auto, problem_type_auto

    # Sélection par numéro
    if reponse.isdigit():

        index = int(reponse)

        if index < 0 or index >= len(colonnes):

            raise ValueError(
                f"Numéro de colonne invalide : {index}"
            )

        target = colonnes[index]

    else:

        # Sélection par nom
        if reponse not in colonnes:

            raise ValueError(
                f"La colonne '{reponse}' n'existe pas "
                f"dans le dataset."
            )

        target = reponse

    # ------------------------------------------------------------
    # DETERMINATION DU TYPE DE PROBLEME POUR LA CIBLE CHOISIE
    # ------------------------------------------------------------

    serie = df[target]

    if pd.api.types.is_numeric_dtype(serie) and serie.nunique() > 15:

        problem_type = "regression"

    else:

        nb_classes = serie.nunique()

        if nb_classes == 2:

            problem_type = "classification_binaire"

        else:

            problem_type = "classification_multiclasse"

    print(
        f"\n🎯 Cible choisie : {target}"
    )

    print(
        f"🧠 Type de problème déterminé : {problem_type}"
    )

    return target, problem_type


# ================================================================
# ETAPE 6 — PREPARATION DES DONNEES
# ================================================================

def preparer_donnees(
    df,
    target
):

    afficher_titre(
        6,
        "🧩 PRÉPARATION DES DONNÉES"
    )

    if target not in df.columns:

        raise ValueError(
            f"La cible '{target}' "
            f"n'existe pas dans le dataset."
        )

    X = df.drop(
        columns=[target]
    ).copy()

    y = df[target].copy()

    print(
        f"📊 X : "
        f"{X.shape[0]} lignes × "
        f"{X.shape[1]} colonnes"
    )

    print(
        f"🎯 y : "
        f"{len(y)} observations"
    )

    return X, y


# ================================================================
# ETAPE 7 — FEATURE ENGINEERING
# ================================================================

def effectuer_feature_engineering(
    X,
    y=None
):

    afficher_titre(
        7,
        "⚙️ FEATURE ENGINEERING AUTOMATIQUE"
    )

    ingenieur = FeatureEngineer()

    # ------------------------------------------------------------
    # fit_transform
    # ------------------------------------------------------------

    if hasattr(
        ingenieur,
        "fit_transform"
    ):

        try:

            resultat = ingenieur.fit_transform(
                X,
                y
            )

        except TypeError:

            resultat = ingenieur.fit_transform(
                X
            )

    elif hasattr(
        ingenieur,
        "fit"
    ) and hasattr(
        ingenieur,
        "transform"
    ):

        ingenieur.fit(
            X,
            y
        )

        resultat = ingenieur.transform(
            X
        )

    else:

        raise AttributeError(
            "FeatureEngineer ne possède pas "
            "de méthode compatible."
        )

    # ------------------------------------------------------------
    # RESULTAT
    # ------------------------------------------------------------

    if isinstance(
        resultat,
        tuple
    ):

        X_features = resultat[0]

    else:

        X_features = resultat

    # ------------------------------------------------------------
    # CONVERSION EN DATAFRAME
    # ------------------------------------------------------------

    if hasattr(
        X_features,
        "shape"
    ):

        nombre_features = X_features.shape[1]

    else:

        nombre_features = len(
            X_features
        )

    print(
        "\n✅ Feature Engineering terminé."
    )

    print(
        f"🔢 Nombre de features : "
        f"{nombre_features}"
    )

    return (
        ingenieur,
        X_features
    )


# ================================================================
# ETAPE 8 — TRAIN / TEST
# ================================================================

def separer_train_test(
    X,
    y,
    problem_type
):

    afficher_titre(
        8,
        "✂️ SÉPARATION ENTRAÎNEMENT / TEST"
    )

    trainer = ModelTrainer()

    # ------------------------------------------------------------
    # Utilisation du trainer si compatible
    # ------------------------------------------------------------

    methode = trouver_methode(
        trainer,
        [
            "split_data",
            "train_test_split",
            "prepare_data",
            "split"
        ]
    )

    resultat = None

    if methode is not None:

        try:

            resultat = methode(
                X,
                y
            )

        except Exception:

            resultat = None

    # ------------------------------------------------------------
    # FALLBACK SKLEARN
    # ------------------------------------------------------------

    if resultat is None:

        from sklearn.model_selection import train_test_split

        stratify = None

        if problem_type.startswith(
            "classification"
        ):

            try:

                stratify = y

            except Exception:

                stratify = None

        resultat = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=stratify
        )

    # ------------------------------------------------------------
    # NORMALISATION DU FORMAT
    # ------------------------------------------------------------

    if len(resultat) != 4:

        raise RuntimeError(
            "La séparation Train/Test "
            "n'a pas retourné 4 éléments."
        )

    X_train, X_test, y_train, y_test = resultat

    print(
        f"\n📚 X_train : {X_train.shape}"
    )

    print(
        f"🧪 X_test : {X_test.shape}"
    )

    print(
        f"🎯 y_train : {y_train.shape}"
    )

    print(
        f"🎯 y_test : {y_test.shape}"
    )

    print(
        "\n" + "-" * 70
    )

    print(
        "DONNÉES PRÊTES POUR L'AUTO ML"
    )

    print(
        "-" * 70
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )


# ================================================================
# ETAPE 9 — PREPARATION DES MODELES
# ================================================================

def preparer_modeles():

    afficher_titre(
        9,
        "🤖 PRÉPARATION DES MODÈLES"
    )

    factory = ModelFactory()

    print(
        "✅ Fabrique de modèles préparée."
    )

    return factory


# ================================================================
# ETAPE 10 — AUTO ML
# ================================================================

def lancer_automl(
    problem_type,
    X_train,
    y_train,
    X_test,
    y_test
):

    afficher_titre(
        10,
        "🧠 AUTO ML ET OPTIMISATION"
    )

    print(
        f"\nType de problème : "
        f"{problem_type}"
    )

    print(
        f"Trials : {N_TRIALS}"
    )

    # IMPORTANT :
    # problem_type doit être fourni au constructeur.

    automl = AutoML(
        problem_type=problem_type,
        n_trials=N_TRIALS
    )

    resultats = automl.run(
        X_train,
        y_train,
        X_test,
        y_test
    )

    champion = automl.get_champion()

    if champion is None:

        raise RuntimeError(
            "AutoML n'a retourné aucun champion."
        )

    print(
        "\n🏆 CHAMPION AUTO ML"
    )

    print(
        f"Modèle : {champion['model']}"
    )

    print(
        f"Score : {champion['score']:.4f}"
    )

    return (
        automl,
        champion,
        resultats
    )


# ================================================================
# ETAPE 10bis — ENCODAGE DE LA TARGET POUR L'ENTRAINEMENT FINAL
# ================================================================
#
# IMPORTANT :
#
# Le champion AutoML est créé (mais pas entraîné) par
# model_factory.create_model(). Pour les modèles de
# classification (XGBoost, LightGBM, CatBoost, etc.), la target
# DOIT être encodée en entiers (0, 1, 2, ...) avant le .fit().
#
# L'encodeur utilisé pendant l'optimisation (LabelEncoder) est
# renvoyé dans le dict "champion" sous la clé "label_encoder".
# On le réutilise ici pour encoder y_train / y_test de façon
# cohérente avec ce qui a été fait pendant la validation croisée.
#
# Sans cela :
#
# ValueError: Invalid classes inferred from unique values of `y`.
# Expected: [0 1], got ['New' 'Returning']
# ================================================================

def encoder_target_pour_entrainement(
    champion,
    y_train,
    y_test,
    problem_type
):

    if champion is None:

        return y_train, y_test, None

    if not problem_type.startswith("classification"):

        return y_train, y_test, None

    label_encoder = champion.get("label_encoder")

    if label_encoder is None:

        print(
            "⚠️ Aucun label_encoder trouvé dans le champion. "
            "La target n'a pas pu être ré-encodée."
        )

        return y_train, y_test, None

    try:

        y_train_encoded = label_encoder.transform(
            y_train.astype(str)
        )

        y_test_encoded = label_encoder.transform(
            y_test.astype(str)
        )

        print(
            "\n✅ Target ré-encodée pour l'entraînement final "
            "(cohérente avec l'AutoML)."
        )

        return y_train_encoded, y_test_encoded, label_encoder

    except Exception as e:

        print(
            f"⚠️ Impossible de ré-encoder la target : {e}"
        )

        return y_train, y_test, None


# ================================================================
# ETAPE 11 — EVALUATION
# ================================================================

def evaluer_modele(
    champion,
    X_test,
    y_test,
    problem_type
):

    afficher_titre(
        11,
        "🧪 ÉVALUATION DU MODÈLE CHAMPION"
    )

    if champion is None:

        print(
            "⚠️ Aucun champion disponible."
        )

        return None

    model = champion.get(
        "model_object"
    )

    if model is None:

        print(
            "⚠️ Aucun modèle disponible."
        )

        return None

    # ------------------------------------------------------------
    # ATTENTION :
    #
    # Le modèle a déjà été entraîné sur X_train / y_train
    # (encodés) juste avant l'appel à cette fonction, dans main().
    # On ne le ré-entraîne PAS ici sur X_test : ce serait à la
    # fois inutile et une fuite de données (le modèle serait
    # évalué sur les mêmes données que celles utilisées pour
    # l'entraîner).
    # ------------------------------------------------------------

    evaluation = {}

    try:

        if hasattr(
            model,
            "predict"
        ):

            predictions = model.predict(
                X_test
            )

            evaluation["predictions"] = predictions

    except Exception as e:

        print(
            f"⚠️ Impossible de prédire : {e}"
        )

    # ------------------------------------------------------------
    # METRIQUES
    # ------------------------------------------------------------

    try:

        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
            mean_absolute_error,
            mean_squared_error,
            r2_score
        )

        predictions = evaluation.get(
            "predictions"
        )

        if predictions is not None:

            if problem_type.startswith(
                "classification"
            ):

                evaluation["accuracy"] = (
                    accuracy_score(
                        y_test,
                        predictions
                    )
                )

                evaluation["precision"] = (
                    precision_score(
                        y_test,
                        predictions,
                        average="weighted",
                        zero_division=0
                    )
                )

                evaluation["recall"] = (
                    recall_score(
                        y_test,
                        predictions,
                        average="weighted",
                        zero_division=0
                    )
                )

                evaluation["f1"] = (
                    f1_score(
                        y_test,
                        predictions,
                        average="weighted",
                        zero_division=0
                    )
                )

                print(
                    f"\nAccuracy : "
                    f"{evaluation['accuracy']:.4f}"
                )

                print(
                    f"Precision : "
                    f"{evaluation['precision']:.4f}"
                )

                print(
                    f"Recall : "
                    f"{evaluation['recall']:.4f}"
                )

                print(
                    f"F1 Score : "
                    f"{evaluation['f1']:.4f}"
                )

            else:

                evaluation["mae"] = (
                    mean_absolute_error(
                        y_test,
                        predictions
                    )
                )

                evaluation["rmse"] = (
                    np.sqrt(
                        mean_squared_error(
                            y_test,
                            predictions
                        )
                    )
                )

                evaluation["r2"] = (
                    r2_score(
                        y_test,
                        predictions
                    )
                )

                print(
                    f"\nMAE : "
                    f"{evaluation['mae']:.4f}"
                )

                print(
                    f"RMSE : "
                    f"{evaluation['rmse']:.4f}"
                )

                print(
                    f"R² : "
                    f"{evaluation['r2']:.4f}"
                )

    except Exception as e:

        print(
            f"⚠️ Erreur pendant l'évaluation : {e}"
        )

    print(
        "\n✅ Évaluation terminée."
    )

    return evaluation


# ================================================================
# ETAPE 12 — EXPLICABILITE
# ================================================================

def expliquer_modele(
    champion,
    X_train,
    problem_type
):

    afficher_titre(
        12,
        "🔎 EXPLICABILITÉ DU MODÈLE"
    )

    if champion is None:

        print(
            "⚠️ Aucun modèle disponible."
        )

        return None

    model = champion.get(
        "model_object"
    )

    if model is None:

        print(
            "⚠️ Aucun modèle disponible."
        )

        return None

    if ModelExplainer is None:

        print(
            "⚠️ Module d'explicabilité indisponible."
        )

        return None

    try:

        explainer = ModelExplainer()

        methode = trouver_methode(
            explainer,
            [
                "explain",
                "analyze",
                "run",
                "expliquer"
            ]
        )

        if methode is None:

            print(
                "⚠️ Aucune méthode d'explicabilité."
            )

            return None

        try:

            resultat = methode(
                model,
                X_train
            )

        except TypeError:

            resultat = methode(
                model
            )

        print(
            "\n✅ Explicabilité terminée."
        )

        return resultat

    except Exception as e:

        print(
            f"⚠️ Explicabilité non disponible : {e}"
        )

        return None


# ================================================================
# ETAPE 13 — ANALYSE DES ERREURS
# ================================================================

def analyser_erreurs(
    champion,
    X_test,
    y_test
):

    afficher_titre(
        13,
        "🕵️ ANALYSE AUTOMATIQUE DES ERREURS"
    )

    if champion is None:

        print(
            "⚠️ Aucun modèle disponible."
        )

        return None

    model = champion.get(
        "model_object"
    )

    if model is None:

        print(
            "⚠️ Aucun modèle disponible."
        )

        return None

    if ErrorAnalyzer is None:

        print(
            "⚠️ Module d'analyse des erreurs indisponible."
        )

        return None

    try:

        analyzer = ErrorAnalyzer()

        methode = trouver_methode(
            analyzer,
            [
                "analyze",
                "analyse",
                "analyze_errors",
                "run"
            ]
        )

        if methode is None:

            print(
                "⚠️ Aucune méthode d'analyse disponible."
            )

            return None

        try:

            resultat = methode(
                model,
                X_test,
                y_test
            )

        except TypeError:

            resultat = methode(
                model
            )

        print(
            "\n✅ Analyse des erreurs terminée."
        )

        return resultat

    except Exception as e:

        print(
            f"⚠️ Analyse des erreurs impossible : {e}"
        )

        return None


# ================================================================
# ETAPE 14 — PREDICTION
# ================================================================

def effectuer_prediction(
    champion,
    X_test,
    label_encoder=None
):

    afficher_titre(
        14,
        "🔮 PRÉDICTION AUTOMATIQUE"
    )

    if champion is None:

        print(
            "⚠️ Aucun modèle disponible."
        )

        return None

    model = champion.get(
        "model_object"
    )

    if model is None:

        print(
            "⚠️ Aucun modèle disponible."
        )

        return None

    try:

        predictions = model.predict(
            X_test
        )

        # --------------------------------------------------------
        # DECODAGE DES PREDICTIONS
        #
        # Pour la classification, les prédictions du modèle sont
        # des entiers (0, 1, ...). On les reconvertit dans les
        # labels d'origine ('New', 'Returning', ...) pour un
        # affichage / rapport lisible.
        # --------------------------------------------------------

        if label_encoder is not None:

            try:

                predictions = label_encoder.inverse_transform(
                    predictions
                )

            except Exception:

                pass

        print(
            f"✅ {len(predictions)} prédictions générées."
        )

        return predictions

    except Exception as e:

        print(
            f"⚠️ Impossible de générer les prédictions : {e}"
        )

        return None


# ================================================================
# ETAPE 15 — RAPPORT
# ================================================================

def generer_rapport(
    df_initial,
    df_clean,
    target,
    problem_type,
    X_features,
    champion,
    evaluation
):

    afficher_titre(
        15,
        "📄 GÉNÉRATION DU RAPPORT AUTOMATIQUE"
    )

    rapport = {
        "dataset_initial": {
            "lignes": int(df_initial.shape[0]),
            "colonnes": int(df_initial.shape[1])
        },

        "dataset_nettoye": {
            "lignes": int(df_clean.shape[0]),
            "colonnes": int(df_clean.shape[1])
        },

        "cible": target,

        "type_probleme": problem_type,

        "nombre_features": (
            int(X_features.shape[1])
            if hasattr(
                X_features,
                "shape"
            )
            else None
        ),

        "champion": None,

        "evaluation": {}
    }

    if champion is not None:

        rapport["champion"] = {
            "modele": champion.get(
                "model"
            ),
            "score_cv": champion.get(
                "score"
            ),
            "parametres": champion.get(
                "params",
                {}
            )
        }

    if evaluation:

        for key, value in evaluation.items():

            if isinstance(
                value,
                (int, float, np.integer, np.floating)
            ):

                rapport["evaluation"][key] = float(
                    value
                )

    # ------------------------------------------------------------
    # JSON
    # ------------------------------------------------------------

    import json

    chemin_json = (
        "reports/final/rapport_final.json"
    )

    try:

        with open(
            chemin_json,
            "w",
            encoding="utf-8"
        ) as fichier:

            json.dump(
                rapport,
                fichier,
                indent=4,
                ensure_ascii=False
            )

        print(
            f"✅ Rapport JSON : {chemin_json}"
        )

    except Exception as e:

        print(
            f"⚠️ Rapport JSON impossible : {e}"
        )

    # ------------------------------------------------------------
    # RAPPORT TXT
    # ------------------------------------------------------------

    chemin_txt = (
        "reports/final/rapport_final.txt"
    )

    try:

        with open(
            chemin_txt,
            "w",
            encoding="utf-8"
        ) as fichier:

            fichier.write(
                "=" * 70 + "\n"
            )

            fichier.write(
                "              RAPPORT FINAL — IA DATA SCIENTIST\n"
            )

            fichier.write(
                "=" * 70 + "\n\n"
            )

            fichier.write(
                f"Dataset initial : "
                f"{df_initial.shape[0]} lignes × "
                f"{df_initial.shape[1]} colonnes\n"
            )

            fichier.write(
                f"Dataset nettoyé : "
                f"{df_clean.shape[0]} lignes × "
                f"{df_clean.shape[1]} colonnes\n\n"
            )

            fichier.write(
                f"Cible : {target}\n"
            )

            fichier.write(
                f"Type de problème : "
                f"{problem_type}\n"
            )

            fichier.write(
                f"Features : "
                f"{X_features.shape[1]}\n\n"
            )

            if champion is not None:

                fichier.write(
                    "MODELE CHAMPION\n"
                )

                fichier.write(
                    "-" * 50 + "\n"
                )

                fichier.write(
                    f"Modèle : "
                    f"{champion.get('model')}\n"
                )

                fichier.write(
                    f"Score CV : "
                    f"{champion.get('score'):.4f}\n\n"
                )

                fichier.write(
                    "Paramètres :\n"
                )

                for key, value in champion.get(
                    "params",
                    {}
                ).items():

                    fichier.write(
                        f"   {key} : {value}\n"
                    )

            if evaluation:

                fichier.write(
                    "\nEVALUATION\n"
                )

                fichier.write(
                    "-" * 50 + "\n"
                )

                for key, value in evaluation.items():

                    if isinstance(
                        value,
                        (int, float, np.integer, np.floating)
                    ):

                        fichier.write(
                            f"{key} : {float(value):.4f}\n"
                        )

        print(
            f"✅ Rapport TXT : {chemin_txt}"
        )

    except Exception as e:

        print(
            f"⚠️ Rapport TXT impossible : {e}"
        )

    return rapport


# ================================================================
# ETAPE 15bis — RAPPORT PDF PROFESSIONNEL
# ================================================================
#
# Génère un rapport PDF avec mise en page soignée :
#
# - Page de garde
# - Résumé exécutif
# - Aperçu des données
# - Modèle retenu + hyperparamètres
# - Évaluation (tableau + graphique de métriques)
# - Aperçu des prédictions
#
# Nécessite : reportlab, matplotlib
#
#   pip install reportlab matplotlib
#
# Si reportlab n'est pas installé, la fonction avertit et
# n'interrompt pas le pipeline (le rapport JSON/TXT reste généré
# par generer_rapport()).
# ================================================================

def generer_graphique_metriques(
    evaluation,
    problem_type,
    chemin_image
):

    if not evaluation:

        return None

    try:

        import matplotlib

        matplotlib.use("Agg")

        import matplotlib.pyplot as plt

    except ImportError:

        print(
            "⚠️ matplotlib n'est pas installé. "
            "Graphique de métriques non généré."
        )

        return None

    if problem_type.startswith("classification"):

        cles = ["accuracy", "precision", "recall", "f1"]

    else:

        cles = ["mae", "rmse", "r2"]

    labels = []

    valeurs = []

    for cle in cles:

        if cle in evaluation:

            try:

                valeur = float(evaluation[cle])

            except (TypeError, ValueError):

                continue

            labels.append(cle.upper())

            valeurs.append(valeur)

    if not valeurs:

        return None

    try:

        fig, ax = plt.subplots(figsize=(6, 3.2))

        couleur_barres = "#2E86AB"

        barres = ax.bar(
            labels,
            valeurs,
            color=couleur_barres
        )

        borne_haute = max(
            max(valeurs) * 1.25,
            1
        )

        ax.set_ylim(0, borne_haute)

        ax.set_title(
            "Métriques de performance du modèle champion",
            fontsize=11,
            fontweight="bold",
            color="#1F3B57"
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        for barre, valeur in zip(barres, valeurs):

            ax.text(
                barre.get_x() + barre.get_width() / 2,
                valeur + borne_haute * 0.02,
                f"{valeur:.3f}",
                ha="center",
                va="bottom",
                fontsize=9
            )

        plt.tight_layout()

        os.makedirs(
            os.path.dirname(chemin_image),
            exist_ok=True
        )

        plt.savefig(
            chemin_image,
            dpi=150
        )

        plt.close(fig)

        return chemin_image

    except Exception as e:

        print(
            f"⚠️ Graphique de métriques impossible : {e}"
        )

        return None


def generer_rapport_pdf(
    df_initial,
    df_clean,
    target,
    problem_type,
    X_features,
    champion,
    evaluation,
    predictions=None,
    chemin_pdf="reports/final/rapport_final.pdf"
):

    try:

        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import cm
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_CENTER
        from reportlab.platypus import (
            SimpleDocTemplate,
            Paragraph,
            Spacer,
            Table,
            TableStyle,
            Image,
            PageBreak,
            HRFlowable
        )

    except ImportError:

        print(
            "⚠️ reportlab n'est pas installé. "
            "Rapport PDF non généré. "
            "Installez-le avec : pip install reportlab"
        )

        return None

    from datetime import datetime

    COULEUR_PRIMAIRE = colors.HexColor("#1F3B57")
    COULEUR_ACCENT = colors.HexColor("#2E86AB")
    COULEUR_GRIS_CLAIR = colors.HexColor("#F2F4F7")
    COULEUR_BORDURE = colors.HexColor("#D9D9D9")

    styles = getSampleStyleSheet()

    style_titre = ParagraphStyle(
        "TitrePrincipal",
        parent=styles["Title"],
        fontSize=24,
        textColor=COULEUR_PRIMAIRE,
        spaceAfter=6,
        alignment=TA_CENTER
    )

    style_soustitre = ParagraphStyle(
        "SousTitre",
        parent=styles["Normal"],
        fontSize=12,
        textColor=colors.grey,
        alignment=TA_CENTER,
        spaceAfter=24
    )

    style_section = ParagraphStyle(
        "Section",
        parent=styles["Heading1"],
        fontSize=15,
        textColor=COULEUR_PRIMAIRE,
        spaceBefore=18,
        spaceAfter=10
    )

    style_corps = ParagraphStyle(
        "Corps",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        spaceAfter=6
    )

    style_puce = ParagraphStyle(
        "Puce",
        parent=style_corps,
        leftIndent=14,
        bulletIndent=4
    )

    elements = []

    # ------------------------------------------------------------
    # PAGE DE GARDE
    # ------------------------------------------------------------

    elements.append(Spacer(1, 4 * cm))

    elements.append(
        Paragraph(
            "IA DATA SCIENTIST",
            style_titre
        )
    )

    elements.append(
        Paragraph(
            "Rapport d'analyse automatique de données",
            style_soustitre
        )
    )

    elements.append(Spacer(1, 1 * cm))

    elements.append(
        HRFlowable(
            width="60%",
            thickness=1,
            color=COULEUR_ACCENT,
            hAlign="CENTER"
        )
    )

    elements.append(Spacer(1, 1 * cm))

    infos_garde = [
        ["Variable cible", str(target)],
        [
            "Type de problème",
            str(problem_type).replace("_", " ").capitalize()
        ],
        [
            "Date de génération",
            datetime.now().strftime("%d/%m/%Y à %H:%M")
        ],
    ]

    table_garde = Table(
        infos_garde,
        colWidths=[6 * cm, 8 * cm]
    )

    table_garde.setStyle(
        TableStyle([
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("TEXTCOLOR", (0, 0), (0, -1), COULEUR_PRIMAIRE),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ])
    )

    elements.append(table_garde)

    elements.append(PageBreak())

    # ------------------------------------------------------------
    # RESUME EXECUTIF
    # ------------------------------------------------------------

    elements.append(
        Paragraph("Résumé exécutif", style_section)
    )

    resume_points = []

    resume_points.append(
        f"Le dataset initial contenait {df_initial.shape[0]} lignes et "
        f"{df_initial.shape[1]} colonnes. Après nettoyage automatique, "
        f"{df_clean.shape[0]} lignes et {df_clean.shape[1]} colonnes "
        f"ont été conservées."
    )

    resume_points.append(
        f"La variable à prédire est « {target} », traitée comme un "
        f"problème de {str(problem_type).replace('_', ' ')}."
    )

    if X_features is not None and hasattr(X_features, "shape"):

        resume_points.append(
            f"{X_features.shape[1]} variables explicatives ont été "
            f"utilisées après feature engineering automatique."
        )

    if champion is not None:

        resume_points.append(
            f"Le modèle retenu (champion) est {champion.get('model')}, "
            f"avec un score de validation croisée de "
            f"{champion.get('score'):.4f}."
        )

    else:

        resume_points.append(
            "Aucun modèle champion n'a pu être déterminé automatiquement."
        )

    for point in resume_points:

        elements.append(
            Paragraph(f"• {point}", style_puce)
        )

    elements.append(Spacer(1, 0.5 * cm))

    # ------------------------------------------------------------
    # APERCU DES DONNEES
    # ------------------------------------------------------------

    elements.append(
        Paragraph("Aperçu des données", style_section)
    )

    data_table = [
        ["", "Lignes", "Colonnes"],
        [
            "Dataset initial",
            str(df_initial.shape[0]),
            str(df_initial.shape[1])
        ],
        [
            "Dataset nettoyé",
            str(df_clean.shape[0]),
            str(df_clean.shape[1])
        ],
    ]

    table_donnees = Table(
        data_table,
        colWidths=[6 * cm, 4 * cm, 4 * cm]
    )

    table_donnees.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), COULEUR_PRIMAIRE),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ALIGN", (1, 0), (-1, -1), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.5, COULEUR_BORDURE),
            (
                "ROWBACKGROUNDS",
                (0, 1),
                (-1, -1),
                [colors.white, COULEUR_GRIS_CLAIR]
            ),
            ("FONTSIZE", (0, 0), (-1, -1), 9.5),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ])
    )

    elements.append(table_donnees)

    elements.append(Spacer(1, 0.6 * cm))

    # ------------------------------------------------------------
    # MODELE CHAMPION
    # ------------------------------------------------------------

    elements.append(
        Paragraph("Modèle retenu", style_section)
    )

    if champion is not None:

        infos_modele = [
            ["Modèle", str(champion.get("model"))],
            [
                "Score (validation croisée)",
                f"{champion.get('score'):.4f}"
            ],
        ]

        table_modele = Table(
            infos_modele,
            colWidths=[6 * cm, 8 * cm]
        )

        table_modele.setStyle(
            TableStyle([
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, COULEUR_BORDURE),
                ("BACKGROUND", (0, 0), (0, -1), COULEUR_GRIS_CLAIR),
                ("FONTSIZE", (0, 0), (-1, -1), 9.5),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ])
        )

        elements.append(table_modele)

        elements.append(Spacer(1, 0.3 * cm))

        params = champion.get("params", {})

        if params:

            elements.append(
                Paragraph(
                    "Hyperparamètres optimisés :",
                    style_corps
                )
            )

            lignes_params = [["Paramètre", "Valeur"]]

            for cle, valeur in params.items():

                lignes_params.append(
                    [str(cle), str(valeur)]
                )

            table_params = Table(
                lignes_params,
                colWidths=[7 * cm, 7 * cm]
            )

            table_params.setStyle(
                TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), COULEUR_ACCENT),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("GRID", (0, 0), (-1, -1), 0.5, COULEUR_BORDURE),
                    (
                        "ROWBACKGROUNDS",
                        (0, 1),
                        (-1, -1),
                        [colors.white, COULEUR_GRIS_CLAIR]
                    ),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ])
            )

            elements.append(table_params)

    else:

        elements.append(
            Paragraph(
                "Aucun modèle champion n'a pu être déterminé.",
                style_corps
            )
        )

    elements.append(Spacer(1, 0.6 * cm))

    # ------------------------------------------------------------
    # EVALUATION
    # ------------------------------------------------------------

    elements.append(
        Paragraph("Évaluation des performances", style_section)
    )

    if evaluation:

        chemin_graphique = os.path.join(
            "reports",
            "final",
            "graphique_metriques.png"
        )

        graphique = generer_graphique_metriques(
            evaluation,
            problem_type,
            chemin_graphique
        )

        if graphique and os.path.exists(graphique):

            elements.append(
                Image(graphique, width=14 * cm, height=7.5 * cm)
            )

            elements.append(Spacer(1, 0.4 * cm))

        lignes_eval = [["Métrique", "Valeur"]]

        for cle, valeur in evaluation.items():

            if cle == "predictions":

                continue

            if isinstance(valeur, (int, float, np.integer, np.floating)):

                lignes_eval.append(
                    [cle.upper(), f"{float(valeur):.4f}"]
                )

        if len(lignes_eval) > 1:

            table_eval = Table(
                lignes_eval,
                colWidths=[7 * cm, 7 * cm]
            )

            table_eval.setStyle(
                TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), COULEUR_PRIMAIRE),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("GRID", (0, 0), (-1, -1), 0.5, COULEUR_BORDURE),
                    (
                        "ROWBACKGROUNDS",
                        (0, 1),
                        (-1, -1),
                        [colors.white, COULEUR_GRIS_CLAIR]
                    ),
                    ("FONTSIZE", (0, 0), (-1, -1), 9.5),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ])
            )

            elements.append(table_eval)

    else:

        elements.append(
            Paragraph("Aucune évaluation disponible.", style_corps)
        )

    elements.append(Spacer(1, 0.6 * cm))

    # ------------------------------------------------------------
    # APERCU DES PREDICTIONS
    # ------------------------------------------------------------

    elements.append(
        Paragraph("Aperçu des prédictions", style_section)
    )

    if predictions is not None and len(predictions) > 0:

        try:

            valeurs_uniques, comptes = np.unique(
                predictions,
                return_counts=True
            )

            lignes_pred = [
                ["Valeur prédite", "Occurrences", "Part (%)"]
            ]

            total = int(comptes.sum())

            for valeur, compte in zip(valeurs_uniques, comptes):

                part = (
                    (compte / total) * 100
                    if total
                    else 0
                )

                lignes_pred.append(
                    [str(valeur), str(int(compte)), f"{part:.1f}%"]
                )

            table_pred = Table(
                lignes_pred,
                colWidths=[6 * cm, 4 * cm, 4 * cm]
            )

            table_pred.setStyle(
                TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), COULEUR_ACCENT),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                    ("GRID", (0, 0), (-1, -1), 0.5, COULEUR_BORDURE),
                    (
                        "ROWBACKGROUNDS",
                        (0, 1),
                        (-1, -1),
                        [colors.white, COULEUR_GRIS_CLAIR]
                    ),
                    ("FONTSIZE", (0, 0), (-1, -1), 9.5),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ])
            )

            elements.append(table_pred)

            elements.append(Spacer(1, 0.3 * cm))

            elements.append(
                Paragraph(
                    f"{total} prédictions générées au total sur le "
                    f"jeu de test.",
                    style_corps
                )
            )

        except Exception:

            elements.append(
                Paragraph(
                    f"{len(predictions)} prédictions générées "
                    f"(détail non disponible).",
                    style_corps
                )
            )

    else:

        elements.append(
            Paragraph("Aucune prédiction disponible.", style_corps)
        )

    # ------------------------------------------------------------
    # EN-TETE / PIED DE PAGE
    # ------------------------------------------------------------

    def _entete_pied(canvas, doc):

        canvas.saveState()

        canvas.setFont("Helvetica", 8)

        canvas.setFillColor(colors.grey)

        canvas.drawString(
            2 * cm,
            1.2 * cm,
            "Généré automatiquement par IA Data Scientist"
        )

        canvas.drawRightString(
            19 * cm,
            1.2 * cm,
            f"Page {doc.page}"
        )

        canvas.restoreState()

    # ------------------------------------------------------------
    # GENERATION DU FICHIER
    # ------------------------------------------------------------

    os.makedirs(
        os.path.dirname(chemin_pdf),
        exist_ok=True
    )

    doc = SimpleDocTemplate(
        chemin_pdf,
        pagesize=A4,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        title="Rapport IA Data Scientist"
    )

    try:

        doc.build(
            elements,
            onFirstPage=_entete_pied,
            onLaterPages=_entete_pied
        )

        print(
            f"✅ Rapport PDF professionnel : {chemin_pdf}"
        )

        return chemin_pdf

    except Exception as e:

        print(
            f"⚠️ Rapport PDF impossible : {e}"
        )

        return None


# ================================================================
# RAPPORT FINAL CONSOLE
# ================================================================

def afficher_rapport_final(
    df_initial,
    df_clean,
    target,
    problem_type,
    X_features,
    champion
):

    print("\n")
    print("=" * 70)

    print(
        "                  RAPPORT FINAL — IA DATA SCIENTIST"
    )

    print("=" * 70)

    print(
        f"📂 Dataset initial : "
        f"{df_initial.shape[0]} lignes × "
        f"{df_initial.shape[1]} colonnes"
    )

    print(
        f"🧹 Dataset nettoyé : "
        f"{df_clean.shape[0]} lignes × "
        f"{df_clean.shape[1]} colonnes"
    )

    print(
        f"\n🎯 Cible : {target}"
    )

    print(
        f"🧠 Type de problème : "
        f"{problem_type}"
    )

    print(
        f"📊 Features : "
        f"{X_features.shape[1]}"
    )

    print(
        "\n🏆 MODELE CHAMPION"
    )

    if champion is not None:

        print(
            f"   Modèle : "
            f"{champion.get('model')}"
        )

        print(
            f"   Score CV : "
            f"{champion.get('score'):.4f}"
        )

    else:

        print(
            "   Aucun champion disponible."
        )

    print(
        "\n📁 Résultats disponibles :"
    )

    print(
        "   reports/"
    )

    print(
        "   ├── eda/"
    )

    print(
        "   ├── errors/"
    )

    print(
        "   ├── models/"
    )

    print(
        "   ├── predictions/"
    )

    print(
        "   ├── explainability/"
    )

    print(
        "   └── final/"
    )

    print("\n")
    print("=" * 70)

    print(
        "                    ✅ ANALYSE TERMINÉE"
    )

    print(
        "\n        L'IA DATA SCIENTIST a terminé son analyse."
    )

    print("=" * 70)


# ================================================================
# MAIN
# ================================================================

def main():

    afficher_intro()

    creer_dossiers()

    # Variables globales du pipeline

    df_initial = None

    df_clean = None

    profil = None

    eda_result = None

    target = None

    problem_type = None

    features_candidates = None

    X = None

    y = None

    feature_engineer = None

    X_features = None

    X_train = None

    X_test = None

    y_train = None

    y_test = None

    factory = None

    automl = None

    champion = None

    resultats_automl = None

    evaluation = None

    explainability = None

    errors = None

    predictions = None

    rapport = None

    # ============================================================
    # ETAPE 1
    # ============================================================

    try:

        df_initial = charger_donnees()

    except Exception as e:

        afficher_erreur(
            "Impossible de charger les données.",
            e
        )

        return

    # ============================================================
    # ETAPE 2
    # ============================================================

    try:

        profil = profiler_donnees(
            df_initial
        )

    except Exception as e:

        afficher_erreur(
            "Le profilage a rencontré un problème.",
            e
        )

    # ============================================================
    # ETAPE 3
    # ============================================================

    try:

        df_clean = nettoyer_donnees(
            df_initial
        )

    except Exception as e:

        afficher_erreur(
            "Le nettoyage a rencontré un problème.",
            e
        )

        return

    # ------------------------------------------------------------
    # ETAPE 3bis — export du dataset nettoyé dans un fichier séparé
    # ------------------------------------------------------------

    try:

        exporter_dataset_nettoye(
            df_clean
        )

    except Exception as e:

        afficher_erreur(
            "L'export du dataset nettoyé a rencontré un problème.",
            e
        )

    # ============================================================
    # ETAPE 4
    # ============================================================

    try:

        eda_result = effectuer_eda(
            df_clean
        )

    except Exception as e:

        afficher_erreur(
            "L'EDA a rencontré un problème.",
            e
        )

    # ============================================================
    # ETAPE 5
    # ============================================================

    try:

        (
            target,
            problem_type,
            features_candidates
        ) = detecter_probleme(
            df_clean
        )

    except Exception as e:

        afficher_erreur(
            "La détection automatique du problème a échoué.",
            e
        )

        return

    # ------------------------------------------------------------
    # ETAPE 5bis — l'utilisateur peut choisir la variable cible
    # ------------------------------------------------------------

    try:

        target, problem_type = choisir_cible_manuellement(
            df_clean,
            target_auto=target,
            problem_type_auto=problem_type
        )

    except Exception as e:

        afficher_erreur(
            "Le choix manuel de la cible a rencontré un problème. "
            "La suggestion automatique est conservée.",
            e
        )

    # ============================================================
    # ETAPE 6
    # ============================================================

    try:

        X, y = preparer_donnees(
            df_clean,
            target
        )

    except Exception as e:

        afficher_erreur(
            "La préparation des données a échoué.",
            e
        )

        return

    # ============================================================
    # ETAPE 7
    # ============================================================

    try:

        (
            feature_engineer,
            X_features
        ) = effectuer_feature_engineering(
            X,
            y
        )

    except Exception as e:

        afficher_erreur(
            "Erreur pendant le Feature Engineering.",
            e
        )

        return

    # ============================================================
    # ETAPE 8
    # ============================================================

    try:

        (
            X_train,
            X_test,
            y_train,
            y_test
        ) = separer_train_test(
            X_features,
            y,
            problem_type
        )

    except Exception as e:

        afficher_erreur(
            "La séparation Train/Test a échoué.",
            e
        )

        return

    # ============================================================
    # ETAPE 9
    # ============================================================

    try:

        factory = preparer_modeles()

    except Exception as e:

        afficher_erreur(
            "La fabrique de modèles a rencontré un problème.",
            e
        )

        return

    # ============================================================
    # ETAPE 10
    # ============================================================

    try:

        (
            automl,
            champion,
            resultats_automl
        ) = lancer_automl(
            problem_type,
            X_train,
            y_train,
            X_test,
            y_test
        )

    except Exception as e:

        afficher_erreur(
            "L'AutoML a rencontré un problème.",
            e
        )

    # ============================================================
    # ETAPE 11
    # ============================================================

    label_encoder_final = None

    try:

        if champion is not None:

            # --------------------------------------------------
            # RE-ENCODAGE DE LA TARGET
            #
            # XGBoost / LightGBM / CatBoost exigent une target
            # entière (0, 1, ...) et non les labels d'origine
            # ('New', 'Returning', ...).
            # --------------------------------------------------

            (
                y_train_ml,
                y_test_ml,
                label_encoder_final
            ) = encoder_target_pour_entrainement(
                champion,
                y_train,
                y_test,
                problem_type
            )

            # --------------------------------------------------
            # ENTRAINEMENT FINAL SUR X_train / y_train (encodés)
            # --------------------------------------------------

            modele = champion.get(
                "model_object"
            )

            if modele is not None:

                modele.fit(
                    X_train,
                    y_train_ml
                )

            evaluation = evaluer_modele(
                champion,
                X_test,
                y_test_ml,
                problem_type
            )

        else:

            print(
                "\n⚠️ Aucun champion récupéré automatiquement."
            )

    except Exception as e:

        afficher_erreur(
            "L'évaluation du modèle a rencontré un problème.",
            e
        )

    # ============================================================
    # ETAPE 12
    # ============================================================

    try:

        explainability = expliquer_modele(
            champion,
            X_train,
            problem_type
        )

    except Exception as e:

        afficher_erreur(
            "L'explicabilité a rencontré un problème.",
            e
        )

    # ============================================================
    # ETAPE 13
    # ============================================================

    try:

        errors = analyser_erreurs(
            champion,
            X_test,
            y_test_ml if label_encoder_final is not None else y_test
        )

    except Exception as e:

        afficher_erreur(
            "L'analyse des erreurs a rencontré un problème.",
            e
        )

    # ============================================================
    # ETAPE 14
    # ============================================================

    try:

        predictions = effectuer_prediction(
            champion,
            X_test,
            label_encoder=label_encoder_final
        )

    except Exception as e:

        afficher_erreur(
            "La prédiction a rencontré un problème.",
            e
        )

    # ============================================================
    # ETAPE 15
    # ============================================================

    try:

        rapport = generer_rapport(
            df_initial,
            df_clean,
            target,
            problem_type,
            X_features,
            champion,
            evaluation
        )

    except Exception as e:

        afficher_erreur(
            "Le rapport automatique n'a pas pu être généré.",
            e
        )

    # ------------------------------------------------------------
    # ETAPE 15bis — rapport PDF professionnel
    # ------------------------------------------------------------

    try:

        generer_rapport_pdf(
            df_initial,
            df_clean,
            target,
            problem_type,
            X_features,
            champion,
            evaluation,
            predictions=predictions
        )

    except Exception as e:

        afficher_erreur(
            "Le rapport PDF n'a pas pu être généré.",
            e
        )

    # ============================================================
    # RAPPORT FINAL
    # ============================================================

    afficher_rapport_final(
        df_initial,
        df_clean,
        target,
        problem_type,
        X_features,
        champion
    )


# ================================================================
# LANCEMENT
# ================================================================

if __name__ == "__main__":

    main()