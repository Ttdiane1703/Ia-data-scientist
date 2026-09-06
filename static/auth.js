(() => {
// ================================================================
// IA DATA SCIENTIST — Authentification Google + Email/Mot de passe
// ================================================================

const SUPABASE_URL = "https://tbljaeeeknfhpscqhlwg.supabase.co";
const SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InRibGphZWVla25maHBzY3FobHdnIiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODgyOTk0MzUsImV4cCI6MjEwMzg3NTQzNX0._Ii6_1NC7d2pjuMp5BUz3Y4nQ0Mp38vebr0-qzJQuZk";

const MODE_INVITE = !SUPABASE_URL || !SUPABASE_ANON_KEY;

// ================================================================
// TRADUCTIONS
// ================================================================

const TRADUCTIONS = {
  fr: {
    "nav.dataset": "Dataset", "nav.target": "Target", "nav.overview": "Overview",
    "nav.quality": "Data Quality", "nav.statistics": "Statistics", "nav.eda": "EDA",
    "nav.features": "Features", "nav.models": "Models", "nav.bestmodel": "Best Model",
    "nav.report": "Report", "nav.notebook": "Notebook", "nav.export": "Export",
    "nav.settings": "Paramètres",
    "dataset.title": "Charger votre dataset",
    "dataset.subtitle": "Importez votre fichier CSV, Excel ou autre format tabulaire",
    "dataset.dropzone": "Glissez-déposez votre fichier ici", "dataset.or": "ou",
    "dataset.browse": "Choisir un fichier",
    "target.title": "Sélectionner la variable cible",
    "target.subtitle": "Choisissez la variable que vous souhaitez prédire",
    "target.field": "Variable cible", "target.launch": "Lancer l'analyse",
    "overview.running.title": "Analyse en cours…",
    "overview.running.subtitle": "AI Data Scientist travaille pour vous",
    "settings.title": "Paramètres", "settings.appearance": "Apparence",
    "settings.theme": "Thème", "settings.theme.dark": "Sombre", "settings.theme.light": "Clair",
    "settings.language": "Langue", "settings.account": "Compte",
    "settings.logout": "Se déconnecter", "settings.save": "Enregistrer",
    "settings.saved": "Préférences enregistrées",
    "settings.restart": "Démarrer une nouvelle analyse",
    "settings.analysis": "Paramètres d'analyse",
    "settings.trials": "Nombre d'essais AutoML (trials)",
    "settings.trials.hint": "Plus de trials = meilleur modèle, mais plus lent (10–100)",
    "settings.testsize": "Proportion de données de test (%)",
    "settings.testsize.hint": "Pourcentage du dataset réservé à l'évaluation (10–40%)",
    "settings.notifications": "Notifications",
    "settings.notif.email": "Recevoir un email à la fin de chaque analyse",
    "settings.advanced": "Avancé", "settings.delete.account": "Supprimer mon compte",
    "settings.delete.confirm": "Tapez SUPPRIMER pour confirmer",
    "auth.welcome": "Bienvenue sur IA Data Scientist",
    "auth.subtitle": "Connectez-vous pour lancer vos analyses",
    "auth.google": "Continuer avec Google",
    "auth.email": "Adresse e-mail", "auth.password": "Mot de passe",
    "auth.login": "Se connecter", "auth.signup": "Créer un compte",
    "auth.no_account": "Pas encore de compte ?",
    "auth.has_account": "Déjà un compte ?",
    "auth.forgot": "Mot de passe oublié ?",
    "auth.reset_sent": "Email de réinitialisation envoyé ! Vérifiez votre boîte mail.",
    "auth.or": "ou avec un e-mail",
    "auth.error.generic": "Une erreur est survenue. Réessayez.",
    "auth.error.network": "Impossible de joindre le serveur d'authentification.",
    "auth.error.invalid": "Email ou mot de passe incorrect.",
    "auth.error.weak_password": "Le mot de passe doit contenir au moins 6 caractères.",
    "auth.error.email_taken": "Cet email est déjà utilisé.",
    "auth.dev.mode": "Mode invité (Supabase non configuré)",
    "auth.reset.title": "Choisir un nouveau mot de passe",
    "auth.reset.subtitle": "Entrez votre nouveau mot de passe ci-dessous",
    "auth.reset.confirm": "Confirmer le mot de passe",
    "auth.reset.submit": "Enregistrer le mot de passe",
    "auth.reset.mismatch": "Les mots de passe ne correspondent pas.",
    "auth.reset.success": "Mot de passe mis à jour ! Vous pouvez continuer.",
    "auth.signup.confirm_needed": "Compte créé ! Vérifiez votre boîte mail et cliquez sur le lien de confirmation avant de vous connecter.",
    "history.title": "Historique des analyses", "history.empty": "Aucune analyse enregistrée.",
    "history.delete": "Supprimer", "history.reload": "Recharger",
    "history.date": "Date", "history.file": "Fichier", "history.target": "Cible",
    "history.model": "Modèle", "history.score": "Score CV", "history.type": "Type",

    "dataset.info.title": "Informations du dataset",
    "dataset.info.filename": "Nom du fichier",
    "dataset.info.rows": "Lignes",
    "dataset.info.columns": "Colonnes",
    "dataset.info.size": "Taille",
    "dataset.info.format": "Format",
    "dataset.info.encoding": "Encodage",
    "dataset.info.success": "Dataset chargé avec succès",
    "dataset.error.notcsv": "Ce fichier n'est pas un .csv. Choisissez un fichier au format CSV.",
    "dataset.error.backend_not_configured": "⚠️ Le backend n'est pas configuré. Dans static/app.js, remplace API_BASE par ton URL Render (ex: https://ia-data-scientist-xxxx.onrender.com) puis redéploie sur Vercel.",
    "dataset.error.upload_failed": "Le dépôt du fichier a échoué.",
    "dataset.error.fetch_info": "Impossible de récupérer les informations du dataset.",
    "dataset.error.network": "Impossible de joindre le backend Render. Vérifie que le service est bien démarré (Render Dashboard > ton service) et que l'URL dans API_BASE est correcte.",
    "dataset.uploading_wait": "Connexion au serveur… cela peut prendre jusqu'à 50 secondes si le service vient de se réveiller.",

    "target.type_detected": "Type détecté",
    "target.preview_title": "Aperçu de la variable cible",
    "target.table.value": "Valeur",
    "target.table.count": "Nombre",
    "target.table.percentage": "Pourcentage",
    "target.type.classification_binaire": "Classification binaire",
    "target.type.classification_multiclasse": "Classification multiclasse",
    "target.type.regression": "Régression",
    "target.stats.min": "Min", "target.stats.max": "Max",
    "target.stats.mean": "Moyenne", "target.stats.median": "Médiane",
    "target.error.distribution": "Impossible de charger l'aperçu de la variable cible.",
    "target.error.launch": "Impossible de lancer l'analyse.",

    "overview.elapsed": "Temps écoulé : ",
    "overview.error.lost_connection": "Connexion au serveur perdue pendant l'analyse.",
    "overview.error.generic": "L'analyse a rencontré un problème.",
    "overview.done.title": "Analyse terminée",
    "overview.done.subtitle": "Analyse terminée avec succès",
    "overview.summary.title": "Résumé exécutif",
    "overview.summary.template": "AI Data Scientist a analysé votre dataset et identifié le meilleur modèle pour prédire {target}. Les données ont été nettoyées, transformées en {n} variables, et {model} a été retenu comme champion avec un score de validation croisée de {score}.",
    "overview.best.badge": "★ MEILLEUR MODÈLE",
    "overview.best.viewdetails": "Voir les détails du modèle",
    "overview.stats.dataset": "Dataset",
    "overview.stats.rows_suffix": "lignes",
    "overview.stats.features_final": "Features finales",
    "overview.stats.best_model": "Meilleur modèle",
    "overview.stats.problem_type": "Type de problème",
    "overview.stats.target": "Cible",
    "overview.stats.score_cv": "Score CV",

    "quality.title": "Data Quality Report",
    "quality.details_title": "Détails des problèmes",
    "quality.table.problem": "Problème", "quality.table.column": "Colonne",
    "quality.table.count": "Nombre", "quality.table.action": "Action",
    "quality.ready": "Les données sont prêtes pour l'analyse",
    "quality.missing": "Missing Values", "quality.duplicates": "Duplicates",
    "quality.invalid": "Invalid Values", "quality.outliers": "Outliers",
    "quality.leakage": "Data Leakage",
    "quality.leakage.detected": "Détectée", "quality.leakage.none": "Aucune détectée",
    "quality.no_issues": "Aucun problème détecté.",
    "quality.error": "Impossible de charger le rapport de qualité des données.",

    "statistics.title": "Statistiques descriptives",
    "statistics.numeric_title": "Statistiques numériques",
    "statistics.categorical_title": "Statistiques catégorielles",
    "statistics.table.variable": "Variable", "statistics.table.count": "Count",
    "statistics.table.mean": "Mean", "statistics.table.std": "Std",
    "statistics.table.min": "Min", "statistics.table.max": "Max",
    "statistics.table.skew": "Skew", "statistics.table.kurt": "Kurt",
    "statistics.table.unique": "Unique", "statistics.table.most_frequent": "Most Frequent",
    "statistics.table.frequency": "Frequency", "statistics.table.percentage": "Percentage",
    "statistics.no_numeric": "Aucune variable numérique.",
    "statistics.no_categorical": "Aucune variable catégorielle.",
    "statistics.error": "Impossible de charger les statistiques.",

    "eda.title": "EDA — Visualisations",
    "eda.distributions_title": "Distributions",
    "eda.correlations_title": "Corrélations",
    "eda.distribution_of": "Distribution de",
    "eda.no_numeric": "Aucune variable numérique à visualiser.",
    "eda.not_enough_for_corr": "Pas assez de variables numériques pour calculer des corrélations.",
    "eda.error": "Impossible de charger les visualisations EDA.",

    "features.title": "Feature Engineering",
    "features.note": "Le détail colonne par colonne des features créées est disponible dans le notebook Jupyter généré (section \"Statistiques descriptives\") et dans le rapport PDF.",
    "features.original_columns": "Colonnes originales",
    "features.final_features": "Features finales",
    "features.rows_used": "Lignes utilisées",

    "models.title": "Comparaison des modèles",
    "models.table.model": "Modèle", "models.table.score_cv": "Score CV",
    "models.no_data": "Aucune donnée.",
    "models.error": "Impossible de charger la comparaison des modèles.",

    "bestmodel.title": "Meilleur modèle",
    "bestmodel.performance": "Performances",
    "bestmodel.information": "Informations",
    "bestmodel.hyperparams": "Hyperparamètres",
    "bestmodel.metric.f1": "F1 Score", "bestmodel.metric.accuracy": "Accuracy",
    "bestmodel.metric.precision": "Precision", "bestmodel.metric.recall": "Recall",
    "bestmodel.metric.mae": "MAE", "bestmodel.metric.rmse": "RMSE", "bestmodel.metric.r2": "R2",
    "bestmodel.info.problem_type": "Type de problème",
    "bestmodel.info.score_cv": "Score CV",
    "bestmodel.info.features_used": "Features utilisées",
    "bestmodel.confirm_template": "{model} est le meilleur modèle parmi ceux testés selon la procédure d'évaluation utilisée.",

    "report.title": "Rapport PDF",
    "report.subtitle": "Rapport complet, explicatif, pour profils techniques et non-techniques",
    "report.download": "Télécharger le rapport PDF",

    "notebook.title": "Notebook Jupyter",
    "notebook.subtitle": "Ouvrable dans Jupyter, VSCode ou JupyterLab pour continuer l'analyse",
    "notebook.download": "Télécharger le notebook (.ipynb)",

    "export.title": "Exporter les résultats",
    "export.subtitle": "Tous les résultats de l'analyse regroupés",
    "export.available_title": "Export disponible",
    "export.available_text": "Téléchargez chaque livrable individuellement ci-dessous.",
    "export.btn.pdf": "Rapport PDF", "export.btn.notebook": "Notebook",
    "export.btn.dataset": "Dataset nettoyé", "export.btn.model": "Modèle (.joblib)",
    "export.checklist.dataset": "Dataset nettoyé",
    "export.checklist.pdf": "Rapport PDF",
    "export.checklist.notebook": "Notebook Jupyter",
    "export.checklist.model": "Modèle entraîné (.joblib)",
    "export.checklist.metadata": "Métadonnées",
    "export.checklist.stats": "Statistiques",

    "steps.0": "Chargement des données", "steps.1": "Profilage automatique",
    "steps.2": "Nettoyage intelligent", "steps.3": "Analyse exploratoire",
    "steps.4": "Détection de la cible", "steps.5": "Préparation des données",
    "steps.6": "Feature engineering", "steps.7": "Train / test",
    "steps.8": "Choix des modèles", "steps.9": "AutoML & optimisation",
    "steps.10": "Évaluation", "steps.11": "Explicabilité",
    "steps.12": "Analyse des erreurs", "steps.13": "Prédiction",
    "steps.14": "Génération du rapport",

    "nav.admin": "Admin",
    "admin.title": "Administration",
    "admin.users_title": "Utilisateurs inscrits",
    "admin.analyses_title": "Toutes les analyses",
    "admin.table.email": "Email",
    "admin.table.name": "Nom",
    "admin.table.admin": "Admin",
    "admin.table.created": "Inscrit le",
    "admin.table.user": "Utilisateur",
    "admin.badge.yes": "Oui",
    "admin.badge.no": "Non",
    "admin.no_users": "Aucun utilisateur.",
    "admin.no_analyses": "Aucune analyse enregistrée.",
    "admin.refresh": "Recharger",
    "admin.error": "Impossible de charger les données d'administration. Vérifie tes politiques RLS Supabase.",
  },
  en: {
    "nav.dataset": "Dataset", "nav.target": "Target", "nav.overview": "Overview",
    "nav.quality": "Data Quality", "nav.statistics": "Statistics", "nav.eda": "EDA",
    "nav.features": "Features", "nav.models": "Models", "nav.bestmodel": "Best Model",
    "nav.report": "Report", "nav.notebook": "Notebook", "nav.export": "Export",
    "nav.settings": "Settings",
    "dataset.title": "Upload your dataset",
    "dataset.subtitle": "Import your CSV, Excel, or other tabular file",
    "dataset.dropzone": "Drag and drop your file here", "dataset.or": "or",
    "dataset.browse": "Choose a file",
    "target.title": "Select the target variable",
    "target.subtitle": "Choose the variable you want to predict",
    "target.field": "Target variable", "target.launch": "Run analysis",
    "overview.running.title": "Analysis in progress…",
    "overview.running.subtitle": "AI Data Scientist is working for you",
    "settings.title": "Settings", "settings.appearance": "Appearance",
    "settings.theme": "Theme", "settings.theme.dark": "Dark", "settings.theme.light": "Light",
    "settings.language": "Language", "settings.account": "Account",
    "settings.logout": "Sign out", "settings.save": "Save",
    "settings.saved": "Preferences saved",
    "settings.restart": "Start a new analysis",
    "settings.analysis": "Analysis settings", "settings.trials": "AutoML trials",
    "settings.trials.hint": "More trials = better model, but slower (10–100)",
    "settings.testsize": "Test data percentage (%)",
    "settings.testsize.hint": "Percentage of dataset reserved for evaluation (10–40%)",
    "settings.notifications": "Notifications",
    "settings.notif.email": "Receive an email when analysis is complete",
    "settings.advanced": "Advanced", "settings.delete.account": "Delete my account",
    "settings.delete.confirm": "Type DELETE to confirm",
    "auth.welcome": "Welcome to AI Data Scientist",
    "auth.subtitle": "Sign in to run your analyses",
    "auth.google": "Continue with Google",
    "auth.email": "Email address", "auth.password": "Password",
    "auth.login": "Sign in", "auth.signup": "Create account",
    "auth.no_account": "Don't have an account?",
    "auth.has_account": "Already have an account?",
    "auth.forgot": "Forgot password?",
    "auth.reset_sent": "Reset email sent! Check your inbox.",
    "auth.or": "or with email",
    "auth.error.generic": "Something went wrong. Please try again.",
    "auth.error.network": "Cannot reach the authentication server.",
    "auth.error.invalid": "Invalid email or password.",
    "auth.error.weak_password": "Password must be at least 6 characters.",
    "auth.error.email_taken": "This email is already in use.",
    "auth.dev.mode": "Guest mode (Supabase not configured)",
    "auth.reset.title": "Choose a new password",
    "auth.reset.subtitle": "Enter your new password below",
    "auth.reset.confirm": "Confirm password",
    "auth.reset.submit": "Save password",
    "auth.reset.mismatch": "Passwords do not match.",
    "auth.reset.success": "Password updated! You can continue.",
    "auth.signup.confirm_needed": "Account created! Check your inbox and click the confirmation link before signing in.",
    "history.title": "Analysis History", "history.empty": "No analyses saved yet.",
    "history.delete": "Delete", "history.reload": "Refresh",
    "history.date": "Date", "history.file": "File", "history.target": "Target",
    "history.model": "Model", "history.score": "CV Score", "history.type": "Type",

    "dataset.info.title": "Dataset information",
    "dataset.info.filename": "File name",
    "dataset.info.rows": "Rows",
    "dataset.info.columns": "Columns",
    "dataset.info.size": "Size",
    "dataset.info.format": "Format",
    "dataset.info.encoding": "Encoding",
    "dataset.info.success": "Dataset loaded successfully",
    "dataset.error.notcsv": "This file is not a .csv. Please choose a CSV file.",
    "dataset.error.backend_not_configured": "⚠️ The backend is not configured. In static/app.js, replace API_BASE with your Render URL (e.g. https://ia-data-scientist-xxxx.onrender.com) then redeploy on Vercel.",
    "dataset.error.upload_failed": "The file upload failed.",
    "dataset.error.fetch_info": "Unable to retrieve dataset information.",
    "dataset.error.network": "Unable to reach the Render backend. Check that the service is running (Render Dashboard > your service) and that the API_BASE URL is correct.",
    "dataset.uploading_wait": "Connecting to server… this can take up to 50 seconds if the service just woke up.",

    "target.type_detected": "Detected type",
    "target.preview_title": "Target variable preview",
    "target.table.value": "Value",
    "target.table.count": "Count",
    "target.table.percentage": "Percentage",
    "target.type.classification_binaire": "Binary classification",
    "target.type.classification_multiclasse": "Multiclass classification",
    "target.type.regression": "Regression",
    "target.stats.min": "Min", "target.stats.max": "Max",
    "target.stats.mean": "Mean", "target.stats.median": "Median",
    "target.error.distribution": "Unable to load the target variable preview.",
    "target.error.launch": "Unable to start the analysis.",

    "overview.elapsed": "Elapsed time: ",
    "overview.error.lost_connection": "Connection to the server was lost during the analysis.",
    "overview.error.generic": "The analysis encountered a problem.",
    "overview.done.title": "Analysis Completed",
    "overview.done.subtitle": "Analysis completed successfully",
    "overview.summary.title": "Executive summary",
    "overview.summary.template": "AI Data Scientist analyzed your dataset and identified the best model to predict {target}. The data was cleaned and transformed into {n} variables, and {model} was selected as the champion with a cross-validation score of {score}.",
    "overview.best.badge": "★ BEST MODEL",
    "overview.best.viewdetails": "View model details",
    "overview.stats.dataset": "Dataset",
    "overview.stats.rows_suffix": "rows",
    "overview.stats.features_final": "Final features",
    "overview.stats.best_model": "Best model",
    "overview.stats.problem_type": "Problem type",
    "overview.stats.target": "Target",
    "overview.stats.score_cv": "CV Score",

    "quality.title": "Data Quality Report",
    "quality.details_title": "Issue details",
    "quality.table.problem": "Issue", "quality.table.column": "Column",
    "quality.table.count": "Count", "quality.table.action": "Action",
    "quality.ready": "The data is ready for analysis",
    "quality.missing": "Missing Values", "quality.duplicates": "Duplicates",
    "quality.invalid": "Invalid Values", "quality.outliers": "Outliers",
    "quality.leakage": "Data Leakage",
    "quality.leakage.detected": "Detected", "quality.leakage.none": "None detected",
    "quality.no_issues": "No issues detected.",
    "quality.error": "Unable to load the data quality report.",

    "statistics.title": "Descriptive statistics",
    "statistics.numeric_title": "Numeric statistics",
    "statistics.categorical_title": "Categorical statistics",
    "statistics.table.variable": "Variable", "statistics.table.count": "Count",
    "statistics.table.mean": "Mean", "statistics.table.std": "Std",
    "statistics.table.min": "Min", "statistics.table.max": "Max",
    "statistics.table.skew": "Skew", "statistics.table.kurt": "Kurt",
    "statistics.table.unique": "Unique", "statistics.table.most_frequent": "Most Frequent",
    "statistics.table.frequency": "Frequency", "statistics.table.percentage": "Percentage",
    "statistics.no_numeric": "No numeric variable.",
    "statistics.no_categorical": "No categorical variable.",
    "statistics.error": "Unable to load statistics.",

    "eda.title": "EDA — Visualizations",
    "eda.distributions_title": "Distributions",
    "eda.correlations_title": "Correlations",
    "eda.distribution_of": "Distribution of",
    "eda.no_numeric": "No numeric variable to visualize.",
    "eda.not_enough_for_corr": "Not enough numeric variables to compute correlations.",
    "eda.error": "Unable to load EDA visualizations.",

    "features.title": "Feature Engineering",
    "features.note": "The column-by-column detail of the created features is available in the generated Jupyter notebook (\"Descriptive statistics\" section) and in the PDF report.",
    "features.original_columns": "Original columns",
    "features.final_features": "Final features",
    "features.rows_used": "Rows used",

    "models.title": "Model comparison",
    "models.table.model": "Model", "models.table.score_cv": "CV Score",
    "models.no_data": "No data.",
    "models.error": "Unable to load model comparison.",

    "bestmodel.title": "Best model",
    "bestmodel.performance": "Performance",
    "bestmodel.information": "Information",
    "bestmodel.hyperparams": "Hyperparameters",
    "bestmodel.metric.f1": "F1 Score", "bestmodel.metric.accuracy": "Accuracy",
    "bestmodel.metric.precision": "Precision", "bestmodel.metric.recall": "Recall",
    "bestmodel.metric.mae": "MAE", "bestmodel.metric.rmse": "RMSE", "bestmodel.metric.r2": "R2",
    "bestmodel.info.problem_type": "Problem type",
    "bestmodel.info.score_cv": "CV Score",
    "bestmodel.info.features_used": "Features used",
    "bestmodel.confirm_template": "{model} is the best model among those tested according to the evaluation procedure used.",

    "report.title": "PDF Report",
    "report.subtitle": "Full, explanatory report, for technical and non-technical profiles",
    "report.download": "Download PDF report",

    "notebook.title": "Jupyter Notebook",
    "notebook.subtitle": "Can be opened in Jupyter, VSCode, or JupyterLab to continue the analysis",
    "notebook.download": "Download notebook (.ipynb)",

    "export.title": "Export results",
    "export.subtitle": "All analysis results grouped together",
    "export.available_title": "Export available",
    "export.available_text": "Download each deliverable individually below.",
    "export.btn.pdf": "PDF Report", "export.btn.notebook": "Notebook",
    "export.btn.dataset": "Cleaned dataset", "export.btn.model": "Model (.joblib)",
    "export.checklist.dataset": "Cleaned dataset",
    "export.checklist.pdf": "PDF report",
    "export.checklist.notebook": "Jupyter notebook",
    "export.checklist.model": "Trained model (.joblib)",
    "export.checklist.metadata": "Metadata",
    "export.checklist.stats": "Statistics",

    "steps.0": "Loading data", "steps.1": "Automatic profiling",
    "steps.2": "Smart cleaning", "steps.3": "Exploratory analysis",
    "steps.4": "Target detection", "steps.5": "Data preparation",
    "steps.6": "Feature engineering", "steps.7": "Train / test",
    "steps.8": "Model selection", "steps.9": "AutoML & optimization",
    "steps.10": "Evaluation", "steps.11": "Explainability",
    "steps.12": "Error analysis", "steps.13": "Prediction",
    "steps.14": "Report generation",

    "nav.admin": "Admin",
    "admin.title": "Administration",
    "admin.users_title": "Registered users",
    "admin.analyses_title": "All analyses",
    "admin.table.email": "Email",
    "admin.table.name": "Name",
    "admin.table.admin": "Admin",
    "admin.table.created": "Signed up",
    "admin.table.user": "User",
    "admin.badge.yes": "Yes",
    "admin.badge.no": "No",
    "admin.no_users": "No users.",
    "admin.no_analyses": "No analyses saved yet.",
    "admin.refresh": "Refresh",
    "admin.error": "Unable to load admin data. Check your Supabase RLS policies.",
  },
};

function t(key) {
  const langue = document.documentElement.dataset.lang || "fr";
  return (TRADUCTIONS[langue] && TRADUCTIONS[langue][key]) || key;
}

function appliquerTraductions() {
  document.querySelectorAll("[data-i18n]").forEach((elt) => {
    const key = elt.dataset.i18n;
    const texte = t(key);
    if (elt.dataset.i18nAttr) elt.setAttribute(elt.dataset.i18nAttr, texte);
    else elt.textContent = texte;
  });
}

function appliquerTheme(theme) {
  document.documentElement.dataset.theme = theme === "light" ? "light" : "dark";
  localStorage.setItem("ia_ds_theme", document.documentElement.dataset.theme);
  const select = document.getElementById("settings-theme-select");
  if (select) select.value = document.documentElement.dataset.theme;
}

function appliquerLangue(langue) {
  document.documentElement.dataset.lang = langue === "en" ? "en" : "fr";
  localStorage.setItem("ia_ds_lang", document.documentElement.dataset.lang);
  appliquerTraductions();
  const select = document.getElementById("settings-lang-select");
  if (select) select.value = document.documentElement.dataset.lang;
  // Permet à app.js de retraduire le contenu déjà généré dynamiquement
  // (tableaux, résumés, checklist...) sans recharger la page.
  window.dispatchEvent(new CustomEvent("ia-ds-lang-changed"));
}

// Rendu accessible à app.js pour traduire tout texte généré dynamiquement.
window.t = t;

function chargerPrefsAnalyse() {
  return {
    trials: parseInt(localStorage.getItem("ia_ds_trials") || "10", 10),
    testsize: parseInt(localStorage.getItem("ia_ds_testsize") || "20", 10),
    notif_email: localStorage.getItem("ia_ds_notif_email") === "true",
  };
}

function sauvegarderPrefsAnalyse(prefs) {
  localStorage.setItem("ia_ds_trials", String(prefs.trials));
  localStorage.setItem("ia_ds_testsize", String(prefs.testsize));
  localStorage.setItem("ia_ds_notif_email", String(Boolean(prefs.notif_email)));
}

window.getPrefsAnalyse = chargerPrefsAnalyse;

appliquerTheme(localStorage.getItem("ia_ds_theme") || "dark");
appliquerLangue(localStorage.getItem("ia_ds_lang") || "fr");

let supabase = null;

// Devient vrai dès qu'on détecte un événement PASSWORD_RECOVERY.
// Empêche afficherDashboard() de s'exécuter tant que l'utilisateur
// n'a pas choisi son nouveau mot de passe.
let modeRecuperationMotDePasse = false;

function initSupabase() {
  if (MODE_INVITE) return false;
  if (!window.supabase) { console.warn("SDK Supabase non chargé."); return false; }
  try {
    supabase = window.supabase.createClient(SUPABASE_URL, SUPABASE_ANON_KEY);
    window.supabaseClient = supabase;
    return true;
  } catch (error) { console.error("Erreur init Supabase :", error); return false; }
}

// ================================================================
// HISTORIQUE DES ANALYSES
// ================================================================

async function sauvegarderAnalyse(nomFichier, resultat) {
  if (!supabase || !window.currentUser) return;
  const { error } = await supabase.from("analyses").insert({
    user_id:         window.currentUser.id,
    nom_fichier:     nomFichier,
    target:          resultat.target,
    problem_type:    resultat.problem_type,
    champion_modele: resultat.champion?.modele || null,
    score_cv:        resultat.champion?.score_cv || null,
    nb_features:     resultat.nombre_features || null,
    nb_lignes:       resultat.dataset_initial?.lignes || null,
    resultat:        resultat,
  });
  if (error) console.error("Erreur sauvegarde analyse :", error.message);
  else chargerHistorique();
}
window.sauvegarderAnalyse = sauvegarderAnalyse;

async function chargerHistorique() {
  const conteneur = document.getElementById("history-list");
  if (!conteneur) return;
  if (!supabase || !window.currentUser) {
    conteneur.innerHTML = `<p class="muted-text">${t("history.empty")}</p>`;
    return;
  }
  conteneur.innerHTML = `<p class="muted-text">Chargement…</p>`;
  const { data, error } = await supabase
    .from("analyses").select("*")
    .eq("user_id", window.currentUser.id)
    .order("created_at", { ascending: false });
  if (error || !data || data.length === 0) {
    conteneur.innerHTML = `<p class="muted-text">${t("history.empty")}</p>`;
    return;
  }
  const rows = data.map((a) => {
    const date = new Date(a.created_at).toLocaleString(
      document.documentElement.dataset.lang === "en" ? "en-GB" : "fr-FR"
    );
    return `<tr>
      <td>${date}</td>
      <td>${escapeHtml(a.nom_fichier || "—")}</td>
      <td>${escapeHtml(a.target || "—")}</td>
      <td>${escapeHtml((a.problem_type || "").replace(/_/g, " ") || "—")}</td>
      <td>${escapeHtml(a.champion_modele || "—")}</td>
      <td>${a.score_cv != null ? Number(a.score_cv).toFixed(4) : "—"}</td>
      <td><button class="btn-delete-history" data-id="${a.id}">${t("history.delete")}</button></td>
    </tr>`;
  }).join("");
  conteneur.innerHTML = `
    <table class="data-table history-table">
      <thead><tr>
        <th>${t("history.date")}</th><th>${t("history.file")}</th>
        <th>${t("history.target")}</th><th>${t("history.type")}</th>
        <th>${t("history.model")}</th><th>${t("history.score")}</th><th></th>
      </tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
  conteneur.querySelectorAll(".btn-delete-history").forEach((btn) => {
    btn.addEventListener("click", () => supprimerAnalyse(btn.dataset.id));
  });
}

async function supprimerAnalyse(id) {
  if (!supabase || !window.currentUser) return;
  await supabase.from("analyses").delete().eq("id", id).eq("user_id", window.currentUser.id);
  chargerHistorique();
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

// ================================================================
// AUTHENTIFICATION EMAIL / MOT DE PASSE
// ================================================================

// Mode : "login" ou "signup"
let modeAuth = "login";

function afficherErreurAuth(message) {
  const box = document.getElementById("auth-error");
  if (box) { box.textContent = message; box.hidden = false; }
}

function masquerErreurAuth() {
  const box = document.getElementById("auth-error");
  if (box) box.hidden = true;
}

function basculerModeAuth() {
  modeAuth = modeAuth === "login" ? "signup" : "login";
  masquerErreurAuth();

  const btnSubmit  = document.getElementById("auth-email-btn");
  const btnToggle  = document.getElementById("auth-toggle-btn");
  const btnForgot  = document.getElementById("auth-forgot-btn");

  if (modeAuth === "signup") {
    if (btnSubmit) btnSubmit.textContent = t("auth.signup");
    if (btnToggle) btnToggle.textContent = t("auth.has_account") + " " + t("auth.login");
    if (btnForgot) btnForgot.hidden = true;
  } else {
    if (btnSubmit) btnSubmit.textContent = t("auth.login");
    if (btnToggle) btnToggle.textContent = t("auth.no_account") + " " + t("auth.signup");
    if (btnForgot) btnForgot.hidden = false;
  }
}

async function connexionEmail() {
  if (!supabase) { afficherErreurAuth(t("auth.error.network")); return; }

  const email    = document.getElementById("auth-email-input")?.value?.trim();
  const password = document.getElementById("auth-password-input")?.value;

  if (!email || !password) {
    afficherErreurAuth(t("auth.error.invalid"));
    return;
  }

  masquerErreurAuth();
  const btn = document.getElementById("auth-email-btn");
  if (btn) btn.disabled = true;

  try {
    let result;
    if (modeAuth === "signup") {
      if (password.length < 6) {
        afficherErreurAuth(t("auth.error.weak_password"));
        return;
      }
      result = await supabase.auth.signUp({ email, password });
    } else {
      result = await supabase.auth.signInWithPassword({ email, password });
    }

    const { data, error } = result;
    if (error) {
      if (error.message.includes("Invalid login")) afficherErreurAuth(t("auth.error.invalid"));
      else if (error.message.includes("already registered")) afficherErreurAuth(t("auth.error.email_taken"));
      else if (error.message.includes("weak")) afficherErreurAuth(t("auth.error.weak_password"));
      else afficherErreurAuth(error.message || t("auth.error.generic"));
    } else if (modeAuth === "signup" && data && !data.session) {
      // Le compte a bien été créé côté Supabase, mais aucune session n'est
      // renvoyée : la confirmation par email est activée et l'utilisateur
      // doit cliquer sur le lien reçu avant de pouvoir se connecter.
      // Sans ce message, l'écran ne fait rien après "Créer un compte".
      afficherErreurAuth(t("auth.signup.confirm_needed"));
    }
  } catch (e) {
    afficherErreurAuth(t("auth.error.generic"));
  } finally {
    if (btn) btn.disabled = false;
  }
}

async function reinitialiserMotDePasse() {
  if (!supabase) { afficherErreurAuth(t("auth.error.network")); return; }
  const email = document.getElementById("auth-email-input")?.value?.trim();
  if (!email) { afficherErreurAuth(t("auth.error.invalid")); return; }
  masquerErreurAuth();
  const { error } = await supabase.auth.resetPasswordForEmail(email, {
    redirectTo: window.location.origin,
  });
  if (error) afficherErreurAuth(error.message || t("auth.error.generic"));
  else afficherErreurAuth(t("auth.reset_sent"));
}

async function connexionGoogle() {
  if (!supabase) { afficherErreurAuth(t("auth.error.network")); return; }
  masquerErreurAuth();
  const { error } = await supabase.auth.signInWithOAuth({
    provider: "google",
    options: { redirectTo: window.location.origin },
  });
  if (error) afficherErreurAuth(error.message || t("auth.error.generic"));
}

async function deconnexion() {
  if (supabase) await supabase.auth.signOut();
  window.currentUser = null;
  modeRecuperationMotDePasse = false;
  masquerEcranResetPassword();
  document.getElementById("auth-screen").hidden = false;
  document.getElementById("app-shell").hidden = true;
}

function afficherEcranAuth() {
  const authScreen = document.getElementById("auth-screen");
  const appShell = document.getElementById("app-shell");
  const resetScreen = document.getElementById("reset-password-screen");
  if (resetScreen) resetScreen.hidden = true;
  if (authScreen) authScreen.hidden = false;
  if (appShell) appShell.hidden = true;
}

// ================================================================
// REINITIALISATION DU MOT DE PASSE (après clic sur le lien reçu par mail)
// ================================================================

function afficherEcranResetPassword() {
  const authScreen  = document.getElementById("auth-screen");
  const appShell    = document.getElementById("app-shell");
  const resetScreen = document.getElementById("reset-password-screen");
  if (authScreen) authScreen.hidden = true;
  if (appShell) appShell.hidden = true;
  if (resetScreen) resetScreen.hidden = false;
}

function masquerEcranResetPassword() {
  const resetScreen = document.getElementById("reset-password-screen");
  if (resetScreen) resetScreen.hidden = true;
}

function afficherErreurReset(message) {
  const box = document.getElementById("reset-password-error");
  if (box) { box.textContent = message; box.hidden = false; }
}

function masquerErreurReset() {
  const box = document.getElementById("reset-password-error");
  if (box) box.hidden = true;
}

async function validerNouveauMotDePasse() {
  if (!supabase) { afficherErreurReset(t("auth.error.network")); return; }

  const password  = document.getElementById("reset-password-input")?.value || "";
  const password2 = document.getElementById("reset-password-confirm-input")?.value || "";

  masquerErreurReset();

  if (password.length < 6) {
    afficherErreurReset(t("auth.error.weak_password"));
    return;
  }
  if (password !== password2) {
    afficherErreurReset(t("auth.reset.mismatch"));
    return;
  }

  const btn = document.getElementById("reset-password-btn");
  if (btn) btn.disabled = true;

  try {
    const { error } = await supabase.auth.updateUser({ password });
    if (error) {
      afficherErreurReset(error.message || t("auth.error.generic"));
      return;
    }
    modeRecuperationMotDePasse = false;
    masquerEcranResetPassword();
    const { data: { session } } = await supabase.auth.getSession();
    if (session) afficherDashboard(session);
    else afficherEcranAuth();
  } catch (e) {
    afficherErreurReset(t("auth.error.generic"));
  } finally {
    if (btn) btn.disabled = false;
  }
}

// ================================================================
// PROFIL
// ================================================================

async function chargerProfil(userId) {
  if (!supabase) return null;
  const { data, error } = await supabase
    .from("profiles").select("theme, language, full_name, avatar_url, is_admin")
    .eq("id", userId).single();
  if (error) return null;
  if (data?.theme) appliquerTheme(data.theme);
  if (data?.language) appliquerLangue(data.language);
  return data;
}

async function sauvegarderProfil(userId, { theme, language }) {
  if (!supabase) return true;
  const { error } = await supabase.from("profiles").update({ theme, language }).eq("id", userId);
  return !error;
}

function syncControlesSettings() {
  const prefs = chargerPrefsAnalyse();
  const themeSelect = document.getElementById("settings-theme-select");
  if (themeSelect) themeSelect.value = document.documentElement.dataset.theme;
  const langSelect = document.getElementById("settings-lang-select");
  if (langSelect) langSelect.value = document.documentElement.dataset.lang;
  const trialsInput = document.getElementById("settings-trials");
  if (trialsInput) trialsInput.value = prefs.trials;
  const testsizeInput = document.getElementById("settings-testsize");
  if (testsizeInput) testsizeInput.value = prefs.testsize;
  const notifCheckbox = document.getElementById("settings-notif-email");
  if (notifCheckbox) notifCheckbox.checked = prefs.notif_email;
}

async function afficherDashboard(session) {
  // Si on est en plein flux de récupération de mot de passe, on ne
  // bascule pas sur le dashboard tant que le nouveau mot de passe
  // n'a pas été défini.
  if (modeRecuperationMotDePasse) {
    afficherEcranResetPassword();
    return;
  }

  const authScreen = document.getElementById("auth-screen");
  const appShell = document.getElementById("app-shell");
  const resetScreen = document.getElementById("reset-password-screen");
  if (authScreen) authScreen.hidden = true;
  if (resetScreen) resetScreen.hidden = true;
  if (appShell) appShell.hidden = false;

  window.currentUser = session.user;
  let profil = null;
  if (supabase) profil = await chargerProfil(session.user.id);
  // En mode invité (pas de Supabase configuré), on autorise l'accès à la
  // page admin pour permettre de la tester localement.
  window.currentUserIsAdmin = supabase ? !!profil?.is_admin : true;

  const nomAffiche =
    session.user.user_metadata?.full_name ||
    session.user.user_metadata?.name ||
    session.user.email;

  const elNom = document.getElementById("settings-user-name");
  if (elNom) elNom.textContent = nomAffiche;
  const elEmail = document.getElementById("settings-user-email");
  if (elEmail) elEmail.textContent = session.user.email;

  const avatarUrl = session.user.user_metadata?.avatar_url;
  const elAvatar = document.getElementById("settings-user-avatar");
  if (elAvatar && avatarUrl) { elAvatar.src = avatarUrl; elAvatar.hidden = false; }

  syncControlesSettings();
  chargerHistorique();
  window.dispatchEvent(new CustomEvent("ia-ds-authenticated"));
}

// ================================================================
// INITIALISATION
// ================================================================

document.addEventListener("DOMContentLoaded", () => {
  appliquerTraductions();
  const supabaseOk = initSupabase();

  if (MODE_INVITE || !supabaseOk) {
    const badge = document.getElementById("auth-dev-badge");
    if (badge) badge.hidden = false;
    afficherDashboard({
      user: { id: "local-user", email: "local@ia-ds.app", user_metadata: { full_name: "Utilisateur local" } },
    });
  } else {
    supabase.auth.onAuthStateChange((event, session) => {
      if (event === "PASSWORD_RECOVERY") {
        // L'utilisateur vient de cliquer sur le lien reçu par email.
        // On l'empêche d'atterrir directement sur le dashboard et on
        // affiche le formulaire de nouveau mot de passe à la place.
        modeRecuperationMotDePasse = true;
        afficherEcranResetPassword();
        return;
      }
      if (session) afficherDashboard(session);
      else afficherEcranAuth();
    });
    supabase.auth.getSession().then(({ data: { session } }) => {
      if (modeRecuperationMotDePasse) return;
      if (session) afficherDashboard(session);
      else afficherEcranAuth();
    });
  }

  // Boutons auth
  document.getElementById("auth-google-btn")?.addEventListener("click", connexionGoogle);
  document.getElementById("auth-email-btn")?.addEventListener("click", connexionEmail);
  document.getElementById("auth-toggle-btn")?.addEventListener("click", basculerModeAuth);
  document.getElementById("auth-forgot-btn")?.addEventListener("click", reinitialiserMotDePasse);

  // Permettre Entrée dans les champs
  ["auth-email-input", "auth-password-input"].forEach((id) => {
    document.getElementById(id)?.addEventListener("keydown", (e) => {
      if (e.key === "Enter") connexionEmail();
    });
  });

  // Ecran nouveau mot de passe
  document.getElementById("reset-password-btn")?.addEventListener("click", validerNouveauMotDePasse);
  ["reset-password-input", "reset-password-confirm-input"].forEach((id) => {
    document.getElementById(id)?.addEventListener("keydown", (e) => {
      if (e.key === "Enter") validerNouveauMotDePasse();
    });
  });

  // Paramètres
  document.getElementById("settings-logout-btn")?.addEventListener("click", deconnexion);
  document.getElementById("settings-theme-select")?.addEventListener("change", (e) => appliquerTheme(e.target.value));
  document.getElementById("settings-lang-select")?.addEventListener("change", (e) => appliquerLangue(e.target.value));
  document.getElementById("history-reload-btn")?.addEventListener("click", chargerHistorique);

  document.getElementById("settings-save-btn")?.addEventListener("click", async () => {
    const theme    = document.getElementById("settings-theme-select")?.value || "dark";
    const language = document.getElementById("settings-lang-select")?.value || "fr";
    const trials   = parseInt(document.getElementById("settings-trials")?.value || "10", 10);
    const testsize = parseInt(document.getElementById("settings-testsize")?.value || "20", 10);
    const notif_email = document.getElementById("settings-notif-email")?.checked || false;

    appliquerTheme(theme);
    appliquerLangue(language);
    sauvegarderPrefsAnalyse({ trials, testsize, notif_email });

    if (supabase && window.currentUser) {
      await sauvegarderProfil(window.currentUser.id, { theme, language });
    }

    const msg = document.getElementById("settings-save-msg");
    if (msg) {
      msg.textContent = t("settings.saved");
      msg.hidden = false;
      setTimeout(() => { msg.hidden = true; }, 3000);
    }
  });

  document.getElementById("settings-delete-btn")?.addEventListener("click", async () => {
    const input  = document.getElementById("settings-delete-confirm")?.value || "";
    const motCle = document.documentElement.dataset.lang === "en" ? "DELETE" : "SUPPRIMER";
    if (input !== motCle) { alert(t("settings.delete.confirm")); return; }
    if (supabase && window.currentUser) {
      await supabase.auth.admin.deleteUser(window.currentUser.id).catch(() => {});
      await deconnexion();
    }
  });
});
})();