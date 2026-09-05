(() => {
// ================================================================
// IA DATA SCIENTIST — Authentification Google uniquement
// ================================================================

const SUPABASE_URL = "https://tbljaeeeknfhpscqhlwg.supabase.co";
const SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InRibGphZWVla25maHBzY3FobHdnIiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODgyOTk0MzUsImV4cCI6MjEwMzg3NTQzNX0._Ii6_1NC7d2pjuMp5BUz3Y4nQ0Mp38vebr0-qzJQuZk";

// CORRECTION : ne plus considérer les vraies clés comme des placeholders
const MODE_INVITE = !SUPABASE_URL || !SUPABASE_ANON_KEY;

const TRADUCTIONS = {
  fr: {
    "nav.dataset": "Dataset",
    "nav.target": "Target",
    "nav.overview": "Overview",
    "nav.quality": "Data Quality",
    "nav.statistics": "Statistics",
    "nav.eda": "EDA",
    "nav.features": "Features",
    "nav.models": "Models",
    "nav.bestmodel": "Best Model",
    "nav.report": "Report",
    "nav.notebook": "Notebook",
    "nav.export": "Export",
    "nav.settings": "Paramètres",
    "dataset.title": "Charger votre dataset",
    "dataset.subtitle": "Importez votre fichier CSV, Excel ou autre format tabulaire",
    "dataset.dropzone": "Glissez-déposez votre fichier ici",
    "dataset.or": "ou",
    "dataset.browse": "Choisir un fichier",
    "target.title": "Sélectionner la variable cible",
    "target.subtitle": "Choisissez la variable que vous souhaitez prédire",
    "target.field": "Variable cible",
    "target.launch": "Lancer l'analyse",
    "overview.running.title": "Analyse en cours…",
    "overview.running.subtitle": "AI Data Scientist travaille pour vous",
    "settings.title": "Paramètres",
    "settings.appearance": "Apparence",
    "settings.theme": "Thème",
    "settings.theme.dark": "Sombre",
    "settings.theme.light": "Clair",
    "settings.language": "Langue",
    "settings.account": "Compte",
    "settings.logout": "Se déconnecter",
    "settings.save": "Enregistrer",
    "settings.saved": "Préférences enregistrées",
    "settings.restart": "Démarrer une nouvelle analyse",
    "settings.analysis": "Paramètres d'analyse",
    "settings.trials": "Nombre d'essais AutoML (trials)",
    "settings.trials.hint": "Plus de trials = meilleur modèle, mais plus lent (10–100)",
    "settings.testsize": "Proportion de données de test (%)",
    "settings.testsize.hint": "Pourcentage du dataset réservé à l'évaluation (10–40%)",
    "settings.notifications": "Notifications",
    "settings.notif.email": "Recevoir un email à la fin de chaque analyse",
    "settings.advanced": "Avancé",
    "settings.delete.account": "Supprimer mon compte",
    "settings.delete.confirm": "Tapez SUPPRIMER pour confirmer",
    "auth.welcome": "Bienvenue sur IA Data Scientist",
    "auth.subtitle": "Connectez-vous pour lancer vos analyses",
    "auth.google": "Continuer avec Google",
    "auth.error.generic": "Une erreur est survenue. Réessayez.",
    "auth.error.network": "Impossible de joindre le serveur d'authentification.",
    "auth.dev.mode": "Mode invité (Supabase non configuré)",
  },
  en: {
    "nav.dataset": "Dataset",
    "nav.target": "Target",
    "nav.overview": "Overview",
    "nav.quality": "Data Quality",
    "nav.statistics": "Statistics",
    "nav.eda": "EDA",
    "nav.features": "Features",
    "nav.models": "Models",
    "nav.bestmodel": "Best Model",
    "nav.report": "Report",
    "nav.notebook": "Notebook",
    "nav.export": "Export",
    "nav.settings": "Settings",
    "dataset.title": "Upload your dataset",
    "dataset.subtitle": "Import your CSV, Excel, or other tabular file",
    "dataset.dropzone": "Drag and drop your file here",
    "dataset.or": "or",
    "dataset.browse": "Choose a file",
    "target.title": "Select the target variable",
    "target.subtitle": "Choose the variable you want to predict",
    "target.field": "Target variable",
    "target.launch": "Run analysis",
    "overview.running.title": "Analysis in progress…",
    "overview.running.subtitle": "AI Data Scientist is working for you",
    "settings.title": "Settings",
    "settings.appearance": "Appearance",
    "settings.theme": "Theme",
    "settings.theme.dark": "Dark",
    "settings.theme.light": "Light",
    "settings.language": "Language",
    "settings.account": "Account",
    "settings.logout": "Sign out",
    "settings.save": "Save",
    "settings.saved": "Preferences saved",
    "settings.restart": "Start a new analysis",
    "settings.analysis": "Analysis settings",
    "settings.trials": "AutoML trials",
    "settings.trials.hint": "More trials = better model, but slower (10–100)",
    "settings.testsize": "Test data percentage (%)",
    "settings.testsize.hint": "Percentage of dataset reserved for evaluation (10–40%)",
    "settings.notifications": "Notifications",
    "settings.notif.email": "Receive an email when analysis is complete",
    "settings.advanced": "Advanced",
    "settings.delete.account": "Delete my account",
    "settings.delete.confirm": "Type DELETE to confirm",
    "auth.welcome": "Welcome to AI Data Scientist",
    "auth.subtitle": "Sign in to run your analyses",
    "auth.google": "Continue with Google",
    "auth.error.generic": "Something went wrong. Please try again.",
    "auth.error.network": "Cannot reach the authentication server.",
    "auth.dev.mode": "Guest mode (Supabase not configured)",
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
    if (elt.dataset.i18nAttr) {
      elt.setAttribute(elt.dataset.i18nAttr, texte);
    } else {
      elt.textContent = texte;
    }
  });
}

function appliquerTheme(theme) {
  document.documentElement.dataset.theme = theme === "light" ? "light" : "dark";
  localStorage.setItem("ia_ds_theme", document.documentElement.dataset.theme);
  // CORRECTION : synchroniser le select si déjà dans le DOM
  const select = document.getElementById("settings-theme-select");
  if (select) select.value = document.documentElement.dataset.theme;
}

function appliquerLangue(langue) {
  document.documentElement.dataset.lang = langue === "en" ? "en" : "fr";
  localStorage.setItem("ia_ds_lang", document.documentElement.dataset.lang);
  appliquerTraductions();
  // CORRECTION : synchroniser le select si déjà dans le DOM
  const select = document.getElementById("settings-lang-select");
  if (select) select.value = document.documentElement.dataset.lang;
}

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

function initSupabase() {
  if (MODE_INVITE) return false;
  if (!window.supabase) {
    console.warn("SDK Supabase non chargé.");
    return false;
  }

  try {
    supabase = window.supabase.createClient(SUPABASE_URL, SUPABASE_ANON_KEY);
    return true;
  } catch (error) {
    console.error("Erreur init Supabase :", error);
    return false;
  }
}

async function chargerProfil(userId) {
  if (!supabase) return null;

  const { data, error } = await supabase
    .from("profiles")
    .select("theme, language, full_name, avatar_url")
    .eq("id", userId)
    .single();

  if (error) return null;
  if (data?.theme) appliquerTheme(data.theme);
  if (data?.language) appliquerLangue(data.language);
  return data;
}

async function sauvegarderProfil(userId, { theme, language }) {
  if (!supabase) return true;
  const { error } = await supabase
    .from("profiles")
    .update({ theme, language })
    .eq("id", userId);
  return !error;
}

function afficherErreurAuth(message) {
  const box = document.getElementById("auth-error");
  if (box) {
    box.textContent = message;
    box.hidden = false;
  }
}

function masquerErreurAuth() {
  const box = document.getElementById("auth-error");
  if (box) box.hidden = true;
}

async function connexionGoogle() {
  if (!supabase) {
    afficherErreurAuth(t("auth.error.network"));
    return;
  }

  masquerErreurAuth();

  const { error } = await supabase.auth.signInWithOAuth({
    provider: "google",
    options: { redirectTo: window.location.origin },
  });

  if (error) {
    afficherErreurAuth(error.message || t("auth.error.generic"));
  }
}

async function deconnexion() {
  if (supabase) await supabase.auth.signOut();
  window.currentUser = null;
  document.getElementById("auth-screen").hidden = false;
  document.getElementById("app-shell").hidden = true;
}

function afficherEcranAuth() {
  const authScreen = document.getElementById("auth-screen");
  const appShell = document.getElementById("app-shell");
  if (authScreen) authScreen.hidden = false;
  if (appShell) appShell.hidden = true;
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
  const authScreen = document.getElementById("auth-screen");
  const appShell = document.getElementById("app-shell");
  if (authScreen) authScreen.hidden = true;
  if (appShell) appShell.hidden = false;

  window.currentUser = session.user;

  if (supabase) await chargerProfil(session.user.id);

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
  if (elAvatar && avatarUrl) {
    elAvatar.src = avatarUrl;
    elAvatar.hidden = false;
  }

  syncControlesSettings();
  window.dispatchEvent(new CustomEvent("ia-ds-authenticated"));
}

document.addEventListener("DOMContentLoaded", () => {
  appliquerTraductions();

  const supabaseOk = initSupabase();

  if (MODE_INVITE || !supabaseOk) {
    const badge = document.getElementById("auth-dev-badge");
    if (badge) badge.hidden = false;

    const fakeSession = {
      user: {
        id: "local-user",
        email: "local@ia-ds.app",
        user_metadata: { full_name: "Utilisateur local" },
      },
    };

    afficherDashboard(fakeSession);
  } else {
    supabase.auth.onAuthStateChange((_event, session) => {
      if (session) afficherDashboard(session);
      else afficherEcranAuth();
    });

    supabase.auth.getSession().then(({ data: { session } }) => {
      if (session) afficherDashboard(session);
      else afficherEcranAuth();
    });
  }

  document.getElementById("auth-google-btn")?.addEventListener("click", connexionGoogle);
  document.getElementById("settings-logout-btn")?.addEventListener("click", deconnexion);

  document.getElementById("settings-theme-select")?.addEventListener("change", (e) => appliquerTheme(e.target.value));
  document.getElementById("settings-lang-select")?.addEventListener("change", (e) => appliquerLangue(e.target.value));

  // CORRECTION : sauvegarder thème et langue + prefs analyse
  document.getElementById("settings-save-btn")?.addEventListener("click", async () => {
    const theme = document.getElementById("settings-theme-select")?.value || "dark";
    const language = document.getElementById("settings-lang-select")?.value || "fr";
    const trials = parseInt(document.getElementById("settings-trials")?.value || "10", 10);
    const testsize = parseInt(document.getElementById("settings-testsize")?.value || "20", 10);
    const notif_email = document.getElementById("settings-notif-email")?.checked || false;

    // Appliquer immédiatement
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
    const input = document.getElementById("settings-delete-confirm")?.value || "";
    const motCle = document.documentElement.dataset.lang === "en" ? "DELETE" : "SUPPRIMER";

    if (input !== motCle) {
      alert(t("settings.delete.confirm"));
      return;
    }

    if (supabase && window.currentUser) {
      await supabase.auth.admin.deleteUser(window.currentUser.id).catch(() => {});
      await deconnexion();
    }
  });
});
})();