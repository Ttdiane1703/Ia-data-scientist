// ================================================================
// AI DATA SCIENTIST — Dashboard
// ================================================================

// ================================================================
// CONFIGURATION — ADRESSE DU BACKEND RENDER
//
// OPTION 1 — Modifier ce fichier :
//   Remplace la valeur ci-dessous par ton URL Render exacte.
//   Elle ressemble à : https://ia-data-scientist-xxxx.onrender.com
//
// OPTION 2 — Via les Paramètres du site (recommandé) :
//   Va dans Paramètres > "Connexion backend" et colle ton URL.
//   Elle sera sauvegardée dans le navigateur automatiquement.
//
// EN LOCAL (uvicorn api:app) : laisser vide ""
// ================================================================

const _API_BASE_DEFAUT = "https://ia-data-scientist.onrender.com";

function getApiBase() {
  const hostname = window.location.hostname;
  const isLocalHost = hostname === "localhost" || hostname === "127.0.0.1" || hostname === "::1";

  if (isLocalHost) {
    return window.location.origin;
  }

  return window._API_BASE_OVERRIDE
    || localStorage.getItem("ia_ds_api_base")
    || _API_BASE_DEFAUT;
}

const API_BASE = getApiBase();

// Pré-chauffage du backend Render : le plan gratuit met le service en veille
// après 15 min d'inactivité, et la première requête après une veille peut
// prendre 30 à 50 secondes. On envoie une requête silencieuse dès le
// chargement de la page pour lancer le réveil avant même que l'utilisateur
// ne dépose un fichier — ça ne garantit rien mais ça réduit l'attente perçue.
if (API_BASE) {
  fetch(API_BASE, { method: "GET" }).catch(() => {});
}

const el = (id) => document.getElementById(id);

// t() est exposé globalement par auth.js. Fallback si jamais absent.
const tr = (key, vars) => {
  let texte = (window.t ? window.t(key) : key);
  if (vars) {
    Object.entries(vars).forEach(([k, v]) => {
      texte = texte.replace(`{${k}}`, v);
    });
  }
  return texte;
};

function getEtapes() {
  return Array.from({ length: 15 }, (_, i) => tr(`steps.${i}`));
}

const COULEURS = ["#3B82F6", "#EF4444", "#F5B942", "#A78BFA", "#22C55E", "#EC4899", "#38BDF8", "#FB923C"];

const state = {
  jobId: null,
  colonnes: [],
  resultat: null,
  pollTimer: null,
  chronoTimer: null,
  debutAnalyse: null,
  chargeQuality: false,
  chargeStats: false,
  chargeEda: false,
  chargeModels: false,
  chargeAdmin: false,
  nomFichierCourant: null,
  targetCourante: null,
};

// ----------------------------------------------------------------
// NAVIGATION
// ----------------------------------------------------------------

function activerNav(page, enabled = true) {
  const btn = document.querySelector(`.nav-item[data-page="${page}"]`);
  if (btn) btn.dataset.enabled = enabled ? "true" : "false";
}

function allerA(page) {
  const btn = document.querySelector(`.nav-item[data-page="${page}"]`);
  if (!btn || btn.dataset.enabled === "false") return;

  document.querySelectorAll(".nav-item").forEach((b) => b.classList.remove("is-active"));
  btn.classList.add("is-active");

  document.querySelectorAll(".page").forEach((p) => p.classList.remove("active"));
  const target = document.querySelector(`.page[data-page="${page}"]`);
  if (target) target.classList.add("active");

  if (page === "quality" && !state.chargeQuality) chargerQuality();
  if (page === "statistics" && !state.chargeStats) chargerStatistics();
  if (page === "eda" && !state.chargeEda) chargerEda();
  if (page === "models" && !state.chargeModels) chargerModels();
  if (page === "admin" && !state.chargeAdmin) chargerAdmin();
  if (page === "bestmodel") remplirBestModel();
  if (page === "features") remplirFeatures();
  if (page === "report" || page === "notebook" || page === "export") remplirLiens();
}

document.querySelectorAll(".nav-item").forEach((btn) => {
  btn.addEventListener("click", () => allerA(btn.dataset.page));
});

document.querySelectorAll("[data-goto]").forEach((btn) => {
  btn.addEventListener("click", () => allerA(btn.dataset.goto));
});

// ----------------------------------------------------------------
// ERREURS
// ----------------------------------------------------------------

function toast(message) {
  const t = el("toast-error");
  t.textContent = message;
  t.hidden = false;
  clearTimeout(toast._timer);
  toast._timer = setTimeout(() => { t.hidden = true; }, 5000);
}

// ----------------------------------------------------------------
// 1. UPLOAD
// ----------------------------------------------------------------

const dropzone = el("dropzone");
const fileInput = el("file-input");

dropzone.addEventListener("click", () => fileInput.click());
el("browse-btn").addEventListener("click", (e) => { e.stopPropagation(); fileInput.click(); });

dropzone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); }
});

["dragenter", "dragover"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => { e.preventDefault(); dropzone.classList.add("dragover"); })
);
["dragleave", "drop"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => { e.preventDefault(); dropzone.classList.remove("dragover"); })
);
dropzone.addEventListener("drop", (e) => {
  const fichier = e.dataTransfer.files[0];
  if (fichier) traiterFichier(fichier);
});
fileInput.addEventListener("change", (e) => {
  const fichier = e.target.files[0];
  if (fichier) traiterFichier(fichier);
});

async function traiterFichier(fichier) {
  el("dataset-error").hidden = true;

  // ---- Vérification de la configuration ----
  if (!API_BASE || API_BASE.includes("REMPLACE-PAR-TON-URL")) {
    afficherErreurDataset(tr("dataset.error.backend_not_configured"));
    return;
  }

  if (!fichier.name.toLowerCase().endsWith(".csv")) {
    afficherErreurDataset(tr("dataset.error.notcsv"));
    return;
  }

  const formData = new FormData();
  formData.append("fichier", fichier);

  toast(tr("dataset.uploading_wait"));

  try {
    const reponse = await fetch(`${getApiBase()}/api/upload`, { method: "POST", body: formData });
    if (!reponse.ok) {
      const detail = await reponse.json().catch(() => ({}));
      throw new Error(detail.detail || tr("dataset.error.upload_failed"));
    }
    const data = await reponse.json();
    state.jobId = data.job_id;
    state.colonnes = data.colonnes;
    state.nomFichierCourant = fichier.name;

    await afficherInfosDataset(fichier.name);

    activerNav("target");
    activerNav("quality");
    activerNav("statistics");
    activerNav("eda");

    remplirCiblePossibles(data.colonnes);
    allerA("target");
  } catch (err) {
    if (err.message.includes("Failed to fetch") || err.message.includes("NetworkError")) {
      afficherErreurDataset(tr("dataset.error.network"));
    } else {
      afficherErreurDataset(err.message);
    }
  }
}

function afficherErreurDataset(message) {
  const box = el("dataset-error");
  box.textContent = message;
  box.hidden = false;
}

async function afficherInfosDataset(nomFichier) {
  try {
    const reponse = await fetch(`${getApiBase()}/api/job/${state.jobId}/dataset-info`);
    const info = await reponse.json();

    const langue = document.documentElement.dataset.lang === "en" ? "en-GB" : "fr-FR";

    el("info-nom").textContent = nomFichier;
    el("info-lignes").textContent = info.lignes.toLocaleString(langue);
    el("info-colonnes").textContent = info.colonnes;
    el("info-taille").textContent = `${info.taille_ko} KB`;
    el("info-format").textContent = info.format;
    el("info-encodage").textContent = info.encodage;

    el("dataset-info-card").hidden = false;
  } catch (err) {
    toast(tr("dataset.error.fetch_info"));
  }
}

// ----------------------------------------------------------------
// 2. TARGET
// ----------------------------------------------------------------

function remplirCiblePossibles(colonnes) {
  const select = el("target-select");
  select.innerHTML = "";
  colonnes.forEach((c) => {
    const opt = document.createElement("option");
    opt.value = c; opt.textContent = c;
    select.appendChild(opt);
  });
  chargerDistributionCible();
}

el("target-select").addEventListener("change", chargerDistributionCible);

async function chargerDistributionCible() {
  const target = el("target-select").value;
  if (!target) return;
  state.targetCourante = target;

  try {
    const reponse = await fetch(`${getApiBase()}/api/job/${state.jobId}/target-distribution?target=${encodeURIComponent(target)}`);
    if (!reponse.ok) throw new Error(tr("target.error.distribution"));
    const data = await reponse.json();

    const typeBox = el("target-type-box");
    const typeValue = el("target-type-value");
    const labels = {
      classification_binaire: tr("target.type.classification_binaire"),
      classification_multiclasse: tr("target.type.classification_multiclasse"),
      regression: tr("target.type.regression"),
    };
    typeValue.textContent = labels[data.type] || data.type;
    typeBox.hidden = false;

    const donut = el("target-donut");
    const tbody = document.querySelector("#target-table tbody");
    tbody.innerHTML = "";

    if (data.type === "regression") {
      donut.innerHTML = "";
      tbody.innerHTML = `
        <tr><td>${tr("target.stats.min")}</td><td colspan="2">${data.min}</td></tr>
        <tr><td>${tr("target.stats.max")}</td><td colspan="2">${data.max}</td></tr>
        <tr><td>${tr("target.stats.mean")}</td><td colspan="2">${data.moyenne?.toFixed(2)}</td></tr>
        <tr><td>${tr("target.stats.median")}</td><td colspan="2">${data.mediane?.toFixed(2)}</td></tr>
      `;
    } else {
      donut.innerHTML = construireDonutSVG(data.repartition);
      data.repartition.forEach((r, i) => {
        const tr2 = document.createElement("tr");
        tr2.innerHTML = `
          <td><span class="dot-swatch" style="background:${COULEURS[i % COULEURS.length]}"></span></td>
          <td>${escapeHtml(r.valeur)}</td>
          <td>${r.count}</td>
          <td>${r.pourcentage}%</td>
        `;
        tbody.appendChild(tr2);
      });
    }
  } catch (err) {
    toast(tr("target.error.distribution"));
  }
}

function construireDonutSVG(repartition) {
  const size = 120, radius = 45, strokeWidth = 18;
  const circonference = 2 * Math.PI * radius;
  let cumule = 0;

  const segments = repartition.map((item, i) => {
    const frac = item.pourcentage / 100;
    const dash = frac * circonference;
    const svg = `<circle cx="${size/2}" cy="${size/2}" r="${radius}" fill="none"
      stroke="${COULEURS[i % COULEURS.length]}" stroke-width="${strokeWidth}"
      stroke-dasharray="${dash} ${circonference - dash}"
      stroke-dashoffset="${-cumule}"
      transform="rotate(-90 ${size/2} ${size/2})" />`;
    cumule += dash;
    return svg;
  }).join("");

  return `<svg viewBox="0 0 ${size} ${size}" width="${size}" height="${size}">${segments}</svg>`;
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

// ----------------------------------------------------------------
// 3. LANCEMENT DE L'ANALYSE
// ----------------------------------------------------------------

el("launch-btn").addEventListener("click", async () => {
  const target = el("target-select").value;
  if (!target || !state.jobId) return;

  const btn = el("launch-btn");
  btn.disabled = true;

  const formData = new FormData();
  formData.append("target", target);
  const prefs = window.getPrefsAnalyse ? window.getPrefsAnalyse() : { trials: 10, testsize: 20 };
  formData.append("trials", String(prefs.trials));
  formData.append("testsize", String(prefs.testsize));

  try {
    const reponse = await fetch(`${getApiBase()}/api/analyze/${state.jobId}`, { method: "POST", body: formData });
    if (!reponse.ok) {
      const detail = await reponse.json().catch(() => ({}));
      throw new Error(detail.detail || tr("target.error.launch"));
    }

    activerNav("overview");
    construireChecklist();
    el("overview-running").hidden = false;
    el("overview-done").hidden = true;
    state.debutAnalyse = Date.now();
    demarrerChrono();
    allerA("overview");
    demarrerSuivi();
  } catch (err) {
    toast(err.message);
  } finally {
    btn.disabled = false;
  }
});

function construireChecklist() {
  const ul = el("checklist");
  ul.innerHTML = "";
  getEtapes().forEach((label, i) => {
    const li = document.createElement("li");
    li.dataset.step = i + 1;
    li.innerHTML = `<span class="chk-icon"></span> ${label}`;
    ul.appendChild(li);
  });
}

function demarrerChrono() {
  clearInterval(state.chronoTimer);
  state.chronoTimer = setInterval(() => {
    const secondes = Math.floor((Date.now() - state.debutAnalyse) / 1000);
    const h = String(Math.floor(secondes / 3600)).padStart(2, "0");
    const m = String(Math.floor((secondes % 3600) / 60)).padStart(2, "0");
    const s = String(secondes % 60).padStart(2, "0");
    el("temps-ecoule").textContent = `${tr("overview.elapsed")}${h}:${m}:${s}`;
  }, 1000);
}

// ----------------------------------------------------------------
// 4. SUIVI DE PROGRESSION
// ----------------------------------------------------------------

function demarrerSuivi() {
  clearInterval(state.pollTimer);
  state.pollTimer = setInterval(async () => {
    try {
      const reponse = await fetch(`${getApiBase()}/api/status/${state.jobId}`);
      if (!reponse.ok) throw new Error();
      const etat = await reponse.json();

      mettreAJourChecklist(etat.etape, etat.statut);
      el("progress-fill").style.width = `${etat.progression}%`;
      el("progress-label").textContent = `${etat.progression}%`;

      if (etat.statut === "termine") {
        clearInterval(state.pollTimer);
        clearInterval(state.chronoTimer);
        state.resultat = etat.resultat;
        if (window.sauvegarderAnalyse) {
          window.sauvegarderAnalyse(state.nomFichierCourant || "—", state.resultat);
        }
        afficherOverviewDone();
      } else if (etat.statut === "erreur") {
        clearInterval(state.pollTimer);
        clearInterval(state.chronoTimer);
        toast(etat.erreur || tr("overview.error.generic"));
      }
    } catch (err) {
      clearInterval(state.pollTimer);
      toast(tr("overview.error.lost_connection"));
    }
  }, 1500);
}

function mettreAJourChecklist(etapeCourante, statut) {
  const numero = parseInt(String(etapeCourante || "0").match(/\d+/)?.[0] || "0", 10);
  document.querySelectorAll("#checklist li").forEach((li) => {
    const step = parseInt(li.dataset.step, 10);
    li.classList.remove("done", "active");
    if (step < numero || statut === "termine") li.classList.add("done");
    else if (step === numero) li.classList.add("active");
  });
}

// ----------------------------------------------------------------
// 5. OVERVIEW — RESULTATS
// ----------------------------------------------------------------

function afficherOverviewDone() {
  const r = state.resultat;
  if (!r) return;

  activerNav("features");
  activerNav("models");
  activerNav("bestmodel");
  activerNav("report");
  activerNav("notebook");
  activerNav("export");

  el("overview-running").hidden = true;
  el("overview-done").hidden = false;

  const grid = el("overview-stats");
  grid.innerHTML = "";
  const items = [
    [tr("overview.stats.dataset"), `${r.dataset_initial.lignes} ${tr("overview.stats.rows_suffix")}`],
    [tr("overview.stats.features_final"), r.nombre_features ?? "—"],
    [tr("overview.stats.best_model"), r.champion?.modele ?? "—"],
    [tr("overview.stats.problem_type"), (r.problem_type || "").replace(/_/g, " ")],
    [tr("overview.stats.target"), r.target],
    [tr("overview.stats.score_cv"), r.champion?.score_cv?.toFixed(4) ?? "—"],
  ];
  items.forEach(([label, value]) => {
    const div = document.createElement("div");
    div.innerHTML = `<span class="info-label">${label}</span><span class="info-value">${value}</span>`;
    grid.appendChild(div);
  });

  el("overview-resume").textContent = tr("overview.summary.template", {
    target: r.target,
    n: r.nombre_features ?? "—",
    model: r.champion?.modele ?? "—",
    score: r.champion?.score_cv?.toFixed(4) ?? "—",
  });

  el("overview-best-name").textContent = r.champion?.modele ?? "—";

  const isClassif = (r.problem_type || "").startsWith("classification");
  if (isClassif) {
    el("overview-f1").textContent = r.evaluation?.f1?.toFixed(4) ?? "—";
    el("overview-acc").textContent = r.evaluation?.accuracy?.toFixed(4) ?? "—";
  } else {
    el("overview-f1").textContent = r.evaluation?.r2?.toFixed(4) ?? "—";
    el("overview-acc").textContent = r.evaluation?.rmse?.toFixed(4) ?? "—";
  }
}

// ----------------------------------------------------------------
// 6. DATA QUALITY
// ----------------------------------------------------------------

let dernierQuality = null;

async function chargerQuality() {
  state.chargeQuality = true;
  try {
    const reponse = await fetch(`${getApiBase()}/api/job/${state.jobId}/data-quality`);
    const q = await reponse.json();
    dernierQuality = q;
    rendreQuality(q);
  } catch (err) {
    toast(tr("quality.error"));
  }
}

function rendreQuality(q) {
  const stats = el("quality-stats");
  stats.innerHTML = `
    <div><span class="q-label">${tr("quality.missing")}</span><span class="q-value ${q.valeurs_manquantes ? "q-warn" : "q-ok"}">${q.valeurs_manquantes}</span></div>
    <div><span class="q-label">${tr("quality.duplicates")}</span><span class="q-value ${q.doublons ? "q-warn" : "q-ok"}">${q.doublons}</span></div>
    <div><span class="q-label">${tr("quality.invalid")}</span><span class="q-value ${q.valeurs_invalides ? "q-warn" : "q-ok"}">${q.valeurs_invalides}</span></div>
    <div><span class="q-label">${tr("quality.outliers")}</span><span class="q-value ${q.outliers ? "q-warn" : "q-ok"}">${q.outliers}</span></div>
    <div><span class="q-label">${tr("quality.leakage")}</span><span class="q-value q-none">${q.fuite_donnees_detectee ? tr("quality.leakage.detected") : tr("quality.leakage.none")}</span></div>
  `;

  const tbody = document.querySelector("#quality-table tbody");
  tbody.innerHTML = "";
  if (q.details.length === 0) {
    tbody.innerHTML = `<tr><td colspan="4">${tr("quality.no_issues")}</td></tr>`;
  } else {
    q.details.forEach((d) => {
      const tr2 = document.createElement("tr");
      tr2.innerHTML = `<td>${escapeHtml(d.probleme)}</td><td>${escapeHtml(d.colonne)}</td><td>${d.nombre}</td><td>${escapeHtml(d.action)}</td>`;
      tbody.appendChild(tr2);
    });
  }
}

// ----------------------------------------------------------------
// 7. STATISTICS
// ----------------------------------------------------------------

let dernieresStats = null;

async function chargerStatistics() {
  state.chargeStats = true;
  try {
    const reponse = await fetch(`${getApiBase()}/api/job/${state.jobId}/statistics`);
    const s = await reponse.json();
    dernieresStats = s;
    rendreStatistics(s);
  } catch (err) {
    toast(tr("statistics.error"));
  }
}

function rendreStatistics(s) {
  const tbodyNum = document.querySelector("#stats-num-table tbody");
  tbodyNum.innerHTML = s.numeriques.map((v) => `
    <tr>
      <td>${escapeHtml(v.variable)}</td><td>${v.count}</td><td>${fmt(v.mean)}</td><td>${fmt(v.std)}</td>
      <td>${fmt(v.min)}</td><td>${fmt(v.p25)}</td><td>${fmt(v.p50)}</td><td>${fmt(v.p75)}</td>
      <td>${fmt(v.max)}</td><td>${fmt(v.skew)}</td><td>${fmt(v.kurt)}</td>
    </tr>`).join("") || `<tr><td colspan="11">${tr("statistics.no_numeric")}</td></tr>`;

  const tbodyCat = document.querySelector("#stats-cat-table tbody");
  tbodyCat.innerHTML = s.categorielles.map((v) => `
    <tr>
      <td>${escapeHtml(v.variable)}</td><td>${v.unique}</td><td>${escapeHtml(v.plus_frequent)}</td>
      <td>${v.frequence}</td><td>${v.pourcentage}%</td>
    </tr>`).join("") || `<tr><td colspan="5">${tr("statistics.no_categorical")}</td></tr>`;
}

function fmt(v) { return (v === null || v === undefined) ? "—" : Number(v).toFixed(2); }

// ----------------------------------------------------------------
// 8. EDA
// ----------------------------------------------------------------

let derniereEda = null;

async function chargerEda() {
  state.chargeEda = true;
  try {
    const reponse = await fetch(`${getApiBase()}/api/job/${state.jobId}/eda`);
    const d = await reponse.json();
    derniereEda = d;
    rendreEda(d);
  } catch (err) {
    toast(tr("eda.error"));
  }
}

function rendreEda(d) {
  const wrap = el("eda-histograms");
  wrap.innerHTML = "";
  d.distributions.forEach((dist, i) => {
    const max = Math.max(...dist.counts, 1);
    const bars = dist.counts.map((c) =>
      `<div class="bar" style="height:${(c / max) * 100}%; background:${COULEURS[i % COULEURS.length]}"></div>`
    ).join("");
    const card = document.createElement("div");
    card.className = "chart-card";
    card.innerHTML = `<h4>${tr("eda.distribution_of")} ${escapeHtml(dist.variable)}</h4><div class="bars">${bars}</div>`;
    wrap.appendChild(card);
  });
  if (d.distributions.length === 0) {
    wrap.innerHTML = `<p class="muted-text">${tr("eda.no_numeric")}</p>`;
  }

  const corrBox = el("eda-correlation");
  if (d.correlations) {
    const vars = d.correlations.variables;
    let html = `<table class="data-table"><thead><tr><th></th>${vars.map((v) => `<th>${escapeHtml(v)}</th>`).join("")}</tr></thead><tbody>`;
    d.correlations.matrice.forEach((ligne, i) => {
      html += `<tr><th>${escapeHtml(vars[i])}</th>`;
      ligne.forEach((val) => {
        html += `<td class="corr-cell" style="background:${couleurCorrelation(val)}">${val ?? "—"}</td>`;
      });
      html += `</tr>`;
    });
    html += `</tbody></table>`;
    corrBox.innerHTML = html;
  } else {
    corrBox.innerHTML = `<p class="muted-text">${tr("eda.not_enough_for_corr")}</p>`;
  }
}

function couleurCorrelation(v) {
  if (v === null || v === undefined) return "#1B2233";
  const intensite = Math.abs(v);
  if (v >= 0) return `rgba(59,130,246,${0.15 + intensite * 0.7})`;
  return `rgba(239,68,68,${0.15 + intensite * 0.7})`;
}

// ----------------------------------------------------------------
// 9. FEATURES
// ----------------------------------------------------------------

function remplirFeatures() {
  const r = state.resultat;
  const grid = el("features-stats");
  if (!r) { grid.innerHTML = ""; return; }
  grid.innerHTML = "";
  const items = [
    [tr("features.original_columns"), r.dataset_initial.colonnes],
    [tr("features.final_features"), r.nombre_features ?? "—"],
    [tr("features.rows_used"), r.dataset_nettoye.lignes],
  ];
  items.forEach(([label, value]) => {
    const div = document.createElement("div");
    div.innerHTML = `<span class="info-label">${label}</span><span class="info-value">${value}</span>`;
    grid.appendChild(div);
  });
}

// ----------------------------------------------------------------
// 10. MODELS
// ----------------------------------------------------------------

let derniersModels = null;

async function chargerModels() {
  state.chargeModels = true;
  try {
    const reponse = await fetch(`${getApiBase()}/api/job/${state.jobId}/models`);
    const d = await reponse.json();
    derniersModels = d;
    rendreModels(d);
  } catch (err) {
    toast(tr("models.error"));
  }
}

function rendreModels(d) {
  const tbody = document.querySelector("#models-table tbody");
  tbody.innerHTML = d.modeles.map((m) => `
    <tr class="${m.est_champion ? "model-best-row" : ""}">
      <td>${m.est_champion ? "★ " : ""}${escapeHtml(m.nom)}</td>
      <td>${m.score_cv?.toFixed(4) ?? "—"}</td>
    </tr>`).join("") || `<tr><td colspan="2">${tr("models.no_data")}</td></tr>`;

  el("models-note").textContent = d.note || "";
}

// ----------------------------------------------------------------
// 11. BEST MODEL
// ----------------------------------------------------------------

function remplirBestModel() {
  const r = state.resultat;
  if (!r || !r.champion) return;

  el("bestmodel-name").textContent = r.champion.modele;

  const isClassif = (r.problem_type || "").startsWith("classification");

  const perf = el("bestmodel-perf");
  perf.innerHTML = "";
  const lignesPerf = isClassif
    ? [[tr("bestmodel.metric.f1"), r.evaluation?.f1], [tr("bestmodel.metric.accuracy"), r.evaluation?.accuracy],
       [tr("bestmodel.metric.precision"), r.evaluation?.precision], [tr("bestmodel.metric.recall"), r.evaluation?.recall]]
    : [[tr("bestmodel.metric.mae"), r.evaluation?.mae], [tr("bestmodel.metric.rmse"), r.evaluation?.rmse], [tr("bestmodel.metric.r2"), r.evaluation?.r2]];
  lignesPerf.forEach(([k, v]) => {
    const div = document.createElement("div");
    div.innerHTML = `<span class="kv-key">${k}</span><span class="kv-val">${v?.toFixed(4) ?? "—"}</span>`;
    perf.appendChild(div);
  });

  const info = el("bestmodel-info");
  info.innerHTML = "";
  [
    [tr("bestmodel.info.problem_type"), (r.problem_type || "").replace(/_/g, " ")],
    [tr("bestmodel.info.score_cv"), r.champion.score_cv?.toFixed(4) ?? "—"],
    [tr("bestmodel.info.features_used"), r.nombre_features ?? "—"],
  ].forEach(([k, v]) => {
    const div = document.createElement("div");
    div.innerHTML = `<span class="kv-key">${k}</span><span class="kv-val">${v}</span>`;
    info.appendChild(div);
  });

  const params = el("bestmodel-params");
  params.innerHTML = "";
  Object.entries(r.champion.params || {}).forEach(([k, v]) => {
    const div = document.createElement("div");
    div.innerHTML = `<span class="kv-key">${k}</span><span class="kv-val">${v}</span>`;
    params.appendChild(div);
  });

  el("bestmodel-confirm").textContent = tr("bestmodel.confirm_template", { model: r.champion.modele });
}

// ----------------------------------------------------------------
// 12. REPORT / NOTEBOOK / EXPORT — LIENS DE TELECHARGEMENT
// ----------------------------------------------------------------

function remplirLiens() {
  if (!state.jobId) return;
  const base = `${getApiBase()}/api/download/${state.jobId}`;
  el("report-download").href = `${base}/pdf`;
  el("notebook-download").href = `${base}/notebook`;
  el("export-dl-pdf").href = `${base}/pdf`;
  el("export-dl-notebook").href = `${base}/notebook`;
  el("export-dl-dataset").href = `${base}/dataset`;
  el("export-dl-model").href = `${base}/model`;

  const checklist = el("export-checklist");
  const items = [
    tr("export.checklist.dataset"), tr("export.checklist.pdf"), tr("export.checklist.notebook"),
    tr("export.checklist.model"), tr("export.checklist.metadata"), tr("export.checklist.stats"),
  ];
  checklist.innerHTML = items.map((i) =>
    `<div class="export-item"><svg viewBox="0 0 24 24"><path d="M5 12l4 4 10-10" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"/></svg>${i}</div>`
  ).join("");
}

// ----------------------------------------------------------------
// 13. ADMIN
// ----------------------------------------------------------------

let dernierAdminUsers = null;
let dernieresAdminAnalyses = null;

function afficherOuMasquerNavAdmin() {
  const btn = el("nav-admin-btn");
  if (!btn) return;
  const estAdmin = !!window.currentUserIsAdmin;
  btn.hidden = !estAdmin;
  btn.dataset.enabled = estAdmin ? "true" : "false";
}

window.addEventListener("ia-ds-authenticated", afficherOuMasquerNavAdmin);

async function chargerAdmin() {
  const sb = window.supabaseClient;
  if (!sb) { toast(tr("admin.error")); return; }
  state.chargeAdmin = true;

  try {
    const [resProfils, resAnalyses] = await Promise.all([
      sb.from("profiles").select("id, email, full_name, is_admin, created_at").order("created_at", { ascending: false }),
      sb.from("analyses").select("*").order("created_at", { ascending: false }),
    ]);
    if (resProfils.error) throw resProfils.error;
    if (resAnalyses.error) throw resAnalyses.error;

    dernierAdminUsers = resProfils.data || [];
    dernieresAdminAnalyses = resAnalyses.data || [];
    rendreAdminUsers(dernierAdminUsers);
    rendreAdminAnalyses(dernieresAdminAnalyses, dernierAdminUsers);
  } catch (err) {
    toast(tr("admin.error"));
  }
}

function formaterDate(dateStr) {
  if (!dateStr) return "—";
  const langue = document.documentElement.dataset.lang === "en" ? "en-GB" : "fr-FR";
  return new Date(dateStr).toLocaleString(langue);
}

function rendreAdminUsers(profils) {
  const tbody = document.querySelector("#admin-users-table tbody");
  if (!tbody) return;
  if (!profils || profils.length === 0) {
    tbody.innerHTML = `<tr><td colspan="5">${tr("admin.no_users")}</td></tr>`;
    return;
  }
  const moiId = window.currentUser?.id;
  tbody.innerHTML = profils.map((p) => {
    const estMoi = p.id === moiId;
    const boutonRole = estMoi
      ? tr("admin.action.you")
      : `<button type="button" class="btn-outline admin-toggle-role" data-id="${p.id}" data-current="${p.is_admin ? "1" : "0"}">${p.is_admin ? tr("admin.action.demote") : tr("admin.action.promote")}</button>`;
    const boutonDelete = estMoi
      ? ""
      : `<button type="button" class="btn-outline admin-delete-user" data-id="${p.id}" data-email="${escapeHtml(p.email || "")}">${tr("admin.action.delete_user")}</button>`;
    return `
    <tr>
      <td>${escapeHtml(p.email || "—")}</td>
      <td>${escapeHtml(p.full_name || "—")}</td>
      <td>${p.is_admin ? tr("admin.badge.yes") : tr("admin.badge.no")}</td>
      <td>${formaterDate(p.created_at)}</td>
      <td style="display:flex;gap:8px;flex-wrap:wrap;">${boutonRole} ${boutonDelete}</td>
    </tr>`;
  }).join("");

  tbody.querySelectorAll(".admin-toggle-role").forEach((btn) => {
    btn.addEventListener("click", () => basculerRoleAdmin(btn.dataset.id, btn.dataset.current === "1"));
  });
  tbody.querySelectorAll(".admin-delete-user").forEach((btn) => {
    btn.addEventListener("click", () => supprimerUtilisateur(btn.dataset.id, btn.dataset.email));
  });
}

async function basculerRoleAdmin(userId, estActuellementAdmin) {
  const sb = window.supabaseClient;
  if (!sb) return;
  try {
    const { error } = await sb.from("profiles").update({ is_admin: !estActuellementAdmin }).eq("id", userId);
    if (error) throw error;
    toast(tr("admin.success.role_updated"));
    state.chargeAdmin = false;
    chargerAdmin();
  } catch (err) {
    toast(tr("admin.error.action_failed"));
  }
}

async function supprimerUtilisateur(userId, email) {
  const message = tr("admin.confirm.delete_user", { email: email || userId });
  if (!window.confirm(message)) return;

  const sb = window.supabaseClient;
  if (!sb) return;
  try {
    const { error: errAnalyses } = await sb.from("analyses").delete().eq("user_id", userId);
    if (errAnalyses) throw errAnalyses;
    const { error: errProfil } = await sb.from("profiles").delete().eq("id", userId);
    if (errProfil) throw errProfil;
    toast(tr("admin.success.user_deleted"));
    state.chargeAdmin = false;
    chargerAdmin();
  } catch (err) {
    toast(tr("admin.error.action_failed"));
  }
}

function rendreAdminAnalyses(analyses, profils) {
  const tbody = document.querySelector("#admin-analyses-table tbody");
  if (!tbody) return;
  if (!analyses || analyses.length === 0) {
    tbody.innerHTML = `<tr><td colspan="7">${tr("admin.no_analyses")}</td></tr>`;
    return;
  }
  const emailParId = {};
  (profils || []).forEach((p) => { emailParId[p.id] = p.email; });

  tbody.innerHTML = analyses.map((a) => `
    <tr>
      <td>${formaterDate(a.created_at)}</td>
      <td>${escapeHtml(emailParId[a.user_id] || a.user_id || "—")}</td>
      <td>${escapeHtml(a.nom_fichier || "—")}</td>
      <td>${escapeHtml(a.target || "—")}</td>
      <td>${escapeHtml(a.champion_modele || "—")}</td>
      <td>${a.score_cv != null ? Number(a.score_cv).toFixed(4) : "—"}</td>
      <td><button type="button" class="btn-outline admin-delete-analysis" data-id="${a.id}">${tr("admin.action.delete_analysis")}</button></td>
    </tr>`).join("");

  tbody.querySelectorAll(".admin-delete-analysis").forEach((btn) => {
    btn.addEventListener("click", () => supprimerAnalyseAdmin(btn.dataset.id));
  });
}

async function supprimerAnalyseAdmin(analyseId) {
  if (!window.confirm(tr("admin.confirm.delete_analysis"))) return;

  const sb = window.supabaseClient;
  if (!sb) return;
  try {
    const { error } = await sb.from("analyses").delete().eq("id", analyseId);
    if (error) throw error;
    toast(tr("admin.success.analysis_deleted"));
    state.chargeAdmin = false;
    chargerAdmin();
  } catch (err) {
    toast(tr("admin.error.action_failed"));
  }
}

el("admin-refresh-btn")?.addEventListener("click", () => {
  state.chargeAdmin = false;
  chargerAdmin();
});

// ----------------------------------------------------------------
// SETTINGS — RESTART
// ----------------------------------------------------------------

el("restart-btn").addEventListener("click", () => {
  window.location.reload();
});

// ----------------------------------------------------------------
// RETRADUCTION A CHAUD (changement de langue sans recharger la page)
// ----------------------------------------------------------------

window.addEventListener("ia-ds-lang-changed", () => {
  // Contenu déjà chargé depuis le backend : on le redessine avec les
  // données en cache, sans refaire d'appel réseau.
  if (dernierQuality) rendreQuality(dernierQuality);
  if (dernieresStats) rendreStatistics(dernieresStats);
  if (derniereEda) rendreEda(derniereEda);
  if (derniersModels) rendreModels(derniersModels);
  if (dernierAdminUsers) rendreAdminUsers(dernierAdminUsers);
  if (dernieresAdminAnalyses) rendreAdminAnalyses(dernieresAdminAnalyses, dernierAdminUsers);

  // Contenu dérivé du résultat d'analyse, déjà en mémoire.
  if (state.resultat) {
    if (!el("overview-done").hidden) afficherOverviewDone();
    remplirFeatures();
    remplirBestModel();
    remplirLiens();
  }

  // Type détecté + tableau de la variable cible.
  if (state.targetCourante && !el("target-type-box").hidden) {
    chargerDistributionCible();
  }

  // Infos dataset (unité "KB" ne change pas, mais le séparateur des
  // milliers dans "Lignes" dépend de la langue).
  if (state.jobId && !el("dataset-info-card").hidden && state.nomFichierCourant) {
    afficherInfosDataset(state.nomFichierCourant);
  }
});