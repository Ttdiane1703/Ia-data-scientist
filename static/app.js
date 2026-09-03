// ================================================================
// AI DATA SCIENTIST — Dashboard
// ================================================================

const el = (id) => document.getElementById(id);

const ETAPES = [
  "Chargement des données", "Profilage automatique", "Nettoyage intelligent",
  "Analyse exploratoire", "Détection de la cible", "Préparation des données",
  "Feature engineering", "Train / test", "Choix des modèles",
  "AutoML & optimisation", "Évaluation", "Explicabilité",
  "Analyse des erreurs", "Prédiction", "Génération du rapport",
];

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

  if (!fichier.name.toLowerCase().endsWith(".csv")) {
    afficherErreurDataset("Ce fichier n'est pas un .csv. Choisissez un fichier au format CSV.");
    return;
  }

  const formData = new FormData();
  formData.append("fichier", fichier);

  try {
    const reponse = await fetch("/api/upload", { method: "POST", body: formData });
    if (!reponse.ok) {
      const detail = await reponse.json().catch(() => ({}));
      throw new Error(detail.detail || "Le dépôt du fichier a échoué.");
    }
    const data = await reponse.json();
    state.jobId = data.job_id;
    state.colonnes = data.colonnes;

    await afficherInfosDataset(fichier.name);

    activerNav("target");
    activerNav("quality");
    activerNav("statistics");
    activerNav("eda");

    remplirCiblePossibles(data.colonnes);
    allerA("target");
  } catch (err) {
    if (err instanceof TypeError && err.message.toLowerCase().includes("fetch")) {
      afficherErreurDataset(
        "Impossible de joindre l'API. Lancez start_api.bat, puis ouvrez " +
        "http://127.0.0.1:8000/ dans le navigateur."
      );
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
    const reponse = await fetch(`/api/job/${state.jobId}/dataset-info`);
    const info = await reponse.json();

    el("info-nom").textContent = nomFichier;
    el("info-lignes").textContent = info.lignes.toLocaleString("fr-FR");
    el("info-colonnes").textContent = info.colonnes;
    el("info-taille").textContent = `${info.taille_ko} KB`;
    el("info-format").textContent = info.format;
    el("info-encodage").textContent = info.encodage;

    el("dataset-info-card").hidden = false;
  } catch (err) {
    toast("Impossible de récupérer les informations du dataset.");
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

  try {
    const reponse = await fetch(`/api/job/${state.jobId}/target-distribution?target=${encodeURIComponent(target)}`);
    if (!reponse.ok) throw new Error("Impossible de charger la distribution.");
    const data = await reponse.json();

    const typeBox = el("target-type-box");
    const typeValue = el("target-type-value");
    const labels = {
      classification_binaire: "Classification binaire",
      classification_multiclasse: "Classification multiclasse",
      regression: "Régression",
    };
    typeValue.textContent = labels[data.type] || data.type;
    typeBox.hidden = false;

    const donut = el("target-donut");
    const tbody = document.querySelector("#target-table tbody");
    tbody.innerHTML = "";

    if (data.type === "regression") {
      donut.innerHTML = "";
      tbody.innerHTML = `
        <tr><td>Min</td><td colspan="2">${data.min}</td></tr>
        <tr><td>Max</td><td colspan="2">${data.max}</td></tr>
        <tr><td>Moyenne</td><td colspan="2">${data.moyenne?.toFixed(2)}</td></tr>
        <tr><td>Médiane</td><td colspan="2">${data.mediane?.toFixed(2)}</td></tr>
      `;
    } else {
      donut.innerHTML = construireDonutSVG(data.repartition);
      data.repartition.forEach((r, i) => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td><span class="dot-swatch" style="background:${COULEURS[i % COULEURS.length]}"></span></td>
          <td>${escapeHtml(r.valeur)}</td>
          <td>${r.count}</td>
          <td>${r.pourcentage}%</td>
        `;
        tbody.appendChild(tr);
      });
    }
  } catch (err) {
    toast("Impossible de charger l'aperçu de la variable cible.");
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

  try {
    const reponse = await fetch(`/api/analyze/${state.jobId}`, { method: "POST", body: formData });
    if (!reponse.ok) {
      const detail = await reponse.json().catch(() => ({}));
      throw new Error(detail.detail || "Impossible de lancer l'analyse.");
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
  ETAPES.forEach((label, i) => {
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
    el("temps-ecoule").textContent = `Temps écoulé : ${h}:${m}:${s}`;
  }, 1000);
}

// ----------------------------------------------------------------
// 4. SUIVI DE PROGRESSION
// ----------------------------------------------------------------

function demarrerSuivi() {
  clearInterval(state.pollTimer);
  state.pollTimer = setInterval(async () => {
    try {
      const reponse = await fetch(`/api/status/${state.jobId}`);
      if (!reponse.ok) throw new Error();
      const etat = await reponse.json();

      mettreAJourChecklist(etat.etape, etat.statut);
      el("progress-fill").style.width = `${etat.progression}%`;
      el("progress-label").textContent = `${etat.progression}%`;

      if (etat.statut === "termine") {
        clearInterval(state.pollTimer);
        clearInterval(state.chronoTimer);
        state.resultat = etat.resultat;
        afficherOverviewDone();
      } else if (etat.statut === "erreur") {
        clearInterval(state.pollTimer);
        clearInterval(state.chronoTimer);
        toast(etat.erreur || "L'analyse a rencontré un problème.");
      }
    } catch (err) {
      clearInterval(state.pollTimer);
      toast("Connexion au serveur perdue pendant l'analyse.");
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
    ["Dataset", `${r.dataset_initial.lignes} lignes`],
    ["Features finales", r.nombre_features ?? "—"],
    ["Meilleur modèle", r.champion?.modele ?? "—"],
    ["Type de problème", (r.problem_type || "").replace(/_/g, " ")],
    ["Cible", r.target],
    ["Score CV", r.champion?.score_cv?.toFixed(4) ?? "—"],
  ];
  items.forEach(([label, value]) => {
    const div = document.createElement("div");
    div.innerHTML = `<span class="info-label">${label}</span><span class="info-value">${value}</span>`;
    grid.appendChild(div);
  });

  el("overview-resume").textContent =
    `AI Data Scientist a analysé votre dataset et identifié le meilleur modèle pour ` +
    `prédire ${r.target}. Les données ont été nettoyées, transformées en ${r.nombre_features ?? "—"} ` +
    `variables, et ${r.champion?.modele ?? "le modèle"} a été retenu comme champion ` +
    `avec un score de validation croisée de ${r.champion?.score_cv?.toFixed(4) ?? "—"}.`;

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

async function chargerQuality() {
  state.chargeQuality = true;
  try {
    const reponse = await fetch(`/api/job/${state.jobId}/data-quality`);
    const q = await reponse.json();

    const stats = el("quality-stats");
    stats.innerHTML = `
      <div><span class="q-label">Missing Values</span><span class="q-value ${q.valeurs_manquantes ? "q-warn" : "q-ok"}">${q.valeurs_manquantes}</span></div>
      <div><span class="q-label">Duplicates</span><span class="q-value ${q.doublons ? "q-warn" : "q-ok"}">${q.doublons}</span></div>
      <div><span class="q-label">Invalid Values</span><span class="q-value ${q.valeurs_invalides ? "q-warn" : "q-ok"}">${q.valeurs_invalides}</span></div>
      <div><span class="q-label">Outliers</span><span class="q-value ${q.outliers ? "q-warn" : "q-ok"}">${q.outliers}</span></div>
      <div><span class="q-label">Data Leakage</span><span class="q-value q-none">${q.fuite_donnees_detectee ? "Détectée" : "Aucune détectée"}</span></div>
    `;

    const tbody = document.querySelector("#quality-table tbody");
    tbody.innerHTML = "";
    if (q.details.length === 0) {
      tbody.innerHTML = `<tr><td colspan="4">Aucun problème détecté.</td></tr>`;
    } else {
      q.details.forEach((d) => {
        const tr = document.createElement("tr");
        tr.innerHTML = `<td>${escapeHtml(d.probleme)}</td><td>${escapeHtml(d.colonne)}</td><td>${d.nombre}</td><td>${escapeHtml(d.action)}</td>`;
        tbody.appendChild(tr);
      });
    }
  } catch (err) {
    toast("Impossible de charger le rapport de qualité des données.");
  }
}

// ----------------------------------------------------------------
// 7. STATISTICS
// ----------------------------------------------------------------

async function chargerStatistics() {
  state.chargeStats = true;
  try {
    const reponse = await fetch(`/api/job/${state.jobId}/statistics`);
    const s = await reponse.json();

    const tbodyNum = document.querySelector("#stats-num-table tbody");
    tbodyNum.innerHTML = s.numeriques.map((v) => `
      <tr>
        <td>${escapeHtml(v.variable)}</td><td>${v.count}</td><td>${fmt(v.mean)}</td><td>${fmt(v.std)}</td>
        <td>${fmt(v.min)}</td><td>${fmt(v.p25)}</td><td>${fmt(v.p50)}</td><td>${fmt(v.p75)}</td>
        <td>${fmt(v.max)}</td><td>${fmt(v.skew)}</td><td>${fmt(v.kurt)}</td>
      </tr>`).join("") || `<tr><td colspan="11">Aucune variable numérique.</td></tr>`;

    const tbodyCat = document.querySelector("#stats-cat-table tbody");
    tbodyCat.innerHTML = s.categorielles.map((v) => `
      <tr>
        <td>${escapeHtml(v.variable)}</td><td>${v.unique}</td><td>${escapeHtml(v.plus_frequent)}</td>
        <td>${v.frequence}</td><td>${v.pourcentage}%</td>
      </tr>`).join("") || `<tr><td colspan="5">Aucune variable catégorielle.</td></tr>`;
  } catch (err) {
    toast("Impossible de charger les statistiques.");
  }
}

function fmt(v) { return (v === null || v === undefined) ? "—" : Number(v).toFixed(2); }

// ----------------------------------------------------------------
// 8. EDA
// ----------------------------------------------------------------

async function chargerEda() {
  state.chargeEda = true;
  try {
    const reponse = await fetch(`/api/job/${state.jobId}/eda`);
    const d = await reponse.json();

    const wrap = el("eda-histograms");
    wrap.innerHTML = "";
    d.distributions.forEach((dist, i) => {
      const max = Math.max(...dist.counts, 1);
      const bars = dist.counts.map((c) =>
        `<div class="bar" style="height:${(c / max) * 100}%; background:${COULEURS[i % COULEURS.length]}"></div>`
      ).join("");
      const card = document.createElement("div");
      card.className = "chart-card";
      card.innerHTML = `<h4>Distribution de ${escapeHtml(dist.variable)}</h4><div class="bars">${bars}</div>`;
      wrap.appendChild(card);
    });
    if (d.distributions.length === 0) {
      wrap.innerHTML = `<p class="muted-text">Aucune variable numérique à visualiser.</p>`;
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
      corrBox.innerHTML = `<p class="muted-text">Pas assez de variables numériques pour calculer des corrélations.</p>`;
    }
  } catch (err) {
    toast("Impossible de charger les visualisations EDA.");
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
    ["Colonnes originales", r.dataset_initial.colonnes],
    ["Features finales", r.nombre_features ?? "—"],
    ["Lignes utilisées", r.dataset_nettoye.lignes],
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

async function chargerModels() {
  state.chargeModels = true;
  try {
    const reponse = await fetch(`/api/job/${state.jobId}/models`);
    const d = await reponse.json();

    const tbody = document.querySelector("#models-table tbody");
    tbody.innerHTML = d.modeles.map((m) => `
      <tr class="${m.est_champion ? "model-best-row" : ""}">
        <td>${m.est_champion ? "★ " : ""}${escapeHtml(m.nom)}</td>
        <td>${m.score_cv?.toFixed(4) ?? "—"}</td>
      </tr>`).join("") || `<tr><td colspan="2">Aucune donnée.</td></tr>`;

    el("models-note").textContent = d.note || "";
  } catch (err) {
    toast("Impossible de charger la comparaison des modèles.");
  }
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
    ? [["F1 Score", r.evaluation?.f1], ["Accuracy", r.evaluation?.accuracy],
       ["Precision", r.evaluation?.precision], ["Recall", r.evaluation?.recall]]
    : [["MAE", r.evaluation?.mae], ["RMSE", r.evaluation?.rmse], ["R2", r.evaluation?.r2]];
  lignesPerf.forEach(([k, v]) => {
    const div = document.createElement("div");
    div.innerHTML = `<span class="kv-key">${k}</span><span class="kv-val">${v?.toFixed(4) ?? "—"}</span>`;
    perf.appendChild(div);
  });

  const info = el("bestmodel-info");
  info.innerHTML = "";
  [
    ["Type de problème", (r.problem_type || "").replace(/_/g, " ")],
    ["Score CV", r.champion.score_cv?.toFixed(4) ?? "—"],
    ["Features utilisées", r.nombre_features ?? "—"],
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

  el("bestmodel-confirm").textContent =
    `${r.champion.modele} est le meilleur modèle parmi ceux testés selon la procédure d'évaluation utilisée.`;
}

// ----------------------------------------------------------------
// 12. REPORT / NOTEBOOK / EXPORT — LIENS DE TELECHARGEMENT
// ----------------------------------------------------------------

function remplirLiens() {
  if (!state.jobId) return;
  const base = `/api/download/${state.jobId}`;
  el("report-download").href = `${base}/pdf`;
  el("notebook-download").href = `${base}/notebook`;
  el("export-dl-pdf").href = `${base}/pdf`;
  el("export-dl-notebook").href = `${base}/notebook`;
  el("export-dl-dataset").href = `${base}/dataset`;
  el("export-dl-model").href = `${base}/model`;

  const checklist = el("export-checklist");
  const items = ["Dataset nettoyé", "Rapport PDF", "Notebook Jupyter", "Modèle entraîné (.joblib)", "Métadonnées", "Statistiques"];
  checklist.innerHTML = items.map((i) =>
    `<div class="export-item"><svg viewBox="0 0 24 24"><path d="M5 12l4 4 10-10" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"/></svg>${i}</div>`
  ).join("");
}

// ----------------------------------------------------------------
// SETTINGS — RESTART
// ----------------------------------------------------------------

el("restart-btn").addEventListener("click", () => {
  window.location.reload();
});