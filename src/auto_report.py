from pathlib import Path
from datetime import datetime
import json
import html


class AutoReport:

    def __init__(self, output_dir="reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _safe(self, value):
        if value is None:
            return "N/A"

        if isinstance(value, float):
            return f"{value:.4f}"

        return html.escape(str(value))

    def generate(
        self,
        dataset_shape,
        cleaned_shape,
        target,
        problem_type,
        feature_count,
        champion_name,
        cv_score,
        evaluation=None,
        ranking=None,
        feature_importance=None,
        predictions=None
    ):

        evaluation = evaluation or {}
        ranking = ranking or []
        feature_importance = feature_importance or []
        predictions = predictions or []

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        test_score = evaluation.get("test_score", "N/A")
        train_score = evaluation.get("train_score", "N/A")
        validation_score = evaluation.get("validation_score", "N/A")
        overfitting = evaluation.get("overfitting", "N/A")
        quality = evaluation.get("quality", "N/A")
        decision = evaluation.get("decision", "N/A")

        ranking_html = ""

        for i, model in enumerate(ranking, start=1):

            if isinstance(model, dict):
                name = model.get("model", model.get("name", "Unknown"))
                score = model.get("score", model.get("cv_score", "N/A"))
            else:
                try:
                    name = model[0]
                    score = model[1]
                except Exception:
                    name = str(model)
                    score = "N/A"

            ranking_html += f"""
            <tr>
                <td>{i}</td>
                <td>{self._safe(name)}</td>
                <td>{self._safe(score)}</td>
            </tr>
            """

        importance_html = ""

        for i, item in enumerate(feature_importance[:20], start=1):

            if isinstance(item, dict):
                feature = item.get("feature", item.get("name", "Unknown"))
                importance = item.get(
                    "importance",
                    item.get("value", "N/A")
                )
            else:
                try:
                    feature = item[0]
                    importance = item[1]
                except Exception:
                    feature = str(item)
                    importance = "N/A"

            importance_html += f"""
            <tr>
                <td>{i}</td>
                <td>{self._safe(feature)}</td>
                <td>{self._safe(importance)}</td>
            </tr>
            """

        prediction_html = ""

        for prediction in predictions[:10]:
            prediction_html += f"""
            <li>{self._safe(prediction)}</li>
            """

        if not prediction_html:
            prediction_html = "<li>Aucune prédiction disponible</li>"

        html_report = f"""
<!DOCTYPE html>

<html lang="fr">

<head>

<meta charset="UTF-8">

<title>AI Data Scientist - Rapport</title>

<style>

body {{
    font-family: Arial, Helvetica, sans-serif;
    background: #f4f6f8;
    color: #222;
    margin: 0;
    padding: 0;
}}

.container {{
    width: 90%;
    max-width: 1200px;
    margin: auto;
}}

header {{
    background: #111827;
    color: white;
    padding: 40px;
    margin-bottom: 30px;
}}

header h1 {{
    margin: 0;
    font-size: 32px;
}}

header p {{
    opacity: 0.8;
}}

section {{
    background: white;
    padding: 25px;
    margin-bottom: 25px;
    border-radius: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}}

h2 {{
    margin-top: 0;
    color: #111827;
}}

.grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 15px;
}}

.card {{
    background: #f8fafc;
    padding: 20px;
    border-radius: 10px;
}}

.card .value {{
    font-size: 25px;
    font-weight: bold;
    margin-top: 8px;
}}

table {{
    width: 100%;
    border-collapse: collapse;
}}

th, td {{
    padding: 12px;
    border-bottom: 1px solid #ddd;
    text-align: left;
}}

th {{
    background: #f1f5f9;
}}

.badge {{
    display: inline-block;
    padding: 7px 12px;
    border-radius: 20px;
    background: #e5e7eb;
}}

.warning {{
    background: #fef3c7;
    padding: 15px;
    border-radius: 8px;
}}

.danger {{
    background: #fee2e2;
    padding: 15px;
    border-radius: 8px;
}}

.success {{
    background: #dcfce7;
    padding: 15px;
    border-radius: 8px;
}}

footer {{
    text-align: center;
    padding: 30px;
    color: #777;
}}

</style>

</head>

<body>

<header>

<div class="container">

<h1>🤖 AI DATA SCIENTIST</h1>

<p>Rapport automatique d'analyse Data Science</p>

<p>Généré le {timestamp}</p>

</div>

</header>

<div class="container">

<section>

<h2>📊 Vue d'ensemble</h2>

<div class="grid">

<div class="card">
Dataset initial
<div class="value">
{dataset_shape[0]} × {dataset_shape[1]}
</div>
</div>

<div class="card">
Dataset nettoyé
<div class="value">
{cleaned_shape[0]} × {cleaned_shape[1]}
</div>
</div>

<div class="card">
Variable cible
<div class="value">
{self._safe(target)}
</div>
</div>

<div class="card">
Problème
<div class="value">
{self._safe(problem_type)}
</div>
</div>

<div class="card">
Features
<div class="value">
{feature_count}
</div>
</div>

</div>

</section>


<section>

<h2>🎯 Problème détecté</h2>

<p>
L'agent a automatiquement identifié le problème suivant :
</p>

<p>
<strong>{self._safe(problem_type)}</strong>
</p>

<p>
Variable cible :
<strong>{self._safe(target)}</strong>
</p>

</section>


<section>

<h2>🏆 Modèle champion</h2>

<div class="grid">

<div class="card">
Modèle
<div class="value">
{self._safe(champion_name)}
</div>
</div>

<div class="card">
Score CV
<div class="value">
{self._safe(cv_score)}
</div>
</div>

<div class="card">
Qualité
<div class="value">
{self._safe(quality)}
</div>
</div>

</div>

</section>


<section>

<h2>🧪 Évaluation</h2>

<div class="grid">

<div class="card">
Score Train
<div class="value">
{self._safe(train_score)}
</div>
</div>

<div class="card">
Score Validation
<div class="value">
{self._safe(validation_score)}
</div>
</div>

<div class="card">
Score Test
<div class="value">
{self._safe(test_score)}
</div>
</div>

<div class="card">
Overfitting
<div class="value">
{self._safe(overfitting)}
</div>
</div>

</div>

<br>

<div class="warning">

<strong>Décision :</strong>

{self._safe(decision)}

</div>

</section>


<section>

<h2>🤖 Classement AutoML</h2>

<table>

<thead>

<tr>
<th>Rang</th>
<th>Modèle</th>
<th>Score CV</th>
</tr>

</thead>

<tbody>

{ranking_html}

</tbody>

</table>

</section>


<section>

<h2>🔎 Variables importantes</h2>

<table>

<thead>

<tr>
<th>Rang</th>
<th>Variable</th>
<th>Importance</th>
</tr>

</thead>

<tbody>

{importance_html}

</tbody>

</table>

</section>


<section>

<h2>🔮 Prédictions</h2>

<ul>

{prediction_html}

</ul>

</section>


<section>

<h2>🧠 Analyse automatique</h2>

<p>

L'agent Data Scientist a automatiquement exécuté les différentes
étapes du processus d'analyse :

</p>

<ul>

<li>Chargement des données</li>

<li>Profilage automatique</li>

<li>Nettoyage intelligent</li>

<li>Analyse exploratoire</li>

<li>Détection du problème ML</li>

<li>Feature Engineering</li>

<li>Séparation Train/Test</li>

<li>Entraînement de plusieurs modèles</li>

<li>Optimisation des hyperparamètres</li>

<li>Sélection du meilleur modèle</li>

<li>Évaluation</li>

<li>Analyse de l'importance des variables</li>

<li>Prédiction</li>

</ul>

</section>


<section>

<h2>🚨 Recommandation</h2>

<div class="danger">

L'agent recommande de ne pas utiliser le modèle en production
si les performances sont insuffisantes ou si l'overfitting est élevé.

</div>

</section>

</div>


<footer>

AI DATA SCIENTIST — Automated Machine Learning System

</footer>

</body>

</html>
"""

        report_path = self.output_dir / "ai_data_scientist_report.html"

        report_path.write_text(
            html_report,
            encoding="utf-8"
        )

        print()
        print("=" * 70)
        print("📄 RAPPORT AUTOMATIQUE")
        print("=" * 70)

        print(f"✅ Rapport généré : {report_path}")

        return str(report_path)