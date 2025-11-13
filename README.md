# BO Benchmark MVP

Eine minimale, reproduzierbare Pipeline, um **Bayesian Optimization (BO)** gegen **DOE-Baselines** auf klassischen Benchmarks zu vergleichen – mit sauberem Logging, Auswertung und Diagnoseplots.

## Inhalt

- `problems.py` — Benchmark-Probleme (Branin2D, SphereND, RosenbrockND, RastriginND, GoldsteinPrice2D, Hartmann6D) inkl. Mapping zwischen Originalraum und $0,1$^d.
- `methods.py` — Methoden-Adapter: `BO_EI`, `BO_KG`, `DOE_LHS`, `DOE_FF`, `DOE_CCD`, `OD`, `Random`.
- `runner.py` — Führt **einen** Run (problem × method × seed × budget × noise) aus und loggt CSV.
- `evaluator.py` — Lädt/aggregiert Logs und erzeugt:

  - Regret-Kurven (Median ± IQR), normalisierte Regret-Kurven
  - Boxplots @T (regret, regret_obs, regret_norm, best_so_far/best_so_far_obs)
  - Heatmaps (Average Rank & Win-Rate) – optional gefacetet nach `noise_pct`
  - Tabellen (AUC(Regret), durchschnittliche Ränge)

- `main.py` — Orchestrator: definiert den Versuchsplan, startet alle Runs, erzeugt erste Plots/Tables.
- `evaluate_latest.py` — Findet den neuesten `results/<timestamp>_*`-Ordner, fährt die Standard-Auswertung und erstellt Transparenz-Plots (2D/3D Samples).
- `viz_samples.py` — Visuelle Diagnostik für 2D-Probleme: Kontur- und 3D-Oberflächenplots + Stichprobenverlauf, je k-te Iteration.
- `tools/check_logs_monotonicity.py` — Konsistenz-Checks für Logs (Monotonie, Iterationslücken, Noise-Kohärenz, optionale f\*-Konsistenz).
- `requirements.txt` — Minimal benötigte Abhängigkeiten.

## Schnellstart

```bash
pip install -r requirements.txt
python main.py
# oder:
python evaluate_latest.py
```

Artefakte liegen danach unter `results/<timestamp>_<exp_name>/`.

## Versuchsplan (main.py)

- **Probleme**: z. B. `"Branin"`, `"Sphere"`, `"Rosenbrock"`, `"Rastrigin"`, `"GoldsteinPrice"`, `"Hartmann6"`.

  - **Dimensionen** für skalierbare Probleme bequem im Namen:
    `"Sphere5"`, `"sphere_3"`, `"Rosenbrock-10"` … → Dimension d=5/3/10.
    (Branin/GoldsteinPrice sind 2D; Hartmann6 ist 6D.)

- **Methoden**: `["BO_EI", "BO_KG", "DOE_LHS", "DOE_FF", "DOE_CCD", "OD", "Random"]` (beliebig kombinierbar).
- **Budgets**: als **Faktoren pro Dimension** (z. B. `[10, 20, 50]` → für d=5 entspricht das T∈{50, 100, 250}).
- **Seeds**: Liste von Seeds, z. B. `list(range(10))`.
- **init_factor** (nur BO): setzt `init_n = init_factor * dim`.
- **Noise**: `noise_pct` als **Vielfaches der std(f) auf $0,1$^d** (z. B. `0.0, 0.5, 1.5` → 0 %, 50 %, 150 %).

`main.py` schreibt außerdem:

- `config.json` (Plan & aufgelöste Budgets),
- `env.json` (Python/Plattform/optional git/pip freeze).

## Logging-Schema

**Eine CSV pro Run** (problem × method × seed × budget × noise). Spalten:

```
problem, method, seed, noise_seed, noise_pct, sigma_noise,
budget, iter, x, y, y_noisy,
best_so_far, best_so_far_obs,
regret, regret_obs, time_s
```

- `x` – Punkt im **Originalraum** (JSON-Liste).
- `y` – **true** objective (noise-frei).
- `y_noisy` – beobachteter Wert (mit Noise).
- `best_so_far` / `best_so_far_obs` – kumulative Minima (true/observed).
- `regret` / `regret_obs` – bzgl. bekanntem f\* (falls vorhanden).
- `noise_pct` – Anteil an `std(f)` auf $0,1$^d; `sigma_noise` ist die daraus resultierende Standardabweichung in Funktionswert-Einheiten.

> **Hinweis:** `regret_norm` wird **nicht** geloggt, sondern in der Auswertung erzeugt:
>
> $$
> \text{regret\_norm} = \frac{\text{best\_so\_far} - f^\*}{q_{95} - f^\*}\ \in [0,1]
> $$
>
> mit `q95`≈95-Perzentil von f(x) unter Uniform auf der Originaldomäne (per Monte-Carlo geschätzt).

## Auswertung & Plots

- **Kurven**: Median ± IQR über Seeds; facettiert nach Problem / Budget / `noise_pct` (wo vorhanden).
- **Boxplots @T**: je Problem × Budget × Noise, mit optionalem **Seed-Overlay** (alle Seeds als graue Punkte; wählbarer Highlight-Seed).

  - Regret (true & observed), normalisierter Regret, best value (true & observed).
  - Für normalisierten Regret: feste y-Achse $0,1$.
  - Für Regret (true/obs) kann die y-Achse über Noise-Facetten **fixiert** werden, um Vergleiche zu erleichtern.

- **Heatmaps**:

  - _Average Rank_ und _Win-Rate_ pro Gruppe (z. B. (Problem, Budget, Noise)).
  - Win-Rate = Anteil Seeds, in denen die Methode Rang 1 hat (ties erlaubt).

- **Tabellen** (`results/.../tables`):

  - `auc_regret_runs.csv` (pro Run),
  - `auc_regret_summary.csv` (Mittelwert/Std/95%-CI),
  - `average_rank_*` (final/AUC).

## Transparenz-Plots (Samples)

`evaluate_latest.py` erzeugt zusätzlich **2D-Konturen** und **3D-Oberflächen** mit den abgetasteten Punkten und der best-so-far-Trajektorie – **nur für 2D-Probleme**.
Standardmäßig werden pro Kombination (Problem, Methode) die Seeds `seeds_for_samples=[0]` betrachtet und alle vorhandenen `noise_pct`-Level separat geplottet. Frequenz via `k_every`.

Beispiel:

```bash
python evaluate_latest.py
# Probleme/Methoden automatisch aus Logs; Seeds & k_every oben im File anpassen
```

## Tools & Checks

**Monotonie-Check:**

```bash
python tools/check_logs_monotonicity.py results/<timestamp>_<exp_name>
```

Prüft u. a.:

- `iter` startet bei 0 und hat Schrittweite 1,
- `time_s` nicht fallend,
- `best_so_far` / `best_so_far_obs` monoton fallend & konsistent zu `cummin(y)` / `cummin(y_noisy)`,
- `regret` / `regret_obs` monoton,
- optionale f\*-Konsistenz,
- `noise_pct` konstant pro Datei und Übereinstimmung mit `_Nxx` im Dateinamen.

Schreibt `monotonicity_report.csv`.

## Ergebnisstruktur

```
results/<timestamp>_<exp_name>/
  all_runs.csv
  log_<Problem>_<Method>_seedS_BB_NNN.csv
  config.json
  env.json
  plots/
    regret_curves*/        # true/observed
    regret_norm_curves/
    regret/                # Boxplots (true/obs)
    regret_norm/           # Boxplots
    ybest/                 # y_best-Kurven
    heatmaps/              # Rank/Win-Rate
  samples/
    <Problem>/<Method>_seedS_noiseNN/
      samples2d_itK.png, samples3d_itK.png, ...
  tables/
    auc_regret_runs.csv
    auc_regret_summary.csv
    average_rank_final_regret.csv
    average_rank_auc_regret.csv
```

## Hinweise & Best Practices

- **Determinismus**: Runs sind deterministisch pro (problem, method, seed, noise_seed).
- **Noise** wirkt auf das **Lernen** (BO lernt aus `y_noisy`) und wird separat als `regret_obs`/`best_so_far_obs` sichtbar gemacht.
- **Budgets** immer als Faktor × d denken: so bleiben Probleme mit unterschiedlichen Dimensionen vergleichbar.
- **Methoden-Reihenfolge** in Boxplots über `METHOD_ORDER` festlegen, damit Plots konsistent bleiben.

## Abhängigkeiten

Siehe `requirements.txt`. Typisch:

- `numpy`, `pandas`, `matplotlib`
- `scipy`, `scikit-learn` (für GP-BO/Kernel usw.)
