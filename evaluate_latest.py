"""
evaluate_latest.py
------------------
Convenience entry point:

1) Find the most recent results/<timestamp>_* folder.
2) (Optionally) run the standard evaluation (aggregate curves/boxplots/tables).
3) (Optionally) generate transparency plots (2D contour + 3D surface) for selected
   problems/methods/seeds (every k-th iteration or @T only).
4) Build an HTML report 'benchmark_report.html' (Boxplots + Tabellen nebeneinander).

Defaults:
- problems_for_samples = "auto"  -> all 2D problems present in logs
- methods_for_samples  = "auto"  -> all methods present in logs
- seeds_for_samples    = "auto"  -> all seeds present in logs
- final_samples_at_T   = True    -> only the final T
- k_every              = 5       -> used if final_samples_at_T=False
"""
from __future__ import annotations

try:
    from tb_style import apply as _apply_style
    _apply_style('thesis_v1')
except Exception:
    pass
import os
import re
import html
import datetime
from pathlib import Path
from typing import List, Sequence, Union, Optional
from evaluator import render_boxplot_side_tables, save_best_at_T_tables_as_png


import pandas as pd

# Aggregates & tables
from evaluator import evaluate_folder, export_best_at_T_table
try:
    # robust loader (supports logs/, all_runs.csv)
    from evaluator import load_logs as _load_any_logs
except Exception:
    _load_any_logs = None

# Samples (2D/3D)
from viz_samples import plot_samples_every_k
from problems import get_problem


# ---------------------- user-facing knobs (easy to change) ----------------------
# "auto" = automatisch aus Logs erkennen
problems_for_samples: Union[str, Sequence[str]] = "auto"
methods_for_samples:  Union[str, Sequence[str]] = "auto"
seeds_for_samples:    Union[str, List[int]] = "auto"

# Nur finale Sampleplots @T? (reduziert Informationsflut)
final_samples_at_T: bool = True   # nur @T
# Jede k-te Iteration; für @T-only egal
k_every: int = 5                  # egal bei @T-only

# Samples hier zentral einschalten
# auf True setzen, wenn Samples erstellt werden sollen
generate_samples: bool = True

# Rebuild aggregates (curves/boxplots/heatmaps) on this run?
REBUILD_AGGREGATES: bool = True  # -> False = Aggregate nicht neu rendern

BESTAT_T_DECIMALS: int = 4

# --------------------------------------------------------------------------------


def find_latest_results(root: Path) -> Path:
    """Pick the most recently modified folder under ./results (timestamp_*)."""
    results_root = root / "results"
    if not results_root.exists():
        raise FileNotFoundError(
            "No 'results' folder found. Run main.py first.")
    candidates = [p for p in results_root.glob("*_*") if p.is_dir()]
    if not candidates:
        raise FileNotFoundError("No results folders found under ./results")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def detect_2d_problems_from_logs(df: pd.DataFrame) -> List[str]:
    """
    Detect which 2D problems are present in the logs by checking dim via get_problem(...).
    Returns the *logged* names.
    """
    probs = sorted(df["problem"].astype(str).unique().tolist())
    out = []
    for p in probs:
        try:
            if get_problem(p).dim == 2:
                out.append(p)
        except Exception:
            if any(k in p.lower() for k in ("branin", "goldstein", "rastrigin", "sphere", "rosenbrock")):
                out.append(p)
    # preserve order while deduping
    seen = set()
    dedup = []
    for x in out:
        if x not in seen:
            seen.add(x)
            dedup.append(x)
    return dedup


def _collect_images(root_dir):
    """Sammelt alle PNGs rekursiv als Path-Objekte."""
    root = Path(root_dir)
    return sorted(root.rglob("*.png"), key=lambda p: str(p).lower())


def _classify(img: Path):
    """Klassifiziert Bilddateien nach Namenskonventionen."""
    name = img.name
    is_boxplot = name.startswith(
        "bestvalue_box_") and not name.endswith("_table.png")
    is_boxplot_table = name.startswith(
        "bestvalue_box_") and name.endswith("_table.png")
    is_big_best_at_T = name.startswith(
        "best_at_T_by_seed_method_") and name.endswith(".png")
    return is_boxplot, is_boxplot_table, is_big_best_at_T


def _parse_box_meta(img_path):
    """
    Erwartet Namen wie: bestvalue_box_Branin2D_T20_N50.png
    Liefert (problem, T, N_float) = ("Branin2D", 20, 0.50)
    """
    name = Path(img_path).name
    stem = Path(img_path).stem  # ohne .png
    # Muster: bestvalue_box_<problem>_T<budget>_N<noise*100>
    m = re.match(r"bestvalue_box_(.+?)_T(\d+)_N(\d+)$",
                 stem, flags=re.IGNORECASE)
    if m:
        problem = m.group(1)
        T = int(m.group(2))
        N100 = int(m.group(3))  # 0, 50, 150, ...
        N = N100 / 100.0
        return (problem, T, N)
    # Fallback: alles 0, damit es nicht crasht
    return (stem, 0, 0.0)


def _pair_plots_with_tables(images):
    """
    Paart ALLE Bilder, für die ein <basename>_table.png existiert.
    Rückgabe:
      pairs: Liste (plot_path, table_path, title),
      used:  Set verwendeter Pfade
    """
    from pathlib import Path
    used = set()
    pairs = []
    by_name = {img.name: img for img in images}

    for img in images:
        name = img.name
        if name.endswith("_table.png"):
            continue
        base = name[:-4]  # ohne .png
        table_name = f"{base}_table.png"
        table = by_name.get(table_name)
        if table is not None:
            title = base.replace("_", " ")
            pairs.append((img, table, title))
            used.add(img)
            used.add(table)

    # Stabile Sortierung: Problem → Methode → T → N (falls parsbar)
    import re

    def _sort_key(p):
        nm = Path(p[0]).name  # Paar anhand Plot-Namens
        # häufige Muster:
        # bestvalue_box_<Prob>_T<T>_N<N...>
        # bestvalue_box_method_<Prob>_<Meth>_T<T>_Nall
        m = re.search(r"_T(\d+)_N(\d+|all)", nm, flags=re.I)
        T = int(m.group(1)) if m else 0
        N = m.group(2) if m else "0"
        return (nm.lower(), T, str(N))
    pairs.sort(key=_sort_key)
    return pairs, used


def _write_html_gallery(report_path, box_pairs, big_tables, other_plots, results_dir):
    """Schreibt eine klare, deduplizierte HTML-Seite mit 3 Abschnitten."""
    def rel(p): return os.path.relpath(
        p, start=os.path.dirname(report_path)).replace("\\", "/")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("<!doctype html><html><head><meta charset='utf-8'>")
        f.write("<style>")
        f.write(
            "body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu; margin:16px;}")
        f.write(
            ".grid2{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:28px;}")
        f.write(
            ".grid3{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:12px;}")
        f.write(
            ".card{border:1px solid #ddd;border-radius:10px;padding:10px;background:#fff;}")
        f.write(".title{font-weight:600;margin:6px 0 8px 0;}")
        f.write(
            "img{max-width:100%;height:auto;border-radius:6px;display:block;}")
        f.write(
            "h2{margin-top:28px;border-bottom:1px solid #eee;padding-bottom:6px;}")
        f.write("</style></head><body>")
        f.write(f"<h1>Benchmark Report</h1>")
        f.write(
            f"<p><small>Results: {html.escape(str(results_dir))}</small></p>")

        # Section 1: Boxplots + Tabellen (gepairt)
        f.write("<h2>Boxplots mit Seitentabellen</h2>")
        for plot, table, title in box_pairs:
            f.write("<div class='grid2'>")
            f.write("<div class='card'>")
            f.write(
                f"<div class='title'>{html.escape('best_value — ' + title)}</div>")
            f.write(f"<img src='{rel(str(plot))}' loading='lazy'/>")
            f.write("</div>")
            f.write("<div class='card'>")
            if table is not None:
                f.write(
                    f"<div class='title'>{html.escape('Tabelle — ' + title)}</div>")
                f.write(f"<img src='{rel(str(table))}' loading='lazy'/>")
            else:
                f.write("<div class='title'>Keine Tabelle gefunden</div>")
                f.write("<p>Für diesen Boxplot wurde keine *_table.png erzeugt.</p>")
            f.write("</div></div>")

        # Section 2: Große Best-at-T Tabellen
        if big_tables:
            f.write("<h2>Best-at-T Gesamt-Tabellen</h2>")
            f.write("<div class='grid3'>")
            for img in big_tables:
                f.write("<div class='card'>")
                f.write(
                    f"<div class='title'>{html.escape(img.name.replace('_', ' '))}</div>")
                f.write(f"<img src='{rel(str(img))}' loading='lazy'/>")
                f.write("</div>")
            f.write("</div>")

        # Section 3: Weitere Plots
        if other_plots:
            f.write("<h2>Weitere Plots</h2>")
            f.write("<div class='grid3'>")
            for img in other_plots:
                f.write("<div class='card'>")
                f.write(
                    f"<div class='title'>{html.escape(img.name.replace('_', ' '))}</div>")
                f.write(f"<img src='{rel(str(img))}' loading='lazy'/>")
                f.write("</div>")
            f.write("</div>")

        f.write("</body></html>")


def _load_all_runs(results_dir: Path) -> pd.DataFrame:
    """Prefer evaluator.load_logs; fallback to legacy root log_*.csv."""
    if _load_any_logs is not None:
        return _load_any_logs(results_dir)
    all_runs_csv = results_dir / "all_runs.csv"
    if all_runs_csv.exists():
        return pd.read_csv(all_runs_csv)
    logs = sorted(results_dir.glob("log_*.csv"))
    if not logs:
        raise FileNotFoundError(
            f"No 'all_runs.csv' or 'log_*.csv' found in {results_dir}")
    return pd.concat([pd.read_csv(p) for p in logs], ignore_index=True)


def build_benchmark_report(results_dir: Path) -> Path:
    """
    Build a clickable HTML report with:
      - all PNGs under plots/** (recursive)
      - all PNGs under samples/** (recursive, faceted by problem → run)
      - all PNGs under tables/ (flat)
    and pair boxplots with best-at-T table images side-by-side.
    Always overwrites 'benchmark_report.html'.
    """
    results_dir = Path(results_dir)
    plots_dir = results_dir / "plots"
    samples_dir = results_dir / "samples"
    tables_dir = results_dir / "tables"
    out_html = results_dir / "benchmark_report.html"

    def _rel(p: Path) -> str:
        return os.path.relpath(p, results_dir).replace("\\", "/")

    def _src(p: Path) -> str:
        # Cache-Busting via mtime
        try:
            ver = int(p.stat().st_mtime)
        except Exception:
            ver = 0
        return f"{_rel(p)}?v={ver}"

    def _rglob_pngs(root: Path) -> List[Path]:
        if not root.exists():
            return []
        return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".png"]

    plot_imgs = _rglob_pngs(plots_dir)
    sample_imgs = _rglob_pngs(samples_dir)
    table_imgs = [p for p in (tables_dir.glob(
        "*.png") if tables_dir.exists() else [])]

    try:
        print(
            f"[report] found: plots={len(plot_imgs)}, samples={len(sample_imgs)}, tables={len(table_imgs)}")
    except Exception:
        pass

    # --- Boxplot ↔ Tabelle Zuordnung (per Dateiname) ---
    boxplot_re = re.compile(
        r"(regret_box_|regret_norm_box_|bestvalue_box_)", re.IGNORECASE)
    key_boxplots = [p for p in plot_imgs if boxplot_re.search(p.name)]

    def _find_table_for_boxplot(bp: Path) -> Optional[Path]:
        # 1) Sibling bevorzugen
        sib = bp.with_name(bp.stem + "_table.png")
        if sib.exists():
            return sib

        # 2) Fallback: globale Best-at-T PNGs
        name_lower = bp.name.lower()
        candidates: List[str] = []
        if "regret_norm" in name_lower:
            candidates = ["best_at_t_by_seed_method_norm.png",
                          "best_at_t_by_seed_method_norm-1.png"]
        elif "bestvalue_box_" in name_lower and "_obs" in name_lower:
            candidates = ["best_at_t_by_seed_method_obs.png",
                          "best_at_t_by_seed_method_obs-1.png"]
        elif "bestvalue_box_" in name_lower:
            candidates = ["best_at_t_by_seed_method_true.png",
                          "best_at_t_by_seed_method_true-1.png"]
        elif "regret_box_" in name_lower and "_obs" in name_lower:
            candidates = ["best_at_t_by_seed_method_obs.png",
                          "best_at_t_by_seed_method_obs-1.png"]
        else:
            candidates = ["best_at_t_by_seed_method_true.png",
                          "best_at_t_by_seed_method_true-1.png"]

        for nm in candidates:
            for t in table_imgs:
                if t.name.lower() == nm:
                    return t
        return None

    def _pairs_section() -> str:
        if not key_boxplots:
            return "<i>Keine Boxplots gefunden.</i>"
        rows = []
        for bp in sorted(key_boxplots):
            tbl = _find_table_for_boxplot(bp)
            right_html = (
                f'<img src="{_src(tbl)}" alt="{html.escape(tbl.name)}" loading="lazy">'
                if tbl is not None else "<div class='muted'>Keine passende Tabelle gefunden.</div>"
            )
            rows.append(f"""
            <div class="row">
              <div class="col">
                <div class="cap">{html.escape(bp.parent.name + ' — ' + bp.name)}</div>
                <a href="{_rel(bp)}" target="_blank">
                  <img src="{_src(bp)}" loading="lazy" alt="{html.escape(bp.name)}">
                </a>
              </div>
              <div class="col">
                <div class="cap">{html.escape(tbl.name) if tbl else 'Tabelle'}</div>
                {right_html}
              </div>
            </div>
            """)
        return "".join(rows)

    def _grid_cards(img_paths: List[Path], caption_fn):
        if not img_paths:
            return "<i>Keine Artefakte gefunden.</i>"
        cards = []
        for p in sorted(img_paths):
            cap = caption_fn(p)
            cards.append(f"""
            <div class="card">
              <a href="{_rel(p)}" target="_blank">
                <img src="{_src(p)}" loading="lazy" alt="{html.escape(cap)}">
              </a>
              <div class="cap">{html.escape(cap)}</div>
            </div>
            """)
        return "<div class='grid'>" + "".join(cards) + "</div>"

    def _samples_faceted() -> str:
        if not samples_dir.exists():
            return "<i>Keine Samples gefunden (Ordner fehlt).</i>"
        prob_dirs = [d for d in sorted(samples_dir.iterdir()) if d.is_dir()]
        if not prob_dirs:
            return "<i>Keine Samples gefunden.</i>"

        def _thumbs(run_dir: Path) -> str:
            imgs = [p for p in sorted(run_dir.glob("*.png"))]
            if not imgs:
                return ""

            def _key(p: Path):
                n = p.name.lower()
                # Tfinal zuerst
                return (0 if "_tfinal" in n else 1, n)
            imgs.sort(key=_key)
            return "".join(
                f'<a href="{_rel(img)}" target="_blank"><img src="{_src(img)}" loading="lazy"></a>'
                for img in imgs
            )

        sections = []
        for prob_dir in prob_dirs:
            run_dirs = [d for d in sorted(prob_dir.iterdir()) if d.is_dir()]
            blocks = []
            for run_dir in run_dirs:
                thumbs_html = _thumbs(run_dir)
                if not thumbs_html:
                    continue
                blocks.append(f"""
                <details open>
                  <summary>{html.escape(run_dir.name)}</summary>
                  <div class="thumbs">{thumbs_html}</div>
                </details>
                """)
            if blocks:
                sections.append(f"""
                <section>
                  <h3>Samples – {html.escape(prob_dir.name)}</h3>
                  {''.join(blocks)}
                </section>
                """)
        return "".join(sections) if sections else "<i>Keine Samples gefunden.</i>"

    def _section(title: str, content_html: str) -> str:
        return f"""
        <section>
          <h2>{html.escape(title)}</h2>
          {content_html if content_html.strip() else "<i>Keine Artefakte gefunden.</i>"}
        </section>
        """

    built_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    plots_html = _grid_cards(
        plot_imgs, lambda p: f"{p.parent.name} — {p.name}")
    tables_html = _grid_cards(table_imgs, lambda p: p.name)

    html_s = f"""
<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="utf-8" />
<title>Benchmark Report – {html.escape(results_dir.name)}</title>
<style>
  :root {{
    --card-border: #e0e0e0;
    --cap-color: #333;
    --muted: #666;
  }}
  body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 24px; }}
  h1 {{ margin-bottom: .25rem; }}
  h2 {{ margin: 2rem 0 .75rem; }}
  h3 {{ margin: 1.25rem 0 .5rem; }}
  .small {{ color: var(--muted); font-size: 12px; }}
  section {{ margin-bottom: 24px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 12px; }}
  .card {{ border: 1px solid var(--card-border); border-radius: 10px; overflow: hidden; background:#fff; }}
  .card img {{ width: 100%; height: auto; display: block; }}
  .card .cap {{ font-size: 12px; padding: 8px; color: var(--cap-color); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
  details summary {{ cursor: pointer; font-weight: 600; margin: 6px 0; }}
  .thumbs img {{ width: 320px; height: auto; margin: 4px; border: 1px solid #ddd; border-radius: 8px; }}
  .row {{ display: flex; flex-wrap: wrap; gap: 14px; align-items: flex-start; margin-bottom: 16px; }}
  .col {{ flex: 1 1 420px; min-width: 320px; }}
  .col img {{ width: 100%; height: auto; border: 1px solid #e6e6e6; border-radius: 8px; }}
  .cap {{ font-size: 12px; color: var(--muted); margin: 4px 0 6px; }}
  .muted {{ color: var(--muted); }}
</style>
</head>
<body>
  <h1>Benchmark Report</h1>
  <div class="small">{html.escape(str(results_dir))} — gebaut am {html.escape(built_at)}</div>

  { _section("Boxplots ↔ Tabellen (Side-by-Side)", _pairs_section()) }
  { _section("Alle Plots (rekursiv)", plots_html) }
  { _section("Samples (facettiert: Problem → Runs)", _samples_faceted()) }
  { _section("Alle Tabellen (PNG)", tables_html) }

</body>
</html>
"""
    out_html.write_text(html_s, encoding="utf-8")
    return out_html

# ==== RSM-Auswertung mitlaufen lassen (DROP-IN) ============================
# Einfügen ans Dateiende von evaluate_latest.py


try:
    from evaluator import evaluate_doe_with_rsm
except Exception:
    evaluate_doe_with_rsm = None


def _find_latest_results_dir() -> str | None:
    root = Path("results")
    if not root.exists():
        return None
    cand = [p for p in root.iterdir() if p.is_dir()]
    if not cand:
        return None
    latest = max(cand, key=lambda p: p.stat().st_mtime)
    return str(latest)


if evaluate_doe_with_rsm is not None:
    _dir = locals().get("RESULTS_DIR", None) or locals().get(
        "out_dir", None) or _find_latest_results_dir()
    if _dir:
        print(f"[RSM] Starte RSM-Auswertung für DOE-Methoden in: {_dir}")
        df_rsm = evaluate_doe_with_rsm(_dir, methods=("DOE_LHS", "DOE_CCD", "DOE_FF", "OD", "Random"),
                                       use_observed=True, save_csv=True)
        print(
            f"[RSM] Fertig. Zeilen: {len(df_rsm)} — gespeichert nach tables/rsm_summary.csv")
        try:
            # Kleiner Preview
            print(df_rsm.head(min(5, len(df_rsm))))
        except Exception:
            pass
    else:
        print("[RSM] Kein results/-Ordner gefunden.")
# ==== /RSM DROP-IN =========================================================
# --- RSM2/RSM3 OFFLINE EVALUATION ON DOE ------------------------------------
try:
    from evaluator import evaluate_doe_with_rsm_variants, rsm_variants_boxplots
    print("→ RSM2/RSM3: running offline evaluation on DOE_LHS prefixes…")
    _summary, _progress = evaluate_doe_with_rsm_variants(
        results_dir=str(_dir),
        # falls du weitere DOE-Quellen willst: ("DOE_LHS","DOE_CCD",...)
        methods=("DOE_LHS",),
        variants=("RSM2", "RSM3"),
        use_observed=True,             # nutze y_noisy, wie im Clean-Notebook üblich
        ridge=1e-8,
        save_csv=True
    )
    if not _summary.empty:
        _pngs = rsm_variants_boxplots(str(_dir), variants=("RSM2", "RSM3"))
        print(
            f"✓ RSM variants summary/progress written. Boxplots: {len(_pngs)}")
    else:
        print("! RSM variants produced no rows (check DOE methods / filters).")
except Exception as e:
    print("! RSM2/RSM3 evaluation failed:", e)
# ---------------------------------------------------------------------------


def main():
    proj = Path(".").resolve()
    latest = find_latest_results(proj)

    print(f"[1/3] Evaluating aggregate metrics in: {latest}")
    if REBUILD_AGGREGATES:
        _ = evaluate_folder(latest)
        print("    Aggregate plots written. (Regret/RegretNorm curves, y_best, boxplots, heatmaps, tables)")
    else:
        print("    Skipped aggregate rebuild (REBUILD_AGGREGATES=False).")

    # Load logs if available (continue even if missing)
    df: Optional[pd.DataFrame] = None
    load_err: Optional[Exception] = None
    try:
        df = _load_all_runs(latest)
    except Exception as e:
        load_err = e
        print(f"[2/3] Transparency plots (samples): {e}")
        print("      Proceeding without samples (no logs available).")

    # Export Best-at-T CSVs (and try to render table PNGs if helper exists)
    if df is not None:
        try:
            csv_true = export_best_at_T_table(
                df, latest, kind="true", decimals=BESTAT_T_DECIMALS)
            csv_obs = export_best_at_T_table(
                df, latest, kind="obs",  decimals=BESTAT_T_DECIMALS)
            if "regret_norm" in df.columns:
                csv_norm = export_best_at_T_table(
                    df, latest, kind="norm", decimals=BESTAT_T_DECIMALS)
                print("    Best-at-T tables:", csv_true, csv_obs, csv_norm)
            else:
                print("    Best-at-T tables:", csv_true, csv_obs)
        except Exception as e:
            print("    Best-at-T export skipped:", e)

    # Try to convert best-at-T CSVs to PNGs (if your evaluator provides it)
    try:
        from evaluator import save_best_at_T_tables_as_png
        save_best_at_T_tables_as_png(latest, decimals=BESTAT_T_DECIMALS)
    except Exception:
        pass

    # Create per-plot side tables (one PNG per boxplot facet)
    try:
        from evaluator import render_boxplot_side_tables
        render_boxplot_side_tables(latest, decimals=BESTAT_T_DECIMALS)
    except Exception as e:
        print("[tables] side-table generation skipped:", e)

    # Determine problems/methods for samples
    prob_list: List[str] = []
    meth_list: List[str] = []
    if df is not None:
        if problems_for_samples == "auto":
            prob_list = detect_2d_problems_from_logs(df)
            if not prob_list:
                try:
                    prob_list = sorted([p for p in df["problem"].astype(str).unique().tolist()
                                        if get_problem(p).dim == 2])
                except Exception:
                    prob_list = sorted(
                        df["problem"].astype(str).unique().tolist())
        else:
            prob_list = list(problems_for_samples)

        if methods_for_samples == "auto":
            meth_list = sorted(df["method"].astype(str).unique().tolist())
        else:
            meth_list = list(methods_for_samples)

    # Optional sample generation
    count = 0
    if generate_samples and df is not None and prob_list and meth_list:
        # noise facets (if present)
        noise_vals = [None]
        if "noise_pct" in df.columns:
            try:
                noise_vals = sorted(df["noise_pct"].dropna().astype(
                    float).unique().tolist()) or [None]
            except Exception:
                pass

        k_eff = (10**9) if final_samples_at_T else max(1, int(k_every))

        print("[2/3] Transparency plots (samples):")
        print(f"    Problems: {prob_list}")
        print(f"    Methods : {meth_list}")
        print(f"    Seeds   : {seeds_for_samples}")
        print(f"    Every k : {'T only' if final_samples_at_T else k_every}")

        for p in prob_list:
            for m in meth_list:
                mask_pm = (df["problem"].astype(str) == str(p)) & (
                    df["method"].astype(str) == str(m))
                for n_pct in noise_vals:
                    if seeds_for_samples == "auto":
                        mask = mask_pm
                        if n_pct is not None and "noise_pct" in df.columns:
                            mask = mask & (df["noise_pct"].astype(
                                float) == float(n_pct))
                        seed_list = sorted(
                            df.loc[mask, "seed"].astype(int).unique().tolist())
                    else:
                        seed_list = [int(s) for s in seeds_for_samples]

                    for s in seed_list:
                        try:
                            kwargs = dict(
                                df=df, problem_name=p, method=m, seed=int(s),
                                k=int(k_eff), outdir=latest
                            )
                            if n_pct is not None:
                                kwargs["noise_pct"] = float(n_pct)
                            paths = plot_samples_every_k(**kwargs)
                            count += len(paths)
                            n_str = "none" if n_pct is None else f"{int(round(100*float(n_pct)))}%"
                            print(
                                f"      {p} / {m} / seed={s} / noise={n_str}: {len(paths)} figures")
                        except Exception as e:
                            n_str = "none" if n_pct is None else f"{int(round(100*float(n_pct)))}%"
                            print(
                                f"      Skip {p} / {m} / seed={s} / noise={n_str}: {e}")
    else:
        print("[2/3] Skipping sample generation "
              f"(generate_samples={generate_samples}, logs={'present' if df is not None else 'absent'}).")

    #     # --- Gruppierte Boxplots (Methode × Noise) erzeugen ---
    # try:
    #     from evaluator import render_grouped_boxplots_by_method_noise
    #     render_grouped_boxplots_by_method_noise(
    #         latest, decimals=BESTAT_T_DECIMALS)
    # except Exception as e:
    #     print("[grouped] skipped:", e)

    # --- Pro Methode: Boxplot (Noise gruppiert) + Tabelle ---
    try:
        from evaluator import render_bestvalue_boxplots_per_method_grouped_by_noise
        render_bestvalue_boxplots_per_method_grouped_by_noise(
            latest,
            decimals=BESTAT_T_DECIMALS,
            metric="best_so_far",         # oder "best_so_far_obs"
            show_all_seeds=True,
            highlight_seed=None,          # optional z.B. 0
        )
    except Exception as e:
        print("[per-method] skipped:", e)

    import glob
    import os
    pp = os.path.join(latest, "plots", "bestvalue_box_method_*_Nall.png")
    hits = glob.glob(pp)
    print(
        f"[per-method] found {len(hits)} per-method Nall plots (expected >= 1). Example:", hits[:3])

    # 3) Always (re)build the report at the end
    try:
        # --- Deduped Report mit sauberem Pairing bauen ---
        report_path = os.path.join(latest, "benchmark_report.html")
        all_imgs = _collect_images(latest)
        pairs, used = _pair_plots_with_tables(all_imgs)

        big_tables = [p for p in all_imgs if _classify(
            p)[2] and p not in used]  # best_at_T_by_seed_method_*.png
        other_plots = [p for p in all_imgs
                       if p not in used
                       and not _classify(p)[1]  # keine *_table.png
                       # keine großen Best-at-T Tabellen
                       and not _classify(p)[2]
                       ]

        _write_html_gallery(report_path, pairs, big_tables,
                            other_plots, latest)
        print(f"Report: {report_path}")

    except Exception as e:
        print("[warn] building benchmark_report.html failed:", e)

    # Final line
    if generate_samples:
        print(f"[3/3] Done. Generated {count} sample figures.")
        print("     (Look for files: samples2d_*.png and samples3d_*.png)")
    else:
        print("[3/3] Done.")


if __name__ == "__main__":
    main()
