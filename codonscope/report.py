"""CodonScope HTML Report Generator.

Generates a single self-contained HTML file with inline CSS and
base64-embedded matplotlib plots. Runs all applicable analysis modes
and presents results in a clean, readable format.
"""

import base64
import io
import logging
import time
from datetime import datetime
from html import escape
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from codonscope import __version__

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════

def generate_report(
    species: str,
    gene_ids: list[str],
    output: str | Path = "report.html",
    species2: str | None = None,
    tissue: str | None = None,
    n_bootstrap: int = 10_000,
    seed: int | None = None,
    data_dir: str | Path | None = None,
) -> Path:
    """Generate a comprehensive HTML report for a gene list.

    Runs all applicable analysis modes and produces a self-contained
    HTML file with embedded plots.

    Args:
        species: Primary species (e.g. "yeast", "human").
        gene_ids: List of gene identifiers.
        output: Output HTML file path.
        species2: Second species for cross-species comparison (Mode 6).
        tissue: GTEx tissue for human demand analysis (Mode 2).
        n_bootstrap: Bootstrap iterations for all modes.
        seed: Random seed for reproducibility.
        data_dir: Override default data directory.

    Returns:
        Path to the generated HTML file.
    """
    output = Path(output)
    t0 = time.time()
    sections: list[str] = []

    # ── Gene list resolution ──────────────────────────────────────────────
    from codonscope.core.sequences import SequenceDB
    db = SequenceDB(species, data_dir=data_dir)
    id_mapping = db.resolve_ids(gene_ids)
    gene_meta = db.get_gene_metadata()

    sections.append(_section_gene_summary(
        species, gene_ids, id_mapping, gene_meta, db,
    ))

    # ── Mode 1: Monocodon composition ────────────────────────────────────
    try:
        logger.info("Running Mode 1: monocodon composition...")
        from codonscope.modes.mode1_composition import run_composition
        mono_result = run_composition(
            species=species, gene_ids=gene_ids, k=1,
            n_bootstrap=n_bootstrap, seed=seed, data_dir=data_dir,
        )
        sections.append(_section_mode1(mono_result, k=1))
    except Exception as exc:
        logger.warning("Mode 1 (monocodon) failed: %s", exc)
        sections.append(_section_error("Mode 1: Monocodon Composition", exc))

    # ── Mode 1: Dicodon composition ──────────────────────────────────────
    try:
        logger.info("Running Mode 1: dicodon composition...")
        di_result = run_composition(
            species=species, gene_ids=gene_ids, k=2,
            n_bootstrap=n_bootstrap, seed=seed, data_dir=data_dir,
        )
        sections.append(_section_mode1(di_result, k=2))
    except Exception as exc:
        logger.warning("Mode 1 (dicodon) failed: %s", exc)
        sections.append(_section_error("Mode 1: Dicodon Composition", exc))

    # ── Mode 5: Disentanglement ──────────────────────────────────────────
    try:
        logger.info("Running Mode 5: AA vs codon disentanglement...")
        from codonscope.modes.mode5_disentangle import run_disentangle
        dis_result = run_disentangle(
            species=species, gene_ids=gene_ids,
            n_bootstrap=n_bootstrap, seed=seed, data_dir=data_dir,
        )
        sections.append(_section_mode5(dis_result))
    except Exception as exc:
        logger.warning("Mode 5 failed: %s", exc)
        sections.append(_section_error("Mode 5: Disentanglement", exc))

    # ── Mode 3: Optimality Profile ───────────────────────────────────────
    try:
        logger.info("Running Mode 3: optimality profile...")
        from codonscope.modes.mode3_profile import run_profile
        prof_result = run_profile(
            species=species, gene_ids=gene_ids, data_dir=data_dir,
        )
        sections.append(_section_mode3(prof_result))
    except Exception as exc:
        logger.warning("Mode 3 failed: %s", exc)
        sections.append(_section_error("Mode 3: Optimality Profile", exc))

    # ── Mode 4: Collision Potential ──────────────────────────────────────
    try:
        logger.info("Running Mode 4: collision potential...")
        from codonscope.modes.mode4_collision import run_collision
        col_result = run_collision(
            species=species, gene_ids=gene_ids, data_dir=data_dir,
        )
        sections.append(_section_mode4(col_result))
    except Exception as exc:
        logger.warning("Mode 4 failed: %s", exc)
        sections.append(_section_error("Mode 4: Collision Potential", exc))

    # ── Mode 2: Translational Demand ─────────────────────────────────────
    try:
        logger.info("Running Mode 2: translational demand...")
        from codonscope.modes.mode2_demand import run_demand
        dem_result = run_demand(
            species=species, gene_ids=gene_ids,
            tissue=tissue,
            n_bootstrap=n_bootstrap, seed=seed, data_dir=data_dir,
        )
        sections.append(_section_mode2(dem_result))
    except Exception as exc:
        logger.warning("Mode 2 failed: %s", exc)
        sections.append(_section_error("Mode 2: Translational Demand", exc))

    # ── Mode 6: Cross-Species Comparison ─────────────────────────────────
    if species2:
        try:
            logger.info("Running Mode 6: cross-species comparison...")
            from codonscope.modes.mode6_compare import run_compare
            cmp_result = run_compare(
                species1=species, species2=species2,
                gene_ids=gene_ids, from_species=species,
                n_bootstrap=n_bootstrap, seed=seed, data_dir=data_dir,
            )
            sections.append(_section_mode6(cmp_result))
        except Exception as exc:
            logger.warning("Mode 6 failed: %s", exc)
            sections.append(_section_error("Mode 6: Cross-Species Comparison", exc))

    elapsed = time.time() - t0
    html = _build_html(species, gene_ids, sections, elapsed)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    logger.info("Report written to %s (%.1fs)", output, elapsed)
    return output


# ═══════════════════════════════════════════════════════════════════════════════
# HTML building
# ═══════════════════════════════════════════════════════════════════════════════

CSS = """
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    max-width: 1100px;
    margin: 0 auto;
    padding: 20px 30px;
    color: #1a1a2e;
    background: #f8f9fa;
    line-height: 1.5;
}
h1 {
    color: #16213e;
    border-bottom: 3px solid #0f3460;
    padding-bottom: 10px;
    margin-top: 0;
}
h2 {
    color: #0f3460;
    border-bottom: 2px solid #e2e8f0;
    padding-bottom: 6px;
    margin-top: 40px;
}
h3 {
    color: #16213e;
    margin-top: 25px;
}
.summary-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 12px;
    margin: 15px 0;
}
.summary-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 14px 18px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
.summary-card .label {
    font-size: 0.82em;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.summary-card .value {
    font-size: 1.5em;
    font-weight: 700;
    color: #0f3460;
    margin-top: 2px;
}
table {
    border-collapse: collapse;
    width: 100%;
    margin: 10px 0;
    font-size: 0.88em;
    background: white;
}
th {
    background: #0f3460;
    color: white;
    padding: 8px 10px;
    text-align: left;
    font-weight: 600;
}
td {
    padding: 6px 10px;
    border-bottom: 1px solid #e2e8f0;
}
tr:nth-child(even) td {
    background: #f8fafc;
}
tr:hover td {
    background: #eef2ff;
}
.sig-pos { color: #b91c1c; font-weight: 600; }
.sig-neg { color: #1d4ed8; font-weight: 600; }
.warn { color: #d97706; }
.ok { color: #16a34a; }
.plot-container {
    text-align: center;
    margin: 20px 0;
}
.plot-container img {
    max-width: 100%;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.08);
}
.section {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 20px 25px;
    margin: 20px 0;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.section h2 {
    margin-top: 0;
    border-bottom: 2px solid #e2e8f0;
}
.error-box {
    background: #fef2f2;
    border: 1px solid #fecaca;
    border-radius: 6px;
    padding: 12px 16px;
    color: #991b1b;
    font-size: 0.9em;
}
.footer {
    text-align: center;
    color: #94a3b8;
    font-size: 0.82em;
    margin-top: 40px;
    padding-top: 15px;
    border-top: 1px solid #e2e8f0;
}
.badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.82em;
    font-weight: 600;
}
.badge-enriched { background: #fee2e2; color: #991b1b; }
.badge-depleted { background: #dbeafe; color: #1e40af; }
.badge-aa { background: #fef3c7; color: #92400e; }
.badge-syn { background: #d1fae5; color: #065f46; }
.badge-both { background: #ede9fe; color: #5b21b6; }
"""


def _build_html(
    species: str,
    gene_ids: list[str],
    sections: list[str],
    elapsed: float,
) -> str:
    """Assemble the full HTML document."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    body = "\n".join(sections)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CodonScope Report &mdash; {escape(species)} ({len(gene_ids)} genes)</title>
<style>{CSS}</style>
</head>
<body>
<h1>CodonScope Report</h1>
<p><strong>Species:</strong> {escape(species)} &nbsp;|&nbsp;
   <strong>Input genes:</strong> {len(gene_ids)} &nbsp;|&nbsp;
   <strong>Generated:</strong> {now}</p>
{body}
<div class="footer">
  CodonScope v{__version__} &mdash; generated in {elapsed:.1f}s &mdash; {now}
</div>
</body>
</html>
"""


# ═══════════════════════════════════════════════════════════════════════════════
# Section builders
# ═══════════════════════════════════════════════════════════════════════════════

def _section_gene_summary(
    species: str,
    gene_ids: list[str],
    id_mapping,
    gene_meta: pd.DataFrame,
    db,
) -> str:
    """Gene list summary with diagnostics."""
    n_input = len(gene_ids)
    n_mapped = id_mapping.n_mapped
    n_unmapped = id_mapping.n_unmapped

    # Get mapped gene metadata
    mapped_sys = list(id_mapping.values())
    meta_sub = gene_meta[gene_meta["systematic_name"].isin(mapped_sys)]
    mean_len = meta_sub["cds_length"].mean() if len(meta_sub) > 0 else 0
    mean_gc = meta_sub["gc_content"].mean() if len(meta_sub) > 0 else 0

    # Genome stats for comparison
    genome_mean_len = gene_meta["cds_length"].mean()
    genome_mean_gc = gene_meta["gc_content"].mean()

    # Unmapped list
    unmapped_html = ""
    if n_unmapped > 0:
        unmapped_list = id_mapping.unmapped[:20]
        unmapped_str = ", ".join(escape(u) for u in unmapped_list)
        if n_unmapped > 20:
            unmapped_str += f" ... and {n_unmapped - 20} more"
        unmapped_html = f"""
        <p class="warn">Unmapped IDs ({n_unmapped}): {unmapped_str}</p>
        """

    return f"""
<div class="section">
<h2>Gene List Summary</h2>
<div class="summary-grid">
  <div class="summary-card">
    <div class="label">Input IDs</div>
    <div class="value">{n_input}</div>
  </div>
  <div class="summary-card">
    <div class="label">Mapped</div>
    <div class="value {'ok' if n_mapped >= 10 else 'warn'}">{n_mapped}</div>
  </div>
  <div class="summary-card">
    <div class="label">Unmapped</div>
    <div class="value {'warn' if n_unmapped > 0 else 'ok'}">{n_unmapped}</div>
  </div>
  <div class="summary-card">
    <div class="label">Mean CDS length</div>
    <div class="value">{mean_len:.0f} nt</div>
  </div>
  <div class="summary-card">
    <div class="label">Mean GC content</div>
    <div class="value">{mean_gc:.1%}</div>
  </div>
  <div class="summary-card">
    <div class="label">Genome mean length</div>
    <div class="value">{genome_mean_len:.0f} nt</div>
  </div>
</div>
{unmapped_html}
</div>
"""


def _section_mode1(result: dict, k: int) -> str:
    """Mode 1: Composition analysis section."""
    kname = {1: "Monocodon", 2: "Dicodon", 3: "Tricodon"}[k]
    df = result["results"]
    n_genes = result["n_genes"]
    diag = result["diagnostics"]

    sig = df[df["adjusted_p"] < 0.05]
    n_sig = len(sig)
    enriched = sig[sig["z_score"] > 0].sort_values("z_score", ascending=False)
    depleted = sig[sig["z_score"] < 0].sort_values("z_score", ascending=True)

    # Diagnostics
    diag_html = ""
    warns = []
    if diag.get("length_warning"):
        warns.append(f"CDS length differs from background (KS p={diag['length_p']:.2e})")
    if diag.get("gc_warning"):
        warns.append(f"GC content differs from background (KS p={diag['gc_p']:.2e})")
    for w in diag.get("power_warnings", []):
        warns.append(w)
    if warns:
        items = "".join(f"<li>{escape(w)}</li>" for w in warns)
        diag_html = f'<div class="error-box"><strong>Diagnostics:</strong><ul style="margin:4px 0">{items}</ul></div>'

    # Volcano plot
    plot_b64 = _plot_volcano(df, f"Mode 1: {kname} Composition (n={n_genes} genes)")

    # Top tables
    top_enriched = _results_table(enriched.head(15), k)
    top_depleted = _results_table(depleted.head(15), k)

    return f"""
<div class="section">
<h2>Mode 1: {kname} Composition</h2>
<p>Genes analyzed: <strong>{n_genes}</strong> &nbsp;|&nbsp;
   Significant {kname.lower()}s (adj_p &lt; 0.05): <strong>{n_sig}</strong></p>
{diag_html}
<div class="plot-container">{_img_tag(plot_b64)}</div>
<h3>Top Enriched ({len(enriched)} total)</h3>
{top_enriched}
<h3>Top Depleted ({len(depleted)} total)</h3>
{top_depleted}
</div>
"""


def _section_mode5(result: dict) -> str:
    """Mode 5: Disentanglement section."""
    summary = result["summary"]
    attr_df = result["attribution"]
    rscu_df = result["rscu_results"]
    aa_df = result["aa_results"]
    n_genes = result["n_genes"]

    # Summary text
    s = summary
    summary_text = escape(s["summary_text"])

    # Attribution pie chart
    pie_b64 = _plot_attribution_pie(s)

    # Attribution table (significant codons only)
    sig_attr = attr_df[attr_df["attribution"] != "None"].sort_values(
        "rscu_z_score", key=abs, ascending=False
    ).head(20)

    rows = ""
    for _, row in sig_attr.iterrows():
        badge_cls = {
            "AA-driven": "badge-aa",
            "Synonymous-driven": "badge-syn",
            "Both": "badge-both",
        }.get(row["attribution"], "")
        z_cls = "sig-pos" if row["rscu_z_score"] > 0 else "sig-neg"
        rows += f"""<tr>
          <td><strong>{escape(str(row['codon']))}</strong></td>
          <td>{escape(str(row['amino_acid']))}</td>
          <td class="{z_cls}">{row.get('aa_z_score', 0):+.2f}</td>
          <td class="{z_cls}">{row['rscu_z_score']:+.2f}</td>
          <td>{row.get('aa_adj_p', 1):.2e}</td>
          <td>{row.get('rscu_adj_p', 1):.2e}</td>
          <td><span class="badge {badge_cls}">{escape(str(row['attribution']))}</span></td>
        </tr>"""

    # Synonymous drivers
    syn_df = result.get("synonymous_drivers", pd.DataFrame())
    driver_html = ""
    if len(syn_df) > 0:
        sig_syn = syn_df[syn_df["driver"] != "not_applicable"].head(15)
        if len(sig_syn) > 0:
            driver_rows = ""
            for _, row in sig_syn.iterrows():
                driver_rows += f"""<tr>
                  <td>{escape(str(row['codon']))}</td>
                  <td>{escape(str(row['amino_acid']))}</td>
                  <td>{row['rscu_z_score']:+.2f}</td>
                  <td>{escape(str(row['driver']))}</td>
                </tr>"""
            driver_html = f"""
            <h3>Synonymous Drivers</h3>
            <table>
            <tr><th>Codon</th><th>AA</th><th>RSCU Z</th><th>Driver</th></tr>
            {driver_rows}
            </table>"""

    return f"""
<div class="section">
<h2>Mode 5: AA vs Codon Disentanglement</h2>
<p>Genes analyzed: <strong>{n_genes}</strong></p>
<p>{summary_text}</p>
<div class="plot-container">{_img_tag(pie_b64)}</div>
<h3>Attribution Table</h3>
<table>
<tr><th>Codon</th><th>AA</th><th>AA Z</th><th>RSCU Z</th><th>AA adj_p</th><th>RSCU adj_p</th><th>Attribution</th></tr>
{rows}
</table>
{driver_html}
</div>
"""


def _section_mode3(result: dict) -> str:
    """Mode 3: Optimality profile section."""
    n_genes = result["n_genes"]
    metagene_gs = result["metagene_geneset"]
    metagene_bg = result["metagene_genome"]
    ramp = result["ramp_analysis"]
    scores = result["per_gene_scores"]

    # Metagene plot + ramp bar chart
    plot_b64 = _plot_metagene(metagene_gs, metagene_bg, ramp, n_genes)

    col = "wtai" if "wtai" in scores.columns else "tai"
    mean_score = scores[col].mean()
    std_score = scores[col].std()

    return f"""
<div class="section">
<h2>Mode 3: Optimality Profile</h2>
<p>Genes analyzed: <strong>{n_genes}</strong> &nbsp;|&nbsp;
   Mean {col.upper()}: <strong>{mean_score:.4f}</strong> (std {std_score:.4f})</p>
<div class="summary-grid">
  <div class="summary-card">
    <div class="label">Gene set ramp</div>
    <div class="value">{ramp['geneset_ramp_mean']:.3f}</div>
  </div>
  <div class="summary-card">
    <div class="label">Gene set body</div>
    <div class="value">{ramp['geneset_body_mean']:.3f}</div>
  </div>
  <div class="summary-card">
    <div class="label">Ramp delta</div>
    <div class="value">{ramp['geneset_ramp_delta']:+.3f}</div>
  </div>
  <div class="summary-card">
    <div class="label">Genome ramp delta</div>
    <div class="value">{ramp['genome_ramp_delta']:+.3f}</div>
  </div>
</div>
<div class="plot-container">{_img_tag(plot_b64)}</div>
</div>
"""


def _section_mode4(result: dict) -> str:
    """Mode 4: Collision potential section."""
    n_genes = result["n_genes"]
    gs = result["transition_matrix_geneset"]
    bg = result["transition_matrix_genome"]
    fs_enrich = result["fs_enrichment"]
    chi2 = result["chi2_stat"]
    chi2_p = result["chi2_p"]

    # Transition bar plot
    plot_b64 = _plot_transitions(gs, bg, fs_enrich, chi2_p, n_genes)

    return f"""
<div class="section">
<h2>Mode 4: Collision Potential</h2>
<p>Genes analyzed: <strong>{n_genes}</strong> &nbsp;|&nbsp;
   Fast/slow threshold: {result['threshold']:.4f}</p>
<div class="summary-grid">
  <div class="summary-card">
    <div class="label">FS Enrichment</div>
    <div class="value">{fs_enrich:.3f}</div>
  </div>
  <div class="summary-card">
    <div class="label">Chi-squared</div>
    <div class="value">{chi2:.1f}</div>
  </div>
  <div class="summary-card">
    <div class="label">Chi-sq p-value</div>
    <div class="value">{chi2_p:.2e}</div>
  </div>
  <div class="summary-card">
    <div class="label">FS/SF ratio (gene set)</div>
    <div class="value">{result['fs_sf_ratio_geneset']:.3f}</div>
  </div>
</div>
<div class="plot-container">{_img_tag(plot_b64)}</div>
<table>
<tr><th>Transition</th><th>Gene Set</th><th>Genome</th><th>Fold Change</th></tr>
{"".join(f'<tr><td><strong>{t}</strong></td><td>{gs[t]:.4f}</td><td>{bg[t]:.4f}</td><td>{gs[t]/bg[t]:.3f}</td></tr>' if bg[t] > 0 else f'<tr><td><strong>{t}</strong></td><td>{gs[t]:.4f}</td><td>{bg[t]:.4f}</td><td>-</td></tr>' for t in ('FF','FS','SF','SS'))}
</table>
</div>
"""


def _section_mode2(result: dict) -> str:
    """Mode 2: Translational demand section."""
    df = result["results"]
    top = result["top_genes"]
    n_genes = result["n_genes"]
    tissue = result["tissue"]

    sig = df[df["adjusted_p"] < 0.05]
    n_sig = len(sig)
    enriched = sig[sig["z_score"] > 0].sort_values("z_score", ascending=False)
    depleted = sig[sig["z_score"] < 0].sort_values("z_score", ascending=True)

    # Demand bar plot
    plot_b64 = _plot_demand_bars(enriched.head(15), depleted.head(15), n_genes, tissue)

    # Top genes table
    top_rows = ""
    for _, row in top.head(10).iterrows():
        top_rows += f"""<tr>
          <td><strong>{escape(str(row['gene']))}</strong></td>
          <td>{row['tpm']:.1f}</td>
          <td>{row['n_codons']}</td>
          <td>{row['demand_fraction']*100:.1f}%</td>
        </tr>"""

    return f"""
<div class="section">
<h2>Mode 2: Translational Demand</h2>
<p>Genes analyzed: <strong>{n_genes}</strong> &nbsp;|&nbsp;
   Expression: <strong>{escape(str(tissue))}</strong> &nbsp;|&nbsp;
   Significant codons: <strong>{n_sig}</strong></p>
<h3>Top Demand-Contributing Genes</h3>
<table>
<tr><th>Gene</th><th>TPM</th><th>Codons</th><th>Demand %</th></tr>
{top_rows}
</table>
<div class="plot-container">{_img_tag(plot_b64)}</div>
<h3>Top Demand-Enriched Codons</h3>
{_demand_table(enriched.head(15))}
<h3>Top Demand-Depleted Codons</h3>
{_demand_table(depleted.head(15))}
</div>
"""


def _section_mode6(result: dict) -> str:
    """Mode 6: Cross-species comparison section."""
    summary = result["summary"]
    pg = result["per_gene"]
    n_ortho = result["n_orthologs"]
    n_genome = result["n_genome_orthologs"]
    from_sp = result["from_species"]
    to_sp = result["to_species"]

    # Correlation histogram
    genome_corr = result["genome_correlations"]
    plot_b64 = _plot_correlation_hist(pg, genome_corr, summary, from_sp, to_sp)

    from_col = f"{from_sp}_gene"
    to_col = f"{to_sp}_gene"

    # Top conserved
    top_cons = pg.nlargest(10, "rscu_correlation")
    cons_rows = ""
    for _, row in top_cons.iterrows():
        cons_rows += f"""<tr>
          <td>{escape(str(row[from_col]))}</td>
          <td>{escape(str(row[to_col]))}</td>
          <td>{row['rscu_correlation']:.4f}</td>
        </tr>"""

    # Top divergent
    top_div = pg.nsmallest(10, "rscu_correlation")
    div_rows = ""
    for _, row in top_div.iterrows():
        div_rows += f"""<tr>
          <td>{escape(str(row[from_col]))}</td>
          <td>{escape(str(row[to_col]))}</td>
          <td>{row['rscu_correlation']:.4f}</td>
        </tr>"""

    return f"""
<div class="section">
<h2>Mode 6: Cross-Species Comparison</h2>
<p>Ortholog pairs: <strong>{n_ortho}</strong> (gene set)  /
   <strong>{n_genome}</strong> (genome)</p>
<div class="summary-grid">
  <div class="summary-card">
    <div class="label">Gene set mean r</div>
    <div class="value">{summary['geneset_mean_r']:.3f}</div>
  </div>
  <div class="summary-card">
    <div class="label">Genome mean r</div>
    <div class="value">{summary['genome_mean_r']:.3f}</div>
  </div>
  <div class="summary-card">
    <div class="label">Z-score</div>
    <div class="value">{summary['z_score']:+.2f}</div>
  </div>
  <div class="summary-card">
    <div class="label">P-value</div>
    <div class="value">{summary['p_value']:.2e}</div>
  </div>
</div>
<div class="plot-container">{_img_tag(plot_b64)}</div>
<h3>Most Conserved Orthologs</h3>
<table>
<tr><th>{escape(from_sp)} gene</th><th>{escape(to_sp)} gene</th><th>RSCU r</th></tr>
{cons_rows}
</table>
<h3>Most Divergent Orthologs</h3>
<table>
<tr><th>{escape(from_sp)} gene</th><th>{escape(to_sp)} gene</th><th>RSCU r</th></tr>
{div_rows}
</table>
</div>
"""


def _section_error(title: str, exc: Exception) -> str:
    """Error section when a mode fails."""
    return f"""
<div class="section">
<h2>{escape(title)}</h2>
<div class="error-box">Analysis failed: {escape(str(exc))}</div>
</div>
"""


# ═══════════════════════════════════════════════════════════════════════════════
# Helper tables
# ═══════════════════════════════════════════════════════════════════════════════

def _results_table(df: pd.DataFrame, k: int) -> str:
    """Build an HTML table for Mode 1 results."""
    if len(df) == 0:
        return "<p><em>No significant results.</em></p>"

    kname = "kmer" if "kmer" in df.columns else "codon"
    rows = ""
    for _, row in df.iterrows():
        z = row["z_score"]
        z_cls = "sig-pos" if z > 0 else "sig-neg"
        rows += f"""<tr>
          <td><strong>{escape(str(row[kname]))}</strong></td>
          <td class="{z_cls}">{z:+.2f}</td>
          <td>{row['observed_freq']:.4f}</td>
          <td>{row['expected_freq']:.4f}</td>
          <td>{row['adjusted_p']:.2e}</td>
          <td>{row.get('cohens_d', 0):.2f}</td>
        </tr>"""

    return f"""<table>
<tr><th>K-mer</th><th>Z-score</th><th>Observed</th><th>Expected</th><th>Adj. p</th><th>Cohen's d</th></tr>
{rows}
</table>"""


def _demand_table(df: pd.DataFrame) -> str:
    """Build an HTML table for Mode 2 demand results."""
    if len(df) == 0:
        return "<p><em>No significant results.</em></p>"

    rows = ""
    for _, row in df.iterrows():
        z = row["z_score"]
        z_cls = "sig-pos" if z > 0 else "sig-neg"
        rows += f"""<tr>
          <td><strong>{escape(str(row['kmer']))}</strong></td>
          <td class="{z_cls}">{z:+.2f}</td>
          <td>{row['demand_geneset']:.4f}</td>
          <td>{row['demand_genome']:.4f}</td>
          <td>{row['adjusted_p']:.2e}</td>
        </tr>"""

    return f"""<table>
<tr><th>K-mer</th><th>Z-score</th><th>Gene Set Demand</th><th>Genome Demand</th><th>Adj. p</th></tr>
{rows}
</table>"""


# ═══════════════════════════════════════════════════════════════════════════════
# Plot generators (return base64 PNG strings)
# ═══════════════════════════════════════════════════════════════════════════════

def _fig_to_b64(fig) -> str:
    """Convert matplotlib figure to base64 PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _img_tag(b64: str) -> str:
    """Wrap base64 string in an <img> tag."""
    return f'<img src="data:image/png;base64,{b64}" alt="plot">'


def _plot_volcano(df: pd.DataFrame, title: str) -> str:
    """Volcano plot: Z-score vs -log10(adjusted_p)."""
    fig, ax = plt.subplots(figsize=(10, 5))

    z = df["z_score"].values
    p = df["adjusted_p"].values
    neglog_p = -np.log10(np.clip(p, 1e-300, 1.0))

    sig_mask = p < 0.05
    ax.scatter(z[~sig_mask], neglog_p[~sig_mask],
               c="#94a3b8", s=20, alpha=0.5, label="Not significant")
    if sig_mask.any():
        colors = np.where(z[sig_mask] > 0, "#dc2626", "#2563eb")
        ax.scatter(z[sig_mask], neglog_p[sig_mask],
                   c=colors, s=30, alpha=0.7, label="Significant (adj_p < 0.05)")

    ax.axhline(-np.log10(0.05), color="gray", ls="--", lw=0.8, alpha=0.5)
    ax.axvline(0, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax.set_xlabel("Z-score")
    ax.set_ylabel("-log10(adjusted p-value)")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    return _fig_to_b64(fig)


def _plot_attribution_pie(summary: dict) -> str:
    """Pie chart of attribution categories."""
    fig, ax = plt.subplots(figsize=(5, 4))
    labels = []
    sizes = []
    colors_map = {
        "AA-driven": "#f59e0b",
        "Synonymous-driven": "#10b981",
        "Both": "#8b5cf6",
    }
    for cat in ("AA-driven", "Synonymous-driven", "Both"):
        key = f"n_{cat.lower().replace('-', '_')}"
        n = summary.get(key, 0)
        if n > 0:
            labels.append(f"{cat} ({n})")
            sizes.append(n)

    if not sizes:
        ax.text(0.5, 0.5, "No significant attributions",
                ha="center", va="center", transform=ax.transAxes)
    else:
        cols = [colors_map.get(l.split(" (")[0], "#94a3b8") for l in labels]
        ax.pie(sizes, labels=labels, colors=cols, autopct="%1.0f%%",
               startangle=90, textprops={"fontsize": 9})
        ax.set_title("Codon Bias Attribution")

    fig.tight_layout()
    return _fig_to_b64(fig)


def _plot_metagene(
    metagene_gs: np.ndarray,
    metagene_bg: np.ndarray,
    ramp: dict,
    n_genes: int,
) -> str:
    """Metagene profile + ramp bar chart."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Panel 1: Metagene profile
    x = np.arange(len(metagene_gs))
    ax1.plot(x, metagene_gs, color="#2563eb", lw=1.5, label="Gene set")
    ax1.plot(x, metagene_bg, color="#94a3b8", lw=1.5, label="Genome")
    ax1.set_xlabel("Normalised position (0=5', 99=3')")
    ax1.set_ylabel("Mean optimality (wtAI)")
    ax1.set_title(f"Metagene Optimality Profile (n={n_genes})")
    ax1.legend(fontsize=8)

    # Panel 2: Ramp analysis
    labels = ["Gene set\nramp", "Gene set\nbody", "Genome\nramp", "Genome\nbody"]
    values = [
        ramp["geneset_ramp_mean"], ramp["geneset_body_mean"],
        ramp["genome_ramp_mean"], ramp["genome_body_mean"],
    ]
    colors = ["#93c5fd", "#2563eb", "#d1d5db", "#6b7280"]
    ax2.bar(labels, values, color=colors, edgecolor="white", width=0.6)
    ax2.set_ylabel("Mean optimality (wtAI)")
    ax2.set_title("Ramp vs Body")

    fig.tight_layout()
    return _fig_to_b64(fig)


def _plot_transitions(
    gs: dict, bg: dict,
    fs_enrich: float, chi2_p: float,
    n_genes: int,
) -> str:
    """Transition proportion bar chart."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    types = ["FF", "FS", "SF", "SS"]
    x = np.arange(len(types))
    w = 0.35

    gs_vals = [gs[t] for t in types]
    bg_vals = [bg[t] for t in types]

    ax.bar(x - w/2, gs_vals, w, label="Gene set", color="#2563eb", edgecolor="white")
    ax.bar(x + w/2, bg_vals, w, label="Genome", color="#94a3b8", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(types)
    ax.set_ylabel("Proportion")
    ax.set_title(f"Codon Transition Proportions (n={n_genes})\n"
                 f"FS enrichment={fs_enrich:.3f}, chi2 p={chi2_p:.2e}")
    ax.legend(fontsize=8)

    fig.tight_layout()
    return _fig_to_b64(fig)


def _plot_demand_bars(
    enriched: pd.DataFrame,
    depleted: pd.DataFrame,
    n_genes: int,
    tissue: str,
) -> str:
    """Horizontal bar chart of top demand-enriched/depleted codons."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    if len(enriched) > 0:
        y = np.arange(len(enriched))
        ax1.barh(y, enriched["z_score"].values, color="#dc2626", edgecolor="white")
        ax1.set_yticks(y)
        ax1.set_yticklabels(enriched["kmer"].values, fontsize=8)
        ax1.invert_yaxis()
        ax1.set_xlabel("Z-score")
        ax1.set_title("Demand-Enriched")
    else:
        ax1.text(0.5, 0.5, "None", ha="center", va="center", transform=ax1.transAxes)

    if len(depleted) > 0:
        y = np.arange(len(depleted))
        ax2.barh(y, depleted["z_score"].values, color="#2563eb", edgecolor="white")
        ax2.set_yticks(y)
        ax2.set_yticklabels(depleted["kmer"].values, fontsize=8)
        ax2.invert_yaxis()
        ax2.set_xlabel("Z-score")
        ax2.set_title("Demand-Depleted")
    else:
        ax2.text(0.5, 0.5, "None", ha="center", va="center", transform=ax2.transAxes)

    fig.suptitle(f"Translational Demand (n={n_genes}, {tissue})", fontsize=11)
    fig.tight_layout()
    return _fig_to_b64(fig)


def _plot_correlation_hist(
    per_gene: pd.DataFrame,
    genome_corr: pd.DataFrame,
    summary: dict,
    from_sp: str,
    to_sp: str,
) -> str:
    """Correlation histogram: gene set vs genome."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    gs_r = per_gene["rscu_correlation"].dropna()
    bg_r = genome_corr["rscu_correlation"].dropna()

    bins = np.linspace(-1, 1, 41)
    ax1.hist(bg_r, bins=bins, alpha=0.5, label="Genome", density=True,
             color="#94a3b8", edgecolor="white")
    ax1.hist(gs_r, bins=bins, alpha=0.7, label="Gene set", density=True,
             color="#2563eb", edgecolor="white")
    ax1.axvline(summary["geneset_mean_r"], color="#2563eb", ls="--",
                label=f"Gene set mean={summary['geneset_mean_r']:.3f}")
    ax1.axvline(summary["genome_mean_r"], color="#6b7280", ls="--",
                label=f"Genome mean={summary['genome_mean_r']:.3f}")
    ax1.set_xlabel("RSCU Pearson r")
    ax1.set_ylabel("Density")
    ax1.set_title(f"{from_sp} vs {to_sp} RSCU Correlation\n"
                  f"Z={summary['z_score']:.2f}, p={summary['p_value']:.2e}")
    ax1.legend(fontsize=7)

    # Ranked bar chart
    sorted_r = gs_r.sort_values().reset_index(drop=True)
    ax2.bar(range(len(sorted_r)), sorted_r, color="#2563eb", width=1.0)
    ax2.axhline(summary["genome_mean_r"], color="#6b7280", ls="--",
                label=f"Genome mean={summary['genome_mean_r']:.3f}")
    ax2.set_xlabel("Ortholog pair (ranked)")
    ax2.set_ylabel("RSCU Pearson r")
    ax2.set_title("Per-Gene Cross-Species RSCU Correlation")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    return _fig_to_b64(fig)
