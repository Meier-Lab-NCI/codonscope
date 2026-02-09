"""Two-List Differential Codon Usage Analysis.

Compares codon usage between two gene lists using Mann-Whitney U test.
For each of the 61 sense codons, compute:
  - Mann-Whitney U statistic comparing per-gene frequencies
  - Effect size: rank-biserial r = 1 - 2U/(n_a * n_b)
  - Fold change: mean_a / mean_b
  - Benjamini-Hochberg correction across 61 tests
"""

import base64
import logging
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from codonscope.core.codons import CODON_TABLE, SENSE_CODONS, annotate_kmer
from codonscope.core.sequences import SequenceDB
from codonscope.core.statistics import (
    benjamini_hochberg,
    compute_geneset_frequencies,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════

def run_differential(
    species: str,
    gene_ids_a: list[str],
    gene_ids_b: list[str],
    labels: tuple[str, str] = ("List A", "List B"),
    n_bootstrap: int = 10000,
    seed: int | None = None,
    output_dir: str | Path | None = None,
    data_dir: str | Path | None = None,
) -> dict:
    """Run differential codon usage analysis between two gene lists.

    Args:
        species: Species name (e.g. "yeast", "human").
        gene_ids_a: First gene list (any ID format).
        gene_ids_b: Second gene list (any ID format).
        labels: Labels for the two lists (default: ("List A", "List B")).
        n_bootstrap: Bootstrap iterations for attribution analysis (default: 10000).
        seed: Random seed for reproducibility.
        output_dir: Directory for output files. None = no file output.
        data_dir: Override default data directory.

    Returns:
        dict with keys:
            "results": DataFrame with columns [kmer, amino_acid, mean_a, mean_b,
                       fold_change, u_stat, p_value, adjusted_p, rank_biserial_r,
                       z_genome_a, z_genome_b]
            "n_genes_a": int
            "n_genes_b": int
            "labels": tuple[str, str]
            "species": str
            "attribution_a": dict (optional, from run_disentangle)
            "attribution_b": dict (optional, from run_disentangle)
    """
    db = SequenceDB(species, data_dir=data_dir)

    # Resolve gene IDs
    id_result_a = db.resolve_ids(gene_ids_a)
    id_result_b = db.resolve_ids(gene_ids_b)
    gene_seqs_a = db.get_sequences(id_result_a)
    gene_seqs_b = db.get_sequences(id_result_b)
    n_genes_a = len(gene_seqs_a)
    n_genes_b = len(gene_seqs_b)

    logger.info("Differential analysis: %s (%d genes) vs %s (%d genes)",
                labels[0], n_genes_a, labels[1], n_genes_b)

    if n_genes_a < 5 or n_genes_b < 5:
        logger.warning("Gene lists are small (n_a=%d, n_b=%d). Results may be unreliable.",
                       n_genes_a, n_genes_b)

    # Compute per-gene frequency matrices
    per_gene_a, mean_a, kmer_names = compute_geneset_frequencies(gene_seqs_a, k=1, trim_ramp=0)
    per_gene_b, mean_b, _ = compute_geneset_frequencies(gene_seqs_b, k=1, trim_ramp=0)

    # Mann-Whitney U test for each codon
    results = []
    for i, kmer in enumerate(kmer_names):
        freq_a = per_gene_a[:, i]
        freq_b = per_gene_b[:, i]

        # Mann-Whitney U test
        try:
            u_stat, p_value = mannwhitneyu(freq_a, freq_b, alternative='two-sided')
        except ValueError:
            # Handle case where all values are identical
            u_stat = n_genes_a * n_genes_b / 2.0
            p_value = 1.0

        # Effect size: rank-biserial r = 1 - 2U/(n_a * n_b)
        rank_biserial_r = 1.0 - 2.0 * u_stat / (n_genes_a * n_genes_b)

        # Fold change (guard against zero)
        epsilon = 1e-9
        fold_change = (mean_a[i] + epsilon) / (mean_b[i] + epsilon)

        results.append({
            "kmer": kmer,
            "amino_acid": CODON_TABLE.get(kmer, "?"),
            "mean_a": mean_a[i],
            "mean_b": mean_b[i],
            "fold_change": fold_change,
            "u_stat": u_stat,
            "p_value": p_value,
            "rank_biserial_r": rank_biserial_r,
        })

    df = pd.DataFrame(results)

    # Benjamini-Hochberg correction
    df["adjusted_p"] = benjamini_hochberg(df["p_value"].values)

    # Load genome background for Z-scores vs genome
    species_dir = db.species_dir
    bg_mono = np.load(species_dir / "background_mono.npz")
    bg_per_gene = bg_mono["per_gene"]

    # Compute Z-scores vs genome for both lists
    from codonscope.core.statistics import bootstrap_zscores
    z_a, _, _ = bootstrap_zscores(mean_a, bg_per_gene, n_genes_a, n_bootstrap=200, seed=seed)
    z_b, _, _ = bootstrap_zscores(mean_b, bg_per_gene, n_genes_b, n_bootstrap=200, seed=seed)

    df["z_genome_a"] = z_a
    df["z_genome_b"] = z_b

    # Sort by absolute rank-biserial r (effect size)
    df = df.sort_values("rank_biserial_r", key=np.abs, ascending=False).reset_index(drop=True)

    result = {
        "results": df,
        "n_genes_a": n_genes_a,
        "n_genes_b": n_genes_b,
        "labels": labels,
        "species": species,
    }

    # Optional: run attribution analysis for both lists
    # Wrap in try/except since it's optional
    try:
        from codonscope.modes.mode5_disentangle import run_disentangle
        logger.info("Running attribution analysis for %s...", labels[0])
        attribution_a = run_disentangle(
            species=species,
            gene_ids=gene_ids_a,
            n_bootstrap=n_bootstrap,
            seed=seed,
            output_dir=None,
            data_dir=data_dir,
        )
        logger.info("Running attribution analysis for %s...", labels[1])
        attribution_b = run_disentangle(
            species=species,
            gene_ids=gene_ids_b,
            n_bootstrap=n_bootstrap,
            seed=seed,
            output_dir=None,
            data_dir=data_dir,
        )
        result["attribution_a"] = attribution_a
        result["attribution_b"] = attribution_b
    except Exception as exc:
        logger.warning("Attribution analysis failed: %s", exc)

    # Output files
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        df.to_csv(out / "differential_results.tsv", sep="\t", index=False, float_format="%.6g")
        logger.info("Differential results written to %s", out / "differential_results.tsv")

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# HTML Report Generation
# ═══════════════════════════════════════════════════════════════════════════════

def generate_differential_report(
    species: str,
    gene_ids_a: list[str],
    gene_ids_b: list[str],
    labels: tuple[str, str] = ("List A", "List B"),
    output: str | Path = "differential_report.html",
    n_bootstrap: int = 10000,
    seed: int | None = None,
    output_dir: str | Path | None = None,
    data_dir: str | Path | None = None,
) -> Path:
    """Generate a standalone HTML report for differential analysis.

    Args:
        species: Species name (e.g. "yeast", "human").
        gene_ids_a: First gene list (any ID format).
        gene_ids_b: Second gene list (any ID format).
        labels: Labels for the two lists (default: ("List A", "List B")).
        output: Output HTML file path.
        n_bootstrap: Bootstrap iterations.
        seed: Random seed.
        output_dir: Directory for TSV output files.
        data_dir: Override default data directory.

    Returns:
        Path to the generated HTML report.
    """
    # Run analysis
    result = run_differential(
        species=species,
        gene_ids_a=gene_ids_a,
        gene_ids_b=gene_ids_b,
        labels=labels,
        n_bootstrap=n_bootstrap,
        seed=seed,
        output_dir=output_dir,
        data_dir=data_dir,
    )

    df = result["results"]
    n_genes_a = result["n_genes_a"]
    n_genes_b = result["n_genes_b"]
    label_a, label_b = result["labels"]

    # Generate plots
    waterfall_img = _plot_differential_waterfall(df, labels)
    scatter_img = _plot_z_genome_scatter(df, labels)

    # Build HTML
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Differential Codon Usage: {label_a} vs {label_b}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            max-width: 1200px;
            margin: 40px auto;
            padding: 0 20px;
            line-height: 1.6;
            color: #333;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #2c3e50;
            margin-top: 40px;
            border-bottom: 2px solid #ecf0f1;
            padding-bottom: 8px;
        }}
        .summary {{
            background: #f8f9fa;
            border-left: 4px solid #3498db;
            padding: 15px;
            margin: 20px 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            font-size: 14px;
        }}
        th, td {{
            padding: 8px 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background: #f8f9fa;
            font-weight: 600;
            color: #2c3e50;
        }}
        tr:hover {{
            background: #f8f9fa;
        }}
        .significant {{
            background: #fff3cd;
        }}
        img {{
            max-width: 100%;
            height: auto;
            display: block;
            margin: 20px auto;
        }}
        .enriched {{
            color: #d73027;
        }}
        .depleted {{
            color: #4575b4;
        }}
    </style>
</head>
<body>
    <h1>Differential Codon Usage Analysis</h1>

    <div class="summary">
        <strong>Species:</strong> {species}<br>
        <strong>{label_a}:</strong> {n_genes_a} genes<br>
        <strong>{label_b}:</strong> {n_genes_b} genes<br>
        <strong>Test:</strong> Mann-Whitney U (two-sided)<br>
        <strong>Multiple testing correction:</strong> Benjamini-Hochberg FDR
    </div>
"""

    # Summary of significant codons
    sig = df[df["adjusted_p"] < 0.05]
    n_sig = len(sig)
    a_enriched = sig[sig["rank_biserial_r"] > 0]
    b_enriched = sig[sig["rank_biserial_r"] < 0]

    html += f"""
    <h2>Summary</h2>
    <p><strong>{n_sig}</strong> codons show significant differential usage (adjusted p &lt; 0.05):</p>
    <ul>
        <li><span class="enriched">Enriched in {label_a}:</span> {len(a_enriched)} codons</li>
        <li><span class="depleted">Enriched in {label_b}:</span> {len(b_enriched)} codons</li>
    </ul>
"""

    # Differential waterfall chart
    html += f"""
    <h2>Differential Codon Usage</h2>
    <p>Codons are ranked by effect size (rank-biserial r). Blue = enriched in {label_a}, Red = enriched in {label_b}.</p>
    <img src="data:image/png;base64,{waterfall_img}" alt="Differential Waterfall Chart">
"""

    # Z-genome scatter plot
    html += f"""
    <h2>Z-scores vs Genome Background</h2>
    <p>Each point represents a codon. X-axis: Z-score vs genome for {label_a}. Y-axis: Z-score vs genome for {label_b}.</p>
    <img src="data:image/png;base64,{scatter_img}" alt="Z-genome Scatter Plot">
"""

    # Table of significant codons
    if n_sig > 0:
        html += f"""
    <h2>Significant Differential Codons (adjusted p &lt; 0.05)</h2>
    <table>
        <tr>
            <th>Codon</th>
            <th>AA</th>
            <th>{label_a} Mean</th>
            <th>{label_b} Mean</th>
            <th>Fold Change</th>
            <th>Rank-Biserial r</th>
            <th>p-value</th>
            <th>Adjusted p</th>
        </tr>
"""
        for _, row in sig.iterrows():
            color_class = "enriched" if row["rank_biserial_r"] > 0 else "depleted"
            html += f"""
        <tr class="significant">
            <td><strong>{row['kmer']}</strong></td>
            <td>{row['amino_acid']}</td>
            <td>{row['mean_a']:.4f}</td>
            <td>{row['mean_b']:.4f}</td>
            <td>{row['fold_change']:.2f}</td>
            <td class="{color_class}">{row['rank_biserial_r']:+.3f}</td>
            <td>{row['p_value']:.2e}</td>
            <td>{row['adjusted_p']:.2e}</td>
        </tr>
"""
        html += """
    </table>
"""
    else:
        html += """
    <h2>Significant Differential Codons</h2>
    <p>No codons reached statistical significance (adjusted p &lt; 0.05).</p>
"""

    # Full results table
    html += f"""
    <h2>All Codons</h2>
    <table>
        <tr>
            <th>Codon</th>
            <th>AA</th>
            <th>{label_a} Mean</th>
            <th>{label_b} Mean</th>
            <th>Fold Change</th>
            <th>Rank-Biserial r</th>
            <th>p-value</th>
            <th>Adjusted p</th>
            <th>Z (vs genome) {label_a}</th>
            <th>Z (vs genome) {label_b}</th>
        </tr>
"""
    for _, row in df.iterrows():
        sig_class = "significant" if row["adjusted_p"] < 0.05 else ""
        color_class = "enriched" if row["rank_biserial_r"] > 0 else "depleted"
        html += f"""
        <tr class="{sig_class}">
            <td><strong>{row['kmer']}</strong></td>
            <td>{row['amino_acid']}</td>
            <td>{row['mean_a']:.4f}</td>
            <td>{row['mean_b']:.4f}</td>
            <td>{row['fold_change']:.2f}</td>
            <td class="{color_class}">{row['rank_biserial_r']:+.3f}</td>
            <td>{row['p_value']:.2e}</td>
            <td>{row['adjusted_p']:.2e}</td>
            <td>{row['z_genome_a']:+.2f}</td>
            <td>{row['z_genome_b']:+.2f}</td>
        </tr>
"""
    html += """
    </table>
</body>
</html>
"""

    # Write HTML file
    output_path = Path(output)
    output_path.write_text(html)
    logger.info("Differential report written to %s", output_path)

    return output_path


# ═══════════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════════

def _plot_differential_waterfall(df: pd.DataFrame, labels: tuple[str, str]) -> str:
    """Generate waterfall chart of differential codon usage.

    Returns base64-encoded PNG image.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 8))

    # Sort by rank-biserial r
    df_sorted = df.sort_values("rank_biserial_r")

    # Color by direction (blue = A enriched, red = B enriched)
    colors = ["#4575b4" if r > 0 else "#d73027" for r in df_sorted["rank_biserial_r"]]

    # Highlight significant codons with black edge
    sig = df_sorted["adjusted_p"] < 0.05
    edge_colors = ["black" if s else "none" for s in sig]
    edge_widths = [1.0 if s else 0 for s in sig]

    # Plot
    y_positions = range(len(df_sorted))
    ax.barh(y_positions, df_sorted["rank_biserial_r"], color=colors,
            edgecolor=edge_colors, linewidth=edge_widths)

    # Labels
    codon_labels = [f"{r['kmer']} ({r['amino_acid']})" for _, r in df_sorted.iterrows()]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(codon_labels, fontsize=6)
    ax.set_xlabel("Rank-Biserial r", fontsize=10)
    ax.set_title(f"Differential Codon Usage: {labels[0]} vs {labels[1]}", fontsize=12)
    ax.axvline(0, color="black", linewidth=0.8)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#4575b4", label=f"Enriched in {labels[0]}"),
        Patch(facecolor="#d73027", label=f"Enriched in {labels[1]}"),
        Patch(facecolor="white", edgecolor="black", label="Significant (adj_p < 0.05)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

    fig.tight_layout()

    # Convert to base64
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")

    return img_b64


def _plot_z_genome_scatter(df: pd.DataFrame, labels: tuple[str, str]) -> str:
    """Generate scatter plot of Z-scores vs genome for both lists.

    Returns base64-encoded PNG image.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8))

    # Color by significance
    sig = df["adjusted_p"] < 0.05
    colors = ["#e74c3c" if s else "#95a5a6" for s in sig]
    sizes = [50 if s else 20 for s in sig]

    # Scatter plot
    ax.scatter(df["z_genome_a"], df["z_genome_b"], c=colors, s=sizes, alpha=0.6, edgecolors="black", linewidth=0.5)

    # Reference lines
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.plot([-10, 10], [-10, 10], color="gray", linewidth=0.5, linestyle=":")  # diagonal

    # Labels
    ax.set_xlabel(f"Z-score vs Genome ({labels[0]})", fontsize=10)
    ax.set_ylabel(f"Z-score vs Genome ({labels[1]})", fontsize=10)
    ax.set_title(f"Codon Enrichment vs Genome Background", fontsize=12)

    # Set equal aspect ratio
    ax.set_aspect("equal", adjustable="box")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#e74c3c", label="Differentially used (adj_p < 0.05)"),
        Patch(facecolor="#95a5a6", label="Not significant"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=8)

    fig.tight_layout()

    # Convert to base64
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")

    return img_b64
