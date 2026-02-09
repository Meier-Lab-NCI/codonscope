"""Mode 1: Sequence Composition Analysis.

Does a gene list have unusual codon/dicodon/tricodon frequencies
compared to the genome background?
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from codonscope.core.codons import CODON_TABLE, all_possible_kmers, annotate_kmer, kmer_frequencies
from codonscope.core.sequences import SequenceDB
from codonscope.core.statistics import (
    benjamini_hochberg,
    binomial_glm_zscores,
    bootstrap_pvalues,
    bootstrap_zscores,
    cohens_d,
    compare_to_background,
    compute_geneset_frequencies,
    diagnostic_ks_tests,
    power_check,
)

logger = logging.getLogger(__name__)


def run_composition(
    species: str,
    gene_ids: list[str],
    k: int | None = None,
    kmer: int | None = None,
    kmer_size: int | None = None,
    background: str = "all",
    trim_ramp: int = 0,
    min_genes: int = 10,
    n_bootstrap: int = 10_000,
    output_dir: str | Path | None = None,
    seed: int | None = None,
    data_dir: str | Path | None = None,
    model: str = "bootstrap",
) -> dict:
    """Run Mode 1 Sequence Composition analysis.

    Args:
        species: Species name (e.g. "yeast").
        gene_ids: List of gene identifiers (any format).
        k: k-mer size (1=mono, 2=di, 3=tri).  Aliases: kmer, kmer_size.
        kmer: Alias for k.
        kmer_size: Alias for k.
        background: "all" for whole genome, "matched" for length+GC matched.
        trim_ramp: Number of 5' codons to exclude (default 0).
        min_genes: Minimum gene list size.
        n_bootstrap: Bootstrap iterations.
        output_dir: Directory for output files. None = no file output.
        seed: Random seed for reproducibility.
        data_dir: Override default data directory.

    Returns:
        dict with keys:
            "results": DataFrame (kmer, observed_freq, expected_freq,
                       z_score, p_value, adjusted_p, cohens_d)
            "diagnostics": dict with KS test results and warnings
            "id_summary": dict with mapping stats
            "n_genes": int
    """
    # Resolve k-mer size from aliases (k, kmer, kmer_size)
    k_resolved = k or kmer or kmer_size or 1
    if k_resolved not in (1, 2, 3):
        raise ValueError(f"k-mer size must be 1, 2, or 3, got {k_resolved}")
    k = k_resolved

    # Load database
    db = SequenceDB(species, data_dir=data_dir)
    species_dir = db.species_dir

    # Resolve gene IDs
    id_result = db.resolve_ids(gene_ids)
    sys_names = list(id_result.values())

    if len(sys_names) < min_genes:
        logger.warning(
            "Only %d genes mapped (minimum %d). Results may be unreliable.",
            len(sys_names), min_genes,
        )

    # Power check
    warnings_list = power_check(len(sys_names), k)
    for w in warnings_list:
        logger.warning(w)

    # Get sequences
    gene_seqs = db.get_sequences(sys_names)
    n_genes = len(gene_seqs)
    logger.info("Analyzing %d genes with k=%d", n_genes, k)

    # Validate model parameter
    if model not in ("bootstrap", "binomial"):
        raise ValueError(f"model must be 'bootstrap' or 'binomial', got {model!r}")

    if model == "binomial":
        if k != 1:
            raise ValueError(
                "Binomial GLM only supports monocodon (k=1). "
                f"Got k={k}. Use model='bootstrap' for dicodons/tricodons."
            )
        # Load ALL genome sequences for the GLM
        all_seqs = db.get_all_sequences()
        results_df = binomial_glm_zscores(
            gene_seqs, all_seqs, k=1, trim_ramp=trim_ramp,
        )
    else:
        # Choose background file
        bg_key = {1: "mono", 2: "di", 3: "tri"}[k]
        bg_path = species_dir / f"background_{bg_key}.npz"

        if background == "matched":
            results_df = _run_matched_background(
                gene_seqs, db, bg_path, k=k, trim_ramp=trim_ramp,
                n_bootstrap=n_bootstrap, seed=seed,
            )
        else:
            results_df = compare_to_background(
                gene_seqs, bg_path, k=k,
                n_bootstrap=n_bootstrap, trim_ramp=trim_ramp, seed=seed,
            )

    # Add amino acid annotation column (if not already present from GLM)
    def _kmer_to_aa(kmer: str) -> str:
        codons = [kmer[i:i + 3] for i in range(0, len(kmer), 3)]
        return "-".join(CODON_TABLE.get(c, "?") for c in codons)

    if "amino_acid" not in results_df.columns:
        results_df["amino_acid"] = results_df["kmer"].apply(_kmer_to_aa)

    # Run diagnostics
    diagnostics = _run_diagnostics(gene_seqs, species_dir)
    diagnostics["power_warnings"] = warnings_list

    # Write output files
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        _write_outputs(results_df, diagnostics, out, k)

    result = {
        "results": results_df,
        "diagnostics": diagnostics,
        "id_summary": id_result,
        "n_genes": n_genes,
    }
    if model == "binomial":
        result["model"] = "binomial"
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Matched background
# ═══════════════════════════════════════════════════════════════════════════════

def _run_matched_background(
    gene_seqs: dict[str, str],
    db: SequenceDB,
    bg_path: Path,
    k: int,
    trim_ramp: int,
    n_bootstrap: int,
    seed: int | None,
) -> pd.DataFrame:
    """Run analysis with length+GC matched background.

    Select background genes within ±20% CDS length and ±5% GC content
    of the gene set distribution.
    """
    # Compute gene set length and GC stats
    gs_lengths = np.array([len(s) for s in gene_seqs.values()])
    gs_gc = np.array([
        (s.count("G") + s.count("C")) / len(s) if len(s) > 0 else 0.0
        for s in gene_seqs.values()
    ])

    len_lo = gs_lengths.min() * 0.8
    len_hi = gs_lengths.max() * 1.2
    gc_lo = gs_gc.min() - 0.05
    gc_hi = gs_gc.max() + 0.05

    # Load background and filter
    bg = np.load(bg_path, allow_pickle=True)
    bg_gene_names = bg["gene_names"]

    # Load metadata for filtering
    meta_path = db.species_dir / "gene_metadata.npz"
    meta = np.load(meta_path)
    meta_names = meta["gene_names"]
    meta_lengths = meta["cds_lengths"]
    meta_gc = meta["gc_contents"]

    # Build name→index map for metadata
    meta_idx = {name: i for i, name in enumerate(meta_names)}

    # Build name→index map for background matrix
    bg_idx = {name: i for i, name in enumerate(bg_gene_names)}

    # Find matching genes
    matched_bg_indices = []
    for name in bg_gene_names:
        if name not in meta_idx:
            continue
        mi = meta_idx[name]
        length = meta_lengths[mi]
        gc = meta_gc[mi]
        if len_lo <= length <= len_hi and gc_lo <= gc <= gc_hi:
            matched_bg_indices.append(bg_idx[name])

    matched_bg_indices = np.array(matched_bg_indices)
    logger.info(
        "Matched background: %d of %d genes (length %.0f-%.0f, GC %.2f-%.2f)",
        len(matched_bg_indices), len(bg_gene_names),
        len_lo, len_hi, gc_lo, gc_hi,
    )

    if len(matched_bg_indices) < 100:
        logger.warning(
            "Matched background has only %d genes. "
            "Results may be unreliable. Consider using --background all.",
            len(matched_bg_indices),
        )

    # Compute gene set frequencies
    _, geneset_mean, kmer_names = compute_geneset_frequencies(
        gene_seqs, k=k, trim_ramp=trim_ramp
    )

    if "per_gene" in bg:
        # Filter the per-gene matrix
        matched_per_gene = bg["per_gene"][matched_bg_indices]

        z_scores, boot_mean, boot_std = bootstrap_zscores(
            geneset_mean, matched_per_gene,
            n_genes=len(gene_seqs), n_bootstrap=n_bootstrap, seed=seed,
        )
    else:
        # Tricodon fallback: use genome-wide std with analytic SE
        bg_std = bg["std"].astype(np.float64)
        bg_mean = bg["mean"].astype(np.float64)
        se = bg_std / np.sqrt(len(gene_seqs))
        with np.errstate(divide="ignore", invalid="ignore"):
            z_scores = np.where(se > 0, (geneset_mean - bg_mean) / se, 0.0)
        boot_mean = bg_mean

    p_values = bootstrap_pvalues(z_scores)
    adj_p = benjamini_hochberg(p_values)
    d = cohens_d(geneset_mean, bg["mean"].astype(np.float64), bg["std"].astype(np.float64))

    df = pd.DataFrame({
        "kmer": kmer_names,
        "observed_freq": geneset_mean,
        "expected_freq": boot_mean,
        "z_score": z_scores,
        "p_value": p_values,
        "adjusted_p": adj_p,
        "cohens_d": d,
    })
    return df.sort_values("z_score", key=np.abs, ascending=False).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Diagnostics
# ═══════════════════════════════════════════════════════════════════════════════

def _run_diagnostics(gene_seqs: dict[str, str], species_dir: Path) -> dict:
    """Run KS diagnostic tests for length and GC bias."""
    meta_path = species_dir / "gene_metadata.npz"
    if not meta_path.exists():
        return {"length_warning": False, "gc_warning": False}

    meta = np.load(meta_path)
    bg_lengths = meta["cds_lengths"].astype(np.float64)
    bg_gc = meta["gc_contents"].astype(np.float64)

    gs_lengths = np.array([len(s) for s in gene_seqs.values()], dtype=np.float64)
    gs_gc = np.array([
        (s.count("G") + s.count("C")) / len(s) if len(s) > 0 else 0.0
        for s in gene_seqs.values()
    ], dtype=np.float64)

    return diagnostic_ks_tests(gs_lengths, gs_gc, bg_lengths, bg_gc)


# ═══════════════════════════════════════════════════════════════════════════════
# Output
# ═══════════════════════════════════════════════════════════════════════════════

def _write_outputs(
    results_df: pd.DataFrame,
    diagnostics: dict,
    output_dir: Path,
    k: int,
) -> None:
    """Write TSV results and plots to output directory."""
    k_label = {1: "mono", 2: "di", 3: "tri"}[k]

    # TSV table
    tsv_path = output_dir / f"composition_{k_label}codons.tsv"
    results_df.to_csv(tsv_path, sep="\t", index=False, float_format="%.6g")
    logger.info("Results written to %s", tsv_path)

    # Plots
    try:
        _plot_volcano(results_df, output_dir / f"volcano_{k_label}.png")
        _plot_top_kmers(results_df, output_dir / f"top_kmers_{k_label}.png", k)
    except Exception as exc:
        logger.warning("Plot generation failed: %s", exc)


def _plot_volcano(df: pd.DataFrame, path: Path) -> None:
    """Volcano plot: Z-score vs -log10(adjusted p)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))

    neg_log_p = -np.log10(df["adjusted_p"].clip(lower=1e-300))
    z = df["z_score"]

    # Color by significance
    sig = df["adjusted_p"] < 0.05
    ax.scatter(z[~sig], neg_log_p[~sig], c="grey", alpha=0.4, s=10, label="NS")
    ax.scatter(z[sig], neg_log_p[sig], c="red", alpha=0.7, s=15, label="adj_p < 0.05")

    # Label top hits
    k = len(df["kmer"].iloc[0]) // 3 if len(df) > 0 else 1
    top = df.head(10)
    for _, row in top.iterrows():
        if row["adjusted_p"] < 0.05:
            ax.annotate(
                annotate_kmer(row["kmer"], k),
                (row["z_score"], -np.log10(max(row["adjusted_p"], 1e-300))),
                fontsize=7, alpha=0.8,
            )

    ax.set_xlabel("Z-score")
    ax.set_ylabel("-log10(adjusted p-value)")
    ax.set_title("Codon Usage: Volcano Plot")
    ax.legend(fontsize=8)
    ax.axhline(-np.log10(0.05), color="blue", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Volcano plot saved to %s", path)


def _plot_top_kmers(df: pd.DataFrame, path: Path, k: int) -> None:
    """Bar chart of top enriched and depleted k-mers."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sig = df[df["adjusted_p"] < 0.05].copy()
    if len(sig) == 0:
        sig = df.head(20).copy()

    # Top 10 enriched + top 10 depleted
    enriched = sig[sig["z_score"] > 0].head(10)
    depleted = sig[sig["z_score"] < 0].head(10)
    plot_df = pd.concat([enriched, depleted]).sort_values("z_score")

    if len(plot_df) == 0:
        return

    fig, ax = plt.subplots(figsize=(8, max(4, len(plot_df) * 0.35)))
    colors = ["#d73027" if z > 0 else "#4575b4" for z in plot_df["z_score"]]
    ax.barh(range(len(plot_df)), plot_df["z_score"], color=colors)
    ax.set_yticks(range(len(plot_df)))
    labels = [annotate_kmer(km, k) for km in plot_df["kmer"]]
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Z-score")

    k_label = {1: "Monocodon", 2: "Dicodon", 3: "Tricodon"}[k]
    ax.set_title(f"Top Enriched/Depleted {k_label}s")
    ax.axvline(0, color="black", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Top k-mers plot saved to %s", path)
