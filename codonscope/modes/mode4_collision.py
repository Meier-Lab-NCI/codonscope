"""Mode 4: Collision Potential.

Classify codons as fast/slow and count dicodon transition types
(FF, FS, SF, SS).  FS transitions mark sites where a trailing
ribosome (on fast codons) catches a leading ribosome (on a slow codon)
— the collision-prone junctions described by Aguilar Rangel et al. 2024.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from codonscope.core.codons import SENSE_CODONS, sequence_to_codons
from codonscope.core.optimality import OptimalityScorer
from codonscope.core.sequences import SequenceDB
from codonscope.core.statistics import power_check

logger = logging.getLogger(__name__)

TRANSITION_TYPES = ("FF", "FS", "SF", "SS")


def run_collision(
    species: str,
    gene_ids: list[str],
    wobble_penalty: float = 0.5,
    threshold: float | None = None,
    method: str = "wtai",
    output_dir: str | Path | None = None,
    data_dir: str | Path | None = None,
) -> dict:
    """Run Mode 4: Collision Potential analysis.

    Args:
        species: Species name (e.g. "yeast", "human").
        gene_ids: List of gene identifiers (any format).
        wobble_penalty: Penalty for wobble-decoded codons (default 0.5).
        threshold: Fast/slow cutoff.  *None* uses the median wtAI.
        method: Scoring method — ``"tai"`` or ``"wtai"`` (default).
        output_dir: Directory for output files.
        data_dir: Override default data directory.

    Returns:
        dict with keys:
            "transition_matrix_geneset" : dict {FF/FS/SF/SS: proportion}
            "transition_matrix_genome"  : dict {FF/FS/SF/SS: proportion}
            "fs_enrichment"             : float — observed/expected FS ratio
            "fs_sf_ratio_geneset"       : float — FS/SF for gene set
            "fs_sf_ratio_genome"        : float — FS/SF for genome
            "chi2_stat"                 : float
            "chi2_p"                    : float
            "per_gene_fs_frac"          : DataFrame per-gene FS fraction
            "fs_positions"              : DataFrame of clustered FS positions
            "fast_codons"               : set
            "slow_codons"               : set
            "threshold"                 : float used
            "id_summary"                : IDMapping
            "n_genes"                   : int
    """
    db = SequenceDB(species, data_dir=data_dir)
    scorer = OptimalityScorer(
        db.species_dir, wobble_penalty=wobble_penalty,
    )

    # Resolve gene IDs
    id_result = db.resolve_ids(gene_ids)
    gene_seqs = db.get_sequences(id_result)
    n_genes = len(gene_seqs)
    logger.info("Analyzing %d genes for collision potential", n_genes)

    for w in power_check(n_genes, k=1):
        logger.warning(w)

    all_seqs = db.get_all_sequences()

    # Classify codons
    fast, slow = scorer.classify_codons(threshold=threshold, method=method)
    used_threshold = threshold
    if used_threshold is None:
        weights = scorer.wtai_weights if method == "wtai" else scorer.tai_weights
        used_threshold = float(np.median(list(weights.values())))

    # ── Transition matrices ───────────────────────────────────────────
    gs_counts, gs_per_gene = _count_transitions(gene_seqs, fast, slow)
    bg_counts, _ = _count_transitions(all_seqs, fast, slow)

    gs_matrix = _to_proportions(gs_counts)
    bg_matrix = _to_proportions(bg_counts)

    # ── FS enrichment ─────────────────────────────────────────────────
    fs_enrichment = _fs_enrichment(gs_matrix, bg_matrix)
    fs_sf_gs = _fs_sf_ratio(gs_matrix)
    fs_sf_bg = _fs_sf_ratio(bg_matrix)

    # ── Chi-squared test ──────────────────────────────────────────────
    chi2, chi2_p = _chi_squared_test(gs_counts, bg_counts)

    # ── Per-gene FS fraction ──────────────────────────────────────────
    per_gene_df = _per_gene_fs(gs_per_gene)

    # ── Positional FS analysis ────────────────────────────────────────
    fs_positions = _fs_position_analysis(gene_seqs, fast, slow)

    # ── Output ────────────────────────────────────────────────────────
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        _write_outputs(
            gs_matrix, bg_matrix, fs_enrichment, fs_sf_gs, fs_sf_bg,
            chi2, chi2_p, per_gene_df, fs_positions, out,
        )

    return {
        "transition_matrix_geneset": gs_matrix,
        "transition_matrix_genome": bg_matrix,
        "fs_enrichment": fs_enrichment,
        "fs_sf_ratio_geneset": fs_sf_gs,
        "fs_sf_ratio_genome": fs_sf_bg,
        "chi2_stat": chi2,
        "chi2_p": chi2_p,
        "per_gene_fs_frac": per_gene_df,
        "fs_positions": fs_positions,
        "fast_codons": fast,
        "slow_codons": slow,
        "threshold": used_threshold,
        "id_summary": id_result,
        "n_genes": n_genes,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Transition counting
# ═══════════════════════════════════════════════════════════════════════════════

def _count_transitions(
    sequences: dict[str, str],
    fast: set[str],
    slow: set[str],
) -> tuple[dict[str, int], dict[str, dict[str, int]]]:
    """Count FF/FS/SF/SS transitions across all genes.

    Returns:
        (total_counts, per_gene_counts)
        total_counts: {FF: int, FS: int, SF: int, SS: int}
        per_gene_counts: {gene_name: {FF: int, ...}}
    """
    total = {t: 0 for t in TRANSITION_TYPES}
    per_gene: dict[str, dict[str, int]] = {}

    for gene, seq in sequences.items():
        codons = sequence_to_codons(seq)
        gene_counts = {t: 0 for t in TRANSITION_TYPES}

        for i in range(len(codons) - 1):
            c1 = codons[i]
            c2 = codons[i + 1]
            speed1 = "F" if c1 in fast else "S"
            speed2 = "F" if c2 in fast else "S"
            tt = speed1 + speed2
            gene_counts[tt] += 1
            total[tt] += 1

        per_gene[gene] = gene_counts

    return total, per_gene


def _to_proportions(counts: dict[str, int]) -> dict[str, float]:
    """Convert raw counts to proportions summing to 1."""
    total = sum(counts.values())
    if total == 0:
        return {t: 0.0 for t in TRANSITION_TYPES}
    return {t: c / total for t, c in counts.items()}


# ═══════════════════════════════════════════════════════════════════════════════
# FS enrichment metrics
# ═══════════════════════════════════════════════════════════════════════════════

def _fs_enrichment(
    gs_matrix: dict[str, float],
    bg_matrix: dict[str, float],
) -> float:
    """Ratio of gene-set FS proportion to genome FS proportion.

    > 1 means gene set has more collision-prone junctions than expected.
    """
    bg_fs = bg_matrix.get("FS", 0.0)
    gs_fs = gs_matrix.get("FS", 0.0)
    if bg_fs <= 0:
        return float("inf") if gs_fs > 0 else 1.0
    return gs_fs / bg_fs


def _fs_sf_ratio(matrix: dict[str, float]) -> float:
    """Ratio of FS to SF proportions.

    > 1 means more fast→slow than slow→fast transitions.
    """
    fs = matrix.get("FS", 0.0)
    sf = matrix.get("SF", 0.0)
    if sf <= 0:
        return float("inf") if fs > 0 else 1.0
    return fs / sf


# ═══════════════════════════════════════════════════════════════════════════════
# Chi-squared test
# ═══════════════════════════════════════════════════════════════════════════════

def _chi_squared_test(
    gs_counts: dict[str, int],
    bg_counts: dict[str, int],
) -> tuple[float, float]:
    """Chi-squared test comparing gene-set and genome transition counts.

    Null hypothesis: the gene set has the same transition distribution
    as the genome.  We compute expected counts for the gene set using
    genome proportions.
    """
    gs_total = sum(gs_counts.values())
    bg_total = sum(bg_counts.values())
    if gs_total == 0 or bg_total == 0:
        return 0.0, 1.0

    bg_props = {t: c / bg_total for t, c in bg_counts.items()}
    observed = np.array([gs_counts[t] for t in TRANSITION_TYPES], dtype=np.float64)
    expected = np.array([bg_props[t] * gs_total for t in TRANSITION_TYPES], dtype=np.float64)

    # Avoid division by zero for any zero-expected cell
    mask = expected > 0
    if not mask.any():
        return 0.0, 1.0

    chi2 = float(np.sum((observed[mask] - expected[mask]) ** 2 / expected[mask]))
    df = int(mask.sum()) - 1
    if df <= 0:
        return chi2, 1.0
    p = float(sp_stats.chi2.sf(chi2, df))
    return chi2, p


# ═══════════════════════════════════════════════════════════════════════════════
# Per-gene FS fraction
# ═══════════════════════════════════════════════════════════════════════════════

def _per_gene_fs(
    per_gene_counts: dict[str, dict[str, int]],
) -> pd.DataFrame:
    """Build DataFrame with per-gene FS fraction."""
    rows = []
    for gene, counts in sorted(per_gene_counts.items()):
        total = sum(counts.values())
        fs_frac = counts["FS"] / total if total > 0 else 0.0
        rows.append({
            "gene": gene,
            "n_transitions": total,
            "FF": counts["FF"],
            "FS": counts["FS"],
            "SF": counts["SF"],
            "SS": counts["SS"],
            "fs_fraction": fs_frac,
        })
    return pd.DataFrame(rows).sort_values(
        "fs_fraction", ascending=False
    ).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Positional FS analysis
# ═══════════════════════════════════════════════════════════════════════════════

def _fs_position_analysis(
    sequences: dict[str, str],
    fast: set[str],
    slow: set[str],
    bin_size: int = 10,
) -> pd.DataFrame:
    """Count FS transitions by CDS position (binned as % of CDS).

    Returns DataFrame with columns: position_pct, fs_count, total_transitions.
    """
    n_bins = 100 // bin_size
    fs_hist = np.zeros(n_bins, dtype=np.int64)
    total_hist = np.zeros(n_bins, dtype=np.int64)

    for seq in sequences.values():
        codons = sequence_to_codons(seq)
        n_codons = len(codons)
        if n_codons < 2:
            continue

        for i in range(n_codons - 1):
            c1 = codons[i]
            c2 = codons[i + 1]
            speed1 = "F" if c1 in fast else "S"
            speed2 = "F" if c2 in fast else "S"

            # Position as % of CDS
            pct = i / (n_codons - 1) * 100
            bin_idx = min(int(pct // bin_size), n_bins - 1)

            total_hist[bin_idx] += 1
            if speed1 == "F" and speed2 == "S":
                fs_hist[bin_idx] += 1

    rows = []
    for b in range(n_bins):
        pct_start = b * bin_size
        fs_frac = fs_hist[b] / total_hist[b] if total_hist[b] > 0 else 0.0
        rows.append({
            "position_pct": pct_start,
            "fs_count": int(fs_hist[b]),
            "total_transitions": int(total_hist[b]),
            "fs_fraction": fs_frac,
        })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# Output
# ═══════════════════════════════════════════════════════════════════════════════

def _write_outputs(
    gs_matrix: dict[str, float],
    bg_matrix: dict[str, float],
    fs_enrichment: float,
    fs_sf_gs: float,
    fs_sf_bg: float,
    chi2: float,
    chi2_p: float,
    per_gene_df: pd.DataFrame,
    fs_positions: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Write TSV results and plots."""
    # Transition matrix
    tm_df = pd.DataFrame({
        "transition": list(TRANSITION_TYPES),
        "geneset_proportion": [gs_matrix[t] for t in TRANSITION_TYPES],
        "genome_proportion": [bg_matrix[t] for t in TRANSITION_TYPES],
    })
    tm_df.to_csv(
        output_dir / "collision_transitions.tsv", sep="\t", index=False,
        float_format="%.6g",
    )

    # Per-gene FS
    per_gene_df.to_csv(
        output_dir / "collision_per_gene.tsv", sep="\t", index=False,
        float_format="%.6g",
    )

    # Positional FS
    fs_positions.to_csv(
        output_dir / "collision_positions.tsv", sep="\t", index=False,
        float_format="%.6g",
    )

    logger.info("Collision results written to %s", output_dir)

    # Plot
    try:
        _plot_collision(gs_matrix, bg_matrix, fs_positions, output_dir / "collision.png")
    except Exception as exc:
        logger.warning("Plot generation failed: %s", exc)


def _plot_collision(
    gs_matrix: dict[str, float],
    bg_matrix: dict[str, float],
    fs_positions: pd.DataFrame,
    path: Path,
) -> None:
    """Two-panel collision plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: transition matrix comparison
    x = np.arange(4)
    width = 0.35
    gs_vals = [gs_matrix[t] for t in TRANSITION_TYPES]
    bg_vals = [bg_matrix[t] for t in TRANSITION_TYPES]

    ax1.bar(x - width / 2, gs_vals, width, color="#d73027", label="Gene set")
    ax1.bar(x + width / 2, bg_vals, width, color="#4575b4", label="Genome")
    ax1.set_xticks(x)
    ax1.set_xticklabels(TRANSITION_TYPES)
    ax1.set_ylabel("Proportion")
    ax1.set_title("(A) Transition Type Distribution")
    ax1.legend(fontsize=9)

    # Panel B: positional FS fraction
    if len(fs_positions) > 0:
        ax2.bar(
            fs_positions["position_pct"],
            fs_positions["fs_fraction"],
            width=fs_positions["position_pct"].diff().dropna().iloc[0]
            if len(fs_positions) > 1 else 10,
            color="#d73027", alpha=0.7,
        )
    ax2.set_xlabel("CDS position (%)")
    ax2.set_ylabel("FS fraction")
    ax2.set_title("(B) FS Transitions Along CDS")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Collision plot saved to %s", path)
