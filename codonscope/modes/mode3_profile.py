"""Mode 3: Optimality Profile.

Per-position codon optimality along CDSs, with metagene averaging and
ramp analysis.  Compares a gene set's optimality profile to the genome
background.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from codonscope.core.codons import CODON_TABLE, sequence_to_codons
from codonscope.core.optimality import OptimalityScorer
from codonscope.core.sequences import SequenceDB
from codonscope.core.statistics import (
    bootstrap_pvalues,
    bootstrap_zscores,
    benjamini_hochberg,
    compute_geneset_frequencies,
    power_check,
)

logger = logging.getLogger(__name__)

# Default parameters
DEFAULT_WINDOW = 10
DEFAULT_WOBBLE_PENALTY = 0.5
DEFAULT_RAMP_CODONS = 50
N_METAGENE_BINS = 100  # normalise CDSs to 100 bins (0–99)


def run_profile(
    species: str,
    gene_ids: list[str],
    window: int = DEFAULT_WINDOW,
    wobble_penalty: float = DEFAULT_WOBBLE_PENALTY,
    ramp_codons: int = DEFAULT_RAMP_CODONS,
    method: str = "wtai",
    n_bootstrap: int = 1_000,
    seed: int | None = None,
    output_dir: str | Path | None = None,
    data_dir: str | Path | None = None,
) -> dict:
    """Run Mode 3: Optimality Profile analysis.

    Args:
        species: Species name (e.g. "yeast", "human").
        gene_ids: List of gene identifiers (any format).
        window: Sliding-window size in codons for smoothing.
        wobble_penalty: Penalty for wobble-decoded codons (default 0.5).
        ramp_codons: Number of 5' codons for ramp analysis (default 50).
        method: Scoring method — ``"tai"`` or ``"wtai"`` (default).
        n_bootstrap: Bootstrap iterations for ramp/body composition.
        seed: Random seed for reproducibility.
        output_dir: Directory for output files.  *None* = no file output.
        data_dir: Override default data directory.

    Returns:
        dict with keys:
            "metagene_geneset"  : 1-D array (N_METAGENE_BINS,) — gene-set
            "metagene_genome"   : 1-D array (N_METAGENE_BINS,) — genome bg
            "ramp_analysis"     : dict with ramp statistics
            "ramp_composition"  : DataFrame — ramp region codon composition
            "body_composition"  : DataFrame — body region codon composition
            "per_gene_scores"   : DataFrame with per-gene mean tAI/wtAI
            "scorer"            : OptimalityScorer instance
            "id_summary"        : IDMapping
            "n_genes"           : int
    """
    db = SequenceDB(species, data_dir=data_dir)
    scorer = OptimalityScorer(
        db.species_dir, wobble_penalty=wobble_penalty,
    )

    # Resolve gene IDs
    id_result = db.resolve_ids(gene_ids)
    gene_seqs = db.get_sequences(id_result)
    n_genes = len(gene_seqs)
    logger.info("Analyzing %d genes for optimality profile", n_genes)

    for w in power_check(n_genes, k=1):
        logger.warning(w)

    # Genome background sequences
    all_seqs = db.get_all_sequences()

    # ── Per-gene mean scores ──────────────────────────────────────────
    per_gene_df = _per_gene_scores(gene_seqs, scorer, method)

    # ── Metagene profiles ─────────────────────────────────────────────
    metagene_gs = _metagene_profile(gene_seqs, scorer, window, method)
    metagene_bg = _metagene_profile(all_seqs, scorer, window, method)

    # ── Ramp analysis ─────────────────────────────────────────────────
    ramp = _ramp_analysis(
        gene_seqs, all_seqs, scorer, ramp_codons, method,
    )

    # ── Ramp vs body codon composition ────────────────────────────────
    ramp_comp = _ramp_body_composition(
        gene_seqs, all_seqs, scorer, ramp_codons,
        n_bootstrap=n_bootstrap, seed=seed, method=method,
    )

    # ── Output ────────────────────────────────────────────────────────
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        _write_outputs(
            metagene_gs, metagene_bg, ramp, per_gene_df, out, method,
        )

    return {
        "metagene_geneset": metagene_gs,
        "metagene_genome": metagene_bg,
        "ramp_analysis": ramp,
        "ramp_composition": ramp_comp["ramp_composition"],
        "body_composition": ramp_comp["body_composition"],
        "per_gene_scores": per_gene_df,
        "scorer": scorer,
        "id_summary": id_result,
        "n_genes": n_genes,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Per-gene scores
# ═══════════════════════════════════════════════════════════════════════════════

def _per_gene_scores(
    sequences: dict[str, str],
    scorer: OptimalityScorer,
    method: str,
) -> pd.DataFrame:
    """Compute per-gene mean tAI and wtAI."""
    rows = []
    for gene, seq in sorted(sequences.items()):
        tai = scorer.gene_tai(seq)
        wtai = scorer.gene_wtai(seq)
        n_codons = len(seq) // 3
        rows.append({
            "gene": gene,
            "n_codons": n_codons,
            "tai": tai,
            "wtai": wtai,
        })
    df = pd.DataFrame(rows)
    sort_col = "wtai" if method == "wtai" else "tai"
    return df.sort_values(sort_col, ascending=False).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Metagene profile
# ═══════════════════════════════════════════════════════════════════════════════

def _metagene_profile(
    sequences: dict[str, str],
    scorer: OptimalityScorer,
    window: int,
    method: str,
    min_codons: int = 30,
) -> np.ndarray:
    """Average per-position optimality across genes, normalised to % CDS.

    Each gene is rescaled to ``N_METAGENE_BINS`` positional bins (0–99%).
    Genes shorter than *min_codons* are excluded.

    Returns:
        1-D float64 array of shape ``(N_METAGENE_BINS,)``.
    """
    accum = np.zeros(N_METAGENE_BINS, dtype=np.float64)
    count = 0

    for seq in sequences.values():
        n_codons = len(seq) // 3
        if n_codons < min_codons:
            continue

        raw = scorer.per_position_scores(seq, method=method)
        smoothed = scorer.smooth_profile(raw, window=window)

        # Resample to N_METAGENE_BINS via linear interpolation
        x_orig = np.linspace(0, 1, len(smoothed))
        x_bins = np.linspace(0, 1, N_METAGENE_BINS)
        resampled = np.interp(x_bins, x_orig, smoothed)

        accum += resampled
        count += 1

    if count == 0:
        return accum
    return accum / count


# ═══════════════════════════════════════════════════════════════════════════════
# Ramp analysis
# ═══════════════════════════════════════════════════════════════════════════════

def _ramp_analysis(
    gene_seqs: dict[str, str],
    all_seqs: dict[str, str],
    scorer: OptimalityScorer,
    ramp_codons: int,
    method: str,
) -> dict:
    """Compare mean optimality of the first *ramp_codons* to the CDS body.

    The "ramp" hypothesis: slow codons enriched near 5' end to prevent
    ribosome collisions further downstream.

    Returns dict with:
        geneset_ramp_mean:   mean optimality, first ramp_codons codons
        geneset_body_mean:   mean optimality, remainder of CDS
        geneset_ramp_delta:  body - ramp (positive = ramp is slower)
        genome_ramp_mean:    genome background ramp mean
        genome_body_mean:    genome background body mean
        genome_ramp_delta:   genome background delta
        per_gene_deltas:     1-D array of per-gene (body − ramp) values
    """
    gs_ramp, gs_body, gs_deltas = _ramp_scores(
        gene_seqs, scorer, ramp_codons, method,
    )
    bg_ramp, bg_body, _ = _ramp_scores(
        all_seqs, scorer, ramp_codons, method,
    )

    return {
        "geneset_ramp_mean": gs_ramp,
        "geneset_body_mean": gs_body,
        "geneset_ramp_delta": gs_body - gs_ramp,
        "genome_ramp_mean": bg_ramp,
        "genome_body_mean": bg_body,
        "genome_ramp_delta": bg_body - bg_ramp,
        "per_gene_deltas": gs_deltas,
        "ramp_codons": ramp_codons,
    }


def _ramp_scores(
    sequences: dict[str, str],
    scorer: OptimalityScorer,
    ramp_codons: int,
    method: str,
) -> tuple[float, float, np.ndarray]:
    """Return (mean_ramp, mean_body, per_gene_deltas)."""
    ramp_vals: list[float] = []
    body_vals: list[float] = []
    deltas: list[float] = []

    for seq in sequences.values():
        n_codons = len(seq) // 3
        if n_codons <= ramp_codons:
            continue

        scores = scorer.per_position_scores(seq, method=method)
        ramp_mean = float(scores[:ramp_codons].mean())
        body_mean = float(scores[ramp_codons:].mean())
        ramp_vals.append(ramp_mean)
        body_vals.append(body_mean)
        deltas.append(body_mean - ramp_mean)

    if not ramp_vals:
        return 0.0, 0.0, np.array([])

    return (
        float(np.mean(ramp_vals)),
        float(np.mean(body_vals)),
        np.array(deltas, dtype=np.float64),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Ramp vs body codon composition
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_region_seqs(
    sequences: dict[str, str],
    start_codon: int,
    end_codon: int | None,
) -> dict[str, str]:
    """Extract a codon region [start_codon, end_codon) from each sequence.

    Genes shorter than start_codon+1 codons are skipped.
    """
    result = {}
    for gene, seq in sequences.items():
        n_codons = len(seq) // 3
        if n_codons <= start_codon:
            continue
        start_nt = start_codon * 3
        end_nt = end_codon * 3 if end_codon is not None else len(seq)
        sub = seq[start_nt:end_nt]
        # Ensure divisible by 3
        remainder = len(sub) % 3
        if remainder:
            sub = sub[: len(sub) - remainder]
        if len(sub) >= 3:
            result[gene] = sub
    return result


def _region_composition(
    geneset_seqs: dict[str, str],
    background_seqs: dict[str, str],
    n_bootstrap: int,
    seed: int | None,
) -> pd.DataFrame:
    """Compute monocodon composition with bootstrap Z-scores for a region."""
    _, gs_mean, kmer_names = compute_geneset_frequencies(geneset_seqs, k=1)
    bg_per_gene, bg_mean, _ = compute_geneset_frequencies(background_seqs, k=1)

    n_genes = len(geneset_seqs)
    z_scores, boot_mean, boot_std = bootstrap_zscores(
        gs_mean, bg_per_gene, n_genes,
        n_bootstrap=n_bootstrap, seed=seed,
    )
    p_values = bootstrap_pvalues(z_scores)
    adj_p = benjamini_hochberg(p_values)

    df = pd.DataFrame({
        "codon": kmer_names,
        "freq_geneset": gs_mean,
        "freq_genome": boot_mean,
        "z_score": z_scores,
        "p_value": p_values,
        "adjusted_p": adj_p,
        "amino_acid": [CODON_TABLE.get(c, "?") for c in kmer_names],
    })
    return df.sort_values("z_score", key=np.abs, ascending=False).reset_index(
        drop=True
    )


def _ramp_body_composition(
    gene_seqs: dict[str, str],
    all_seqs: dict[str, str],
    scorer: OptimalityScorer,
    ramp_codons: int,
    n_bootstrap: int = 1_000,
    seed: int | None = None,
    method: str = "wtai",
) -> dict:
    """Compare codon composition of ramp vs body regions.

    Splits each gene into ramp (first *ramp_codons* codons) and body
    (remainder), computes monocodon frequencies for each region in
    gene set vs genome, and bootstrap Z-scores.

    Returns dict with:
        ramp_composition: DataFrame — per-codon frequencies and Z-scores
        body_composition: DataFrame — per-codon frequencies and Z-scores
    Each DataFrame has columns: codon, freq_geneset, freq_genome,
        z_score, p_value, adjusted_p, amino_acid, speed
    """
    # Extract ramp and body subsequences
    gs_ramp = _extract_region_seqs(gene_seqs, 0, ramp_codons)
    gs_body = _extract_region_seqs(gene_seqs, ramp_codons, None)
    bg_ramp = _extract_region_seqs(all_seqs, 0, ramp_codons)
    bg_body = _extract_region_seqs(all_seqs, ramp_codons, None)

    logger.info(
        "Ramp/body composition: %d/%d gene-set, %d/%d genome genes",
        len(gs_ramp), len(gs_body), len(bg_ramp), len(bg_body),
    )

    # Composition analysis per region
    ramp_df = _region_composition(gs_ramp, bg_ramp, n_bootstrap, seed)
    body_df = _region_composition(gs_body, bg_body, n_bootstrap, seed)

    # Annotate fast/slow from scorer
    fast, slow = scorer.classify_codons(method=method)
    for df in (ramp_df, body_df):
        df["speed"] = df["codon"].apply(
            lambda c: "fast" if c in fast else "slow"
        )

    return {
        "ramp_composition": ramp_df,
        "body_composition": body_df,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Output
# ═══════════════════════════════════════════════════════════════════════════════

def _write_outputs(
    metagene_gs: np.ndarray,
    metagene_bg: np.ndarray,
    ramp: dict,
    per_gene_df: pd.DataFrame,
    output_dir: Path,
    method: str,
) -> None:
    """Write TSV results and plots."""
    # Per-gene scores
    per_gene_df.to_csv(
        output_dir / "profile_per_gene.tsv", sep="\t", index=False,
        float_format="%.6g",
    )

    # Metagene TSV
    meta_df = pd.DataFrame({
        "position_pct": np.arange(N_METAGENE_BINS),
        "geneset": metagene_gs,
        "genome": metagene_bg,
    })
    meta_df.to_csv(
        output_dir / "profile_metagene.tsv", sep="\t", index=False,
        float_format="%.6g",
    )

    # Ramp summary
    ramp_df = pd.DataFrame([{
        "region": "ramp",
        "geneset_mean": ramp["geneset_ramp_mean"],
        "genome_mean": ramp["genome_ramp_mean"],
    }, {
        "region": "body",
        "geneset_mean": ramp["geneset_body_mean"],
        "genome_mean": ramp["genome_body_mean"],
    }])
    ramp_df.to_csv(
        output_dir / "profile_ramp.tsv", sep="\t", index=False,
        float_format="%.6g",
    )

    logger.info("Profile results written to %s", output_dir)

    # Plot
    try:
        _plot_metagene(metagene_gs, metagene_bg, ramp, method,
                       output_dir / "profile_metagene.png")
    except Exception as exc:
        logger.warning("Plot generation failed: %s", exc)


def _plot_metagene(
    metagene_gs: np.ndarray,
    metagene_bg: np.ndarray,
    ramp: dict,
    method: str,
    path: Path,
) -> None:
    """Metagene optimality plot: gene set vs genome."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(N_METAGENE_BINS)

    # Panel A: metagene profiles
    ax1.plot(x, metagene_gs, color="#d73027", linewidth=1.5, label="Gene set")
    ax1.plot(x, metagene_bg, color="#4575b4", linewidth=1.5,
             alpha=0.7, label="Genome")
    ax1.set_xlabel("CDS position (%)")
    ax1.set_ylabel(f"Mean {method.upper()}")
    ax1.set_title("(A) Metagene Optimality Profile")
    ax1.legend(fontsize=9)

    # Mark ramp region
    ramp_pct = int(ramp["ramp_codons"] / 3)  # rough %, varies by gene
    ax1.axvline(ramp_pct, color="grey", linestyle="--", alpha=0.5)
    ax1.text(ramp_pct + 1, ax1.get_ylim()[1] * 0.95, "ramp",
             fontsize=8, color="grey")

    # Panel B: ramp vs body bar chart
    labels = ["Ramp\n(gene set)", "Body\n(gene set)",
              "Ramp\n(genome)", "Body\n(genome)"]
    values = [
        ramp["geneset_ramp_mean"], ramp["geneset_body_mean"],
        ramp["genome_ramp_mean"], ramp["genome_body_mean"],
    ]
    colors = ["#d73027", "#d73027", "#4575b4", "#4575b4"]
    alphas = [0.6, 1.0, 0.6, 1.0]

    bars = ax2.bar(range(4), values, color=colors)
    for bar, a in zip(bars, alphas):
        bar.set_alpha(a)
    ax2.set_xticks(range(4))
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylabel(f"Mean {method.upper()}")
    ax2.set_title("(B) Ramp vs Body")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Metagene plot saved to %s", path)
