"""Mode 6: Cross-Species Comparison.

For a gene list in one species, map to orthologs in the other species.
For each ortholog pair, compute per-gene RSCU in both species and
calculate the Pearson correlation.

Compares the gene-set correlation distribution to genome-wide
ortholog correlation distribution.

For divergent genes (low correlation), tests whether the divergence
tracks tRNA pool differences — does each species use its own optimal
codon for the same amino acid?
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from codonscope.core.codons import SENSE_CODONS, sequence_to_codons
from codonscope.core.orthologs import OrthologDB
from codonscope.core.sequences import SequenceDB

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# Genetic code tables (shared with mode5_disentangle)
# ═══════════════════════════════════════════════════════════════════════════════

CODON_TABLE: dict[str, str] = {
    "TTT": "Phe", "TTC": "Phe",
    "TTA": "Leu", "TTG": "Leu", "CTT": "Leu", "CTC": "Leu",
    "CTA": "Leu", "CTG": "Leu",
    "ATT": "Ile", "ATC": "Ile", "ATA": "Ile",
    "ATG": "Met",
    "GTT": "Val", "GTC": "Val", "GTA": "Val", "GTG": "Val",
    "TCT": "Ser", "TCC": "Ser", "TCA": "Ser", "TCG": "Ser",
    "AGT": "Ser", "AGC": "Ser",
    "CCT": "Pro", "CCC": "Pro", "CCA": "Pro", "CCG": "Pro",
    "ACT": "Thr", "ACC": "Thr", "ACA": "Thr", "ACG": "Thr",
    "GCT": "Ala", "GCC": "Ala", "GCA": "Ala", "GCG": "Ala",
    "TAT": "Tyr", "TAC": "Tyr",
    "CAT": "His", "CAC": "His",
    "CAA": "Gln", "CAG": "Gln",
    "AAT": "Asn", "AAC": "Asn",
    "AAA": "Lys", "AAG": "Lys",
    "GAT": "Asp", "GAC": "Asp",
    "GAA": "Glu", "GAG": "Glu",
    "TGT": "Cys", "TGC": "Cys",
    "TGG": "Trp",
    "CGT": "Arg", "CGC": "Arg", "CGA": "Arg", "CGG": "Arg",
    "AGA": "Arg", "AGG": "Arg",
    "GGT": "Gly", "GGC": "Gly", "GGA": "Gly", "GGG": "Gly",
}

AMINO_ACIDS = sorted(set(CODON_TABLE.values()))

AA_FAMILIES: dict[str, list[str]] = {}
for _c, _a in CODON_TABLE.items():
    AA_FAMILIES.setdefault(_a, []).append(_c)
for _a in AA_FAMILIES:
    AA_FAMILIES[_a].sort()


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════

def run_compare(
    species1: str,
    species2: str,
    gene_ids: list[str],
    from_species: str | None = None,
    n_bootstrap: int = 10_000,
    seed: int | None = None,
    output_dir: str | Path | None = None,
    data_dir: str | Path | None = None,
) -> dict:
    """Run Mode 6: Cross-Species Comparison.

    Args:
        species1: First species (e.g. "yeast").
        species2: Second species (e.g. "human").
        gene_ids: Gene identifiers (in from_species).
        from_species: Which species the gene IDs belong to.
            Defaults to species1 if not specified.
        n_bootstrap: Bootstrap iterations for significance test.
        seed: Random seed.
        output_dir: Directory for output files. None = no file output.
        data_dir: Override default data directory.

    Returns:
        dict with keys:
            "per_gene": DataFrame with per-gene RSCU correlations
            "summary": dict with mean/median/std of gene-set vs genome
            "divergent_analysis": DataFrame with divergent-gene tRNA analysis
            "scatter_data": dict of {amino_acid: DataFrame} for scatter plots
            "n_orthologs": int
            "n_genome_orthologs": int
            "from_species": str
            "to_species": str
            "id_summary": dict
    """
    species1 = species1.lower()
    species2 = species2.lower()
    if from_species is None:
        from_species = species1
    from_species = from_species.lower()
    to_species = species2 if from_species == species1 else species1

    # Load sequence databases
    base = Path(data_dir) if data_dir else None
    db_from = SequenceDB(from_species, data_dir=base)
    db_to = SequenceDB(to_species, data_dir=base)

    # Load ortholog mapping
    ortho_db = OrthologDB(species1, species2, data_dir=base.parent if base else None)

    # Resolve gene IDs in the from-species
    id_mapping = db_from.resolve_ids(gene_ids)
    id_summary = {
        "n_input": len(gene_ids),
        "n_mapped": id_mapping.n_mapped,
        "n_unmapped": id_mapping.n_unmapped,
    }

    # Map resolved genes to orthologs
    from_sys_names = list(id_mapping.values())
    ortho_map = ortho_db.map_genes(from_sys_names, from_species=from_species)

    # Get sequences for both species
    from_genes_with_ortho = [g for g in from_sys_names if g in ortho_map]
    to_genes = [ortho_map[g] for g in from_genes_with_ortho]

    from_seqs = db_from.get_sequences(from_genes_with_ortho)
    to_seqs = db_to.get_sequences(to_genes)

    # Filter to pairs where both have sequences
    valid_pairs: list[tuple[str, str]] = []
    for fg in from_genes_with_ortho:
        tg = ortho_map[fg]
        if fg in from_seqs and tg in to_seqs:
            valid_pairs.append((fg, tg))

    if len(valid_pairs) == 0:
        raise ValueError(
            f"No ortholog pairs found with sequences in both species. "
            f"Input genes: {len(gene_ids)}, mapped: {id_mapping.n_mapped}, "
            f"with orthologs: {len(from_genes_with_ortho)}"
        )

    logger.info(
        "Gene set: %d input → %d mapped → %d with orthologs → %d with sequences",
        len(gene_ids), id_mapping.n_mapped,
        len(from_genes_with_ortho), len(valid_pairs),
    )

    id_summary["n_with_orthologs"] = len(from_genes_with_ortho)
    id_summary["n_valid_pairs"] = len(valid_pairs)

    # ── Compute per-gene RSCU correlations (gene set) ──────────────────────
    gs_corrs = _per_gene_rscu_correlations(valid_pairs, from_seqs, to_seqs)

    # ── Compute genome-wide ortholog correlations ─────────────────────────
    all_ortho_pairs = ortho_db.get_all_pairs()

    # Determine which column is from_species vs to_species
    # OrthologDB stores as (species1_id, species2_id)
    if from_species == species1:
        genome_from_genes = [p[0] for p in all_ortho_pairs]
        genome_to_genes = [p[1] for p in all_ortho_pairs]
    else:
        genome_from_genes = [p[1] for p in all_ortho_pairs]
        genome_to_genes = [p[0] for p in all_ortho_pairs]

    all_from_seqs = db_from.get_all_sequences()
    all_to_seqs = db_to.get_all_sequences()

    genome_pairs: list[tuple[str, str]] = []
    for fg, tg in zip(genome_from_genes, genome_to_genes):
        if fg in all_from_seqs and tg in all_to_seqs:
            genome_pairs.append((fg, tg))

    genome_corrs = _per_gene_rscu_correlations(
        genome_pairs, all_from_seqs, all_to_seqs
    )

    # ── Build per-gene results DataFrame ──────────────────────────────────
    per_gene_df = pd.DataFrame({
        f"{from_species}_gene": [p[0] for p in valid_pairs],
        f"{to_species}_gene": [p[1] for p in valid_pairs],
        "rscu_correlation": gs_corrs,
    }).sort_values("rscu_correlation", ascending=True).reset_index(drop=True)

    # ── Summary statistics ────────────────────────────────────────────────
    rng = np.random.default_rng(seed)
    gs_mean = np.nanmean(gs_corrs)
    gs_median = np.nanmedian(gs_corrs)
    gs_std = np.nanstd(gs_corrs, ddof=1) if len(gs_corrs) > 1 else 0.0
    bg_mean = np.nanmean(genome_corrs)
    bg_median = np.nanmedian(genome_corrs)
    bg_std = np.nanstd(genome_corrs, ddof=1) if len(genome_corrs) > 1 else 0.0

    # Bootstrap test: is the gene-set mean correlation different from genome?
    n_gs = len(gs_corrs)
    boot_means = np.empty(n_bootstrap)
    genome_corrs_arr = np.array(genome_corrs, dtype=np.float64)
    valid_genome = genome_corrs_arr[~np.isnan(genome_corrs_arr)]

    for i in range(n_bootstrap):
        idx = rng.choice(len(valid_genome), size=n_gs, replace=True)
        boot_means[i] = valid_genome[idx].mean()

    boot_se = boot_means.std(ddof=1)
    z_score = (gs_mean - bg_mean) / boot_se if boot_se > 0 else 0.0
    p_value = 2.0 * stats.norm.sf(abs(z_score))  # two-sided

    # Mann-Whitney U test (non-parametric)
    valid_gs = np.array(gs_corrs)[~np.isnan(gs_corrs)]
    if len(valid_gs) >= 3 and len(valid_genome) >= 3:
        mw_stat, mw_p = stats.mannwhitneyu(
            valid_gs, valid_genome, alternative="two-sided"
        )
    else:
        mw_stat, mw_p = np.nan, np.nan

    summary = {
        "geneset_mean_r": float(gs_mean),
        "geneset_median_r": float(gs_median),
        "geneset_std_r": float(gs_std),
        "genome_mean_r": float(bg_mean),
        "genome_median_r": float(bg_median),
        "genome_std_r": float(bg_std),
        "z_score": float(z_score),
        "p_value": float(p_value),
        "mannwhitney_stat": float(mw_stat),
        "mannwhitney_p": float(mw_p),
    }

    # ── Divergent gene analysis ───────────────────────────────────────────
    divergent_df = _analyse_divergent(
        per_gene_df, from_seqs, to_seqs,
        from_species, to_species,
    )

    # ── Scatter data for selected amino acids ─────────────────────────────
    scatter_data = _build_scatter_data(
        valid_pairs, from_seqs, to_seqs,
        from_species, to_species,
    )

    # ── Genome correlation distribution (for histogram overlay) ───────────
    genome_corr_df = pd.DataFrame({
        f"{from_species}_gene": [p[0] for p in genome_pairs],
        f"{to_species}_gene": [p[1] for p in genome_pairs],
        "rscu_correlation": genome_corrs,
    })

    # ── Output files ──────────────────────────────────────────────────────
    if output_dir:
        _write_outputs(
            output_dir, per_gene_df, genome_corr_df,
            divergent_df, summary, from_species, to_species,
        )

    return {
        "per_gene": per_gene_df,
        "genome_correlations": genome_corr_df,
        "summary": summary,
        "divergent_analysis": divergent_df,
        "scatter_data": scatter_data,
        "n_orthologs": len(valid_pairs),
        "n_genome_orthologs": len(genome_pairs),
        "from_species": from_species,
        "to_species": to_species,
        "id_summary": id_summary,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# RSCU computation
# ═══════════════════════════════════════════════════════════════════════════════

def _gene_rscu_vector(sequence: str) -> np.ndarray:
    """Compute RSCU vector (61 sense codons) for a single gene.

    Returns NaN-filled vector if gene has too few codons.
    """
    codon_to_idx = {c: i for i, c in enumerate(SENSE_CODONS)}
    codons = sequence_to_codons(sequence)
    if len(codons) < 10:
        return np.full(61, np.nan, dtype=np.float64)

    counts: dict[str, int] = {}
    for c in codons:
        if c in codon_to_idx:
            counts[c] = counts.get(c, 0) + 1

    rscu = np.zeros(61, dtype=np.float64)
    for aa, family in AA_FAMILIES.items():
        n_syn = len(family)
        total_aa = sum(counts.get(c, 0) for c in family)
        if total_aa == 0:
            for c in family:
                rscu[codon_to_idx[c]] = np.nan
            continue
        for c in family:
            rscu[codon_to_idx[c]] = n_syn * counts.get(c, 0) / total_aa

    return rscu


def _per_gene_rscu_correlations(
    pairs: list[tuple[str, str]],
    from_seqs: dict[str, str],
    to_seqs: dict[str, str],
) -> list[float]:
    """Compute Pearson r of RSCU vectors for each ortholog pair.

    Only uses codons from amino acids with >=2 synonyms (excludes Met, Trp).
    """
    # Indices of codons with >=2 synonyms
    multi_syn_idx = []
    for i, c in enumerate(SENSE_CODONS):
        aa = CODON_TABLE[c]
        if len(AA_FAMILIES[aa]) >= 2:
            multi_syn_idx.append(i)
    multi_syn_idx = np.array(multi_syn_idx)

    correlations: list[float] = []
    for fg, tg in pairs:
        rscu_from = _gene_rscu_vector(from_seqs[fg])
        rscu_to = _gene_rscu_vector(to_seqs[tg])

        # Use only multi-synonym codons
        v1 = rscu_from[multi_syn_idx]
        v2 = rscu_to[multi_syn_idx]

        # Remove NaN positions
        valid = ~np.isnan(v1) & ~np.isnan(v2)
        if valid.sum() < 5:
            correlations.append(np.nan)
            continue

        r, _ = stats.pearsonr(v1[valid], v2[valid])
        correlations.append(float(r))

    return correlations


# ═══════════════════════════════════════════════════════════════════════════════
# Divergent gene analysis
# ═══════════════════════════════════════════════════════════════════════════════

def _analyse_divergent(
    per_gene_df: pd.DataFrame,
    from_seqs: dict[str, str],
    to_seqs: dict[str, str],
    from_species: str,
    to_species: str,
) -> pd.DataFrame:
    """Analyse most divergent orthologs (lowest RSCU correlation).

    For each divergent pair, check whether each species uses its own
    preferred codon for each amino acid (tracking tRNA pool differences).
    """
    # Consider bottom quartile as divergent
    valid = per_gene_df.dropna(subset=["rscu_correlation"])
    if len(valid) < 4:
        return pd.DataFrame()

    q25 = valid["rscu_correlation"].quantile(0.25)
    divergent = valid[valid["rscu_correlation"] <= q25].copy()

    records = []
    from_col = f"{from_species}_gene"
    to_col = f"{to_species}_gene"

    for _, row in divergent.iterrows():
        fg = row[from_col]
        tg = row[to_col]

        rscu_f = _gene_rscu_vector(from_seqs[fg])
        rscu_t = _gene_rscu_vector(to_seqs[tg])

        # Count AA families where preferred codon differs
        n_diff_preferred = 0
        n_total_families = 0

        for aa, family in AA_FAMILIES.items():
            if len(family) < 2:
                continue

            idxs = [SENSE_CODONS.index(c) for c in family]
            f_vals = rscu_f[idxs]
            t_vals = rscu_t[idxs]

            if np.any(np.isnan(f_vals)) or np.any(np.isnan(t_vals)):
                continue

            n_total_families += 1
            if family[np.argmax(f_vals)] != family[np.argmax(t_vals)]:
                n_diff_preferred += 1

        frac_diff = n_diff_preferred / n_total_families if n_total_families > 0 else np.nan

        records.append({
            from_col: fg,
            to_col: tg,
            "rscu_correlation": row["rscu_correlation"],
            "n_different_preferred_codons": n_diff_preferred,
            "n_aa_families_compared": n_total_families,
            "fraction_different": frac_diff,
        })

    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════════════════════
# Scatter data
# ═══════════════════════════════════════════════════════════════════════════════

def _build_scatter_data(
    pairs: list[tuple[str, str]],
    from_seqs: dict[str, str],
    to_seqs: dict[str, str],
    from_species: str,
    to_species: str,
) -> dict[str, pd.DataFrame]:
    """Build scatter-plot data: species1 vs species2 RSCU per amino acid.

    Returns dict mapping amino acid (3-letter) to DataFrame with columns:
        from_gene, to_gene, codon, rscu_{from_species}, rscu_{to_species}
    Only includes AAs with >=2 synonymous codons.
    """
    scatter: dict[str, list[dict]] = {}

    for aa, family in AA_FAMILIES.items():
        if len(family) < 2:
            continue
        scatter[aa] = []

    for fg, tg in pairs:
        rscu_f = _gene_rscu_vector(from_seqs[fg])
        rscu_t = _gene_rscu_vector(to_seqs[tg])

        for aa, family in AA_FAMILIES.items():
            if len(family) < 2:
                continue
            for codon in family:
                idx = SENSE_CODONS.index(codon)
                if np.isnan(rscu_f[idx]) or np.isnan(rscu_t[idx]):
                    continue
                scatter[aa].append({
                    f"{from_species}_gene": fg,
                    f"{to_species}_gene": tg,
                    "codon": codon,
                    f"rscu_{from_species}": rscu_f[idx],
                    f"rscu_{to_species}": rscu_t[idx],
                })

    result = {}
    for aa, records in scatter.items():
        if records:
            result[aa] = pd.DataFrame(records)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Output
# ═══════════════════════════════════════════════════════════════════════════════

def _write_outputs(
    output_dir: str | Path,
    per_gene_df: pd.DataFrame,
    genome_corr_df: pd.DataFrame,
    divergent_df: pd.DataFrame,
    summary: dict,
    from_species: str,
    to_species: str,
) -> None:
    """Write Mode 6 output files."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    per_gene_df.to_csv(out / "compare_per_gene.tsv", sep="\t", index=False)
    genome_corr_df.to_csv(out / "compare_genome_correlations.tsv", sep="\t", index=False)

    if len(divergent_df) > 0:
        divergent_df.to_csv(out / "compare_divergent.tsv", sep="\t", index=False)

    # Summary
    with open(out / "compare_summary.txt", "w") as fh:
        fh.write(f"Cross-species comparison: {from_species} → {to_species}\n\n")
        for k, v in summary.items():
            fh.write(f"{k}: {v}\n")

    # Plot
    try:
        _plot_compare(
            per_gene_df, genome_corr_df, summary,
            from_species, to_species, out,
        )
    except Exception as exc:
        logger.warning("Plotting failed: %s", exc)


def _plot_compare(
    per_gene_df: pd.DataFrame,
    genome_corr_df: pd.DataFrame,
    summary: dict,
    from_species: str,
    to_species: str,
    output_dir: Path,
) -> None:
    """Two-panel plot: correlation histogram + divergent gene scatter."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Correlation histogram
    ax = axes[0]
    gs_r = per_gene_df["rscu_correlation"].dropna()
    bg_r = genome_corr_df["rscu_correlation"].dropna()

    bins = np.linspace(-1, 1, 41)
    ax.hist(bg_r, bins=bins, alpha=0.5, label="Genome orthologs", density=True,
            color="gray", edgecolor="white")
    ax.hist(gs_r, bins=bins, alpha=0.7, label="Gene set", density=True,
            color="steelblue", edgecolor="white")
    ax.axvline(summary["geneset_mean_r"], color="steelblue", ls="--",
               label=f"Gene set mean r={summary['geneset_mean_r']:.3f}")
    ax.axvline(summary["genome_mean_r"], color="gray", ls="--",
               label=f"Genome mean r={summary['genome_mean_r']:.3f}")
    ax.set_xlabel("Per-gene RSCU Pearson r")
    ax.set_ylabel("Density")
    ax.set_title(f"Cross-species RSCU correlation\n"
                 f"Z={summary['z_score']:.2f}, p={summary['p_value']:.2e}")
    ax.legend(fontsize=8)

    # Panel 2: Gene-set RSCU correlation ranked
    ax2 = axes[1]
    sorted_r = gs_r.sort_values().reset_index(drop=True)
    ax2.bar(range(len(sorted_r)), sorted_r, color="steelblue", width=1.0)
    ax2.axhline(summary["genome_mean_r"], color="gray", ls="--",
                label=f"Genome mean r={summary['genome_mean_r']:.3f}")
    ax2.set_xlabel("Ortholog pair (ranked)")
    ax2.set_ylabel("RSCU Pearson r")
    ax2.set_title("Per-gene cross-species RSCU correlation")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(output_dir / "compare_plot.png", dpi=150)
    plt.close(fig)
    logger.info("Plot saved to %s", output_dir / "compare_plot.png")
