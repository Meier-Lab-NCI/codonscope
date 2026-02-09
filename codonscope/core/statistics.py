"""Bootstrap Z-scores, multiple testing correction, and effect sizes.

Core statistical engine for comparing gene-set k-mer frequencies against
genome-wide backgrounds.  All bootstrap operations are vectorized with numpy.
"""

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from codonscope.core.codons import all_possible_kmers, kmer_frequencies

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Gene-set frequency computation
# ═══════════════════════════════════════════════════════════════════════════════

def compute_geneset_frequencies(
    sequences: dict[str, str],
    k: int = 1,
    trim_ramp: int = 0,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Compute per-gene k-mer frequency matrix and mean for a gene set.

    Each gene contributes equally (frequencies computed per-gene, then averaged).

    Args:
        sequences: {gene_name: cds_sequence} dict.
        k: k-mer size (1=mono, 2=di, 3=tri).
        trim_ramp: Number of leading codons to exclude (default 0).

    Returns:
        (per_gene_matrix, mean_vector, kmer_names)
        - per_gene_matrix: shape (n_genes, n_kmers), float32
        - mean_vector: shape (n_kmers,), float64
        - kmer_names: sorted list of k-mer strings
    """
    kmer_names = all_possible_kmers(k=k, sense_only=True)
    kmer_to_idx = {km: i for i, km in enumerate(kmer_names)}
    n_kmers = len(kmer_names)
    gene_names = sorted(sequences.keys())
    n_genes = len(gene_names)

    per_gene = np.zeros((n_genes, n_kmers), dtype=np.float32)

    for gi, gene in enumerate(gene_names):
        seq = sequences[gene]
        if trim_ramp > 0:
            # Skip first trim_ramp codons (trim_ramp * 3 nucleotides)
            skip_nt = trim_ramp * 3
            if skip_nt >= len(seq):
                continue
            seq = seq[skip_nt:]
            # Ensure still divisible by 3
            remainder = len(seq) % 3
            if remainder:
                seq = seq[: len(seq) - remainder]
        if len(seq) < 3 * k:
            continue
        freqs = kmer_frequencies(seq, k=k)
        for kmer, freq in freqs.items():
            if kmer in kmer_to_idx:
                per_gene[gi, kmer_to_idx[kmer]] = freq

    mean_vec = per_gene.mean(axis=0).astype(np.float64)
    return per_gene, mean_vec, kmer_names


# ═══════════════════════════════════════════════════════════════════════════════
# Bootstrap Z-scores
# ═══════════════════════════════════════════════════════════════════════════════

def bootstrap_zscores(
    geneset_mean: np.ndarray,
    background_per_gene: np.ndarray,
    n_genes: int,
    n_bootstrap: int = 10_000,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute bootstrap Z-scores comparing gene set to genome background.

    Resamples *n_genes* rows from *background_per_gene* (with replacement)
    *n_bootstrap* times, computes the mean of each resample, then:

        Z = (geneset_mean - bootstrap_mean) / bootstrap_SE

    Args:
        geneset_mean: shape (n_kmers,) — observed gene-set mean frequencies.
        background_per_gene: shape (N_genome, n_kmers) — per-gene frequency
            matrix for the whole genome (from background .npz).
        n_genes: number of genes in the user gene set (resample size).
        n_bootstrap: number of bootstrap resamples (default 10,000).
        seed: random seed for reproducibility.

    Returns:
        (z_scores, bootstrap_mean, bootstrap_std)
        All shape (n_kmers,).
    """
    rng = np.random.default_rng(seed)
    n_genome = background_per_gene.shape[0]

    # Vectorized bootstrap: draw all resample indices at once
    # Shape: (n_bootstrap, n_genes)
    indices = rng.integers(0, n_genome, size=(n_bootstrap, n_genes))

    # Compute resample means: for each bootstrap, mean over sampled genes
    # Use advanced indexing — process in chunks to limit memory
    chunk_size = min(1000, n_bootstrap)
    n_kmers = background_per_gene.shape[1]
    resample_means = np.zeros((n_bootstrap, n_kmers), dtype=np.float64)

    for start in range(0, n_bootstrap, chunk_size):
        end = min(start + chunk_size, n_bootstrap)
        chunk_idx = indices[start:end]  # (chunk, n_genes)
        # Gather: (chunk, n_genes, n_kmers) → mean → (chunk, n_kmers)
        sampled = background_per_gene[chunk_idx]  # (chunk, n_genes, n_kmers)
        resample_means[start:end] = sampled.mean(axis=1)

    bootstrap_mean = resample_means.mean(axis=0)
    bootstrap_std = resample_means.std(axis=0, ddof=1)

    # Avoid division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        z_scores = np.where(
            bootstrap_std > 0,
            (geneset_mean - bootstrap_mean) / bootstrap_std,
            0.0,
        )

    return z_scores, bootstrap_mean, bootstrap_std


def bootstrap_pvalues(z_scores: np.ndarray) -> np.ndarray:
    """Convert Z-scores to two-sided p-values (normal approximation).

    Args:
        z_scores: array of Z-scores.

    Returns:
        Array of two-sided p-values.
    """
    return 2.0 * stats.norm.sf(np.abs(z_scores))


# ═══════════════════════════════════════════════════════════════════════════════
# Multiple testing correction
# ═══════════════════════════════════════════════════════════════════════════════

def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Apply Benjamini-Hochberg FDR correction.

    Args:
        p_values: 1-D array of raw p-values.

    Returns:
        Array of adjusted p-values (same shape), capped at 1.0.
    """
    p = np.asarray(p_values, dtype=np.float64)
    n = len(p)
    if n == 0:
        return p.copy()

    # Sort p-values
    sorted_idx = np.argsort(p)
    sorted_p = p[sorted_idx]

    # BH adjustment: p_adj[i] = p[i] * n / (rank)
    # Then enforce monotonicity from the bottom up
    ranks = np.arange(1, n + 1)
    adjusted = sorted_p * n / ranks

    # Enforce monotonicity (step-up): walk backwards, take cumulative min
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)

    # Unsort
    result = np.empty(n, dtype=np.float64)
    result[sorted_idx] = adjusted
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Effect sizes
# ═══════════════════════════════════════════════════════════════════════════════

def cohens_d(
    geneset_mean: np.ndarray,
    background_mean: np.ndarray,
    background_std: np.ndarray,
) -> np.ndarray:
    """Compute Cohen's d effect size for each k-mer.

    d = (geneset_mean - background_mean) / background_std

    Args:
        geneset_mean: shape (n_kmers,).
        background_mean: shape (n_kmers,) — genome-wide mean per-gene frequencies.
        background_std: shape (n_kmers,) — genome-wide per-gene standard deviations.

    Returns:
        Array of Cohen's d values. Where background_std is 0, returns 0.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.where(
            background_std > 0,
            (geneset_mean - background_mean) / background_std,
            0.0,
        )
    return d


# ═══════════════════════════════════════════════════════════════════════════════
# Power warnings
# ═══════════════════════════════════════════════════════════════════════════════

def power_check(n_genes: int, k: int) -> list[str]:
    """Check if gene list is adequately powered for the chosen k-mer size.

    Args:
        n_genes: number of genes in the user gene set.
        k: k-mer size (1, 2, or 3).

    Returns:
        List of warning strings (empty if adequately powered).
    """
    warns: list[str] = []

    if n_genes < 10:
        warns.append(
            f"Gene list has only {n_genes} genes. Minimum recommended is 10."
        )

    if k == 2 and n_genes < 30:
        warns.append(
            f"Gene list has {n_genes} genes. Dicodon analysis (k=2) is "
            f"underpowered with fewer than 30 genes."
        )

    if k == 3 and n_genes < 100:
        warns.append(
            f"Gene list has {n_genes} genes. Tricodon analysis (k=3) is "
            f"underpowered with fewer than 100 genes."
        )

    return warns


# ═══════════════════════════════════════════════════════════════════════════════
# Diagnostic tests
# ═══════════════════════════════════════════════════════════════════════════════

def diagnostic_ks_tests(
    geneset_lengths: np.ndarray,
    geneset_gc: np.ndarray,
    background_lengths: np.ndarray,
    background_gc: np.ndarray,
) -> dict:
    """Run KS tests comparing gene-set length and GC to background.

    Args:
        geneset_lengths: CDS lengths for gene set.
        geneset_gc: GC content for gene set.
        background_lengths: CDS lengths for all genome genes.
        background_gc: GC content for all genome genes.

    Returns:
        dict with keys:
            "length_ks_stat", "length_p", "length_warning",
            "gc_ks_stat", "gc_p", "gc_warning"
    """
    len_stat, len_p = stats.ks_2samp(geneset_lengths, background_lengths)
    gc_stat, gc_p = stats.ks_2samp(geneset_gc, background_gc)

    result = {
        "length_ks_stat": float(len_stat),
        "length_p": float(len_p),
        "length_warning": len_p < 0.01,
        "gc_ks_stat": float(gc_stat),
        "gc_p": float(gc_p),
        "gc_warning": gc_p < 0.01,
    }

    if result["length_warning"]:
        logger.warning(
            "Gene-set CDS lengths differ significantly from background "
            "(KS p=%.2e). Consider --background matched.", len_p
        )
    if result["gc_warning"]:
        logger.warning(
            "Gene-set GC content differs significantly from background "
            "(KS p=%.2e). Consider --background matched.", gc_p
        )

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Full comparison pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def compare_to_background(
    gene_sequences: dict[str, str],
    background_npz_path: str | Path,
    k: int = 1,
    n_bootstrap: int = 10_000,
    trim_ramp: int = 0,
    seed: int | None = None,
) -> pd.DataFrame:
    """Full pipeline: frequencies → bootstrap Z-scores → BH correction.

    Args:
        gene_sequences: {gene_name: cds_sequence} for the user gene set.
        background_npz_path: path to background .npz file (background_mono.npz, etc.)
        k: k-mer size (must match the background file).
        n_bootstrap: bootstrap iterations.
        trim_ramp: codons to skip at 5' end.
        seed: random seed.

    Returns:
        DataFrame with columns:
            kmer, observed_freq, expected_freq, z_score, p_value,
            adjusted_p, cohens_d
        Sorted by absolute Z-score descending.
    """
    n_genes = len(gene_sequences)

    # Power check
    for warn in power_check(n_genes, k):
        logger.warning(warn)

    # Compute gene-set frequencies
    _, geneset_mean, kmer_names = compute_geneset_frequencies(
        gene_sequences, k=k, trim_ramp=trim_ramp
    )

    # Load background
    bg = np.load(background_npz_path)
    bg_mean = bg["mean"].astype(np.float64)
    bg_std = bg["std"].astype(np.float64)

    if "per_gene" in bg:
        bg_per_gene = bg["per_gene"]
    else:
        # Tricodon: no per-gene matrix.  Use analytic SE approximation.
        logger.info(
            "No per-gene matrix in background (tricodon). "
            "Using analytic SE = std / sqrt(n_genes)."
        )
        se = bg_std / np.sqrt(n_genes)
        with np.errstate(divide="ignore", invalid="ignore"):
            z_scores = np.where(
                se > 0, (geneset_mean - bg_mean) / se, 0.0
            )
        p_values = bootstrap_pvalues(z_scores)
        adj_p = benjamini_hochberg(p_values)
        d = cohens_d(geneset_mean, bg_mean, bg_std)

        df = pd.DataFrame({
            "kmer": kmer_names,
            "observed_freq": geneset_mean,
            "expected_freq": bg_mean,
            "z_score": z_scores,
            "p_value": p_values,
            "adjusted_p": adj_p,
            "cohens_d": d,
        })
        return df.sort_values("z_score", key=np.abs, ascending=False).reset_index(drop=True)

    # Bootstrap
    z_scores, boot_mean, boot_std = bootstrap_zscores(
        geneset_mean, bg_per_gene, n_genes,
        n_bootstrap=n_bootstrap, seed=seed,
    )
    p_values = bootstrap_pvalues(z_scores)
    adj_p = benjamini_hochberg(p_values)
    d = cohens_d(geneset_mean, bg_mean, bg_std)

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
# Binomial GLM (GC3-corrected)
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_gc3(sequence: str) -> float:
    """Compute GC content at third codon positions.

    Args:
        sequence: CDS nucleotide sequence (must be divisible by 3).

    Returns:
        Fraction of G+C at the third position of each codon (0.0–1.0).
    """
    if len(sequence) < 3:
        return 0.0
    n_codons = len(sequence) // 3
    gc_count = 0
    for i in range(n_codons):
        base = sequence[i * 3 + 2]  # third position
        if base in ("G", "C"):
            gc_count += 1
    return gc_count / n_codons if n_codons > 0 else 0.0


def binomial_glm_zscores(
    geneset_sequences: dict[str, str],
    all_sequences: dict[str, str],
    k: int = 1,
    trim_ramp: int = 0,
) -> pd.DataFrame:
    """Compute GC3-corrected binomial GLM Z-scores for monocodon composition.

    For each codon, fits: logit(p_i) = beta_0 + beta_1*gene_set_i + beta_2*gc3_i
    where p_i is the proportion of that codon within its synonymous family
    for gene i.

    Only supports k=1 (monocodon). Raises ValueError for k>1.

    Args:
        geneset_sequences: {gene_name: cds_sequence} for the user gene set.
        all_sequences: {gene_name: cds_sequence} for ALL genome genes.
        k: Must be 1. Raises ValueError otherwise.
        trim_ramp: Number of leading codons to exclude.

    Returns:
        DataFrame with columns: kmer, observed_freq, expected_freq,
        z_score, p_value, adjusted_p, cohens_d, amino_acid,
        gc3_beta, gc3_pvalue
    """
    if k != 1:
        raise ValueError(
            f"Binomial GLM only supports monocodon (k=1), got k={k}. "
            "Dicodons have no natural synonymous family grouping."
        )

    import statsmodels.api as sm

    from codonscope.core.codons import CODON_TABLE, SENSE_CODONS

    # Build AA → codon family mapping
    aa_to_codons: dict[str, list[str]] = {}
    for c, aa in CODON_TABLE.items():
        aa_to_codons.setdefault(aa, []).append(c)

    # Skip single-codon families
    multi_codon_aas = {aa: codons for aa, codons in aa_to_codons.items() if len(codons) > 1}

    geneset_names = set(geneset_sequences.keys())

    # Pre-compute per-gene: codon counts and GC3
    gene_names = sorted(all_sequences.keys())
    gene_codon_counts: dict[str, dict[str, int]] = {}
    gene_gc3: dict[str, float] = {}

    for gene in gene_names:
        seq = all_sequences[gene]
        if trim_ramp > 0:
            skip_nt = trim_ramp * 3
            if skip_nt >= len(seq):
                continue
            seq = seq[skip_nt:]
            remainder = len(seq) % 3
            if remainder:
                seq = seq[:len(seq) - remainder]
        if len(seq) < 3:
            continue
        gene_gc3[gene] = _compute_gc3(seq)
        counts: dict[str, int] = {}
        n_codons_gene = len(seq) // 3
        for i in range(n_codons_gene):
            c = seq[i * 3: i * 3 + 3]
            counts[c] = counts.get(c, 0) + 1
        gene_codon_counts[gene] = counts

    # Also compute gene-set frequencies (for observed_freq column)
    _, gs_mean, kmer_names = compute_geneset_frequencies(
        geneset_sequences, k=1, trim_ramp=trim_ramp,
    )
    kmer_to_idx = {km: i for i, km in enumerate(kmer_names)}

    # Background mean (for expected_freq and cohens_d)
    _, all_mean, _ = compute_geneset_frequencies(all_sequences, k=1, trim_ramp=trim_ramp)
    # Background std
    all_gene_names_sorted = sorted(all_sequences.keys())
    from codonscope.core.codons import kmer_frequencies as _kmer_freq
    n_kmers = len(kmer_names)
    per_gene_mat = np.zeros((len(all_gene_names_sorted), n_kmers), dtype=np.float32)
    for gi, gene in enumerate(all_gene_names_sorted):
        seq = all_sequences[gene]
        if trim_ramp > 0:
            skip_nt = trim_ramp * 3
            if skip_nt < len(seq):
                seq = seq[skip_nt:]
                remainder = len(seq) % 3
                if remainder:
                    seq = seq[:len(seq) - remainder]
        if len(seq) >= 3:
            freqs = _kmer_freq(seq, k=1)
            for km, f in freqs.items():
                idx = kmer_to_idx.get(km)
                if idx is not None:
                    per_gene_mat[gi, idx] = f
    bg_std = per_gene_mat.std(axis=0).astype(np.float64)

    valid_genes = [g for g in gene_names if g in gene_codon_counts]

    results = []

    for codon in SENSE_CODONS:
        aa = CODON_TABLE.get(codon)
        if aa is None or aa not in multi_codon_aas:
            continue
        family = multi_codon_aas[aa]

        # Build per-gene data for this codon
        k_vals = []  # count of focal codon
        n_vals = []  # total count of synonymous family
        gs_indicator = []
        gc3_vals = []

        for gene in valid_genes:
            counts = gene_codon_counts[gene]
            k_i = counts.get(codon, 0)
            n_i = sum(counts.get(c, 0) for c in family)
            if n_i == 0:
                continue
            k_vals.append(k_i)
            n_vals.append(n_i)
            gs_indicator.append(1.0 if gene in geneset_names else 0.0)
            gc3_vals.append(gene_gc3.get(gene, 0.5))

        if len(k_vals) < 10:
            continue

        k_arr = np.array(k_vals, dtype=np.float64)
        n_arr = np.array(n_vals, dtype=np.float64)
        gs_arr = np.array(gs_indicator, dtype=np.float64)
        gc3_arr = np.array(gc3_vals, dtype=np.float64)

        # Binomial GLM: endog = (successes, failures), exog = [1, gene_set, gc3]
        endog = np.column_stack([k_arr, n_arr - k_arr])
        exog = sm.add_constant(np.column_stack([gs_arr, gc3_arr]))

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = sm.GLM(endog, exog, family=sm.families.Binomial())
                fit = model.fit(disp=0)

            z_score = float(fit.tvalues[1])
            p_value = float(fit.pvalues[1])
            gc3_beta = float(fit.params[2])
            gc3_pvalue = float(fit.pvalues[2])
        except Exception:
            z_score = 0.0
            p_value = 1.0
            gc3_beta = 0.0
            gc3_pvalue = 1.0

        idx = kmer_to_idx.get(codon)
        obs_freq = float(gs_mean[idx]) if idx is not None else 0.0
        exp_freq = float(all_mean[idx]) if idx is not None else 0.0
        d = float((obs_freq - exp_freq) / bg_std[idx]) if idx is not None and bg_std[idx] > 0 else 0.0

        results.append({
            "kmer": codon,
            "observed_freq": obs_freq,
            "expected_freq": exp_freq,
            "z_score": z_score,
            "p_value": p_value,
            "cohens_d": d,
            "amino_acid": aa,
            "gc3_beta": gc3_beta,
            "gc3_pvalue": gc3_pvalue,
        })

    df = pd.DataFrame(results)
    if len(df) == 0:
        return df

    # BH correction
    df["adjusted_p"] = benjamini_hochberg(df["p_value"].values)

    return df.sort_values("z_score", key=np.abs, ascending=False).reset_index(drop=True)
