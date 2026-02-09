"""Reverse mode: find genes enriched for specific codons.

Given target codons, computes per-gene Z-scores relative to the genome
and returns a ranked gene list.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from codonscope.core.codons import CODON_TABLE, SENSE_CODONS, all_possible_kmers
from codonscope.core.statistics import compute_geneset_frequencies

logger = logging.getLogger(__name__)


def run_reverse(
    species: str,
    codons: list[str],
    top: int | None = None,
    zscore_cutoff: float | None = None,
    percentile: float | None = None,
    data_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    check: bool = False,
) -> dict:
    """Find genes enriched for specific codons.

    Args:
        species: Species name (e.g. "yeast", "human").
        codons: Target codons (e.g. ["AGA", "GAA"]).
        top: Return top N genes (default: all passing cutoff).
        zscore_cutoff: Minimum combined Z-score (default: 2.0 if no other filter).
        percentile: Minimum percentile rank (0-100).
        data_dir: Override data directory.
        output_dir: Directory to write output TSV.
        check: If True, run enrichment on result gene list.

    Returns:
        Dict with:
        - "gene_table": DataFrame with gene, combined_z, min_z, max_z, sum_freq,
          mean_freq, gene_name (common name), per-codon Z-scores
        - "n_genome_genes": Total genome genes
        - "target_codons": List of target codons used
        - "species": Species name
        - "check_result": Enrichment result if check=True, else None
    """
    from codonscope.core.sequences import SequenceDB

    # Validate codons
    valid_codons = set(SENSE_CODONS)
    codons_upper = [c.upper().replace("U", "T") for c in codons]
    invalid = [c for c in codons_upper if c not in valid_codons]
    if invalid:
        raise ValueError(f"Invalid codons: {invalid}. Must be sense codons (61 possible).")

    db = SequenceDB(species, data_dir=data_dir)
    all_seqs = db.get_all_sequences()

    # Compute per-gene frequency matrix for monocodons (k=1)
    per_gene_matrix, genome_mean, kmer_names = compute_geneset_frequencies(all_seqs, k=1)
    kmer_to_idx = {km: i for i, km in enumerate(kmer_names)}
    gene_names = sorted(all_seqs.keys())

    # Genome std per codon
    genome_std = per_gene_matrix.std(axis=0, ddof=1).astype(np.float64)

    # Per-gene Z-scores: (gene_freq - genome_mean) / genome_std
    z_matrix = (per_gene_matrix.astype(np.float64) - genome_mean) / np.where(genome_std > 0, genome_std, 1.0)

    # Get indices for target codons
    target_indices = [kmer_to_idx[c] for c in codons_upper]

    # Extract Z-scores and frequencies for target codons
    target_z = z_matrix[:, target_indices]  # (n_genes, n_targets)
    target_freq = per_gene_matrix[:, target_indices].astype(np.float64)

    # Combine multi-codon scores
    combined_z = target_z.mean(axis=1)  # mean Z across target codons
    min_z = target_z.min(axis=1)
    max_z = target_z.max(axis=1)
    sum_freq = target_freq.sum(axis=1)
    mean_freq = target_freq.mean(axis=1)

    # Sort by combined_z descending and apply filters early (before DataFrame)
    sorted_idx = np.argsort(-combined_z)

    if top is not None:
        # Only need top N — much faster than building full DataFrame
        keep_idx = sorted_idx[:top]
    elif zscore_cutoff is not None:
        keep_idx = sorted_idx[combined_z[sorted_idx] >= zscore_cutoff]
    elif percentile is not None:
        threshold = np.percentile(combined_z, percentile)
        keep_idx = sorted_idx[combined_z[sorted_idx] >= threshold]
    else:
        # Default: Z >= 2.0
        keep_idx = sorted_idx[combined_z[sorted_idx] >= 2.0]

    # Build DataFrame only for kept genes (vectorized)
    kept_genes = [gene_names[i] for i in keep_idx]
    data = {
        "gene": kept_genes,
        "combined_z": combined_z[keep_idx],
        "min_z": min_z[keep_idx],
        "max_z": max_z[keep_idx],
        "sum_freq": sum_freq[keep_idx],
        "mean_freq": mean_freq[keep_idx],
    }
    for j, codon in enumerate(codons_upper):
        data[f"z_{codon}"] = target_z[keep_idx, j]

    df = pd.DataFrame(data).reset_index(drop=True)

    # Add common names
    common_names = db.get_common_names(kept_genes)
    df["gene_name"] = df["gene"].map(lambda g: common_names.get(g, g))

    # Add amino acid info for target codons
    aa_info = ", ".join(f"{c} ({CODON_TABLE.get(c, '?')})" for c in codons_upper)

    # Output
    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        codons_str = "_".join(codons_upper)
        df.to_csv(out / f"reverse_{codons_str}.tsv", sep="\t", index=False, float_format="%.6g")

    # Optional check: run enrichment on result genes
    check_result = None
    if check and len(df) >= 10:
        try:
            from codonscope.modes.mode1_composition import run_composition
            check_genes = df["gene"].tolist()
            check_result = run_composition(
                species=species, gene_ids=check_genes, k=1,
                n_bootstrap=1000, seed=42, data_dir=data_dir,
            )
        except Exception as exc:
            logger.warning("Check enrichment failed: %s", exc)

    return {
        "gene_table": df,
        "n_genome_genes": len(gene_names),
        "target_codons": codons_upper,
        "codon_info": aa_info,
        "species": species,
        "check_result": check_result,
    }
