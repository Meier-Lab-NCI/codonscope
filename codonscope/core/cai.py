"""Codon Adaptation Index (CAI) — Sharp & Li 1987.

Computes reference codon weights from highly-expressed genes and
per-gene CAI scores (geometric mean of per-codon weights).
"""

import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from codonscope.core.codons import CODON_TABLE, SENSE_CODONS, sequence_to_codons
from codonscope.core.sequences import SequenceDB

logger = logging.getLogger(__name__)

# Amino acids with a single codon — uninformative for CAI
_SINGLE_CODON_AAS = {"Met", "Trp"}

# Build AA → codon family mapping
_AA_TO_CODONS: dict[str, list[str]] = defaultdict(list)
for _c, _aa in CODON_TABLE.items():
    _AA_TO_CODONS[_aa].append(_c)
_AA_TO_CODONS = dict(_AA_TO_CODONS)


def compute_reference_weights(
    species: str,
    data_dir: Path | None = None,
    top_fraction: float = 0.05,
) -> dict[str, float]:
    """Compute CAI reference weights from the most highly-expressed genes.

    Args:
        species: Species name (e.g. "yeast", "human", "mouse").
        data_dir: Override default data directory.
        top_fraction: Fraction of genes to use as reference set (default 5%).

    Returns:
        {codon: w_value} for all 61 sense codons.
        Within each synonymous family, max w = 1.0.
    """
    from codonscope.modes.mode2_demand import _load_expression

    db = SequenceDB(species, data_dir=data_dir)
    species_dir = db.species_dir

    # Load expression
    expression, _tissue, _tissues = _load_expression(species, species_dir)

    # Get all CDS sequences
    all_seqs = db.get_all_sequences()

    # Select top genes by TPM
    gene_tpm = []
    for gene in all_seqs:
        tpm = expression.get(gene, 0.0)
        gene_tpm.append((gene, tpm))
    gene_tpm.sort(key=lambda x: x[1], reverse=True)

    n_ref = max(10, int(len(gene_tpm) * top_fraction))
    ref_genes = [g for g, _ in gene_tpm[:n_ref]]
    logger.info("CAI reference set: %d genes (top %.1f%% by TPM)", n_ref, top_fraction * 100)

    # Count codons in reference set
    codon_counts: dict[str, int] = defaultdict(int)
    for gene in ref_genes:
        seq = all_seqs.get(gene)
        if seq is None:
            continue
        codons = sequence_to_codons(seq)
        for c in codons:
            if c in CODON_TABLE:
                codon_counts[c] += 1

    # Compute weights: w(codon) = freq_in_ref / max_freq_for_that_AA
    weights: dict[str, float] = {}
    for aa, codons in _AA_TO_CODONS.items():
        if aa in _SINGLE_CODON_AAS:
            for c in codons:
                weights[c] = 1.0
            continue
        counts = [codon_counts.get(c, 0) for c in codons]
        total = sum(counts)
        if total == 0:
            for c in codons:
                weights[c] = 1.0
            continue
        freqs = [cnt / total for cnt in counts]
        max_freq = max(freqs)
        if max_freq == 0:
            for c in codons:
                weights[c] = 1.0
            continue
        for c, freq in zip(codons, freqs):
            w = freq / max_freq
            # Pseudocount: avoid w=0 which makes geometric mean undefined
            weights[c] = max(w, 0.01)

    return weights


def compute_cai(
    gene_sequences: dict[str, str],
    weights: dict[str, float],
) -> dict[str, float]:
    """Compute per-gene CAI (geometric mean of reference weights).

    Args:
        gene_sequences: {gene_name: cds_sequence}.
        weights: {codon: w_value} from compute_reference_weights().

    Returns:
        {gene_name: cai_value}.
    """
    result: dict[str, float] = {}
    for gene, seq in gene_sequences.items():
        codons = sequence_to_codons(seq)
        log_sum = 0.0
        n = 0
        for c in codons:
            aa = CODON_TABLE.get(c)
            if aa is None or aa in _SINGLE_CODON_AAS:
                continue
            w = weights.get(c)
            if w is not None and w > 0:
                log_sum += np.log(w)
                n += 1
        if n == 0:
            result[gene] = 0.0
        else:
            result[gene] = float(np.exp(log_sum / n))
    return result


def cai_analysis(
    species: str,
    gene_ids: list[str],
    data_dir: Path | None = None,
    top_fraction: float = 0.05,
) -> dict:
    """Full CAI analysis for a gene set.

    Args:
        species: Species name.
        gene_ids: List of gene identifiers.
        data_dir: Override default data directory.
        top_fraction: Fraction of genes for reference set.

    Returns:
        dict with keys:
            per_gene: DataFrame (gene, cai)
            geneset_mean, geneset_median: float
            genome_per_gene: DataFrame (gene, cai)
            genome_mean, genome_median: float
            percentile_rank: float
            mann_whitney_u, mann_whitney_p: float
            reference_n_genes: int
            weights: dict
    """
    db = SequenceDB(species, data_dir=data_dir)

    # Compute reference weights
    weights = compute_reference_weights(species, data_dir=data_dir, top_fraction=top_fraction)

    # Resolve gene IDs and get sequences
    id_mapping = db.resolve_ids(gene_ids)
    sys_names = list(id_mapping.values())
    gene_seqs = db.get_sequences(sys_names)

    # Compute CAI for gene set
    geneset_cai = compute_cai(gene_seqs, weights)

    # Compute CAI for all genome genes
    all_seqs = db.get_all_sequences()
    genome_cai = compute_cai(all_seqs, weights)

    # Build DataFrames
    gs_df = pd.DataFrame([
        {"gene": g, "cai": v} for g, v in sorted(geneset_cai.items())
    ])
    genome_df = pd.DataFrame([
        {"gene": g, "cai": v} for g, v in sorted(genome_cai.items())
    ])

    gs_values = np.array(list(geneset_cai.values()))
    genome_values = np.array(list(genome_cai.values()))

    gs_mean = float(gs_values.mean()) if len(gs_values) > 0 else 0.0
    gs_median = float(np.median(gs_values)) if len(gs_values) > 0 else 0.0
    gen_mean = float(genome_values.mean()) if len(genome_values) > 0 else 0.0
    gen_median = float(np.median(genome_values)) if len(genome_values) > 0 else 0.0

    # Percentile rank of gene set mean within genome distribution
    percentile = float(np.mean(genome_values <= gs_mean) * 100) if len(genome_values) > 0 else 50.0

    # Mann-Whitney U test
    if len(gs_values) >= 2 and len(genome_values) >= 2:
        u_stat, u_p = stats.mannwhitneyu(gs_values, genome_values, alternative="two-sided")
    else:
        u_stat, u_p = 0.0, 1.0

    n_ref = max(10, int(len(all_seqs) * top_fraction))

    return {
        "per_gene": gs_df,
        "geneset_mean": gs_mean,
        "geneset_median": gs_median,
        "genome_per_gene": genome_df,
        "genome_mean": gen_mean,
        "genome_median": gen_median,
        "percentile_rank": percentile,
        "mann_whitney_u": float(u_stat),
        "mann_whitney_p": float(u_p),
        "reference_n_genes": n_ref,
        "weights": weights,
    }
