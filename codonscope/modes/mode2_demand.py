"""Mode 2: Translational Demand Analysis.

Which codons are most in demand when we account for how much each gene
is actually translated?  Weight each gene's codon frequencies by its
expression level (TPM) × number of codons, so highly-expressed genes
dominate the demand vector.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from codonscope.core.codons import (
    CODON_TABLE,
    SENSE_CODONS,
    all_possible_kmers,
    annotate_kmer,
    kmer_frequencies,
)
from codonscope.core.sequences import SequenceDB
from codonscope.core.statistics import (
    benjamini_hochberg,
    bootstrap_pvalues,
)

logger = logging.getLogger(__name__)


def run_demand(
    species: str,
    gene_ids: list[str],
    k: int = 1,
    tissue: str | None = None,
    cell_line: str | None = None,
    expression_file: str | Path | None = None,
    top_n: int | None = None,
    n_bootstrap: int = 10_000,
    seed: int | None = None,
    output_dir: str | Path | None = None,
    data_dir: str | Path | None = None,
) -> dict:
    """Run Mode 2 Translational Demand analysis.

    For each gene, weight = TPM × (CDS_length / 3).  The demand vector is
    the weighted sum of per-gene codon frequencies, normalised to proportions.
    Compare gene-set demand to whole-transcriptome demand using weighted
    bootstrap Z-scores.

    Args:
        species: Species name ("yeast" or "human").
        gene_ids: Gene identifiers (any format).
        k: k-mer size (1=mono, 2=di, 3=tri).
        tissue: GTEx tissue name for human (default: HEK293T proxy).
        cell_line: CCLE cell line for human (e.g. HEK293T, HeLa, K562).
        expression_file: Path to user-supplied expression TSV
            (columns: gene_id, tpm).
        top_n: Only use top-N expressed genes in background (None = all).
        n_bootstrap: Bootstrap iterations.
        seed: Random seed.
        output_dir: Directory for output files.
        data_dir: Override data directory.

    Returns:
        dict with keys:
            "results": DataFrame (kmer, demand_geneset, demand_genome,
                       z_score, p_value, adjusted_p)
            "top_genes": DataFrame of genes ranked by demand contribution
            "tissue": tissue name used (or "rich_media" for yeast)
            "available_tissues": list of tissue names (human only)
            "n_genes": int
            "id_summary": IDMapping
    """
    if k not in (1, 2, 3):
        raise ValueError(f"k must be 1, 2, or 3, got {k}")

    # Load database
    db = SequenceDB(species, data_dir=data_dir)
    species_dir = db.species_dir

    # Resolve gene IDs
    id_result = db.resolve_ids(gene_ids)
    sys_names = list(id_result.values())
    if not sys_names:
        raise ValueError("No gene IDs could be mapped.")

    # Get sequences for gene set
    gene_seqs = db.get_sequences(sys_names)
    n_genes = len(gene_seqs)
    logger.info("Mapped %d genes for demand analysis", n_genes)

    # Load expression data
    expression, tissue_used, available_tissues = _load_expression(
        species, species_dir, tissue=tissue, cell_line=cell_line,
        expression_file=expression_file,
    )

    # Get all genome sequences for background
    all_seqs = db.get_all_sequences()

    # Compute demand vectors
    kmer_names = all_possible_kmers(k=k, sense_only=True)

    gs_demand, gs_weights = _compute_demand_vector(
        gene_seqs, expression, kmer_names, k=k,
    )
    genome_demand, genome_weights = _compute_demand_vector(
        all_seqs, expression, kmer_names, k=k, top_n=top_n,
    )

    # Weighted bootstrap Z-scores
    z_scores, boot_mean, boot_std = _weighted_bootstrap_zscores(
        gs_demand,
        all_seqs,
        expression,
        n_genes=n_genes,
        kmer_names=kmer_names,
        k=k,
        top_n=top_n,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )

    p_values = bootstrap_pvalues(z_scores)
    adj_p = benjamini_hochberg(p_values)

    results_df = pd.DataFrame({
        "kmer": kmer_names,
        "demand_geneset": gs_demand,
        "demand_genome": genome_demand,
        "z_score": z_scores,
        "p_value": p_values,
        "adjusted_p": adj_p,
    })
    results_df = results_df.sort_values(
        "z_score", key=np.abs, ascending=False,
    ).reset_index(drop=True)

    # Add amino acid annotation column
    def _kmer_to_aa(kmer: str) -> str:
        codons = [kmer[i:i + 3] for i in range(0, len(kmer), 3)]
        return "-".join(CODON_TABLE.get(c, "?") for c in codons)

    results_df["amino_acid"] = results_df["kmer"].apply(_kmer_to_aa)

    # Build systematic_name → common_name mapping for gene display
    gene_name_map = _load_gene_name_map(species_dir)

    # Top demand-contributing genes
    top_genes = _rank_demand_genes(
        gene_seqs, expression, kmer_names, k=k,
        gene_name_map=gene_name_map,
    )

    # Write outputs
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        _write_outputs(results_df, top_genes, out, k, tissue_used)

    return {
        "results": results_df,
        "top_genes": top_genes,
        "tissue": tissue_used,
        "available_tissues": available_tissues,
        "n_genes": n_genes,
        "id_summary": id_result,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Expression loading
# ═══════════════════════════════════════════════════════════════════════════════

def _load_expression(
    species: str,
    species_dir: Path,
    tissue: str | None = None,
    cell_line: str | None = None,
    expression_file: str | Path | None = None,
) -> tuple[dict[str, float], str, list[str]]:
    """Load expression data (TPM per gene).

    Returns:
        (expression_dict, tissue_name, available_tissues)
        expression_dict: {systematic_name: tpm}
    """
    if expression_file is not None:
        return _load_custom_expression(expression_file, species_dir, species)

    if species == "yeast":
        if cell_line:
            raise ValueError("Cell line expression not available for yeast.")
        return _load_yeast_expression(species_dir)
    elif species == "human":
        if cell_line is not None:
            return _load_ccle_expression(species_dir, cell_line)
        if tissue is not None:
            return _load_human_expression(species_dir, tissue=tissue)
        # Default: HEK293T (proxy: Kidney - Cortex)
        return _load_ccle_expression(species_dir, "HEK293T")
    else:
        raise ValueError(f"No expression data for species: {species!r}")


def _load_yeast_expression(
    species_dir: Path,
) -> tuple[dict[str, float], str, list[str]]:
    """Load yeast rich-media expression estimates."""
    expr_path = species_dir / "expression_rich_media.tsv"
    if not expr_path.exists():
        raise FileNotFoundError(
            f"Expression data not found: {expr_path}. "
            f"Run: codonscope download --species yeast"
        )

    df = pd.read_csv(expr_path, sep="\t")
    expr = dict(zip(df["systematic_name"], df["tpm"]))
    return expr, "rich_media", ["rich_media"]


def _load_human_expression(
    species_dir: Path,
    tissue: str | None = None,
) -> tuple[dict[str, float], str, list[str]]:
    """Load human GTEx expression data for a specific tissue."""
    expr_path = species_dir / "expression_gtex.tsv.gz"
    if not expr_path.exists():
        raise FileNotFoundError(
            f"GTEx expression data not found: {expr_path}. "
            f"Run: codonscope download --species human"
        )

    df = pd.read_csv(expr_path, sep="\t", compression="gzip")
    tissue_cols = [c for c in df.columns if c not in ("ensembl_gene", "symbol")]
    available = sorted(tissue_cols)

    if tissue is None:
        # Compute cross-tissue median as general-purpose background
        median_tpm = df[tissue_cols].median(axis=1)
        expr = dict(zip(df["ensembl_gene"], median_tpm.astype(float)))
        logger.info("No tissue specified, using cross-tissue median TPM.")
        return expr, "cross_tissue_median", available

    # Fuzzy match tissue name (case-insensitive substring)
    tissue_lower = tissue.lower().replace("_", " ").replace("-", " ")
    matched = None
    for t in tissue_cols:
        t_lower = t.lower().replace("_", " ").replace("-", " ")
        if tissue_lower == t_lower:
            matched = t
            break
    if matched is None:
        # Try substring match
        for t in tissue_cols:
            t_lower = t.lower().replace("_", " ").replace("-", " ")
            if tissue_lower in t_lower:
                matched = t
                break

    if matched is None:
        raise ValueError(
            f"Tissue '{tissue}' not found in GTEx data. "
            f"Available tissues ({len(available)}): "
            f"{', '.join(available[:10])}..."
        )

    logger.info("Using GTEx tissue: %s", matched)
    expr = dict(zip(df["ensembl_gene"], df[matched].astype(float)))
    return expr, matched, available


def _normalize_cell_line_name(name: str) -> str:
    """Normalize cell line name for fuzzy matching.

    Strips spaces, hyphens, parentheses, colons; uppercases.
    """
    return (
        name.upper()
        .replace(" ", "")
        .replace("-", "")
        .replace("(", "")
        .replace(")", "")
        .replace(":", "")
    )


def _load_ccle_expression(
    species_dir: Path,
    cell_line: str,
) -> tuple[dict[str, float], str, list[str]]:
    """Load CCLE cell line expression data.

    Tries loading from expression_ccle.tsv.gz first. If not available,
    falls back to GTEx tissue proxy using CELL_LINE_TISSUE_PROXY mapping.
    """
    from codonscope.data.download import CELL_LINE_TISSUE_PROXY

    ccle_path = species_dir / "expression_ccle.tsv.gz"
    norm_query = _normalize_cell_line_name(cell_line)

    if ccle_path.exists():
        df = pd.read_csv(ccle_path, sep="\t", compression="gzip")
        data_cols = [c for c in df.columns if c not in ("ensembl_gene", "symbol")]

        # Build normalized name → actual column name mapping
        norm_to_col = {}
        for c in data_cols:
            norm_to_col[_normalize_cell_line_name(c)] = c

        # Exact match on normalized name
        matched = norm_to_col.get(norm_query)

        # Substring match
        if matched is None:
            for norm_name, col_name in norm_to_col.items():
                if norm_query in norm_name or norm_name in norm_query:
                    matched = col_name
                    break

        if matched is not None:
            expr = dict(zip(df["ensembl_gene"], df[matched].astype(float)))
            logger.info("Using CCLE expression for cell line: %s", matched)
            return expr, matched, sorted(data_cols)

        # Cell line not found in CCLE data — try GTEx proxy fallback
        logger.warning(
            "Cell line '%s' not found in CCLE data. Checking GTEx proxy...",
            cell_line,
        )

    # Fall back to GTEx tissue proxy
    proxy_tissue = None
    for proxy_name, tissue in CELL_LINE_TISSUE_PROXY.items():
        if _normalize_cell_line_name(proxy_name) == norm_query:
            proxy_tissue = tissue
            break
    # Substring match on proxy names
    if proxy_tissue is None:
        for proxy_name, tissue in CELL_LINE_TISSUE_PROXY.items():
            norm_proxy = _normalize_cell_line_name(proxy_name)
            if norm_query in norm_proxy or norm_proxy in norm_query:
                proxy_tissue = tissue
                break

    if proxy_tissue is not None:
        logger.warning(
            "CCLE data not available. Using GTEx '%s' as proxy for '%s'.",
            proxy_tissue, cell_line,
        )
        expr, _, available = _load_human_expression(
            species_dir, tissue=proxy_tissue,
        )
        tissue_used = f"{cell_line} (proxy: {proxy_tissue})"
        return expr, tissue_used, available

    # Cell line not found anywhere
    available_names = sorted(CELL_LINE_TISSUE_PROXY.keys())
    if ccle_path.exists():
        df = pd.read_csv(ccle_path, sep="\t", compression="gzip", nrows=0)
        ccle_cols = [c for c in df.columns if c not in ("ensembl_gene", "symbol")]
        available_names = sorted(set(available_names) | set(ccle_cols))

    raise ValueError(
        f"Cell line '{cell_line}' not found. "
        f"Available cell lines ({len(available_names)}): "
        f"{', '.join(available_names[:20])}..."
    )


def _load_custom_expression(
    expression_file: str | Path,
    species_dir: Path,
    species: str,
) -> tuple[dict[str, float], str, list[str]]:
    """Load user-supplied expression file.

    Expected format: TSV with columns gene_id and tpm (at minimum).
    gene_id can be any identifier supported by the species.
    """
    path = Path(expression_file)
    if not path.exists():
        raise FileNotFoundError(f"Expression file not found: {path}")

    df = pd.read_csv(path, sep="\t")
    if "tpm" not in df.columns:
        raise ValueError(
            f"Expression file must have a 'tpm' column. "
            f"Found: {list(df.columns)}"
        )

    # Find gene ID column
    id_col = None
    for col in ("gene_id", "systematic_name", "ensembl_gene", "symbol", "gene"):
        if col in df.columns:
            id_col = col
            break
    if id_col is None:
        id_col = df.columns[0]

    # Map to systematic names using SequenceDB
    from codonscope.core.sequences import SequenceDB
    db = SequenceDB(species, data_dir=species_dir.parent)
    gene_ids = df[id_col].tolist()
    mapping = db.resolve_ids(gene_ids)

    expr = {}
    for _, row in df.iterrows():
        gene = row[id_col]
        if gene in mapping:
            expr[mapping[gene]] = float(row["tpm"])

    logger.info(
        "Loaded custom expression: %d/%d genes mapped",
        len(expr), len(df),
    )
    return expr, "custom", ["custom"]


# ═══════════════════════════════════════════════════════════════════════════════
# Demand computation
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_demand_vector(
    sequences: dict[str, str],
    expression: dict[str, float],
    kmer_names: list[str],
    k: int = 1,
    top_n: int | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Compute demand-weighted codon frequency vector.

    For each gene: weight = TPM × (n_codons).
    Demand = sum(weight_i × freq_i) / sum(weight_i).

    Args:
        sequences: {gene: cds_sequence}
        expression: {gene: tpm}
        kmer_names: sorted list of k-mer names
        k: k-mer size
        top_n: if set, only use top-N expressed genes

    Returns:
        (demand_vector, gene_weights)
        demand_vector: shape (n_kmers,) — normalised demand proportions
        gene_weights: {gene: weight} for genes that contributed
    """
    kmer_to_idx = {km: i for i, km in enumerate(kmer_names)}
    n_kmers = len(kmer_names)

    # Build gene list with weights
    gene_weight_list = []
    for gene, seq in sequences.items():
        tpm = expression.get(gene, 0.0)
        if tpm <= 0:
            continue
        n_codons = len(seq) // 3
        if n_codons < k:
            continue
        weight = tpm * n_codons
        gene_weight_list.append((gene, seq, weight))

    # Apply top_n filter
    if top_n is not None and len(gene_weight_list) > top_n:
        gene_weight_list.sort(key=lambda x: x[2], reverse=True)
        gene_weight_list = gene_weight_list[:top_n]

    if not gene_weight_list:
        return np.zeros(n_kmers, dtype=np.float64), {}

    # Compute weighted demand
    demand = np.zeros(n_kmers, dtype=np.float64)
    total_weight = 0.0
    gene_weights = {}

    for gene, seq, weight in gene_weight_list:
        freqs = kmer_frequencies(seq, k=k)
        for kmer, freq in freqs.items():
            if kmer in kmer_to_idx:
                demand[kmer_to_idx[kmer]] += weight * freq
        total_weight += weight
        gene_weights[gene] = weight

    if total_weight > 0:
        demand /= total_weight

    return demand, gene_weights


def _weighted_bootstrap_zscores(
    geneset_demand: np.ndarray,
    all_seqs: dict[str, str],
    expression: dict[str, float],
    n_genes: int,
    kmer_names: list[str],
    k: int = 1,
    top_n: int | None = None,
    n_bootstrap: int = 10_000,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Weighted bootstrap Z-scores for demand analysis.

    For each bootstrap iteration: sample n_genes from the genome,
    compute their demand-weighted frequency vector, collect the
    distribution of means, then compute Z = (observed - mean) / SE.

    To avoid recomputing per-gene frequencies each iteration, we
    pre-compute all per-gene frequency vectors and weights, then
    resample from those.
    """
    rng = np.random.default_rng(seed)
    n_kmers = len(kmer_names)
    kmer_to_idx = {km: i for i, km in enumerate(kmer_names)}

    # Pre-compute per-gene frequency vectors and weights
    gene_names = []
    freq_matrix = []
    weight_vec = []

    for gene, seq in all_seqs.items():
        tpm = expression.get(gene, 0.0)
        if tpm <= 0:
            continue
        n_codons = len(seq) // 3
        if n_codons < k:
            continue
        weight = tpm * n_codons

        freqs = kmer_frequencies(seq, k=k)
        row = np.zeros(n_kmers, dtype=np.float32)
        for kmer, freq in freqs.items():
            if kmer in kmer_to_idx:
                row[kmer_to_idx[kmer]] = freq

        gene_names.append(gene)
        freq_matrix.append(row)
        weight_vec.append(weight)

    freq_arr = np.array(freq_matrix, dtype=np.float32)  # (N_genome, n_kmers)
    weight_arr = np.array(weight_vec, dtype=np.float64)  # (N_genome,)
    n_genome = len(gene_names)

    if n_genome == 0:
        return (
            np.zeros(n_kmers),
            np.zeros(n_kmers),
            np.ones(n_kmers),
        )

    # Apply top_n: only sample from top-N expressed genes
    if top_n is not None and n_genome > top_n:
        top_idx = np.argsort(weight_arr)[::-1][:top_n]
        freq_arr = freq_arr[top_idx]
        weight_arr = weight_arr[top_idx]
        n_genome = top_n

    # Bootstrap: sample n_genes, compute weighted demand each time
    chunk_size = min(500, n_bootstrap)
    resample_demands = np.zeros((n_bootstrap, n_kmers), dtype=np.float64)

    for start in range(0, n_bootstrap, chunk_size):
        end = min(start + chunk_size, n_bootstrap)
        n_chunk = end - start

        # Draw random gene indices: (n_chunk, n_genes)
        indices = rng.integers(0, n_genome, size=(n_chunk, n_genes))

        # For each bootstrap sample, compute weighted demand
        sampled_freqs = freq_arr[indices]   # (n_chunk, n_genes, n_kmers)
        sampled_weights = weight_arr[indices]  # (n_chunk, n_genes)

        # Weighted mean: sum(w * f) / sum(w) for each bootstrap × kmer
        w_expanded = sampled_weights[:, :, np.newaxis]  # (n_chunk, n_genes, 1)
        weighted = sampled_freqs * w_expanded  # (n_chunk, n_genes, n_kmers)
        sum_weighted = weighted.sum(axis=1)    # (n_chunk, n_kmers)
        sum_w = sampled_weights.sum(axis=1, keepdims=True)  # (n_chunk, 1)

        with np.errstate(divide="ignore", invalid="ignore"):
            resample_demands[start:end] = np.where(
                sum_w > 0,
                sum_weighted / sum_w,
                0.0,
            )

    boot_mean = resample_demands.mean(axis=0)
    boot_std = resample_demands.std(axis=0, ddof=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        z_scores = np.where(
            boot_std > 0,
            (geneset_demand - boot_mean) / boot_std,
            0.0,
        )

    return z_scores, boot_mean, boot_std


# ═══════════════════════════════════════════════════════════════════════════════
# Gene name mapping
# ═══════════════════════════════════════════════════════════════════════════════

def _load_gene_name_map(species_dir: Path) -> dict[str, str]:
    """Load systematic_name → common_name mapping from gene_id_map.tsv."""
    map_path = species_dir / "gene_id_map.tsv"
    if not map_path.exists():
        return {}
    df = pd.read_csv(map_path, sep="\t")
    if "systematic_name" not in df.columns or "common_name" not in df.columns:
        return {}
    mapping = {}
    for _, row in df.iterrows():
        sys_name = str(row["systematic_name"])
        common = str(row.get("common_name", ""))
        if common and common != "nan" and common != sys_name:
            mapping[sys_name] = common
    return mapping


# ═══════════════════════════════════════════════════════════════════════════════
# Demand gene ranking
# ═══════════════════════════════════════════════════════════════════════════════

def _rank_demand_genes(
    gene_seqs: dict[str, str],
    expression: dict[str, float],
    kmer_names: list[str],
    k: int = 1,
    gene_name_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Rank genes in the set by their contribution to total demand.

    Returns DataFrame with columns:
        gene, gene_name, tpm, n_codons, demand_weight, demand_fraction
    Sorted by demand_weight descending.
    """
    rows = []
    total_weight = 0.0

    for gene, seq in gene_seqs.items():
        tpm = expression.get(gene, 0.0)
        n_codons = len(seq) // 3
        weight = tpm * n_codons
        total_weight += weight
        gene_name = gene
        if gene_name_map:
            gene_name = gene_name_map.get(gene, gene)
        rows.append({
            "gene": gene,
            "gene_name": gene_name,
            "tpm": tpm,
            "n_codons": n_codons,
            "demand_weight": weight,
        })

    df = pd.DataFrame(rows)
    if total_weight > 0:
        df["demand_fraction"] = df["demand_weight"] / total_weight
    else:
        df["demand_fraction"] = 0.0

    return df.sort_values("demand_weight", ascending=False).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Output
# ═══════════════════════════════════════════════════════════════════════════════

def _write_outputs(
    results_df: pd.DataFrame,
    top_genes: pd.DataFrame,
    output_dir: Path,
    k: int,
    tissue: str,
) -> None:
    """Write TSV results and plots."""
    k_label = {1: "mono", 2: "di", 3: "tri"}[k]

    # Results table
    tsv_path = output_dir / f"demand_{k_label}codons_{tissue}.tsv"
    results_df.to_csv(tsv_path, sep="\t", index=False, float_format="%.6g")
    logger.info("Demand results written to %s", tsv_path)

    # Top genes table
    genes_path = output_dir / f"demand_top_genes_{tissue}.tsv"
    top_genes.to_csv(genes_path, sep="\t", index=False, float_format="%.4g")
    logger.info("Top demand genes written to %s", genes_path)

    # Plot
    try:
        _plot_demand(results_df, output_dir / f"demand_{k_label}_{tissue}.png", k)
    except Exception as exc:
        logger.warning("Plot generation failed: %s", exc)


def _plot_demand(df: pd.DataFrame, path: Path, k: int) -> None:
    """Bar chart of top demand-enriched and demand-depleted k-mers."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sig = df[df["adjusted_p"] < 0.05].copy()
    if len(sig) == 0:
        sig = df.head(20).copy()

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
    ax.set_xlabel("Z-score (demand-weighted)")

    k_label = {1: "Monocodon", 2: "Dicodon", 3: "Tricodon"}[k]
    ax.set_title(f"Translational Demand: Top {k_label}s")
    ax.axvline(0, color="black", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Demand plot saved to %s", path)
