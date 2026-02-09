"""Deep-dive analyses: driver gene identification, positional enrichment, cluster scanning.

These analyses build on Tier 1 report output, providing detailed follow-up
for significant findings.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_tier1_data(results_dir: str | Path) -> dict:
    """Load pre-computed Tier 1 analysis data.

    Args:
        results_dir: Path to the {stem}_data/ directory from generate_report().

    Returns:
        Dict with keys: 'enrichment', 'per_gene_freq', 'config', 'gene_metadata'
    """
    results_dir = Path(results_dir)

    data = {}

    # Enrichment results
    enrich_path = results_dir / "mode1_monocodon.tsv"
    if enrich_path.exists():
        data["enrichment"] = pd.read_csv(enrich_path, sep="\t")
    else:
        raise FileNotFoundError(f"Enrichment results not found: {enrich_path}")

    # Per-gene frequency matrix
    freq_path = results_dir / "per_gene_codon_freq.tsv"
    if freq_path.exists():
        data["per_gene_freq"] = pd.read_csv(freq_path, sep="\t", index_col=0)
    else:
        raise FileNotFoundError(f"Per-gene freq matrix not found: {freq_path}")

    # Config
    config_path = results_dir / "analysis_config.json"
    if config_path.exists():
        data["config"] = json.loads(config_path.read_text())
    else:
        data["config"] = {}

    # Gene metadata
    meta_path = results_dir / "gene_metadata.tsv"
    if meta_path.exists():
        data["gene_metadata"] = pd.read_csv(meta_path, sep="\t")
    else:
        data["gene_metadata"] = None

    return data


def run_driver_analysis(
    results_dir: str | Path,
    adj_p_threshold: float = 0.05,
    output_dir: str | Path | None = None,
) -> dict:
    """Identify driver genes for each significant codon.

    For each codon with adj_p < threshold:
    1. Compute per-gene contribution (gene_freq - genome_mean)
    2. Cumulative contribution curve with N50/N80
    3. Gini coefficient on absolute contributions
    4. Jackknife analysis to flag influential genes

    Args:
        results_dir: Path to Tier 1 data directory.
        adj_p_threshold: Significance threshold (default 0.05).
        output_dir: Directory for output files.

    Returns:
        Dict with:
        - 'driver_tables': {codon: DataFrame} for each significant codon
        - 'summary': DataFrame with per-codon Gini, N50, N80
        - 'significant_codons': list of significant codon strings
        - 'n_genes': number of genes in the set
    """
    tier1 = load_tier1_data(results_dir)
    enrichment = tier1["enrichment"]
    per_gene_freq = tier1["per_gene_freq"]
    config = tier1["config"]

    # Get significant codons
    sig = enrichment[enrichment["adjusted_p"] < adj_p_threshold].copy()
    sig_codons = sig["kmer"].tolist()

    if len(sig_codons) == 0:
        return {
            "driver_tables": {},
            "summary": pd.DataFrame(),
            "significant_codons": [],
            "n_genes": len(per_gene_freq),
        }

    # Load genome background for expected frequencies
    species = config.get("species", "yeast")

    from codonscope.core.sequences import SequenceDB
    from codonscope.core.statistics import compute_geneset_frequencies

    db = SequenceDB(species)
    all_seqs = db.get_all_sequences()
    bg_per_gene, bg_mean, kmer_names = compute_geneset_frequencies(all_seqs, k=1)
    kmer_to_idx = {km: i for i, km in enumerate(kmer_names)}
    bg_std = bg_per_gene.std(axis=0, ddof=1).astype(np.float64)

    gene_names = list(per_gene_freq.index)
    n_genes = len(gene_names)

    driver_tables = {}
    summary_records = []

    for codon in sig_codons:
        if codon not in per_gene_freq.columns:
            continue

        idx = kmer_to_idx.get(codon)
        if idx is None:
            continue

        genome_mean = float(bg_mean[idx])
        genome_std_val = float(bg_std[idx])

        # Per-gene contribution
        gene_freqs = per_gene_freq[codon].values.astype(np.float64)
        contributions = gene_freqs - genome_mean

        # Gene set mean
        gs_mean = gene_freqs.mean()

        # Sort by contribution (descending for enriched, ascending for depleted)
        z_row = sig[sig["kmer"] == codon].iloc[0]
        is_enriched = z_row["z_score"] > 0

        sorted_idx = np.argsort(-contributions if is_enriched else contributions)
        sorted_contribs = contributions[sorted_idx]
        sorted_genes = [gene_names[i] for i in sorted_idx]
        sorted_freqs = gene_freqs[sorted_idx]

        # Cumulative contribution
        abs_contribs = np.abs(contributions)
        total_abs = abs_contribs.sum()

        # Sort by absolute contribution for N50/N80
        abs_sorted_idx = np.argsort(-abs_contribs)
        abs_cumsum = np.cumsum(abs_contribs[abs_sorted_idx])

        n50 = n80 = n_genes
        if total_abs > 0:
            cumfrac = abs_cumsum / total_abs
            n50_arr = np.where(cumfrac >= 0.5)[0]
            n80_arr = np.where(cumfrac >= 0.8)[0]
            n50 = int(n50_arr[0]) + 1 if len(n50_arr) > 0 else n_genes
            n80 = int(n80_arr[0]) + 1 if len(n80_arr) > 0 else n_genes

        # Gini coefficient
        gini = _gini_coefficient(abs_contribs)

        # Jackknife: remove each gene, recompute mean Z
        jackknife_flags = []
        original_z = z_row["z_score"]
        for i in range(n_genes):
            leave_out = np.delete(gene_freqs, i)
            loo_mean = leave_out.mean()
            if genome_std_val > 0:
                loo_z = (loo_mean - genome_mean) / (genome_std_val / np.sqrt(n_genes - 1))
            else:
                loo_z = 0.0
            delta_z = abs(original_z - loo_z)
            jackknife_flags.append(delta_z > 2.0)

        # Build driver table
        records = []
        for i, gi in enumerate(sorted_idx):
            records.append({
                "gene": gene_names[gi],
                "frequency": gene_freqs[gi],
                "contribution": contributions[gi],
                "abs_contribution": abs_contribs[gi],
                "jackknife_influential": jackknife_flags[gi],
            })

        driver_df = pd.DataFrame(records)

        # Add common names
        common_names = db.get_common_names([r["gene"] for r in records])
        driver_df["gene_name"] = driver_df["gene"].map(lambda g: common_names.get(g, g))

        driver_tables[codon] = driver_df

        summary_records.append({
            "codon": codon,
            "amino_acid": z_row.get("amino_acid", ""),
            "z_score": z_row["z_score"],
            "adjusted_p": z_row["adjusted_p"],
            "gini": gini,
            "n50": n50,
            "n80": n80,
            "n_jackknife_influential": sum(jackknife_flags),
            "top_driver_gene": sorted_genes[0] if sorted_genes else "",
            "top_driver_contribution": sorted_contribs[0] if len(sorted_contribs) > 0 else 0,
        })

    summary_df = pd.DataFrame(summary_records)

    # Output
    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(out / "driver_summary.tsv", sep="\t", index=False, float_format="%.6g")
        for codon, df in driver_tables.items():
            df.to_csv(out / f"driver_{codon}.tsv", sep="\t", index=False, float_format="%.6g")

    return {
        "driver_tables": driver_tables,
        "summary": summary_df,
        "significant_codons": sig_codons,
        "n_genes": n_genes,
    }


def _gini_coefficient(values: np.ndarray) -> float:
    """Compute the Gini coefficient of an array of values."""
    if len(values) == 0 or values.sum() == 0:
        return 0.0
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    index = np.arange(1, n + 1)
    return float((2.0 * np.sum(index * sorted_vals) / (n * np.sum(sorted_vals))) - (n + 1) / n)


def _synonymous_shuffle(codon_list: list[str], rng: np.random.Generator) -> list[str]:
    """Shuffle codons within amino acid families, preserving AA sequence.

    Args:
        codon_list: List of codon strings.
        rng: Numpy random generator.

    Returns:
        Shuffled codon list with same amino acid sequence.
    """
    from codonscope.core.codons import CODON_TABLE

    # Group positions by amino acid
    aa_groups: dict[str, list[int]] = {}
    for i, codon in enumerate(codon_list):
        aa = CODON_TABLE.get(codon, "X")
        aa_groups.setdefault(aa, []).append(i)

    # Shuffle within each group
    result = list(codon_list)
    for aa, positions in aa_groups.items():
        if len(positions) > 1:
            codons_at_pos = [codon_list[p] for p in positions]
            rng.shuffle(codons_at_pos)
            for p, c in zip(positions, codons_at_pos):
                result[p] = c

    return result


def _compute_positional_freqs(seq: str, n_bins: int) -> np.ndarray:
    """Compute codon frequencies at each positional bin for a single gene.

    Args:
        seq: CDS sequence string.
        n_bins: Number of positional bins.

    Returns:
        (n_bins, 61) array of codon frequencies per bin.
    """
    from codonscope.core.codons import sequence_to_codons, SENSE_CODONS

    codons = sequence_to_codons(seq)
    n_codons = len(codons)

    if n_codons == 0:
        return np.zeros((n_bins, 61), dtype=np.float64)

    # Map codons to indices
    codon_to_idx = {c: i for i, c in enumerate(SENSE_CODONS)}

    # Count codons per bin
    bin_counts = np.zeros((n_bins, 61), dtype=np.float64)
    bin_totals = np.zeros(n_bins, dtype=np.float64)

    for pos, codon in enumerate(codons):
        bin_idx = int(pos / n_codons * n_bins)
        if bin_idx >= n_bins:  # Edge case for rounding
            bin_idx = n_bins - 1

        codon_idx = codon_to_idx.get(codon)
        if codon_idx is not None:
            bin_counts[bin_idx, codon_idx] += 1
            bin_totals[bin_idx] += 1

    # Normalize to frequencies
    bin_freqs = np.zeros_like(bin_counts)
    for b in range(n_bins):
        if bin_totals[b] > 0:
            bin_freqs[b, :] = bin_counts[b, :] / bin_totals[b]

    return bin_freqs


def run_positional_enrichment(
    results_dir: str | Path,
    n_bins: int = 100,
    output_dir: str | Path | None = None,
) -> dict:
    """Analyze positional enrichment of codons across the CDS.

    For each gene, normalize CDS to n_bins positional bins and compute
    codon frequency per bin. Compare gene set mean to genome mean per bin.

    Args:
        results_dir: Path to Tier 1 data directory.
        n_bins: Number of positional bins (default: 100).
        output_dir: Directory for output files.

    Returns:
        Dict with:
        - 'z_matrix': (n_bins, 61) Z-score matrix
        - 'sig_matrix': (n_bins, 61) boolean mask (adj_p < 0.05)
        - 'kmer_names': list of 61 codon names
        - 'n_bins': int
        - 'n_genes': int
        - 'species': str
        - 'top_enriched': list of tuples (codon, max_z)
        - 'top_depleted': list of tuples (codon, min_z)
    """
    tier1 = load_tier1_data(results_dir)
    config = tier1["config"]
    per_gene_freq = tier1["per_gene_freq"]
    species = config.get("species", "yeast")

    # Get gene list from per_gene_freq index (systematic names)
    gene_list = list(per_gene_freq.index)

    from codonscope.core.sequences import SequenceDB
    from codonscope.core.codons import SENSE_CODONS
    from codonscope.core.statistics import benjamini_hochberg
    from scipy import stats

    logger.info(f"Running positional enrichment for {species} with {n_bins} bins")

    db = SequenceDB(species)

    # Get gene set sequences (gene_list is already systematic names from per_gene_freq)
    gs_seqs = db.get_sequences(gene_list)
    n_gs_genes = len(gs_seqs)

    if n_gs_genes == 0:
        raise ValueError("No valid gene sequences found")

    logger.info(f"  Gene set: {n_gs_genes} genes")

    # Compute gene set positional frequencies
    gs_per_gene_bins = np.zeros((n_gs_genes, n_bins, 61), dtype=np.float64)
    for i, (gene_id, seq) in enumerate(gs_seqs.items()):
        gs_per_gene_bins[i] = _compute_positional_freqs(seq, n_bins)

    gs_mean = gs_per_gene_bins.mean(axis=0)  # (n_bins, 61)

    # Compute genome background statistics incrementally to avoid memory issues
    logger.info("  Computing genome background...")
    all_seqs = db.get_all_sequences()
    n_genome = len(all_seqs)

    # Pass 1: compute means
    genome_sum = np.zeros((n_bins, 61), dtype=np.float64)
    genome_count = 0
    for gene_name, seq in all_seqs.items():
        bin_freqs = _compute_positional_freqs(seq, n_bins)
        genome_sum += bin_freqs
        genome_count += 1

    genome_mean = genome_sum / genome_count

    # Pass 2: compute stds
    logger.info("  Computing genome standard deviation...")
    genome_sq_sum = np.zeros((n_bins, 61), dtype=np.float64)
    for gene_name, seq in all_seqs.items():
        bin_freqs = _compute_positional_freqs(seq, n_bins)
        genome_sq_sum += (bin_freqs - genome_mean) ** 2

    genome_std = np.sqrt(genome_sq_sum / (genome_count - 1))

    # Compute Z-scores: Z = (gs_mean - genome_mean) / (genome_std / sqrt(n_gs_genes))
    se = genome_std / np.sqrt(n_gs_genes)

    # Avoid division by zero
    z_matrix = np.zeros((n_bins, 61), dtype=np.float64)
    valid_mask = se > 0
    z_matrix[valid_mask] = (gs_mean[valid_mask] - genome_mean[valid_mask]) / se[valid_mask]

    # Compute p-values and apply BH correction
    p_matrix = 2 * (1 - stats.norm.cdf(np.abs(z_matrix)))
    p_flat = p_matrix.flatten()
    adj_p_flat = benjamini_hochberg(p_flat)
    adj_p_matrix = adj_p_flat.reshape((n_bins, 61))

    sig_matrix = adj_p_matrix < 0.05

    # Mask non-significant Z-scores
    z_matrix_masked = z_matrix.copy()
    z_matrix_masked[~sig_matrix] = 0.0

    # Identify top enriched and depleted codons by max absolute Z
    max_abs_z = np.abs(z_matrix).max(axis=0)  # (61,)
    enriched_idx = np.argsort(-max_abs_z)[:5]
    depleted_idx = np.argsort(z_matrix.min(axis=0))[:5]

    top_enriched = [(SENSE_CODONS[i], float(max_abs_z[i])) for i in enriched_idx]
    top_depleted = [(SENSE_CODONS[i], float(z_matrix[:, i].min())) for i in depleted_idx]

    logger.info(f"  Top enriched codons: {[c for c, _ in top_enriched]}")
    logger.info(f"  Top depleted codons: {[c for c, _ in top_depleted]}")

    # Output
    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Write Z-matrix
        z_df = pd.DataFrame(z_matrix, columns=SENSE_CODONS)
        z_df.insert(0, "bin", range(n_bins))
        z_df.to_csv(out / "positional_z_matrix.tsv", sep="\t", index=False, float_format="%.4f")

        logger.info(f"  Wrote positional_z_matrix.tsv to {output_dir}/")

    return {
        "z_matrix": z_matrix,
        "sig_matrix": sig_matrix,
        "kmer_names": SENSE_CODONS,
        "n_bins": n_bins,
        "n_genes": n_gs_genes,
        "species": species,
        "top_enriched": top_enriched,
        "top_depleted": top_depleted,
    }


def run_cluster_scan(
    results_dir: str | Path,
    codons: list[str] | None = None,
    top_n: int = 5,
    window: int = 15,
    n_permutations: int = 1000,
    seed: int | None = None,
    output_dir: str | Path | None = None,
) -> dict:
    """Scan for clusters of target codons in gene sequences.

    Args:
        results_dir: Path to Tier 1 data directory.
        codons: Target codons to scan for. If None, auto-select top N enriched.
        top_n: Number of top enriched codons to auto-select (default: 5).
        window: Sliding window size for cluster detection (default: 15).
        n_permutations: Number of permutation tests (default: 1000).
        seed: Random seed for reproducibility.
        output_dir: Directory for output files.

    Returns:
        Dict with:
        - 'per_gene': DataFrame with cluster statistics per gene
        - 'target_codons': list[str]
        - 'n_genes': int
        - 'summary': dict with aggregate statistics
    """
    tier1 = load_tier1_data(results_dir)
    enrichment = tier1["enrichment"]
    per_gene_freq = tier1["per_gene_freq"]
    config = tier1["config"]
    species = config.get("species", "yeast")

    # Get gene list from per_gene_freq index (systematic names)
    gene_list = list(per_gene_freq.index)

    from codonscope.core.sequences import SequenceDB
    from codonscope.core.codons import sequence_to_codons

    logger.info(f"Running codon cluster scan for {species}")

    # Auto-select codons if not provided
    if codons is None:
        sig = enrichment[enrichment["adjusted_p"] < 0.05].copy()
        sig = sig.sort_values("z_score", ascending=False)
        codons = sig.head(top_n)["kmer"].tolist()
        logger.info(f"  Auto-selected top {top_n} enriched codons: {codons}")
    else:
        logger.info(f"  Target codons: {codons}")

    target_set = set(codons)

    db = SequenceDB(species)
    # Get gene set sequences (gene_list is already systematic names from per_gene_freq)
    gs_seqs = db.get_sequences(gene_list)
    common_names = db.get_common_names(list(gs_seqs.keys()))

    rng = np.random.default_rng(seed)

    records = []

    for gene_id, seq in gs_seqs.items():
        codon_list = sequence_to_codons(seq)
        n_codons = len(codon_list)

        if n_codons == 0:
            continue

        # Count target codons
        n_target = sum(1 for c in codon_list if c in target_set)

        if n_target == 0:
            records.append({
                "gene": gene_id,
                "gene_name": common_names.get(gene_id, gene_id),
                "n_codons": n_codons,
                "n_target": 0,
                "n_runs": 0,
                "max_run_length": 0,
                "n_clusters": 0,
                "perm_p": 1.0,
            })
            continue

        # Count runs (2+ consecutive target codons)
        def count_runs(codons: list[str]) -> tuple[int, int]:
            runs = []
            current_run = 0
            for c in codons:
                if c in target_set:
                    current_run += 1
                else:
                    if current_run >= 2:
                        runs.append(current_run)
                    current_run = 0
            if current_run >= 2:
                runs.append(current_run)
            return len(runs), max(runs) if runs else 0

        n_runs, max_run = count_runs(codon_list)

        # Sliding window cluster detection
        gene_freq = n_target / n_codons
        threshold_density = 2 * gene_freq
        n_clusters = 0

        if n_codons >= window:
            for i in range(n_codons - window + 1):
                window_codons = codon_list[i:i + window]
                window_targets = sum(1 for c in window_codons if c in target_set)
                window_density = window_targets / window
                if window_density > threshold_density and window_targets >= 2:
                    n_clusters += 1

        # Permutation test: synonymous shuffle
        shuffles_gte = 0
        for _ in range(n_permutations):
            shuffled = _synonymous_shuffle(codon_list, rng)
            perm_runs, _ = count_runs(shuffled)
            if perm_runs >= n_runs:
                shuffles_gte += 1

        perm_p = (shuffles_gte + 1) / (n_permutations + 1)

        records.append({
            "gene": gene_id,
            "gene_name": common_names.get(gene_id, gene_id),
            "n_codons": n_codons,
            "n_target": n_target,
            "n_runs": n_runs,
            "max_run_length": max_run,
            "n_clusters": n_clusters,
            "perm_p": perm_p,
        })

    per_gene_df = pd.DataFrame(records)

    # Summary statistics
    if len(per_gene_df) > 0:
        sig_genes = per_gene_df[per_gene_df["perm_p"] < 0.05]
        summary = {
            "n_genes": len(per_gene_df),
            "n_significant": len(sig_genes),
            "frac_significant": len(sig_genes) / len(per_gene_df),
            "mean_runs": float(per_gene_df["n_runs"].mean()),
            "mean_max_run": float(per_gene_df["max_run_length"].mean()),
            "mean_clusters": float(per_gene_df["n_clusters"].mean()),
        }
    else:
        summary = {
            "n_genes": 0,
            "n_significant": 0,
            "frac_significant": 0.0,
            "mean_runs": 0.0,
            "mean_max_run": 0.0,
            "mean_clusters": 0.0,
        }

    logger.info(f"  {summary['n_significant']}/{summary['n_genes']} genes with significant clusters (p < 0.05)")
    logger.info(f"  Mean runs: {summary['mean_runs']:.2f}, mean max run: {summary['mean_max_run']:.2f}")

    # Output
    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        per_gene_df.to_csv(out / "cluster_scan.tsv", sep="\t", index=False, float_format="%.4f")
        logger.info(f"  Wrote cluster_scan.tsv to {output_dir}/")

    return {
        "per_gene": per_gene_df,
        "target_codons": codons,
        "n_genes": len(per_gene_df),
        "summary": summary,
    }


def generate_deep_dive_report(
    results_dir: str | Path,
    output: str | Path = "deep_dive.html",
    adj_p_threshold: float = 0.05,
) -> Path:
    """Generate a deep-dive HTML report from Tier 1 data.

    Args:
        results_dir: Path to Tier 1 data directory.
        output: Output HTML file path.
        adj_p_threshold: Significance threshold.

    Returns:
        Path to the generated HTML file.
    """
    import base64
    import io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from datetime import datetime
    from html import escape

    output = Path(output)
    result = run_driver_analysis(results_dir, adj_p_threshold=adj_p_threshold)

    # Build HTML sections
    sections = []
    sections.append('<h1>CodonScope Deep-Dive: Driver Gene Analysis</h1>')

    summary = result["summary"]
    if len(summary) > 0:
        # Summary table
        rows = ""
        for _, row in summary.iterrows():
            rows += (
                f'<tr><td><strong>{escape(str(row["codon"]))}</strong></td>'
                f'<td>{escape(str(row.get("amino_acid", "")))}</td>'
                f'<td>{row["z_score"]:+.2f}</td>'
                f'<td>{row["gini"]:.3f}</td>'
                f'<td>{row["n50"]}</td>'
                f'<td>{row["n80"]}</td>'
                f'<td>{row["n_jackknife_influential"]}</td>'
                f'<td>{escape(str(row["top_driver_gene"]))}</td></tr>\n'
            )
        sections.append(f'''
        <div class="section">
        <h2>Driver Summary ({len(summary)} significant codons)</h2>
        <p>Genes in set: <strong>{result["n_genes"]}</strong></p>
        <table>
        <tr><th>Codon</th><th>AA</th><th>Z</th><th>Gini</th><th>N50</th><th>N80</th><th>Influential</th><th>Top Driver</th></tr>
        {rows}
        </table>
        </div>
        ''')

        # Per-codon sections
        for codon in result["significant_codons"]:
            if codon not in result["driver_tables"]:
                continue
            df = result["driver_tables"][codon]

            # Cumulative contribution plot
            abs_c = df["abs_contribution"].values
            abs_sorted = np.sort(abs_c)[::-1]
            cumsum = np.cumsum(abs_sorted) / abs_sorted.sum() if abs_sorted.sum() > 0 else np.zeros_like(abs_sorted)

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(range(1, len(cumsum) + 1), cumsum, color="#2563eb", lw=1.5)
            ax.axhline(0.5, color="#f59e0b", ls="--", lw=0.8, label="50%")
            ax.axhline(0.8, color="#dc2626", ls="--", lw=0.8, label="80%")
            ax.set_xlabel("Number of genes (ranked by |contribution|)")
            ax.set_ylabel("Cumulative fraction of total contribution")
            ax.set_title(f"{codon}: Cumulative Contribution Curve")
            ax.legend(fontsize=8)
            fig.tight_layout()

            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            buf.seek(0)
            plot_b64 = base64.b64encode(buf.read()).decode("ascii")

            # Top/bottom 10 table
            top10 = df.head(10)
            bot10 = df.tail(10)

            def _gene_rows(sub_df):
                rows = ""
                for _, row in sub_df.iterrows():
                    flag = " *" if row["jackknife_influential"] else ""
                    rows += (
                        f'<tr><td>{escape(str(row["gene_name"]))}</td>'
                        f'<td>{escape(str(row["gene"]))}</td>'
                        f'<td>{row["frequency"]:.5f}</td>'
                        f'<td>{row["contribution"]:+.5f}{flag}</td></tr>\n'
                    )
                return rows

            codon_summary = summary[summary["codon"] == codon].iloc[0]

            sections.append(f'''
            <details class="collapsible-section">
            <summary>{codon} ({codon_summary.get("amino_acid", "")}) -- Z={codon_summary["z_score"]:+.2f}, Gini={codon_summary["gini"]:.3f}, N50={codon_summary["n50"]}</summary>
            <div class="section">
            <h3>{codon} Driver Analysis</h3>
            <div class="plot-container"><img src="data:image/png;base64,{plot_b64}" alt="cumulative"></div>
            <h4>Top 10 Positive Contributors</h4>
            <table>
            <tr><th>Gene</th><th>Systematic</th><th>Frequency</th><th>Contribution</th></tr>
            {_gene_rows(top10)}
            </table>
            <h4>Top 10 Negative Contributors</h4>
            <table>
            <tr><th>Gene</th><th>Systematic</th><th>Frequency</th><th>Contribution</th></tr>
            {_gene_rows(bot10)}
            </table>
            </div>
            </details>
            ''')
    else:
        sections.append('<p><em>No significant codons found at the given threshold.</em></p>')

    # Assemble HTML
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    body = "\n".join(sections)
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>CodonScope Deep-Dive Report</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 1100px; margin: 0 auto; padding: 20px; color: #1a1a2e; background: #f8f9fa; }}
h1 {{ color: #16213e; border-bottom: 3px solid #0f3460; padding-bottom: 10px; }}
h2 {{ color: #0f3460; }}
.section {{ background: white; border: 1px solid #e2e8f0; border-radius: 10px; padding: 20px; margin: 20px 0; }}
table {{ border-collapse: collapse; width: 100%; font-size: 0.88em; background: white; }}
th {{ background: #0f3460; color: white; padding: 8px 10px; text-align: left; }}
td {{ padding: 6px 10px; border-bottom: 1px solid #e2e8f0; }}
.plot-container {{ text-align: center; margin: 20px 0; }}
.plot-container img {{ max-width: 100%; border: 1px solid #e2e8f0; border-radius: 6px; }}
details.collapsible-section {{ margin: 20px 0; }}
details.collapsible-section > summary {{ cursor: pointer; padding: 12px 18px; background: #f0f4ff; border: 1px solid #c7d2fe; border-radius: 8px; font-weight: 600; color: #1e3a5f; }}
.footer {{ text-align: center; color: #94a3b8; font-size: 0.82em; margin-top: 40px; }}
</style>
</head>
<body>
{body}
<div class="footer">CodonScope Deep-Dive &mdash; {now}</div>
</body>
</html>'''

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    return output
