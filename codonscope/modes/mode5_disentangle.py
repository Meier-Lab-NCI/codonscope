"""Mode 5: Amino Acid vs Codon Disentanglement.

Is codon bias driven by protein composition or synonymous codon choice?

Two-layer decomposition:
  Layer 1 — Amino acid composition vs genome
  Layer 2 — RSCU (Relative Synonymous Codon Usage) within each AA family

Attribution for each codon:
  "AA-driven"         amino acid enriched; synonymous usage normal
  "Synonymous-driven" amino acid normal; this synonym preferred
  "Both"              both effects
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from codonscope.core.codons import SENSE_CODONS, sequence_to_codons
from codonscope.core.sequences import SequenceDB
from codonscope.core.statistics import (
    benjamini_hochberg,
    bootstrap_pvalues,
    bootstrap_zscores,
    power_check,
)

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# Standard genetic code
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

# Amino acids sorted alphabetically (20 standard)
AMINO_ACIDS = sorted(set(CODON_TABLE.values()))

# AA → list of synonymous codons
AA_FAMILIES: dict[str, list[str]] = {}
for _codon, _aa in CODON_TABLE.items():
    AA_FAMILIES.setdefault(_aa, []).append(_codon)
for _aa in AA_FAMILIES:
    AA_FAMILIES[_aa].sort()

# Number of synonyms per AA family
N_SYNONYMS: dict[str, int] = {aa: len(codons) for aa, codons in AA_FAMILIES.items()}


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════

def run_disentangle(
    species: str,
    gene_ids: list[str],
    n_bootstrap: int = 10_000,
    seed: int | None = None,
    output_dir: str | Path | None = None,
    data_dir: str | Path | None = None,
) -> dict:
    """Run Mode 5: AA vs Codon Disentanglement.

    Args:
        species: Species name (e.g. "yeast", "human").
        gene_ids: List of gene identifiers (any format).
        n_bootstrap: Bootstrap iterations.
        seed: Random seed for reproducibility.
        output_dir: Directory for output files. None = no file output.
        data_dir: Override default data directory.

    Returns:
        dict with keys:
            "aa_results": DataFrame — amino acid enrichment Z-scores
            "rscu_results": DataFrame — per-codon RSCU deviation Z-scores
            "attribution": DataFrame — codon attribution (AA-driven/Synonymous/Both)
            "synonymous_drivers": DataFrame — classification of synonymous deviations
            "summary": dict with "pct_aa_driven", "pct_synonymous_driven", etc.
            "id_summary": IDMapping
            "n_genes": int
    """
    db = SequenceDB(species, data_dir=data_dir)
    species_dir = db.species_dir

    # Resolve gene IDs
    id_result = db.resolve_ids(gene_ids)
    gene_seqs = db.get_sequences(id_result)
    n_genes = len(gene_seqs)
    logger.info("Analyzing %d genes for disentanglement", n_genes)

    # Power check
    for w in power_check(n_genes, k=1):
        logger.warning(w)

    # Get all genome sequences for background
    all_seqs = db.get_all_sequences()

    # ── Layer 1: Amino acid composition ──────────────────────────────────
    aa_results = _amino_acid_analysis(gene_seqs, all_seqs, n_bootstrap, seed)

    # ── Layer 2: RSCU ────────────────────────────────────────────────────
    rscu_results = _rscu_analysis(gene_seqs, all_seqs, n_bootstrap, seed)

    # ── Attribution ──────────────────────────────────────────────────────
    attribution = _build_attribution(aa_results, rscu_results)

    # ── Synonymous deviation classification ──────────────────────────────
    syn_drivers = _classify_synonymous_drivers(rscu_results, species_dir)

    # ── Summary stats ────────────────────────────────────────────────────
    summary = _compute_summary(attribution)

    # ── Output ───────────────────────────────────────────────────────────
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        _write_outputs(aa_results, rscu_results, attribution, syn_drivers,
                       summary, out)

    return {
        "aa_results": aa_results,
        "rscu_results": rscu_results,
        "attribution": attribution,
        "synonymous_drivers": syn_drivers,
        "summary": summary,
        "id_summary": id_result,
        "n_genes": n_genes,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 1: Amino acid composition
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_aa_frequencies(sequences: dict[str, str]) -> np.ndarray:
    """Compute per-gene amino acid frequency matrix.

    Returns:
        (n_genes, 20) array of per-gene AA frequencies.
    """
    gene_names = sorted(sequences.keys())
    n_genes = len(gene_names)
    aa_to_idx = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
    per_gene = np.zeros((n_genes, 20), dtype=np.float32)

    for gi, gene in enumerate(gene_names):
        seq = sequences[gene]
        codons = sequence_to_codons(seq)
        if not codons:
            continue
        aa_counts = np.zeros(20, dtype=np.float64)
        for codon in codons:
            aa = CODON_TABLE.get(codon)
            if aa is not None:
                aa_counts[aa_to_idx[aa]] += 1
        total = aa_counts.sum()
        if total > 0:
            per_gene[gi] = (aa_counts / total).astype(np.float32)

    return per_gene


def _amino_acid_analysis(
    gene_seqs: dict[str, str],
    all_seqs: dict[str, str],
    n_bootstrap: int,
    seed: int | None,
) -> pd.DataFrame:
    """Compare gene-set AA frequencies to genome via bootstrap Z-scores."""
    # Gene set
    gs_per_gene = _compute_aa_frequencies(gene_seqs)
    gs_mean = gs_per_gene.mean(axis=0).astype(np.float64)

    # Background
    bg_per_gene = _compute_aa_frequencies(all_seqs)
    bg_mean = bg_per_gene.mean(axis=0).astype(np.float64)
    bg_std = bg_per_gene.std(axis=0, ddof=1).astype(np.float64)

    # Bootstrap
    z_scores, boot_mean, _ = bootstrap_zscores(
        gs_mean, bg_per_gene, n_genes=len(gene_seqs),
        n_bootstrap=n_bootstrap, seed=seed,
    )
    p_values = bootstrap_pvalues(z_scores)
    adj_p = benjamini_hochberg(p_values)

    df = pd.DataFrame({
        "amino_acid": AMINO_ACIDS,
        "observed_freq": gs_mean,
        "expected_freq": boot_mean,
        "z_score": z_scores,
        "p_value": p_values,
        "adjusted_p": adj_p,
    })
    return df.sort_values("z_score", key=np.abs, ascending=False).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 2: RSCU
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_rscu_per_gene(sequences: dict[str, str]) -> np.ndarray:
    """Compute per-gene RSCU for each of the 61 sense codons.

    RSCU = n_synonyms * (count_codon / count_AA_family)

    When RSCU = 1.0, the codon is used equally with its synonyms.
    RSCU > 1 means preferred, RSCU < 1 means avoided.

    Returns:
        (n_genes, 61) array of RSCU values.
    """
    gene_names = sorted(sequences.keys())
    n_genes = len(gene_names)
    codon_to_idx = {c: i for i, c in enumerate(SENSE_CODONS)}
    per_gene = np.zeros((n_genes, 61), dtype=np.float32)

    for gi, gene in enumerate(gene_names):
        seq = sequences[gene]
        codons = sequence_to_codons(seq)
        if not codons:
            continue

        # Count codons
        codon_counts: dict[str, int] = {}
        for codon in codons:
            if codon in codon_to_idx:
                codon_counts[codon] = codon_counts.get(codon, 0) + 1

        # Compute RSCU per AA family
        for aa, family_codons in AA_FAMILIES.items():
            n_syn = len(family_codons)
            total_aa = sum(codon_counts.get(c, 0) for c in family_codons)
            if total_aa == 0:
                continue
            for codon in family_codons:
                count = codon_counts.get(codon, 0)
                rscu = n_syn * count / total_aa
                per_gene[gi, codon_to_idx[codon]] = rscu

    return per_gene


def _rscu_analysis(
    gene_seqs: dict[str, str],
    all_seqs: dict[str, str],
    n_bootstrap: int,
    seed: int | None,
) -> pd.DataFrame:
    """Compare gene-set RSCU to genome RSCU via bootstrap Z-scores."""
    # Gene set
    gs_rscu = _compute_rscu_per_gene(gene_seqs)
    gs_mean = gs_rscu.mean(axis=0).astype(np.float64)

    # Background
    bg_rscu = _compute_rscu_per_gene(all_seqs)
    bg_mean = bg_rscu.mean(axis=0).astype(np.float64)
    bg_std = bg_rscu.std(axis=0, ddof=1).astype(np.float64)

    # Bootstrap
    z_scores, boot_mean, _ = bootstrap_zscores(
        gs_mean, bg_rscu, n_genes=len(gene_seqs),
        n_bootstrap=n_bootstrap, seed=seed,
    )
    p_values = bootstrap_pvalues(z_scores)
    adj_p = benjamini_hochberg(p_values)

    # Build amino acid annotation for each codon
    codon_aa = [CODON_TABLE[c] for c in SENSE_CODONS]
    n_syn = [N_SYNONYMS[CODON_TABLE[c]] for c in SENSE_CODONS]

    df = pd.DataFrame({
        "codon": SENSE_CODONS,
        "amino_acid": codon_aa,
        "n_synonyms": n_syn,
        "observed_rscu": gs_mean,
        "expected_rscu": boot_mean,
        "z_score": z_scores,
        "p_value": p_values,
        "adjusted_p": adj_p,
    })
    return df.sort_values("z_score", key=np.abs, ascending=False).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Attribution
# ═══════════════════════════════════════════════════════════════════════════════

def _build_attribution(
    aa_results: pd.DataFrame,
    rscu_results: pd.DataFrame,
) -> pd.DataFrame:
    """Classify each codon's deviation as AA-driven, synonymous-driven, or both.

    Thresholds:
        AA significant:  adjusted_p < 0.05 for the amino acid
        RSCU significant: adjusted_p < 0.05 for the codon's RSCU

    Attribution:
        AA-driven:         AA significant, RSCU not significant
        Synonymous-driven: AA not significant, RSCU significant
        Both:              both significant
        None:              neither significant
    """
    # Build AA significance lookup
    aa_sig = set(
        aa_results.loc[aa_results["adjusted_p"] < 0.05, "amino_acid"]
    )

    rows = []
    for _, row in rscu_results.iterrows():
        codon = row["codon"]
        aa = row["amino_acid"]
        rscu_significant = row["adjusted_p"] < 0.05
        aa_significant = aa in aa_sig

        # Get AA Z-score for context
        aa_row = aa_results[aa_results["amino_acid"] == aa]
        aa_z = float(aa_row["z_score"].iloc[0]) if len(aa_row) > 0 else 0.0

        if aa_significant and rscu_significant:
            attribution = "Both"
        elif aa_significant:
            attribution = "AA-driven"
        elif rscu_significant:
            attribution = "Synonymous-driven"
        else:
            attribution = "None"

        rows.append({
            "codon": codon,
            "amino_acid": aa,
            "aa_z_score": aa_z,
            "rscu_z_score": row["z_score"],
            "aa_adj_p": float(aa_row["adjusted_p"].iloc[0]) if len(aa_row) > 0 else 1.0,
            "rscu_adj_p": row["adjusted_p"],
            "attribution": attribution,
        })

    df = pd.DataFrame(rows)
    return df.sort_values("rscu_z_score", key=np.abs, ascending=False).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Synonymous deviation classification
# ═══════════════════════════════════════════════════════════════════════════════

def _classify_synonymous_drivers(
    rscu_results: pd.DataFrame,
    species_dir: Path,
) -> pd.DataFrame:
    """For codons with significant RSCU deviations, classify the cause.

    Drivers:
        - tRNA supply:     deviation correlates with tRNA gene copy number
        - GC3 bias:        codons ending in G/C are systematically preferred
        - Wobble avoidance: Watson-Crick decoded codons preferred over wobble
    """
    sig = rscu_results[rscu_results["adjusted_p"] < 0.05].copy()
    if len(sig) == 0:
        return pd.DataFrame(columns=[
            "codon", "amino_acid", "rscu_z_score", "driver",
            "gc3", "decoding_type", "trna_copies",
        ])

    # Load wobble rules
    wobble_path = species_dir / "wobble_rules.tsv"
    wobble_info: dict[str, dict] = {}
    if wobble_path.exists():
        wobble_df = pd.read_csv(wobble_path, sep="\t")
        for _, row in wobble_df.iterrows():
            wobble_info[row["codon"]] = {
                "decoding_type": row["decoding_type"],
                "trna_copies": int(row["trna_gene_copies"]),
            }

    rows = []
    for _, row in sig.iterrows():
        codon = row["codon"]
        aa = row["amino_acid"]
        z = row["z_score"]

        # GC3: does this codon end in G or C?
        gc3 = 1 if codon[2] in ("G", "C") else 0

        # Wobble info
        winfo = wobble_info.get(codon, {})
        decoding = winfo.get("decoding_type", "unknown")
        trna = winfo.get("trna_copies", 0)

        # Classify driver (heuristic)
        driver = _classify_single_driver(z, gc3, decoding, trna, aa)

        rows.append({
            "codon": codon,
            "amino_acid": aa,
            "rscu_z_score": z,
            "driver": driver,
            "gc3": gc3,
            "decoding_type": decoding,
            "trna_copies": trna,
        })

    return pd.DataFrame(rows).sort_values(
        "rscu_z_score", key=np.abs, ascending=False
    ).reset_index(drop=True)


def _classify_single_driver(
    z_score: float,
    gc3: int,
    decoding_type: str,
    trna_copies: int,
    amino_acid: str,
) -> str:
    """Heuristic classification of a single synonymous deviation.

    Rules:
    1. If codon is enriched (Z > 0) and Watson-Crick decoded → "wobble avoidance"
    2. If codon is enriched and has high tRNA copies → "tRNA supply"
    3. If enriched codon ends in G/C (gc3=1) and Z > 0 → "GC3 bias"
    4. If depleted codon ends in A/T (gc3=0) and Z < 0 → "GC3 bias"
    5. Otherwise → "unclassified"

    For single-codon AAs (Met, Trp), RSCU is always 1.0 — skip.
    """
    if N_SYNONYMS.get(amino_acid, 1) <= 1:
        return "not_applicable"

    enriched = z_score > 0

    # Watson-Crick and enriched → wobble avoidance
    if enriched and decoding_type == "watson_crick" and trna_copies > 0:
        return "wobble_avoidance"

    # High tRNA supply (above median for that family would be better,
    # but heuristic: tRNA copies > 5) and enriched
    if enriched and trna_copies > 5:
        return "tRNA_supply"

    # GC3 bias: enriched G/C-ending or depleted A/T-ending
    if enriched and gc3 == 1:
        return "GC3_bias"
    if not enriched and gc3 == 0:
        return "GC3_bias"

    return "unclassified"


# ═══════════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_summary(attribution: pd.DataFrame) -> dict:
    """Compute summary statistics from attribution table."""
    sig = attribution[attribution["attribution"] != "None"]
    total = len(sig)
    if total == 0:
        return {
            "n_significant_codons": 0,
            "pct_aa_driven": 0.0,
            "pct_synonymous_driven": 0.0,
            "pct_both": 0.0,
            "summary_text": "No significant codon deviations detected.",
        }

    n_aa = len(sig[sig["attribution"] == "AA-driven"])
    n_syn = len(sig[sig["attribution"] == "Synonymous-driven"])
    n_both = len(sig[sig["attribution"] == "Both"])

    pct_aa = 100 * n_aa / total
    pct_syn = 100 * n_syn / total
    pct_both = 100 * n_both / total

    summary_text = (
        f"{total} codons with significant deviation: "
        f"{pct_aa:.0f}% AA-driven, "
        f"{pct_syn:.0f}% synonymous-driven, "
        f"{pct_both:.0f}% both."
    )

    return {
        "n_significant_codons": total,
        "n_aa_driven": n_aa,
        "n_synonymous_driven": n_syn,
        "n_both": n_both,
        "pct_aa_driven": pct_aa,
        "pct_synonymous_driven": pct_syn,
        "pct_both": pct_both,
        "summary_text": summary_text,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Output
# ═══════════════════════════════════════════════════════════════════════════════

def _write_outputs(
    aa_results: pd.DataFrame,
    rscu_results: pd.DataFrame,
    attribution: pd.DataFrame,
    syn_drivers: pd.DataFrame,
    summary: dict,
    output_dir: Path,
) -> None:
    """Write TSV results and plots to output directory."""
    aa_results.to_csv(
        output_dir / "disentangle_aa.tsv", sep="\t", index=False,
        float_format="%.6g",
    )
    rscu_results.to_csv(
        output_dir / "disentangle_rscu.tsv", sep="\t", index=False,
        float_format="%.6g",
    )
    attribution.to_csv(
        output_dir / "disentangle_attribution.tsv", sep="\t", index=False,
        float_format="%.6g",
    )
    if len(syn_drivers) > 0:
        syn_drivers.to_csv(
            output_dir / "disentangle_drivers.tsv", sep="\t", index=False,
            float_format="%.6g",
        )
    logger.info("Disentanglement results written to %s", output_dir)

    # Plots
    try:
        _plot_two_panel(aa_results, rscu_results, output_dir / "disentangle.png")
    except Exception as exc:
        logger.warning("Plot generation failed: %s", exc)


def _plot_two_panel(
    aa_results: pd.DataFrame,
    rscu_results: pd.DataFrame,
    path: Path,
) -> None:
    """Two-panel figure: (A) AA deviation, (B) within-AA RSCU deviation."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: AA deviation bar chart
    aa_sorted = aa_results.sort_values("z_score")
    colors_a = ["#d73027" if z > 0 else "#4575b4" for z in aa_sorted["z_score"]]
    sig_a = aa_sorted["adjusted_p"] < 0.05
    edge_colors = ["black" if s else "none" for s in sig_a]

    ax1.barh(
        range(len(aa_sorted)), aa_sorted["z_score"],
        color=colors_a, edgecolor=edge_colors, linewidth=0.5,
    )
    ax1.set_yticks(range(len(aa_sorted)))
    ax1.set_yticklabels(aa_sorted["amino_acid"], fontsize=8)
    ax1.set_xlabel("Z-score")
    ax1.set_title("(A) Amino Acid Composition")
    ax1.axvline(0, color="black", linewidth=0.5)

    # Panel B: Top RSCU deviations
    sig_rscu = rscu_results[rscu_results["adjusted_p"] < 0.05]
    if len(sig_rscu) == 0:
        sig_rscu = rscu_results.head(20)

    enriched = sig_rscu[sig_rscu["z_score"] > 0].head(10)
    depleted = sig_rscu[sig_rscu["z_score"] < 0].head(10)
    plot_rscu = pd.concat([enriched, depleted]).sort_values("z_score")

    if len(plot_rscu) > 0:
        labels = [f"{r['codon']} ({r['amino_acid']})" for _, r in plot_rscu.iterrows()]
        colors_b = ["#d73027" if z > 0 else "#4575b4" for z in plot_rscu["z_score"]]
        ax2.barh(range(len(plot_rscu)), plot_rscu["z_score"], color=colors_b)
        ax2.set_yticks(range(len(plot_rscu)))
        ax2.set_yticklabels(labels, fontsize=7)
    ax2.set_xlabel("RSCU Z-score")
    ax2.set_title("(B) Synonymous Codon Preference (RSCU)")
    ax2.axvline(0, color="black", linewidth=0.5)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Disentanglement plot saved to %s", path)
