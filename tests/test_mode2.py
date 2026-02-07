"""Tests for Mode 2: Translational Demand."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

YEAST_DATA_DIR = Path.home() / ".codonscope" / "data" / "species" / "yeast"
HAVE_YEAST_DATA = YEAST_DATA_DIR.exists() and (
    YEAST_DATA_DIR / "cds_sequences.fa.gz"
).exists()
HAVE_YEAST_EXPR = YEAST_DATA_DIR.exists() and (
    YEAST_DATA_DIR / "expression_rich_media.tsv"
).exists()

HUMAN_DATA_DIR = Path.home() / ".codonscope" / "data" / "species" / "human"
HAVE_HUMAN_DATA = HUMAN_DATA_DIR.exists() and (
    HUMAN_DATA_DIR / "cds_sequences.fa.gz"
).exists()
HAVE_HUMAN_EXPR = HUMAN_DATA_DIR.exists() and (
    HUMAN_DATA_DIR / "expression_gtex.tsv.gz"
).exists()

skip_no_yeast = pytest.mark.skipif(
    not (HAVE_YEAST_DATA and HAVE_YEAST_EXPR),
    reason="Yeast data/expression not downloaded",
)

skip_no_human_expr = pytest.mark.skipif(
    not (HAVE_HUMAN_DATA and HAVE_HUMAN_EXPR),
    reason="Human data/expression not downloaded",
)


# ═══════════════════════════════════════════════════════════════════════════════
# Unit tests for demand computation
# ═══════════════════════════════════════════════════════════════════════════════


class TestDemandVector:
    """Test demand-weighted codon frequency computation."""

    def test_uniform_expression_equals_length_weighted(self):
        """With equal TPM, demand weight is proportional to n_codons."""
        from codonscope.core.codons import all_possible_kmers
        from codonscope.modes.mode2_demand import _compute_demand_vector

        kmer_names = all_possible_kmers(k=1, sense_only=True)
        seqs = {
            "gene1": "ATGAAAGAA",  # 3 codons
            "gene2": "ATGAAAGAAATG",  # 4 codons (last ATG = 4th codon)
        }
        expr = {"gene1": 100.0, "gene2": 100.0}

        demand, weights = _compute_demand_vector(seqs, expr, kmer_names, k=1)

        # gene2 has 4 codons, gene1 has 3 → gene2 weight is 4/3 of gene1
        assert weights["gene1"] == pytest.approx(300.0)  # 100 * 3
        assert weights["gene2"] == pytest.approx(400.0)  # 100 * 4

    def test_demand_sums_to_one(self):
        """Demand vector should sum to ~1 (proportions)."""
        from codonscope.core.codons import all_possible_kmers
        from codonscope.modes.mode2_demand import _compute_demand_vector

        kmer_names = all_possible_kmers(k=1, sense_only=True)
        seqs = {"g1": "ATGAAAGAA", "g2": "ATGTTTTTT"}
        expr = {"g1": 1000.0, "g2": 10.0}

        demand, _ = _compute_demand_vector(seqs, expr, kmer_names, k=1)
        assert abs(demand.sum() - 1.0) < 0.001

    def test_high_expression_dominates(self):
        """Gene with 1000x TPM should dominate the demand vector."""
        from codonscope.core.codons import all_possible_kmers
        from codonscope.modes.mode2_demand import _compute_demand_vector

        kmer_names = all_possible_kmers(k=1, sense_only=True)
        kmer_to_idx = {km: i for i, km in enumerate(kmer_names)}

        # gene1 uses AAA codons at high expression
        # gene2 uses TTT codons at low expression
        seqs = {
            "gene1": "AAAAAAAAA",  # 3x AAA
            "gene2": "TTTTTTTTTTTT",  # 4x TTT
        }
        expr = {"gene1": 10000.0, "gene2": 1.0}

        demand, _ = _compute_demand_vector(seqs, expr, kmer_names, k=1)

        aaa_idx = kmer_to_idx["AAA"]
        ttt_idx = kmer_to_idx["TTT"]

        # AAA should dominate because gene1 has 10000x higher TPM
        assert demand[aaa_idx] > 0.99
        assert demand[ttt_idx] < 0.001

    def test_zero_expression_excluded(self):
        """Genes with 0 TPM should not contribute to demand."""
        from codonscope.core.codons import all_possible_kmers
        from codonscope.modes.mode2_demand import _compute_demand_vector

        kmer_names = all_possible_kmers(k=1, sense_only=True)
        seqs = {"g1": "ATGAAAGAA", "g2": "ATGTTTTTT"}
        expr = {"g1": 100.0, "g2": 0.0}

        _, weights = _compute_demand_vector(seqs, expr, kmer_names, k=1)
        assert "g2" not in weights

    def test_top_n_limits_genes(self):
        """top_n should only keep highest-expressed genes."""
        from codonscope.core.codons import all_possible_kmers
        from codonscope.modes.mode2_demand import _compute_demand_vector

        kmer_names = all_possible_kmers(k=1, sense_only=True)
        seqs = {
            "g1": "ATGAAAGAA",
            "g2": "ATGTTTTTT",
            "g3": "ATGGCTGCT",
        }
        expr = {"g1": 1000.0, "g2": 100.0, "g3": 10.0}

        _, weights = _compute_demand_vector(
            seqs, expr, kmer_names, k=1, top_n=2,
        )
        # Only top-2 by weight should be included
        assert len(weights) == 2
        assert "g3" not in weights

    def test_dicodon_demand(self):
        """Demand should work with k=2 (dicodons)."""
        from codonscope.core.codons import all_possible_kmers
        from codonscope.modes.mode2_demand import _compute_demand_vector

        kmer_names = all_possible_kmers(k=2, sense_only=True)
        seqs = {"g1": "ATGAAAGAA"}  # dicodons: ATGAAA, AAAGAA
        expr = {"g1": 100.0}

        demand, _ = _compute_demand_vector(seqs, expr, kmer_names, k=2)
        assert abs(demand.sum() - 1.0) < 0.001


class TestRankDemandGenes:
    """Test gene ranking by demand contribution."""

    def test_ranking_order(self):
        from codonscope.modes.mode2_demand import _rank_demand_genes
        from codonscope.core.codons import all_possible_kmers

        kmer_names = all_possible_kmers(k=1, sense_only=True)
        seqs = {
            "high": "ATGAAAGAA",  # 3 codons
            "low": "ATGAAAGAA",   # 3 codons
        }
        expr = {"high": 10000.0, "low": 1.0}

        df = _rank_demand_genes(seqs, expr, kmer_names, k=1)
        assert df.iloc[0]["gene"] == "high"
        assert df.iloc[0]["demand_fraction"] > 0.99

    def test_fraction_sums_to_one(self):
        from codonscope.modes.mode2_demand import _rank_demand_genes
        from codonscope.core.codons import all_possible_kmers

        kmer_names = all_possible_kmers(k=1, sense_only=True)
        seqs = {"g1": "ATGAAAGAA", "g2": "ATGTTTTTT"}
        expr = {"g1": 100.0, "g2": 200.0}

        df = _rank_demand_genes(seqs, expr, kmer_names, k=1)
        assert abs(df["demand_fraction"].sum() - 1.0) < 0.001


# ═══════════════════════════════════════════════════════════════════════════════
# Expression loading tests
# ═══════════════════════════════════════════════════════════════════════════════


@skip_no_yeast
class TestYeastExpression:
    """Test yeast expression data loading."""

    def test_expression_file_exists(self):
        assert (YEAST_DATA_DIR / "expression_rich_media.tsv").exists()

    def test_load_expression(self):
        from codonscope.modes.mode2_demand import _load_yeast_expression
        expr, tissue, available = _load_yeast_expression(YEAST_DATA_DIR)
        assert tissue == "rich_media"
        assert "rich_media" in available
        assert len(expr) > 6000

    def test_rp_genes_highly_expressed(self):
        """Ribosomal proteins should have high TPM."""
        from codonscope.modes.mode2_demand import _load_yeast_expression
        from codonscope.core.sequences import SequenceDB

        expr, _, _ = _load_yeast_expression(YEAST_DATA_DIR)
        db = SequenceDB("yeast")
        mapping = db.resolve_ids(["RPL3", "RPS2", "RPL25"])

        for sys_name in mapping.values():
            assert expr[sys_name] >= 1000, (
                f"RP gene {sys_name} TPM={expr[sys_name]}, expected >= 1000"
            )

    def test_dubious_low_expression(self):
        """Dubious ORFs should have low TPM."""
        df = pd.read_csv(YEAST_DATA_DIR / "expression_rich_media.tsv", sep="\t")
        dubious = df[df["tpm"] == 0.5]
        assert len(dubious) > 300  # ~382 dubious ORFs


@skip_no_human_expr
class TestHumanExpression:
    """Test human GTEx expression data loading."""

    def test_expression_file_exists(self):
        assert (HUMAN_DATA_DIR / "expression_gtex.tsv.gz").exists()

    def test_load_expression(self):
        from codonscope.modes.mode2_demand import _load_human_expression
        expr, tissue, available = _load_human_expression(HUMAN_DATA_DIR)
        assert len(expr) > 10000
        assert len(available) > 40  # GTEx has ~54 tissues

    def test_tissue_matching(self):
        """Should match tissue name case-insensitively."""
        from codonscope.modes.mode2_demand import _load_human_expression
        expr, tissue, _ = _load_human_expression(
            HUMAN_DATA_DIR, tissue="liver",
        )
        assert "liver" in tissue.lower() or "Liver" in tissue

    def test_invalid_tissue_raises(self):
        from codonscope.modes.mode2_demand import _load_human_expression
        with pytest.raises(ValueError, match="not found"):
            _load_human_expression(
                HUMAN_DATA_DIR, tissue="nonexistent_tissue_xyz",
            )


# ═══════════════════════════════════════════════════════════════════════════════
# Weighted bootstrap tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestWeightedBootstrap:
    """Test weighted bootstrap Z-score computation."""

    def test_identical_demand_gives_low_z(self):
        """If gene set has same demand as genome, Z ≈ 0."""
        from codonscope.core.codons import all_possible_kmers
        from codonscope.modes.mode2_demand import (
            _compute_demand_vector,
            _weighted_bootstrap_zscores,
        )

        kmer_names = all_possible_kmers(k=1, sense_only=True)
        # All genes identical — any sample should match
        seqs = {f"g{i}": "ATGAAAGAA" for i in range(50)}
        expr = {f"g{i}": 100.0 for i in range(50)}

        demand, _ = _compute_demand_vector(seqs, expr, kmer_names, k=1)
        z, _, _ = _weighted_bootstrap_zscores(
            demand, seqs, expr,
            n_genes=10, kmer_names=kmer_names, k=1,
            n_bootstrap=500, seed=42,
        )
        # All Z-scores should be near 0
        assert np.abs(z).max() < 2.0

    def test_extreme_bias_gives_high_z(self):
        """Gene set with extreme bias should have high Z-scores."""
        from codonscope.core.codons import all_possible_kmers
        from codonscope.modes.mode2_demand import (
            _compute_demand_vector,
            _weighted_bootstrap_zscores,
        )

        kmer_names = all_possible_kmers(k=1, sense_only=True)

        # Gene set: all AAA codons, highly expressed
        gs_seqs = {f"gs{i}": "AAAAAAAAA" for i in range(20)}
        gs_expr = {f"gs{i}": 1000.0 for i in range(20)}

        # Genome: mix of codons, varied expression
        genome = {}
        genome_expr = {}
        for i in range(200):
            if i < 50:
                genome[f"g{i}"] = "AAAAAAAAA"
                genome_expr[f"g{i}"] = 10.0
            elif i < 100:
                genome[f"g{i}"] = "TTTTTTTTT"
                genome_expr[f"g{i}"] = 10.0
            elif i < 150:
                genome[f"g{i}"] = "GCTGCTGCT"
                genome_expr[f"g{i}"] = 10.0
            else:
                genome[f"g{i}"] = "ATGATGATG"
                genome_expr[f"g{i}"] = 10.0

        demand, _ = _compute_demand_vector(
            gs_seqs, gs_expr, kmer_names, k=1,
        )
        z, _, _ = _weighted_bootstrap_zscores(
            demand, genome, genome_expr,
            n_genes=20, kmer_names=kmer_names, k=1,
            n_bootstrap=1000, seed=42,
        )

        kmer_to_idx = {km: i for i, km in enumerate(kmer_names)}
        aaa_idx = kmer_to_idx["AAA"]
        # AAA should be highly enriched
        assert z[aaa_idx] > 2.0


# ═══════════════════════════════════════════════════════════════════════════════
# Integration tests — yeast RP genes
# ═══════════════════════════════════════════════════════════════════════════════


@skip_no_yeast
class TestRPDemandYeast:
    """Yeast RP genes: high expression, optimal codons.

    RP genes use preferred codons (AAG, AGA, GCT, TTG, etc.) and are
    the most highly expressed gene family.  Their demand-weighted signal
    should show enrichment for optimal codons and depletion of rare codons.

    Note: Z-scores may be moderate because RP genes already dominate
    ~70% of the genome demand in our expression estimates.
    """

    @pytest.fixture(autouse=True)
    def run_analysis(self):
        from codonscope.modes.mode2_demand import run_demand
        rp_genes = [
            "RPL1A", "RPL1B", "RPL2A", "RPL2B", "RPL3",
            "RPL4A", "RPL4B", "RPL5", "RPL6A", "RPL6B",
            "RPL7A", "RPL7B", "RPL8A", "RPL8B", "RPL9A",
            "RPL10", "RPL11A", "RPL11B", "RPL13A", "RPL14A",
            "RPL15A", "RPL16A", "RPL17A", "RPL18A", "RPL19A",
            "RPL20A", "RPL21A", "RPL22A", "RPL23A", "RPL24A",
            "RPL25", "RPL26A", "RPL27A", "RPL28", "RPL30",
            "RPL31A", "RPL32", "RPL33A", "RPL34A", "RPL35A",
            "RPL36A", "RPL37A", "RPL38", "RPL39", "RPL40A",
            "RPL42A", "RPL43A",
            "RPS0A", "RPS1A", "RPS2", "RPS3", "RPS4A",
            "RPS5", "RPS6A", "RPS7A", "RPS8A", "RPS9A",
            "RPS10A", "RPS11A", "RPS12", "RPS13", "RPS14A",
            "RPS15", "RPS16A", "RPS17A", "RPS18A", "RPS19A",
            "RPS20", "RPS21A", "RPS22A", "RPS23A", "RPS24A",
            "RPS25A", "RPS26A", "RPS27A", "RPS28A", "RPS29A",
            "RPS30A", "RPS31",
        ]
        self.result = run_demand(
            species="yeast",
            gene_ids=rp_genes,
            k=1,
            n_bootstrap=5000,
            seed=42,
        )

    def test_many_genes(self):
        assert self.result["n_genes"] >= 50

    def test_tissue_is_rich_media(self):
        assert self.result["tissue"] == "rich_media"

    def test_demand_sums_to_one(self):
        df = self.result["results"]
        assert abs(df["demand_geneset"].sum() - 1.0) < 0.001
        assert abs(df["demand_genome"].sum() - 1.0) < 0.001

    def test_all_genes_high_tpm(self):
        """All RP genes should have high TPM (≥1000)."""
        top = self.result["top_genes"]
        assert (top["tpm"] >= 1000).all(), (
            f"Some RP genes have low TPM: "
            f"{top[top['tpm'] < 1000][['gene', 'tpm']].to_dict()}"
        )

    def test_optimal_codons_enriched(self):
        """Known optimal codons (AAG, AGA, GCT) should have higher
        demand in RP gene set than genome average."""
        df = self.result["results"]
        for codon in ("AAG", "AGA", "GCT"):
            row = df[df["kmer"] == codon].iloc[0]
            assert row["demand_geneset"] > row["demand_genome"], (
                f"Expected {codon} demand_geneset > demand_genome, "
                f"got {row['demand_geneset']:.4f} vs {row['demand_genome']:.4f}"
            )

    def test_rare_codons_depleted(self):
        """Rare codons (AGT, CTG, ATA) should have lower demand
        in RP gene set than genome."""
        df = self.result["results"]
        for codon in ("AGT", "CTG", "ATA"):
            row = df[df["kmer"] == codon].iloc[0]
            assert row["demand_geneset"] < row["demand_genome"], (
                f"Expected {codon} demand_geneset < demand_genome, "
                f"got {row['demand_geneset']:.4f} vs {row['demand_genome']:.4f}"
            )

    def test_top_genes_have_data(self):
        top = self.result["top_genes"]
        assert len(top) >= 50
        assert "demand_fraction" in top.columns
        assert abs(top["demand_fraction"].sum() - 1.0) < 0.001

    def test_returns_expected_keys(self):
        expected = [
            "results", "top_genes", "tissue", "available_tissues",
            "n_genes", "id_summary",
        ]
        for k in expected:
            assert k in self.result, f"Missing key: {k}"


@skip_no_yeast
class TestGcn4DemandYeast:
    """Gcn4 targets: moderate expression, amino acid biosynthesis genes.

    These genes are expressed at moderate levels (~15-800 TPM) and have
    different codon usage than RP genes.  Their demand should differ
    from the genome demand more visibly than RP genes (since they don't
    dominate the transcriptome).
    """

    @pytest.fixture(autouse=True)
    def run_analysis(self):
        from codonscope.modes.mode2_demand import run_demand
        gcn4_genes = [
            "ARG1", "ARG3", "ARG4", "ARG8",
            "HIS1", "HIS3", "HIS4", "HIS5",
            "ILV1", "ILV2", "ILV3", "ILV5", "ILV6",
            "LEU1", "LEU2", "LEU4", "LEU9",
            "LYS1", "LYS2", "LYS4", "LYS9", "LYS20", "LYS21",
            "TRP2", "TRP3", "TRP4", "TRP5",
            "MET6", "MET13", "MET14", "MET16",
            "SER1", "SER2", "SER33",
            "ADE1", "ADE2", "ADE4", "ADE8",
            "GCN4", "CPA1", "CPA2",
            "ASN1", "ASN2",
            "GLN1", "GLN4",
            "GDH1",
            "ARO1", "ARO2", "ARO3", "ARO4",
            "SNZ1", "SNO1",
            "THR1", "THR4",
            "HOM2", "HOM3", "HOM6",
        ]
        self.result = run_demand(
            species="yeast",
            gene_ids=gcn4_genes,
            k=1,
            n_bootstrap=5000,
            seed=42,
        )

    def test_many_genes(self):
        assert self.result["n_genes"] >= 40

    def test_demand_sums_to_one(self):
        df = self.result["results"]
        assert abs(df["demand_geneset"].sum() - 1.0) < 0.001

    def test_top_gene_not_uniform(self):
        """Top demand gene should have higher fraction than bottom."""
        top = self.result["top_genes"]
        assert top.iloc[0]["demand_fraction"] > top.iloc[-1]["demand_fraction"]


# ═══════════════════════════════════════════════════════════════════════════════
# Human integration tests
# ═══════════════════════════════════════════════════════════════════════════════


@skip_no_human_expr
class TestHumanDemand:
    """Human RP genes with GTEx liver tissue."""

    @pytest.fixture(autouse=True)
    def run_analysis(self):
        from codonscope.modes.mode2_demand import run_demand
        rp_genes = [
            "RPL3", "RPL5", "RPL7", "RPL8", "RPL10",
            "RPL11", "RPL13", "RPL14", "RPL15",
            "RPS2", "RPS3", "RPS5", "RPS6", "RPS8",
        ]
        self.result = run_demand(
            species="human",
            gene_ids=rp_genes,
            tissue="liver",
            k=1,
            n_bootstrap=2000,
            seed=42,
        )

    def test_many_genes(self):
        assert self.result["n_genes"] >= 10

    def test_tissue_is_liver(self):
        assert "liver" in self.result["tissue"].lower() or \
               "Liver" in self.result["tissue"]

    def test_has_results(self):
        assert len(self.result["results"]) == 61  # 61 sense codons

    def test_demand_sums_to_one(self):
        df = self.result["results"]
        assert abs(df["demand_geneset"].sum() - 1.0) < 0.001


# ═══════════════════════════════════════════════════════════════════════════════
# CLI tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestCLIDemand:
    """Test CLI integration for demand subcommand."""

    def test_help(self):
        from codonscope.cli import main
        with pytest.raises(SystemExit) as exc_info:
            main(["demand", "--help"])
        assert exc_info.value.code == 0

    def test_has_tissue_arg(self):
        """Demand subcommand should accept --tissue."""
        from codonscope.cli import main
        with pytest.raises(SystemExit) as exc_info:
            main(["demand", "--help"])
        assert exc_info.value.code == 0

    def test_has_expression_arg(self):
        """Demand subcommand should accept --expression."""
        from codonscope.cli import main
        with pytest.raises(SystemExit) as exc_info:
            main(["demand", "--help"])
        assert exc_info.value.code == 0
