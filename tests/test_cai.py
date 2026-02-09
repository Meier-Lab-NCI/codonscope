"""Tests for CAI (Codon Adaptation Index) computation."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from codonscope.core.cai import (
    _AA_TO_CODONS,
    _SINGLE_CODON_AAS,
    cai_analysis,
    compute_cai,
    compute_reference_weights,
)
from codonscope.core.codons import CODON_TABLE, SENSE_CODONS
from codonscope.core.sequences import SequenceDB

DATA_DIR = os.environ.get(
    "CODONSCOPE_TEST_DATA_DIR",
    str(Path.home() / ".codonscope" / "data" / "species"),
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def yeast_db():
    return SequenceDB("yeast", data_dir=DATA_DIR)


@pytest.fixture(scope="session")
def yeast_weights():
    return compute_reference_weights("yeast", data_dir=DATA_DIR)


@pytest.fixture(scope="session")
def human_weights():
    return compute_reference_weights("human", data_dir=DATA_DIR)


@pytest.fixture(scope="session")
def yeast_rp_genes():
    """Yeast ribosomal protein gene names."""
    db = SequenceDB("yeast", data_dir=DATA_DIR)
    meta = db.get_gene_metadata()
    rp_genes = meta[
        meta["common_name"].str.startswith("RPL", na=False)
        | meta["common_name"].str.startswith("RPS", na=False)
    ]["common_name"].tolist()
    return rp_genes


@pytest.fixture(scope="session")
def yeast_gcn4_genes():
    """Yeast Gcn4 target gene names (amino acid biosynthesis)."""
    return [
        "ARG1", "ARG3", "ARG4", "ARG5,6", "ARG8",
        "HIS1", "HIS4", "HIS5",
        "ILV1", "ILV2", "ILV5",
        "LEU1", "LEU4",
        "LYS1", "LYS2", "LYS9", "LYS20",
        "TRP2", "TRP3", "TRP4", "TRP5",
        "SER1", "SER33",
        "MET6", "MET17",
        "THR1", "THR4",
        "ADE1", "ADE4",
        "GLN1",
        "ASN1", "ASN2",
        "GCN4",
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# Unit tests — compute_reference_weights
# ═══════════════════════════════════════════════════════════════════════════════

class TestReferenceWeights:
    def test_all_sense_codons_present(self, yeast_weights):
        """All 61 sense codons should have a weight."""
        for c in SENSE_CODONS:
            assert c in yeast_weights, f"Missing weight for {c}"

    def test_weights_in_valid_range(self, yeast_weights):
        """All weights should be in (0, 1]."""
        for c, w in yeast_weights.items():
            assert 0 < w <= 1.0, f"Weight for {c} out of range: {w}"

    def test_max_per_family_is_one(self, yeast_weights):
        """Within each AA family, the max weight should be 1.0."""
        for aa, codons in _AA_TO_CODONS.items():
            max_w = max(yeast_weights[c] for c in codons)
            assert np.isclose(max_w, 1.0), f"Max weight for {aa} is {max_w}, not 1.0"

    def test_single_codon_aas_are_one(self, yeast_weights):
        """Met (ATG) and Trp (TGG) should have weight 1.0."""
        assert np.isclose(yeast_weights["ATG"], 1.0)
        assert np.isclose(yeast_weights["TGG"], 1.0)

    def test_human_weights_valid(self, human_weights):
        """Human weights should also satisfy basic constraints."""
        for c in SENSE_CODONS:
            assert c in human_weights
            assert 0 < human_weights[c] <= 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# Unit tests — compute_cai
# ═══════════════════════════════════════════════════════════════════════════════

class TestComputeCAI:
    def test_cai_values_in_range(self, yeast_weights):
        """CAI values should be in (0, 1]."""
        # Use a simple test sequence: all optimal codons for Lys (AAG has higher w)
        seqs = {"test_gene": "AAGAAGAAGAAGAAG"}
        result = compute_cai(seqs, yeast_weights)
        assert 0 < result["test_gene"] <= 1.0

    def test_cai_empty_sequence(self, yeast_weights):
        """Empty codons should return 0."""
        seqs = {"test_gene": ""}
        result = compute_cai(seqs, yeast_weights)
        assert result["test_gene"] == 0.0

    def test_cai_met_only(self, yeast_weights):
        """A sequence of only Met/Trp codons should return 0 (all skipped)."""
        seqs = {"test_gene": "ATGATGATG"}
        result = compute_cai(seqs, yeast_weights)
        # Met is skipped — no informative codons
        assert result["test_gene"] == 0.0

    def test_optimal_codons_high_cai(self, yeast_weights):
        """Sequence using only top-weight codons should have CAI near 1.0."""
        # Build a sequence using the max-weight codon for each AA
        best_codons = []
        for aa, codons in _AA_TO_CODONS.items():
            if aa in _SINGLE_CODON_AAS:
                continue
            best = max(codons, key=lambda c: yeast_weights.get(c, 0))
            best_codons.append(best)
        # Use each best codon twice
        seq = "".join(best_codons * 2)
        result = compute_cai({"optimal": seq}, yeast_weights)
        assert result["optimal"] > 0.95, f"Optimal CAI should be near 1.0, got {result['optimal']}"

    def test_multiple_genes(self, yeast_weights):
        """Multiple genes should each get a CAI value."""
        seqs = {
            "gene1": "AAGAAGAAGAAGAAG",
            "gene2": "AAAAAAAAAAAAAAA",
        }
        result = compute_cai(seqs, yeast_weights)
        assert "gene1" in result
        assert "gene2" in result
        assert len(result) == 2


# ═══════════════════════════════════════════════════════════════════════════════
# Integration tests — cai_analysis with real data
# ═══════════════════════════════════════════════════════════════════════════════

class TestCAIAnalysisYeast:
    def test_rp_genes_high_cai(self, yeast_rp_genes):
        """Yeast RP genes should have high CAI (>0.7)."""
        result = cai_analysis("yeast", yeast_rp_genes, data_dir=DATA_DIR)
        assert result["geneset_mean"] > 0.7, (
            f"RP genes mean CAI {result['geneset_mean']:.3f} should be > 0.7"
        )

    def test_rp_genes_mann_whitney_significant(self, yeast_rp_genes):
        """RP genes CAI should differ significantly from genome."""
        result = cai_analysis("yeast", yeast_rp_genes, data_dir=DATA_DIR)
        assert result["mann_whitney_p"] < 0.001, (
            f"Mann-Whitney p={result['mann_whitney_p']:.2e} should be < 0.001"
        )

    def test_rp_percentile_high(self, yeast_rp_genes):
        """RP genes should rank in top percentiles of genome CAI."""
        result = cai_analysis("yeast", yeast_rp_genes, data_dir=DATA_DIR)
        assert result["percentile_rank"] > 80, (
            f"RP percentile {result['percentile_rank']:.1f} should be > 80"
        )

    def test_gcn4_moderate_cai(self, yeast_gcn4_genes):
        """Gcn4 targets should have moderate CAI, lower than RP genes."""
        result = cai_analysis("yeast", yeast_gcn4_genes, data_dir=DATA_DIR)
        assert 0.2 < result["geneset_mean"] < 0.8, (
            f"Gcn4 mean CAI {result['geneset_mean']:.3f} should be moderate"
        )

    def test_genome_distribution_spread(self, yeast_rp_genes):
        """Genome CAI should have reasonable spread."""
        result = cai_analysis("yeast", yeast_rp_genes, data_dir=DATA_DIR)
        genome_cai = result["genome_per_gene"]["cai"].values
        assert genome_cai.std() > 0.05, (
            f"Genome CAI std {genome_cai.std():.3f} should be > 0.05"
        )

    def test_result_structure(self, yeast_rp_genes):
        """Result dict should have all expected keys."""
        result = cai_analysis("yeast", yeast_rp_genes, data_dir=DATA_DIR)
        expected_keys = {
            "per_gene", "geneset_mean", "geneset_median",
            "genome_per_gene", "genome_mean", "genome_median",
            "percentile_rank", "mann_whitney_u", "mann_whitney_p",
            "reference_n_genes", "weights",
        }
        assert expected_keys.issubset(result.keys())

    def test_per_gene_dataframe(self, yeast_rp_genes):
        """per_gene DataFrame should have gene and cai columns."""
        result = cai_analysis("yeast", yeast_rp_genes, data_dir=DATA_DIR)
        df = result["per_gene"]
        assert "gene" in df.columns
        assert "cai" in df.columns
        assert len(df) > 50  # Should have most RP genes

    def test_weights_dict(self, yeast_rp_genes):
        """weights should have entries for all 61 sense codons."""
        result = cai_analysis("yeast", yeast_rp_genes, data_dir=DATA_DIR)
        assert len(result["weights"]) == 61


class TestCAIAnalysisHuman:
    def test_human_rp_genes_high_cai(self):
        """Human RP genes should have high CAI."""
        rp_genes = [
            "RPL3", "RPL4", "RPL5", "RPL6", "RPL7", "RPL8", "RPL9",
            "RPL10", "RPL11", "RPL13", "RPL14",
            "RPS3", "RPS5", "RPS6", "RPS8",
        ]
        result = cai_analysis("human", rp_genes, data_dir=DATA_DIR)
        assert result["geneset_mean"] > 0.6, (
            f"Human RP mean CAI {result['geneset_mean']:.3f} should be > 0.6"
        )

    def test_human_genome_has_spread(self):
        """Human genome CAI should have reasonable distribution."""
        rp_genes = ["RPL3", "RPL4", "RPL5", "RPS3", "RPS5"]
        result = cai_analysis("human", rp_genes, data_dir=DATA_DIR)
        genome_cai = result["genome_per_gene"]["cai"].values
        assert genome_cai.std() > 0.03


class TestCAIEdgeCases:
    def test_single_gene(self):
        """Should work with a single gene."""
        result = cai_analysis("yeast", ["RPL3"], data_dir=DATA_DIR)
        assert result["geneset_mean"] > 0
        assert len(result["per_gene"]) == 1

    def test_small_gene_set(self):
        """Should work with a small gene set."""
        result = cai_analysis("yeast", ["RPL3", "RPL4A", "RPL5"], data_dir=DATA_DIR)
        assert result["geneset_mean"] > 0
        assert len(result["per_gene"]) == 3
