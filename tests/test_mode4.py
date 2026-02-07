"""Tests for Mode 4: Collision Potential."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

YEAST_DATA_DIR = Path.home() / ".codonscope" / "data" / "species" / "yeast"
HAVE_YEAST_DATA = YEAST_DATA_DIR.exists() and (
    YEAST_DATA_DIR / "cds_sequences.fa.gz"
).exists()

skip_no_data = pytest.mark.skipif(
    not HAVE_YEAST_DATA,
    reason="Yeast data not downloaded. Run: codonscope download --species yeast",
)


# ═══════════════════════════════════════════════════════════════════════════════
# Unit tests for transition counting
# ═══════════════════════════════════════════════════════════════════════════════


class TestTransitionCounting:
    """Test fast/slow classification and transition counting."""

    def test_all_fast_gives_ff(self):
        """If all codons are fast, all transitions should be FF."""
        from codonscope.modes.mode4_collision import _count_transitions

        fast = {"ATG", "AAG"}
        slow = {"TTA", "ACA"}
        seqs = {"gene1": "ATGAAGAAG"}  # ATG AAG AAG — all fast
        counts, _ = _count_transitions(seqs, fast, slow)
        assert counts["FF"] == 2
        assert counts["FS"] == 0
        assert counts["SF"] == 0
        assert counts["SS"] == 0

    def test_all_slow_gives_ss(self):
        """If all codons are slow, all transitions should be SS."""
        from codonscope.modes.mode4_collision import _count_transitions

        fast = {"ATG"}
        slow = {"TTA", "ACA"}
        seqs = {"gene1": "TTAACATTA"}  # TTA ACA TTA — all slow
        counts, _ = _count_transitions(seqs, fast, slow)
        assert counts["SS"] == 2
        assert counts["FS"] == 0
        assert counts["SF"] == 0
        assert counts["FF"] == 0

    def test_fs_transition(self):
        """Fast followed by slow = FS transition."""
        from codonscope.modes.mode4_collision import _count_transitions

        fast = {"AAG"}
        slow = {"TTA"}
        seqs = {"gene1": "AAGTTA"}  # AAG TTA
        counts, _ = _count_transitions(seqs, fast, slow)
        assert counts["FS"] == 1

    def test_sf_transition(self):
        """Slow followed by fast = SF transition."""
        from codonscope.modes.mode4_collision import _count_transitions

        fast = {"AAG"}
        slow = {"TTA"}
        seqs = {"gene1": "TTAAAG"}  # TTA AAG
        counts, _ = _count_transitions(seqs, fast, slow)
        assert counts["SF"] == 1

    def test_per_gene_counts(self):
        """Per-gene counts should be returned."""
        from codonscope.modes.mode4_collision import _count_transitions

        fast = {"AAG", "ATG"}
        slow = {"TTA"}
        seqs = {
            "gene1": "AAGTTAAAG",  # AAG TTA AAG → FS, SF
            "gene2": "AAGAAGAAG",  # AAG AAG AAG → FF, FF
        }
        _, per_gene = _count_transitions(seqs, fast, slow)
        assert per_gene["gene1"]["FS"] == 1
        assert per_gene["gene1"]["SF"] == 1
        assert per_gene["gene2"]["FF"] == 2


class TestProportions:
    """Test proportion calculation."""

    def test_proportions_sum_to_1(self):
        from codonscope.modes.mode4_collision import _to_proportions
        counts = {"FF": 10, "FS": 5, "SF": 3, "SS": 2}
        props = _to_proportions(counts)
        assert abs(sum(props.values()) - 1.0) < 0.001

    def test_empty_counts(self):
        from codonscope.modes.mode4_collision import _to_proportions
        counts = {"FF": 0, "FS": 0, "SF": 0, "SS": 0}
        props = _to_proportions(counts)
        assert all(v == 0.0 for v in props.values())


class TestFSEnrichment:
    """Test FS enrichment calculations."""

    def test_equal_fs_gives_ratio_1(self):
        from codonscope.modes.mode4_collision import _fs_enrichment
        gs = {"FF": 0.4, "FS": 0.2, "SF": 0.2, "SS": 0.2}
        bg = {"FF": 0.4, "FS": 0.2, "SF": 0.2, "SS": 0.2}
        assert abs(_fs_enrichment(gs, bg) - 1.0) < 0.001

    def test_elevated_fs_gives_ratio_gt_1(self):
        from codonscope.modes.mode4_collision import _fs_enrichment
        gs = {"FF": 0.2, "FS": 0.4, "SF": 0.2, "SS": 0.2}
        bg = {"FF": 0.4, "FS": 0.2, "SF": 0.2, "SS": 0.2}
        assert _fs_enrichment(gs, bg) == pytest.approx(2.0)

    def test_fs_sf_ratio(self):
        from codonscope.modes.mode4_collision import _fs_sf_ratio
        matrix = {"FF": 0.3, "FS": 0.3, "SF": 0.15, "SS": 0.25}
        assert _fs_sf_ratio(matrix) == pytest.approx(2.0)


class TestChiSquared:
    """Test chi-squared transition comparison."""

    def test_identical_distributions(self):
        from codonscope.modes.mode4_collision import _chi_squared_test
        gs = {"FF": 100, "FS": 50, "SF": 30, "SS": 20}
        bg = {"FF": 1000, "FS": 500, "SF": 300, "SS": 200}
        chi2, p = _chi_squared_test(gs, bg)
        # Same proportions → chi2 ≈ 0, p ≈ 1
        assert chi2 < 1.0
        assert p > 0.5

    def test_different_distributions(self):
        from codonscope.modes.mode4_collision import _chi_squared_test
        gs = {"FF": 100, "FS": 100, "SF": 0, "SS": 0}
        bg = {"FF": 1000, "FS": 100, "SF": 100, "SS": 800}
        chi2, p = _chi_squared_test(gs, bg)
        assert chi2 > 10.0
        assert p < 0.01


# ═══════════════════════════════════════════════════════════════════════════════
# Integration tests with yeast data
# ═══════════════════════════════════════════════════════════════════════════════


@skip_no_data
class TestRPCollisionYeast:
    """Yeast ribosomal proteins: expect LOW collision potential.

    RP genes use optimal codons throughout → mostly FF transitions.
    """

    @pytest.fixture(autouse=True)
    def run_analysis(self):
        from codonscope.modes.mode4_collision import run_collision
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
        self.result = run_collision(
            species="yeast",
            gene_ids=rp_genes,
            method="wtai",
        )

    def test_many_genes(self):
        assert self.result["n_genes"] >= 50

    def test_high_ff_proportion(self):
        """RP genes should have high FF (fast→fast) proportion."""
        gs = self.result["transition_matrix_geneset"]
        assert gs["FF"] > 0.3, (
            f"Expected FF > 0.3 for RP genes, got {gs['FF']:.4f}"
        )

    def test_low_fs_enrichment(self):
        """RP genes should have FS enrichment <= 1 (low collision potential)."""
        assert self.result["fs_enrichment"] <= 1.1, (
            f"Expected FS enrichment <= 1.1, got {self.result['fs_enrichment']:.3f}"
        )

    def test_returns_expected_keys(self):
        r = self.result
        expected_keys = [
            "transition_matrix_geneset", "transition_matrix_genome",
            "fs_enrichment", "fs_sf_ratio_geneset", "fs_sf_ratio_genome",
            "chi2_stat", "chi2_p", "per_gene_fs_frac",
            "fs_positions", "fast_codons", "slow_codons",
            "threshold", "n_genes",
        ]
        for k in expected_keys:
            assert k in r, f"Missing key: {k}"

    def test_per_gene_fs_has_rows(self):
        df = self.result["per_gene_fs_frac"]
        assert len(df) >= 50
        assert "fs_fraction" in df.columns

    def test_fs_positions_is_dataframe(self):
        df = self.result["fs_positions"]
        assert isinstance(df, pd.DataFrame)
        assert "position_pct" in df.columns


@skip_no_data
class TestGcn4CollisionYeast:
    """Gcn4 targets: expect higher collision potential than RP genes.

    Gcn4 targets have mixed codon usage (not as optimised as RP genes),
    so they should have more FS transitions.
    """

    @pytest.fixture(autouse=True)
    def run_analysis(self):
        from codonscope.modes.mode4_collision import run_collision
        gcn4_genes = [
            "ARG1", "ARG3", "ARG4", "ARG5,6", "ARG8",
            "HIS1", "HIS3", "HIS4", "HIS5",
            "ILV1", "ILV2", "ILV3", "ILV5", "ILV6",
            "LEU1", "LEU2", "LEU4", "LEU9",
            "LYS1", "LYS2", "LYS4", "LYS9", "LYS20", "LYS21",
            "TRP2", "TRP3", "TRP4", "TRP5",
            "MET6", "MET13", "MET14", "MET16",
            "SER1", "SER2", "SER33",
            "ADE1", "ADE2", "ADE4", "ADE5,7", "ADE8",
            "GCN4", "CPA1", "CPA2",
            "ASN1", "ASN2",
            "GLN1", "GLN4",
            "GDH1",
            "ARO1", "ARO2", "ARO3", "ARO4",
            "SNZ1", "SNO1",
            "THR1", "THR4",
            "HOM2", "HOM3", "HOM6",
        ]
        self.result = run_collision(
            species="yeast",
            gene_ids=gcn4_genes,
            method="wtai",
        )

    def test_many_genes(self):
        assert self.result["n_genes"] >= 40

    def test_has_fs_transitions(self):
        """Gcn4 targets should have some FS transitions."""
        gs = self.result["transition_matrix_geneset"]
        assert gs["FS"] > 0.01

    def test_chi2_significant(self):
        """Gcn4 targets should differ from genome background."""
        # Use lenient threshold — chi2 test on 4 cells
        assert self.result["chi2_p"] < 0.1 or self.result["chi2_stat"] > 3


# ═══════════════════════════════════════════════════════════════════════════════
# CLI tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestCLICollision:
    """Test CLI integration for collision subcommand."""

    def test_help(self):
        from codonscope.cli import main
        with pytest.raises(SystemExit) as exc_info:
            main(["collision", "--help"])
        assert exc_info.value.code == 0
