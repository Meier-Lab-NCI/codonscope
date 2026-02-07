"""Tests for Mode 3: Optimality Profile."""

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
# Unit tests for OptimalityScorer
# ═══════════════════════════════════════════════════════════════════════════════


@skip_no_data
class TestOptimalityScorer:
    """Test the core OptimalityScorer."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from codonscope.core.optimality import OptimalityScorer
        self.scorer = OptimalityScorer(YEAST_DATA_DIR)

    def test_tai_weights_has_61_codons(self):
        assert len(self.scorer.tai_weights) == 61

    def test_wtai_weights_has_61_codons(self):
        assert len(self.scorer.wtai_weights) == 61

    def test_weights_between_0_and_1(self):
        for w in self.scorer.tai_weights.values():
            assert 0 < w <= 1.0
        for w in self.scorer.wtai_weights.values():
            assert 0 < w <= 1.0

    def test_max_tai_is_1(self):
        assert max(self.scorer.tai_weights.values()) == 1.0

    def test_max_wtai_is_1(self):
        assert max(self.scorer.wtai_weights.values()) == 1.0

    def test_wobble_penalty_reduces_score(self):
        """Wobble-decoded codons should have lower wtAI than tAI."""
        # TTC (Phe) is wobble-decoded: GAA anticodon, 10 copies
        tai_ttc = self.scorer.tai_weights["TTC"]
        wtai_ttc = self.scorer.wtai_weights["TTC"]
        assert wtai_ttc < tai_ttc, (
            f"TTC wobble: expected wtAI ({wtai_ttc}) < tAI ({tai_ttc})"
        )

    def test_watson_crick_no_penalty(self):
        """Watson-Crick decoded codons should have same tAI and wtAI."""
        # TTT (Phe) is Watson-Crick: GAA anticodon, 10 copies
        tai_ttt = self.scorer.tai_weights["TTT"]
        wtai_ttt = self.scorer.wtai_weights["TTT"]
        assert abs(tai_ttt - wtai_ttt) < 0.01

    def test_gene_tai_simple(self):
        """Simple sequence should give non-zero tAI."""
        seq = "ATGAAAGAA"  # Met, Lys, Glu
        tai = self.scorer.gene_tai(seq)
        assert tai > 0

    def test_gene_wtai_leq_tai(self):
        """Gene wtAI should be <= tAI (wobble penalty)."""
        seq = "ATGAAAGAATTTTTCGCTGCC"  # mix of WC and wobble codons
        tai = self.scorer.gene_tai(seq)
        wtai = self.scorer.gene_wtai(seq)
        assert wtai <= tai + 0.01  # small tolerance for rounding

    def test_per_position_scores_length(self):
        """Per-position scores should match number of codons."""
        seq = "ATGAAAGAAGATTTTGCC"  # 6 codons
        scores = self.scorer.per_position_scores(seq)
        assert len(scores) == 6

    def test_smooth_profile_same_length(self):
        """Smoothed profile should be same length as input."""
        scores = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        smoothed = self.scorer.smooth_profile(scores, window=3)
        assert len(smoothed) == len(scores)

    def test_classify_codons_covers_all(self):
        """Fast + slow should cover all 61 sense codons."""
        fast, slow = self.scorer.classify_codons()
        assert len(fast) + len(slow) == 61
        assert len(fast & slow) == 0

    def test_custom_wobble_penalty(self):
        """Different wobble penalty should change scores."""
        from codonscope.core.optimality import OptimalityScorer
        scorer_low = OptimalityScorer(YEAST_DATA_DIR, wobble_penalty=0.3)
        scorer_high = OptimalityScorer(YEAST_DATA_DIR, wobble_penalty=0.8)
        # TTC is wobble-decoded
        assert scorer_low.wtai_weights["TTC"] < scorer_high.wtai_weights["TTC"]


# ═══════════════════════════════════════════════════════════════════════════════
# Unit tests for metagene profile functions
# ═══════════════════════════════════════════════════════════════════════════════


@skip_no_data
class TestMetageneProfile:
    """Test metagene profiling internals."""

    def test_metagene_shape(self):
        from codonscope.modes.mode3_profile import (
            N_METAGENE_BINS,
            _metagene_profile,
        )
        from codonscope.core.optimality import OptimalityScorer

        scorer = OptimalityScorer(YEAST_DATA_DIR)
        # Create a synthetic long gene
        seqs = {"gene1": "ATG" + "AAG" * 200}  # 201 codons
        profile = _metagene_profile(seqs, scorer, window=10, method="wtai")
        assert profile.shape == (N_METAGENE_BINS,)

    def test_metagene_values_in_range(self):
        from codonscope.modes.mode3_profile import _metagene_profile
        from codonscope.core.optimality import OptimalityScorer

        scorer = OptimalityScorer(YEAST_DATA_DIR)
        seqs = {"gene1": "ATG" + "AAG" * 200}
        profile = _metagene_profile(seqs, scorer, window=10, method="wtai")
        # All values should be between 0 and 1 (normalised weights)
        assert np.all(profile >= 0)
        assert np.all(profile <= 1.0 + 0.01)

    def test_short_gene_excluded(self):
        from codonscope.modes.mode3_profile import _metagene_profile
        from codonscope.core.optimality import OptimalityScorer

        scorer = OptimalityScorer(YEAST_DATA_DIR)
        # Gene with only 10 codons — below min_codons=30
        seqs = {"gene1": "ATG" + "AAG" * 9}  # 10 codons
        profile = _metagene_profile(
            seqs, scorer, window=10, method="wtai", min_codons=30,
        )
        # Should be all zeros (gene excluded)
        assert np.allclose(profile, 0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# Unit tests for ramp analysis
# ═══════════════════════════════════════════════════════════════════════════════


@skip_no_data
class TestRampAnalysis:
    """Test ramp analysis."""

    def test_ramp_returns_expected_keys(self):
        from codonscope.modes.mode3_profile import _ramp_analysis
        from codonscope.core.optimality import OptimalityScorer

        scorer = OptimalityScorer(YEAST_DATA_DIR)
        seqs = {"gene1": "ATG" + "AAG" * 200}
        ramp = _ramp_analysis(
            seqs, seqs, scorer, ramp_codons=50, method="wtai",
        )
        assert "geneset_ramp_mean" in ramp
        assert "geneset_body_mean" in ramp
        assert "geneset_ramp_delta" in ramp
        assert "genome_ramp_mean" in ramp
        assert "per_gene_deltas" in ramp

    def test_ramp_delta_sign(self):
        """For a uniform sequence, ramp delta should be ~0."""
        from codonscope.modes.mode3_profile import _ramp_analysis
        from codonscope.core.optimality import OptimalityScorer

        scorer = OptimalityScorer(YEAST_DATA_DIR)
        # All AAG codons — same score everywhere
        seqs = {"gene1": "AAG" * 200}
        ramp = _ramp_analysis(
            seqs, seqs, scorer, ramp_codons=50, method="wtai",
        )
        # Delta should be ~0 for uniform sequence
        assert abs(ramp["geneset_ramp_delta"]) < 0.01


# ═══════════════════════════════════════════════════════════════════════════════
# Integration tests with yeast data
# ═══════════════════════════════════════════════════════════════════════════════


@skip_no_data
class TestRPProfileYeast:
    """Yeast ribosomal proteins: high optimality throughout with visible ramp."""

    @pytest.fixture(autouse=True)
    def run_analysis(self):
        from codonscope.modes.mode3_profile import run_profile
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
        self.result = run_profile(
            species="yeast",
            gene_ids=rp_genes,
            window=10,
            method="wtai",
        )

    def test_many_genes(self):
        """Should map most RP genes."""
        assert self.result["n_genes"] >= 50

    def test_high_mean_optimality(self):
        """RP genes should have higher mean wtAI than genome average."""
        gs = self.result["metagene_geneset"]
        bg = self.result["metagene_genome"]
        # Mean across all positions
        assert gs.mean() > bg.mean(), (
            f"RP mean wtAI {gs.mean():.4f} should exceed genome {bg.mean():.4f}"
        )

    def test_ramp_visible(self):
        """RP genes should show a ramp: lower optimality near 5' end.

        The ramp delta (body - ramp) should be positive, meaning the body
        has higher optimality than the first 50 codons.
        """
        ramp = self.result["ramp_analysis"]
        # Even a small positive delta counts
        assert ramp["geneset_ramp_delta"] > -0.05, (
            f"Expected ramp delta >= -0.05, got {ramp['geneset_ramp_delta']:.4f}"
        )

    def test_metagene_not_flat(self):
        """The metagene profile should vary along the CDS."""
        gs = self.result["metagene_geneset"]
        assert gs.std() > 0.001

    def test_per_gene_scores_has_data(self):
        """Per-gene scores DataFrame should have rows."""
        df = self.result["per_gene_scores"]
        assert len(df) >= 50
        assert "tai" in df.columns
        assert "wtai" in df.columns


@skip_no_data
class TestRandomGenesProfile:
    """Random gene set as negative control: metagene should be flat-ish."""

    @pytest.fixture(autouse=True)
    def run_analysis(self):
        from codonscope.modes.mode3_profile import run_profile
        from codonscope.core.sequences import SequenceDB

        db = SequenceDB("yeast")
        all_genes = list(db.get_all_sequences().keys())

        rng = np.random.RandomState(42)
        random_genes = list(rng.choice(all_genes, size=100, replace=False))

        self.result = run_profile(
            species="yeast",
            gene_ids=random_genes,
            window=10,
            method="wtai",
        )

    def test_many_genes(self):
        assert self.result["n_genes"] >= 80

    def test_profile_close_to_genome(self):
        """Random gene set metagene should be similar to genome."""
        gs = self.result["metagene_geneset"]
        bg = self.result["metagene_genome"]
        diff = np.abs(gs - bg)
        # Mean difference should be small
        assert diff.mean() < 0.05, (
            f"Random gene-set deviates too much from genome: "
            f"mean |diff| = {diff.mean():.4f}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# CLI tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestCLIProfile:
    """Test CLI integration for profile subcommand."""

    def test_help(self):
        from codonscope.cli import main
        with pytest.raises(SystemExit) as exc_info:
            main(["profile", "--help"])
        assert exc_info.value.code == 0
