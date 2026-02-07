"""Tests for Chunk 2: Core statistics engine."""

import os
from pathlib import Path

import numpy as np
import pytest

from codonscope.core.codons import all_possible_kmers, kmer_frequencies
from codonscope.core.sequences import SequenceDB
from codonscope.core.statistics import (
    benjamini_hochberg,
    bootstrap_pvalues,
    bootstrap_zscores,
    cohens_d,
    compare_to_background,
    compute_geneset_frequencies,
    diagnostic_ks_tests,
    power_check,
)

DATA_DIR = os.environ.get(
    "CODONSCOPE_TEST_DATA_DIR",
    str(Path.home() / ".codonscope" / "data" / "species"),
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def yeast_db():
    return SequenceDB("yeast", data_dir=DATA_DIR)


@pytest.fixture(scope="session")
def yeast_dir():
    return Path(DATA_DIR) / "yeast"


# ═══════════════════════════════════════════════════════════════════════════════
# Unit tests — pure functions, no downloaded data needed
# ═══════════════════════════════════════════════════════════════════════════════

class TestBenjaminiHochberg:
    def test_single_pvalue(self):
        result = benjamini_hochberg(np.array([0.05]))
        assert np.isclose(result[0], 0.05)

    def test_all_significant(self):
        p = np.array([0.001, 0.002, 0.003])
        adj = benjamini_hochberg(p)
        # All should remain significant after BH
        assert all(adj < 0.01)
        # Monotonicity: adjusted p-values for sorted inputs should be non-decreasing
        sorted_idx = np.argsort(p)
        assert np.all(np.diff(adj[sorted_idx]) >= -1e-15)

    def test_caps_at_one(self):
        p = np.array([0.5, 0.8, 0.95])
        adj = benjamini_hochberg(p)
        assert np.all(adj <= 1.0)

    def test_preserves_ordering(self):
        """Smaller raw p-values should have smaller or equal adjusted p-values."""
        p = np.array([0.001, 0.05, 0.5, 0.9])
        adj = benjamini_hochberg(p)
        for i in range(len(p)):
            for j in range(i + 1, len(p)):
                if p[i] < p[j]:
                    assert adj[i] <= adj[j] + 1e-15

    def test_empty_input(self):
        result = benjamini_hochberg(np.array([]))
        assert len(result) == 0

    def test_known_values(self):
        """Compare to hand-computed BH correction."""
        # 4 tests, p = [0.01, 0.04, 0.03, 0.20]
        # Sorted: [0.01, 0.03, 0.04, 0.20] at ranks [1,2,3,4]
        # Raw adj: [0.04, 0.06, 0.0533, 0.20]
        # After monotonicity: [0.04, 0.0533, 0.0533, 0.20]
        p = np.array([0.01, 0.04, 0.03, 0.20])
        adj = benjamini_hochberg(p)
        assert adj[0] < 0.05  # 0.01 * 4/1 = 0.04
        assert adj[3] == pytest.approx(0.20, abs=1e-10)


class TestBootstrapPvalues:
    def test_zero_zscore(self):
        p = bootstrap_pvalues(np.array([0.0]))
        assert np.isclose(p[0], 1.0)

    def test_large_zscore(self):
        p = bootstrap_pvalues(np.array([10.0]))
        assert p[0] < 1e-20

    def test_symmetric(self):
        """Positive and negative Z-scores of same magnitude give same p."""
        p = bootstrap_pvalues(np.array([3.0, -3.0]))
        assert np.isclose(p[0], p[1])

    def test_standard_values(self):
        """Z=1.96 should give p≈0.05."""
        p = bootstrap_pvalues(np.array([1.96]))
        assert abs(p[0] - 0.05) < 0.001


class TestCohensD:
    def test_no_difference(self):
        mean = np.array([0.5, 0.3])
        d = cohens_d(mean, mean, np.array([0.1, 0.1]))
        assert np.allclose(d, 0.0)

    def test_one_sd_difference(self):
        geneset = np.array([0.6])
        bg_mean = np.array([0.5])
        bg_std = np.array([0.1])
        d = cohens_d(geneset, bg_mean, bg_std)
        assert np.isclose(d[0], 1.0)

    def test_zero_std(self):
        """Zero std should give 0 Cohen's d, not inf."""
        d = cohens_d(np.array([0.5]), np.array([0.3]), np.array([0.0]))
        assert d[0] == 0.0


class TestPowerCheck:
    def test_adequate_power_mono(self):
        warns = power_check(50, k=1)
        assert len(warns) == 0

    def test_low_power_dicodon(self):
        warns = power_check(20, k=2)
        assert any("dicodon" in w.lower() or "underpowered" in w.lower() for w in warns)

    def test_low_power_tricodon(self):
        warns = power_check(50, k=3)
        assert any("tricodon" in w.lower() or "underpowered" in w.lower() for w in warns)

    def test_very_small_list(self):
        warns = power_check(5, k=1)
        assert len(warns) > 0


class TestBootstrapZscores:
    def test_known_signal(self):
        """A gene set drawn from one tail should produce positive Z-scores."""
        rng = np.random.default_rng(42)
        n_genome = 1000
        n_kmers = 10
        # Background: standard normal frequencies
        bg = rng.normal(0.5, 0.05, size=(n_genome, n_kmers)).astype(np.float32)
        bg = np.clip(bg, 0, 1)
        # Gene set: shifted up
        geneset_mean = np.full(n_kmers, 0.6, dtype=np.float64)

        z, _, _ = bootstrap_zscores(geneset_mean, bg, n_genes=50, seed=42)
        # All Z-scores should be positive (gene set is above background)
        assert np.all(z > 0)

    def test_null_distribution(self):
        """Gene set drawn from background should give Z-scores near 0."""
        rng = np.random.default_rng(42)
        n_genome = 1000
        n_kmers = 10
        bg = rng.normal(0.5, 0.05, size=(n_genome, n_kmers)).astype(np.float32)

        # Gene set mean = background mean (should give Z ≈ 0)
        geneset_mean = bg.mean(axis=0).astype(np.float64)

        z, _, _ = bootstrap_zscores(geneset_mean, bg, n_genes=50, seed=42)
        # Most Z-scores should be small
        assert np.median(np.abs(z)) < 2.0

    def test_reproducible_with_seed(self):
        rng = np.random.default_rng(42)
        bg = rng.normal(0.5, 0.05, size=(500, 5)).astype(np.float32)
        mean = np.full(5, 0.55)

        z1, _, _ = bootstrap_zscores(mean, bg, n_genes=30, seed=123)
        z2, _, _ = bootstrap_zscores(mean, bg, n_genes=30, seed=123)
        assert np.allclose(z1, z2)


class TestComputeGenesetFrequencies:
    def test_simple_sequences(self):
        seqs = {
            "G1": "ATGAAAGAA",  # ATG, AAA, GAA
            "G2": "ATGATGGAA",  # ATG, ATG, GAA
        }
        per_gene, mean_vec, kmer_names = compute_geneset_frequencies(seqs, k=1)
        assert per_gene.shape[0] == 2
        assert per_gene.shape[1] == 61  # sense codons
        # Mean should sum to ~1.0
        assert abs(mean_vec.sum() - 1.0) < 0.01

    def test_trim_ramp(self):
        # 5 codons: skip first 2 → count last 3
        seq = "ATGAAAGAACCCTTT"  # ATG AAA GAA CCC TTT
        seqs = {"G1": seq}
        _, mean_no_trim, _ = compute_geneset_frequencies(seqs, k=1, trim_ramp=0)
        _, mean_trim2, _ = compute_geneset_frequencies(seqs, k=1, trim_ramp=2)
        # With trim=2, only GAA, CCC, TTT should be counted
        assert not np.allclose(mean_no_trim, mean_trim2)


# ═══════════════════════════════════════════════════════════════════════════════
# Integration tests — require downloaded yeast data
# ═══════════════════════════════════════════════════════════════════════════════

class TestDiagnosticKS:
    def test_same_distribution(self):
        """Identical distributions should not trigger warnings."""
        rng = np.random.default_rng(42)
        data = rng.normal(1000, 200, size=500).astype(np.float32)
        result = diagnostic_ks_tests(data[:50], data[:50], data, data)
        assert not result["length_warning"]
        assert not result["gc_warning"]


class TestCompareToBackgroundMono:
    """Integration tests using real yeast mono background."""

    def test_returns_dataframe(self, yeast_db, yeast_dir):
        seqs = yeast_db.get_sequences(["YLR249W", "YEL009C", "YAL001C"])
        bg_path = yeast_dir / "background_mono.npz"
        df = compare_to_background(seqs, bg_path, k=1, n_bootstrap=1000, seed=42)
        assert len(df) == 61
        expected_cols = {
            "kmer", "observed_freq", "expected_freq",
            "z_score", "p_value", "adjusted_p", "cohens_d",
        }
        assert expected_cols.issubset(set(df.columns))

    def test_frequencies_valid(self, yeast_db, yeast_dir):
        seqs = yeast_db.get_sequences(["YLR249W"])
        bg_path = yeast_dir / "background_mono.npz"
        df = compare_to_background(seqs, bg_path, k=1, n_bootstrap=500, seed=42)
        # Observed frequencies should sum to ~1.0
        assert abs(df["observed_freq"].sum() - 1.0) < 0.01
        # Expected frequencies should sum to ~1.0
        assert abs(df["expected_freq"].sum() - 1.0) < 0.01

    def test_sorted_by_abs_zscore(self, yeast_db, yeast_dir):
        seqs = yeast_db.get_sequences(["YLR249W", "YEL009C"])
        bg_path = yeast_dir / "background_mono.npz"
        df = compare_to_background(seqs, bg_path, k=1, n_bootstrap=500, seed=42)
        abs_z = df["z_score"].abs().values
        assert np.all(abs_z[:-1] >= abs_z[1:] - 1e-10)


class TestCompareToBackgroundDi:
    """Integration test with dicodon background."""

    def test_returns_correct_count(self, yeast_db, yeast_dir):
        seqs = yeast_db.get_sequences(["YLR249W", "YEL009C", "YAL001C"])
        bg_path = yeast_dir / "background_di.npz"
        df = compare_to_background(seqs, bg_path, k=2, n_bootstrap=500, seed=42)
        assert len(df) == 61 * 61  # 3721 dicodons


class TestYeastPositiveControls:
    """Validate statistics against known yeast positive controls.

    These are smoke tests — they confirm the statistical machinery
    produces biologically expected results.
    """

    def _get_ribosomal_protein_genes(self, yeast_db):
        """Get a sample of known yeast ribosomal protein genes."""
        rp_genes = [
            "RPL1A", "RPL2A", "RPL3", "RPL4A", "RPL5", "RPL6A", "RPL7A",
            "RPL8A", "RPL9A", "RPL10", "RPL11A", "RPL12A", "RPL13A",
            "RPL14A", "RPL15A", "RPL16A", "RPL17A", "RPL18A", "RPL19A",
            "RPL20A", "RPL21A", "RPL22A", "RPL23A", "RPL24A", "RPL25",
            "RPL26A", "RPL27A", "RPL28", "RPL29", "RPL30", "RPL31A",
            "RPL32", "RPL33A", "RPL34A", "RPL35A", "RPL36A", "RPL37A",
            "RPL38", "RPL39", "RPL40A", "RPL41A", "RPL42A", "RPL43A",
            "RPS0A", "RPS1A", "RPS2", "RPS3", "RPS4A", "RPS5", "RPS6A",
            "RPS7A", "RPS8A", "RPS9A", "RPS10A", "RPS11A", "RPS12",
            "RPS13", "RPS14A", "RPS15", "RPS16A", "RPS17A", "RPS18A",
            "RPS19A", "RPS20", "RPS21A", "RPS22A", "RPS23A", "RPS24A",
            "RPS25A", "RPS26A", "RPS27A", "RPS28A", "RPS29A", "RPS30A",
            "RPS31",
        ]
        result = yeast_db.resolve_ids(rp_genes)
        sys_names = list(result["mapping"].values())
        return yeast_db.get_sequences(sys_names)

    def test_ribosomal_proteins_strong_bias(self, yeast_db, yeast_dir):
        """Ribosomal proteins should show significant codon bias (mono)."""
        seqs = self._get_ribosomal_protein_genes(yeast_db)
        assert len(seqs) >= 50, f"Only resolved {len(seqs)} RP genes"

        bg_path = yeast_dir / "background_mono.npz"
        df = compare_to_background(seqs, bg_path, k=1, n_bootstrap=2000, seed=42)

        # Should have multiple codons with |Z| > 3
        n_significant = (df["z_score"].abs() > 3).sum()
        assert n_significant >= 5, (
            f"Expected >=5 codons with |Z|>3 for ribosomal proteins, got {n_significant}"
        )

    def test_yef3_aga_enrichment(self, yeast_db, yeast_dir):
        """YEF3 (YLR249W) should show AGA enrichment (monocodon)."""
        seqs = yeast_db.get_sequences(["YLR249W"])
        bg_path = yeast_dir / "background_mono.npz"
        df = compare_to_background(seqs, bg_path, k=1, n_bootstrap=2000, seed=42)

        aga_row = df[df["kmer"] == "AGA"]
        assert len(aga_row) == 1
        # YEF3 is known to be enriched for AGA
        assert aga_row.iloc[0]["observed_freq"] > aga_row.iloc[0]["expected_freq"], (
            "YEF3 should have higher AGA frequency than genome average"
        )
