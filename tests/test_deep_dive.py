"""Tests for deep-dive driver gene analysis."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

DATA_DIR = Path.home() / ".codonscope" / "data"
YEAST_DIR = DATA_DIR / "species" / "yeast"

pytestmark = pytest.mark.skipif(
    not YEAST_DIR.exists(),
    reason="Yeast data not downloaded",
)

YEAST_RP_GENES = [
    "RPL5", "RPL10", "RPL11A", "RPL3", "RPL4A", "RPL6A", "RPL7A",
    "RPL8A", "RPL13A", "RPL15A", "RPL19A", "RPL22A", "RPL25",
    "RPL30", "RPL31A", "RPS3", "RPS5", "RPS8A", "RPS15", "RPS20",
]


@pytest.fixture(scope="module")
def tier1_data_dir():
    """Generate Tier 1 report and return the data directory."""
    from codonscope.report import generate_report

    with tempfile.TemporaryDirectory() as tmpdir:
        output = Path(tmpdir) / "rp_report.html"
        generate_report(
            species="yeast",
            gene_ids=YEAST_RP_GENES,
            output=output,
            n_bootstrap=200,
            seed=42,
        )
        data_dir = Path(tmpdir) / "rp_report_data"
        yield data_dir


class TestLoadTier1Data:
    def test_load_basic(self, tier1_data_dir):
        from codonscope.modes.deep_dive import load_tier1_data
        data = load_tier1_data(tier1_data_dir)
        assert "enrichment" in data
        assert "per_gene_freq" in data
        assert "config" in data

    def test_enrichment_has_columns(self, tier1_data_dir):
        from codonscope.modes.deep_dive import load_tier1_data
        data = load_tier1_data(tier1_data_dir)
        df = data["enrichment"]
        assert "kmer" in df.columns
        assert "z_score" in df.columns
        assert "adjusted_p" in df.columns

    def test_per_gene_freq_shape(self, tier1_data_dir):
        from codonscope.modes.deep_dive import load_tier1_data
        data = load_tier1_data(tier1_data_dir)
        freq = data["per_gene_freq"]
        assert freq.shape[1] == 61  # 61 sense codons
        assert freq.shape[0] > 10  # at least some genes

    def test_missing_dir_raises(self):
        from codonscope.modes.deep_dive import load_tier1_data
        with pytest.raises(FileNotFoundError):
            load_tier1_data("/nonexistent/path")


class TestRunDriverAnalysis:
    def test_basic(self, tier1_data_dir):
        from codonscope.modes.deep_dive import run_driver_analysis
        result = run_driver_analysis(tier1_data_dir)
        assert "driver_tables" in result
        assert "summary" in result
        assert "significant_codons" in result
        assert result["n_genes"] > 10

    def test_has_significant_codons(self, tier1_data_dir):
        from codonscope.modes.deep_dive import run_driver_analysis
        result = run_driver_analysis(tier1_data_dir)
        assert len(result["significant_codons"]) > 0

    def test_summary_columns(self, tier1_data_dir):
        from codonscope.modes.deep_dive import run_driver_analysis
        result = run_driver_analysis(tier1_data_dir)
        if len(result["summary"]) > 0:
            expected = {"codon", "z_score", "gini", "n50", "n80", "n_jackknife_influential"}
            assert expected.issubset(set(result["summary"].columns))

    def test_gini_range(self, tier1_data_dir):
        from codonscope.modes.deep_dive import run_driver_analysis
        result = run_driver_analysis(tier1_data_dir)
        if len(result["summary"]) > 0:
            gini_vals = result["summary"]["gini"].values
            assert all(0 <= g <= 1 for g in gini_vals)

    def test_n50_reasonable(self, tier1_data_dir):
        from codonscope.modes.deep_dive import run_driver_analysis
        result = run_driver_analysis(tier1_data_dir)
        if len(result["summary"]) > 0:
            n50_vals = result["summary"]["n50"].values
            assert all(1 <= n <= result["n_genes"] for n in n50_vals)

    def test_driver_table_has_contributions(self, tier1_data_dir):
        from codonscope.modes.deep_dive import run_driver_analysis
        result = run_driver_analysis(tier1_data_dir)
        for codon, df in result["driver_tables"].items():
            assert "contribution" in df.columns
            assert "gene" in df.columns
            assert "frequency" in df.columns
            break  # just check first one

    def test_jackknife_flags(self, tier1_data_dir):
        from codonscope.modes.deep_dive import run_driver_analysis
        result = run_driver_analysis(tier1_data_dir)
        for codon, df in result["driver_tables"].items():
            assert "jackknife_influential" in df.columns
            # Should be boolean values
            assert df["jackknife_influential"].dtype == bool
            break

    def test_output_dir(self, tier1_data_dir):
        from codonscope.modes.deep_dive import run_driver_analysis
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_driver_analysis(tier1_data_dir, output_dir=tmpdir)
            out = Path(tmpdir)
            assert (out / "driver_summary.tsv").exists()
            # Should have at least one driver TSV
            driver_files = list(out.glob("driver_*.tsv"))
            assert len(driver_files) >= 2  # summary + at least one codon

    def test_strict_threshold(self, tier1_data_dir):
        from codonscope.modes.deep_dive import run_driver_analysis
        result = run_driver_analysis(tier1_data_dir, adj_p_threshold=1e-10)
        # Very strict threshold may return fewer codons
        assert len(result["significant_codons"]) <= 61


class TestGiniCoefficient:
    def test_uniform(self):
        from codonscope.modes.deep_dive import _gini_coefficient
        values = np.ones(100)
        assert _gini_coefficient(values) < 0.01

    def test_concentrated(self):
        from codonscope.modes.deep_dive import _gini_coefficient
        values = np.zeros(100)
        values[0] = 1.0
        assert _gini_coefficient(values) > 0.9

    def test_empty(self):
        from codonscope.modes.deep_dive import _gini_coefficient
        assert _gini_coefficient(np.array([])) == 0.0


class TestDeepDiveReport:
    def test_generates_html(self, tier1_data_dir):
        from codonscope.modes.deep_dive import generate_deep_dive_report
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "dd.html"
            result = generate_deep_dive_report(tier1_data_dir, output=output)
            assert result.exists()
            content = result.read_text()
            assert "<!DOCTYPE html>" in content
            assert "Driver" in content

    def test_has_cumulative_plots(self, tier1_data_dir):
        from codonscope.modes.deep_dive import generate_deep_dive_report
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "dd.html"
            result = generate_deep_dive_report(tier1_data_dir, output=output)
            content = result.read_text()
            assert "data:image/png;base64" in content

    def test_has_collapsible_sections(self, tier1_data_dir):
        from codonscope.modes.deep_dive import generate_deep_dive_report
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "dd.html"
            result = generate_deep_dive_report(tier1_data_dir, output=output)
            content = result.read_text()
            assert "collapsible-section" in content


class TestDeepDiveCLI:
    def test_cli_deep_dive(self, tier1_data_dir):
        from codonscope.cli import main
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "dd.html"
            ret = main([
                "deep-dive",
                "--results-dir", str(tier1_data_dir),
                "--output", str(output),
            ])
            assert ret == 0
            assert output.exists()


class TestPositionalEnrichment:
    def test_basic(self, tier1_data_dir):
        from codonscope.modes.deep_dive import run_positional_enrichment
        result = run_positional_enrichment(tier1_data_dir)
        assert "z_matrix" in result
        assert result["z_matrix"].shape == (100, 61)

    def test_sig_matrix_shape(self, tier1_data_dir):
        from codonscope.modes.deep_dive import run_positional_enrichment
        result = run_positional_enrichment(tier1_data_dir)
        assert "sig_matrix" in result
        assert result["sig_matrix"].shape == result["z_matrix"].shape
        assert result["sig_matrix"].dtype == bool

    def test_z_values_finite(self, tier1_data_dir):
        from codonscope.modes.deep_dive import run_positional_enrichment
        result = run_positional_enrichment(tier1_data_dir)
        z = result["z_matrix"]
        assert np.all(np.isfinite(z))

    def test_has_kmer_names(self, tier1_data_dir):
        from codonscope.modes.deep_dive import run_positional_enrichment
        result = run_positional_enrichment(tier1_data_dir)
        assert "kmer_names" in result
        assert len(result["kmer_names"]) == 61

    def test_top_codons_identified(self, tier1_data_dir):
        from codonscope.modes.deep_dive import run_positional_enrichment
        result = run_positional_enrichment(tier1_data_dir)
        assert "top_enriched" in result
        assert "top_depleted" in result
        assert len(result["top_enriched"]) == 5
        assert len(result["top_depleted"]) == 5

    def test_output_dir(self, tier1_data_dir):
        from codonscope.modes.deep_dive import run_positional_enrichment
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_positional_enrichment(tier1_data_dir, output_dir=tmpdir)
            out = Path(tmpdir)
            assert (out / "positional_z_matrix.tsv").exists()

    def test_custom_bins(self, tier1_data_dir):
        from codonscope.modes.deep_dive import run_positional_enrichment
        result = run_positional_enrichment(tier1_data_dir, n_bins=50)
        assert result["z_matrix"].shape == (50, 61)
        assert result["n_bins"] == 50

    def test_has_species(self, tier1_data_dir):
        from codonscope.modes.deep_dive import run_positional_enrichment
        result = run_positional_enrichment(tier1_data_dir)
        assert "species" in result
        assert result["species"] == "yeast"


class TestClusterScan:
    def test_basic(self, tier1_data_dir):
        from codonscope.modes.deep_dive import run_cluster_scan
        result = run_cluster_scan(tier1_data_dir, n_permutations=100, seed=42)
        assert "per_gene" in result
        assert "summary" in result

    def test_per_gene_columns(self, tier1_data_dir):
        from codonscope.modes.deep_dive import run_cluster_scan
        result = run_cluster_scan(tier1_data_dir, n_permutations=100, seed=42)
        df = result["per_gene"]
        expected = {"gene", "gene_name", "n_codons", "n_target", "n_runs", "max_run_length", "n_clusters", "perm_p"}
        assert expected.issubset(set(df.columns))

    def test_custom_codons(self, tier1_data_dir):
        from codonscope.modes.deep_dive import run_cluster_scan
        result = run_cluster_scan(tier1_data_dir, codons=["AGA"], n_permutations=100, seed=42)
        assert result["target_codons"] == ["AGA"]

    def test_summary_keys(self, tier1_data_dir):
        from codonscope.modes.deep_dive import run_cluster_scan
        result = run_cluster_scan(tier1_data_dir, n_permutations=100, seed=42)
        summary = result["summary"]
        expected = {"n_genes", "n_significant", "frac_significant", "mean_runs", "mean_max_run", "mean_clusters"}
        assert expected.issubset(set(summary.keys()))

    def test_perm_p_range(self, tier1_data_dir):
        from codonscope.modes.deep_dive import run_cluster_scan
        result = run_cluster_scan(tier1_data_dir, n_permutations=100, seed=42)
        df = result["per_gene"]
        assert all(0 <= p <= 1 for p in df["perm_p"])

    def test_output_dir(self, tier1_data_dir):
        from codonscope.modes.deep_dive import run_cluster_scan
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_cluster_scan(tier1_data_dir, n_permutations=100, seed=42, output_dir=tmpdir)
            out = Path(tmpdir)
            assert (out / "cluster_scan.tsv").exists()

    def test_top_n_auto_select(self, tier1_data_dir):
        from codonscope.modes.deep_dive import run_cluster_scan
        result = run_cluster_scan(tier1_data_dir, top_n=3, n_permutations=100, seed=42)
        assert len(result["target_codons"]) <= 3  # May be fewer if fewer significant codons


class TestSynonymousShuffle:
    def test_preserves_amino_acids(self):
        from codonscope.modes.deep_dive import _synonymous_shuffle
        from codonscope.core.codons import CODON_TABLE
        codons = ["GCT", "GCC", "GCA", "GCG", "AAA", "AAG", "AAA"]
        rng = np.random.default_rng(42)
        shuffled = _synonymous_shuffle(codons, rng)
        orig_aas = [CODON_TABLE[c] for c in codons]
        shuf_aas = [CODON_TABLE[c] for c in shuffled]
        assert orig_aas == shuf_aas

    def test_shuffles_within_family(self):
        from codonscope.modes.deep_dive import _synonymous_shuffle
        codons = ["GCT", "GCT", "GCT", "GCC", "GCC", "GCA", "GCG"]
        rng = np.random.default_rng(42)
        shuffled = _synonymous_shuffle(codons, rng)
        # Should still have same codon counts
        from collections import Counter
        assert Counter(shuffled) == Counter(codons)

    def test_single_codon_aa_unchanged(self):
        from codonscope.modes.deep_dive import _synonymous_shuffle
        codons = ["ATG", "TGG", "ATG", "TGG"]
        rng = np.random.default_rng(42)
        shuffled = _synonymous_shuffle(codons, rng)
        # Met and Trp have only one codon each, so positions must stay the same
        assert shuffled == codons

    def test_mixed_families(self):
        from codonscope.modes.deep_dive import _synonymous_shuffle
        from codonscope.core.codons import CODON_TABLE
        # Mix of multi-codon and single-codon AAs
        codons = ["ATG", "GCT", "GCC", "TGG", "AAA", "AAG"]
        rng = np.random.default_rng(42)
        shuffled = _synonymous_shuffle(codons, rng)
        orig_aas = [CODON_TABLE[c] for c in codons]
        shuf_aas = [CODON_TABLE[c] for c in shuffled]
        assert orig_aas == shuf_aas

    def test_deterministic_with_seed(self):
        from codonscope.modes.deep_dive import _synonymous_shuffle
        codons = ["GCT", "GCC", "GCA", "GCG", "AAA", "AAG"] * 5
        rng1 = np.random.default_rng(123)
        rng2 = np.random.default_rng(123)
        shuffled1 = _synonymous_shuffle(codons, rng1)
        shuffled2 = _synonymous_shuffle(codons, rng2)
        assert shuffled1 == shuffled2


class TestComputePositionalFreqs:
    def test_basic(self):
        from codonscope.modes.deep_dive import _compute_positional_freqs
        # Simple sequence: 10 codons, split into 10 bins
        seq = "GCTAGCGCAGCTTTTAAACCCTTTGGGAAACCCAAAGGG"
        result = _compute_positional_freqs(seq, n_bins=10)
        assert result.shape == (10, 61)
        # Each bin should have frequencies summing to ~1 (or 0 if empty)
        for i in range(10):
            bin_sum = result[i].sum()
            assert bin_sum == pytest.approx(1.0, abs=1e-6) or bin_sum == 0.0

    def test_empty_sequence(self):
        from codonscope.modes.deep_dive import _compute_positional_freqs
        seq = ""
        result = _compute_positional_freqs(seq, n_bins=10)
        assert result.shape == (10, 61)
        assert np.all(result == 0.0)

    def test_single_codon(self):
        from codonscope.modes.deep_dive import _compute_positional_freqs
        seq = "GCT"
        result = _compute_positional_freqs(seq, n_bins=5)
        assert result.shape == (5, 61)
        # Only one bin should be non-zero
        assert np.sum(result.sum(axis=1) > 0) == 1


class TestDeepDiveCLIExtended:
    def test_cli_positional(self, tier1_data_dir):
        from codonscope.cli import main
        with tempfile.TemporaryDirectory() as tmpdir:
            ret = main([
                "deep-dive",
                "--results-dir", str(tier1_data_dir),
                "--output", str(Path(tmpdir) / "dd.html"),
                "--positional",
            ])
            assert ret == 0

    def test_cli_cluster(self, tier1_data_dir):
        from codonscope.cli import main
        with tempfile.TemporaryDirectory() as tmpdir:
            ret = main([
                "deep-dive",
                "--results-dir", str(tier1_data_dir),
                "--output", str(Path(tmpdir) / "dd.html"),
                "--cluster",
                "--permutations", "100",
                "--seed", "42",
            ])
            assert ret == 0

    def test_cli_both_features(self, tier1_data_dir):
        from codonscope.cli import main
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            ret = main([
                "deep-dive",
                "--results-dir", str(tier1_data_dir),
                "--output", str(output_dir / "dd.html"),
                "--output-dir", str(output_dir),
                "--positional",
                "--cluster",
                "--permutations", "100",
                "--seed", "42",
            ])
            assert ret == 0
            assert (output_dir / "positional_z_matrix.tsv").exists()
            assert (output_dir / "cluster_scan.tsv").exists()

    def test_cli_custom_cluster_codons(self, tier1_data_dir):
        from codonscope.cli import main
        with tempfile.TemporaryDirectory() as tmpdir:
            ret = main([
                "deep-dive",
                "--results-dir", str(tier1_data_dir),
                "--output", str(Path(tmpdir) / "dd.html"),
                "--cluster",
                "--cluster-codons", "AGA", "AAG",
                "--permutations", "100",
                "--seed", "42",
            ])
            assert ret == 0
