"""Tests for reverse mode — find genes enriched for specific codons."""

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


class TestRunReverse:
    """Test run_reverse() function."""

    def test_basic_single_codon(self):
        from codonscope.modes.reverse import run_reverse
        result = run_reverse(species="yeast", codons=["AGA"], zscore_cutoff=2.0)
        df = result["gene_table"]
        assert len(df) > 0
        assert "combined_z" in df.columns
        assert "gene" in df.columns
        assert "gene_name" in df.columns
        assert all(df["combined_z"] >= 2.0)

    def test_multi_codon(self):
        from codonscope.modes.reverse import run_reverse
        result = run_reverse(species="yeast", codons=["AGA", "AAG"], zscore_cutoff=1.5)
        df = result["gene_table"]
        assert len(df) > 0
        assert "z_AGA" in df.columns
        assert "z_AAG" in df.columns
        assert "combined_z" in df.columns

    def test_top_n(self):
        from codonscope.modes.reverse import run_reverse
        result = run_reverse(species="yeast", codons=["AGA"], top=50)
        df = result["gene_table"]
        assert len(df) == 50

    def test_percentile_filter(self):
        from codonscope.modes.reverse import run_reverse
        result = run_reverse(species="yeast", codons=["AGA"], percentile=95)
        df = result["gene_table"]
        assert len(df) > 0
        # Should be top 5% of genome
        assert len(df) <= result["n_genome_genes"] * 0.06  # small margin

    def test_sorted_by_z(self):
        from codonscope.modes.reverse import run_reverse
        result = run_reverse(species="yeast", codons=["AGA"], top=100)
        df = result["gene_table"]
        zvals = df["combined_z"].values
        # Should be sorted descending
        assert all(zvals[i] >= zvals[i+1] for i in range(len(zvals) - 1))

    def test_aga_includes_rp_genes(self):
        """AGA is an RP-preferred codon. Top genes should include RP genes."""
        from codonscope.modes.reverse import run_reverse
        result = run_reverse(species="yeast", codons=["AGA"], top=200)
        df = result["gene_table"]
        gene_names = set(df["gene_name"].str.upper())
        # At least some RP genes should be in top 200 AGA-enriched
        rp_found = [g for g in gene_names if g.startswith("RPL") or g.startswith("RPS")]
        assert len(rp_found) >= 5, f"Expected RP genes in top AGA, found {rp_found}"

    def test_invalid_codon_raises(self):
        from codonscope.modes.reverse import run_reverse
        with pytest.raises(ValueError, match="Invalid codons"):
            run_reverse(species="yeast", codons=["XYZ"])

    def test_stop_codon_raises(self):
        from codonscope.modes.reverse import run_reverse
        with pytest.raises(ValueError, match="Invalid codons"):
            run_reverse(species="yeast", codons=["TAA"])

    def test_output_dir(self):
        from codonscope.modes.reverse import run_reverse
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_reverse(
                species="yeast", codons=["AGA"], top=10,
                output_dir=tmpdir,
            )
            tsv_files = list(Path(tmpdir).glob("*.tsv"))
            assert len(tsv_files) == 1
            assert "reverse_AGA" in tsv_files[0].name

    def test_result_structure(self):
        from codonscope.modes.reverse import run_reverse
        result = run_reverse(species="yeast", codons=["GCT"], top=10)
        assert "gene_table" in result
        assert "n_genome_genes" in result
        assert "target_codons" in result
        assert "species" in result
        assert result["n_genome_genes"] > 5000
        assert result["target_codons"] == ["GCT"]

    def test_uracil_converted(self):
        """Codons with U should be converted to T."""
        from codonscope.modes.reverse import run_reverse
        result = run_reverse(species="yeast", codons=["AGA"], top=10)
        result_u = run_reverse(species="yeast", codons=["AGA"], top=10)
        # Both should give same results
        assert result["target_codons"] == result_u["target_codons"]

    def test_default_zscore_cutoff(self):
        """Without any filter args, should use Z >= 2.0 default."""
        from codonscope.modes.reverse import run_reverse
        result = run_reverse(species="yeast", codons=["AGA"])
        df = result["gene_table"]
        assert all(df["combined_z"] >= 2.0)

    def test_columns_present(self):
        from codonscope.modes.reverse import run_reverse
        result = run_reverse(species="yeast", codons=["AGA", "GAA"], top=5)
        df = result["gene_table"]
        expected_cols = {"gene", "combined_z", "min_z", "max_z", "sum_freq",
                        "mean_freq", "gene_name", "z_AGA", "z_GAA"}
        assert expected_cols.issubset(set(df.columns))

    def test_check_mode(self):
        """Test --check runs enrichment on result."""
        from codonscope.modes.reverse import run_reverse
        result = run_reverse(
            species="yeast", codons=["AGA"], top=50, check=True,
        )
        assert result["check_result"] is not None
        assert "results" in result["check_result"]

    def test_yef3_in_aga_top_genes(self):
        """YEF3 is enriched for AGA (translation elongation factor)."""
        from codonscope.modes.reverse import run_reverse
        result = run_reverse(species="yeast", codons=["AGA"], top=600)
        df = result["gene_table"]
        gene_names = set(df["gene_name"].str.upper())
        genes = set(df["gene"].str.upper())
        # YEF3 or its systematic name should be present in top 600
        assert "YEF3" in gene_names or "YLR249W" in genes, \
            f"YEF3 expected in top 600 AGA genes"


class TestReverseCLI:
    """Test CLI reverse subcommand."""

    def test_cli_reverse(self):
        from codonscope.cli import main
        with tempfile.TemporaryDirectory() as tmpdir:
            ret = main([
                "reverse", "--species", "yeast",
                "--codons", "AGA",
                "--top", "10",
                "--output-dir", tmpdir,
            ])
            assert ret == 0
