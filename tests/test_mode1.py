"""Tests for Chunk 3: Mode 1 (Composition) + CLI."""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from codonscope.cli import _parse_gene_list, main
from codonscope.core.sequences import SequenceDB
from codonscope.modes.mode1_composition import run_composition

DATA_DIR = os.environ.get(
    "CODONSCOPE_TEST_DATA_DIR",
    str(Path.home() / ".codonscope" / "data" / "species"),
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def yeast_db():
    return SequenceDB("yeast", data_dir=DATA_DIR)


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def gene_list_file(tmp_dir):
    """Write a small gene list file for testing."""
    genes = [
        "RPL5", "RPL10", "RPL25", "RPL30", "RPL32",
        "RPS3", "RPS5", "RPS12", "RPS15", "RPS20",
        "RPL3", "RPL6A", "RPL7A", "RPL8A", "RPL9A",
    ]
    path = tmp_dir / "genes.txt"
    path.write_text("\n".join(genes))
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# Gene list parsing
# ═══════════════════════════════════════════════════════════════════════════════

class TestParseGeneList:
    def test_one_per_line(self, tmp_dir):
        path = tmp_dir / "genes.txt"
        path.write_text("GCN4\nYEF3\nRPL5\n")
        ids = _parse_gene_list(str(path))
        assert ids == ["GCN4", "YEF3", "RPL5"]

    def test_comma_separated(self, tmp_dir):
        path = tmp_dir / "genes.txt"
        path.write_text("GCN4,YEF3,RPL5")
        ids = _parse_gene_list(str(path))
        assert ids == ["GCN4", "YEF3", "RPL5"]

    def test_mixed_format(self, tmp_dir):
        path = tmp_dir / "genes.txt"
        path.write_text("GCN4, YEF3\nRPL5\tRPS3\n")
        ids = _parse_gene_list(str(path))
        assert set(ids) == {"GCN4", "YEF3", "RPL5", "RPS3"}

    def test_comments_skipped(self, tmp_dir):
        path = tmp_dir / "genes.txt"
        path.write_text("# header comment\nGCN4\n# another comment\nYEF3\n")
        ids = _parse_gene_list(str(path))
        assert ids == ["GCN4", "YEF3"]

    def test_empty_lines_skipped(self, tmp_dir):
        path = tmp_dir / "genes.txt"
        path.write_text("GCN4\n\n\nYEF3\n")
        ids = _parse_gene_list(str(path))
        assert ids == ["GCN4", "YEF3"]

    def test_nonexistent_file(self):
        ids = _parse_gene_list("/nonexistent/path.txt")
        assert ids == []


# ═══════════════════════════════════════════════════════════════════════════════
# Mode 1 analysis
# ═══════════════════════════════════════════════════════════════════════════════

class TestRunComposition:
    def test_basic_mono(self, tmp_dir):
        """Basic monocodon composition analysis should return valid results."""
        genes = [
            "RPL5", "RPL10", "RPL25", "RPL30", "RPL32",
            "RPS3", "RPS5", "RPS12", "RPS15", "RPS20",
        ]
        result = run_composition(
            species="yeast",
            gene_ids=genes,
            k=1,
            n_bootstrap=500,
            output_dir=str(tmp_dir),
            seed=42,
            data_dir=DATA_DIR,
        )
        df = result["results"]
        assert len(df) == 61
        assert result["n_genes"] == 10
        assert "z_score" in df.columns
        assert "adjusted_p" in df.columns

        # Output files should exist
        assert (tmp_dir / "composition_monocodons.tsv").exists()
        assert (tmp_dir / "volcano_mono.png").exists()
        assert (tmp_dir / "top_kmers_mono.png").exists()

    def test_dicodon_analysis(self, tmp_dir):
        """Dicodon analysis should return 3721 rows."""
        genes = [
            "RPL5", "RPL10", "RPL25", "RPL30", "RPL32",
            "RPS3", "RPS5", "RPS12", "RPS15", "RPS20",
            "RPL3", "RPL6A", "RPL7A", "RPL8A", "RPL9A",
            "RPL11A", "RPL12A", "RPL13A", "RPL14A", "RPL15A",
            "RPL16A", "RPL17A", "RPL18A", "RPL19A", "RPL20A",
            "RPL21A", "RPL22A", "RPL23A", "RPL24A", "RPL28",
        ]
        result = run_composition(
            species="yeast",
            gene_ids=genes,
            k=2,
            n_bootstrap=500,
            output_dir=str(tmp_dir),
            seed=42,
            data_dir=DATA_DIR,
        )
        df = result["results"]
        assert len(df) == 61 * 61

    def test_matched_background(self):
        """Matched background analysis should complete without error."""
        genes = [
            "RPL5", "RPL10", "RPL25", "RPL30", "RPL32",
            "RPS3", "RPS5", "RPS12", "RPS15", "RPS20",
        ]
        result = run_composition(
            species="yeast",
            gene_ids=genes,
            k=1,
            background="matched",
            n_bootstrap=500,
            seed=42,
            data_dir=DATA_DIR,
        )
        df = result["results"]
        assert len(df) == 61
        # Should still have valid frequencies
        assert abs(df["observed_freq"].sum() - 1.0) < 0.01

    def test_trim_ramp(self):
        """Trimming ramp should change results."""
        genes = ["RPL5", "RPL10", "RPL25", "RPL30", "RPL32",
                 "RPS3", "RPS5", "RPS12", "RPS15", "RPS20"]
        r1 = run_composition(
            species="yeast", gene_ids=genes, k=1,
            trim_ramp=0, n_bootstrap=500, seed=42, data_dir=DATA_DIR,
        )
        r2 = run_composition(
            species="yeast", gene_ids=genes, k=1,
            trim_ramp=50, n_bootstrap=500, seed=42, data_dir=DATA_DIR,
        )
        # Frequencies should differ
        assert not np.allclose(
            r1["results"]["observed_freq"].values,
            r2["results"]["observed_freq"].values,
        )

    def test_diagnostics_present(self):
        """Diagnostics should include KS test results."""
        genes = ["RPL5", "RPL10", "RPL25", "RPL30", "RPL32",
                 "RPS3", "RPS5", "RPS12", "RPS15", "RPS20"]
        result = run_composition(
            species="yeast", gene_ids=genes, k=1,
            n_bootstrap=500, seed=42, data_dir=DATA_DIR,
        )
        diag = result["diagnostics"]
        assert "length_p" in diag
        assert "gc_p" in diag

    def test_unmapped_genes_reported(self):
        """Unmapped gene IDs should be reported in summary."""
        genes = ["RPL5", "FAKEGENE999", "RPS3"]
        result = run_composition(
            species="yeast", gene_ids=genes, k=1,
            min_genes=1, n_bootstrap=100, seed=42, data_dir=DATA_DIR,
        )
        assert result["id_summary"]["n_unmapped"] == 1
        assert "FAKEGENE999" in result["id_summary"]["unmapped"]

    def test_no_output_dir(self):
        """Should work without writing output files."""
        genes = ["RPL5", "RPL10", "RPL25", "RPL30", "RPL32",
                 "RPS3", "RPS5", "RPS12", "RPS15", "RPS20"]
        result = run_composition(
            species="yeast", gene_ids=genes, k=1,
            n_bootstrap=100, seed=42, data_dir=DATA_DIR,
            output_dir=None,
        )
        assert len(result["results"]) == 61


# ═══════════════════════════════════════════════════════════════════════════════
# Positive controls via Mode 1
# ═══════════════════════════════════════════════════════════════════════════════

class TestRibosomalProteinComposition:
    """Ribosomal proteins should show strong codon bias in Mode 1."""

    RP_GENES = [
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

    def test_mono_strong_bias(self):
        result = run_composition(
            species="yeast", gene_ids=self.RP_GENES, k=1,
            n_bootstrap=2000, seed=42, data_dir=DATA_DIR,
        )
        df = result["results"]
        n_sig = (df["adjusted_p"] < 0.05).sum()
        assert n_sig >= 10, (
            f"Ribosomal proteins should have >=10 significant monocodons, got {n_sig}"
        )

    def test_di_strong_bias(self):
        result = run_composition(
            species="yeast", gene_ids=self.RP_GENES, k=2,
            n_bootstrap=2000, seed=42, data_dir=DATA_DIR,
        )
        df = result["results"]
        n_sig = (df["adjusted_p"] < 0.05).sum()
        assert n_sig >= 20, (
            f"Ribosomal proteins should have >=20 significant dicodons, got {n_sig}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# CLI smoke tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestCLI:
    def test_help(self):
        """--help should exit with code 0."""
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0

    def test_version(self):
        """--version should exit with code 0."""
        with pytest.raises(SystemExit) as exc_info:
            main(["--version"])
        assert exc_info.value.code == 0

    def test_no_command(self):
        """No command should return code 1."""
        code = main([])
        assert code == 1

    def test_composition_cli(self, gene_list_file, tmp_dir):
        """Full CLI composition run should succeed."""
        out_dir = tmp_dir / "output"
        code = main([
            "composition",
            "--species", "yeast",
            "--genes", str(gene_list_file),
            "--kmer", "1",
            "--n-bootstrap", "200",
            "--seed", "42",
            "--output-dir", str(out_dir),
            "--data-dir", DATA_DIR,
        ])
        assert code == 0
        assert (out_dir / "composition_monocodons.tsv").exists()
