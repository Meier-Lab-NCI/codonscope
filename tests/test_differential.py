"""Tests for differential codon usage analysis."""

import tempfile
from pathlib import Path

import pytest

from codonscope.modes.differential import generate_differential_report, run_differential

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

YEAST_GCN4_GENES = [
    "ARG1", "ARG3", "ARG4", "ARG8", "ARO1", "ARO2", "ARO3", "ARO4",
    "ASN1", "ASN2", "CPA1", "CPA2", "GLN1", "GLT1",
    "HIS1", "HIS3", "HIS4", "HIS5", "HOM2", "HOM3", "HOM6",
    "ILV1", "ILV2", "ILV3", "ILV5", "LEU1", "LEU2", "LEU4",
    "LYS1", "LYS2", "LYS4", "LYS9", "LYS20",
    "MET6", "MET17", "SER1", "SER2", "SHM1", "SHM2",
    "THR1", "THR4", "TRP2", "TRP3", "TRP4", "TRP5", "TYR1",
]


# ═══════════════════════════════════════════════════════════════════════════════
# Basic tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_basic_differential():
    """Basic differential analysis should return valid results."""
    result = run_differential(
        species="yeast",
        gene_ids_a=YEAST_RP_GENES,
        gene_ids_b=YEAST_GCN4_GENES,
        n_bootstrap=200,
        seed=42,
    )

    # Check required keys
    assert "results" in result
    assert "n_genes_a" in result
    assert "n_genes_b" in result
    assert "labels" in result
    assert "species" in result


def test_results_columns():
    """Results DataFrame should have all required columns."""
    result = run_differential(
        species="yeast",
        gene_ids_a=YEAST_RP_GENES,
        gene_ids_b=YEAST_GCN4_GENES,
        n_bootstrap=200,
        seed=42,
    )

    df = result["results"]
    required_cols = [
        "kmer", "amino_acid", "mean_a", "mean_b", "fold_change",
        "u_stat", "p_value", "adjusted_p", "rank_biserial_r",
        "z_genome_a", "z_genome_b",
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing column: {col}"


def test_results_has_61_codons():
    """Results should have exactly 61 rows (one per sense codon)."""
    result = run_differential(
        species="yeast",
        gene_ids_a=YEAST_RP_GENES,
        gene_ids_b=YEAST_GCN4_GENES,
        n_bootstrap=200,
        seed=42,
    )

    df = result["results"]
    assert len(df) == 61


def test_rp_vs_gcn4_has_significant():
    """RP vs Gcn4 should have significant differential codons."""
    result = run_differential(
        species="yeast",
        gene_ids_a=YEAST_RP_GENES,
        gene_ids_b=YEAST_GCN4_GENES,
        n_bootstrap=200,
        seed=42,
    )

    df = result["results"]
    sig = df[df["adjusted_p"] < 0.05]
    assert len(sig) > 0, "Expected some significant differential codons"


def test_fold_change_reasonable():
    """All fold changes should be positive."""
    result = run_differential(
        species="yeast",
        gene_ids_a=YEAST_RP_GENES,
        gene_ids_b=YEAST_GCN4_GENES,
        n_bootstrap=200,
        seed=42,
    )

    df = result["results"]
    assert (df["fold_change"] > 0).all(), "All fold changes should be positive"


def test_rank_biserial_range():
    """Rank-biserial r should be between -1 and 1."""
    result = run_differential(
        species="yeast",
        gene_ids_a=YEAST_RP_GENES,
        gene_ids_b=YEAST_GCN4_GENES,
        n_bootstrap=200,
        seed=42,
    )

    df = result["results"]
    assert (df["rank_biserial_r"] >= -1.0).all(), "r should be >= -1"
    assert (df["rank_biserial_r"] <= 1.0).all(), "r should be <= 1"


def test_adjusted_p_range():
    """Adjusted p-values should be between 0 and 1."""
    result = run_differential(
        species="yeast",
        gene_ids_a=YEAST_RP_GENES,
        gene_ids_b=YEAST_GCN4_GENES,
        n_bootstrap=200,
        seed=42,
    )

    df = result["results"]
    assert (df["adjusted_p"] >= 0.0).all(), "adj_p should be >= 0"
    assert (df["adjusted_p"] <= 1.0).all(), "adj_p should be <= 1"


def test_labels_stored():
    """Labels should be stored correctly."""
    result = run_differential(
        species="yeast",
        gene_ids_a=YEAST_RP_GENES,
        gene_ids_b=YEAST_GCN4_GENES,
        labels=("Ribosomal", "Gcn4 targets"),
        n_bootstrap=200,
        seed=42,
    )

    assert result["labels"] == ("Ribosomal", "Gcn4 targets")


def test_n_genes():
    """n_genes_a and n_genes_b should be correct."""
    result = run_differential(
        species="yeast",
        gene_ids_a=YEAST_RP_GENES,
        gene_ids_b=YEAST_GCN4_GENES,
        n_bootstrap=200,
        seed=42,
    )

    assert result["n_genes_a"] == len(YEAST_RP_GENES)
    assert result["n_genes_b"] <= len(YEAST_GCN4_GENES)  # Some may be unmapped


# ═══════════════════════════════════════════════════════════════════════════════
# Output tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_output_dir():
    """Should write TSV files when output_dir is specified."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        result = run_differential(
            species="yeast",
            gene_ids_a=YEAST_RP_GENES,
            gene_ids_b=YEAST_GCN4_GENES,
            n_bootstrap=200,
            seed=42,
            output_dir=output_dir,
        )

        # Check TSV file exists
        tsv_path = output_dir / "differential_results.tsv"
        assert tsv_path.exists(), "TSV file should be created"

        # Check contents
        import pandas as pd
        df = pd.read_csv(tsv_path, sep="\t")
        assert len(df) == 61


def test_self_vs_self_no_significant():
    """Same gene list vs itself should have no significant results."""
    result = run_differential(
        species="yeast",
        gene_ids_a=YEAST_RP_GENES,
        gene_ids_b=YEAST_RP_GENES,
        n_bootstrap=200,
        seed=42,
    )

    df = result["results"]
    sig = df[df["adjusted_p"] < 0.05]
    assert len(sig) == 0, "Same list vs itself should have no significant differences"


def test_custom_labels():
    """Custom labels should be stored correctly."""
    result = run_differential(
        species="yeast",
        gene_ids_a=YEAST_RP_GENES,
        gene_ids_b=YEAST_GCN4_GENES,
        labels=("Custom A", "Custom B"),
        n_bootstrap=200,
        seed=42,
    )

    assert result["labels"] == ("Custom A", "Custom B")


# ═══════════════════════════════════════════════════════════════════════════════
# Report tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_report_generates():
    """HTML report should generate and be valid."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "report.html"
        result_path = generate_differential_report(
            species="yeast",
            gene_ids_a=YEAST_RP_GENES,
            gene_ids_b=YEAST_GCN4_GENES,
            output=output_path,
            n_bootstrap=200,
            seed=42,
        )

        # Check file exists
        assert result_path.exists()
        assert result_path == output_path

        # Check HTML is valid
        html = result_path.read_text()
        assert "<!DOCTYPE html>" in html
        assert "<html>" in html
        assert "Differential Codon Usage" in html


def test_report_has_plots():
    """Report should contain base64-encoded images."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "report.html"
        generate_differential_report(
            species="yeast",
            gene_ids_a=YEAST_RP_GENES,
            gene_ids_b=YEAST_GCN4_GENES,
            output=output_path,
            n_bootstrap=200,
            seed=42,
        )

        html = output_path.read_text()
        # Check for base64 images
        assert "data:image/png;base64," in html
        # Should have 2 plots
        assert html.count("data:image/png;base64,") == 2


# ═══════════════════════════════════════════════════════════════════════════════
# CLI tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_cli_differential():
    """Test CLI differential command."""
    from codonscope.cli import main

    with tempfile.TemporaryDirectory() as tmpdir:
        # Write gene lists
        list_a_path = Path(tmpdir) / "list_a.txt"
        list_b_path = Path(tmpdir) / "list_b.txt"
        output_path = Path(tmpdir) / "differential_report.html"

        list_a_path.write_text("\n".join(YEAST_RP_GENES))
        list_b_path.write_text("\n".join(YEAST_GCN4_GENES))

        # Run CLI
        ret = main([
            "differential",
            "--species", "yeast",
            "--list-a", str(list_a_path),
            "--list-b", str(list_b_path),
            "--labels", "RP", "Gcn4",
            "--output", str(output_path),
            "--n-bootstrap", "200",
            "--seed", "42",
        ])

        # Check return code
        assert ret == 0

        # Check output file
        assert output_path.exists()
        html = output_path.read_text()
        assert "Differential Codon Usage" in html
        assert "RP" in html
        assert "Gcn4" in html
