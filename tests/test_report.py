"""Tests for HTML report generation."""

import os
import tempfile
from pathlib import Path

import pytest

DATA_DIR = Path.home() / ".codonscope" / "data"
YEAST_DIR = DATA_DIR / "species" / "yeast"
HUMAN_DIR = DATA_DIR / "species" / "human"
ORTHO_FILE = DATA_DIR / "orthologs" / "human_yeast.tsv"

pytestmark = pytest.mark.skipif(
    not YEAST_DIR.exists(),
    reason="Yeast data not downloaded",
)

# ── Yeast RP gene list ───────────────────────────────────────────────────────
YEAST_RP_GENES = [
    "RPL5", "RPL10", "RPL11A", "RPL3", "RPL4A", "RPL6A", "RPL7A",
    "RPL8A", "RPL13A", "RPL15A", "RPL19A", "RPL22A", "RPL25",
    "RPL30", "RPL31A", "RPS3", "RPS5", "RPS8A", "RPS15", "RPS20",
]

# ── Yeast Gcn4 target gene list ──────────────────────────────────────────────
YEAST_GCN4_GENES = [
    "ARG1", "ARG3", "ARG4", "ARG8", "ARO1", "ARO2", "ARO3", "ARO4",
    "ASN1", "ASN2", "CPA1", "CPA2", "GLN1", "GLT1",
    "HIS1", "HIS3", "HIS4", "HIS5", "HOM2", "HOM3", "HOM6",
    "ILV1", "ILV2", "ILV3", "ILV5", "LEU1", "LEU2", "LEU4",
    "LYS1", "LYS2", "LYS4", "LYS9", "LYS20",
    "MET6", "MET17", "SER1", "SER2", "SHM1", "SHM2",
    "THR1", "THR4", "TRP2", "TRP3", "TRP4", "TRP5", "TYR1",
    "ADE1", "ADE2", "ADE4", "ADE8", "ADE16", "ADE17",
    "GDH1", "GAP1", "PUT1", "PUT2", "ACO1",
]


# ═══════════════════════════════════════════════════════════════════════════════
# Core report generation tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestReportGeneration:
    """Test generate_report() function."""

    def test_rp_report_basic(self):
        from codonscope.report import generate_report

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "rp_report.html"
            result = generate_report(
                species="yeast",
                gene_ids=YEAST_RP_GENES,
                output=output,
                n_bootstrap=200,
                seed=42,
            )
            assert result.exists()
            assert result.stat().st_size > 10_000  # at least 10 KB

    def test_report_is_valid_html(self):
        from codonscope.report import generate_report

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "report.html"
            generate_report(
                species="yeast",
                gene_ids=YEAST_RP_GENES,
                output=output,
                n_bootstrap=200,
                seed=42,
            )
            content = output.read_text()
            assert content.startswith("<!DOCTYPE html>")
            assert "</html>" in content
            assert "<style>" in content

    def test_report_contains_all_mode_sections(self):
        from codonscope.report import generate_report

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "report.html"
            generate_report(
                species="yeast",
                gene_ids=YEAST_RP_GENES,
                output=output,
                n_bootstrap=200,
                seed=42,
            )
            content = output.read_text()
            assert "Gene List Summary" in content
            assert "Mode 1: Monocodon Composition" in content
            assert "Mode 1: Dicodon Composition" in content
            assert "Mode 5: AA vs Codon Disentanglement" in content
            assert "Mode 3: Optimality Profile" in content
            assert "Mode 4: Collision Potential" in content
            assert "Mode 2: Translational Demand" in content

    def test_report_has_embedded_images(self):
        from codonscope.report import generate_report

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "report.html"
            generate_report(
                species="yeast",
                gene_ids=YEAST_RP_GENES,
                output=output,
                n_bootstrap=200,
                seed=42,
            )
            content = output.read_text()
            # Should have base64 PNG images
            assert "data:image/png;base64," in content
            # At least 5 plots (volcano x2, pie, metagene, transitions, demand)
            assert content.count("data:image/png;base64,") >= 5

    def test_report_gene_summary_section(self):
        from codonscope.report import generate_report

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "report.html"
            generate_report(
                species="yeast",
                gene_ids=YEAST_RP_GENES,
                output=output,
                n_bootstrap=200,
                seed=42,
            )
            content = output.read_text()
            # Should show gene counts
            assert "20" in content  # 20 input genes
            assert "Mean CDS length" in content
            assert "Mean GC content" in content

    def test_report_shows_unmapped_genes(self):
        from codonscope.report import generate_report

        genes_with_bad = YEAST_RP_GENES + ["FAKE_GENE_1", "FAKE_GENE_2"]
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "report.html"
            generate_report(
                species="yeast",
                gene_ids=genes_with_bad,
                output=output,
                n_bootstrap=200,
                seed=42,
            )
            content = output.read_text()
            assert "FAKE_GENE_1" in content
            assert "Unmapped" in content


class TestGcn4Report:
    """Test report with Gcn4 target genes."""

    def test_gcn4_report_generates(self):
        from codonscope.report import generate_report

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "gcn4_report.html"
            result = generate_report(
                species="yeast",
                gene_ids=YEAST_GCN4_GENES,
                output=output,
                n_bootstrap=200,
                seed=42,
            )
            assert result.exists()
            content = result.read_text()
            assert "Mode 1: Monocodon" in content
            assert "Mode 5" in content

    def test_gcn4_dicodon_shows_enrichment(self):
        """Gcn4 genes should show dicodon enrichment results."""
        from codonscope.report import generate_report

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "gcn4_report.html"
            generate_report(
                species="yeast",
                gene_ids=YEAST_GCN4_GENES,
                output=output,
                n_bootstrap=200,
                seed=42,
            )
            content = output.read_text()
            assert "Mode 1: Dicodon Composition" in content
            assert "Top Enriched" in content


# ═══════════════════════════════════════════════════════════════════════════════
# Cross-species report tests
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.skipif(
    not HUMAN_DIR.exists() or not ORTHO_FILE.exists(),
    reason="Human or ortholog data not downloaded",
)
class TestCrossSpeciesReport:
    """Test report with cross-species comparison."""

    def test_rp_report_with_human(self):
        from codonscope.report import generate_report

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "rp_cross.html"
            result = generate_report(
                species="yeast",
                gene_ids=YEAST_RP_GENES,
                output=output,
                species2="human",
                n_bootstrap=200,
                seed=42,
            )
            assert result.exists()
            content = result.read_text()
            assert "Mode 6: Cross-Species Comparison" in content
            assert "Most Conserved" in content
            assert "Most Divergent" in content

    def test_cross_species_has_correlation_plot(self):
        from codonscope.report import generate_report

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "rp_cross.html"
            generate_report(
                species="yeast",
                gene_ids=YEAST_RP_GENES,
                output=output,
                species2="human",
                n_bootstrap=200,
                seed=42,
            )
            content = output.read_text()
            # Mode 6 should add at least one more plot
            assert content.count("data:image/png;base64,") >= 6


# ═══════════════════════════════════════════════════════════════════════════════
# CLI tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestCLIReport:
    """Test the report CLI subcommand."""

    def test_report_help(self):
        from codonscope.cli import main
        with pytest.raises(SystemExit) as exc_info:
            main(["report", "--help"])
        assert exc_info.value.code == 0

    def test_report_requires_species(self):
        from codonscope.cli import main
        with pytest.raises(SystemExit) as exc_info:
            main(["report", "--genes", "test.txt"])
        assert exc_info.value.code != 0

    def test_report_requires_genes(self):
        from codonscope.cli import main
        with pytest.raises(SystemExit) as exc_info:
            main(["report", "--species", "yeast"])
        assert exc_info.value.code != 0

    def test_report_cli_runs(self):
        """Test full CLI report generation."""
        from codonscope.cli import main
        import tempfile

        # Create a gene list file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("\n".join(YEAST_RP_GENES[:15]))
            gene_file = f.name

        with tempfile.TemporaryDirectory() as tmpdir:
            output = os.path.join(tmpdir, "cli_report.html")
            try:
                ret = main([
                    "report",
                    "--species", "yeast",
                    "--genes", gene_file,
                    "--output", output,
                    "--n-bootstrap", "200",
                    "--seed", "42",
                ])
                assert ret == 0
                assert os.path.exists(output)
                assert os.path.getsize(output) > 10_000
            finally:
                os.unlink(gene_file)


# ═══════════════════════════════════════════════════════════════════════════════
# Edge cases
# ═══════════════════════════════════════════════════════════════════════════════


class TestReportEdgeCases:
    """Test edge cases and error handling."""

    def test_report_self_contained(self):
        """Report should be a single self-contained HTML file with no external refs."""
        from codonscope.report import generate_report

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "report.html"
            generate_report(
                species="yeast",
                gene_ids=YEAST_RP_GENES,
                output=output,
                n_bootstrap=200,
                seed=42,
            )
            content = output.read_text()
            # No external CSS or JS references
            assert 'href="http' not in content
            assert 'src="http' not in content
            # All images are inline base64
            assert ".png" not in content or "base64" in content

    def test_report_inline_css(self):
        """Report should have inline CSS in a <style> tag."""
        from codonscope.report import generate_report

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "report.html"
            generate_report(
                species="yeast",
                gene_ids=YEAST_RP_GENES,
                output=output,
                n_bootstrap=200,
                seed=42,
            )
            content = output.read_text()
            assert "<style>" in content
            assert "font-family" in content

    def test_report_has_version(self):
        from codonscope.report import generate_report

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "report.html"
            generate_report(
                species="yeast",
                gene_ids=YEAST_RP_GENES,
                output=output,
                n_bootstrap=200,
                seed=42,
            )
            content = output.read_text()
            assert "CodonScope v" in content

    def test_output_directory_created(self):
        """Report should create parent directories if needed."""
        from codonscope.report import generate_report

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "subdir" / "nested" / "report.html"
            generate_report(
                species="yeast",
                gene_ids=YEAST_RP_GENES,
                output=output,
                n_bootstrap=200,
                seed=42,
            )
            assert output.exists()
