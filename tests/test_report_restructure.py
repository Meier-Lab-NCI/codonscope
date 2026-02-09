"""Tests for Tier 1 report restructure: executive summary, collapsible sections, data export."""

import json
import tempfile
from pathlib import Path

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
def rp_report():
    """Generate a report once, shared by all tests in this module."""
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
        content = result.read_text()
        data_dir = Path(tmpdir) / "rp_report_data"
        yield {
            "content": content,
            "output": result,
            "data_dir": data_dir,
            "tmpdir": tmpdir,
        }


class TestExecutiveSummary:
    """Test executive summary section."""

    def test_executive_summary_present(self, rp_report):
        assert "Executive Summary" in rp_report["content"]

    def test_executive_summary_has_top_codons(self, rp_report):
        content = rp_report["content"]
        assert "Top Enriched" in content
        assert "Top Depleted" in content

    def test_executive_summary_has_attribution(self, rp_report):
        assert "Attribution:" in rp_report["content"]

    def test_executive_summary_has_collision(self, rp_report):
        assert "Collision potential:" in rp_report["content"]

    def test_executive_summary_has_cai(self, rp_report):
        assert "CAI:" in rp_report["content"]

    def test_executive_summary_has_waterfall(self, rp_report):
        # Waterfall chart should be base64-embedded image
        content = rp_report["content"]
        assert "Executive Summary: Ranked Codon Z-scores" in content or "data:image/png;base64" in content


class TestCollapsibleSections:
    """Test collapsible <details> sections."""

    def test_has_collapsible_sections(self, rp_report):
        content = rp_report["content"]
        assert "collapsible-section" in content

    def test_has_details_elements(self, rp_report):
        content = rp_report["content"]
        assert "<details" in content
        assert "<summary>" in content

    def test_section_headers_still_present(self, rp_report):
        """Backward compat: existing section headers still in HTML inside details."""
        content = rp_report["content"]
        assert "1. Codon Enrichment Analysis" in content
        assert "2. Dicodon Enrichment Analysis" in content
        assert "3. AA vs Synonymous Attribution" in content
        assert "4. Weighted tRNA Adaptation Index" in content
        assert "5. Collision Potential Analysis" in content
        assert "6. Translational Demand Analysis" in content

    def test_summary_lines_contain_metrics(self, rp_report):
        """Each collapsible summary should contain relevant metrics."""
        content = rp_report["content"]
        # Check that at least some summary lines have metrics
        assert "enriched" in content.lower()
        assert "depleted" in content.lower()


class TestExpandCollapseControls:
    """Test expand/collapse all buttons."""

    def test_expand_all_button(self, rp_report):
        content = rp_report["content"]
        assert "Expand All" in content

    def test_collapse_all_button(self, rp_report):
        content = rp_report["content"]
        assert "Collapse All" in content

    def test_expand_controls_div(self, rp_report):
        content = rp_report["content"]
        assert "expand-controls" in content


class TestDataExport:
    """Test additional data export files."""

    def test_per_gene_codon_freq(self, rp_report):
        freq_file = rp_report["data_dir"] / "per_gene_codon_freq.tsv"
        assert freq_file.exists(), f"Missing {freq_file}"

    def test_gene_metadata(self, rp_report):
        meta_file = rp_report["data_dir"] / "gene_metadata.tsv"
        assert meta_file.exists(), f"Missing {meta_file}"

    def test_analysis_config_json(self, rp_report):
        config_file = rp_report["data_dir"] / "analysis_config.json"
        assert config_file.exists(), f"Missing {config_file}"

    def test_analysis_config_content(self, rp_report):
        config_file = rp_report["data_dir"] / "analysis_config.json"
        config = json.loads(config_file.read_text())
        assert config["species"] == "yeast"
        assert config["n_input_genes"] == len(YEAST_RP_GENES)
        assert config["n_bootstrap"] == 200
        assert config["seed"] == 42
        assert "codonscope_version" in config
        assert "timestamp" in config

    def test_per_gene_freq_has_all_codons(self, rp_report):
        import pandas as pd
        freq_file = rp_report["data_dir"] / "per_gene_codon_freq.tsv"
        df = pd.read_csv(freq_file, sep="\t", index_col=0)
        # Should have 61 sense codons as columns
        assert df.shape[1] == 61


class TestFooter:
    """Test footer content."""

    def test_footer_has_deep_dive_note(self, rp_report):
        content = rp_report["content"]
        assert "deep-dive" in content.lower() or "deep dive" in content.lower()


class TestCSS:
    """Test CSS additions."""

    def test_collapsible_css(self, rp_report):
        content = rp_report["content"]
        assert "details.collapsible-section" in content

    def test_executive_summary_css(self, rp_report):
        content = rp_report["content"]
        assert ".executive-summary" in content
