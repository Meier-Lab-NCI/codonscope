"""Tests for Mode 6: Cross-Species Comparison.

Requires both yeast and human data downloaded, plus ortholog mapping.
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ─── Skip if data not available ──────────────────────────────────────────────
DATA_DIR = Path.home() / ".codonscope" / "data"
YEAST_DIR = DATA_DIR / "species" / "yeast"
HUMAN_DIR = DATA_DIR / "species" / "human"
ORTHO_FILE = DATA_DIR / "orthologs" / "human_yeast.tsv"

pytestmark = pytest.mark.skipif(
    not YEAST_DIR.exists() or not HUMAN_DIR.exists() or not ORTHO_FILE.exists(),
    reason="Yeast, human, or ortholog data not downloaded",
)


# ═══════════════════════════════════════════════════════════════════════════════
# OrthologDB unit tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestOrthologDB:
    """Test the OrthologDB class."""

    def test_load_human_yeast(self):
        from codonscope.core.orthologs import OrthologDB
        odb = OrthologDB("human", "yeast")
        assert odb.n_pairs >= 500

    def test_load_yeast_human(self):
        """Order shouldn't matter."""
        from codonscope.core.orthologs import OrthologDB
        odb = OrthologDB("yeast", "human")
        assert odb.n_pairs >= 500

    def test_bidirectional_mapping(self):
        from codonscope.core.orthologs import OrthologDB
        odb = OrthologDB("human", "yeast")

        pairs = odb.get_all_pairs()
        assert len(pairs) > 0

        # Forward: human → yeast
        human_gene = pairs[0][0]
        yeast_gene = pairs[0][1]

        fwd = odb.map_genes([human_gene], from_species="human")
        assert human_gene in fwd
        assert fwd[human_gene] == yeast_gene

        # Reverse: yeast → human
        rev = odb.map_genes([yeast_gene], from_species="yeast")
        assert yeast_gene in rev
        assert rev[yeast_gene] == human_gene

    def test_unmapped_gene(self):
        from codonscope.core.orthologs import OrthologDB
        odb = OrthologDB("human", "yeast")
        result = odb.map_genes(["FAKE_GENE_123"], from_species="human")
        assert len(result) == 0

    def test_invalid_from_species(self):
        from codonscope.core.orthologs import OrthologDB
        odb = OrthologDB("human", "yeast")
        with pytest.raises(ValueError, match="from_species"):
            odb.map_genes(["X"], from_species="mouse")

    def test_missing_species_pair(self):
        from codonscope.core.orthologs import OrthologDB
        with pytest.raises(FileNotFoundError):
            OrthologDB("mouse", "zebrafish")

    def test_repr(self):
        from codonscope.core.orthologs import OrthologDB
        odb = OrthologDB("human", "yeast")
        r = repr(odb)
        assert "human" in r
        assert "yeast" in r
        assert str(odb.n_pairs) in r


# ═══════════════════════════════════════════════════════════════════════════════
# RSCU correlation unit tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestRSCUCorrelation:
    """Test per-gene RSCU vector and correlation computation."""

    def test_gene_rscu_vector_basic(self):
        from codonscope.modes.mode6_compare import _gene_rscu_vector
        from codonscope.core.codons import SENSE_CODONS

        # A sequence with known codon content
        # 20 codons to exceed minimum length
        seq = "ATG" + "AAA" * 10 + "AAG" * 5 + "GCT" * 3 + "TGA"
        # Strip stop codon for counting
        seq_no_stop = seq[:-3]
        rscu = _gene_rscu_vector(seq_no_stop)

        assert rscu.shape == (61,)
        # AAA and AAG are Lys (2-fold): RSCU for AAA should be > 1
        aaa_idx = SENSE_CODONS.index("AAA")
        aag_idx = SENSE_CODONS.index("AAG")
        assert rscu[aaa_idx] > 1.0  # preferred
        assert rscu[aag_idx] < 1.0  # avoided

    def test_gene_rscu_vector_short_returns_nan(self):
        from codonscope.modes.mode6_compare import _gene_rscu_vector
        # Very short sequence (< 10 codons)
        seq = "ATGAAAGAA"
        rscu = _gene_rscu_vector(seq)
        assert np.all(np.isnan(rscu))

    def test_per_gene_correlations_identical(self):
        """Identical sequences should have r = 1.0."""
        from codonscope.modes.mode6_compare import _per_gene_rscu_correlations
        # Build a long enough sequence
        np.random.seed(42)
        codons = ["ATG"] + list(np.random.choice(
            ["AAA", "AAG", "GAA", "GAG", "GCT", "GCC", "GCA", "GCG",
             "TTT", "TTC", "GTT", "GTC", "GTA", "GTG"], size=100
        ))
        seq = "".join(codons)

        pairs = [("geneA", "geneB")]
        seqs_from = {"geneA": seq}
        seqs_to = {"geneB": seq}

        corrs = _per_gene_rscu_correlations(pairs, seqs_from, seqs_to)
        assert len(corrs) == 1
        assert corrs[0] > 0.99  # should be ~1.0

    def test_per_gene_correlations_range(self):
        """Correlations should be in [-1, 1]."""
        from codonscope.modes.mode6_compare import _per_gene_rscu_correlations
        from codonscope.core.sequences import SequenceDB

        db_y = SequenceDB("yeast")
        db_h = SequenceDB("human")

        from codonscope.core.orthologs import OrthologDB
        odb = OrthologDB("human", "yeast")
        all_pairs = odb.get_all_pairs()[:20]

        y_seqs = db_y.get_all_sequences()
        h_seqs = db_h.get_all_sequences()

        pairs = [(p[0], p[1]) for p in all_pairs
                 if p[0] in h_seqs and p[1] in y_seqs]

        if len(pairs) < 3:
            pytest.skip("Not enough matching pairs")

        corrs = _per_gene_rscu_correlations(pairs, h_seqs, y_seqs)
        valid = [c for c in corrs if not np.isnan(c)]
        assert len(valid) > 0
        for r in valid:
            assert -1.0 <= r <= 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# Ortholog download tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestOrthologDownload:
    """Test ortholog download infrastructure."""

    def test_ortho_file_exists(self):
        assert ORTHO_FILE.exists()

    def test_ortho_file_has_enough_pairs(self):
        df = pd.read_csv(ORTHO_FILE, sep="\t")
        assert len(df) >= 500

    def test_ortho_file_columns(self):
        df = pd.read_csv(ORTHO_FILE, sep="\t")
        assert "human_gene" in df.columns
        assert "yeast_gene" in df.columns

    def test_ortho_human_genes_are_ensg(self):
        df = pd.read_csv(ORTHO_FILE, sep="\t")
        ensg_count = df["human_gene"].str.startswith("ENSG").sum()
        assert ensg_count == len(df)

    def test_ortho_yeast_genes_are_systematic(self):
        df = pd.read_csv(ORTHO_FILE, sep="\t")
        # Yeast systematic names match Y[A-P][LR]\\d{3}[WC]
        import re
        pattern = re.compile(r"Y[A-P][LR]\d{3}[WC]")
        matches = df["yeast_gene"].apply(lambda x: bool(pattern.match(str(x)))).sum()
        # Most should match (some edge cases possible)
        assert matches / len(df) > 0.95

    def test_rp_genes_in_orthologs(self):
        """RP genes should be well-represented."""
        df = pd.read_csv(ORTHO_FILE, sep="\t")
        yeast_df = pd.read_csv(YEAST_DIR / "gene_id_map.tsv", sep="\t")
        sys_to_name = dict(zip(yeast_df["systematic_name"], yeast_df["common_name"]))

        rp_orthologs = [
            g for g in df["yeast_gene"]
            if str(sys_to_name.get(g, "")).startswith(("RPL", "RPS"))
        ]
        assert len(rp_orthologs) >= 40


# ═══════════════════════════════════════════════════════════════════════════════
# Mode 6 integration tests — Yeast RP genes
# ═══════════════════════════════════════════════════════════════════════════════


class TestRPCompare:
    """Integration tests: RP gene cross-species comparison."""

    @pytest.fixture(scope="class")
    def rp_result_from_yeast(self):
        from codonscope.modes.mode6_compare import run_compare
        rp_genes = [
            "RPL5", "RPL10", "RPL11A", "RPL3", "RPL4A", "RPL6A", "RPL7A",
            "RPL8A", "RPL13A", "RPL15A", "RPL19A", "RPL22A", "RPL25",
            "RPL30", "RPL31A", "RPS3", "RPS5", "RPS8A", "RPS15", "RPS20",
            "RPS4A", "RPS6A", "RPS9A", "RPS10A", "RPL9A", "RPL35A",
        ]
        return run_compare(
            species1="yeast", species2="human",
            gene_ids=rp_genes, from_species="yeast",
            n_bootstrap=1000, seed=42,
        )

    def test_has_orthologs(self, rp_result_from_yeast):
        assert rp_result_from_yeast["n_orthologs"] >= 10

    def test_genome_orthologs(self, rp_result_from_yeast):
        assert rp_result_from_yeast["n_genome_orthologs"] >= 500

    def test_per_gene_dataframe(self, rp_result_from_yeast):
        pg = rp_result_from_yeast["per_gene"]
        assert isinstance(pg, pd.DataFrame)
        assert "rscu_correlation" in pg.columns
        assert "yeast_gene" in pg.columns
        assert "human_gene" in pg.columns

    def test_correlations_are_mostly_low(self, rp_result_from_yeast):
        """RP genes should have low RSCU correlation because species use
        different preferred codons (17/18 AAs have different preferred codons)."""
        pg = rp_result_from_yeast["per_gene"]
        median_r = pg["rscu_correlation"].median()
        # Median correlation should be modest (< 0.5) since codon preferences diverge
        assert median_r < 0.5

    def test_summary_keys(self, rp_result_from_yeast):
        s = rp_result_from_yeast["summary"]
        expected_keys = [
            "geneset_mean_r", "geneset_median_r", "geneset_std_r",
            "genome_mean_r", "genome_median_r", "genome_std_r",
            "z_score", "p_value", "mannwhitney_stat", "mannwhitney_p",
        ]
        for k in expected_keys:
            assert k in s, f"Missing key: {k}"

    def test_summary_values_are_finite(self, rp_result_from_yeast):
        s = rp_result_from_yeast["summary"]
        for k in ["geneset_mean_r", "genome_mean_r", "z_score", "p_value"]:
            assert np.isfinite(s[k]), f"{k} is not finite: {s[k]}"

    def test_divergent_analysis(self, rp_result_from_yeast):
        div = rp_result_from_yeast["divergent_analysis"]
        assert isinstance(div, pd.DataFrame)
        if len(div) > 0:
            assert "fraction_different" in div.columns

    def test_scatter_data_has_amino_acids(self, rp_result_from_yeast):
        scatter = rp_result_from_yeast["scatter_data"]
        assert isinstance(scatter, dict)
        # Should have scatter data for multi-synonym AAs
        assert len(scatter) > 0
        # Check one AA has the expected columns
        for aa, df in scatter.items():
            assert "codon" in df.columns
            assert f"rscu_yeast" in df.columns
            assert f"rscu_human" in df.columns
            break

    def test_id_summary(self, rp_result_from_yeast):
        ids = rp_result_from_yeast["id_summary"]
        assert ids["n_input"] > 0
        assert ids["n_mapped"] > 0
        assert ids["n_valid_pairs"] > 0

    def test_from_to_species(self, rp_result_from_yeast):
        assert rp_result_from_yeast["from_species"] == "yeast"
        assert rp_result_from_yeast["to_species"] == "human"


# ═══════════════════════════════════════════════════════════════════════════════
# Mode 6 integration tests — from human side
# ═══════════════════════════════════════════════════════════════════════════════


class TestHumanRPCompare:
    """Test Mode 6 with human RP genes mapped to yeast."""

    @pytest.fixture(scope="class")
    def rp_result_from_human(self):
        from codonscope.modes.mode6_compare import run_compare
        human_rp = [
            "RPL5", "RPL10", "RPL11", "RPL3", "RPL4", "RPL6", "RPL7",
            "RPL8", "RPL13", "RPL15", "RPL19", "RPL22", "RPL25",
            "RPL30", "RPS3", "RPS5", "RPS8", "RPS15", "RPS20",
            "RPS4X", "RPS6", "RPS9", "RPS10",
        ]
        return run_compare(
            species1="yeast", species2="human",
            gene_ids=human_rp, from_species="human",
            n_bootstrap=1000, seed=42,
        )

    def test_has_orthologs(self, rp_result_from_human):
        assert rp_result_from_human["n_orthologs"] >= 5

    def test_from_to_species(self, rp_result_from_human):
        assert rp_result_from_human["from_species"] == "human"
        assert rp_result_from_human["to_species"] == "yeast"

    def test_correlations_low(self, rp_result_from_human):
        pg = rp_result_from_human["per_gene"]
        median_r = pg["rscu_correlation"].median()
        assert median_r < 0.5

    def test_genome_correlations_exist(self, rp_result_from_human):
        gc = rp_result_from_human["genome_correlations"]
        assert isinstance(gc, pd.DataFrame)
        assert len(gc) >= 500


# ═══════════════════════════════════════════════════════════════════════════════
# Divergent gene analysis tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestDivergentAnalysis:
    """Test divergent gene analysis."""

    def test_divergent_genes_have_different_preferred_codons(self):
        """Most divergent orthologs should use different preferred codons
        for many amino acids."""
        from codonscope.modes.mode6_compare import run_compare

        # Use a broad set of genes to get divergent ones
        from codonscope.core.orthologs import OrthologDB
        odb = OrthologDB("human", "yeast")
        all_pairs = odb.get_all_pairs()
        yeast_genes = [p[1] for p in all_pairs[:50]]

        result = run_compare(
            species1="yeast", species2="human",
            gene_ids=yeast_genes, from_species="yeast",
            n_bootstrap=500, seed=42,
        )

        div = result["divergent_analysis"]
        if len(div) == 0:
            pytest.skip("No divergent genes found")

        # Divergent genes should have different preferred codons for many AAs
        mean_frac_diff = div["fraction_different"].mean()
        assert mean_frac_diff > 0.3, (
            f"Expected divergent genes to differ in >30% of AA families, "
            f"got {mean_frac_diff:.2f}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Output file tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestOutputFiles:
    """Test that output files are written correctly."""

    def test_output_files_created(self):
        from codonscope.modes.mode6_compare import run_compare

        with tempfile.TemporaryDirectory() as tmpdir:
            from codonscope.core.orthologs import OrthologDB
            odb = OrthologDB("human", "yeast")
            all_pairs = odb.get_all_pairs()
            yeast_genes = [p[1] for p in all_pairs[:30]]

            run_compare(
                species1="yeast", species2="human",
                gene_ids=yeast_genes, from_species="yeast",
                n_bootstrap=500, seed=42,
                output_dir=tmpdir,
            )

            assert (Path(tmpdir) / "compare_per_gene.tsv").exists()
            assert (Path(tmpdir) / "compare_genome_correlations.tsv").exists()
            assert (Path(tmpdir) / "compare_summary.txt").exists()


# ═══════════════════════════════════════════════════════════════════════════════
# CLI tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestCLICompare:
    """Test the compare CLI subcommand."""

    def test_compare_help(self):
        from codonscope.cli import main
        with pytest.raises(SystemExit) as exc_info:
            main(["compare", "--help"])
        assert exc_info.value.code == 0

    def test_compare_requires_species(self):
        from codonscope.cli import main
        with pytest.raises(SystemExit) as exc_info:
            main(["compare", "--genes", "test.txt"])
        assert exc_info.value.code != 0

    def test_compare_from_species_arg(self):
        from codonscope.cli import main
        with pytest.raises(SystemExit) as exc_info:
            main(["compare", "--species1", "yeast", "--species2", "human",
                  "--genes", "test.txt", "--from-species", "human", "--help"])
        # --help exits with 0
        assert exc_info.value.code == 0
