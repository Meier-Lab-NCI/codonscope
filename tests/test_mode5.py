"""Tests for Mode 5: AA vs Codon Disentanglement."""

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
# Unit tests for disentanglement internals
# ═══════════════════════════════════════════════════════════════════════════════


class TestCodonTable:
    """Verify the codon table and AA families are well-formed."""

    def test_codon_table_covers_61_sense_codons(self):
        from codonscope.modes.mode5_disentangle import CODON_TABLE
        assert len(CODON_TABLE) == 61

    def test_20_amino_acids(self):
        from codonscope.modes.mode5_disentangle import AMINO_ACIDS
        assert len(AMINO_ACIDS) == 20

    def test_aa_families_sum_to_61(self):
        from codonscope.modes.mode5_disentangle import AA_FAMILIES
        total = sum(len(codons) for codons in AA_FAMILIES.values())
        assert total == 61

    def test_met_has_one_codon(self):
        from codonscope.modes.mode5_disentangle import AA_FAMILIES
        assert AA_FAMILIES["Met"] == ["ATG"]

    def test_trp_has_one_codon(self):
        from codonscope.modes.mode5_disentangle import AA_FAMILIES
        assert AA_FAMILIES["Trp"] == ["TGG"]

    def test_leu_has_six_codons(self):
        from codonscope.modes.mode5_disentangle import AA_FAMILIES
        assert len(AA_FAMILIES["Leu"]) == 6

    def test_ser_has_six_codons(self):
        from codonscope.modes.mode5_disentangle import AA_FAMILIES
        assert len(AA_FAMILIES["Ser"]) == 6

    def test_arg_has_six_codons(self):
        from codonscope.modes.mode5_disentangle import AA_FAMILIES
        assert len(AA_FAMILIES["Arg"]) == 6


class TestAAFrequencies:
    """Test amino acid frequency computation."""

    def test_simple_sequence(self):
        from codonscope.modes.mode5_disentangle import (
            AMINO_ACIDS,
            _compute_aa_frequencies,
        )
        # ATG AAA GAA = Met, Lys, Glu — 3 codons, 3 different AAs
        seqs = {"gene1": "ATGAAAGAA"}
        freqs = _compute_aa_frequencies(seqs)
        assert freqs.shape == (1, 20)
        # Each AA should be 1/3
        aa_idx = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
        assert abs(freqs[0, aa_idx["Met"]] - 1 / 3) < 0.01
        assert abs(freqs[0, aa_idx["Lys"]] - 1 / 3) < 0.01
        assert abs(freqs[0, aa_idx["Glu"]] - 1 / 3) < 0.01

    def test_frequencies_sum_to_one(self):
        from codonscope.modes.mode5_disentangle import _compute_aa_frequencies
        seqs = {"g1": "ATGGCTGCTGCTAAAGAAGAA"}  # Met Ala Ala Ala Lys Glu Glu
        freqs = _compute_aa_frequencies(seqs)
        assert abs(freqs[0].sum() - 1.0) < 0.01


class TestRSCU:
    """Test RSCU computation."""

    def test_equal_usage_gives_rscu_1(self):
        from codonscope.modes.mode5_disentangle import (
            SENSE_CODONS,
            _compute_rscu_per_gene,
        )
        # Leu has 6 codons. If we use each once: RSCU = 6 * 1/6 = 1.0
        seq = "TTATTGCTTCTCCTACTG"  # TTA TTG CTT CTC CTA CTG — one each Leu
        seqs = {"gene1": seq}
        rscu = _compute_rscu_per_gene(seqs)
        codon_idx = {c: i for i, c in enumerate(SENSE_CODONS)}
        for leu_codon in ["TTA", "TTG", "CTT", "CTC", "CTA", "CTG"]:
            assert abs(rscu[0, codon_idx[leu_codon]] - 1.0) < 0.01

    def test_biased_usage_gives_high_rscu(self):
        from codonscope.modes.mode5_disentangle import (
            SENSE_CODONS,
            _compute_rscu_per_gene,
        )
        # All Leu as CTG: RSCU for CTG = 6 * 6/6 = 6.0
        seq = "CTGCTGCTGCTGCTGCTG"  # 6 CTGs
        seqs = {"gene1": seq}
        rscu = _compute_rscu_per_gene(seqs)
        codon_idx = {c: i for i, c in enumerate(SENSE_CODONS)}
        assert abs(rscu[0, codon_idx["CTG"]] - 6.0) < 0.01
        # Other Leu codons should be 0
        assert abs(rscu[0, codon_idx["TTA"]] - 0.0) < 0.01


class TestAttribution:
    """Test attribution logic."""

    def test_aa_driven_classification(self):
        from codonscope.modes.mode5_disentangle import _build_attribution
        aa_results = pd.DataFrame({
            "amino_acid": ["Gly"],
            "z_score": [5.0],
            "adjusted_p": [0.001],
        })
        rscu_results = pd.DataFrame({
            "codon": ["GGT"],
            "amino_acid": ["Gly"],
            "z_score": [0.5],
            "adjusted_p": [0.5],  # Not significant
        })
        attr = _build_attribution(aa_results, rscu_results)
        assert attr.iloc[0]["attribution"] == "AA-driven"

    def test_synonymous_driven_classification(self):
        from codonscope.modes.mode5_disentangle import _build_attribution
        aa_results = pd.DataFrame({
            "amino_acid": ["Gly"],
            "z_score": [0.5],
            "adjusted_p": [0.5],  # Not significant
        })
        rscu_results = pd.DataFrame({
            "codon": ["GGT"],
            "amino_acid": ["Gly"],
            "z_score": [4.0],
            "adjusted_p": [0.001],  # Significant
        })
        attr = _build_attribution(aa_results, rscu_results)
        assert attr.iloc[0]["attribution"] == "Synonymous-driven"

    def test_both_classification(self):
        from codonscope.modes.mode5_disentangle import _build_attribution
        aa_results = pd.DataFrame({
            "amino_acid": ["Gly"],
            "z_score": [4.0],
            "adjusted_p": [0.001],
        })
        rscu_results = pd.DataFrame({
            "codon": ["GGT"],
            "amino_acid": ["Gly"],
            "z_score": [3.5],
            "adjusted_p": [0.01],
        })
        attr = _build_attribution(aa_results, rscu_results)
        assert attr.iloc[0]["attribution"] == "Both"

    def test_none_classification(self):
        from codonscope.modes.mode5_disentangle import _build_attribution
        aa_results = pd.DataFrame({
            "amino_acid": ["Gly"],
            "z_score": [0.5],
            "adjusted_p": [0.5],
        })
        rscu_results = pd.DataFrame({
            "codon": ["GGT"],
            "amino_acid": ["Gly"],
            "z_score": [0.3],
            "adjusted_p": [0.6],
        })
        attr = _build_attribution(aa_results, rscu_results)
        assert attr.iloc[0]["attribution"] == "None"


class TestSummary:
    """Test summary computation."""

    def test_empty_attribution(self):
        from codonscope.modes.mode5_disentangle import _compute_summary
        attr = pd.DataFrame({
            "attribution": ["None", "None", "None"],
        })
        s = _compute_summary(attr)
        assert s["n_significant_codons"] == 0

    def test_mixed_attribution(self):
        from codonscope.modes.mode5_disentangle import _compute_summary
        attr = pd.DataFrame({
            "attribution": ["AA-driven", "Synonymous-driven", "Both", "None"],
        })
        s = _compute_summary(attr)
        assert s["n_significant_codons"] == 3
        assert abs(s["pct_aa_driven"] - 100 / 3) < 1
        assert abs(s["pct_synonymous_driven"] - 100 / 3) < 1
        assert abs(s["pct_both"] - 100 / 3) < 1


# ═══════════════════════════════════════════════════════════════════════════════
# Integration tests with yeast data
# ═══════════════════════════════════════════════════════════════════════════════


@skip_no_data
class TestGcn4Disentangle:
    """Gcn4 targets: we saw strong GGT dicodon enrichment.

    Expect: GGT signal is primarily AA-driven (glycine enrichment from
    amino acid biosynthesis genes), not synonymous codon preference.
    """

    @pytest.fixture(autouse=True)
    def run_analysis(self):
        from codonscope.modes.mode5_disentangle import run_disentangle
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
        self.result = run_disentangle(
            species="yeast",
            gene_ids=gcn4_genes,
            n_bootstrap=1000,
            seed=42,
        )

    def test_gly_aa_enriched(self):
        """Glycine should be enriched in Gcn4 targets (AA biosynthesis)."""
        aa = self.result["aa_results"]
        gly = aa[aa["amino_acid"] == "Gly"]
        assert len(gly) == 1
        assert gly.iloc[0]["z_score"] > 1.5, (
            f"Expected Gly enrichment Z > 1.5, got {gly.iloc[0]['z_score']:.2f}"
        )

    def test_attribution_contains_ggt(self):
        """GGT should appear in the attribution table."""
        attr = self.result["attribution"]
        ggt = attr[attr["codon"] == "GGT"]
        assert len(ggt) == 1

    def test_summary_has_content(self):
        """Should detect some significant deviations."""
        s = self.result["summary"]
        assert s["n_significant_codons"] > 0

    def test_returns_all_expected_keys(self):
        assert "aa_results" in self.result
        assert "rscu_results" in self.result
        assert "attribution" in self.result
        assert "synonymous_drivers" in self.result
        assert "summary" in self.result
        assert "n_genes" in self.result

    def test_rscu_results_shape(self):
        """RSCU results should have 61 rows (one per sense codon)."""
        assert len(self.result["rscu_results"]) == 61


@skip_no_data
class TestRibosomalProteinDisentangle:
    """Yeast ribosomal proteins: expect synonymous-driven signal.

    Ribosomal proteins are known to have strong codon bias driven by
    translational selection (synonymous codon preference), not unusual
    amino acid composition.
    """

    @pytest.fixture(autouse=True)
    def run_analysis(self):
        from codonscope.modes.mode5_disentangle import run_disentangle
        rp_genes = [
            "RPL1A", "RPL1B", "RPL2A", "RPL2B", "RPL3",
            "RPL4A", "RPL4B", "RPL5", "RPL6A", "RPL6B",
            "RPL7A", "RPL7B", "RPL8A", "RPL8B", "RPL9A",
            "RPL9B", "RPL10", "RPL11A", "RPL11B", "RPL12A",
            "RPL13A", "RPL13B", "RPL14A", "RPL14B", "RPL15A",
            "RPL16A", "RPL16B", "RPL17A", "RPL17B", "RPL18A",
            "RPL19A", "RPL19B", "RPL20A", "RPL20B", "RPL21A",
            "RPL22A", "RPL22B", "RPL23A", "RPL23B", "RPL24A",
            "RPL25", "RPL26A", "RPL26B", "RPL27A", "RPL27B",
            "RPL28", "RPL29", "RPL30", "RPL31A", "RPL31B",
            "RPL32", "RPL33A", "RPL33B", "RPL34A", "RPL34B",
            "RPL35A", "RPL35B", "RPL36A", "RPL36B", "RPL37A",
            "RPL38", "RPL39", "RPL40A", "RPL40B", "RPL41A",
            "RPL42A", "RPL42B", "RPL43A", "RPL43B",
            "RPS0A", "RPS0B", "RPS1A", "RPS1B", "RPS2",
            "RPS3", "RPS4A", "RPS4B", "RPS5", "RPS6A",
            "RPS6B", "RPS7A", "RPS7B", "RPS8A", "RPS8B",
            "RPS9A", "RPS9B", "RPS10A", "RPS10B", "RPS11A",
            "RPS12", "RPS13", "RPS14A", "RPS14B", "RPS15",
            "RPS16A", "RPS16B", "RPS17A", "RPS17B", "RPS18A",
            "RPS19A", "RPS19B", "RPS20", "RPS21A", "RPS21B",
            "RPS22A", "RPS22B", "RPS23A", "RPS23B", "RPS24A",
            "RPS25A", "RPS25B", "RPS26A", "RPS26B", "RPS27A",
            "RPS28A", "RPS28B", "RPS29A", "RPS29B", "RPS30A",
            "RPS31",
        ]
        self.result = run_disentangle(
            species="yeast",
            gene_ids=rp_genes,
            n_bootstrap=1000,
            seed=42,
        )

    def test_significant_rscu_deviations(self):
        """Ribosomal proteins should have strong synonymous codon preferences."""
        rscu = self.result["rscu_results"]
        sig = rscu[rscu["adjusted_p"] < 0.05]
        assert len(sig) >= 5, (
            f"Expected >= 5 significant RSCU deviations, got {len(sig)}"
        )

    def test_synonymous_driven_present(self):
        """Attribution should include synonymous-driven codons."""
        attr = self.result["attribution"]
        syn = attr[attr["attribution"] == "Synonymous-driven"]
        assert len(syn) >= 1, "Expected synonymous-driven codons for RP genes"

    def test_many_genes_analyzed(self):
        """Should map most ribosomal protein genes."""
        assert self.result["n_genes"] >= 80


@skip_no_data
class TestCLIDisentangle:
    """Test CLI integration for disentangle subcommand."""

    def test_help(self):
        from codonscope.cli import main
        with pytest.raises(SystemExit) as exc_info:
            main(["disentangle", "--help"])
        assert exc_info.value.code == 0
