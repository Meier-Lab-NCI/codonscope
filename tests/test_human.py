"""Tests for human species support (MANE Select CDS + ID mapping)."""

import gzip
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HUMAN_DATA_DIR = Path.home() / ".codonscope" / "data" / "species" / "human"
HAVE_HUMAN_DATA = HUMAN_DATA_DIR.exists() and (
    HUMAN_DATA_DIR / "cds_sequences.fa.gz"
).exists()

skip_no_data = pytest.mark.skipif(
    not HAVE_HUMAN_DATA,
    reason="Human data not downloaded. Run: codonscope download --species human",
)


# ═══════════════════════════════════════════════════════════════════════════════
# Download file existence
# ═══════════════════════════════════════════════════════════════════════════════


@skip_no_data
class TestHumanDownloadFiles:
    """Verify all expected files were created by the human download."""

    def test_cds_fasta_exists(self):
        assert (HUMAN_DATA_DIR / "cds_sequences.fa.gz").exists()

    def test_gene_id_map_exists(self):
        assert (HUMAN_DATA_DIR / "gene_id_map.tsv").exists()

    def test_trna_copy_numbers_exists(self):
        assert (HUMAN_DATA_DIR / "trna_copy_numbers.tsv").exists()

    def test_wobble_rules_exists(self):
        assert (HUMAN_DATA_DIR / "wobble_rules.tsv").exists()

    def test_background_mono_exists(self):
        assert (HUMAN_DATA_DIR / "background_mono.npz").exists()

    def test_background_di_exists(self):
        assert (HUMAN_DATA_DIR / "background_di.npz").exists()

    def test_background_tri_exists(self):
        assert (HUMAN_DATA_DIR / "background_tri.npz").exists()

    def test_gene_metadata_exists(self):
        assert (HUMAN_DATA_DIR / "gene_metadata.npz").exists()


# ═══════════════════════════════════════════════════════════════════════════════
# Gene ID map structure
# ═══════════════════════════════════════════════════════════════════════════════


@skip_no_data
class TestHumanGeneMap:
    """Verify the gene ID map has correct structure and reasonable counts."""

    @pytest.fixture(autouse=True)
    def load_map(self):
        self.gene_map = pd.read_csv(
            HUMAN_DATA_DIR / "gene_id_map.tsv", sep="\t"
        )

    def test_has_required_columns(self):
        required = {
            "systematic_name", "common_name", "ensembl_transcript",
            "refseq_transcript", "entrez_id", "cds_length", "gc_content",
        }
        assert required.issubset(set(self.gene_map.columns))

    def test_gene_count_range(self):
        """MANE Select should have ~18,000-20,000 genes."""
        n = len(self.gene_map)
        assert 15_000 <= n <= 22_000, f"Expected ~19K genes, got {n}"

    def test_systematic_names_are_ensg(self):
        """Systematic names should be ENSG IDs."""
        for name in self.gene_map["systematic_name"].head(20):
            assert name.startswith("ENSG"), f"Expected ENSG*, got {name}"

    def test_common_names_are_hgnc(self):
        """Spot-check that common names are HGNC symbols."""
        symbols = set(self.gene_map["common_name"].dropna())
        # Well-known genes should be present
        for gene in ["TP53", "BRCA1", "GAPDH", "ACTB"]:
            assert gene in symbols, f"{gene} not found in HGNC symbols"

    def test_entrez_ids_present(self):
        """Most entries should have Entrez IDs."""
        n_with_entrez = self.gene_map["entrez_id"].notna().sum()
        assert n_with_entrez > 15_000

    def test_cds_lengths_reasonable(self):
        """CDS lengths should all be divisible by 3 and in reasonable range."""
        lengths = self.gene_map["cds_length"]
        assert all(lengths > 0)
        assert all(lengths % 3 == 0)
        median_len = lengths.median()
        assert 500 < median_len < 5000


# ═══════════════════════════════════════════════════════════════════════════════
# CDS validation
# ═══════════════════════════════════════════════════════════════════════════════


@skip_no_data
class TestHumanCDSValidation:
    """Verify CDS sequences are well-formed."""

    @pytest.fixture(autouse=True)
    def load_seqs(self):
        from codonscope.core.sequences import SequenceDB
        self.db = SequenceDB("human")
        self.seqs = self.db.get_all_sequences()

    def test_all_divisible_by_3(self):
        for name, seq in list(self.seqs.items())[:500]:
            assert len(seq) % 3 == 0, f"{name} length {len(seq)} not div by 3"

    def test_all_start_with_atg(self):
        for name, seq in list(self.seqs.items())[:500]:
            assert seq.startswith("ATG"), f"{name} doesn't start with ATG"

    def test_no_internal_stops(self):
        stops = {"TAA", "TAG", "TGA"}
        for name, seq in list(self.seqs.items())[:500]:
            for i in range(0, len(seq), 3):
                codon = seq[i:i+3]
                assert codon not in stops, f"{name} has internal stop at pos {i}"

    def test_only_acgt(self):
        import re
        for name, seq in list(self.seqs.items())[:500]:
            assert re.fullmatch(r"[ACGT]+", seq), f"{name} has non-ACGT chars"


# ═══════════════════════════════════════════════════════════════════════════════
# ID resolution
# ═══════════════════════════════════════════════════════════════════════════════


@skip_no_data
class TestHumanIDResolution:
    """Test gene ID resolution for various human ID types."""

    @pytest.fixture(autouse=True)
    def load_db(self):
        from codonscope.core.sequences import SequenceDB
        self.db = SequenceDB("human")

    def test_hgnc_symbols(self):
        """Common HGNC symbols should resolve."""
        result = self.db.resolve_ids(["TP53", "BRCA1", "GAPDH", "ACTB"])
        assert result.n_mapped == 4
        assert result.n_unmapped == 0

    def test_hgnc_case_insensitive(self):
        """HGNC lookup should be case-insensitive."""
        result = self.db.resolve_ids(["tp53", "Brca1", "gapdh"])
        assert result.n_mapped == 3

    def test_ensg_resolution(self):
        """Ensembl gene IDs should resolve."""
        # Get some known ENSG IDs from the map
        gene_map = self.db.get_gene_metadata()
        sample_ensg = list(gene_map["systematic_name"].head(5))
        result = self.db.resolve_ids(sample_ensg)
        assert result.n_mapped == 5

    def test_enst_resolution(self):
        """Ensembl transcript IDs should resolve to gene systematic name."""
        gene_map = pd.read_csv(
            HUMAN_DATA_DIR / "gene_id_map.tsv", sep="\t"
        )
        # Pick some ENST IDs
        sample = gene_map[gene_map["ensembl_transcript"].notna()].head(3)
        enst_ids = list(sample["ensembl_transcript"])
        result = self.db.resolve_ids(enst_ids)
        assert result.n_mapped == 3

    def test_entrez_resolution(self):
        """Numeric Entrez IDs should resolve."""
        gene_map = pd.read_csv(
            HUMAN_DATA_DIR / "gene_id_map.tsv", sep="\t"
        )
        sample = gene_map[gene_map["entrez_id"].notna()].head(3)
        entrez_ids = [str(int(e)) for e in sample["entrez_id"]]
        result = self.db.resolve_ids(entrez_ids)
        assert result.n_mapped == 3

    def test_unmapped_ids(self):
        """Nonsense IDs should be unmapped."""
        result = self.db.resolve_ids(["FAKEGENE123", "NOTAREALID"])
        assert result.n_mapped == 0
        assert result.n_unmapped == 2

    def test_mixed_id_types(self):
        """Mix of HGNC + ENSG should all resolve."""
        gene_map = pd.read_csv(
            HUMAN_DATA_DIR / "gene_id_map.tsv", sep="\t"
        )
        first_ensg = gene_map["systematic_name"].iloc[0]
        result = self.db.resolve_ids(["TP53", "GAPDH", first_ensg])
        assert result.n_mapped == 3


# ═══════════════════════════════════════════════════════════════════════════════
# Background files
# ═══════════════════════════════════════════════════════════════════════════════


@skip_no_data
class TestHumanBackgrounds:
    """Verify background files have correct shapes."""

    def test_mono_shape(self):
        bg = np.load(HUMAN_DATA_DIR / "background_mono.npz")
        assert bg["mean"].shape == (61,)
        assert bg["std"].shape == (61,)
        assert bg["per_gene"].shape[1] == 61
        assert bg["per_gene"].shape[0] > 15_000

    def test_di_shape(self):
        bg = np.load(HUMAN_DATA_DIR / "background_di.npz")
        assert bg["mean"].shape == (3721,)
        assert bg["std"].shape == (3721,)
        assert bg["per_gene"].shape[1] == 3721

    def test_tri_shape(self):
        bg = np.load(HUMAN_DATA_DIR / "background_tri.npz")
        assert bg["mean"].shape == (226981,)
        assert bg["std"].shape == (226981,)
        # Tricodon should NOT have per_gene (too large)
        assert "per_gene" not in bg

    def test_mono_frequencies_sum_near_one(self):
        bg = np.load(HUMAN_DATA_DIR / "background_mono.npz")
        mean_sum = bg["mean"].sum()
        assert 0.95 < mean_sum < 1.05, f"Mono mean sums to {mean_sum}"


# ═══════════════════════════════════════════════════════════════════════════════
# tRNA and wobble rules
# ═══════════════════════════════════════════════════════════════════════════════


@skip_no_data
class TestHumanTRNA:
    """Verify tRNA and wobble rule files."""

    def test_trna_anticodon_count(self):
        df = pd.read_csv(HUMAN_DATA_DIR / "trna_copy_numbers.tsv", sep="\t")
        assert len(df) >= 30, f"Expected 30+ anticodons, got {len(df)}"

    def test_wobble_rules_cover_all_sense_codons(self):
        df = pd.read_csv(HUMAN_DATA_DIR / "wobble_rules.tsv", sep="\t")
        assert len(df) == 61, f"Expected 61 sense codons, got {len(df)}"


# ═══════════════════════════════════════════════════════════════════════════════
# Mode 1 on human ribosomal proteins
# ═══════════════════════════════════════════════════════════════════════════════


@skip_no_data
class TestHumanMode1RibosomalProteins:
    """Run Mode 1 on human ribosomal proteins as a positive control."""

    def test_ribosomal_proteins_resolve(self):
        from codonscope.core.sequences import SequenceDB
        db = SequenceDB("human")
        rp_genes = [
            "RPL3", "RPL4", "RPL5", "RPL7", "RPL8", "RPL10", "RPL11",
            "RPS2", "RPS3", "RPS5", "RPS6", "RPS8", "RPS9", "RPS14",
        ]
        result = db.resolve_ids(rp_genes)
        assert result.n_mapped >= 12, (
            f"Only {result.n_mapped}/{len(rp_genes)} ribosomal proteins mapped"
        )

    def test_mode1_mono_runs(self):
        """Mode 1 monocodon analysis should run on human RP genes."""
        from codonscope.modes.mode1_composition import run_composition

        rp_genes = [
            "RPL3", "RPL4", "RPL5", "RPL7", "RPL8", "RPL10", "RPL11",
            "RPL13", "RPL14", "RPL15",
            "RPS2", "RPS3", "RPS5", "RPS6", "RPS8", "RPS9", "RPS14",
        ]
        result = run_composition(
            species="human",
            gene_ids=rp_genes,
            k=1,
            n_bootstrap=1000,
            seed=42,
        )
        df = result["results"]
        assert len(df) == 61
        assert result["n_genes"] >= 12
        # Ribosomal proteins should show some significant codon bias
        sig = df[df["adjusted_p"] < 0.05]
        assert len(sig) >= 1, "Expected RP genes to show significant bias"
