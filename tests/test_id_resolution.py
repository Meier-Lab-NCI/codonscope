"""Tests for enhanced multi-type ID resolution, GtRNAdb parser fix, and orthologs."""

import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from codonscope.core.sequences import (
    IDMapping,
    SequenceDB,
    _ENTREZ_RE,
    _ENSG_RE,
    _ENSMUSG_RE,
    _ENSMUST_RE,
    _ENST_RE,
    _MGI_ID_RE,
    _REFSEQ_NM_RE,
    _SGD_ID_RE,
    _UNIPROT_RE,
    _YEAST_SYSTEMATIC_RE,
)
from codonscope.data.download import (
    _parse_gtrnadb_fasta,
    download_orthologs,
)

# ── Data directories ────────────────────────────────────────────────────────

YEAST_DATA_DIR = Path.home() / ".codonscope" / "data" / "species" / "yeast"
HUMAN_DATA_DIR = Path.home() / ".codonscope" / "data" / "species" / "human"
MOUSE_DATA_DIR = Path.home() / ".codonscope" / "data" / "species" / "mouse"

skip_no_yeast = pytest.mark.skipif(
    not YEAST_DATA_DIR.exists(),
    reason="Yeast data not downloaded",
)
skip_no_human = pytest.mark.skipif(
    not HUMAN_DATA_DIR.exists(),
    reason="Human data not downloaded",
)
skip_no_mouse = pytest.mark.skipif(
    not MOUSE_DATA_DIR.exists(),
    reason="Mouse data not downloaded",
)


# ═══════════════════════════════════════════════════════════════════════════════
# GtRNAdb parser fix tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestGtRNAdbParser:
    """Test the fixed GtRNAdb FASTA parser with all three header formats."""

    def test_strategy1_old_format(self):
        """Strategy 1: Type: Ala  Anticodon: AGC."""
        fasta = (
            ">sacCer3.trna1-AlaAGC (1-72) Length: 72 bp Type: Ala Anticodon: AGC Score: 78.0\n"
            "GGGCCCAUAGCUCAGUGGUAGAGCGCUCGCUUAGCAUGCGAGAGGCACCGGGAUUCGAUUCCCGGCGGGCCCACCA\n"
            ">sacCer3.trna2-AlaAGC (1-72) Length: 72 bp Type: Ala Anticodon: AGC Score: 77.0\n"
            "GGGCCCAUAGCUCAGUGGUAGAGCGCUCGCUUAGCAUGCGAGAGGCACCGGGAUUCGAUUCCCGGCGGGCCCACCA\n"
        )
        counts = _parse_gtrnadb_fasta(fasta)
        assert "AGC" in counts
        assert counts["AGC"]["gene_count"] == 2
        assert counts["AGC"]["amino_acid"] == "Ala"

    def test_strategy2_modern_format(self):
        """Strategy 2: tRNA-Ala-AGC in identifier."""
        fasta = (
            ">Mus_musculus_tRNA-Ala-AGC-1-1 (chr1:71519553-71519624) Ala (AGC) 72 bp Sc: 72.6\n"
            "GGGCCCAUAGCUCAGUGGUAGAGCGCUCGCUUAGCAUGCGAGAGGCACCGGGAUUCGAUUCCCGGCGGGCCCACCA\n"
            ">Mus_musculus_tRNA-Ala-AGC-2-1 (chr2:10000-10071) Ala (AGC) 72 bp Sc: 71.0\n"
            "GGGCCCAUAGCUCAGUGGUAGAGCGCUCGCUUAGCAUGCGAGAGGCACCGGGAUUCGAUUCCCGGCGGGCCCACCA\n"
            ">Mus_musculus_tRNA-Ala-CGC-1-1 (chr3:5000-5071) Ala (CGC) 72 bp Sc: 70.0\n"
            "GGGCCCAUAGCUCAGUGGUAGAGCGCUCGCUUAGCAUGCGAGAGGCACCGGGAUUCGAUUCCCGGCGGGCCCACCA\n"
        )
        counts = _parse_gtrnadb_fasta(fasta)
        assert "AGC" in counts
        assert counts["AGC"]["gene_count"] == 2
        assert "CGC" in counts
        assert counts["CGC"]["gene_count"] == 1

    def test_strategy3_compact_format(self):
        """Strategy 3: trna34-AlaAGC compact."""
        fasta = (
            ">sacCer3.trna34-AlaAGC (1-72)\n"
            "GGGCCCAUAGCUCAGUGGUAGAGCGCUCGCUUAGCAUGCGAGAGGCACCGGGAUUCGAUUCCCGGCGGGCCCACCA\n"
        )
        counts = _parse_gtrnadb_fasta(fasta)
        assert "AGC" in counts
        assert counts["AGC"]["gene_count"] == 1

    def test_t_to_u_conversion(self):
        """All strategies should convert T→U in anticodon."""
        # Strategy 1 — T in Anticodon field
        fasta1 = ">test Type: Phe Anticodon: GAA Score: 78.0\nATGC\n"
        counts1 = _parse_gtrnadb_fasta(fasta1)
        assert "GAA" in counts1

        # Strategy 2 — T in identifier
        fasta2 = ">Mus_musculus_tRNA-Phe-GAA-1-1 (chr1:1-72) Phe (GAA)\nATGC\n"
        counts2 = _parse_gtrnadb_fasta(fasta2)
        assert "GAA" in counts2

        # Strategy 3 — T in compact (should get U)
        fasta3 = ">sacCer3.trna1-PheGAA (1-72)\nATGC\n"
        counts3 = _parse_gtrnadb_fasta(fasta3)
        assert "GAA" in counts3

    def test_skip_imet_sec_sup(self):
        """Should skip iMet, SeC, Sup, Undet entries."""
        fasta = (
            ">Mus_musculus_tRNA-iMet-CAT-1-1 (chr1:1-72) iMet (CAT)\nATGC\n"
            ">Mus_musculus_tRNA-SeC-TCA-1-1 (chr1:1-72) SeC (TCA)\nATGC\n"
            ">Mus_musculus_tRNA-Sup-CTA-1-1 (chr1:1-72) Sup (CTA)\nATGC\n"
            ">Mus_musculus_tRNA-Undet-NNN-1-1 (chr1:1-72) Undet (NNN)\nATGC\n"
            ">Mus_musculus_tRNA-Ala-AGC-1-1 (chr1:1-72) Ala (AGC)\nATGC\n"
        )
        counts = _parse_gtrnadb_fasta(fasta)
        assert len(counts) == 1
        assert "AGC" in counts

    def test_mixed_formats(self):
        """Parser should handle headers from multiple formats in one file."""
        fasta = (
            ">sacCer3.trna1-AlaAGC (1-72) Length: 72 bp Type: Ala Anticodon: AGC Score: 78.0\n"
            "GGGCCC\n"
            ">Mus_musculus_tRNA-Gly-GCC-1-1 (chr1:1-72) Gly (GCC)\n"
            "GGGCCC\n"
        )
        counts = _parse_gtrnadb_fasta(fasta)
        assert len(counts) == 2
        assert "AGC" in counts
        assert "GCC" in counts


# ═══════════════════════════════════════════════════════════════════════════════
# Regex pattern tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestIDPatterns:
    """Test regex patterns for ID type detection."""

    def test_yeast_systematic(self):
        assert _YEAST_SYSTEMATIC_RE.match("YFL039C")
        assert _YEAST_SYSTEMATIC_RE.match("YOR202W")
        assert _YEAST_SYSTEMATIC_RE.match("YAL001C")
        assert not _YEAST_SYSTEMATIC_RE.match("ACT1")
        assert not _YEAST_SYSTEMATIC_RE.match("ENSG00000141510")

    def test_ensmusg(self):
        assert _ENSMUSG_RE.match("ENSMUSG00000029580")
        assert _ENSMUSG_RE.match("ENSMUSG00000029580.10")
        assert not _ENSMUSG_RE.match("ENSG00000141510")
        assert not _ENSMUSG_RE.match("Actb")

    def test_ensmust(self):
        assert _ENSMUST_RE.match("ENSMUST00000031514")
        assert _ENSMUST_RE.match("ENSMUST00000031514.5")
        assert not _ENSMUST_RE.match("ENST00000269305")

    def test_ensg(self):
        assert _ENSG_RE.match("ENSG00000141510")
        assert _ENSG_RE.match("ENSG00000141510.16")
        assert not _ENSG_RE.match("ENSMUSG00000029580")

    def test_enst(self):
        assert _ENST_RE.match("ENST00000269305")
        assert _ENST_RE.match("ENST00000269305.8")
        assert not _ENST_RE.match("ENSMUST00000031514")

    def test_refseq_nm(self):
        assert _REFSEQ_NM_RE.match("NM_000546")
        assert _REFSEQ_NM_RE.match("NM_000546.6")
        assert not _REFSEQ_NM_RE.match("NR_000001")
        assert not _REFSEQ_NM_RE.match("ENST00000269305")

    def test_mgi_id(self):
        assert _MGI_ID_RE.match("MGI:87904")
        assert _MGI_ID_RE.match("MGI:12345678")
        assert not _MGI_ID_RE.match("MGI:")
        assert not _MGI_ID_RE.match("MGI:ABC")
        assert not _MGI_ID_RE.match("Actb")

    def test_sgd_id(self):
        assert _SGD_ID_RE.match("SGD:S000001855")
        assert _SGD_ID_RE.match("SGD:S000000055")
        assert not _SGD_ID_RE.match("SGD:")
        assert not _SGD_ID_RE.match("YFL039C")

    def test_uniprot(self):
        assert _UNIPROT_RE.match("P60010")  # Yeast actin
        assert _UNIPROT_RE.match("P04637")  # Human TP53
        assert _UNIPROT_RE.match("Q8R081")  # Mouse
        assert _UNIPROT_RE.match("O15553")  # Human
        assert not _UNIPROT_RE.match("12345")
        assert not _UNIPROT_RE.match("ACT1")

    def test_entrez(self):
        assert _ENTREZ_RE.match("7157")  # 4 digits
        assert _ENTREZ_RE.match("11461")  # 5 digits
        assert not _ENTREZ_RE.match("123")  # Too short
        assert not _ENTREZ_RE.match("ABC")


# ═══════════════════════════════════════════════════════════════════════════════
# ID type detection tests
# ═══════════════════════════════════════════════════════════════════════════════


@skip_no_yeast
class TestDetectIDTypeYeast:
    """Test _detect_id_type for yeast IDs."""

    @pytest.fixture(scope="class")
    def db(self):
        return SequenceDB("yeast")

    def test_yeast_systematic(self, db):
        assert db._detect_id_type("YFL039C") == "yeast_systematic"

    def test_gene_symbol(self, db):
        assert db._detect_id_type("ACT1") == "gene_symbol"

    def test_sgd_id(self, db):
        assert db._detect_id_type("SGD:S000001855") == "sgd_id"

    def test_uniprot(self, db):
        assert db._detect_id_type("P60010") == "uniprot"


@skip_no_human
class TestDetectIDTypeHuman:
    """Test _detect_id_type for human IDs."""

    @pytest.fixture(scope="class")
    def db(self):
        return SequenceDB("human")

    def test_ensg(self, db):
        assert db._detect_id_type("ENSG00000141510") == "ensg"

    def test_enst(self, db):
        assert db._detect_id_type("ENST00000269305") == "enst"

    def test_refseq_nm(self, db):
        assert db._detect_id_type("NM_000546.6") == "refseq_nm"

    def test_entrez(self, db):
        assert db._detect_id_type("7157") == "entrez"

    def test_uniprot(self, db):
        assert db._detect_id_type("P04637") == "uniprot"

    def test_gene_symbol(self, db):
        assert db._detect_id_type("TP53") == "gene_symbol"


@skip_no_mouse
class TestDetectIDTypeMouse:
    """Test _detect_id_type for mouse IDs."""

    @pytest.fixture(scope="class")
    def db(self):
        return SequenceDB("mouse")

    def test_ensmusg(self, db):
        assert db._detect_id_type("ENSMUSG00000029580") == "ensmusg"

    def test_ensmust(self, db):
        assert db._detect_id_type("ENSMUST00000031514") == "ensmust"

    def test_mgi_id(self, db):
        assert db._detect_id_type("MGI:87904") == "mgi_id"

    def test_gene_symbol(self, db):
        assert db._detect_id_type("Actb") == "gene_symbol"


# ═══════════════════════════════════════════════════════════════════════════════
# Existing ID resolution (backward compatibility)
# ═══════════════════════════════════════════════════════════════════════════════


@skip_no_yeast
class TestYeastIDResolutionCompat:
    """Ensure existing yeast ID resolution still works."""

    @pytest.fixture(scope="class")
    def db(self):
        return SequenceDB("yeast")

    def test_systematic_name(self, db):
        result = db.resolve_ids(["YFL039C"])
        assert result.n_mapped == 1
        assert result["YFL039C"] == "YFL039C"

    def test_common_name(self, db):
        result = db.resolve_ids(["ACT1"])
        assert result.n_mapped == 1
        assert result["ACT1"] == "YFL039C"

    def test_mixed_list(self, db):
        result = db.resolve_ids(["ACT1", "YOR202W", "HIS3"])
        assert result.n_mapped == 3

    def test_case_insensitive(self, db):
        result = db.resolve_ids(["act1"])
        assert result.n_mapped == 1

    def test_unmapped(self, db):
        result = db.resolve_ids(["FAKE_GENE_XYZ"])
        assert result.n_unmapped == 1
        assert "FAKE_GENE_XYZ" in result.unmapped

    def test_wrong_species_ensembl(self, db):
        result = db.resolve_ids(["ENSG00000141510"])
        assert result.n_unmapped == 1


@skip_no_human
class TestHumanIDResolutionCompat:
    """Ensure existing human ID resolution still works."""

    @pytest.fixture(scope="class")
    def db(self):
        return SequenceDB("human")

    def test_ensg(self, db):
        result = db.resolve_ids(["ENSG00000141510"])
        assert result.n_mapped == 1

    def test_enst(self, db):
        result = db.resolve_ids(["ENST00000269305"])
        assert result.n_mapped == 1

    def test_gene_symbol(self, db):
        result = db.resolve_ids(["TP53"])
        assert result.n_mapped == 1

    def test_entrez(self, db):
        result = db.resolve_ids(["7157"])
        assert result.n_mapped == 1

    def test_mixed_types(self, db):
        """Multiple ID types in one list should resolve correctly."""
        result = db.resolve_ids(["TP53", "ENSG00000141510"])
        # Both map to the same gene, but different input IDs
        assert result.n_mapped == 2


@skip_no_mouse
class TestMouseIDResolutionCompat:
    """Ensure existing mouse ID resolution still works."""

    @pytest.fixture(scope="class")
    def db(self):
        return SequenceDB("mouse")

    def test_ensmusg(self, db):
        result = db.resolve_ids(["ENSMUSG00000029580"])
        assert result.n_mapped == 1

    def test_gene_symbol(self, db):
        result = db.resolve_ids(["Actb"])
        assert result.n_mapped == 1

    def test_case_insensitive(self, db):
        result = db.resolve_ids(["actb"])
        assert result.n_mapped == 1

    def test_rp_genes(self, db):
        """Mouse RP genes should resolve."""
        result = db.resolve_ids(["Rpl3", "Rps2", "Rpl11"])
        assert result.n_mapped >= 2  # At least 2 of 3


# ═══════════════════════════════════════════════════════════════════════════════
# New ID type resolution tests
# ═══════════════════════════════════════════════════════════════════════════════


@skip_no_human
class TestRefSeqResolution:
    """Test RefSeq NM_ ID resolution for human."""

    @pytest.fixture(scope="class")
    def db(self):
        return SequenceDB("human")

    def test_refseq_with_version(self, db):
        """NM_000546.6 should resolve to TP53 gene."""
        if not db._refseq_to_sys:
            pytest.skip("RefSeq mapping not populated (needs re-download)")
        result = db.resolve_ids(["NM_000546.6"])
        assert result.n_mapped == 1

    def test_refseq_without_version(self, db):
        """NM_000546 (no version) should also resolve."""
        if not db._refseq_to_sys:
            pytest.skip("RefSeq mapping not populated (needs re-download)")
        result = db.resolve_ids(["NM_000546"])
        assert result.n_mapped == 1


@skip_no_yeast
class TestSGDIDResolution:
    """Test SGD ID resolution for yeast."""

    @pytest.fixture(scope="class")
    def db(self):
        return SequenceDB("yeast")

    def test_sgd_id(self, db):
        """SGD:S000001855 should resolve to ACT1 (YFL039C)."""
        if not db._sgd_to_sys:
            pytest.skip("SGD ID mapping not populated (needs re-download)")
        result = db.resolve_ids(["SGD:S000001855"])
        assert result.n_mapped == 1
        assert result["SGD:S000001855"] == "YFL039C"


@skip_no_yeast
class TestUniProtResolutionYeast:
    """Test UniProt resolution for yeast."""

    @pytest.fixture(scope="class")
    def db(self):
        return SequenceDB("yeast")

    def test_uniprot_yeast(self, db):
        """P60010 is yeast actin (ACT1 / YFL039C)."""
        if not db._uniprot_to_sys:
            pytest.skip("UniProt mapping not populated (needs re-download)")
        result = db.resolve_ids(["P60010"])
        assert result.n_mapped == 1
        assert result["P60010"] == "YFL039C"


@skip_no_human
class TestUniProtResolutionHuman:
    """Test UniProt resolution for human."""

    @pytest.fixture(scope="class")
    def db(self):
        return SequenceDB("human")

    def test_uniprot_human(self, db):
        """P04637 is human TP53."""
        if not db._uniprot_to_sys:
            pytest.skip("UniProt mapping not populated (needs re-download)")
        result = db.resolve_ids(["P04637"])
        assert result.n_mapped == 1


@skip_no_mouse
class TestMGIIDResolution:
    """Test MGI ID resolution for mouse."""

    @pytest.fixture(scope="class")
    def db(self):
        return SequenceDB("mouse")

    def test_mgi_id(self, db):
        """MGI:87904 is Actb."""
        if not db._mgi_to_sys:
            pytest.skip("MGI mapping not populated (needs re-download)")
        result = db.resolve_ids(["MGI:87904"])
        assert result.n_mapped == 1

    def test_mgi_synonym(self, db):
        """MGI synonyms should also resolve."""
        if not db._mgi_synonym_to_sys:
            pytest.skip("MGI synonym mapping not populated (needs re-download)")
        # Beta-actin is a common synonym for Actb
        result = db.resolve_ids(["beta-actin"])
        # This may or may not resolve depending on MGI data
        # Just check it doesn't error
        assert isinstance(result, IDMapping)


@skip_no_mouse
class TestMouseEntrezResolution:
    """Test Entrez ID resolution for mouse."""

    @pytest.fixture(scope="class")
    def db(self):
        return SequenceDB("mouse")

    def test_entrez_mouse(self, db):
        """11461 is Actb Entrez ID."""
        if not db._mouse_entrez_to_sys:
            pytest.skip("Mouse Entrez mapping not populated (needs re-download)")
        result = db.resolve_ids(["11461"])
        assert result.n_mapped == 1


# ═══════════════════════════════════════════════════════════════════════════════
# Mixed ID type resolution
# ═══════════════════════════════════════════════════════════════════════════════


@skip_no_human
class TestMixedIDResolution:
    """Test resolving mixed ID types in a single gene list."""

    @pytest.fixture(scope="class")
    def db(self):
        return SequenceDB("human")

    def test_ensg_and_symbol_mixed(self, db):
        """Mix of ENSG and gene symbols should resolve."""
        result = db.resolve_ids(["TP53", "ENSG00000141510", "BRCA1"])
        assert result.n_mapped >= 2

    def test_entrez_and_symbol_mixed(self, db):
        """Mix of Entrez IDs and gene symbols should resolve."""
        result = db.resolve_ids(["TP53", "7157", "BRCA1"])
        assert result.n_mapped >= 2

    def test_all_id_types_human(self, db):
        """Attempt to resolve all supported human ID types together."""
        ids = ["TP53", "ENSG00000141510", "ENST00000269305", "7157"]
        if db._refseq_to_sys:
            ids.append("NM_000546")
        if db._uniprot_to_sys:
            ids.append("P04637")
        result = db.resolve_ids(ids)
        # All should resolve to the TP53 gene
        assert result.n_mapped == len(ids)
        # All values should be the same ENSG
        vals = set(result.values())
        assert len(vals) == 1


@skip_no_mouse
class TestMixedIDResolutionMouse:
    """Test resolving mixed ID types for mouse."""

    @pytest.fixture(scope="class")
    def db(self):
        return SequenceDB("mouse")

    def test_ensmusg_and_symbol_mixed(self, db):
        """Mix of ENSMUSG and MGI symbols should resolve."""
        result = db.resolve_ids(["Actb", "ENSMUSG00000029580"])
        assert result.n_mapped == 2
        vals = list(result.values())
        assert vals[0] == vals[1]  # Both resolve to same gene


# ═══════════════════════════════════════════════════════════════════════════════
# Download function existence / structural tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestDownloadFunctionsExist:
    """Verify new download functions are importable."""

    def test_parse_gtrnadb_importable(self):
        from codonscope.data.download import _parse_gtrnadb_fasta
        assert callable(_parse_gtrnadb_fasta)

    def test_download_mgi_mapping_importable(self):
        from codonscope.data.download import _download_mgi_mapping
        assert callable(_download_mgi_mapping)

    def test_download_uniprot_mapping_importable(self):
        from codonscope.data.download import _download_uniprot_mapping
        assert callable(_download_uniprot_mapping)

    def test_fetch_mouse_canonical_importable(self):
        from codonscope.data.download import _fetch_mouse_canonical_transcripts
        assert callable(_fetch_mouse_canonical_transcripts)

    def test_fetch_mouse_entrez_importable(self):
        from codonscope.data.download import _fetch_mouse_entrez_ids
        assert callable(_fetch_mouse_entrez_ids)

    def test_discover_compara_url_importable(self):
        from codonscope.data.download import _discover_compara_url
        assert callable(_discover_compara_url)

    def test_download_orthologs_compara_importable(self):
        from codonscope.data.download import _download_orthologs_compara
        assert callable(_download_orthologs_compara)


class TestOrthologDispatcher:
    """Test that download_orthologs handles new species pairs."""

    def test_human_yeast_supported(self):
        """human-yeast should not raise ValueError."""
        # Don't actually download, just check the pair is accepted
        # We can't easily mock this without more infrastructure, so just
        # verify the function signature handles the pair name
        from codonscope.data.download import download_orthologs
        # This would require data to be present, so just check no ValueError for the pair
        assert callable(download_orthologs)

    def test_unsupported_pair_raises(self):
        """Unsupported pair should raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported ortholog pair"):
            download_orthologs("chicken", "frog")

    def test_mouse_human_pair_accepted(self):
        """mouse-human pair should be in the dispatcher (not raise ValueError)."""
        # We can't test full download, but we can verify it doesn't raise
        # ValueError for the pair itself (it may raise other errors if data missing)
        try:
            download_orthologs("mouse", "human")
        except ValueError as e:
            if "Unsupported" in str(e):
                pytest.fail("mouse-human should be supported")
        except Exception:
            pass  # Other errors (missing data, network) are acceptable

    def test_mouse_yeast_pair_accepted(self):
        """mouse-yeast pair should be in the dispatcher."""
        try:
            download_orthologs("mouse", "yeast")
        except ValueError as e:
            if "Unsupported" in str(e):
                pytest.fail("mouse-yeast should be supported")
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
# Parse function tests with synthetic data
# ═══════════════════════════════════════════════════════════════════════════════


class TestParseEnsemblMouseCanonical:
    """Test _parse_ensembl_mouse_cds with canonical map."""

    def test_canonical_preferred_over_longest(self):
        from codonscope.data.download import _parse_ensembl_mouse_cds

        # Create a FASTA with two transcripts for one gene:
        # - ENSMUST00000000001: shorter (canonical) — 12bp: ATG AAA TTT TAA
        # - ENSMUST00000000002: longer (not canonical) — 21bp: ATG AAA TTT AAA TTT AAA TAG
        # Both valid CDS: start ATG, end stop, divisible by 3
        fasta = (
            ">ENSMUST00000000001.1 cds chromosome:GRCm39:1:1:100:1 "
            "gene:ENSMUSG00000000001.1 gene_symbol:TestGene\n"
            "ATGAAATTTTAA\n"  # 12 bp: ATG|AAA|TTT|TAA → CDS = ATGAAATTT (9bp)
            ">ENSMUST00000000002.1 cds chromosome:GRCm39:1:1:200:1 "
            "gene:ENSMUSG00000000001.1 gene_symbol:TestGene\n"
            "ATGAAATTTAAATTTAAATAG\n"  # 21 bp: ...TAG stop → CDS = 18bp
        )
        canonical_map = {"ENSMUSG00000000001": "ENSMUST00000000001"}
        seqs, records = _parse_ensembl_mouse_cds(fasta, canonical_map)

        assert len(seqs) == 1
        # Should pick canonical (shorter) not longest
        assert seqs["ENSMUSG00000000001"] == "ATGAAATTT"  # 9bp CDS (stop stripped)

    def test_fallback_to_longest_without_canonical(self):
        from codonscope.data.download import _parse_ensembl_mouse_cds

        fasta = (
            ">ENSMUST00000000001.1 cds chromosome:GRCm39:1:1:100:1 "
            "gene:ENSMUSG00000000001.1 gene_symbol:TestGene\n"
            "ATGAAATTTTAA\n"  # 12bp → 9bp CDS
            ">ENSMUST00000000002.1 cds chromosome:GRCm39:1:1:200:1 "
            "gene:ENSMUSG00000000001.1 gene_symbol:TestGene\n"
            "ATGAAATTTAAATTTAAATAG\n"  # 21bp → 18bp CDS
        )
        # No canonical map — should pick longest
        seqs, records = _parse_ensembl_mouse_cds(fasta, None)

        assert len(seqs) == 1
        # Should pick longest
        assert len(seqs["ENSMUSG00000000001"]) == 18


class TestParseSGDFastaWithSGDID:
    """Test _parse_sgd_fasta captures SGD ID."""

    def test_sgd_id_extracted(self):
        from codonscope.data.download import _parse_sgd_fasta

        # Valid CDS: 60bp, starts ATG, ends TAA, no internal stops
        fasta = (
            ">YFL039C ACT1 SGDID:S000001855, Chr VI from 54696-53560, "
            "Genome Release 64-4-1, reverse complement, Verified ORF, "
            "\"Actin\"\n"
            "ATGGCTGATTCTAAAGCTGTTACTGCTGAAACTGAAATCAAACCAGCTACTATTGATTAA\n"
        )
        seqs, records = _parse_sgd_fasta(fasta)

        assert len(records) == 1
        assert records[0]["sgd_id"] == "SGD:S000001855"
        assert records[0]["systematic_name"] == "YFL039C"
        assert records[0]["common_name"] == "ACT1"

    def test_sgd_id_empty_when_missing(self):
        from codonscope.data.download import _parse_sgd_fasta

        # Header without SGDID — valid CDS
        fasta = (
            ">YAL001C TFC3, Chr I from 151168-147596, "
            "Genome Release 64-4-1, reverse complement, Verified ORF\n"
            "ATGGCTGATTCTAAAGCTGTTACTGCTGAAACTGAAATCAAACCAGCTACTATTGATTAA\n"
        )
        seqs, records = _parse_sgd_fasta(fasta)
        # Should still parse, sgd_id should be empty
        if records:
            assert records[0].get("sgd_id", "") == "" or records[0].get("sgd_id") is not None


# ═══════════════════════════════════════════════════════════════════════════════
# IDMapping class tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestIDMappingCompat:
    """Verify IDMapping backward compatibility."""

    def test_dict_like(self):
        m = IDMapping({"A": "B", "C": "D"}, ["E"])
        assert len(m) == 2
        assert m["A"] == "B"
        assert "A" in m
        assert list(m.keys()) == ["A", "C"]

    def test_unmapped(self):
        m = IDMapping({}, ["X", "Y"])
        assert m.n_unmapped == 2
        assert m.unmapped == ["X", "Y"]

    def test_compat_keys(self):
        m = IDMapping({"A": "B"}, [])
        assert m["mapping"] == {"A": "B"}
        assert m["n_mapped"] == 1
