"""Tests for Chunk 1: Data Layer + Yeast Species."""

import math
import os
from pathlib import Path

import numpy as np
import pytest

from codonscope.core.codons import (
    SENSE_CODONS,
    all_possible_kmers,
    count_kmers,
    kmer_frequencies,
    sequence_to_codons,
)
from codonscope.core.sequences import SequenceDB
from codonscope.data.download import download

# Use a test-specific data dir or the default
DATA_DIR = os.environ.get(
    "CODONSCOPE_TEST_DATA_DIR",
    str(Path.home() / ".codonscope" / "data" / "species"),
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def yeast_data_dir():
    """Ensure yeast data is downloaded and return the species dir."""
    species_dir = Path(DATA_DIR) / "yeast"
    if not (species_dir / "cds_sequences.fa.gz").exists():
        download("yeast", data_dir=DATA_DIR)
    return species_dir


@pytest.fixture(scope="session")
def yeast_db(yeast_data_dir):
    """Return a SequenceDB for yeast."""
    return SequenceDB("yeast", data_dir=DATA_DIR)


# ── Download & File Existence ─────────────────────────────────────────────────

def test_download_yeast(yeast_data_dir):
    """Run download, verify all expected files exist."""
    expected_files = [
        "cds_sequences.fa.gz",
        "gene_id_map.tsv",
        "trna_copy_numbers.tsv",
        "wobble_rules.tsv",
        "background_mono.npz",
        "background_di.npz",
        "background_tri.npz",
        "gene_metadata.npz",
    ]
    for fname in expected_files:
        fpath = yeast_data_dir / fname
        assert fpath.exists(), f"Missing expected file: {fname}"
        assert fpath.stat().st_size > 0, f"File is empty: {fname}"


# ── ID Mapping ────────────────────────────────────────────────────────────────

def test_id_mapping_systematic(yeast_db):
    """YAL001C should resolve to YAL001C."""
    result = yeast_db.resolve_ids(["YAL001C"])
    assert "YAL001C" in result["mapping"]
    assert result["mapping"]["YAL001C"] == "YAL001C"


def test_id_mapping_common(yeast_db):
    """TFC3 should resolve to YAL001C."""
    result = yeast_db.resolve_ids(["TFC3"])
    assert "TFC3" in result["mapping"]
    assert result["mapping"]["TFC3"] == "YAL001C"


def test_id_mapping_case_insensitive(yeast_db):
    """tfc3 should resolve to YAL001C."""
    result = yeast_db.resolve_ids(["tfc3"])
    assert "tfc3" in result["mapping"]
    assert result["mapping"]["tfc3"] == "YAL001C"


def test_id_mapping_unmapped(yeast_db):
    """A clearly fake gene ID should be unmapped."""
    result = yeast_db.resolve_ids(["FAKEGENE123"])
    assert result["n_unmapped"] == 1
    assert "FAKEGENE123" in result["unmapped"]


# ── CDS Validation ────────────────────────────────────────────────────────────

def test_cds_validation(yeast_db):
    """All loaded CDSs should be divisible by 3 and contain only ACGT."""
    import re

    seqs = yeast_db.get_all_sequences()
    assert len(seqs) > 0
    for name, seq in seqs.items():
        assert len(seq) % 3 == 0, f"{name}: length {len(seq)} not divisible by 3"
        assert re.fullmatch(r"[ACGT]+", seq), f"{name}: contains non-ACGT characters"
        # Should not contain stop codons (already stripped)
        codons = [seq[i : i + 3] for i in range(0, len(seq), 3)]
        for codon in codons:
            assert codon not in {"TAA", "TAG", "TGA"}, (
                f"{name}: contains stop codon {codon}"
            )


def test_gene_count(yeast_db):
    """Yeast should have ~6,000-6,700 verified ORFs."""
    seqs = yeast_db.get_all_sequences()
    # SGD has ~6,000+ ORFs after filtering dubious and mito
    assert 4000 < len(seqs) < 7500, (
        f"Expected ~6000-6700 ORFs, got {len(seqs)}"
    )


# ── Kmer Counting ─────────────────────────────────────────────────────────────

def test_kmer_counting_basic():
    """Hand-computed example: ATGAAAGAA has codons ATG,AAA,GAA.
    Monocodons: ATG=1, AAA=1, GAA=1
    Dicodons: ATGAAA=1, AAAGAA=1
    """
    seq = "ATGAAAGAA"
    mono = count_kmers(seq, k=1)
    assert mono == {"ATG": 1, "AAA": 1, "GAA": 1}

    di = count_kmers(seq, k=2)
    assert di == {"ATGAAA": 1, "AAAGAA": 1}


def test_kmer_counting_tricodon():
    """ATGAAAGAACCC has codons ATG,AAA,GAA,CCC → 2 tricodons."""
    seq = "ATGAAAGAACCC"
    tri = count_kmers(seq, k=3)
    assert tri == {"ATGAAAGAA": 1, "AAAGAACCC": 1}


def test_kmer_frequencies_sum_to_one():
    """Frequencies from any sequence should sum to ~1.0."""
    seq = "ATGAAAGAACCCTTTATC"  # 6 codons
    for k in (1, 2, 3):
        freqs = kmer_frequencies(seq, k=k)
        total = sum(freqs.values())
        assert abs(total - 1.0) < 1e-9, (
            f"k={k}: frequencies sum to {total}, expected 1.0"
        )


def test_sequence_to_codons():
    """Basic codon splitting."""
    assert sequence_to_codons("ATGAAAGAA") == ["ATG", "AAA", "GAA"]


def test_sequence_to_codons_invalid_length():
    """Non-divisible-by-3 sequence should raise ValueError."""
    with pytest.raises(ValueError):
        sequence_to_codons("ATGAA")


def test_all_possible_kmers_mono():
    """61 sense codons for k=1."""
    kmers = all_possible_kmers(k=1, sense_only=True)
    assert len(kmers) == 61
    # No stop codons
    for k in kmers:
        assert k not in {"TAA", "TAG", "TGA"}


def test_all_possible_kmers_di():
    """61^2 = 3721 sense dicodons for k=2."""
    kmers = all_possible_kmers(k=2, sense_only=True)
    assert len(kmers) == 61 * 61


# ── Background Files ──────────────────────────────────────────────────────────

def test_background_files_exist(yeast_data_dir):
    """After download, background npz files should exist."""
    for k in ("mono", "di", "tri"):
        path = yeast_data_dir / f"background_{k}.npz"
        assert path.exists(), f"Missing background file: background_{k}.npz"


def test_background_mono_shape(yeast_data_dir):
    """Mono background mean vector should have 61 entries (sense codons)."""
    bg = np.load(yeast_data_dir / "background_mono.npz")
    assert bg["mean"].shape == (61,)
    assert bg["std"].shape == (61,)
    assert bg["per_gene"].shape[1] == 61
    # per_gene rows should match number of genes
    assert bg["per_gene"].shape[0] > 4000


def test_background_di_shape(yeast_data_dir):
    """Di background should have 3721 entries."""
    bg = np.load(yeast_data_dir / "background_di.npz")
    assert bg["mean"].shape == (61 * 61,)


def test_background_tri_has_mean_std(yeast_data_dir):
    """Tri background should have mean and std but not per_gene matrix."""
    bg = np.load(yeast_data_dir / "background_tri.npz")
    assert "mean" in bg
    assert "std" in bg
    assert bg["mean"].shape == (61 ** 3,)


def test_background_mono_frequencies_valid(yeast_data_dir):
    """Mono background mean frequencies should sum to ~1.0."""
    bg = np.load(yeast_data_dir / "background_mono.npz")
    total = bg["mean"].sum()
    assert abs(total - 1.0) < 0.01, f"Mono mean sums to {total}, expected ~1.0"


# ── Specific Gene Checks ─────────────────────────────────────────────────────

def test_yef3_exists(yeast_db):
    """YEF3 / YLR249W should be in the database."""
    result = yeast_db.resolve_ids(["YEF3", "YLR249W"])
    assert result["n_mapped"] == 2
    # Both should map to YLR249W
    assert result["mapping"]["YEF3"] == "YLR249W"
    assert result["mapping"]["YLR249W"] == "YLR249W"

    seqs = yeast_db.get_sequences(["YLR249W"])
    assert "YLR249W" in seqs
    # YEF3 is ~3,141 nt CDS (1,047 codons)
    assert len(seqs["YLR249W"]) > 2000


def test_gcn4_target_resolution(yeast_db):
    """A sample of known Gcn4 target genes should resolve successfully."""
    gcn4_targets = [
        "GCN4",     # YEL009C - the TF itself
        "ARG1",     # YOL058W
        "HIS4",     # YCL030C
        "ILV2",     # YMR108W
        "LEU4",     # YNL104C
        "TRP3",     # YKL211C
        "LYS1",     # YIR034C
        "ARO4",     # YBR249C
    ]
    result = yeast_db.resolve_ids(gcn4_targets)
    # Most should resolve
    assert result["n_mapped"] >= 6, (
        f"Expected at least 6 Gcn4 targets to resolve, got {result['n_mapped']}"
    )


def test_no_mitochondrial_orfs(yeast_db):
    """Mitochondrial ORFs (Q0xxx) should not be in the database."""
    seqs = yeast_db.get_all_sequences()
    mito_genes = [name for name in seqs if name.startswith("Q0")]
    assert len(mito_genes) == 0, (
        f"Found {len(mito_genes)} mitochondrial ORFs that should be excluded"
    )


# ── Wobble Rules ──────────────────────────────────────────────────────────────

def test_wobble_rules_completeness(yeast_data_dir):
    """Wobble rules should cover all 61 sense codons."""
    import pandas as pd

    df = pd.read_csv(yeast_data_dir / "wobble_rules.tsv", sep="\t")
    codons_in_rules = set(df["codon"])
    assert len(codons_in_rules) == 61, (
        f"Expected 61 codons in wobble rules, got {len(codons_in_rules)}"
    )
    # Check key entries from spec
    aga_row = df[df["codon"] == "AGA"].iloc[0]
    assert aga_row["decoding_anticodon"] == "UCU"
    assert aga_row["decoding_type"] == "watson_crick"

    gaa_row = df[df["codon"] == "GAA"].iloc[0]
    assert gaa_row["decoding_anticodon"] == "UUC"
    assert gaa_row["decoding_type"] == "watson_crick"
