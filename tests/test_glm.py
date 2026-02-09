"""Tests for binomial GLM (GC3-corrected) alternative to bootstrap."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from codonscope.core.statistics import (
    _compute_gc3,
    binomial_glm_zscores,
    compare_to_background,
    compute_geneset_frequencies,
)
from codonscope.core.sequences import SequenceDB

DATA_DIR = os.environ.get(
    "CODONSCOPE_TEST_DATA_DIR",
    str(Path.home() / ".codonscope" / "data" / "species"),
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def yeast_db():
    return SequenceDB("yeast", data_dir=DATA_DIR)


@pytest.fixture(scope="session")
def yeast_rp_genes():
    db = SequenceDB("yeast", data_dir=DATA_DIR)
    meta = db.get_gene_metadata()
    return meta[
        meta["common_name"].str.startswith("RPL", na=False)
        | meta["common_name"].str.startswith("RPS", na=False)
    ]["common_name"].tolist()


@pytest.fixture(scope="session")
def yeast_gcn4_genes():
    return [
        "ARG1", "ARG3", "ARG4", "ARG5,6", "ARG8",
        "HIS1", "HIS4", "HIS5",
        "ILV1", "ILV2", "ILV5",
        "LEU1", "LEU4",
        "LYS1", "LYS2", "LYS9",
        "TRP2", "TRP3", "TRP4", "TRP5",
        "SER1", "SER33",
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# Unit tests — _compute_gc3
# ═══════════════════════════════════════════════════════════════════════════════

class TestComputeGC3:
    def test_all_gc_third_position(self):
        """All G/C at third position → gc3 = 1.0."""
        # Three codons: AAG, AAC, AAG → third positions: G, C, G → all GC
        assert _compute_gc3("AAGAACAAG") == 1.0

    def test_no_gc_third_position(self):
        """All A/T at third position → gc3 = 0.0."""
        # Three codons: AAA, AAT, AAA → third positions: A, T, A → no GC
        assert _compute_gc3("AAAAATAAA") == 0.0

    def test_mixed_gc3(self):
        """Mixed GC3 content."""
        # Two codons: AAG (G), AAA (A) → gc3 = 0.5
        assert np.isclose(_compute_gc3("AAGAAA"), 0.5)

    def test_empty_sequence(self):
        """Empty sequence → 0.0."""
        assert _compute_gc3("") == 0.0

    def test_short_sequence(self):
        """Sequence < 3 bases → 0.0."""
        assert _compute_gc3("AA") == 0.0

    def test_valid_range(self):
        """GC3 should always be in [0, 1]."""
        seqs = ["ATGATGATG", "GCCGCCGCC", "AAAGGGCCC", "TTTTTTAAAGGG"]
        for seq in seqs:
            gc3 = _compute_gc3(seq)
            assert 0.0 <= gc3 <= 1.0, f"gc3={gc3} out of range for {seq}"


# ═══════════════════════════════════════════════════════════════════════════════
# Unit tests — binomial_glm_zscores
# ═══════════════════════════════════════════════════════════════════════════════

class TestBinomialGLM:
    def test_raises_for_k2(self, yeast_db):
        """Should raise ValueError for k=2."""
        with pytest.raises(ValueError, match="k=1"):
            binomial_glm_zscores({}, {}, k=2)

    def test_raises_for_k3(self, yeast_db):
        """Should raise ValueError for k=3."""
        with pytest.raises(ValueError, match="k=1"):
            binomial_glm_zscores({}, {}, k=3)

    def test_correct_columns(self, yeast_db, yeast_rp_genes):
        """Output DataFrame should have all expected columns."""
        id_mapping = yeast_db.resolve_ids(yeast_rp_genes)
        gene_seqs = yeast_db.get_sequences(list(id_mapping.values()))
        all_seqs = yeast_db.get_all_sequences()

        df = binomial_glm_zscores(gene_seqs, all_seqs, k=1)

        expected_cols = {
            "kmer", "observed_freq", "expected_freq",
            "z_score", "p_value", "adjusted_p", "cohens_d",
            "amino_acid", "gc3_beta", "gc3_pvalue",
        }
        assert expected_cols.issubset(set(df.columns)), (
            f"Missing columns: {expected_cols - set(df.columns)}"
        )

    def test_has_59_codons(self, yeast_db, yeast_rp_genes):
        """Should have results for 59 codons (61 - Met - Trp)."""
        id_mapping = yeast_db.resolve_ids(yeast_rp_genes)
        gene_seqs = yeast_db.get_sequences(list(id_mapping.values()))
        all_seqs = yeast_db.get_all_sequences()

        df = binomial_glm_zscores(gene_seqs, all_seqs, k=1)
        assert len(df) == 59, f"Expected 59 codons, got {len(df)}"


# ═══════════════════════════════════════════════════════════════════════════════
# Integration tests — GLM vs bootstrap correlation
# ═══════════════════════════════════════════════════════════════════════════════

class TestGLMvsBootstrap:
    def test_rp_zscores_correlated(self, yeast_db, yeast_rp_genes):
        """Yeast RP: GLM and bootstrap Z-scores should be correlated (r > 0.7)."""
        id_mapping = yeast_db.resolve_ids(yeast_rp_genes)
        gene_seqs = yeast_db.get_sequences(list(id_mapping.values()))
        all_seqs = yeast_db.get_all_sequences()

        glm_df = binomial_glm_zscores(gene_seqs, all_seqs, k=1)

        bg_path = yeast_db.species_dir / "background_mono.npz"
        boot_df = compare_to_background(gene_seqs, bg_path, k=1, seed=42)

        # Merge on kmer
        merged = glm_df[["kmer", "z_score"]].merge(
            boot_df[["kmer", "z_score"]],
            on="kmer", suffixes=("_glm", "_boot"),
        )

        from scipy.stats import pearsonr
        r, p = pearsonr(merged["z_score_glm"], merged["z_score_boot"])
        assert r > 0.7, (
            f"GLM vs bootstrap Z-score correlation r={r:.3f} should be > 0.7"
        )

    def test_gcn4_zscores_correlated(self, yeast_db, yeast_gcn4_genes):
        """Gcn4 targets: GLM and bootstrap Z-scores should be correlated."""
        id_mapping = yeast_db.resolve_ids(yeast_gcn4_genes)
        gene_seqs = yeast_db.get_sequences(list(id_mapping.values()))
        all_seqs = yeast_db.get_all_sequences()

        glm_df = binomial_glm_zscores(gene_seqs, all_seqs, k=1)

        bg_path = yeast_db.species_dir / "background_mono.npz"
        boot_df = compare_to_background(gene_seqs, bg_path, k=1, seed=42)

        merged = glm_df[["kmer", "z_score"]].merge(
            boot_df[["kmer", "z_score"]],
            on="kmer", suffixes=("_glm", "_boot"),
        )

        from scipy.stats import pearsonr
        r, p = pearsonr(merged["z_score_glm"], merged["z_score_boot"])
        assert r > 0.5, (
            f"Gcn4 GLM vs bootstrap correlation r={r:.3f} should be > 0.5"
        )

    def test_rp_significant_overlap(self, yeast_db, yeast_rp_genes):
        """RP genes: significant codons should substantially overlap between methods."""
        id_mapping = yeast_db.resolve_ids(yeast_rp_genes)
        gene_seqs = yeast_db.get_sequences(list(id_mapping.values()))
        all_seqs = yeast_db.get_all_sequences()

        glm_df = binomial_glm_zscores(gene_seqs, all_seqs, k=1)

        bg_path = yeast_db.species_dir / "background_mono.npz"
        boot_df = compare_to_background(gene_seqs, bg_path, k=1, seed=42)

        glm_sig = set(glm_df[glm_df["adjusted_p"] < 0.05]["kmer"])
        boot_sig = set(boot_df[boot_df["adjusted_p"] < 0.05]["kmer"])

        if len(boot_sig) > 0:
            overlap = len(glm_sig & boot_sig) / len(boot_sig)
            assert overlap > 0.3, (
                f"Overlap of significant codons {overlap:.1%} should be > 30%"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# Integration tests — run_composition with model="binomial"
# ═══════════════════════════════════════════════════════════════════════════════

class TestRunCompositionBinomial:
    def test_binomial_mode(self, yeast_rp_genes):
        """run_composition with model='binomial' should work."""
        from codonscope.modes.mode1_composition import run_composition

        result = run_composition(
            species="yeast",
            gene_ids=yeast_rp_genes,
            k=1,
            data_dir=DATA_DIR,
            model="binomial",
        )
        assert "results" in result
        assert result.get("model") == "binomial"
        assert len(result["results"]) == 59

    def test_binomial_rejects_k2(self, yeast_rp_genes):
        """model='binomial' with k=2 should raise ValueError."""
        from codonscope.modes.mode1_composition import run_composition

        with pytest.raises(ValueError, match="k=1"):
            run_composition(
                species="yeast",
                gene_ids=yeast_rp_genes,
                k=2,
                data_dir=DATA_DIR,
                model="binomial",
            )

    def test_binomial_has_gc3_columns(self, yeast_rp_genes):
        """Binomial results should include gc3_beta and gc3_pvalue."""
        from codonscope.modes.mode1_composition import run_composition

        result = run_composition(
            species="yeast",
            gene_ids=yeast_rp_genes,
            k=1,
            data_dir=DATA_DIR,
            model="binomial",
        )
        df = result["results"]
        assert "gc3_beta" in df.columns
        assert "gc3_pvalue" in df.columns
