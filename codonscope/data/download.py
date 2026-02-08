"""Data download and pre-computation infrastructure for CodonScope.

Currently supports yeast (S. cerevisiae), human, and mouse (M. musculus).
"""

import gzip
import io
import logging
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy import sparse
from tqdm import tqdm

from codonscope.core.codons import (
    SENSE_CODONS,
    all_possible_kmers,
    kmer_frequencies,
)

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path.home() / ".codonscope" / "data" / "species"

# ── GTEx expression data URL ──────────────────────────────────────────────────
GTEX_MEDIAN_TPM_URL = (
    "https://storage.googleapis.com/adult-gtex/bulk-gex/v8/rna-seq/"
    "GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_median_tpm.gct.gz"
)

# ── DepMap file listing API ──────────────────────────────────────────────────
DEPMAP_FILES_API = "https://depmap.org/portal/api/download/files"

# ── Cell line → GTEx tissue proxy mapping (fallback) ────────────────────────
CELL_LINE_TISSUE_PROXY: dict[str, str] = {
    "HEK293T": "Kidney - Cortex",
    "HEK293": "Kidney - Cortex",
    "HELA": "Uterus",
    "U2OS": "Cells - Cultured fibroblasts",
    "A549": "Lung",
    "MCF7": "Breast - Mammary Tissue",
    "K562": "Cells - EBV-transformed lymphocytes",
    "JURKAT": "Cells - EBV-transformed lymphocytes",
    "HEPG2": "Liver",
    "SHSY5Y": "Brain - Cortex",
    "RPE1": "Cells - Cultured fibroblasts",
}

# ── Yeast highly-expressed genes (rich media / YPD) ──────────────────────────
# Well-established expression values from multiple studies (Holstege 1998,
# Nagalakshmi 2008, Pelechano 2010).  Values are approximate relative mRNA
# abundance.  RP genes (RPL*/RPS*) are assigned categorically.
YEAST_HIGH_EXPRESSION: dict[str, float] = {
    # Glycolytic enzymes
    "TDH3": 5000, "TDH2": 3000, "TDH1": 1000,
    "ENO2": 3500, "ENO1": 1500,
    "PGK1": 3000, "FBA1": 2500,
    "ADH1": 3500, "PDC1": 3000,
    "TPI1": 2000, "GPM1": 2000, "PYK1": 2000, "CDC19": 2000,
    "PFK1": 500, "PFK2": 500, "HXK2": 500,
    # Translation factors
    "TEF1": 3000, "TEF2": 3000, "EFB1": 2000,
    "TIF1": 1500, "YEF3": 1200, "SUP35": 400,
    # Chaperones
    "SSA1": 1500, "SSA2": 1500, "SSB1": 1200, "SSB2": 1200,
    "HSP26": 500, "HSP82": 600, "HSC82": 600,
    # Cytoskeleton / structural
    "ACT1": 1500, "TUB1": 400, "TUB2": 400,
    # Other abundant
    "UBI4": 500, "GDH1": 800, "ALD6": 800,
    "OLE1": 400, "FAS1": 400, "FAS2": 400,
    "PDA1": 500, "PDB1": 500,
    "AHP1": 500, "TSA1": 600,
    "TRX2": 400, "GLK1": 300,
    # Histones
    "HTA1": 300, "HTA2": 300, "HTB1": 300, "HTB2": 300,
    "HHF1": 300, "HHF2": 300, "HHT1": 300, "HHT2": 300,
}

# ── SGD download URL ──────────────────────────────────────────────────────────
SGD_CDS_URL = (
    "https://downloads.yeastgenome.org/sequence/S288C_reference/"
    "orf_dna/orf_coding_all.fasta.gz"
)

# ── GtRNAdb download URL ─────────────────────────────────────────────────────
GTRNADB_YEAST_URL = (
    "http://gtrnadb.ucsc.edu/genomes/eukaryota/Scere3/sacCer3-mature-tRNAs.fa"
)

# ── HGNC complete gene set ────────────────────────────────────────────────────
HGNC_COMPLETE_SET_URL = (
    "https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/"
    "hgnc_complete_set.txt"
)

# ── Human download URLs ──────────────────────────────────────────────────────
# MANE Select summary (NCBI) — maps HGNC symbol, ENSG, ENST, RefSeq NM, Entrez ID
MANE_SUMMARY_URL = (
    "https://ftp.ncbi.nlm.nih.gov/refseq/MANE/MANE_human/current/"
    "MANE.GRCh38.v1.5.summary.txt.gz"
)

# Ensembl CDS FASTA — all human transcripts, we filter to MANE Select
ENSEMBL_HUMAN_CDS_URL = (
    "https://ftp.ensembl.org/pub/current_fasta/homo_sapiens/cds/"
    "Homo_sapiens.GRCh38.cds.all.fa.gz"
)

# GtRNAdb human tRNA
GTRNADB_HUMAN_URL = (
    "http://gtrnadb.ucsc.edu/genomes/eukaryota/Hsapi38/hg38-mature-tRNAs.fa"
)

# ── Mouse download URLs ────────────────────────────────────────────────────
# Ensembl CDS FASTA — all mouse transcripts, we keep longest per gene
ENSEMBL_MOUSE_CDS_URL = (
    "https://ftp.ensembl.org/pub/current_fasta/mus_musculus/cds/"
    "Mus_musculus.GRCm39.cds.all.fa.gz"
)

# GtRNAdb mouse tRNA
GTRNADB_MOUSE_URL = (
    "http://gtrnadb.ucsc.edu/genomes/eukaryota/Mmusc39/mm39-mature-tRNAs.fa"
)

# ── Fallback tRNA gene copy numbers (Phizicky & Hopper 2010, Chan et al. 2010)
YEAST_TRNA_FALLBACK: dict[str, tuple[str, int]] = {
    "AAC": ("Val", 2),
    "AAG": ("Leu", 1),
    "AAU": ("Ile", 2),
    "ACA": ("Cys", 4),
    "ACC": ("Gly", 16),
    "ACG": ("Arg", 6),
    "ACU": ("Ser", 4),
    "AGC": ("Ala", 11),
    "AGG": ("Pro", 10),
    "AGU": ("Thr", 11),
    "AUC": ("Asp", 15),
    "AUG": ("His", 7),
    "AUU": ("Asn", 10),
    "CAA": ("Leu", 10),
    "CAC": ("Val", 2),
    "CAG": ("Leu", 3),
    "CAU": ("Ile", 13),
    "CCA": ("Trp", 6),
    "CCC": ("Gly", 2),
    "CCG": ("Arg", 1),
    "CCU": ("Arg", 1),
    "CGC": ("Ala", 1),
    "CGG": ("Pro", 2),
    "CGU": ("Thr", 4),
    "CUC": ("Glu", 14),
    "CUG": ("Gln", 9),
    "CUU": ("Lys", 14),
    "GAA": ("Phe", 10),
    "GAC": ("Val", 14),
    "GAG": ("Leu", 7),
    "GAU": ("Ile", 2),
    "GCA": ("Cys", 0),
    "GCC": ("Gly", 3),
    "GCG": ("Arg", 6),
    "GCU": ("Ser", 11),
    "GGC": ("Ala", 5),
    "GGG": ("Pro", 0),
    "GGU": ("Thr", 9),
    "GUC": ("Asp", 4),
    "GUG": ("His", 1),
    "GUU": ("Asn", 2),
    "IAU": ("Ile", 1),
    "UCU": ("Arg", 11),
    "UGC": ("Ala", 0),
    "UGG": ("Pro", 10),
    "UGU": ("Thr", 0),
    "UUC": ("Glu", 2),
    "UUG": ("Gln", 1),
    "UUU": ("Lys", 7),
}

# ── Yeast wobble decoding rules ──────────────────────────────────────────────
# Standard eukaryotic wobble rules for S. cerevisiae.
# Each sense codon is mapped to its decoding tRNA anticodon and interaction type.
# Anticodon is written 5'→3'.  decoding_type is watson_crick or wobble.
# modification_notes documents known tRNA modifications relevant to decoding.

YEAST_WOBBLE_RULES: list[dict[str, str]] = [
    # ── Phe (UUU, UUC) ──
    {"codon": "TTT", "amino_acid": "Phe", "decoding_anticodon": "GAA", "decoding_type": "watson_crick", "modification_notes": ""},
    {"codon": "TTC", "amino_acid": "Phe", "decoding_anticodon": "GAA", "decoding_type": "wobble", "modification_notes": ""},
    # ── Leu (UUA, UUG, CUU, CUC, CUA, CUG) ──
    {"codon": "TTA", "amino_acid": "Leu", "decoding_anticodon": "UAG", "decoding_type": "watson_crick", "modification_notes": ""},
    {"codon": "TTG", "amino_acid": "Leu", "decoding_anticodon": "CAA", "decoding_type": "watson_crick", "modification_notes": "m5C(Trm4)"},
    {"codon": "CTT", "amino_acid": "Leu", "decoding_anticodon": "GAG", "decoding_type": "watson_crick", "modification_notes": ""},
    {"codon": "CTC", "amino_acid": "Leu", "decoding_anticodon": "GAG", "decoding_type": "wobble", "modification_notes": ""},
    {"codon": "CTA", "amino_acid": "Leu", "decoding_anticodon": "UAG", "decoding_type": "wobble", "modification_notes": "I34 possible"},
    {"codon": "CTG", "amino_acid": "Leu", "decoding_anticodon": "CAG", "decoding_type": "watson_crick", "modification_notes": ""},
    # ── Ile (AUU, AUC, AUA) ──
    {"codon": "ATT", "amino_acid": "Ile", "decoding_anticodon": "AAU", "decoding_type": "watson_crick", "modification_notes": ""},
    {"codon": "ATC", "amino_acid": "Ile", "decoding_anticodon": "AAU", "decoding_type": "wobble", "modification_notes": ""},
    {"codon": "ATA", "amino_acid": "Ile", "decoding_anticodon": "IAU", "decoding_type": "watson_crick", "modification_notes": "I34(Tad1)"},
    # ── Met (AUG) ──
    {"codon": "ATG", "amino_acid": "Met", "decoding_anticodon": "CAU", "decoding_type": "watson_crick", "modification_notes": ""},
    # ── Val (GUU, GUC, GUA, GUG) ──
    {"codon": "GTT", "amino_acid": "Val", "decoding_anticodon": "AAC", "decoding_type": "watson_crick", "modification_notes": ""},
    {"codon": "GTC", "amino_acid": "Val", "decoding_anticodon": "AAC", "decoding_type": "wobble", "modification_notes": ""},
    {"codon": "GTA", "amino_acid": "Val", "decoding_anticodon": "CAC", "decoding_type": "watson_crick", "modification_notes": ""},
    {"codon": "GTG", "amino_acid": "Val", "decoding_anticodon": "CAC", "decoding_type": "wobble", "modification_notes": ""},
    # ── Ser (UCU, UCC, UCA, UCG, AGU, AGC) ──
    {"codon": "TCT", "amino_acid": "Ser", "decoding_anticodon": "AGA", "decoding_type": "watson_crick", "modification_notes": ""},
    {"codon": "TCC", "amino_acid": "Ser", "decoding_anticodon": "AGA", "decoding_type": "wobble", "modification_notes": ""},
    {"codon": "TCA", "amino_acid": "Ser", "decoding_anticodon": "UGA", "decoding_type": "watson_crick", "modification_notes": ""},
    {"codon": "TCG", "amino_acid": "Ser", "decoding_anticodon": "CGA", "decoding_type": "watson_crick", "modification_notes": ""},
    {"codon": "AGT", "amino_acid": "Ser", "decoding_anticodon": "GCU", "decoding_type": "watson_crick", "modification_notes": ""},
    {"codon": "AGC", "amino_acid": "Ser", "decoding_anticodon": "GCU", "decoding_type": "wobble", "modification_notes": ""},
    # ── Pro (CCU, CCC, CCA, CCG) ──
    {"codon": "CCT", "amino_acid": "Pro", "decoding_anticodon": "AGG", "decoding_type": "watson_crick", "modification_notes": ""},
    {"codon": "CCC", "amino_acid": "Pro", "decoding_anticodon": "AGG", "decoding_type": "wobble", "modification_notes": ""},
    {"codon": "CCA", "amino_acid": "Pro", "decoding_anticodon": "UGG", "decoding_type": "watson_crick", "modification_notes": ""},
    {"codon": "CCG", "amino_acid": "Pro", "decoding_anticodon": "CGG", "decoding_type": "watson_crick", "modification_notes": ""},
    # ── Thr (ACU, ACC, ACA, ACG) ──
    {"codon": "ACT", "amino_acid": "Thr", "decoding_anticodon": "AGU", "decoding_type": "watson_crick", "modification_notes": ""},
    {"codon": "ACC", "amino_acid": "Thr", "decoding_anticodon": "AGU", "decoding_type": "wobble", "modification_notes": ""},
    {"codon": "ACA", "amino_acid": "Thr", "decoding_anticodon": "UGU", "decoding_type": "wobble", "modification_notes": ""},
    {"codon": "ACG", "amino_acid": "Thr", "decoding_anticodon": "CGU", "decoding_type": "watson_crick", "modification_notes": ""},
    # ── Ala (GCU, GCC, GCA, GCG) ──
    {"codon": "GCT", "amino_acid": "Ala", "decoding_anticodon": "AGC", "decoding_type": "watson_crick", "modification_notes": ""},
    {"codon": "GCC", "amino_acid": "Ala", "decoding_anticodon": "AGC", "decoding_type": "wobble", "modification_notes": ""},
    {"codon": "GCA", "amino_acid": "Ala", "decoding_anticodon": "UGC", "decoding_type": "wobble", "modification_notes": ""},
    {"codon": "GCG", "amino_acid": "Ala", "decoding_anticodon": "CGC", "decoding_type": "watson_crick", "modification_notes": ""},
    # ── Tyr (UAU, UAC) ──
    {"codon": "TAT", "amino_acid": "Tyr", "decoding_anticodon": "GUA", "decoding_type": "watson_crick", "modification_notes": ""},
    {"codon": "TAC", "amino_acid": "Tyr", "decoding_anticodon": "GUA", "decoding_type": "wobble", "modification_notes": ""},
    # ── His (CAU, CAC) ──
    {"codon": "CAT", "amino_acid": "His", "decoding_anticodon": "GUG", "decoding_type": "watson_crick", "modification_notes": ""},
    {"codon": "CAC", "amino_acid": "His", "decoding_anticodon": "GUG", "decoding_type": "wobble", "modification_notes": ""},
    # ── Gln (CAA, CAG) ──
    {"codon": "CAA", "amino_acid": "Gln", "decoding_anticodon": "UUG", "decoding_type": "watson_crick", "modification_notes": ""},
    {"codon": "CAG", "amino_acid": "Gln", "decoding_anticodon": "CUG", "decoding_type": "watson_crick", "modification_notes": ""},
    # ── Asn (AAU, AAC) ──
    {"codon": "AAT", "amino_acid": "Asn", "decoding_anticodon": "GUU", "decoding_type": "watson_crick", "modification_notes": ""},
    {"codon": "AAC", "amino_acid": "Asn", "decoding_anticodon": "GUU", "decoding_type": "wobble", "modification_notes": ""},
    # ── Lys (AAA, AAG) ──
    {"codon": "AAA", "amino_acid": "Lys", "decoding_anticodon": "UUU", "decoding_type": "watson_crick", "modification_notes": "mcm5s2U"},
    {"codon": "AAG", "amino_acid": "Lys", "decoding_anticodon": "CUU", "decoding_type": "watson_crick", "modification_notes": ""},
    # ── Asp (GAU, GAC) ──
    {"codon": "GAT", "amino_acid": "Asp", "decoding_anticodon": "GUC", "decoding_type": "watson_crick", "modification_notes": ""},
    {"codon": "GAC", "amino_acid": "Asp", "decoding_anticodon": "GUC", "decoding_type": "wobble", "modification_notes": ""},
    # ── Glu (GAA, GAG) ──
    {"codon": "GAA", "amino_acid": "Glu", "decoding_anticodon": "UUC", "decoding_type": "watson_crick", "modification_notes": "mcm5s2U(Trm9)"},
    {"codon": "GAG", "amino_acid": "Glu", "decoding_anticodon": "CUC", "decoding_type": "watson_crick", "modification_notes": ""},
    # ── Cys (UGU, UGC) ──
    {"codon": "TGT", "amino_acid": "Cys", "decoding_anticodon": "GCA", "decoding_type": "watson_crick", "modification_notes": ""},
    {"codon": "TGC", "amino_acid": "Cys", "decoding_anticodon": "GCA", "decoding_type": "wobble", "modification_notes": ""},
    # ── Trp (UGG) ──
    {"codon": "TGG", "amino_acid": "Trp", "decoding_anticodon": "CCA", "decoding_type": "watson_crick", "modification_notes": ""},
    # ── Arg (CGU, CGC, CGA, CGG, AGA, AGG) ──
    {"codon": "CGT", "amino_acid": "Arg", "decoding_anticodon": "ACG", "decoding_type": "watson_crick", "modification_notes": ""},
    {"codon": "CGC", "amino_acid": "Arg", "decoding_anticodon": "ACG", "decoding_type": "wobble", "modification_notes": ""},
    {"codon": "CGA", "amino_acid": "Arg", "decoding_anticodon": "ICG", "decoding_type": "watson_crick", "modification_notes": "I34"},
    {"codon": "CGG", "amino_acid": "Arg", "decoding_anticodon": "CCG", "decoding_type": "watson_crick", "modification_notes": ""},
    {"codon": "AGA", "amino_acid": "Arg", "decoding_anticodon": "UCU", "decoding_type": "watson_crick", "modification_notes": "mcm5U/mcm5s2U(Trm9)"},
    {"codon": "AGG", "amino_acid": "Arg", "decoding_anticodon": "CCU", "decoding_type": "watson_crick", "modification_notes": ""},
    # ── Gly (GGU, GGC, GGA, GGG) ──
    {"codon": "GGT", "amino_acid": "Gly", "decoding_anticodon": "ACC", "decoding_type": "watson_crick", "modification_notes": ""},
    {"codon": "GGC", "amino_acid": "Gly", "decoding_anticodon": "ACC", "decoding_type": "wobble", "modification_notes": ""},
    {"codon": "GGA", "amino_acid": "Gly", "decoding_anticodon": "UCC", "decoding_type": "wobble", "modification_notes": ""},
    {"codon": "GGG", "amino_acid": "Gly", "decoding_anticodon": "CCC", "decoding_type": "watson_crick", "modification_notes": ""},
]


# ── Human tRNA gene copy numbers fallback (GtRNAdb hg38, ~430 genes) ─────────
# Aggregated by anticodon. Source: Chan & Lowe 2016 / GtRNAdb hg38 build.
HUMAN_TRNA_FALLBACK: dict[str, tuple[str, int]] = {
    "AAC": ("Val", 5),
    "AAG": ("Leu", 5),
    "AAU": ("Ile", 5),
    "ACA": ("Cys", 30),
    "ACC": ("Gly", 10),
    "ACG": ("Arg", 7),
    "ACU": ("Ser", 5),
    "AGC": ("Ala", 30),
    "AGG": ("Pro", 11),
    "AGU": ("Thr", 8),
    "AUC": ("Asp", 19),
    "AUG": ("His", 11),
    "AUU": ("Asn", 32),
    "CAA": ("Leu", 7),
    "CAC": ("Val", 11),
    "CAG": ("Leu", 10),
    "CAU": ("Ile", 5),  # also initiator Met(CAU)
    "CCA": ("Trp", 8),
    "CCC": ("Gly", 10),
    "CCG": ("Arg", 4),
    "CCU": ("Arg", 5),
    "CGC": ("Ala", 5),
    "CGG": ("Pro", 5),
    "CGU": ("Thr", 6),
    "CUC": ("Glu", 12),
    "CUG": ("Gln", 20),
    "CUU": ("Lys", 16),
    "GAA": ("Phe", 8),
    "GAC": ("Val", 10),
    "GAG": ("Leu", 4),
    "GAU": ("Ile", 6),
    "GCA": ("Cys", 0),
    "GCC": ("Gly", 12),
    "GCU": ("Ser", 7),
    "GGC": ("Ala", 5),
    "GGG": ("Pro", 0),
    "GGU": ("Thr", 7),
    "GUC": ("Asp", 7),
    "GUG": ("His", 3),
    "GUU": ("Asn", 6),
    "UCU": ("Arg", 6),
    "UGC": ("Ala", 10),
    "UGG": ("Pro", 6),
    "UGU": ("Thr", 5),
    "UUC": ("Glu", 7),
    "UUG": ("Gln", 5),
    "UUU": ("Lys", 14),
}

# ── Mouse tRNA gene copy numbers fallback (GtRNAdb mm39, ~430 genes) ──────
# Aggregated by anticodon. Source: Chan & Lowe 2016 / GtRNAdb mm39 build.
# Mouse tRNA copy numbers are very similar to human (both mammalian).
MOUSE_TRNA_FALLBACK: dict[str, tuple[str, int]] = {
    "AAC": ("Val", 5),
    "AAG": ("Leu", 5),
    "AAU": ("Ile", 5),
    "ACA": ("Cys", 28),
    "ACC": ("Gly", 9),
    "ACG": ("Arg", 6),
    "ACU": ("Ser", 5),
    "AGC": ("Ala", 28),
    "AGG": ("Pro", 10),
    "AGU": ("Thr", 7),
    "AUC": ("Asp", 18),
    "AUG": ("His", 10),
    "AUU": ("Asn", 30),
    "CAA": ("Leu", 7),
    "CAC": ("Val", 10),
    "CAG": ("Leu", 9),
    "CAU": ("Ile", 5),
    "CCA": ("Trp", 7),
    "CCC": ("Gly", 9),
    "CCG": ("Arg", 4),
    "CCU": ("Arg", 5),
    "CGC": ("Ala", 5),
    "CGG": ("Pro", 4),
    "CGU": ("Thr", 5),
    "CUC": ("Glu", 11),
    "CUG": ("Gln", 18),
    "CUU": ("Lys", 15),
    "GAA": ("Phe", 7),
    "GAC": ("Val", 9),
    "GAG": ("Leu", 4),
    "GAU": ("Ile", 5),
    "GCA": ("Cys", 0),
    "GCC": ("Gly", 11),
    "GCU": ("Ser", 6),
    "GGC": ("Ala", 5),
    "GGG": ("Pro", 0),
    "GGU": ("Thr", 6),
    "GUC": ("Asp", 6),
    "GUG": ("His", 3),
    "GUU": ("Asn", 5),
    "UCU": ("Arg", 5),
    "UGC": ("Ala", 9),
    "UGG": ("Pro", 5),
    "UGU": ("Thr", 5),
    "UUC": ("Glu", 6),
    "UUG": ("Gln", 4),
    "UUU": ("Lys", 13),
}

# ── Mouse highly-expressed genes (general) ──────────────────────────────
# Approximate TPM values based on mouse RNA-seq literature.
# Mouse gene names use Title Case (Rpl*/Rps* for RP genes).
MOUSE_HIGH_EXPRESSION: dict[str, float] = {
    # Glycolytic enzymes
    "Gapdh": 3000, "Eno1": 2000, "Pgk1": 2000,
    "Aldoa": 2000, "Tpi1": 1500, "Pkm": 1500,
    "Pfkl": 500, "Hk1": 500, "Hk2": 500,
    "Gpi1": 500, "Pgam1": 1000, "Ldha": 1500,
    # Translation factors
    "Eef1a1": 3000, "Eef2": 2000,
    "Eif4a1": 1000, "Eif2s1": 500,
    # Chaperones
    "Hspa8": 1500, "Hspa5": 1000, "Hsp90aa1": 1200,
    "Hsp90ab1": 1200, "Hspd1": 800,
    # Cytoskeleton / structural
    "Actb": 5000, "Actg1": 2000,
    "Tuba1a": 500, "Tubb4b": 500, "Tubb5": 500,
    # Histones
    "H3c1": 300, "H4c1": 300, "H2ac1": 300, "H2bc1": 300,
    # Other abundant
    "Ubb": 500, "Ubc": 500,
    "Fasn": 400, "Hmgcr": 300,
    "Ppia": 1500, "Ppib": 800,
    "Rplp0": 3000, "Rplp1": 3000, "Rplp2": 3000,
}

# ── Human wobble decoding rules (standard eukaryotic) ────────────────────────
# Very similar to yeast with human-specific modification enzyme names.
# ALKBH8 is the human homolog of yeast Trm9 (mcm5U/mcm5s2U modifications).
# NSUN2 is the human homolog of yeast Trm4 (m5C modification).
HUMAN_WOBBLE_RULES: list[dict[str, str]] = [
    # ── Phe ──
    {"codon": "TTT", "amino_acid": "Phe", "decoding_anticodon": "GAA", "decoding_type": "watson_crick", "modification_notes": ""},
    {"codon": "TTC", "amino_acid": "Phe", "decoding_anticodon": "GAA", "decoding_type": "wobble", "modification_notes": ""},
    # ── Leu ──
    {"codon": "TTA", "amino_acid": "Leu", "decoding_anticodon": "UAG", "decoding_type": "watson_crick", "modification_notes": ""},
    {"codon": "TTG", "amino_acid": "Leu", "decoding_anticodon": "CAA", "decoding_type": "watson_crick", "modification_notes": "m5C(NSUN2)"},
    {"codon": "CTT", "amino_acid": "Leu", "decoding_anticodon": "AAG", "decoding_type": "watson_crick", "modification_notes": "I34(ADAT1)"},
    {"codon": "CTC", "amino_acid": "Leu", "decoding_anticodon": "AAG", "decoding_type": "wobble", "modification_notes": "I34(ADAT1)"},
    {"codon": "CTA", "amino_acid": "Leu", "decoding_anticodon": "AAG", "decoding_type": "wobble", "modification_notes": "I34(ADAT1)"},
    {"codon": "CTG", "amino_acid": "Leu", "decoding_anticodon": "CAG", "decoding_type": "watson_crick", "modification_notes": ""},
    # ── Ile ──
    {"codon": "ATT", "amino_acid": "Ile", "decoding_anticodon": "AAU", "decoding_type": "watson_crick", "modification_notes": "I34(ADAT1)"},
    {"codon": "ATC", "amino_acid": "Ile", "decoding_anticodon": "AAU", "decoding_type": "wobble", "modification_notes": "I34(ADAT1)"},
    {"codon": "ATA", "amino_acid": "Ile", "decoding_anticodon": "AAU", "decoding_type": "wobble", "modification_notes": "I34(ADAT1)"},
    # ── Met ──
    {"codon": "ATG", "amino_acid": "Met", "decoding_anticodon": "CAU", "decoding_type": "watson_crick", "modification_notes": ""},
    # ── Val ──
    {"codon": "GTT", "amino_acid": "Val", "decoding_anticodon": "AAC", "decoding_type": "watson_crick", "modification_notes": "I34(ADAT1)"},
    {"codon": "GTC", "amino_acid": "Val", "decoding_anticodon": "AAC", "decoding_type": "wobble", "modification_notes": "I34(ADAT1)"},
    {"codon": "GTA", "amino_acid": "Val", "decoding_anticodon": "AAC", "decoding_type": "wobble", "modification_notes": "I34(ADAT1)"},
    {"codon": "GTG", "amino_acid": "Val", "decoding_anticodon": "CAC", "decoding_type": "watson_crick", "modification_notes": ""},
    # ── Ser (6-fold) ──
    {"codon": "TCT", "amino_acid": "Ser", "decoding_anticodon": "AGC", "decoding_type": "watson_crick", "modification_notes": "I34(ADAT1)"},
    {"codon": "TCC", "amino_acid": "Ser", "decoding_anticodon": "AGC", "decoding_type": "wobble", "modification_notes": "I34(ADAT1)"},
    {"codon": "TCA", "amino_acid": "Ser", "decoding_anticodon": "AGC", "decoding_type": "wobble", "modification_notes": "I34(ADAT1)"},
    {"codon": "TCG", "amino_acid": "Ser", "decoding_anticodon": "CGA", "decoding_type": "watson_crick", "modification_notes": ""},
    {"codon": "AGT", "amino_acid": "Ser", "decoding_anticodon": "GCU", "decoding_type": "watson_crick", "modification_notes": ""},
    {"codon": "AGC", "amino_acid": "Ser", "decoding_anticodon": "GCU", "decoding_type": "wobble", "modification_notes": ""},
    # ── Pro ──
    {"codon": "CCT", "amino_acid": "Pro", "decoding_anticodon": "AGG", "decoding_type": "watson_crick", "modification_notes": "I34(ADAT1)"},
    {"codon": "CCC", "amino_acid": "Pro", "decoding_anticodon": "AGG", "decoding_type": "wobble", "modification_notes": "I34(ADAT1)"},
    {"codon": "CCA", "amino_acid": "Pro", "decoding_anticodon": "AGG", "decoding_type": "wobble", "modification_notes": "I34(ADAT1)"},
    {"codon": "CCG", "amino_acid": "Pro", "decoding_anticodon": "CGG", "decoding_type": "watson_crick", "modification_notes": ""},
    # ── Thr ──
    {"codon": "ACT", "amino_acid": "Thr", "decoding_anticodon": "AGU", "decoding_type": "watson_crick", "modification_notes": "I34(ADAT1)"},
    {"codon": "ACC", "amino_acid": "Thr", "decoding_anticodon": "AGU", "decoding_type": "wobble", "modification_notes": "I34(ADAT1)"},
    {"codon": "ACA", "amino_acid": "Thr", "decoding_anticodon": "AGU", "decoding_type": "wobble", "modification_notes": "I34(ADAT1)"},
    {"codon": "ACG", "amino_acid": "Thr", "decoding_anticodon": "CGU", "decoding_type": "watson_crick", "modification_notes": ""},
    # ── Ala ──
    {"codon": "GCT", "amino_acid": "Ala", "decoding_anticodon": "AGC", "decoding_type": "watson_crick", "modification_notes": "I34(ADAT1)"},
    {"codon": "GCC", "amino_acid": "Ala", "decoding_anticodon": "AGC", "decoding_type": "wobble", "modification_notes": "I34(ADAT1)"},
    {"codon": "GCA", "amino_acid": "Ala", "decoding_anticodon": "AGC", "decoding_type": "wobble", "modification_notes": "I34(ADAT1)"},
    {"codon": "GCG", "amino_acid": "Ala", "decoding_anticodon": "CGC", "decoding_type": "watson_crick", "modification_notes": ""},
    # ── Tyr ──
    {"codon": "TAT", "amino_acid": "Tyr", "decoding_anticodon": "GUA", "decoding_type": "watson_crick", "modification_notes": ""},
    {"codon": "TAC", "amino_acid": "Tyr", "decoding_anticodon": "GUA", "decoding_type": "wobble", "modification_notes": ""},
    # ── His ──
    {"codon": "CAT", "amino_acid": "His", "decoding_anticodon": "GUG", "decoding_type": "watson_crick", "modification_notes": ""},
    {"codon": "CAC", "amino_acid": "His", "decoding_anticodon": "GUG", "decoding_type": "wobble", "modification_notes": ""},
    # ── Gln ──
    {"codon": "CAA", "amino_acid": "Gln", "decoding_anticodon": "UUG", "decoding_type": "watson_crick", "modification_notes": "mcm5s2U(ALKBH8)"},
    {"codon": "CAG", "amino_acid": "Gln", "decoding_anticodon": "CUG", "decoding_type": "watson_crick", "modification_notes": ""},
    # ── Asn ──
    {"codon": "AAT", "amino_acid": "Asn", "decoding_anticodon": "GUU", "decoding_type": "watson_crick", "modification_notes": ""},
    {"codon": "AAC", "amino_acid": "Asn", "decoding_anticodon": "GUU", "decoding_type": "wobble", "modification_notes": ""},
    # ── Lys ──
    {"codon": "AAA", "amino_acid": "Lys", "decoding_anticodon": "UUU", "decoding_type": "watson_crick", "modification_notes": "mcm5s2U(ALKBH8)"},
    {"codon": "AAG", "amino_acid": "Lys", "decoding_anticodon": "CUU", "decoding_type": "watson_crick", "modification_notes": ""},
    # ── Asp ──
    {"codon": "GAT", "amino_acid": "Asp", "decoding_anticodon": "GUC", "decoding_type": "watson_crick", "modification_notes": ""},
    {"codon": "GAC", "amino_acid": "Asp", "decoding_anticodon": "GUC", "decoding_type": "wobble", "modification_notes": ""},
    # ── Glu ──
    {"codon": "GAA", "amino_acid": "Glu", "decoding_anticodon": "UUC", "decoding_type": "watson_crick", "modification_notes": "mcm5s2U(ALKBH8)"},
    {"codon": "GAG", "amino_acid": "Glu", "decoding_anticodon": "CUC", "decoding_type": "watson_crick", "modification_notes": ""},
    # ── Cys ──
    {"codon": "TGT", "amino_acid": "Cys", "decoding_anticodon": "GCA", "decoding_type": "watson_crick", "modification_notes": ""},
    {"codon": "TGC", "amino_acid": "Cys", "decoding_anticodon": "GCA", "decoding_type": "wobble", "modification_notes": ""},
    # ── Trp ──
    {"codon": "TGG", "amino_acid": "Trp", "decoding_anticodon": "CCA", "decoding_type": "watson_crick", "modification_notes": ""},
    # ── Arg (6-fold) ──
    {"codon": "CGT", "amino_acid": "Arg", "decoding_anticodon": "ACG", "decoding_type": "watson_crick", "modification_notes": "I34(ADAT1)"},
    {"codon": "CGC", "amino_acid": "Arg", "decoding_anticodon": "ACG", "decoding_type": "wobble", "modification_notes": "I34(ADAT1)"},
    {"codon": "CGA", "amino_acid": "Arg", "decoding_anticodon": "ACG", "decoding_type": "wobble", "modification_notes": "I34(ADAT1)"},
    {"codon": "CGG", "amino_acid": "Arg", "decoding_anticodon": "CCG", "decoding_type": "watson_crick", "modification_notes": ""},
    {"codon": "AGA", "amino_acid": "Arg", "decoding_anticodon": "UCU", "decoding_type": "watson_crick", "modification_notes": "mcm5U(ALKBH8)"},
    {"codon": "AGG", "amino_acid": "Arg", "decoding_anticodon": "CCU", "decoding_type": "watson_crick", "modification_notes": ""},
    # ── Gly ──
    {"codon": "GGT", "amino_acid": "Gly", "decoding_anticodon": "GCC", "decoding_type": "watson_crick", "modification_notes": ""},
    {"codon": "GGC", "amino_acid": "Gly", "decoding_anticodon": "GCC", "decoding_type": "wobble", "modification_notes": ""},
    {"codon": "GGA", "amino_acid": "Gly", "decoding_anticodon": "UCC", "decoding_type": "watson_crick", "modification_notes": ""},
    {"codon": "GGG", "amino_acid": "Gly", "decoding_anticodon": "CCC", "decoding_type": "watson_crick", "modification_notes": ""},
]


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════

def download(species: str, data_dir: str | Path | None = None) -> Path:
    """Download and pre-compute all reference data for a species.

    Args:
        species: Species name (currently only "yeast" supported).
        data_dir: Override default data directory (~/.codonscope/data/species/).

    Returns:
        Path to the species data directory.
    """
    species = species.lower()
    dispatchers = {
        "yeast": _download_yeast,
        "human": _download_human,
        "mouse": _download_mouse,
    }
    if species not in dispatchers:
        raise ValueError(
            f"Unsupported species: {species!r}. "
            f"Supported: {', '.join(dispatchers)}"
        )

    base = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    species_dir = base / species
    species_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s data to %s", species, species_dir)

    dispatchers[species](species_dir)
    return species_dir


# ═══════════════════════════════════════════════════════════════════════════════
# Yeast-specific download
# ═══════════════════════════════════════════════════════════════════════════════

def _download_yeast(species_dir: Path) -> None:
    """Download and process all yeast reference data."""
    # 1. CDS sequences from SGD
    sequences, gene_map = _download_yeast_cds(species_dir)

    # 2. tRNA gene copy numbers
    _download_yeast_trna(species_dir)

    # 3. Wobble decoding rules (curated, not downloaded)
    _save_wobble_rules(species_dir)

    # 4. Pre-compute backgrounds
    _compute_backgrounds(species_dir, sequences)

    # 5. Create expression data (hardcoded rich-media estimates)
    _create_yeast_expression(species_dir)

    logger.info("Yeast download complete. Files in %s", species_dir)


def _download_yeast_cds(species_dir: Path) -> tuple[dict[str, str], pd.DataFrame]:
    """Download yeast CDS sequences from SGD and build gene ID map.

    Returns:
        (sequences dict {systematic_name: cds}, gene_map DataFrame)
    """
    logger.info("Downloading yeast CDS from SGD...")
    resp = requests.get(SGD_CDS_URL, timeout=120)
    resp.raise_for_status()

    raw_fasta = gzip.decompress(resp.content).decode("ascii")
    sequences, records = _parse_sgd_fasta(raw_fasta)

    # Save cleaned FASTA
    cds_path = species_dir / "cds_sequences.fa.gz"
    with gzip.open(cds_path, "wt") as fh:
        for sysname, seq in sorted(sequences.items()):
            fh.write(f">{sysname}\n{seq}\n")
    logger.info("Saved %d validated CDS to %s", len(sequences), cds_path)

    # Save gene ID map
    gene_map = pd.DataFrame(records)
    gene_map.to_csv(species_dir / "gene_id_map.tsv", sep="\t", index=False)
    logger.info("Saved gene ID map with %d entries", len(gene_map))

    return sequences, gene_map


def _parse_sgd_fasta(fasta_text: str) -> tuple[dict[str, str], list[dict]]:
    """Parse SGD orf_coding_all FASTA.

    Returns:
        (sequences dict, list of record dicts for gene_id_map)
    """
    sequences: dict[str, str] = {}
    records: list[dict] = []
    stop_codons = {"TAA", "TAG", "TGA"}

    current_name = None
    current_seq_parts: list[str] = []
    current_header = ""

    def _process_entry(header: str, seq: str) -> None:
        parts = header.split(",")
        name_part = parts[0].strip()
        tokens = name_part.split()
        if len(tokens) < 2:
            return

        systematic_name = tokens[0]
        # Second token is common name or SGDID
        common_name = ""
        if len(tokens) >= 2 and not tokens[1].startswith("SGDID"):
            common_name = tokens[1]

        # Skip mitochondrial ORFs (systematic names starting with Q0)
        if systematic_name.startswith("Q0"):
            logger.debug("Skipping mitochondrial ORF: %s", systematic_name)
            return

        # Determine verification status from header
        status = ""
        header_lower = header.lower()
        if "verified" in header_lower:
            status = "Verified"
        elif "uncharacterized" in header_lower:
            status = "Uncharacterized"
        elif "dubious" in header_lower:
            status = "Dubious"

        seq_upper = seq.upper()

        # Validate: only ACGT
        if not re.fullmatch(r"[ACGT]+", seq_upper):
            logger.debug("Skipping %s: non-ACGT characters", systematic_name)
            return

        # Validate: divisible by 3
        if len(seq_upper) % 3 != 0:
            logger.debug(
                "Skipping %s: length %d not divisible by 3",
                systematic_name, len(seq_upper),
            )
            return

        # Validate: starts with ATG
        if not seq_upper.startswith("ATG"):
            logger.debug("Skipping %s: does not start with ATG", systematic_name)
            return

        # Validate: ends with stop codon
        last_codon = seq_upper[-3:]
        if last_codon not in stop_codons:
            logger.debug("Skipping %s: does not end with stop codon", systematic_name)
            return

        # Strip stop codon
        cds = seq_upper[:-3]

        # Check for internal stop codons
        has_internal_stop = False
        for i in range(0, len(cds), 3):
            if cds[i : i + 3] in stop_codons:
                has_internal_stop = True
                break
        if has_internal_stop:
            logger.debug("Skipping %s: internal stop codon", systematic_name)
            return

        gc_count = cds.count("G") + cds.count("C")
        gc_content = gc_count / len(cds) if len(cds) > 0 else 0.0

        sequences[systematic_name] = cds
        records.append({
            "systematic_name": systematic_name,
            "common_name": common_name,
            "cds_length": len(cds),
            "gc_content": round(gc_content, 4),
            "status": status,
        })

    for line in fasta_text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            # Process previous entry
            if current_name is not None:
                _process_entry(current_header, "".join(current_seq_parts))
            current_header = line[1:]
            current_name = current_header.split()[0]
            current_seq_parts = []
        else:
            current_seq_parts.append(line)

    # Process last entry
    if current_name is not None:
        _process_entry(current_header, "".join(current_seq_parts))

    return sequences, records


def _download_yeast_trna(species_dir: Path) -> None:
    """Download tRNA gene copy numbers from GtRNAdb, with hardcoded fallback."""
    trna_path = species_dir / "trna_copy_numbers.tsv"

    try:
        logger.info("Downloading yeast tRNA data from GtRNAdb...")
        resp = requests.get(GTRNADB_YEAST_URL, timeout=60)
        resp.raise_for_status()
        trna_counts = _parse_gtrnadb_fasta(resp.text)
        if len(trna_counts) < 20:
            raise ValueError(
                f"Only parsed {len(trna_counts)} tRNA anticodons, expected ~40+"
            )
        logger.info("Parsed %d tRNA anticodons from GtRNAdb", len(trna_counts))
    except Exception as exc:
        logger.warning(
            "GtRNAdb download failed (%s), using hardcoded fallback", exc
        )
        trna_counts = {
            anticodon: {"amino_acid": aa, "gene_count": count}
            for anticodon, (aa, count) in YEAST_TRNA_FALLBACK.items()
        }

    rows = []
    for anticodon, info in sorted(trna_counts.items()):
        rows.append({
            "anticodon": anticodon,
            "amino_acid": info["amino_acid"],
            "gene_count": info["gene_count"],
        })

    df = pd.DataFrame(rows)
    df.to_csv(trna_path, sep="\t", index=False)
    logger.info("Saved tRNA copy numbers to %s", trna_path)


def _parse_gtrnadb_fasta(fasta_text: str) -> dict[str, dict]:
    """Parse GtRNAdb mature tRNA FASTA to count genes per anticodon.

    Headers look like:
    >sacCer3.trna1-AlaAGC (1-72)  Length: 72 bp  Type: Ala  Anticodon: AGC ...
    """
    counts: dict[str, dict[str, int | str]] = {}

    for line in fasta_text.splitlines():
        if not line.startswith(">"):
            continue
        # Try to extract amino acid and anticodon from the header
        aa_match = re.search(r"Type:\s*(\w+)", line)
        ac_match = re.search(r"Anticodon:\s*([A-Za-z]+)", line)
        if not aa_match or not ac_match:
            # Try alternate format: header name contains e.g. trna1-AlaAGC
            name_match = re.search(r"trna\d+-(\w{3})(\w{3})", line)
            if name_match:
                aa_code = name_match.group(1)
                anticodon = name_match.group(2).upper()
            else:
                continue
        else:
            aa_code = aa_match.group(1)
            anticodon = ac_match.group(1).upper()
            # Convert DNA anticodon to RNA-style (T→U) for consistency
            anticodon = anticodon.replace("T", "U")

        if anticodon not in counts:
            counts[anticodon] = {"amino_acid": aa_code, "gene_count": 0}
        counts[anticodon]["gene_count"] += 1

    return counts


def _save_wobble_rules(
    species_dir: Path,
    wobble_rules: list[dict[str, str]] | None = None,
) -> None:
    """Save curated wobble decoding rules table for a species."""
    wobble_path = species_dir / "wobble_rules.tsv"
    if wobble_rules is None:
        wobble_rules = YEAST_WOBBLE_RULES

    # Look up tRNA copy numbers if available
    trna_path = species_dir / "trna_copy_numbers.tsv"
    trna_copies: dict[str, int] = {}
    if trna_path.exists():
        trna_df = pd.read_csv(trna_path, sep="\t")
        for _, row in trna_df.iterrows():
            trna_copies[row["anticodon"]] = int(row["gene_count"])

    rows = []
    for rule in wobble_rules:
        ac = rule["decoding_anticodon"]
        copies = trna_copies.get(ac, 0)
        rows.append({
            "codon": rule["codon"],
            "amino_acid": rule["amino_acid"],
            "decoding_anticodon": ac,
            "trna_gene_copies": copies,
            "decoding_type": rule["decoding_type"],
            "modification_notes": rule["modification_notes"],
        })

    df = pd.DataFrame(rows)
    df.to_csv(wobble_path, sep="\t", index=False)
    logger.info("Saved wobble rules to %s", wobble_path)


def _compute_backgrounds(
    species_dir: Path, sequences: dict[str, str]
) -> None:
    """Pre-compute genome-wide background codon frequencies.

    Computes per-gene frequencies for mono-, di-, and tricodons,
    then stores mean, std, and (for mono/di) the full per-gene matrix.
    """
    gene_names = sorted(sequences.keys())
    n_genes = len(gene_names)
    logger.info("Computing backgrounds for %d genes...", n_genes)

    # ── Monocodons ────────────────────────────────────────────────────────
    mono_kmers = all_possible_kmers(k=1, sense_only=True)  # 61
    kmer_to_idx = {k: i for i, k in enumerate(mono_kmers)}
    n_mono = len(mono_kmers)

    per_gene_mono = np.zeros((n_genes, n_mono), dtype=np.float32)
    for gi, gene in enumerate(tqdm(gene_names, desc="Mono backgrounds")):
        freqs = kmer_frequencies(sequences[gene], k=1)
        for kmer, freq in freqs.items():
            if kmer in kmer_to_idx:
                per_gene_mono[gi, kmer_to_idx[kmer]] = freq

    mono_mean = per_gene_mono.mean(axis=0)
    mono_std = per_gene_mono.std(axis=0, ddof=1)

    np.savez_compressed(
        species_dir / "background_mono.npz",
        mean=mono_mean,
        std=mono_std,
        per_gene=per_gene_mono,
        kmer_names=np.array(mono_kmers),
        gene_names=np.array(gene_names),
    )
    logger.info("Saved mono background: %s", species_dir / "background_mono.npz")

    # ── Dicodons ──────────────────────────────────────────────────────────
    # 61^2 = 3721 possible sense dicodons
    di_kmers = all_possible_kmers(k=2, sense_only=True)
    di_kmer_to_idx = {k: i for i, k in enumerate(di_kmers)}
    n_di = len(di_kmers)

    per_gene_di = np.zeros((n_genes, n_di), dtype=np.float32)
    for gi, gene in enumerate(tqdm(gene_names, desc="Di backgrounds")):
        freqs = kmer_frequencies(sequences[gene], k=2)
        for kmer, freq in freqs.items():
            if kmer in di_kmer_to_idx:
                per_gene_di[gi, di_kmer_to_idx[kmer]] = freq

    di_mean = per_gene_di.mean(axis=0)
    di_std = per_gene_di.std(axis=0, ddof=1)

    np.savez_compressed(
        species_dir / "background_di.npz",
        mean=di_mean,
        std=di_std,
        per_gene=per_gene_di,
        kmer_names=np.array(di_kmers),
        gene_names=np.array(gene_names),
    )
    logger.info("Saved di background: %s", species_dir / "background_di.npz")

    # ── Tricodons (mean/std only — matrix too large) ──────────────────────
    logger.info("Computing tricodon backgrounds (mean/std only, no per-gene matrix)...")
    # Use a sparse accumulation approach: track sums and sums-of-squares
    # for each tricodon across genes, without materializing the full matrix.
    tri_kmers = all_possible_kmers(k=3, sense_only=True)
    tri_kmer_to_idx = {k: i for i, k in enumerate(tri_kmers)}
    n_tri = len(tri_kmers)

    tri_sum = np.zeros(n_tri, dtype=np.float64)
    tri_sumsq = np.zeros(n_tri, dtype=np.float64)

    for gi, gene in enumerate(
        tqdm(gene_names, desc="Tri backgrounds")
    ):
        freqs = kmer_frequencies(sequences[gene], k=3)
        gene_vec = np.zeros(n_tri, dtype=np.float64)
        for kmer, freq in freqs.items():
            if kmer in tri_kmer_to_idx:
                gene_vec[tri_kmer_to_idx[kmer]] = freq
        tri_sum += gene_vec
        tri_sumsq += gene_vec ** 2

    tri_mean = (tri_sum / n_genes).astype(np.float32)
    tri_var = (tri_sumsq / n_genes) - (tri_sum / n_genes) ** 2
    # Use Bessel's correction
    tri_var_corrected = tri_var * n_genes / (n_genes - 1)
    tri_var_corrected = np.maximum(tri_var_corrected, 0)  # numerical safety
    tri_std = np.sqrt(tri_var_corrected).astype(np.float32)

    np.savez_compressed(
        species_dir / "background_tri.npz",
        mean=tri_mean,
        std=tri_std,
        kmer_names=np.array(tri_kmers),
        gene_names=np.array(gene_names),
    )
    logger.info("Saved tri background: %s", species_dir / "background_tri.npz")

    # ── Per-gene metadata arrays ──────────────────────────────────────────
    lengths = np.array([len(sequences[g]) for g in gene_names], dtype=np.int32)
    gc_contents = np.array(
        [
            (sequences[g].count("G") + sequences[g].count("C")) / len(sequences[g])
            if len(sequences[g]) > 0
            else 0.0
            for g in gene_names
        ],
        dtype=np.float32,
    )
    np.savez_compressed(
        species_dir / "gene_metadata.npz",
        gene_names=np.array(gene_names),
        cds_lengths=lengths,
        gc_contents=gc_contents,
    )
    logger.info("Background pre-computation complete.")


# ═══════════════════════════════════════════════════════════════════════════════
# Human-specific download
# ═══════════════════════════════════════════════════════════════════════════════

def _download_human(species_dir: Path) -> None:
    """Download and process all human reference data.

    Steps:
    1. Download MANE Select summary from NCBI for ID mapping
    2. Download Ensembl CDS FASTA, filter to MANE Select transcripts
    3. Download tRNA gene copy numbers from GtRNAdb
    4. Save human wobble decoding rules
    5. Pre-compute backgrounds
    6. Download HGNC alias/previous symbol table
    """
    # 1. Download MANE summary + Ensembl CDS
    sequences, gene_map = _download_human_cds(species_dir)

    # 2. tRNA gene copy numbers
    _download_human_trna(species_dir)

    # 3. Wobble rules (curated)
    _save_wobble_rules(species_dir, wobble_rules=HUMAN_WOBBLE_RULES)

    # 4. Pre-compute backgrounds
    _compute_backgrounds(species_dir, sequences)

    # 5. Download GTEx expression data
    _download_human_expression(species_dir)

    # 6. Download HGNC aliases for gene ID resolution
    _download_hgnc_aliases(species_dir)

    # 7. Download CCLE cell line expression data (optional, failure is OK)
    try:
        _download_ccle_expression(species_dir)
    except Exception as exc:
        logger.warning(
            "CCLE download failed (%s). Cell line expression will use "
            "GTEx tissue proxies as fallback.", exc
        )

    logger.info("Human download complete. Files in %s", species_dir)


def _download_human_cds(
    species_dir: Path,
) -> tuple[dict[str, str], pd.DataFrame]:
    """Download human CDS sequences using MANE Select + Ensembl.

    1. Download MANE summary (NCBI) for gene ID mapping
    2. Download Ensembl CDS FASTA (all transcripts)
    3. Filter to MANE Select transcripts only
    4. Validate CDS sequences

    Returns:
        (sequences dict {ensg_id: cds}, gene_map DataFrame)
    """
    # ── Step 1: Download and parse MANE summary ──────────────────────────
    logger.info("Downloading MANE Select summary from NCBI...")
    resp = requests.get(MANE_SUMMARY_URL, timeout=120)
    resp.raise_for_status()
    mane_text = gzip.decompress(resp.content).decode("utf-8")

    mane_records = _parse_mane_summary(mane_text)
    logger.info("Parsed %d MANE Select entries", len(mane_records))

    # Build ENST → record lookup (strip version numbers for matching)
    enst_to_record: dict[str, dict] = {}
    for rec in mane_records:
        enst_base = rec["ensembl_transcript"].split(".")[0]
        enst_to_record[enst_base] = rec

    # ── Step 2: Download Ensembl CDS FASTA ───────────────────────────────
    logger.info("Downloading Ensembl CDS FASTA (this may take a few minutes)...")
    resp = requests.get(ENSEMBL_HUMAN_CDS_URL, timeout=600, stream=True)
    resp.raise_for_status()

    # Decompress in memory
    raw_fasta = gzip.decompress(resp.content).decode("ascii", errors="replace")
    logger.info("Downloaded Ensembl CDS FASTA, parsing...")

    # ── Step 3: Parse and filter to MANE Select ──────────────────────────
    sequences, records = _parse_ensembl_cds_for_mane(raw_fasta, enst_to_record)
    logger.info(
        "Matched %d MANE Select transcripts with validated CDS", len(sequences)
    )

    # ── Step 4: Save cleaned FASTA ───────────────────────────────────────
    cds_path = species_dir / "cds_sequences.fa.gz"
    with gzip.open(cds_path, "wt") as fh:
        for ensg, seq in sorted(sequences.items()):
            fh.write(f">{ensg}\n{seq}\n")
    logger.info("Saved %d validated CDS to %s", len(sequences), cds_path)

    # ── Step 5: Save gene ID map ─────────────────────────────────────────
    gene_map = pd.DataFrame(records)
    gene_map.to_csv(species_dir / "gene_id_map.tsv", sep="\t", index=False)
    logger.info("Saved gene ID map with %d entries", len(gene_map))

    return sequences, gene_map


def _parse_mane_summary(text: str) -> list[dict]:
    """Parse NCBI MANE summary TSV.

    Columns we need:
    - #NCBI_GeneID → entrez_id
    - Ensembl_Gene → ensembl_gene
    - symbol → hgnc_symbol
    - RefSeq_nuc → refseq_transcript
    - Ensembl_nuc → ensembl_transcript
    - MANE_status → filter to "MANE Select"
    """
    records = []
    lines = text.strip().split("\n")
    header = None

    for line in lines:
        if line.startswith("##"):
            continue
        if line.startswith("#"):
            # Header line
            header = line.lstrip("#").strip().split("\t")
            continue
        if header is None:
            continue

        fields = line.split("\t")
        if len(fields) < len(header):
            continue

        row = dict(zip(header, fields))

        # Only keep MANE Select (not MANE Plus Clinical)
        if row.get("MANE_status", "") != "MANE Select":
            continue

        entrez_id = row.get("NCBI_GeneID", "").replace("GeneID:", "")
        records.append({
            "ensembl_gene": row.get("Ensembl_Gene", ""),
            "hgnc_symbol": row.get("symbol", ""),
            "entrez_id": entrez_id,
            "refseq_transcript": row.get("RefSeq_nuc", ""),
            "ensembl_transcript": row.get("Ensembl_nuc", ""),
        })

    return records


def _parse_ensembl_cds_for_mane(
    fasta_text: str,
    enst_to_record: dict[str, dict],
) -> tuple[dict[str, str], list[dict]]:
    """Parse Ensembl CDS FASTA, keeping only MANE Select transcripts.

    Ensembl CDS headers look like:
    >ENST00000341065.8 cds chromosome:GRCh38:1:... gene:ENSG00000187634.13 ...

    Returns:
        (sequences dict {ensg_id: cds}, list of record dicts for gene_id_map)
    """
    sequences: dict[str, str] = {}
    records: list[dict] = []
    stop_codons = {"TAA", "TAG", "TGA"}

    current_enst: str | None = None
    current_seq_parts: list[str] = []

    def _process_entry(enst_id: str, seq: str) -> None:
        # Strip version number for matching
        enst_base = enst_id.split(".")[0]

        if enst_base not in enst_to_record:
            return  # Not a MANE Select transcript

        rec = enst_to_record[enst_base]
        ensg = rec["ensembl_gene"].split(".")[0]  # strip version
        seq_upper = seq.upper()

        # Validate: only ACGT
        if not re.fullmatch(r"[ACGT]+", seq_upper):
            logger.debug("Skipping %s: non-ACGT characters", enst_id)
            return

        # Validate: divisible by 3
        if len(seq_upper) % 3 != 0:
            logger.debug(
                "Skipping %s: length %d not divisible by 3",
                enst_id, len(seq_upper),
            )
            return

        # Validate: starts with ATG
        if not seq_upper.startswith("ATG"):
            logger.debug("Skipping %s: does not start with ATG", enst_id)
            return

        # Validate: ends with stop codon
        last_codon = seq_upper[-3:]
        if last_codon not in stop_codons:
            logger.debug(
                "Skipping %s: does not end with stop codon (%s)",
                enst_id, last_codon,
            )
            return

        # Strip stop codon
        cds = seq_upper[:-3]

        # Check for internal stop codons
        for i in range(0, len(cds), 3):
            if cds[i : i + 3] in stop_codons:
                logger.debug("Skipping %s: internal stop codon", enst_id)
                return

        # Skip if we already have a CDS for this gene (first match wins)
        if ensg in sequences:
            return

        gc_count = cds.count("G") + cds.count("C")
        gc_content = gc_count / len(cds) if len(cds) > 0 else 0.0

        sequences[ensg] = cds
        records.append({
            "systematic_name": ensg,
            "common_name": rec["hgnc_symbol"],
            "ensembl_transcript": rec["ensembl_transcript"],
            "refseq_transcript": rec["refseq_transcript"],
            "entrez_id": rec["entrez_id"],
            "cds_length": len(cds),
            "gc_content": round(gc_content, 4),
        })

    for line in fasta_text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current_enst is not None:
                _process_entry(current_enst, "".join(current_seq_parts))
            # Extract ENST ID from header
            current_enst = line[1:].split()[0]
            current_seq_parts = []
        else:
            current_seq_parts.append(line)

    # Process last entry
    if current_enst is not None:
        _process_entry(current_enst, "".join(current_seq_parts))

    return sequences, records


def _download_human_trna(species_dir: Path) -> None:
    """Download human tRNA gene copy numbers from GtRNAdb, with fallback."""
    trna_path = species_dir / "trna_copy_numbers.tsv"

    try:
        logger.info("Downloading human tRNA data from GtRNAdb...")
        resp = requests.get(GTRNADB_HUMAN_URL, timeout=60)
        resp.raise_for_status()
        trna_counts = _parse_gtrnadb_fasta(resp.text)
        if len(trna_counts) < 20:
            raise ValueError(
                f"Only parsed {len(trna_counts)} tRNA anticodons, expected ~40+"
            )
        logger.info("Parsed %d tRNA anticodons from GtRNAdb", len(trna_counts))
    except Exception as exc:
        logger.warning(
            "GtRNAdb download failed (%s), using hardcoded fallback", exc
        )
        trna_counts = {
            anticodon: {"amino_acid": aa, "gene_count": count}
            for anticodon, (aa, count) in HUMAN_TRNA_FALLBACK.items()
        }

    rows = []
    for anticodon, info in sorted(trna_counts.items()):
        rows.append({
            "anticodon": anticodon,
            "amino_acid": info["amino_acid"],
            "gene_count": info["gene_count"],
        })

    df = pd.DataFrame(rows)
    df.to_csv(trna_path, sep="\t", index=False)
    logger.info("Saved human tRNA copy numbers to %s", trna_path)


# ═══════════════════════════════════════════════════════════════════════════════
# Mouse-specific download
# ═══════════════════════════════════════════════════════════════════════════════

def _download_mouse(species_dir: Path) -> None:
    """Download and process all mouse reference data.

    Steps:
    1. Download Ensembl CDS FASTA, keep longest valid CDS per gene
    2. Download tRNA gene copy numbers from GtRNAdb (with fallback)
    3. Save wobble decoding rules (reuse human/mammalian rules)
    4. Pre-compute backgrounds
    5. Create expression estimates (hardcoded)
    """
    # 1. CDS sequences from Ensembl
    sequences, gene_map = _download_mouse_cds(species_dir)

    # 2. tRNA gene copy numbers
    _download_mouse_trna(species_dir)

    # 3. Wobble rules (same mammalian modifications as human)
    _save_wobble_rules(species_dir, wobble_rules=HUMAN_WOBBLE_RULES)

    # 4. Pre-compute backgrounds
    _compute_backgrounds(species_dir, sequences)

    # 5. Create expression estimates
    _create_mouse_expression(species_dir)

    logger.info("Mouse download complete. Files in %s", species_dir)


def _download_mouse_cds(
    species_dir: Path,
) -> tuple[dict[str, str], pd.DataFrame]:
    """Download mouse CDS sequences from Ensembl and build gene ID map.

    For each gene (ENSMUSG), keeps the longest valid CDS (no MANE for mouse).
    Validates: ACGT only, divisible by 3, starts ATG, ends stop, no internal stops.

    Returns:
        (sequences dict {ensmusg_id: cds}, gene_map DataFrame)
    """
    logger.info("Downloading mouse CDS from Ensembl...")
    resp = requests.get(ENSEMBL_MOUSE_CDS_URL, timeout=600, stream=True)
    resp.raise_for_status()

    raw_fasta = gzip.decompress(resp.content).decode("ascii", errors="replace")
    logger.info("Downloaded Ensembl mouse CDS FASTA, parsing...")

    sequences, records = _parse_ensembl_mouse_cds(raw_fasta)
    logger.info("Validated %d mouse protein-coding genes", len(sequences))

    # Save cleaned FASTA
    cds_path = species_dir / "cds_sequences.fa.gz"
    with gzip.open(cds_path, "wt") as fh:
        for ensmusg, seq in sorted(sequences.items()):
            fh.write(f">{ensmusg}\n{seq}\n")
    logger.info("Saved %d validated CDS to %s", len(sequences), cds_path)

    # Save gene ID map
    gene_map = pd.DataFrame(records)
    gene_map.to_csv(species_dir / "gene_id_map.tsv", sep="\t", index=False)
    logger.info("Saved gene ID map with %d entries", len(gene_map))

    return sequences, gene_map


def _parse_ensembl_mouse_cds(
    fasta_text: str,
) -> tuple[dict[str, str], list[dict]]:
    """Parse Ensembl mouse CDS FASTA, keeping longest valid CDS per gene.

    Ensembl CDS headers look like:
    >ENSMUST00000... cds chromosome:GRCm39:... gene:ENSMUSG00000... gene_symbol:Xyz ...

    Returns:
        (sequences dict {ensmusg_id: cds}, list of record dicts for gene_id_map)
    """
    stop_codons = {"TAA", "TAG", "TGA"}

    # Collect all valid CDS per gene, keep longest
    gene_candidates: dict[str, list[tuple[str, str, str]]] = {}  # ENSMUSG → [(ensmust, symbol, cds), ...]

    current_ensmust: str | None = None
    current_header: str = ""
    current_seq_parts: list[str] = []

    def _process_entry(header: str, seq: str) -> None:
        # Extract ENSMUST ID
        ensmust = header.split()[0]
        ensmust_base = ensmust.split(".")[0]

        # Extract ENSMUSG from header
        gene_match = re.search(r"gene:(ENSMUSG\d+(\.\d+)?)", header)
        if not gene_match:
            return
        ensmusg = gene_match.group(1).split(".")[0]

        # Extract gene symbol
        symbol_match = re.search(r"gene_symbol:(\S+)", header)
        symbol = symbol_match.group(1) if symbol_match else ""

        seq_upper = seq.upper()

        # Validate: only ACGT
        if not re.fullmatch(r"[ACGT]+", seq_upper):
            return

        # Validate: divisible by 3
        if len(seq_upper) % 3 != 0:
            return

        # Validate: starts with ATG
        if not seq_upper.startswith("ATG"):
            return

        # Validate: ends with stop codon
        if seq_upper[-3:] not in stop_codons:
            return

        # Strip stop codon
        cds = seq_upper[:-3]

        # Check for internal stop codons
        for i in range(0, len(cds), 3):
            if cds[i : i + 3] in stop_codons:
                return

        if len(cds) < 9:  # at least 3 codons
            return

        if ensmusg not in gene_candidates:
            gene_candidates[ensmusg] = []
        gene_candidates[ensmusg].append((ensmust_base, symbol, cds))

    for line in fasta_text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current_ensmust is not None:
                _process_entry(current_header, "".join(current_seq_parts))
            current_header = line[1:]
            current_ensmust = current_header.split()[0]
            current_seq_parts = []
        else:
            current_seq_parts.append(line)

    # Process last entry
    if current_ensmust is not None:
        _process_entry(current_header, "".join(current_seq_parts))

    # Keep longest CDS per gene
    sequences: dict[str, str] = {}
    records: list[dict] = []

    for ensmusg, candidates in gene_candidates.items():
        # Sort by CDS length descending, take longest
        candidates.sort(key=lambda x: len(x[2]), reverse=True)
        ensmust, symbol, cds = candidates[0]

        gc_count = cds.count("G") + cds.count("C")
        gc_content = gc_count / len(cds) if len(cds) > 0 else 0.0

        sequences[ensmusg] = cds
        records.append({
            "systematic_name": ensmusg,
            "common_name": symbol,
            "ensembl_transcript": ensmust,
            "cds_length": len(cds),
            "gc_content": round(gc_content, 4),
        })

    return sequences, records


def _download_mouse_trna(species_dir: Path) -> None:
    """Download mouse tRNA gene copy numbers from GtRNAdb, with fallback."""
    trna_path = species_dir / "trna_copy_numbers.tsv"

    try:
        logger.info("Downloading mouse tRNA data from GtRNAdb...")
        resp = requests.get(GTRNADB_MOUSE_URL, timeout=60)
        resp.raise_for_status()
        trna_counts = _parse_gtrnadb_fasta(resp.text)
        if len(trna_counts) < 20:
            raise ValueError(
                f"Only parsed {len(trna_counts)} tRNA anticodons, expected ~40+"
            )
        logger.info("Parsed %d tRNA anticodons from GtRNAdb", len(trna_counts))
    except Exception as exc:
        logger.warning(
            "GtRNAdb download failed (%s), using hardcoded fallback", exc
        )
        trna_counts = {
            anticodon: {"amino_acid": aa, "gene_count": count}
            for anticodon, (aa, count) in MOUSE_TRNA_FALLBACK.items()
        }

    rows = []
    for anticodon, info in sorted(trna_counts.items()):
        rows.append({
            "anticodon": anticodon,
            "amino_acid": info["amino_acid"],
            "gene_count": info["gene_count"],
        })

    df = pd.DataFrame(rows)
    df.to_csv(trna_path, sep="\t", index=False)
    logger.info("Saved mouse tRNA copy numbers to %s", trna_path)


def _create_mouse_expression(species_dir: Path) -> None:
    """Create mouse expression estimates.

    Assigns approximate TPM values:
    - Rpl*/Rps* ribosomal proteins → 3000 TPM
    - Known highly-expressed genes → specific values from MOUSE_HIGH_EXPRESSION
    - Other genes → 15 TPM (median gene)

    Saves: expression_estimates.tsv (systematic_name, common_name, tpm)
    """
    expr_path = species_dir / "expression_estimates.tsv"

    gene_map = pd.read_csv(species_dir / "gene_id_map.tsv", sep="\t")
    rows = []

    for _, row in gene_map.iterrows():
        sys_name = row["systematic_name"]
        common = row["common_name"]

        # Determine TPM
        if isinstance(common, str) and common in MOUSE_HIGH_EXPRESSION:
            tpm = MOUSE_HIGH_EXPRESSION[common]
        elif isinstance(common, str) and (
            common.startswith("Rpl") or common.startswith("Rps")
        ):
            tpm = 3000.0
        else:
            tpm = 15.0

        rows.append({
            "systematic_name": sys_name,
            "common_name": common,
            "tpm": tpm,
        })

    df = pd.DataFrame(rows)
    df.to_csv(expr_path, sep="\t", index=False)
    logger.info(
        "Created mouse expression estimates: %d genes, "
        "mean TPM %.1f, saved to %s",
        len(df), df["tpm"].mean(), expr_path,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Expression data
# ═══════════════════════════════════════════════════════════════════════════════

def _create_yeast_expression(species_dir: Path) -> None:
    """Create yeast expression estimates for rich media (YPD).

    Assigns approximate TPM values from published literature:
    - RPL*/RPS* ribosomal proteins → 3000 TPM
    - Known highly-expressed genes → specific values from YEAST_HIGH_EXPRESSION
    - Other Verified/Uncharacterized → 15 TPM (median gene)
    - Dubious ORFs → 0.5 TPM
    - Mitochondrial (Q0*) → 50 TPM

    Saves: expression_rich_media.tsv (systematic_name, common_name, tpm)
    """
    expr_path = species_dir / "expression_rich_media.tsv"

    gene_map = pd.read_csv(species_dir / "gene_id_map.tsv", sep="\t")
    rows = []

    for _, row in gene_map.iterrows():
        sys_name = row["systematic_name"]
        common = row["common_name"]
        status = row.get("status", "Verified")

        # Determine TPM
        if common in YEAST_HIGH_EXPRESSION:
            tpm = YEAST_HIGH_EXPRESSION[common]
        elif isinstance(common, str) and (
            common.startswith("RPL") or common.startswith("RPS")
        ):
            tpm = 3000.0
        elif isinstance(sys_name, str) and sys_name.startswith("Q0"):
            tpm = 50.0
        elif status == "Dubious":
            tpm = 0.5
        else:
            tpm = 15.0

        rows.append({
            "systematic_name": sys_name,
            "common_name": common,
            "tpm": tpm,
        })

    df = pd.DataFrame(rows)
    df.to_csv(expr_path, sep="\t", index=False)
    logger.info(
        "Created yeast expression estimates: %d genes, "
        "mean TPM %.1f, saved to %s",
        len(df), df["tpm"].mean(), expr_path,
    )


def _download_human_expression(species_dir: Path) -> None:
    """Download GTEx v8 median TPM per tissue.

    Downloads the gene-level median TPM GCT file from the GTEx portal,
    parses it, strips Ensembl gene version suffixes, and saves as a
    compressed TSV.

    Saves: expression_gtex.tsv.gz (ensembl_gene, symbol, <tissue_columns...>)
    """
    expr_path = species_dir / "expression_gtex.tsv.gz"

    logger.info("Downloading GTEx v8 median TPM data...")
    try:
        resp = requests.get(GTEX_MEDIAN_TPM_URL, timeout=300, stream=True)
        resp.raise_for_status()
    except Exception as exc:
        logger.warning(
            "GTEx download failed (%s). Mode 2 will not be available "
            "for human without expression data.", exc
        )
        return

    # GCT format: line 1 = #1.2, line 2 = dimensions, line 3+ = data
    raw = gzip.decompress(resp.content).decode("utf-8")
    lines = raw.split("\n")

    # Skip first 2 header lines (#1.2 and dimensions)
    header_line = lines[2]  # column names
    data_lines = lines[3:]

    cols = header_line.split("\t")
    # cols[0] = "Name" (ENSG with version), cols[1] = "Description" (symbol)
    # cols[2:] = tissue names

    rows = []
    for line in data_lines:
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        ensg_versioned = parts[0]
        symbol = parts[1]
        # Strip version suffix: ENSG00000000003.15 → ENSG00000000003
        ensg = ensg_versioned.split(".")[0]
        tpm_values = parts[2:]
        rows.append([ensg, symbol] + tpm_values)

    tissue_names = cols[2:]
    out_cols = ["ensembl_gene", "symbol"] + tissue_names

    df = pd.DataFrame(rows, columns=out_cols)

    # Convert TPM columns to float
    for col in tissue_names:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df.to_csv(expr_path, sep="\t", index=False, compression="gzip")
    logger.info(
        "Saved GTEx expression: %d genes × %d tissues to %s",
        len(df), len(tissue_names), expr_path,
    )


def _download_ccle_expression(species_dir: Path) -> None:
    """Download CCLE/DepMap cell line expression data.

    Downloads:
    1. Model.csv — cell line metadata (name mapping, lineage)
    2. OmicsExpressionProteinCodingGenesTPMLogp1.csv — expression matrix

    Saves:
    - expression_ccle.tsv.gz  (ensembl_gene, symbol, <cell_line_columns...>)
    - ccle_cell_lines.tsv     (cell_line_name, stripped_name, lineage)

    Values in the DepMap file are log2(TPM+1); we convert back to TPM.
    Gene columns are "SYMBOL (ENTREZ)" format; we map to ENSG via gene_id_map.
    """
    ccle_path = species_dir / "expression_ccle.tsv.gz"
    meta_path = species_dir / "ccle_cell_lines.tsv"

    # Load gene_id_map for Entrez → ENSG mapping
    gene_map_path = species_dir / "gene_id_map.tsv"
    if not gene_map_path.exists():
        raise FileNotFoundError(
            "gene_id_map.tsv not found. Run human download first."
        )
    gene_map = pd.read_csv(gene_map_path, sep="\t")

    # Build Entrez ID → (ENSG, symbol) lookup
    entrez_to_ensg: dict[str, tuple[str, str]] = {}
    if "entrez_id" in gene_map.columns:
        for _, row in gene_map.iterrows():
            eid = row.get("entrez_id")
            if pd.notna(eid):
                eid_str = str(int(float(eid)))
                entrez_to_ensg[eid_str] = (
                    row["systematic_name"],
                    row.get("common_name", ""),
                )

    # Also build symbol → ENSG for fallback
    symbol_to_ensg: dict[str, tuple[str, str]] = {}
    for _, row in gene_map.iterrows():
        cn = row.get("common_name", "")
        if cn:
            symbol_to_ensg[cn.upper()] = (
                row["systematic_name"],
                cn,
            )

    # ── Step 0: Discover download URLs via DepMap API ──────────────────
    logger.info("Querying DepMap file listing API...")
    api_resp = requests.get(DEPMAP_FILES_API, timeout=60)
    api_resp.raise_for_status()
    files_df = pd.read_csv(io.StringIO(api_resp.text))

    # Find the latest release that has both Model.csv and expression TPM
    # Prefer the newest release (first rows in the listing)
    model_url = None
    expr_url = None
    for _, row in files_df.iterrows():
        fname = str(row.get("filename", ""))
        url = str(row.get("url", ""))
        if fname == "Model.csv" and model_url is None:
            model_url = url
        # Match protein-coding TPM files (name varies across releases)
        if ("ExpressionTPMLogp1" in fname or "ExpressionProteinCodingGenesTPMLogp1" in fname) \
                and "ProteinCoding" in fname and "Stranded" not in fname \
                and "Transcript" not in fname and expr_url is None:
            expr_url = url

    # Fallback: accept any protein-coding TPM file
    if expr_url is None:
        for _, row in files_df.iterrows():
            fname = str(row.get("filename", ""))
            url = str(row.get("url", ""))
            if "ExpressionProteinCodingGenesTPMLogp1" in fname \
                    and "Stranded" not in fname and "Transcript" not in fname:
                expr_url = url
                break

    if model_url is None or expr_url is None:
        raise RuntimeError(
            "Could not find Model.csv and/or expression TPM file in DepMap API. "
            f"Model URL: {model_url}, Expression URL: {expr_url}"
        )

    logger.info("Found DepMap Model.csv and expression TPM URLs via API.")

    # ── Step 1: Download Model.csv ──────────────────────────────────────
    logger.info("Downloading DepMap Model.csv...")
    resp = requests.get(model_url, timeout=120)
    resp.raise_for_status()

    model_df = pd.read_csv(io.StringIO(resp.text))
    # Build ModelID → cell line name/lineage mapping
    model_info: dict[str, dict] = {}
    name_col = "CellLineName" if "CellLineName" in model_df.columns else "cell_line_name"
    stripped_col = "StrippedCellLineName" if "StrippedCellLineName" in model_df.columns else None
    lineage_col = "OncotreeLineage" if "OncotreeLineage" in model_df.columns else None
    model_id_col = "ModelID" if "ModelID" in model_df.columns else "model_id"

    for _, row in model_df.iterrows():
        mid = str(row.get(model_id_col, ""))
        cname = str(row.get(name_col, mid))
        stripped = str(row.get(stripped_col, cname)).upper() if stripped_col else cname.upper()
        lineage = str(row.get(lineage_col, "")) if lineage_col else ""
        model_info[mid] = {
            "cell_line_name": cname,
            "stripped_name": stripped,
            "lineage": lineage,
        }

    logger.info("Parsed %d cell line models from DepMap", len(model_info))

    # ── Step 2: Download expression matrix ──────────────────────────────
    logger.info("Downloading DepMap expression matrix (this may take a few minutes)...")
    resp = requests.get(expr_url, timeout=600, stream=True)
    resp.raise_for_status()

    # Parse CSV — format has metadata columns before gene columns
    # Columns: [index], SequencingID, ModelID, IsDefaultEntryForModel,
    #           ModelConditionID, IsDefaultEntryForMC, GENE1 (ENTREZ), ...
    expr_raw = pd.read_csv(io.StringIO(resp.text), index_col=0)

    # Identify gene columns (format "SYMBOL (ENTREZ)") vs metadata columns
    gene_cols = [c for c in expr_raw.columns if re.search(r"\(\d+\)$", str(c))]
    meta_cols = [c for c in expr_raw.columns if c not in gene_cols]
    logger.info(
        "Downloaded expression matrix: %d rows × %d gene columns "
        "(+ %d metadata columns)",
        len(expr_raw), len(gene_cols), len(meta_cols),
    )

    # Use ModelID as index (ACH-xxxxxx), filter to default entries only
    if "ModelID" in meta_cols:
        expr_raw["_model_id"] = expr_raw["ModelID"]
        # Keep only default entry per model to avoid duplicates
        if "IsDefaultEntryForModel" in meta_cols:
            expr_raw = expr_raw[expr_raw["IsDefaultEntryForModel"] == "Yes"]
        expr_df = expr_raw[gene_cols].copy()
        expr_df.index = expr_raw["_model_id"]
    else:
        # Fallback: assume first column is ModelID
        expr_df = expr_raw[gene_cols].copy()

    logger.info(
        "Expression matrix after filtering: %d cell lines × %d genes",
        len(expr_df), len(expr_df.columns),
    )

    # ── Step 3: Map gene columns to ENSG ────────────────────────────────
    # Column format: "SYMBOL (12345)" — extract Entrez ID
    import re as _re
    col_mapping: dict[str, tuple[str, str]] = {}  # old_col → (ensg, symbol)
    for col in expr_df.columns:
        match = _re.match(r"(.+?)\s*\((\d+)\)", col)
        if match:
            symbol = match.group(1).strip()
            entrez = match.group(2)
            if entrez in entrez_to_ensg:
                col_mapping[col] = entrez_to_ensg[entrez]
            elif symbol.upper() in symbol_to_ensg:
                col_mapping[col] = symbol_to_ensg[symbol.upper()]

    if not col_mapping:
        raise ValueError("Could not map any CCLE gene columns to ENSG IDs.")

    logger.info(
        "Mapped %d/%d CCLE gene columns to ENSG",
        len(col_mapping), len(expr_df.columns),
    )

    # ── Step 4: Convert log2(TPM+1) → TPM ──────────────────────────────
    # Select mapped columns only
    mapped_cols = list(col_mapping.keys())
    expr_mapped = expr_df[mapped_cols].copy()

    # Convert from log2(TPM+1) to TPM
    expr_mapped = 2.0 ** expr_mapped - 1.0
    expr_mapped = expr_mapped.clip(lower=0.0)

    # ── Step 5: Rename columns to ENSG, rename rows to cell line names ──
    # Handle duplicate ENSG mappings by keeping first occurrence
    new_col_names = {}
    seen_ensg = set()
    keep_cols = []
    for col in mapped_cols:
        ensg, _ = col_mapping[col]
        if ensg not in seen_ensg:
            new_col_names[col] = ensg
            seen_ensg.add(ensg)
            keep_cols.append(col)

    expr_mapped = expr_mapped[keep_cols].rename(columns=new_col_names)

    # Replace row index (ModelID) with human-readable cell line names
    new_index = []
    for mid in expr_mapped.index:
        if mid in model_info:
            new_index.append(model_info[mid]["cell_line_name"])
        else:
            new_index.append(mid)
    expr_mapped.index = new_index

    # ── Step 6: Transpose to genes × cell_lines format ──────────────────
    # (matching GTEx storage: rows=genes, columns=cell lines)
    out_df = expr_mapped.T
    out_df.index.name = "ensembl_gene"
    out_df = out_df.reset_index()

    # Add symbol column
    ensg_to_symbol = {}
    for _, row in gene_map.iterrows():
        ensg_to_symbol[row["systematic_name"]] = row.get("common_name", "")
    out_df.insert(1, "symbol", out_df["ensembl_gene"].map(ensg_to_symbol).fillna(""))

    # ── Step 7: Save ────────────────────────────────────────────────────
    cell_line_cols = [c for c in out_df.columns if c not in ("ensembl_gene", "symbol")]
    out_df.to_csv(ccle_path, sep="\t", index=False, compression="gzip")
    logger.info(
        "Saved CCLE expression: %d genes × %d cell lines to %s",
        len(out_df), len(cell_line_cols), ccle_path,
    )

    # Save cell line metadata
    meta_rows = []
    for cname in cell_line_cols:
        # Find this cell line in model_info
        found = False
        for mid, info in model_info.items():
            if info["cell_line_name"] == cname:
                meta_rows.append({
                    "cell_line_name": cname,
                    "stripped_name": info["stripped_name"],
                    "lineage": info["lineage"],
                })
                found = True
                break
        if not found:
            cname_str = str(cname)
            meta_rows.append({
                "cell_line_name": cname_str,
                "stripped_name": cname_str.upper().replace("-", "").replace(" ", ""),
                "lineage": "",
            })

    meta_df = pd.DataFrame(meta_rows)
    meta_df.to_csv(meta_path, sep="\t", index=False)
    logger.info("Saved cell line metadata: %d entries to %s", len(meta_df), meta_path)


def download_ccle(data_dir: str | Path | None = None) -> Path:
    """Download CCLE cell line expression data (standalone).

    Args:
        data_dir: Override default data directory.

    Returns:
        Path to the human species directory.
    """
    base = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    species_dir = base / "human"
    if not species_dir.exists():
        raise FileNotFoundError(
            f"Human data directory not found: {species_dir}. "
            f"Run: codonscope download --species human"
        )
    _download_ccle_expression(species_dir)
    return species_dir


def _download_hgnc_aliases(species_dir: Path) -> None:
    """Download HGNC complete set and build alias/previous symbol lookup.

    Creates hgnc_aliases.tsv mapping alias/previous symbols to their
    canonical HGNC symbol and Ensembl gene ID.  Only includes entries
    where the ENSG exists in our gene_id_map (MANE Select genes).
    """
    alias_path = species_dir / "hgnc_aliases.tsv"

    # Load existing gene_id_map to filter aliases to known genes
    gene_map_path = species_dir / "gene_id_map.tsv"
    if not gene_map_path.exists():
        logger.warning("gene_id_map.tsv not found, skipping HGNC alias download")
        return

    gene_map = pd.read_csv(gene_map_path, sep="\t")
    known_ensg = set(gene_map["systematic_name"].str.strip())
    # Also build symbol→ENSG for current symbols (already in common_name)
    symbol_to_ensg: dict[str, str] = {}
    for _, row in gene_map.iterrows():
        sym = str(row["common_name"]).strip().upper()
        ensg = str(row["systematic_name"]).strip()
        if sym and sym != "NAN":
            symbol_to_ensg[sym] = ensg

    try:
        logger.info("Downloading HGNC complete gene set...")
        resp = requests.get(HGNC_COMPLETE_SET_URL, timeout=120)
        resp.raise_for_status()
    except Exception as exc:
        logger.warning("HGNC download failed (%s), alias lookup will not be available", exc)
        return

    # Parse TSV
    hgnc_df = pd.read_csv(io.StringIO(resp.text), sep="\t", low_memory=False)

    rows: list[dict] = []
    for _, row in hgnc_df.iterrows():
        symbol = str(row.get("symbol", "")).strip()
        if not symbol:
            continue

        # Get Ensembl gene ID from HGNC record
        ensg_raw = str(row.get("ensembl_gene_id", "")).strip()
        ensg = ensg_raw if ensg_raw and ensg_raw != "nan" else None

        # Only include if ENSG is in our MANE Select set
        if ensg is None or ensg not in known_ensg:
            continue

        # Previous symbols (pipe-separated)
        prev_raw = str(row.get("prev_symbol", "")).strip()
        if prev_raw and prev_raw != "nan":
            for prev in prev_raw.split("|"):
                prev = prev.strip().strip('"')
                if prev and prev.upper() != symbol.upper():
                    rows.append({
                        "alias": prev,
                        "canonical_symbol": symbol,
                        "ensembl_gene_id": ensg,
                        "alias_type": "previous",
                    })

        # Alias symbols (pipe-separated)
        alias_raw = str(row.get("alias_symbol", "")).strip()
        if alias_raw and alias_raw != "nan":
            for alias in alias_raw.split("|"):
                alias = alias.strip().strip('"')
                if alias and alias.upper() != symbol.upper():
                    rows.append({
                        "alias": alias,
                        "canonical_symbol": symbol,
                        "ensembl_gene_id": ensg,
                        "alias_type": "alias",
                    })

    if not rows:
        logger.warning("No HGNC aliases matched MANE Select genes")
        return

    alias_df = pd.DataFrame(rows)
    # Remove duplicates (keep first — previous takes priority)
    alias_df = alias_df.drop_duplicates(subset=["alias"], keep="first")
    alias_df.to_csv(alias_path, sep="\t", index=False)
    logger.info(
        "Saved %d HGNC aliases (previous + alias symbols) to %s",
        len(alias_df), alias_path,
    )


def download_expression(
    species: str,
    data_dir: str | Path | None = None,
) -> Path:
    """Download/create expression data for a species (standalone).

    Can be called independently of the full download pipeline.
    """
    species = species.lower()
    base = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    species_dir = base / species
    species_dir.mkdir(parents=True, exist_ok=True)

    if species == "yeast":
        _create_yeast_expression(species_dir)
    elif species == "human":
        _download_human_expression(species_dir)
    elif species == "mouse":
        _create_mouse_expression(species_dir)
    else:
        raise ValueError(f"Unsupported species for expression: {species!r}")

    return species_dir


# ═══════════════════════════════════════════════════════════════════════════════
# Ortholog download
# ═══════════════════════════════════════════════════════════════════════════════

BIOMART_URL = "https://www.ensembl.org/biomart/martservice"

# BioMart XML query for human-yeast one-to-one orthologs
BIOMART_HUMAN_YEAST_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE Query>
<Query virtualSchemaName="default" formatter="TSV" header="1"
       uniqueRows="1" count="" datasetConfigVersion="0.6">
  <Dataset name="hsapiens_gene_ensembl" interface="default">
    <Filter name="with_scerevisiae_homolog" excluded="0"/>
    <Filter name="homolog_scerevisiae_orthology_type" value="ortholog_one2one"/>
    <Attribute name="ensembl_gene_id"/>
    <Attribute name="external_gene_name"/>
    <Attribute name="scerevisiae_homolog_ensembl_gene"/>
    <Attribute name="scerevisiae_homolog_associated_gene_name"/>
    <Attribute name="scerevisiae_homolog_orthology_type"/>
  </Dataset>
</Query>"""

# Curated human→yeast gene name mappings for genes whose names differ
# between species.  Only well-established ortholog pairs.
HUMAN_YEAST_CURATED_RENAMES: dict[str, str] = {
    # Glycolysis / gluconeogenesis
    "GAPDH": "TDH3", "GAPDHS": "TDH3",
    "ENO1": "ENO2", "ENO2": "ENO1",  # note: swap
    "PKM": "PYK1", "PKLR": "PYK1",
    "ALDOA": "FBA1", "ALDOB": "FBA1", "ALDOC": "FBA1",
    "PGK1": "PGK1",
    "TPI1": "TPI1",
    "GPI": "PGI1",
    "PFKM": "PFK1", "PFKL": "PFK1", "PFKP": "PFK1",
    "HK1": "HXK2", "HK2": "HXK2",
    "PGAM1": "GPM1", "PGAM2": "GPM1",
    "LDHA": "DLD1", "LDHB": "DLD1",
    # Translation factors
    "EEF1A1": "TEF1", "EEF1A2": "TEF2",
    "EEF2": "EFT1",
    "EIF4A1": "TIF1", "EIF4A2": "TIF1",
    "EIF2S1": "SUI2", "EIF2S2": "SUI3", "EIF2S3": "GCD11",
    "EIF3A": "RPG1", "EIF5B": "FUN12",
    "ETF1": "SUP45", "GSPT1": "SUP35",
    "ABCF1": "GCN20",
    # Ribosomal proteins — yeast has A/B paralogs, human has single genes
    "RPSA": "RPS0A", "RPS0": "RPS0A",
    "FAU": "RPS30A",
    "UBA52": "RPL40A", "UBA80": "RPL40A",
    "RPS27A": "RPS31",
    "RACK1": "ASC1",
    # RPL — human → yeast (map to A paralog by convention)
    "RPL4": "RPL4A", "RPL6": "RPL6A", "RPL7": "RPL7A", "RPL8": "RPL8A",
    "RPL9": "RPL9A",
    "RPL11": "RPL11A", "RPL12": "RPL12A", "RPL13": "RPL13A",
    "RPL14": "RPL14A", "RPL15": "RPL15A",
    "RPL17": "RPL17A", "RPL18": "RPL18A",
    "RPL19": "RPL19A", "RPL21": "RPL21A", "RPL22": "RPL22A",
    "RPL23": "RPL23A", "RPL24": "RPL24A",
    "RPL26": "RPL26A", "RPL27": "RPL27A",
    "RPL31": "RPL31A",
    "RPL34": "RPL34A", "RPL35": "RPL35A", "RPL36": "RPL36A",
    "RPL37": "RPL37A", "RPL41": "RPL41A",
    "RPL1": "RPL1A", "RPL2": "RPL2A",
    "RPL16": "RPL16A", "RPL20": "RPL20A", "RPL33": "RPL33A",
    # RPS — human → yeast
    "RPS6": "RPS6A", "RPS7": "RPS7A", "RPS8": "RPS8A", "RPS9": "RPS9A",
    "RPS10": "RPS10A", "RPS11": "RPS11A",
    "RPS14": "RPS14A", "RPS16": "RPS16A", "RPS17": "RPS17A",
    "RPS18": "RPS18A", "RPS19": "RPS19A",
    "RPS21": "RPS21A", "RPS23": "RPS23A", "RPS24": "RPS24A",
    "RPS25": "RPS25A", "RPS26": "RPS26A",
    "RPS28": "RPS28A", "RPS29": "RPS29A",
    "RPS1": "RPS1A", "RPS4": "RPS4A", "RPS22": "RPS22A",
    "RPS27": "RPS27A",
    # Chaperones
    "HSPA5": "KAR2", "HSPA8": "SSA1", "HSPA1A": "SSA1",
    "HSP90AA1": "HSC82", "HSP90AB1": "HSP82",
    "HSP90B1": "HSP82",
    "HSPD1": "HSP60", "HSPE1": "HSP10",
    "CCT2": "CCT2", "CCT3": "CCT3", "CCT4": "CCT4",
    "CCT5": "CCT5", "CCT6A": "CCT6", "CCT7": "CCT7", "CCT8": "CCT8",
    "TCP1": "CCT1",
    # TCA cycle
    "CS": "CIT1",
    "ACO1": "ACO1", "ACO2": "ACO1",
    "IDH1": "IDP1", "IDH2": "IDP1",
    "OGDH": "KGD1",
    "DLST": "KGD2",
    "SDHA": "SDH1", "SDHB": "SDH2", "SDHC": "SDH3", "SDHD": "SDH4",
    "FH": "FUM1",
    "MDH1": "MDH1", "MDH2": "MDH2",
    # Amino acid biosynthesis
    "ATF4": "GCN4",  # transcription factor (functional ortholog)
    # Ubiquitin-proteasome
    "UBA1": "UBA1",
    "UBB": "UBI4", "UBC": "UBI4",
    "PSMA1": "PRE5", "PSMA2": "PRE8", "PSMA3": "PRE9",
    "PSMA4": "PRE6", "PSMA5": "PUP2", "PSMA6": "PRE5",
    "PSMA7": "PRE4",
    "PSMB1": "PRE3", "PSMB2": "PUP1", "PSMB3": "PUP3",
    "PSMB4": "PRE1", "PSMB5": "PRE2", "PSMB6": "PRE7",
    "PSMB7": "PRE4",
    # RNA processing / splicing
    "SNRPD1": "SMD1", "SNRPD2": "SMD2", "SNRPD3": "SMD3",
    "SNRPB": "SMB1", "SNRPE": "SME1",
    # Histones
    "H3C1": "HHT1", "H4C1": "HHF1",
    "H2AC1": "HTA1", "H2BC1": "HTB1",
    # Actin / cytoskeleton
    "ACTB": "ACT1", "ACTG1": "ACT1",
    "TUBA1A": "TUB1", "TUBA1B": "TUB1",
    "TUBB": "TUB2", "TUBB4B": "TUB2",
    # Metabolism misc
    "FASN": "FAS1",
    "ACLY": "ACB1",
    "HMGCR": "HMG1",
    "IMPDH1": "IMD3", "IMPDH2": "IMD4",
    "PRPS1": "PRS1", "PRPS2": "PRS1",
    # Signalling / kinases
    "CSNK2A1": "CKA1", "CSNK2A2": "CKA2", "CSNK2B": "CKB1",
    "CDK1": "CDC28",
    "MTOR": "TOR1",
    "AKT1": "SCH9",
    "AMPK": "SNF1",  # functional ortholog
}

# Gene name pairs that match by name but are NOT orthologs
NAME_MATCH_BLOCKLIST = {
    "ACT1",   # yeast actin — human ACTA1/ACTB handle separately
    "SEC61A1", "SEC61B", "SEC61G",  # convergent names
}


def download_orthologs(
    species1: str,
    species2: str,
    data_dir: str | Path | None = None,
) -> Path:
    """Download ortholog mapping for a species pair.

    Currently supports human-yeast only.

    Returns:
        Path to the ortholog TSV file.
    """
    species1 = species1.lower()
    species2 = species2.lower()
    pair = tuple(sorted([species1, species2]))

    base = Path(data_dir) if data_dir else DEFAULT_DATA_DIR.parent
    ortho_dir = base / "orthologs"
    ortho_dir.mkdir(parents=True, exist_ok=True)

    if pair == ("human", "yeast"):
        return _download_orthologs_human_yeast(ortho_dir, base / "species")
    else:
        raise ValueError(
            f"Unsupported ortholog pair: {species1}-{species2}. "
            "Supported: human-yeast"
        )


def _download_orthologs_human_yeast(
    ortho_dir: Path,
    species_base: Path,
) -> Path:
    """Download human-yeast orthologs.

    Strategy:
      1. Try BioMart for one-to-one orthologs from Ensembl Compara
      2. Fall back to name-matching + curated renames
    """
    out_path = ortho_dir / "human_yeast.tsv"

    # Try BioMart first
    pairs = _try_biomart_human_yeast(species_base)

    if len(pairs) < 100:
        logger.warning(
            "BioMart returned only %d pairs, falling back to name matching",
            len(pairs),
        )
        pairs = _name_match_human_yeast(species_base)

    # Write TSV
    df = pd.DataFrame(pairs, columns=["human_gene", "yeast_gene"])
    df = df.drop_duplicates().sort_values("human_gene").reset_index(drop=True)
    df.to_csv(out_path, sep="\t", index=False)
    logger.info(
        "Ortholog mapping: %d human-yeast pairs → %s",
        len(df), out_path,
    )
    return out_path


def _try_biomart_human_yeast(species_base: Path) -> list[tuple[str, str]]:
    """Try BioMart query for human-yeast one-to-one orthologs.

    Returns list of (human_ensg, yeast_systematic_name) pairs.
    """
    try:
        logger.info("Querying Ensembl BioMart for human-yeast orthologs...")
        resp = requests.get(
            BIOMART_URL,
            params={"query": BIOMART_HUMAN_YEAST_XML},
            timeout=120,
        )
        resp.raise_for_status()

        text = resp.text.strip()
        if not text or "ERROR" in text[:200]:
            logger.warning("BioMart returned error: %s", text[:200])
            return []

        # Parse TSV response
        lines = text.split("\n")
        if len(lines) < 2:
            return []

        # Load yeast gene map for systematic name lookup
        yeast_dir = species_base / "yeast"
        yeast_map = _load_gene_map(yeast_dir) if yeast_dir.exists() else {}

        pairs = []
        for line in lines[1:]:  # skip header
            parts = line.strip().split("\t")
            if len(parts) < 4:
                continue
            human_ensg = parts[0].strip()
            yeast_gene = parts[3].strip()  # associated gene name
            yeast_ensg = parts[2].strip()

            if not human_ensg or not yeast_gene:
                continue

            # Filter to one2one
            if len(parts) >= 5 and "one2one" not in parts[4]:
                continue

            # Convert yeast common name to systematic name if possible
            if yeast_map:
                sys_name = yeast_map.get(yeast_gene.upper())
                if sys_name:
                    yeast_gene = sys_name

            pairs.append((human_ensg, yeast_gene))

        logger.info("BioMart returned %d ortholog pairs", len(pairs))
        return pairs

    except Exception as exc:
        logger.warning("BioMart query failed: %s", exc)
        return []


def _name_match_human_yeast(species_base: Path) -> list[tuple[str, str]]:
    """Build ortholog table by matching gene names + curated renames.

    Loads gene_id_map.tsv from both species and finds matching gene names.
    Applies curated rename mappings for well-known orthologs.

    Returns list of (human_ensg, yeast_systematic_name) pairs.
    """
    human_dir = species_base / "human"
    yeast_dir = species_base / "yeast"

    if not human_dir.exists() or not yeast_dir.exists():
        raise FileNotFoundError(
            "Both human and yeast data must be downloaded before ortholog mapping. "
            "Run: codonscope download --species human yeast"
        )

    # Load human gene map: HGNC symbol → ENSG
    human_df = pd.read_csv(human_dir / "gene_id_map.tsv", sep="\t")
    human_symbol_to_ensg: dict[str, str] = {}
    for _, row in human_df.iterrows():
        symbol = str(row["common_name"]).strip().upper()
        ensg = str(row["systematic_name"]).strip()
        if symbol and ensg and symbol != "NAN":
            human_symbol_to_ensg[symbol] = ensg

    # Load yeast gene map: common name → systematic name
    yeast_df = pd.read_csv(yeast_dir / "gene_id_map.tsv", sep="\t")
    yeast_name_to_sys: dict[str, str] = {}
    yeast_sys_set: set[str] = set()
    for _, row in yeast_df.iterrows():
        sys_name = str(row["systematic_name"]).strip()
        common = str(row["common_name"]).strip().upper()
        yeast_sys_set.add(sys_name)
        if common and common != "NAN" and common != sys_name:
            yeast_name_to_sys[common] = sys_name
        yeast_name_to_sys[sys_name.upper()] = sys_name

    pairs: list[tuple[str, str]] = []
    used_yeast: set[str] = set()

    # 1. Curated renames (highest priority)
    for human_name, yeast_name in HUMAN_YEAST_CURATED_RENAMES.items():
        human_upper = human_name.upper()
        yeast_upper = yeast_name.upper()

        ensg = human_symbol_to_ensg.get(human_upper)
        if not ensg:
            continue

        yeast_sys = yeast_name_to_sys.get(yeast_upper)
        if not yeast_sys:
            # Try as systematic name directly
            if yeast_name in yeast_sys_set:
                yeast_sys = yeast_name
            else:
                continue

        if yeast_sys not in used_yeast:
            pairs.append((ensg, yeast_sys))
            used_yeast.add(yeast_sys)

    # 2. Direct name matches (case-insensitive)
    for human_symbol, ensg in human_symbol_to_ensg.items():
        if human_symbol in NAME_MATCH_BLOCKLIST:
            continue

        yeast_sys = yeast_name_to_sys.get(human_symbol)
        if yeast_sys and yeast_sys not in used_yeast:
            pairs.append((ensg, yeast_sys))
            used_yeast.add(yeast_sys)

    logger.info(
        "Name matching: %d pairs (%d curated + %d by name)",
        len(pairs),
        sum(1 for h, _ in pairs[:len(HUMAN_YEAST_CURATED_RENAMES)]),
        len(pairs) - sum(1 for h, _ in pairs[:len(HUMAN_YEAST_CURATED_RENAMES)]),
    )
    return pairs


def _load_gene_map(species_dir: Path) -> dict[str, str]:
    """Load gene_id_map.tsv and return {upper_name: systematic_name} lookup."""
    path = species_dir / "gene_id_map.tsv"
    if not path.exists():
        return {}
    df = pd.read_csv(path, sep="\t")
    lookup: dict[str, str] = {}
    for _, row in df.iterrows():
        sys_name = str(row["systematic_name"]).strip()
        common = str(row.get("common_name", "")).strip()
        lookup[sys_name.upper()] = sys_name
        if common and common.upper() != "NAN":
            lookup[common.upper()] = sys_name
    return lookup
