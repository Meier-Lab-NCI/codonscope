"""Data download and pre-computation infrastructure for CodonScope.

Currently supports yeast (S. cerevisiae). Designed for easy extension
to human and mouse.
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

# ── SGD download URL ──────────────────────────────────────────────────────────
SGD_CDS_URL = (
    "https://downloads.yeastgenome.org/sequence/S288C_reference/"
    "orf_dna/orf_coding_all.fasta.gz"
)

# ── GtRNAdb download URL ─────────────────────────────────────────────────────
GTRNADB_YEAST_URL = (
    "http://gtrnadb.ucsc.edu/genomes/eukaryota/Scere3/sacCer3-mature-tRNAs.fa"
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
    """
    # 1. Download MANE summary + Ensembl CDS
    sequences, gene_map = _download_human_cds(species_dir)

    # 2. tRNA gene copy numbers
    _download_human_trna(species_dir)

    # 3. Wobble rules (curated)
    _save_wobble_rules(species_dir, wobble_rules=HUMAN_WOBBLE_RULES)

    # 4. Pre-compute backgrounds
    _compute_backgrounds(species_dir, sequences)

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
