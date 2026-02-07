# CodonScope

Multi-species codon usage analysis tool for gene lists. Replaces the defunct CUT tool (Doyle et al., Methods 2016). Analyzes mono-, di-, and tricodon patterns in user gene lists vs genomic backgrounds.

## Full spec

Read `CodonScope_Project_Spec.md` for the complete project specification including all 6 analysis modes, positive controls, and architecture.

## Current status

Building in chunks. See below for what exists and what's next.

### Chunk order
1. **Data layer + yeast** ✅ DONE
2. Core counting engine + statistics ✅ DONE
3. Mode 1 (composition) + CLI ✅ DONE
4. Mode 5 (AA vs codon disentanglement) ← NEXT
5. Mode 3 (optimality profile) + Mode 4 (collision potential)
6. Mode 2 (translational demand)
7. Mode 6 (cross-species comparison)
8. HTML report generation

### What's built
- [x] Chunk 1: Data layer + yeast (6,685 ORFs, all 23 tests passing)
  - `codonscope/core/codons.py` — k-mer counting (mono/di/tri), fully implemented
  - `codonscope/core/sequences.py` — SequenceDB with ID resolution
  - `codonscope/data/download.py` — SGD CDS, tRNA, wobble rules, backgrounds
  - `tests/test_chunk1.py` — 23 tests
- [x] Chunk 2: Statistics engine (52 total tests passing)
  - `codonscope/core/statistics.py` — bootstrap Z-scores, BH correction, Cohen's d, power warnings, KS diagnostics, full pipeline
  - `tests/test_statistics.py` — 29 tests incl. yeast positive controls
- [x] Chunk 3: Mode 1 (composition) + CLI (71 total tests passing)
  - `codonscope/modes/mode1_composition.py` — run_composition() with matched background, diagnostics, plots
  - `codonscope/cli.py` — argparse CLI with download + composition subcommands
  - `tests/test_mode1.py` — 19 tests incl. ribosomal protein positive controls

## Architecture

```
codonscope/
├── cli.py                  # CLI entry point (argparse)
├── core/
│   ├── sequences.py        # CDS retrieval, isoform selection, ID mapping
│   ├── codons.py           # k-mer counting engine (mono/di/tri)
│   ├── statistics.py       # Z-scores, permutation tests, BH correction
│   ├── optimality.py       # tAI, wobble-aware scoring, codon classification
│   └── orthologs.py        # Cross-species ortholog mapping
├── modes/
│   ├── mode1_composition.py
│   ├── mode2_demand.py
│   ├── mode3_profile.py
│   ├── mode4_collision.py
│   ├── mode5_disentangle.py
│   └── mode6_crossspecies.py
├── data/
│   ├── download.py         # Setup script to fetch all reference data
│   └── species/            # Per-species reference data (downloaded, not in repo)
├── viz/
│   ├── plots.py
│   └── report.py
└── tests/
    ├── test_positive_controls.py
    └── test_statistics.py
```

## Key dependencies

```
python >= 3.9
numpy, scipy, pandas, matplotlib, seaborn, requests, tqdm
```

No BioPython, no pysam. Keep it lightweight.

## Design decisions (read before coding)

1. **One canonical transcript per gene.** Human = MANE Select. Mouse = Ensembl canonical. Yeast = SGD verified ORF. Never use all isoforms — inflates statistics.
2. **Frequencies computed per-gene then averaged.** Not pooled across genes. This treats genes equally regardless of CDS length.
3. **Dicodons/tricodons use sliding windows.** Positions 1-2, 2-3, 3-4... NOT non-overlapping pairs.
4. **Strip stop codon before counting.** Validate CDSs: divisible by 3, starts with ATG, ends with stop.
5. **Bootstrap Z-scores, not analytic.** Resample N genes from genome 10,000 times for SE estimation.
6. **Benjamini-Hochberg correction** across all k-mers within each mode.

## Positive controls (use for testing)

- **Yeast Gcn4 targets (~80 genes)**: Must show AGA-GAA dicodon Z > 3
- **Yeast ribosomal proteins (~128 genes)**: Must show highest codon bias
- **Yeast YEF3 (YLR249W)**: AGA-AGA Z=4.8, GAA-AGA Z=4.1 (published values)
- **Human membrane proteins**: Mode 5 must show bias is AA-driven, not synonymous
