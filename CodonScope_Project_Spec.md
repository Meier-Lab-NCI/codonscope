# CodonScope: Multi-Species Codon Usage Analysis Tool

## Project Specification for Claude Code Implementation

---

## 1. OVERVIEW

CodonScope is a command-line Python tool for analyzing mono-, di-, and tricodon usage patterns in user-supplied gene lists versus genomic backgrounds, across multiple species. It replaces the defunct CUT tool (Doyle et al., Methods 2016) while incorporating modern understanding of translation elongation dynamics from recent ribosome profiling and ML studies (Aguilar Rangel/Frydman, Sci Adv 2024; Cenik et al., bioRxiv 2025).

The tool has six analysis modes, a shared data layer, and a CLI entry point.

---

## 2. ARCHITECTURE

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
│   ├── mode1_composition.py    # Sequence Composition
│   ├── mode2_demand.py         # Translational Demand
│   ├── mode3_profile.py        # Optimality Profile
│   ├── mode4_collision.py      # Collision Potential
│   ├── mode5_disentangle.py    # AA vs Codon Disentanglement
│   └── mode6_crossspecies.py   # Cross-Species Comparison
├── data/
│   ├── download.py         # Setup script to fetch all reference data
│   ├── species/            # Per-species reference data
│   │   ├── human/
│   │   │   ├── cds_sequences.fa.gz
│   │   │   ├── canonical_transcripts.tsv
│   │   │   ├── gene_id_map.tsv
│   │   │   ├── trna_copy_numbers.tsv
│   │   │   ├── wobble_rules.tsv
│   │   │   ├── background_mono.npz
│   │   │   ├── background_di.npz
│   │   │   ├── background_tri.npz
│   │   │   └── gtex_median_tpm.tsv.gz
│   │   ├── mouse/
│   │   ├── yeast/
│   │   └── ... (extendable)
│   └── orthologs/
│       ├── human_mouse.tsv
│       ├── human_yeast.tsv
│       └── ...
├── viz/
│   ├── plots.py            # matplotlib/seaborn output
│   └── report.py           # HTML summary report
└── tests/
    ├── test_positive_controls.py
    └── test_statistics.py
```

---

## 3. SHARED DATA LAYER

### 3.1 CDS Sequences

**Source**: Ensembl BioMart (primary) or NCBI RefSeq
- Human: MANE Select transcripts (one canonical per gene, ~19,400 genes)
- Mouse: Ensembl canonical transcripts
- Yeast (S. cerevisiae): SGD verified ORFs (no isoform problem — one CDS per gene)
- Extensible to zebrafish, fly, worm, rat, E. coli

**Storage**: gzipped FASTA indexed by gene ID. Companion TSV maps gene_symbol → ensembl_gene_id → ensembl_transcript_id → entrez_id (and SGD systematic name for yeast).

**Critical**: All CDSs must be validated as divisible by 3, start with ATG, end with stop codon. Strip stop codon before analysis. Flag and exclude any CDS with internal stop codons or non-ACGT characters.

### 3.2 Gene ID Mapping

Accept any of: HGNC symbol, Ensembl gene ID, Ensembl transcript ID, Entrez ID, yeast systematic name (YAL001C format), yeast common name (TFC3 format).

Use a unified lookup table per species. Log warnings for unmapped IDs. Return mapping summary (N mapped, N unmapped, N ambiguous).

### 3.3 tRNA Data

**Source**: GtRNAdb (http://gtrnadb.ucsc.edu/) — tRNA gene copy numbers per species.

Pre-curate a **wobble decoding rules table** per species encoding:
- Which tRNA anticodon decodes which codons
- Whether each decoding event is Watson-Crick or wobble
- Which modifications are known to affect wobble efficiency

Format:

```
codon  amino_acid  anticodon  trna_gene_copies  decoding_type  modification_dependent
GCA    Ala         UGC        10                 wobble         none
GCG    Ala         UGC        10                 wobble         none
GCU    Ala         AGC        11                 watson_crick   none
GCC    Ala         AGC        11                 wobble         none
AGA    Arg         UCU        11                 watson_crick   mcm5U/mcm5s2U(Trm9)
GAA    Glu         UUC        14                 watson_crick   mcm5s2U(Trm9)
TTG    Leu         CAA        10                 watson_crick   m5C(Trm4)
...
```

### 3.4 Pre-computed Backgrounds

For each species, pre-compute and store:
- Genome-wide mono/di/tricodon frequency vectors (all CDSs, unweighted, one per gene)
- Per-gene frequency vectors (for computing variance)
- Length distribution and GC-content distribution of all CDSs (for matched background sampling)

Store as numpy `.npz` archives. Include bootstrap variance estimates (1000 resamples).

### 3.5 Expression Data

- **Human**: GTEx v8 median TPM per tissue per gene (~54 tissues)
- **Yeast**: Published rich-media expression dataset (Nagalakshmi et al. 2008 or equivalent)
- **Mouse**: ENCODE or Tabula Muris consensus expression
- Allow user-supplied TPM matrix (gene × sample/tissue)

### 3.6 Ortholog Mappings

**Source**: Ensembl Compara one-to-one orthologs. Pre-download for all supported species pairs.

---

## 4. MODE SPECIFICATIONS

### Mode 1: Sequence Composition

**CLI**: `codonscope composition`
**Input**: species, gene list, k-mer size (1/2/3)
**Question**: Does this gene list have unusual codon/dicodon/tricodon frequencies vs genome?

**Algorithm**:
1. Map gene IDs → canonical CDS sequences
2. For each gene, count k-mer frequencies (as proportion of total k-mers in that gene)
3. Compute gene-set mean frequency vector (each gene weighted equally)
4. Compare to pre-computed genome background via Z-score: `Z = (mean_geneset - mean_genome) / SE`
5. SE from bootstrap: resample N genes from genome (where N = gene list size), 10,000 times
6. Apply Benjamini-Hochberg correction across all k-mers
7. Report significant k-mers with Z-scores, observed vs expected frequencies, adjusted p-values

**Options**:
- `--trim-ramp N`: Exclude first N codons (default 0; recommend 50 for ramp-aware analysis)
- `--background matched`: Length-matched and GC-matched background (draw control genes matched on CDS length ±20% and GC ±5%)
- `--background all`: All genome CDSs (default)
- `--min-genes N`: Minimum gene list size (default 10; warn if <30 for tricodon)

**Output**:
- TSV table: k-mer, observed_freq, expected_freq, z_score, p_value, adjusted_p
- Volcano plot (Z-score vs -log10 adjusted p)
- Top enriched/depleted k-mers bar chart

**Automatic diagnostics**:
- Gene list size warning if underpowered for chosen k-mer size
- Length distribution comparison (gene list vs background): KS test
- GC content comparison: KS test
- If either is significant (p<0.01), warn user and suggest `--background matched`

---

### Mode 2: Translational Demand

**CLI**: `codonscope demand`
**Input**: species, gene list, k-mer size, tissue/condition OR user TPM matrix
**Question**: What weighted codon demand does this gene list place on the tRNA pool?

**Algorithm**:
1. Map gene IDs → CDS sequences AND expression values (TPM)
2. For each gene, compute absolute k-mer counts (not frequencies)
3. Weight each gene's counts by: TPM × (CDS_length / 3) = total codon decoding events per cell per unit time
4. Sum weighted counts across gene set → gene set demand vector
5. Same computation for whole transcriptome → genome demand vector
6. Normalize both to proportions → compare as in Mode 1

**Key difference from Mode 1**: A gene at 10,000 TPM with 1,000 codons contributes 10,000,000 decoding events; a gene at 1 TPM with 1,000 codons contributes 1,000. Mode 1 treats them equally.

**Options**:
- `--tissue TISSUE_NAME`: Use GTEx median TPM for specified tissue (human)
- `--expression FILE`: User-supplied TPM matrix
- `--expression-column COL`: Column name/index to use from expression matrix
- `--top-n N`: Only consider top N expressed genes in background

**Output**:
- Same format as Mode 1 but labeled as "demand-weighted"
- Scatter plot of gene-list TPM vs genome TPM distribution
- Table of top demand-contributing genes (which genes dominate the signal?)

---

### Mode 3: Optimality Profile

**CLI**: `codonscope profile`
**Input**: species, gene list
**Question**: Where along transcripts are optimal/non-optimal codons?

**Algorithm**:
1. For each gene, compute per-codon optimality score using:
   - **Classical tAI**: geometric mean of tRNA adaptation weights
   - **Wobble-aware tAI (wtAI)**: penalize wobble-decoded codons (default penalty 0.5). Based on Frydman finding that wobble decoding is slower than Watson-Crick regardless of tRNA abundance.
2. Sliding window (default 10 codons) along each CDS → smoothed optimality profile
3. Normalize each gene to % CDS length (0-100%)
4. Compute metagene average across gene set
5. Compare to metagene average of genome (or matched background)

**Options**:
- `--window N`: Sliding window size in codons (default 10)
- `--wobble-penalty FLOAT`: Wobble decoding penalty factor (default 0.5)
- `--show-ramp`: Highlight first 50 codons separately
- `--compare-to FILE`: Second gene list for head-to-head comparison

**Output**:
- Metagene optimality plot (gene set vs genome)
- Ramp region analysis: mean optimality of first 50 codons vs rest
- Per-gene optimality heatmap (genes as rows, position as columns, color = optimality)
- Distribution of per-gene mean tAI and wtAI

---

### Mode 4: Collision Potential

**CLI**: `codonscope collision`
**Input**: species, gene list
**Question**: Does this gene list have unusual fast→slow transitions (collision-prone)?

**Algorithm**:
1. Classify each codon as "fast" or "slow" using wtAI (threshold: median genome wtAI)
2. For each gene, count dicodon transitions: FF, FS, SF, SS
3. Compute transition matrix proportions for gene set
4. Compare to genome transition matrix
5. For each of 4,096 actual dicodons, classify as FF/FS/SF/SS; report enrichment of FS dicodons

**Biological rationale** (Aguilar Rangel et al. 2024): Disomes arise at junctions where trailing ribosome (on optimal codons) catches leading ribosome (on non-optimal codons). FS transitions are collision-prone.

**Options**:
- `--threshold PERCENTILE`: Percentile for fast/slow classification (default 50)
- `--classify-by {tai,wtai,custom}`: Optimality score for classification

**Output**:
- 2×2 transition matrix (gene set vs genome), chi-squared test
- List of FS dicodons enriched in gene set
- Collision potential score: ratio of observed FS to expected
- Positional analysis: where in CDS do FS transitions cluster?

---

### Mode 5: Amino Acid vs Codon Disentanglement

**CLI**: `codonscope disentangle`
**Input**: species, gene list
**Question**: Is codon bias driven by protein composition or synonymous choice?

**Algorithm** (two-layer decomposition):

1. **Layer 1 — Amino acid composition**: Compare gene set amino acid frequencies to genome. Report over/under-represented AAs.

2. **Layer 2 — RSCU**: For each amino acid family, compute Relative Synonymous Codon Usage within gene set vs genome. RSCU = observed codon freq / expected if all synonyms used equally.

3. **Attribution**: For each codon enriched in Mode 1, classify:
   - "AA-driven": amino acid enriched; synonymous usage normal
   - "Synonymous-driven": amino acid normal; this synonym preferred
   - "Both": both effects

4. **Classify synonymous deviations** by correlation with:
   - tRNA gene copy number → "tRNA supply driven"
   - GC3 content → "mutational bias driven"
   - Wobble avoidance → "wobble avoidance driven"

**Output**:
- Two-panel figure: (A) AA deviation, (B) within-AA RSCU deviation
- Attribution table for all significant codons
- Summary: "X% of signal is AA-driven, Y% is synonymous-driven"

---

### Mode 6: Cross-Species Comparison

**CLI**: `codonscope compare`
**Input**: two species, gene list
**Question**: Is codon usage bias conserved across species for these genes?

**Algorithm**:
1. Map gene list to orthologs via Ensembl Compara
2. For each ortholog pair, compute per-gene RSCU in both species
3. Compute RSCU correlation between species per gene
4. Compare gene-set correlation distribution to genome-wide
5. For divergent genes, test whether divergence tracks tRNA pool differences

**Options**:
- `--species1 human --species2 mouse`
- `--gene-id-species {1,2}`: Which species input IDs are from

**Output**:
- Distribution of per-gene cross-species RSCU correlations (gene set vs genome)
- Most divergent orthologs with annotation
- Scatter of species1 vs species2 RSCU for top codons
- Summary of tRNA pool divergence explanation

---

## 5. CLI INTERFACE

```bash
# Setup (run once)
codonscope download --species human mouse yeast

# Mode 1
codonscope composition --species yeast --genes genelist.txt --kmer 2 --background matched

# Mode 2
codonscope demand --species human --genes genelist.txt --tissue liver --kmer 1

# Mode 3
codonscope profile --species yeast --genes genelist.txt --wobble-penalty 0.5

# Mode 4
codonscope collision --species yeast --genes genelist.txt

# Mode 5
codonscope disentangle --species human --genes genelist.txt

# Mode 6
codonscope compare --species1 human --species2 mouse --genes genelist.txt

# Full report (all modes)
codonscope report --species yeast --genes genelist.txt --output report.html
```

**Gene list format**: One gene ID per line, or comma-separated. Auto-detect ID type.

**Output directory**: `--output-dir DIR` (default: `./codonscope_output/`).

---

## 6. KEY IMPLEMENTATION DETAILS

### 6.1 Statistics

- **Z-scores**: Bootstrap-based. Resample N genes from background 10,000 times. Z = (observed - mean_bootstrap) / std_bootstrap. More robust than analytic Z-scores because it accounts for non-independence of codons within genes.
- **Multiple testing**: Benjamini-Hochberg across all k-mers tested within each mode.
- **Power warnings**: Tricodons (262,144): warn if gene list < 100. Dicodons (4,096): warn if < 30.
- **Effect size**: Report Cohen's d alongside Z-scores and p-values.

### 6.2 Sequence Handling

- All sequences stored 5'→3', sense strand
- Strip start ATG and stop codon before counting
- Dicodons: sliding window of 2 (positions 1-2, 2-3, 3-4...), NOT non-overlapping
- Tricodons: sliding window of 3
- Frame is always codon-aligned (no frame shifting)

### 6.3 Performance

- Pre-computed backgrounds are key optimization
- 500-gene list vs human genome: Mode 1 < 30 seconds, Mode 2 < 1 minute
- Bootstrap is bottleneck; use numpy vectorized operations
- Cache bootstrap distributions for common gene list sizes

### 6.4 Dependencies

```
python >= 3.9
numpy
scipy
pandas
matplotlib
seaborn
requests
tqdm
```

No heavy bioinformatics dependencies (no BioPython, no pysam). Lightweight.

---

## 7. POSITIVE CONTROLS

### Control 1: Yeast Ribosomal Proteins (~128 genes)

**Source**: SGD GO:0003735 "structural constituent of ribosome"
**Expected**:
- Mode 1: Strongest codon bias of any functional class (high CAI)
- Mode 3: High optimality throughout with 5' ramp dip in first 30-50 codons
- Mode 5: Primarily synonymous-driven (diverse AA composition, uniform optimal codon preference)

### Control 2: Yeast Gcn4-Regulated Transcripts (~80 genes)

**Source**: Natarajan et al. 2001 (Mol Cell Biol) Gcn4 microarray targets
**Expected**:
- Mode 1 (dicodon): AGA-GAA enrichment Z > 4, p < 0.00001 (CUT paper flagship result)
- Mode 1 (dicodon): AGA-AGA Z~4.8, GAA-AGA Z~4.1, GAA-GAA Z~3.2
- Mode 1 (monocodon): AGA, GAA, AAG enrichment (Trm9-dependent tRNAs)
- Mode 5: Partly synonymous-driven (AGA preferred over other Arg codons)

### Control 3: Yeast Trm4-Dependent Oxidative Stress Genes

**Source**: Chan et al. 2012 (Nat Commun) — H₂O₂ stress, Trm4-dependent protein upregulation
**Expected**:
- Mode 1: TTG (Leu) codon enrichment
- Mode 2 (stress-weighted): TTG enrichment amplified
- Mode 5: TTG enrichment is synonymous-driven (Leu is 6-fold degenerate)
- Key gene: RPL22A (TTG-enriched) vs RPL22B (TTG-unenriched paralog)

### Control 4: Human 5'TOP mRNA Genes

**Source**: Thoreen et al. 2012 (Nature) or Philippe et al. 2020 (PNAS)
**Expected**:
- Mode 1: Strong codon optimization
- Mode 2: Dominant translational demand in any tissue
- Mode 3: Distinctive 5' optimality profile
- Mode 6 (vs mouse): High conservation (Cenik et al. TE correlation r~0.9)

### Control 5: Yeast YEF3 (Single Gene Sanity Check)

**Source**: YLR249W (translation elongation factor eEF3)
**Expected**:
- Mode 1: 22 of 25 most overused dicodons contain AGA or GAA
- Per-gene Z-scores: AGA-AGA Z=4.8, GAA-AGA Z=4.1, AGA-GAA Z=3.3, GAA-GAA Z=3.2
- Must match published CUT values

### Control 6: Human Membrane Proteins (Negative Control for Mode 5)

**Source**: UniProt "Transmembrane" keyword, human; or GO:0016021
**Expected**:
- Mode 5: Codon biases are largely AA-driven (hydrophobic AA enrichment: Leu, Ile, Val, Phe)
- Mode 1: Will show codon enrichments, but Mode 5 reveals these are AA composition artifacts
- Demonstrates that not all codon bias = tRNA selection

### Control 7: Human Housekeeping vs Tissue-Specific (Mode 2 Validation)

**Source**: Eisenberg & Levanon 2013 (housekeeping); GTEx tau > 0.9 (tissue-specific)
**Expected**:
- Mode 2: Housekeeping genes dominate demand regardless of tissue
- Mode 2: Tissue-specific genes show dramatically different demand across tissues (liver vs brain vs testis)

### Control 8: Cross-Species Ribosomal Proteins (Mode 6 Validation)

**Source**: One-to-one orthologs of ribosomal proteins (human-mouse, human-yeast)
**Expected**:
- Mode 6: Conserved strong optimization in both species, but preferred codons differ due to tRNA pool divergence
- RNA-binding proteins (GO:0003723): more divergent bias per Cenik et al.

---

## 8. TESTING STRATEGY

```python
# tests/test_positive_controls.py

def test_gcn4_aga_gaa_dicodon():
    """Gcn4 targets must show AGA-GAA dicodon enrichment Z > 3, adj_p < 0.001"""

def test_ribosomal_protein_optimality():
    """Yeast ribosomal proteins must have mean tAI > 90th percentile of genome"""

def test_yef3_dicodon_zscores():
    """YEF3 AGA-AGA Z-score must be > 4.0 (published: 4.8)"""

def test_trm4_ttg_enrichment():
    """Trm4-dependent stress genes must show TTG enrichment Z > 2"""

def test_membrane_aa_driven():
    """Membrane protein codon bias must be >70% AA-driven in Mode 5"""

def test_cross_species_ribosomal_conservation():
    """Ribosomal protein orthologs must show RSCU correlation > 0.5"""

def test_demand_tissue_variation():
    """Mode 2 demand profile must differ significantly between liver and brain"""

def test_ramp_detection():
    """Ribosomal proteins must show lower optimality in first 50 codons vs body"""
```

---

## 9. DATA DOWNLOAD SCRIPT

```bash
codonscope download --species human mouse yeast
```

Steps:
1. Create `~/.codonscope/data/` directory structure
2. Download from Ensembl BioMart: CDS sequences, canonical transcript IDs, gene ID mappings, ortholog tables
3. Download from GtRNAdb: tRNA gene copy numbers
4. Download from GTEx portal: median TPM matrix (human only)
5. Download from SGD: verified ORF sequences (yeast)
6. Pre-compute genome-wide background distributions (mono/di/tricodon)
7. Validate all downloads
8. Report total disk usage and time

Estimated: ~3 GB download, ~30 min for background computation.

---

## 10. KNOWN PITFALLS & DESIGN DECISIONS

These are documented here so the implementer understands *why* certain choices were made.

1. **Isoform handling**: Use one canonical transcript per gene (MANE Select for human, Ensembl canonical for mouse, SGD ORF for yeast). This avoids the inflation problem where genes with many isoforms dominate statistics.

2. **Background definition matters**: "Genome average" can mean all CDSs equally weighted, expression-weighted, or top-N expressed. Z-scores change dramatically. The tool offers both `--background all` and `--background matched`.

3. **Gene length bias**: Longer genes dominate pooled frequencies. We compute frequencies per-gene then average (treating genes equally), not pool all codons.

4. **Combinatorial explosion**: 4,096 dicodons, 262,144 tricodons. Power warnings are essential.

5. **Wobble ≠ rare**: A codon decoded by an abundant tRNA through wobble can be SLOWER than a codon decoded by a rare tRNA via Watson-Crick. Standard tAI misses this.

6. **Amino acid confound**: Codon enrichments may reflect protein composition, not synonymous choice. Mode 5 explicitly separates these.

7. **Translation efficiency ≠ protein output**: High ribosome density can reflect stalling (high TE, low throughput) or efficient initiation (high TE, high throughput). Mode 2 weights by RNA abundance (TPM), not TE, to avoid this confound.

8. **Cross-species tRNA pools differ**: The same codon can be "optimal" in yeast and "non-optimal" in human. Mode 6 uses species-specific tRNA data.

9. **Ramp region**: First ~50 codons have systematically different codon usage. Can dominate short-gene signals. Option to trim.

10. **Position matters for collisions**: A dicodon's biological impact depends on whether it's a fast→slow transition (collision-prone) or slow→fast (collision-resolving). Mode 4 captures this directionality.

---

## 11. FUTURE EXTENSIONS (NOT IN V1)

- Web interface (Streamlit or Flask)
- tRNA modification simulation mode
- Disome-seq collision map integration
- Nascent chain / exit tunnel motif flagging (Frydman tripeptide clusters)
- Cenik RiboBase TE values as alternative weighting
- Non-model organisms via user-supplied genome + tRNA annotations
- Batch mode for multiple gene lists
- Gene set enrichment-style analysis (ranked gene list input)
