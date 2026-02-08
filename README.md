# CodonScope

Multi-species codon usage analysis tool for gene lists. Replaces the defunct CUT tool (Doyle et al., *Methods* 2016). Analyzes mono-, di-, and tricodon patterns in user-provided gene lists versus genomic backgrounds, with expression-weighted demand analysis, translational optimality profiling, ribosome collision potential, amino acid vs. synonymous codon disentanglement, and cross-species comparison.

## Supported Species

| Species | Genome | Genes | Source |
|---------|--------|-------|--------|
| Yeast (*S. cerevisiae*) | S288C | 6,685 ORFs | SGD |
| Human | GRCh38 | 19,229 CDS | MANE Select v1.5 + Ensembl |
| Mouse (*M. musculus*) | GRCm39 | ~21,500 CDS | Ensembl |

## Installation

### Dependencies

```
python >= 3.9
numpy, scipy, pandas, matplotlib, requests, tqdm
```

No BioPython or pysam required.

```bash
pip install numpy scipy pandas matplotlib requests tqdm
```

### Setup

Clone the repository and download reference data:

```bash
git clone <repo-url>
cd Codon_analysis

# Download species data (one-time, ~5 min per species)
python3 -c "from codonscope.data.download import download; download('yeast')"
python3 -c "from codonscope.data.download import download; download('human')"
python3 -c "from codonscope.data.download import download; download('mouse')"
```

Data is stored in `~/.codonscope/data/species/{yeast,human,mouse}/`.

## Quick Start

```bash
# 1. Download reference data (if not already done)
python3 -c "from codonscope.data.download import download; download('yeast')"

# 2. Analyze a gene list (HTML report with all modes)
python3 -m codonscope.cli report \
    --species yeast \
    --genes examples/yeast_rp_genes.txt \
    --output rp_report.html

# 3. Or run individual modes
python3 -m codonscope.cli composition --species yeast --genes examples/yeast_rp_genes.txt --kmer 1
python3 -m codonscope.cli demand --species yeast --genes examples/yeast_rp_genes.txt
python3 -m codonscope.cli profile --species yeast --genes examples/yeast_rp_genes.txt
```

## Data Sources

All external data sources used by CodonScope, with exact URLs, versions, and access methods.

### CDS Sequences

| Species | Source | URL | Version | Method |
|---------|--------|-----|---------|--------|
| Yeast | SGD | `https://downloads.yeastgenome.org/sequence/S288C_reference/orf_dna/orf_coding_all.fasta.gz` | S288C reference | All verified ORFs; mitochondrial (Q0*) excluded; validated: ACGT only, divisible by 3, starts ATG, ends stop codon, no internal stops; stop codon stripped |
| Human | NCBI MANE Select + Ensembl | Summary: `https://ftp.ncbi.nlm.nih.gov/refseq/MANE/MANE_human/current/MANE.GRCh38.v1.5.summary.txt.gz`; CDS: `https://ftp.ensembl.org/pub/current_fasta/homo_sapiens/cds/Homo_sapiens.GRCh38.cds.all.fa.gz` | MANE Select v1.5, GRCh38 | MANE summary provides gene ID mapping (ENSG, HGNC, ENST, RefSeq, Entrez); Ensembl CDS FASTA filtered to MANE Select transcripts only; same validation as yeast |
| Mouse | Ensembl | `https://ftp.ensembl.org/pub/current_fasta/mus_musculus/cds/Mus_musculus.GRCm39.cds.all.fa.gz` | GRCm39 | All protein-coding transcripts; for each gene (ENSMUSG), keep longest valid CDS; same validation as yeast |

### tRNA Gene Copy Numbers

| Species | Source | URL | Fallback |
|---------|--------|-----|----------|
| Yeast | GtRNAdb sacCer3 | `http://gtrnadb.ucsc.edu/genomes/eukaryota/Scere3/sacCer3-mature-tRNAs.fa` | Hardcoded from Chan & Lowe 2016 / Phizicky & Hopper 2010 (~275 genes, 42 anticodons) |
| Human | GtRNAdb hg38 | `http://gtrnadb.ucsc.edu/genomes/eukaryota/Hsapi38/hg38-mature-tRNAs.fa` | Hardcoded from Chan & Lowe 2016 (~430 genes, 46 anticodons) |
| Mouse | GtRNAdb mm39 | `http://gtrnadb.ucsc.edu/genomes/eukaryota/Mmusc39/mm39-mature-tRNAs.fa` | Hardcoded from GtRNAdb mm39 (~430 genes, 46 anticodons) |

**Note**: The GtRNAdb FASTA parser currently returns 0 parsed anticodons for all species (header format may have changed). All species use hardcoded fallback tables, which are accurate and well-established.

### Wobble Decoding Rules

| Species | Source | Notes |
|---------|--------|-------|
| Yeast | Johansson & Bystrom 2005; Agris et al. 2017 | Standard eukaryotic wobble rules with yeast-specific modifications (Trm4/m5C, Trm9/mcm5s2U, Tad1/I34) |
| Human | Same base rules, human enzyme names | ALKBH8 (human Trm9), NSUN2 (human Trm4), ADAT1 (human Tad1) |
| Mouse | Reuses human rules | Same mammalian tRNA modification machinery (ALKBH8, NSUN2, ADAT1) |

### Expression Data

| Species | Source | URL | Format |
|---------|--------|-----|--------|
| Yeast | Literature estimates | — | Hardcoded TPM: RP genes 3000, glycolytic enzymes 500-5000, median gene 15, dubious ORFs 0.5 |
| Human | GTEx v8 | `https://storage.googleapis.com/adult-gtex/bulk-gex/v8/rna-seq/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_median_tpm.gct.gz` | ~56K genes x 54 tissues, median TPM per tissue |
| Human | DepMap/CCLE | API: `https://depmap.org/portal/api/download/files` | Cell line expression (log2(TPM+1), converted to TPM); ~1900 cell lines |
| Mouse | Literature estimates | — | Hardcoded TPM: RP genes 3000, housekeeping (Actb 5000, Gapdh 3000), median gene 15 |

### Orthologs

| Pair | Source | Method | Pairs |
|------|--------|--------|-------|
| Human-Yeast | Name matching + curated renames | Gene name matching between species + 150 curated ortholog mappings (RP paralogs: human RPL11 -> yeast RPL11A) | 830 |

### Gene ID Resolution (Human)

| Source | URL | Purpose |
|--------|-----|---------|
| HGNC Complete Set | `https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt` | Alias and previous symbol lookup for gene ID resolution |

## Analysis Methods

### Mode 1: Codon Composition

**Monocodon composition**: For each gene, compute the frequency of each of the 61 sense codons (stop codons excluded). Sliding window of size 1 over the CDS (after stop codon removal). Frequency = count / total_codons. Gene-set frequency = mean across all genes in the set. Genome background = mean across all genes in the species. Bootstrap SE: resample N genes (where N = gene set size) from the genome 10,000 times, compute mean frequency each time, take SD of the 10,000 means as SE. Z = (observed - expected) / SE. Two-sided p-value from Z. Benjamini-Hochberg correction across all 61 codons.

**Dicodon composition**: Same method with k=2. Sliding window: positions 1-2, 2-3, 3-4, etc. (overlapping). 61^2 = 3,721 possible sense dicodons.

**Tricodon composition**: Same method with k=3. 61^3 = 226,981 possible tricodons. Uses analytic SE (std / sqrt(n_genes)) instead of bootstrap because the per-gene matrix would be too large to store.

**Matched background** (`--background matched`): Instead of sampling from the full genome, sample from genes matched on CDS length (within 20%) and GC content (within 5%). This controls for compositional bias.

**KS diagnostics**: Kolmogorov-Smirnov tests compare gene-set CDS length and GC content distributions to the genome background. Significant differences (p < 0.01) trigger a warning suggesting matched background.

### Mode 2: Translational Demand

For each gene, demand weight = TPM x n_codons (expression level x CDS length in codons). The demand vector is the weighted mean of per-gene codon frequencies: demand = sum(weight_i x freq_i) / sum(weight_i). Gene-set demand is compared to genome-wide demand using weighted bootstrap Z-scores.

**Weighted bootstrap**: For each of 10,000 iterations, sample N genes (where N = gene set size) from the genome with their expression weights. Compute the weighted demand vector for the sample. Z = (observed_demand - mean_bootstrap) / SD_bootstrap. BH correction.

**Expression sources**: Yeast uses hardcoded rich-media estimates. Human uses GTEx v8 tissue-specific TPM (54 tissues) or CCLE cell line expression. Mouse uses hardcoded estimates. Custom expression files (TSV with gene_id and tpm columns) are also supported.

### Mode 3: Optimality Profile (Metagene)

Per-position translational optimality scoring using the weighted tRNA Adaptation Index (wtAI).

**wtAI computation**: For each codon, tAI = tRNA_gene_copies / max(tRNA_gene_copies), with a pseudocount for codons with 0 tRNA genes. wtAI applies a wobble penalty (default 0.5) to wobble-decoded codons: wtAI = tAI x wobble_penalty for wobble codons, wtAI = tAI for Watson-Crick codons. Values normalized to [0, 1].

**Metagene profile**: Each CDS is divided into 100 positional bins. Per-position wtAI is computed for each gene, then averaged across the gene set. The metagene profile is compared to the genome-wide average.

**Ramp analysis**: Compare mean wtAI of the first N codons (default 50, the "ramp") to the body (codons 51+). The ramp ratio = mean_ramp / mean_body. Ribosomal protein genes in yeast show a visible ramp (lower optimality at 5' end).

### Mode 4: Collision Potential

Classifies codons as "fast" or "slow" using the median wtAI as threshold. Counts FF (fast-fast), FS (fast-slow), SF (slow-fast), and SS (slow-slow) dicodon transitions.

**FS enrichment ratio**: FS_observed / FS_expected, where FS_expected is computed from individual codon frequencies assuming independence. Ratio > 1 indicates enrichment for ribosome collision sites.

**FS/SF ratio**: Compares forward (fast-to-slow) vs. reverse (slow-to-fast) transitions. High FS/SF ratio suggests directional slowdown patterns.

**Chi-squared test**: Compares the gene-set FF/FS/SF/SS distribution to the genome-wide distribution.

### Mode 5: AA vs. Codon Disentanglement

Two-layer decomposition of codon frequency deviations:

1. **Amino acid composition layer**: Deviations explained by over/under-representation of amino acids (e.g., gene set uses more Lys than genome average)
2. **Synonymous codon choice layer (RSCU)**: Deviations within amino acid families (e.g., among Lys codons, AAG preferred over AAA)

**Attribution**: Each significantly deviating codon is classified as:
- **AA-driven**: Deviation explained primarily by amino acid composition
- **Synonymous-driven**: Deviation explained by RSCU within the amino acid family
- **Both**: Both layers contribute

**Synonymous driver classification** (heuristic): Enriched codons with Watson-Crick decoding and high tRNA gene copies are classified as "tRNA supply" driven. Enriched codons that avoid wobble decoding are classified as "wobble avoidance". GC3-biased codons are classified as "GC3 bias".

### Mode 6: Cross-Species Comparison

Per-gene RSCU (Relative Synonymous Codon Usage) correlation between ortholog pairs.

**RSCU**: For each codon within an amino acid family, RSCU = observed_frequency / expected_frequency, where expected = 1/n_synonyms. Only multi-synonym amino acids are included (Met and Trp excluded — single codon each).

**Per-gene correlation**: For each ortholog pair (e.g., human RPL11 vs. yeast RPL11A), compute Pearson r between the RSCU vectors. This measures whether the two species use similar synonymous codons for the orthologous gene.

**Gene-set test**: Compare the distribution of per-gene RSCU correlations for the gene set to the genome-wide distribution. Bootstrap Z-test (10K resamples) and Mann-Whitney U test.

**Divergent analysis**: Identify genes with low RSCU correlation and determine which amino acid families have the most divergent preferred codons (tracking differences in tRNA pools between species).

## CLI Reference

### Subcommands

```bash
python3 -m codonscope.cli <subcommand> [options]
```

| Subcommand | Description |
|------------|-------------|
| `download` | Download reference data for a species |
| `report` | Generate comprehensive HTML report (all modes) |
| `composition` | Mode 1: Codon composition analysis |
| `demand` | Mode 2: Translational demand analysis |
| `profile` | Mode 3: Optimality profile (metagene + ramp) |
| `collision` | Mode 4: Ribosome collision potential |
| `disentangle` | Mode 5: AA vs. synonymous codon disentanglement |
| `compare` | Mode 6: Cross-species RSCU comparison |

### Common Options

| Option | Description | Default |
|--------|-------------|---------|
| `--species` | Species name | required |
| `--genes` | Path to gene list file | required |
| `--output` | Output file/directory | stdout |
| `--kmer` | k-mer size (1/2/3) for composition/demand | 1 |
| `--background` | Background type: `all` or `matched` | `all` |
| `--n-bootstrap` | Bootstrap iterations | 10000 |
| `--seed` | Random seed | None |

### Mode-Specific Options

| Option | Subcommand | Description |
|--------|-----------|-------------|
| `--tissue` | `demand` | GTEx tissue name (human) |
| `--cell-line` | `demand` | CCLE cell line name (human) |
| `--expression` | `demand` | Custom expression file (TSV) |
| `--top-n` | `demand` | Use only top-N expressed genes |
| `--method` | `profile` | Scoring method: `tai` or `wtai` |
| `--species1` | `compare` | First species |
| `--species2` | `compare` / `report` | Second species (for cross-species) |
| `--from-species` | `compare` | Species of input gene list |

### Gene List Format

One gene per line. Supports:
- Common names: `RPL3`, `GAPDH`, `Actb`
- Systematic names: `YOR063W`, `ENSG00000142541`, `ENSMUSG00000060036`
- Ensembl transcript IDs: `ENST00000341065`, `ENSMUST00000000003`
- Entrez IDs (human): `6122`
- Comments: Lines starting with `#` are ignored
- Comma-separated and tab-separated formats also accepted

## Example Workflows

### Yeast ribosomal protein analysis

```bash
# Full report
python3 -m codonscope.cli report \
    --species yeast \
    --genes examples/yeast_rp_genes.txt \
    --output yeast_rp_report.html

# Individual modes
python3 -m codonscope.cli composition --species yeast --genes examples/yeast_rp_genes.txt --kmer 1
python3 -m codonscope.cli composition --species yeast --genes examples/yeast_rp_genes.txt --kmer 2
python3 -m codonscope.cli demand --species yeast --genes examples/yeast_rp_genes.txt
python3 -m codonscope.cli profile --species yeast --genes examples/yeast_rp_genes.txt
python3 -m codonscope.cli collision --species yeast --genes examples/yeast_rp_genes.txt
python3 -m codonscope.cli disentangle --species yeast --genes examples/yeast_rp_genes.txt
```

### Human tissue-specific analysis

```bash
# Liver-specific demand
python3 -m codonscope.cli demand \
    --species human \
    --genes examples/human_rp_genes.txt \
    --tissue liver

# HEK293T cell line expression (uses CCLE or GTEx Kidney proxy)
python3 -m codonscope.cli demand \
    --species human \
    --genes examples/human_rp_genes.txt \
    --cell-line HEK293T
```

### Mouse analysis

```bash
python3 -m codonscope.cli report \
    --species mouse \
    --genes examples/mouse_rp_genes.txt \
    --output mouse_rp_report.html
```

### Cross-species comparison

```bash
# Download orthologs first
python3 -c "from codonscope.data.download import download_orthologs; download_orthologs('human', 'yeast')"

# Compare yeast gene set to human orthologs
python3 -m codonscope.cli compare \
    --species1 yeast --species2 human \
    --genes examples/yeast_rp_genes.txt

# Compare human gene set to yeast orthologs
python3 -m codonscope.cli compare \
    --species1 yeast --species2 human \
    --genes examples/human_rp_genes.txt \
    --from-species human
```

### Custom expression data

```bash
# User-supplied expression file (TSV with gene_id and tpm columns)
python3 -m codonscope.cli demand \
    --species yeast \
    --genes my_genes.txt \
    --expression my_expression.tsv
```

## Output Files

### HTML Report (`report` subcommand)

Self-contained HTML file with inline CSS and base64-embedded matplotlib plots. Contains:
- Gene summary (mapped/unmapped genes, CDS lengths)
- Mode 1: Volcano plots and bar charts for mono/dicodon composition
- Mode 5: Attribution table (AA-driven vs. synonymous-driven)
- Mode 3: Metagene optimality profile + ramp analysis
- Mode 4: Collision potential (FF/FS/SF/SS transitions)
- Mode 2: Demand-weighted codon analysis
- Mode 6: Cross-species RSCU correlation (if --species2 provided)

### TSV Outputs (individual modes)

Each mode writes TSV files to the output directory:

| File | Contents |
|------|----------|
| `composition_mono.tsv` | kmer, freq_geneset, freq_genome, z_score, p_value, adjusted_p, amino_acid |
| `composition_di.tsv` | Same columns for dicodons |
| `demand_monocodons_*.tsv` | demand_geneset, demand_genome, z_score, p_value, adjusted_p |
| `demand_top_genes_*.tsv` | gene, gene_name, tpm, n_codons, demand_weight, demand_fraction |
| `profile_metagene.tsv` | position, wtai_geneset, wtai_genome |
| `collision_transitions.tsv` | FF, FS, SF, SS counts and proportions |
| `disentangle_attribution.tsv` | codon, aa_deviation, rscu_deviation, attribution |
| `compare_correlations.tsv` | gene1, gene2, rscu_pearson_r |

## Species Data Files

Each species directory (`~/.codonscope/data/species/{species}/`) contains:

| File | Description | Size (approx) |
|------|-------------|---------------|
| `cds_sequences.fa.gz` | Validated CDS sequences (gzipped FASTA) | 2-10 MB |
| `gene_id_map.tsv` | Gene ID mapping (systematic_name, common_name, ...) | 0.3-1.2 MB |
| `trna_copy_numbers.tsv` | tRNA gene counts by anticodon | <1 KB |
| `wobble_rules.tsv` | Codon-anticodon decoding rules with tRNA copies | ~2 KB |
| `background_mono.npz` | Per-gene monocodon frequencies + mean/std | 1-2 MB |
| `background_di.npz` | Per-gene dicodon frequencies + mean/std | 13-280 MB |
| `background_tri.npz` | Tricodon mean/std only (no per-gene matrix) | 2-3 MB |
| `gene_metadata.npz` | CDS lengths and GC contents | <200 KB |
| `expression_*.tsv` | Expression data (format varies by species) | 0.5-15 MB |

## Known Issues

1. **GtRNAdb parser broken**: All species fall back to hardcoded tRNA tables. Low priority.
2. **MANE version hardcoded**: URL contains `v1.5`. Will need updating when NCBI releases v1.6+.
3. **Tricodon backgrounds use analytic SE**: No per-gene matrix (226K x N_genes too large). Less accurate for small gene sets.
4. **Human dicodon backgrounds are large**: ~280 MB (19K genes x 3,721 dicodons). Consider sparse storage if disk space is a concern.
5. **Yeast/mouse expression is approximate**: Hardcoded TPM estimates from literature, not measured. Human GTEx data is real measured data.
6. **Ortholog mapping is name-based**: BioMart query currently fails. Uses gene name matching + curated renames (830 human-yeast pairs).
7. **Mouse has no MANE equivalent**: Uses longest valid CDS per gene from Ensembl. Some genes may have non-canonical transcript selected.
8. **Mouse orthologs not yet implemented**: Mode 6 cross-species comparison requires ortholog tables (currently only human-yeast available).
9. **Mouse expression is estimated**: No tissue-specific expression data. Cell line expression not available.

## References

- Doyle F, Leonardi A, Engel C, et al. CUT (Codon Usage Tool): A web-based resource for studying codon usage bias. *Methods* 2016; 107:98-109.
- Phizicky EM, Hopper AK. tRNA biology charges to the front. *Genes Dev* 2010; 24:1832-60.
- Chan PP, Lowe TM. GtRNAdb 2.0: an expanded database of transfer RNA genes. *Nucleic Acids Res* 2016; 44:D184-9.
- dos Reis M, Savva R, Wernisch L. Solving the riddle of codon usage preferences: a test for translational selection. *Nucleic Acids Res* 2004; 32:5036-44.
- Johansson MJ, Bystrom AS. Transfer RNA modifications and modifying enzymes in *Saccharomyces cerevisiae*. In: *Fine-tuning of RNA functions by modification and editing*. Springer, 2005: 87-120.
- Agris PF, Eruysal ER, Nahar A. tRNA's wobble decoding of the genome: 40 years of modification. *J Mol Biol* 2017; 430:2291-309.
- GTEx Consortium. The GTEx Consortium atlas of genetic regulatory effects across human tissues. *Science* 2020; 369:1318-30.
