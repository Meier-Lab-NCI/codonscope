# CodonScope Scripts

## counts_to_tpm.py

Converts raw RNA-seq counts to TPM (Transcripts Per Million) for use with CodonScope's Translational Demand analysis.

### Why TPM?

RNA-seq quantification produces different metrics that correct for different biases:

| Metric | Corrects for | Good for |
|--------|-------------|----------|
| **Raw counts** | Nothing | Input to DESeq2 / edgeR |
| **DESeq2 normalized** | Library size | Differential expression between conditions |
| **FPKM / RPKM** | Gene length + library size | Deprecated, not comparable across samples |
| **TPM** | Gene length + library size | Relative transcript abundance, comparable across samples |

CodonScope needs TPM because Translational Demand asks "what fraction of ribosomes are translating gene X?" That requires knowing relative molar transcript concentrations, which means correcting for gene length. A 3 kb gene produces 3x more sequencing reads than a 1 kb gene at the same expression level.

**Important:** Start from **raw counts**, not DESeq2 normalized counts. DESeq2 already corrects for library size, and TPM also corrects for library size — using DESeq2 output would double-correct.

### How TPM is calculated

```
RPK_i = count_i / (gene_length_i / 1000)     # reads per kilobase
TPM_i = RPK_i / sum(all RPK_j) * 1,000,000   # scale to per-million
```

TPMs always sum to 1,000,000 per sample.

### Usage

```bash
# featureCounts output (has a Length column):
python counts_to_tpm.py featureCounts.txt --length-column Length -o tpm.tsv

# No length column — pull CDS lengths from CodonScope reference data:
python counts_to_tpm.py counts.tsv --species human -o tpm.tsv
python counts_to_tpm.py counts.tsv --species yeast -o tpm.tsv
python counts_to_tpm.py counts.tsv --species mouse -o tpm.tsv

# Multi-sample file — pick one sample:
python counts_to_tpm.py counts.tsv --species human --sample tumor_rep1 -o tpm.tsv

# Filter low-count genes:
python counts_to_tpm.py counts.tsv --species human --min-count 10 -o tpm.tsv
```

### Input format

Tab-separated file with a gene ID column and one or more count columns. The script auto-detects common formats:

**featureCounts** (most common):
```
Geneid    Chr    Start    End    Strand    Length    sample1    sample2
TP53      chr17  7668402  7687550  -       2591      1523       1847
BRCA1     chr17  43044295 43170245 +       5592      312        287
```
The script auto-detects `Geneid` and `Length` columns, drops `Chr/Start/End/Strand`.

**Simple counts matrix** (DESeq2 input):
```
gene_id    control_1    control_2    treated_1    treated_2
TP53       1523         1410         1847         1902
BRCA1      312          298          287          301
```
The script uses the first sample unless you specify `--sample`.

**HTSeq-count** (two columns):
```
TP53    1523
BRCA1   312
```
Summary rows (`__no_feature`, `__ambiguous`, etc.) are automatically removed.

### Output format

Two-column tab-separated file, sorted by TPM descending:

```
gene_id    tpm
RPL11      15234.7
RPS3       12891.2
TP53       45.3
BRCA1      12.1
```

This file can be uploaded directly in the CodonScope Colab notebook (Translational Demand cell) or passed via CLI:

```bash
# Colab: check "Upload custom expression file" in the Translational Demand cell

# CLI:
python -m codonscope.cli demand --species human --genes genelist.txt --expression tpm.tsv
```

### Options

| Flag | Description |
|------|-------------|
| `-o`, `--output` | Output TPM file path (required) |
| `--species` | Species for CDS lengths: `human`, `yeast`, or `mouse`. Used when the counts file has no length column. Requires CodonScope data to be downloaded first. |
| `--gene-column` | Name of the gene ID column. Auto-detected if omitted (looks for `Geneid`, `gene_id`, `Gene`, etc., falls back to first column). |
| `--length-column` | Name of the gene length column. Auto-detected if a `Length` column exists. |
| `--sample` | Which sample column to convert. Uses the first numeric column if omitted. |
| `--min-count` | Minimum raw count threshold. Genes below this are excluded (default: 0). |

### Gene ID matching

When using `--species` to get CDS lengths from CodonScope, the script matches your gene IDs against:

- **Human**: HGNC symbols, Ensembl gene IDs (ENSG), transcript IDs (ENST), Entrez IDs
- **Yeast**: Systematic names (YFL039C), common names (ACT1)
- **Mouse**: MGI symbols, Ensembl gene IDs (ENSMUSG), transcript IDs (ENSMUST), Entrez IDs

Matching is case-insensitive for gene symbols. Unmatched genes are excluded and reported.

### Dependencies

Only `numpy` and `pandas` (already installed with CodonScope).
