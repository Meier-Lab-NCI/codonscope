# CodonScope

Multi-species codon usage analysis tool for gene lists. Replaces the defunct CUT tool (Doyle et al., *Methods* 2016). Analyzes mono-, di-, and tricodon patterns in user-provided gene lists versus genomic backgrounds, with expression-weighted demand analysis, translational optimality profiling, ribosome collision potential, amino acid vs. synonymous codon disentanglement, and cross-species comparison.

## Supported Species

| Species | Genome | Genes | Source |
|---------|--------|-------|--------|
| Yeast (*S. cerevisiae*) | S288C | 6,685 ORFs | SGD |
| Human | GRCh38 | 19,229 CDS | MANE Select v1.5 + Ensembl |
| Mouse (*M. musculus*) | GRCm39 | 21,556 CDS | Ensembl |

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

---

## Data Sources

All external data sources used by CodonScope, with exact URLs, versions, download dates, and processing methods.

### CDS Sequences

**Yeast (*S. cerevisiae*)**
- **URL**: `https://downloads.yeastgenome.org/sequence/S288C_reference/orf_dna/orf_coding_all.fasta.gz`
- **Version**: S288C reference genome
- **Downloaded**: 2026-02-07
- **Processing**: Parse SGD FASTA headers (`systematic_name common_name, ...`). Exclude mitochondrial ORFs (systematic names starting with `Q0`). Validate each CDS: contains only ACGT, length divisible by 3, starts with ATG, ends with stop codon (TAA/TAG/TGA), no internal stop codons. Strip terminal stop codon. Record verification status (Verified/Uncharacterized/Dubious) from header.
- **Result**: 6,685 validated ORFs saved as `cds_sequences.fa.gz` (keyed by systematic name) and `gene_id_map.tsv` (columns: systematic_name, common_name, cds_length, gc_content, status).

**Human**
- **MANE Summary URL**: `https://ftp.ncbi.nlm.nih.gov/refseq/MANE/MANE_human/current/MANE.GRCh38.v1.5.summary.txt.gz`
- **Ensembl CDS URL**: `https://ftp.ensembl.org/pub/current_fasta/homo_sapiens/cds/Homo_sapiens.GRCh38.cds.all.fa.gz`
- **Version**: MANE Select v1.5 (released December 2025), Ensembl GRCh38
- **Downloaded**: 2026-02-07
- **Processing**: Parse MANE summary TSV to extract gene ID mapping (ENSG, HGNC symbol, ENST, RefSeq NM, Entrez ID); filter to rows where `MANE_status == "MANE Select"` (excluding MANE Plus Clinical). Download Ensembl CDS FASTA; extract ENST ID from headers (`>ENST00000... cds chromosome:GRCh38:...`); match to MANE records by ENST (version-stripped). For each matched transcript, apply same CDS validation as yeast. First valid match per ENSG wins (one transcript per gene).
- **Result**: 19,229 validated CDS. `gene_id_map.tsv` columns: systematic_name (ENSG), common_name (HGNC symbol), ensembl_transcript, refseq_transcript, entrez_id, cds_length, gc_content.

**Mouse (*M. musculus*)**
- **URL**: `https://ftp.ensembl.org/pub/current_fasta/mus_musculus/cds/Mus_musculus.GRCm39.cds.all.fa.gz`
- **Version**: Ensembl GRCm39
- **Downloaded**: 2026-02-07
- **Processing**: Parse Ensembl CDS FASTA headers (`>ENSMUST... cds ... gene:ENSMUSG... gene_symbol:Xyz ...`). Extract ENSMUSG gene ID and gene symbol from each header. Apply same CDS validation as yeast. For each gene (ENSMUSG), collect all valid transcripts and keep the **longest** CDS (no MANE equivalent for mouse). Minimum CDS length: 9 nt (3 codons).
- **Result**: 21,556 validated CDS. `gene_id_map.tsv` columns: systematic_name (ENSMUSG), common_name (MGI symbol), ensembl_transcript (ENSMUST), cds_length, gc_content.

### tRNA Gene Copy Numbers

**Yeast**
- **URL**: `http://gtrnadb.ucsc.edu/genomes/eukaryota/Scere3/sacCer3-mature-tRNAs.fa`
- **Fallback**: Hardcoded table from Chan & Lowe 2016 / Phizicky & Hopper 2010 (~275 genes across 42 anticodons). Activated when GtRNAdb parse returns <20 anticodons.
- **Downloaded**: 2026-02-07 (fallback used; GtRNAdb parser returns 0 anticodons due to changed header format)

**Human**
- **URL**: `http://gtrnadb.ucsc.edu/genomes/eukaryota/Hsapi38/hg38-mature-tRNAs.fa`
- **Fallback**: Hardcoded table from Chan & Lowe 2016 (~430 genes across 46 anticodons). Same activation threshold.
- **Downloaded**: 2026-02-07 (fallback used)

**Mouse**
- **URL**: `http://gtrnadb.ucsc.edu/genomes/eukaryota/Mmusc39/mm39-mature-tRNAs.fa`
- **Fallback**: Hardcoded table based on GtRNAdb mm39 (~430 genes across 46 anticodons, similar to human). Same activation threshold.
- **Downloaded**: 2026-02-07 (fallback used)

**Note**: The GtRNAdb FASTA parser currently returns 0 parsed anticodons for all species. The header format may have changed from the expected `Type: Xxx  Anticodon: YYY` pattern. All species use hardcoded fallback tables, which are accurate and well-established in the literature.

### Wobble Decoding Rules

Curated tables mapping each of the 61 sense codons to its decoding tRNA anticodon, decoding type (Watson-Crick or wobble), and tRNA modification notes. Stored as `wobble_rules.tsv` with columns: codon, amino_acid, decoding_anticodon, trna_gene_copies, decoding_type, modification_notes.

| Species | Source | Notes |
|---------|--------|-------|
| Yeast | Johansson & Bystrom 2005; Agris et al. 2017 | Yeast-specific modifications: Trm4 (m5C at C34), Trm9 (mcm5s2U at U34), Tad1 (I34 at A34). Includes inosine-modified anticodon IAU for Ile-ATA. |
| Human | Same base wobble rules, human enzyme names | ALKBH8 (human Trm9 homolog), NSUN2 (human Trm4 homolog), ADAT1 (human Tad1 homolog). Key difference from yeast: Leu-CTT/CTC/CTA decoded by I34-modified tRNA(AAG) rather than separate anticodons. |
| Mouse | Identical to human rules | Same mammalian tRNA modification machinery. |

### Expression Data

**Yeast** — Literature estimates (not downloaded)
- **Source**: Published studies (Holstege 1998, Nagalakshmi 2008, Pelechano 2010)
- **Method**: Hardcoded TPM assignments: RPL*/RPS* ribosomal proteins = 3,000; glycolytic enzymes = 500-5,000 (per-gene from `YEAST_HIGH_EXPRESSION` dict); other Verified/Uncharacterized ORFs = 15 (median gene); Dubious ORFs = 0.5; mitochondrial (Q0*) = 50.
- **File**: `expression_rich_media.tsv` (columns: systematic_name, common_name, tpm)

**Human** — GTEx v8
- **URL**: `https://storage.googleapis.com/adult-gtex/bulk-gex/v8/rna-seq/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_median_tpm.gct.gz`
- **Version**: GTEx Analysis V8 (2017-06-05 release)
- **Downloaded**: 2026-02-07
- **Processing**: Parse GCT format (skip 2 header lines). Strip Ensembl gene version suffixes (ENSG00000000003.15 -> ENSG00000000003). Convert tissue columns to float. Result: ~56,000 genes x 54 tissues.
- **File**: `expression_gtex.tsv.gz` (columns: ensembl_gene, symbol, <54 tissue columns>)

**Human** — DepMap/CCLE cell line expression
- **API URL**: `https://depmap.org/portal/api/download/files`
- **Downloaded**: 2026-02-07
- **Processing**: Query API for file listing. Download Model.csv (cell line metadata) and OmicsExpressionProteinCodingGenesTPMLogp1.csv (expression matrix). Filter to default entries per model (`IsDefaultEntryForModel == "Yes"`). Parse gene columns in "SYMBOL (ENTREZ)" format. Map Entrez IDs to ENSG via gene_id_map. Convert log2(TPM+1) to TPM: `TPM = 2^value - 1`, clipped at 0. Transpose to genes-as-rows format.
- **File**: `expression_ccle.tsv.gz` (columns: ensembl_gene, symbol, <cell line columns>)

**Mouse** — Literature estimates (not downloaded)
- **Source**: General mouse RNA-seq literature
- **Method**: Hardcoded TPM assignments: Rpl*/Rps* ribosomal proteins = 3,000; housekeeping genes from `MOUSE_HIGH_EXPRESSION` dict (Actb = 5,000, Gapdh = 3,000, Eef1a1 = 3,000, etc.); all other genes = 15.
- **File**: `expression_estimates.tsv` (columns: systematic_name, common_name, tpm)

### Gene ID Resolution (Human)

- **URL**: `https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt`
- **Downloaded**: 2026-02-07
- **Processing**: Parse HGNC complete gene set. Extract previous symbols (pipe-separated `prev_symbol` field) and alias symbols (pipe-separated `alias_symbol` field). Filter to genes whose Ensembl gene ID exists in the MANE Select gene_id_map. Remove duplicates (keep first occurrence; previous symbols take priority over aliases).
- **File**: `hgnc_aliases.tsv` (columns: alias, canonical_symbol, ensembl_gene_id, alias_type)

### Orthologs

- **Human-Yeast ortholog pairs**: 830 one-to-one mappings
- **Primary method**: Ensembl BioMart query (XML query for `with_scerevisiae_homolog` + `ortholog_one2one`). Currently fails (filter name changed in Ensembl).
- **Fallback method**: Gene name matching between species gene_id_map.tsv files (case-insensitive), supplemented with ~150 curated renames in `HUMAN_YEAST_CURATED_RENAMES` dict (e.g., GAPDH -> TDH3, RPL11 -> RPL11A, EEF1A1 -> TEF1). A blocklist (`NAME_MATCH_BLOCKLIST`) excludes false matches. Curated renames take priority; direct name matches fill remaining pairs.
- **File**: `~/.codonscope/data/orthologs/human_yeast.tsv` (columns: human_gene, yeast_gene)
- **Generated**: 2026-02-07

---

## Analysis Methods

### Shared Statistical Framework

All modes share the following statistical components:

**Bootstrap resampling** (default B = 10,000 iterations): To estimate the standard error of the test statistic under the null hypothesis, we repeatedly draw random samples of size N (where N = number of genes in the user's gene set) from the genome background with replacement. Each resample computes the same summary statistic (e.g., mean codon frequency). The standard deviation across B resampled statistics is the bootstrap SE. Resampling is chunked in groups of 1,000 for memory efficiency and uses NumPy's `default_rng` for reproducible results when a seed is provided.

**Z-score**: Z = (observed - expected) / SE, where observed is the gene-set statistic, expected is the mean across B bootstrap resamples, and SE is the SD across B bootstrap resamples (using Bessel's correction, ddof=1). When SE = 0, Z is set to 0.

**Null hypothesis** (Modes 1, 2): The gene set's codon usage is consistent with a random sample of N genes drawn from the genome background. Rejection of the null (large |Z|) indicates the gene set has systematically different codon usage from the genome.

**P-values**: Two-sided p-values from the standard normal distribution: p = 2 * P(Z > |z|), computed via `scipy.stats.norm.sf(|z|) * 2`.

**Benjamini-Hochberg FDR correction**: Applied across all k-mers within each analysis. Step-up procedure: (1) sort p-values in ascending order, (2) compute adjusted_p[i] = p[i] * n / rank[i] where rank is 1-indexed position in the sorted list and n is total number of tests, (3) enforce monotonicity by walking backwards and taking the cumulative minimum, (4) clip to [0, 1]. This controls the false discovery rate at the nominal alpha level (typically 0.05).

**Cohen's d**: Effect size measure reported alongside Z-scores. d = (observed - expected) / pooled_SD.

---

### Mode 1: Codon Composition

Tests whether the user's gene set has different codon (or dicodon/tricodon) frequencies compared to the genome background.

**Per-gene frequency computation**: For each gene, count occurrences of each k-mer using a sliding window of width k codons (k=1: monocodons, k=2: dicodons, k=3: tricodons). The window slides by 1 codon: positions 1, 2, 3, ... for monocodons; positions 1-2, 2-3, 3-4, ... for dicodons (overlapping). Only the 61 sense codons are counted (stop codons excluded by prior CDS validation). Frequency = count / total_windows. This yields 61 frequencies for monocodons, 61^2 = 3,721 for dicodons, 61^3 = 226,981 for tricodons.

**Gene-set statistic**: Mean of per-gene frequencies across all genes in the set.

**Genome background**: Mean of per-gene frequencies across all genes in the species (pre-computed during download and stored in `background_{mono,di,tri}.npz`).

**Bootstrap test (monocodons and dicodons)**: For each of B = 10,000 iterations, draw N gene indices uniformly at random from [0, N_genome), extract their pre-computed per-gene frequency vectors, and compute the resample mean. The bootstrap SE is the SD of these 10,000 resample means (ddof=1). Z = (gene_set_mean - bootstrap_mean) / bootstrap_SE. Two-sided p-value from standard normal. BH correction across all k-mers (61 for mono, 3,721 for di).

**Tricodon exception**: The per-gene frequency matrix for tricodons (226,981 x N_genes) is too large to store. Instead, only the genome-wide mean and SD per tricodon are stored. The analytic SE is computed as SE = SD / sqrt(N), where N is the gene-set size. Z = (gene_set_mean - genome_mean) / SE. This is less accurate for small gene sets because it assumes normality of the sampling distribution.

**Matched background** (`--background matched`): Instead of resampling from all genome genes, restrict the resampling pool to genes that match the gene set in CDS length and GC content. Matching criteria: CDS length within [min(gene_set_lengths) * 0.8, max(gene_set_lengths) * 1.2] (i.e., 80%-120% of the gene set's length range), AND GC content within [min(gene_set_GC) - 0.05, max(gene_set_GC) + 0.05] (i.e., within 5 percentage points of the gene set's GC range). If the matched pool contains fewer than 100 genes, a warning is logged. This controls for length- and GC-driven compositional biases.

**Trim ramp option** (`--trim-ramp N`): Before computing frequencies, skip the first N codons of each CDS (both gene-set and background genes). This removes the 5' translation initiation ramp, which may have distinct codon usage unrelated to the biological signal of interest. Applied per-gene: `seq = seq[N*3:]`, then truncate to a multiple of 3 if needed.

**KS diagnostics**: Two Kolmogorov-Smirnov tests compare the gene-set distributions of CDS length and GC content against the genome background. If either KS p < 0.01, a warning is emitted suggesting the user consider `--background matched`. The KS test uses `scipy.stats.ks_2samp()`.

**Output**: Results DataFrame sorted by |Z-score| descending. Columns: kmer, observed_freq (gene-set mean), expected_freq (bootstrap mean or genome mean), z_score, p_value, adjusted_p, cohens_d, amino_acid annotation.

---

### Mode 2: Translational Demand

Tests whether the gene set's expression-weighted codon demand differs from the genome-wide translational demand.

**Demand weight per gene**: weight = TPM * n_codons, where TPM is the gene's expression level and n_codons = CDS_length / 3. Genes with TPM <= 0 are excluded. This weight reflects the gene's contribution to the total translational workload.

**Demand vector**: For a set of genes, the demand vector is the weighted mean of per-gene codon frequencies:

```
demand[k] = Σ(weight_i * freq_i[k]) / Σ(weight_i)
```

where freq_i[k] is gene i's frequency of k-mer k, and weight_i = TPM_i * n_codons_i.

**Weighted bootstrap Z-scores**: Pre-compute all genome genes' per-gene frequency vectors (float32) and weights (float64). For each of B = 10,000 iterations: draw N gene indices uniformly at random from the genome pool, extract their frequency vectors and weights, compute the weighted demand vector for the sample. Z = (gene_set_demand - mean_bootstrap_demand) / SD_bootstrap_demand (ddof=1). Two-sided p from standard normal. BH correction across all k-mers.

**Null hypothesis**: The gene set's expression-weighted codon demand is consistent with the demand of N random genes drawn from the genome (using each gene's actual expression weight). Rejection indicates the gene set has systematically different translational demand.

**Top-N filtering** (`--top-n`): When set, only the top-N genes by demand weight are included in the background pool. This focuses the comparison on highly-expressed genes.

**Expression sources**: Yeast — hardcoded rich-media estimates (`expression_rich_media.tsv`). Human — GTEx v8 tissue-specific TPM (specify with `--tissue`), CCLE cell line expression (specify with `--cell-line`), or cross-tissue median TPM (default when no tissue specified; falls back to HEK293T proxy via GTEx Kidney-Cortex when using `--cell-line HEK293T`). Mouse — hardcoded estimates (`expression_estimates.tsv`). Custom — user-supplied TSV with gene_id and tpm columns (`--expression`).

**GTEx tissue matching**: Case-insensitive, supports substring matching. For example, `--tissue liver` matches "Liver", `--tissue kidney` matches "Kidney - Cortex" and "Kidney - Medulla".

---

### Mode 3: Optimality Profile (Metagene)

Scores each codon position in the CDS by translational optimality and generates a metagene profile comparing the gene set to the genome.

**Weighted tRNA Adaptation Index (wtAI)**: For each of the 61 sense codons:
1. Raw tAI = tRNA_gene_copies for the decoding anticodon (from `trna_copy_numbers.tsv` via `wobble_rules.tsv`).
2. Normalize: tAI = raw / max(raw) across all 61 codons, so values range (0, 1] with the most abundant anticodon scoring 1.0.
3. Pseudocount: codons with 0 tRNA gene copies get a pseudocount of 0.5 / max(raw), ensuring no codon has exactly zero weight (important for geometric means).
4. Wobble penalty: for codons decoded by wobble base-pairing (decoding_type = "wobble" in wobble_rules.tsv), multiply by `wobble_penalty` (default 0.5): wtAI = tAI * 0.5. Watson-Crick codons retain their full tAI score.
5. Final weights are in range [pseudocount * wobble_penalty, 1.0].

**Per-gene score (geometric mean)**: gene_wtAI = exp(mean(log(wtAI[codon]))) across all codons in the CDS. Codons with weight <= 0 are skipped (should not occur with pseudocount).

**Per-position scores**: For each gene, compute wtAI at each codon position (1, 2, 3, ..., n_codons). Apply sliding-window smoothing with a window of 10 codons: `smoothed = convolve(scores, ones(10)/10, mode='same')`.

**Metagene profile**: Each gene's smoothed positional scores are resampled to exactly 100 positional bins (0-99% of CDS) via linear interpolation (`numpy.interp`). Genes with fewer than 30 codons are excluded. The metagene profile is the mean across all genes at each of the 100 positions. A separate genome-wide metagene is computed identically from all genome genes.

**Ramp analysis**: For each gene with more than `ramp_codons` codons (default 50):
- ramp_mean = mean(per-position wtAI for codons 1 through ramp_codons)
- body_mean = mean(per-position wtAI for codons ramp_codons+1 through end)
- delta = body_mean - ramp_mean

A positive delta indicates higher optimality in the body than the ramp (consistent with the translation initiation ramp hypothesis). The gene-set ramp delta is compared descriptively to the genome-wide distribution. No formal hypothesis test is applied; the delta and per-gene distributions are reported for visual comparison.

---

### Mode 4: Collision Potential

Quantifies the enrichment of fast-to-slow codon transitions (ribosome collision sites) in the gene set compared to the genome.

**Codon classification**: All 61 sense codons are classified as "fast" or "slow" based on the median wtAI score (from Mode 3's OptimalityScorer). Codons with wtAI >= median are "fast"; codons with wtAI < median are "slow". This yields approximately 30/31 codons in each category.

**Transition counting**: For each gene, scan consecutive codon pairs (positions i, i+1) and classify each transition as FF (fast-fast), FS (fast-slow), SF (slow-fast), or SS (slow-slow). Count total transitions per gene and per category. Gene-set counts are summed across all genes.

**Proportions**: prop_XX = count_XX / total_transitions for XX in {FF, FS, SF, SS}. Proportions sum to 1.

**FS enrichment ratio**: FS_enrichment = prop_FS_geneset / prop_FS_genome. Values > 1 indicate the gene set has more fast-to-slow transitions than the genome average, suggesting more ribosome collision-prone junctions.

**FS/SF ratio**: FS_count / SF_count. Values > 1 indicate more forward (fast-to-slow) than reverse (slow-to-fast) transitions, suggesting directional ribosome slowdown patterns.

**Chi-squared test**: Null hypothesis: the gene set has the same FF/FS/SF/SS transition distribution as the genome. Expected counts: expected_XX = prop_XX_genome * total_gs_transitions. Chi-squared statistic: chi2 = sum((observed - expected)^2 / expected) for categories where expected > 0. Degrees of freedom = (number of categories with expected > 0) - 1. P-value from `scipy.stats.chi2.sf()`. This is a goodness-of-fit test against the genome's transition proportions.

**Positional FS clustering**: Each gene's CDS is divided into 10 equal bins (0-10%, 10-20%, ..., 90-100%). For each bin, count FS transitions and total transitions. Report FS fraction per positional bin to detect whether collision sites cluster at specific regions (e.g., 5' ramp). Position is computed as `(codon_index / (n_codons - 1)) * 100` as percent of CDS.

---

### Mode 5: AA vs. Codon Disentanglement

Decomposes codon frequency deviations into amino acid composition effects and synonymous codon choice (RSCU) effects.

**Two-layer decomposition**:

*Layer 1 — Amino acid composition*: For each gene, compute amino acid frequencies (count each AA / total codons). Gene-set AA frequency = mean across genes. Genome AA frequency = mean across all genome genes. Bootstrap Z-scores (B = 10,000): resample N genes from genome, compute mean AA frequency, estimate SE. Z and p-values as in Mode 1.

*Layer 2 — Relative Synonymous Codon Usage (RSCU)*: For each gene, within each amino acid family (e.g., the 4 Ala codons), compute RSCU = n_synonyms * (count_codon / count_AA). If count_AA = 0, RSCU is undefined. An RSCU of 1.0 means the codon is used equally with its synonyms; RSCU > 1 means preferred; RSCU < 1 means avoided. Gene-set RSCU = mean across genes. Bootstrap Z-scores as above.

**Attribution classification**: For each codon, using BH-adjusted p-values at alpha = 0.05:
- **AA-driven**: The amino acid's frequency is significantly different from genome (AA layer adj_p < 0.05) AND the RSCU is not significant (RSCU layer adj_p >= 0.05). The deviation is explained by over/under-representation of the amino acid.
- **Synonymous-driven**: The RSCU is significantly different (adj_p < 0.05) AND the AA frequency is not significant (adj_p >= 0.05). The deviation reflects preferential use of specific synonymous codons.
- **Both**: Both AA frequency and RSCU are significant (both adj_p < 0.05).
- **None**: Neither is significant.

**Synonymous driver classification** (heuristic, applied to codons with significant RSCU): Rules applied in order:
1. If codon is enriched (Z > 0) AND decoding_type is "watson_crick" AND tRNA gene copies > 0 -> **wobble_avoidance** (gene set avoids wobble-decoded alternatives)
2. Else if codon is enriched AND tRNA gene copies > 5 -> **tRNA_supply** (codon matches abundant tRNA)
3. Else if codon is enriched AND third position is G or C -> **GC3_bias**
4. Else if codon is depleted (Z < 0) AND third position is A or T -> **GC3_bias**
5. Else -> **unclassified**

Single-codon amino acids (Met/ATG, Trp/TGG) are labeled "not_applicable" since they have no synonymous alternatives.

---

### Mode 6: Cross-Species Comparison

Compares RSCU patterns between orthologous genes across two species.

**Per-gene RSCU vector**: For each gene, count all 61 sense codons. For each amino acid family with >= 2 synonymous codons (excluding Met and Trp), compute RSCU = n_synonyms * count_codon / total_AA_count. This yields a vector of 59 RSCU values (61 minus Met and Trp). Genes with fewer than 10 total codons are excluded (RSCU vector set to NaN).

**Per-gene RSCU correlation**: For each ortholog pair (gene_A in species 1, gene_B in species 2), compute the Pearson correlation coefficient between their RSCU vectors, using only positions where both values are non-NaN. If fewer than 5 valid positions remain, the correlation is NaN. High r indicates similar synonymous codon preferences; low r indicates divergent codon usage (potentially reflecting different tRNA pools).

**Gene-set vs. genome test**: Two complementary tests compare the distribution of per-gene RSCU correlations for the user's gene set to the genome-wide distribution:

1. **Bootstrap Z-test** (B = 10,000): Resample N genes from the genome ortholog pool, compute mean RSCU correlation for each resample. Z = (gene_set_mean_r - bootstrap_mean_r) / bootstrap_SE_r. Two-sided p from standard normal.

2. **Mann-Whitney U test**: Non-parametric comparison of gene-set correlations vs. genome-wide correlations. `scipy.stats.mannwhitneyu(gs_corrs, genome_corrs, alternative="two-sided")`. Requires >= 3 samples in each group.

**Divergent gene analysis**: Ortholog pairs in the bottom 25th percentile of RSCU correlation (across all genome orthologs) are classified as "divergent." For each divergent pair, identify amino acid families where the preferred codon (highest RSCU) differs between species, indicating that the two species have adapted their codon usage to different tRNA pools for these amino acids.

---

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
| `--species` | Species name (`yeast`, `human`, `mouse`) | required |
| `--genes` | Path to gene list file | required |
| `--output` | Output file/directory | stdout |
| `--kmer` | k-mer size (1/2/3) for composition/demand | 1 |
| `--background` | Background type: `all` or `matched` | `all` |
| `--n-bootstrap` | Bootstrap iterations | 10000 |
| `--seed` | Random seed for reproducibility | None |

### Mode-Specific Options

| Option | Subcommand | Description |
|--------|-----------|-------------|
| `--tissue` | `demand` | GTEx tissue name (human only) |
| `--cell-line` | `demand` | CCLE cell line name (human only) |
| `--expression` | `demand` | Custom expression file path (TSV with gene_id + tpm) |
| `--top-n` | `demand` | Use only top-N expressed genes in background |
| `--method` | `profile` | Scoring method: `tai` or `wtai` (default: `wtai`) |
| `--species1` | `compare` | First species in cross-species comparison |
| `--species2` | `compare` / `report` | Second species (enables Mode 6) |
| `--from-species` | `compare` | Which species the input gene list belongs to |

### Gene List Format

One gene per line. Supports:
- Common names: `RPL3`, `GAPDH`, `Actb`
- Systematic names: `YOR063W`, `ENSG00000142541`, `ENSMUSG00000060036`
- Ensembl transcript IDs: `ENST00000341065`, `ENSMUST00000000003`
- Entrez IDs (human only): `6122`
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

# HEK293T cell line expression (uses CCLE or GTEx Kidney-Cortex proxy)
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
| `gene_metadata.npz` | CDS lengths and GC contents per gene | <200 KB |
| `expression_*.tsv` | Expression data (format varies by species) | 0.5-15 MB |

## Known Issues

1. **GtRNAdb parser broken**: All species fall back to hardcoded tRNA tables. The FASTA header format may have changed from the expected `Type: Xxx  Anticodon: YYY` pattern. Low priority since hardcoded tables are accurate.
2. **MANE version hardcoded**: URL contains `v1.5`. Will need updating when NCBI releases v1.6+. The Ensembl CDS URL uses `current_fasta` and auto-updates.
3. **Tricodon backgrounds use analytic SE**: No per-gene matrix (226,981 x N_genes too large to store). Uses SE = SD / sqrt(N) instead of bootstrap. Less accurate for small gene sets because it assumes the sampling distribution is normal.
4. **Human dicodon backgrounds are large**: ~280 MB (19,229 genes x 3,721 dicodons x float32). Consider sparse storage if disk space is a concern.
5. **Yeast/mouse expression is approximate**: Hardcoded TPM estimates from literature, not measured. Human GTEx data is real measured data.
6. **Ortholog mapping is name-based**: BioMart query currently fails (Ensembl filter name changed). Uses gene name matching + 150 curated renames (830 human-yeast pairs). True Ensembl Compara orthologs would give better coverage (~1,200+ pairs).
7. **Mouse has no MANE equivalent**: Uses longest valid CDS per gene from Ensembl. Some genes may have a non-canonical transcript selected.
8. **Mouse orthologs not yet implemented**: Mode 6 cross-species comparison currently supports human-yeast only.
9. **Mouse expression is estimated**: No tissue-specific measured expression data. Cell line expression not available for mouse.

## References

- Doyle F, Leonardi A, Engel C, et al. CUT (Codon Usage Tool): A web-based resource for studying codon usage bias. *Methods* 2016; 107:98-109.
- Phizicky EM, Hopper AK. tRNA biology charges to the front. *Genes Dev* 2010; 24:1832-60.
- Chan PP, Lowe TM. GtRNAdb 2.0: an expanded database of transfer RNA genes. *Nucleic Acids Res* 2016; 44:D184-9.
- dos Reis M, Savva R, Wernisch L. Solving the riddle of codon usage preferences: a test for translational selection. *Nucleic Acids Res* 2004; 32:5036-44.
- Johansson MJ, Bystrom AS. Transfer RNA modifications and modifying enzymes in *Saccharomyces cerevisiae*. In: *Fine-tuning of RNA functions by modification and editing*. Springer, 2005: 87-120.
- Agris PF, Eruysal ER, Nahar A. tRNA's wobble decoding of the genome: 40 years of modification. *J Mol Biol* 2017; 430:2291-309.
- GTEx Consortium. The GTEx Consortium atlas of genetic regulatory effects across human tissues. *Science* 2020; 369:1318-30.
- Benjamini Y, Hochberg Y. Controlling the false discovery rate: a practical and powerful approach to multiple testing. *J R Stat Soc Series B* 1995; 57:289-300.
