# CodonScope Implementation Status

Last updated: 2026-02-07
338 tests passing + 6 skipped. 23 commits on main. Version 0.1.0.

---

## 1. What's Implemented and Tested

All 6 analysis modes, 3 species, HTML report, and full CLI are complete.

### Core Engine

**`codonscope/core/codons.py`** — K-mer counting engine with amino acid annotations.

```python
SENSE_CODONS: list[str]                                    # 61 sense codons, sorted
sequence_to_codons(sequence: str) -> list[str]
count_kmers(sequence: str, k: int = 1) -> dict[str, int]
kmer_frequencies(sequence: str, k: int = 1) -> dict[str, float]
annotate_kmer(kmer: str, k: int = 1) -> str               # e.g. "AAG" → "AAG (Lys)"
all_possible_kmers(k: int = 1, sense_only: bool = True) -> list[str]
```

**`codonscope/core/sequences.py`** — Gene ID resolution and CDS retrieval. Enhanced multi-type ID resolution with auto-detection.

```python
class IDMapping:
    """Dict-like result from resolve_ids(). Keys are input IDs, values are systematic names."""
    .n_mapped: int
    .n_unmapped: int
    .unmapped: list[str]
    # Supports len(), [], in, .keys(), .values(), .items(), .get()
    # Backwards compat: result["mapping"], result["n_mapped"] still work

class SequenceDB:
    def __init__(self, species: str, data_dir: str | Path | None = None)
    def resolve_ids(self, gene_ids: list[str]) -> IDMapping
    def get_sequences(self, names) -> dict[str, str]          # accepts IDMapping, dict, or list[str]
    def get_sequences_for_ids(self, gene_ids: list[str]) -> dict[str, str]  # convenience combo
    def get_all_sequences(self) -> dict[str, str]
    def get_common_names(self, systematic_names: list[str]) -> dict[str, str]
    def get_gene_metadata(self) -> pd.DataFrame
    @property species_dir -> Path
```

ID types supported per species:
- **Yeast:** systematic name (YFL039C), common name (ACT1), SGD ID (SGD:S000001855), UniProt (P60010)
- **Human:** HGNC symbol (TP53), ENSG, ENST, Entrez ID (7157), RefSeq NM_ (NM_000546.6), UniProt (P04637), HGNC aliases
- **Mouse:** MGI symbol (Actb), ENSMUSG, ENSMUST, MGI ID (MGI:87904), Entrez ID (11461), UniProt, MGI synonyms

Mixed ID types per gene list are supported. Auto-detection via regex patterns with debug logging.

**`codonscope/core/statistics.py`** — Bootstrap Z-scores, multiple testing, effect sizes.

```python
compute_geneset_frequencies(sequences, k=1, trim_ramp=0) -> (per_gene_matrix, mean_vector, kmer_names)
bootstrap_zscores(geneset_mean, background_per_gene, n_genes, n_bootstrap=10000, seed=None) -> (z_scores, boot_mean, boot_std)
bootstrap_pvalues(z_scores) -> np.ndarray
benjamini_hochberg(p_values) -> np.ndarray
cohens_d(geneset_mean, background_mean, background_std) -> np.ndarray
power_check(n_genes, k) -> list[str]
diagnostic_ks_tests(geneset_lengths, geneset_gc, bg_lengths, bg_gc) -> dict
compare_to_background(gene_sequences, background_npz_path, k=1, n_bootstrap=10000, trim_ramp=0, seed=None) -> pd.DataFrame
```

**`codonscope/core/optimality.py`** — tAI and wobble-aware tAI scoring.

```python
class OptimalityScorer:
    def __init__(self, species_dir: str | Path, wobble_penalty: float = 0.5)
    def gene_tai(self, sequence: str) -> float
    def gene_wtai(self, sequence: str) -> float
    def per_position_scores(self, sequence: str, method: str = "wtai") -> np.ndarray
    def smooth_profile(self, scores: np.ndarray, window: int = 10) -> np.ndarray
    def classify_codons(self, threshold: float | None = None, method: str = "wtai") -> tuple[set, set]
    @property tai_weights: dict[str, float]
    @property wtai_weights: dict[str, float]
```

**`codonscope/core/orthologs.py`** — Bidirectional ortholog mapping.

```python
class OrthologDB:
    def __init__(self, species1: str, species2: str, data_dir: str | Path | None = None)
    def map_genes(self, gene_ids: list[str], from_species: str) -> dict[str, str]
    def get_all_pairs(self) -> list[tuple[str, str]]
    @property n_pairs: int
```

### Data Layer

**`codonscope/data/download.py`** — Downloads and pre-computes all reference data.

```python
download(species: str, data_dir: str | Path | None = None) -> Path
download_expression(species: str, data_dir: str | Path | None = None) -> Path
download_orthologs(species1: str, species2: str, data_dir: str | Path | None = None) -> Path
```

Species support: yeast, human, mouse.

Key internal functions:
- `_parse_gtrnadb_fasta(fasta_text)` — 3-strategy parser: old (Type:/Anticodon:), modern (tRNA-Ala-AGC), compact (trna34-AlaAGC). T→U conversion. Skips iMet/SeC/Sup/Undet.
- `_fetch_mouse_canonical_transcripts()` — BioMart canonical transcript query
- `_fetch_mouse_entrez_ids()` — BioMart Entrez gene ID query
- `_download_mgi_mapping()` — MGI ID/synonym mapping from JAX
- `_download_uniprot_mapping()` — UniProt SwissProt mapping via BioMart (all species)
- `_discover_compara_url()` — Auto-discover Ensembl Compara release URL
- `_download_orthologs_compara()` — Bulk Ensembl Compara ortholog download
- `download_ccle_expression()` — CCLE/DepMap cell line expression data (human only)

### Analysis Modes

**Mode 1: Composition** — `codonscope/modes/mode1_composition.py`

```python
run_composition(
    species: str, gene_ids: list[str],
    k: int | None = None,               # preferred; aliases: kmer, kmer_size
    kmer: int | None = None,
    kmer_size: int | None = None,
    background: str = "all",             # "all" or "matched" (length+GC)
    trim_ramp: int = 0,                  # skip first N codons from 5' end
    min_genes: int = 10,
    n_bootstrap: int = 10_000,
    output_dir: str | Path | None = None,
    seed: int | None = None,
    data_dir: str | Path | None = None,
) -> dict
# Returns: results (DataFrame), diagnostics (dict), id_summary (IDMapping), n_genes (int)
# DataFrame columns: kmer, observed_freq, expected_freq, z_score, p_value, adjusted_p, cohens_d
# K-mers annotated with amino acids (e.g., "AAG (Lys)")
```

Features: mono/di/tricodon analysis, matched (length+GC) background, KS diagnostics, ramp trimming, amino acid annotations on k-mers, volcano + bar chart plots, gene name display.

**Mode 2: Translational Demand** — `codonscope/modes/mode2_demand.py`

```python
run_demand(
    species: str, gene_ids: list[str],
    k: int = 1,
    tissue: str | None = None,           # GTEx tissue for human
    cell_line: str | None = None,        # CCLE cell line for human
    expression_file: str | Path | None = None,  # custom expression TSV
    top_n: int | None = None,
    n_bootstrap: int = 10_000,
    seed: int | None = None,
    output_dir: str | Path | None = None,
    data_dir: str | Path | None = None,
) -> dict
# Returns: results (DataFrame), top_genes (DataFrame), tissue (str), available_tissues (list),
#          n_genes (int), id_summary (IDMapping)
```

Features: expression-weighted codon demand, GTEx tissue-specific (human), CCLE cell line expression (human, e.g. HEK293T, HeLa, K562), custom expression file support, weighted bootstrap Z-scores, amino acid annotations, top demand-contributing genes.

**Mode 3: Optimality Profile** — `codonscope/modes/mode3_profile.py`

```python
run_profile(
    species: str, gene_ids: list[str],
    window: int = 10,
    wobble_penalty: float = 0.5,
    ramp_codons: int = 50,               # first N codons for ramp analysis
    method: str = "wtai",                # "tai" or "wtai"
    n_bootstrap: int = 1_000,
    seed: int | None = None,
    output_dir: str | Path | None = None,
    data_dir: str | Path | None = None,
) -> dict
# Returns: metagene_geneset (ndarray), metagene_genome (ndarray), ramp_analysis (dict),
#          ramp_composition (DataFrame), body_composition (DataFrame),
#          per_gene_scores (DataFrame), scorer (OptimalityScorer), id_summary (IDMapping), n_genes (int)
```

Features: per-position tAI/wtAI scoring, sliding window smoothing, normalised to 100 positional bins, ramp analysis (first N codons vs body), ramp codon composition breakdown, metagene line + ramp bar chart plots.

**Mode 4: Collision Potential** — `codonscope/modes/mode4_collision.py`

```python
run_collision(
    species: str, gene_ids: list[str],
    wobble_penalty: float = 0.5,
    threshold: float | None = None,      # custom fast/slow threshold
    method: str = "wtai",
    output_dir: str | Path | None = None,
    data_dir: str | Path | None = None,
) -> dict
# Returns: transition_matrix_geneset (dict), transition_matrix_genome (dict),
#          fs_enrichment (float), fs_sf_ratio_geneset (float), fs_sf_ratio_genome (float),
#          chi2_stat (float), chi2_p (float), per_gene_fs_frac (DataFrame),
#          fs_positions (DataFrame), fs_dicodons (DataFrame),
#          fast_codons (set), slow_codons (set), threshold (float),
#          id_summary (IDMapping), n_genes (int)
```

Features: FF/FS/SF/SS dicodon transition counting, FS enrichment ratio, per-dicodon FS enrichment analysis (`fs_dicodons` DataFrame), per-gene FS fractions, positional FS clustering, chi-squared test vs genome, transition bars + positional FS plots.

**Mode 5: AA vs Codon Disentanglement** — `codonscope/modes/mode5_disentangle.py`

```python
run_disentangle(
    species: str, gene_ids: list[str],
    n_bootstrap: int = 10_000,
    seed: int | None = None,
    output_dir: str | Path | None = None,
    data_dir: str | Path | None = None,
) -> dict
# Returns: aa_results (DataFrame), rscu_results (DataFrame), attribution (DataFrame),
#          synonymous_drivers (DataFrame), summary (dict), id_summary (IDMapping), n_genes (int)
# Attribution values: "AA-driven", "Synonymous-driven", "Both", "None"
# Driver values: "tRNA_supply", "GC3_bias", "wobble_avoidance", "unclassified", "not_applicable"
```

Features: two-layer decomposition (AA composition + RSCU), per-codon attribution, synonymous driver classification (tRNA supply, GC3 bias, wobble avoidance), two-panel plot.

**Mode 6: Cross-Species Comparison** — `codonscope/modes/mode6_compare.py`

```python
run_compare(
    species1: str, species2: str, gene_ids: list[str],
    from_species: str | None = None,     # which species gene IDs belong to
    n_bootstrap: int = 10_000,
    seed: int | None = None,
    output_dir: str | Path | None = None,
    data_dir: str | Path | None = None,
) -> dict
# Returns: per_gene (DataFrame), summary (dict), divergent_analysis (DataFrame),
#          scatter_data (dict of {aa: DataFrame}), n_orthologs (int),
#          n_genome_orthologs (int), from_species (str), to_species (str), id_summary (dict)
```

Features: per-gene RSCU correlation between ortholog pairs, gene-set vs genome-wide distribution, bootstrap Z-test + Mann-Whitney U, divergent gene analysis, scatter data per amino acid, bidirectional analysis (from_species parameter), correlation histogram + ranked bar chart plots.

### Report

**`codonscope/report.py`** — HTML report generator.

```python
generate_report(
    species: str, gene_ids: list[str],
    output: str | Path = "report.html",
    species2: str | None = None,
    tissue: str | None = None,
    cell_line: str | None = None,
    n_bootstrap: int = 10_000,
    seed: int | None = None,
    data_dir: str | Path | None = None,
) -> Path
```

Self-contained HTML with inline CSS and base64-embedded matplotlib plots. Runs all applicable modes (1 mono+di, 5, 3, 4, 2, optionally 6). Includes gene summary with ID mapping table, volcano plots, attribution table, metagene profile, collision bars, demand analysis, cross-species correlation (if species2 provided). Exports `{stem}_results.zip` containing HTML report, data/*.tsv files, and README.txt documenting analysis parameters, input gene list, output file descriptions, and reference data sources.

### CLI

**`codonscope/cli.py`** — 8 subcommands.

```
codonscope download     --species SPECIES [SPECIES ...] [--data-dir DIR]
codonscope report       --species S --genes FILE [--species2 S] [--tissue T] [--cell-line C] [--output FILE] [--n-bootstrap N] [--seed N] [--data-dir DIR]
codonscope composition  --species S --genes FILE [--kmer {1,2,3}] [--background {all,matched}] [--trim-ramp N] [--min-genes N] [--n-bootstrap N] [--output-dir DIR] [--seed N] [--data-dir DIR]
codonscope demand       --species S --genes FILE [--tissue T] [--cell-line C] [--expression-file FILE] [--top-n N] [--n-bootstrap N] [--output-dir DIR] [--seed N] [--data-dir DIR]
codonscope profile      --species S --genes FILE [--window N] [--wobble-penalty P] [--ramp-codons N] [--method {tai,wtai}] [--output-dir DIR] [--data-dir DIR]
codonscope collision    --species S --genes FILE [--wobble-penalty P] [--threshold T] [--method {tai,wtai}] [--output-dir DIR] [--data-dir DIR]
codonscope disentangle  --species S --genes FILE [--n-bootstrap N] [--output-dir DIR] [--seed N] [--data-dir DIR]
codonscope compare      --species1 S --species2 S --genes FILE [--from-species S] [--n-bootstrap N] [--output-dir DIR] [--seed N] [--data-dir DIR]
```

Gene list files support: one ID per line, comma-separated, tab-separated, `#` comments.

### Tests (338 passing, 6 skipped)

| File | Count | What it covers |
|------|-------|----------------|
| `tests/test_chunk1.py` | 23 | Yeast download files, ID mapping, CDS validation, k-mer counting, backgrounds |
| `tests/test_statistics.py` | 29 | Bootstrap Z-scores, BH correction, Cohen's d, KS diagnostics, power, yeast positive controls (RP, YEF3) |
| `tests/test_mode1.py` | 19 | Gene list parsing, composition pipeline (mono/di/matched/trim_ramp), diagnostics, CLI, RP positive controls |
| `tests/test_human.py` | 33 | Human download files, gene map structure/columns, CDS validation, ID resolution (HGNC/ENSG/ENST/Entrez), backgrounds, Mode 1 on human RP genes |
| `tests/test_mode5.py` | 27 | Codon table, AA frequencies, RSCU, attribution logic, summary, Gcn4/RP integration, CLI |
| `tests/test_mode3.py` | 26 | OptimalityScorer unit tests, metagene profile, ramp analysis, yeast RP integration, random gene negative control, CLI |
| `tests/test_mode4.py` | 22 | Transition counting (FF/FS/SF/SS), proportions, FS enrichment, chi-squared, yeast RP integration, Gcn4 comparison, CLI |
| `tests/test_mode2.py` | 36 | Demand vectors, expression loading (yeast + human GTEx), weighted bootstrap, yeast RP demand, Gcn4 demand, human liver demand, CLI |
| `tests/test_mode6.py` | 36 | OrthologDB unit tests, RSCU correlation, ortholog download, RP comparison (yeast→human + human→yeast), divergent analysis, CLI |
| `tests/test_report.py` | 18 | HTML report generation (RP + Gcn4), all mode sections, base64 images, gene summary, cross-species report, CLI |
| `tests/test_id_resolution.py` | 75 (69+6skip) | GtRNAdb parser (3 formats), ID regex patterns, auto-detection, backward-compat resolution, new ID types (RefSeq, SGD, UniProt, MGI, Entrez), mixed ID lists, download function imports, ortholog dispatcher, parse unit tests, IDMapping compat |
| **Total** | **338 + 6 skip** | |

6 tests skipped: UniProt, MGI, SGD ID, and Entrez lookups need data re-download to populate new mapping files.

---

## 2. Species Data (in `~/.codonscope/data/species/`)

### Yeast (S. cerevisiae)
- 6,685 validated ORFs from SGD (`orf_coding_all.fasta.gz`)
- Mitochondrial ORFs (Q0*) excluded from backgrounds
- Gene ID map columns: systematic_name, common_name, sgd_id, cds_length, gc_content, status
- Additional mappings: `uniprot_mapping.tsv` (UniProt SwissProt via BioMart)
- tRNA: 3-strategy GtRNAdb parser + hardcoded fallback (~275 genes)
- Expression: `expression_rich_media.tsv` — hardcoded TPM estimates (RP 3000, glycolytic 500-5000, median 15, dubious 0.5)

### Human
- 19,229 validated CDS from MANE Select v1.5 (NCBI summary + Ensembl CDS FASTA)
- Gene ID map columns: systematic_name (ENSG), common_name (HGNC symbol), ensembl_transcript, refseq_transcript, entrez_id, cds_length, gc_content
- Additional mappings: `uniprot_mapping.tsv` (UniProt SwissProt via BioMart)
- tRNA: 3-strategy GtRNAdb parser + hardcoded fallback (~610 genes)
- Expression: GTEx v8 median TPM per tissue (`expression_gtex.tsv.gz`, ~56K genes × 54 tissues)
- CCLE: `expression_ccle.tsv.gz` + `ccle_cell_lines.tsv` (DepMap cell line expression, downloadable)

### Mouse (M. musculus)
- ~21,500 validated CDS from Ensembl GRCm39 (BioMart canonical transcripts preferred, longest CDS fallback)
- Gene ID map columns: systematic_name (ENSMUSG), common_name (MGI symbol), ensembl_transcript (ENSMUST), entrez_id, cds_length, gc_content
- Additional mappings: `mgi_mapping.tsv` (MGI IDs, synonyms, Entrez from JAX), `uniprot_mapping.tsv` (UniProt SwissProt via BioMart)
- tRNA: 3-strategy GtRNAdb parser + hardcoded fallback (~430 genes), same mammalian counts as human
- Expression: `expression_estimates.tsv` — hardcoded TPM estimates (RP 3000, Actb 5000, Gapdh 3000, median 15)

### Orthologs (in `~/.codonscope/data/orthologs/`)
- `human_yeast.tsv` — 830 pairs via name matching + curated renames
- `human_mouse.tsv` — Ensembl Compara one-to-one orthologs (~16K pairs)
- `mouse_yeast.tsv` — Ensembl Compara one-to-one orthologs (~2-3K pairs)

---

## 3. Features Added Since Original Chunks 1-9

### Chunk 10: Mouse Species Support
- Full mouse CDS download from Ensembl GRCm39
- Longest valid CDS per ENSMUSG gene (later upgraded to BioMart canonical)
- Mouse ID resolution: ENSMUSG, ENSMUST, MGI symbol (Title Case)
- Hardcoded expression estimates
- tRNA: reuses human/mammalian wobble rules
- 99 RP genes (Rpl*/Rps*) found in gene_id_map
- All single-species modes (1-5) work

### Chunk 11: Enhanced ID Resolution + Mouse Data Sources
- **GtRNAdb parser fixed:** 3-strategy parser handles old, modern, and compact header formats. T→U conversion. Skips iMet/SeC/Sup/Undet. Benefits all species.
- **BioMart canonical transcripts:** Mouse CDS selection prefers Ensembl canonical transcript over longest CDS. Falls back to longest if BioMart fails.
- **Multi-type ID resolution:** Auto-detects input ID type by regex pattern. 8-step priority resolution: Ensembl gene → Ensembl transcript → yeast systematic → species-specific (SGD, MGI, RefSeq NM_) → Entrez → UniProt → gene symbol → alias/synonym. Mixed ID types per gene list supported. Debug logging for detected types.
- **New mapping files:** UniProt (all species), MGI mapping (mouse), SGD ID capture (yeast), Entrez IDs (mouse via BioMart)
- **Ensembl Compara orthologs:** Mouse-human and mouse-yeast via bulk FTP download. Auto-discovers release number. One-to-one orthologs only.
- **HGNC alias lookup:** Human genes can be found by HGNC aliases (alternative names)
- **MGI synonym lookup:** Mouse genes can be found by MGI synonyms

### Chunk 12: Gene Name Display, Full Results Export, Colab Fixes
- **`get_common_names()`** added to `SequenceDB` — maps systematic names to gene symbols
- **Context-specific unmapped warnings** — ENSG IDs get "not in MANE Select (may be non-coding, mitochondrial, pseudogene, or retired)", ENST gets "transcript not in MANE Select", ENSMUSG gets "not in Ensembl canonical CDS set"
- **Gene ID mapping table** in HTML report gene summary — shows Input ID → Gene Name → Ensembl/Systematic ID, collapsible `<details>` if >20 genes
- **`gene_mapping.tsv`** exported to data directory with all ID mappings
- **Results zip export** — `{stem}_results.zip` created alongside HTML, containing HTML report, data/*.tsv files, and README.txt
- **README.txt in zip** — documents analysis parameters (species, expression source, bootstrap settings, seed), full input gene list, unmapped IDs, per-file descriptions, analysis mode explanations, and reference data sources
- **Expanded HTML table limits** — Mode 1: 15→30 rows, Mode 2: demand codons 15→30 and top genes 10→20, Mode 4 FS dicodons: 20→30, Mode 5 attribution and drivers: show all significant, Mode 3 ramp slow codons: show all. "Full results in data/ directory" note added to all sections.
- **Colab widget fix** — `enable_custom_widget_manager()` added to install cell for ipywidgets Textarea rendering
- **Colab zip download** — report cell downloads zip instead of bare HTML
- **Colab notebook UX** — tissue/cell line dropdowns (GTEx tissues + common cell lines), parameter descriptions for KMER_SIZE (mono/di/tricodon), BACKGROUND (all vs matched), METHOD (wtai vs tai), expanded interpretation guide with parameter table, expression file format example

### Earlier Feature Additions (Chunks 7-9)
- **CCLE cell line expression:** `download_ccle_expression()` downloads DepMap data. `--cell-line` CLI flag (e.g., HEK293T, HeLa, K562). Human only.
- **Amino acid annotations:** `annotate_kmer()` adds AA labels to k-mers in results (e.g., "AAG (Lys)").
- **Ramp codon analysis:** Mode 3 includes `ramp_composition` and `body_composition` DataFrames breaking down codon usage in ramp vs body regions.
- **Per-dicodon FS enrichment:** Mode 4 `fs_dicodons` DataFrame shows which specific Fast→Slow dicodon transitions are most enriched.
- **Gene name display:** Results include gene names (common names) alongside systematic IDs where applicable.
- **Custom expression file:** `--expression-file` CLI flag for user-supplied expression data.

---

## 4. Deviations from Project Spec

### Implemented differently than specified

1. **`resolve_ids` returns `IDMapping`**, not `dict`. Custom dict-like class with `.n_mapped`, `.n_unmapped`, `.unmapped`. Backwards compat: `result["mapping"]` etc. still works.
2. **`get_sequences` accepts multiple types** — IDMapping, dict, or list[str].
3. **`all_possible_kmers` has `sense_only` parameter** — default True gives 61^k sense-only kmers.
4. **`run_composition` has `kmer`/`kmer_size` aliases for `k`** — resolves first non-None.
5. **Human systematic_name is ENSG, not HGNC symbol** — ENSG is stable; HGNC symbol is `common_name`.
6. **gene_id_map.tsv columns differ by species** — `get_gene_metadata()` normalizes to 4 common columns.
7. **tRNA data uses 3-strategy parser + hardcoded fallbacks** — parser now works but fallbacks remain as backup.
8. **Mode 5 synonymous driver classification uses heuristics** — per-codon rules, not cross-family regression.
9. **Tricodon backgrounds are analytic, not bootstrap** — no per-gene matrix (would be ~17GB for human).

---

## 5. Known Bugs and Issues

### Fixed
1. **GtRNAdb parser** — Now supports 3 header formats. Previously returned 0 parsed anticodons for all species. Fixed in chunk 11.

### Active Issues
2. **MANE version hardcoded** — URL contains `v1.5`. Will 404 when NCBI releases v1.6+. Ensembl CDS URL uses `current_fasta` (auto-updates). GTEx URL updated to `adult-gtex/bulk-gex/v8/rna-seq/`.
3. **Tricodon bootstrap not available** — 226K × N_genes matrix too large. Uses analytic SE instead. Less accurate for small gene sets.
4. **Dicodon background files large** — Human: ~280MB uncompressed (19K genes × 3,721 dicodons × float32).
5. **Mode 5 driver classification is heuristic** — Simple if/else rules, not statistical regression.
6. **Yeast expression is approximate** — Hardcoded TPM estimates. RP genes at uniform 3000 TPM (real values vary). RP genes end up as ~72% of genome demand.
7. **Mouse expression is estimated** — Hardcoded TPM approximations, not tissue-specific measured data.
8. **Ortholog mapping is hybrid** — Human-yeast: name matching + 150 curated renames (830 pairs). Mouse pairs: Ensembl Compara bulk FTP.
9. **Mouse BioMart canonical fallback** — Falls back to longest CDS if BioMart returns <5000 results or fails.
10. **New mapping files need re-download** — UniProt, MGI, SGD ID, and Entrez mappings only created during fresh `download()`. Existing data directories need re-download to populate.
11. **No save-intermediates option** — Mode results are returned in-memory; no built-in option to save intermediate computation files.
12. **No ambiguous ID handling** — If a gene name maps to multiple systematic names, last match wins silently.

---

## 6. Positive Controls (Verified)

- **Yeast Gcn4 targets (~59 mapped genes):** AGA-GAA dicodon Z=3.66 (all bg), Z=4.33 (matched bg, adj_p=9.6e-4). Top dicodon enrichment is GGT-containing (glycine). Mode 5 confirms Gly AA enrichment (AA-driven). Mode 4 shows FS transitions present.
- **Yeast ribosomal proteins (~114 mapped genes):** Strong monocodon and dicodon bias. Mode 5: synonymous-driven RSCU deviations (translational selection). Mode 3: high optimality with visible ramp. Mode 4: high FF proportion, low FS enrichment (≤1.1). Mode 2: optimal codons (AAG, AGA, GCT) enriched in demand; rare codons (AGT, CTG, ATA) depleted. Z-scores moderate (~1.7) because RP genes dominate ~72% of genome demand.
- **Human ribosomal proteins (14 genes):** Mode 1 monocodon shows significant codon bias.
- **Cross-species RP orthologs:** Low RSCU correlation (mean r~0.13) between yeast and human RP genes, confirming different preferred codons (17/18 AAs). Both directions work.
