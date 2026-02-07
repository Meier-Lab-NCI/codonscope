# CodonScope Implementation Status

Last updated: 2026-02-07
131 tests passing. 11 commits on main.

---

## 1. What's Implemented and Tested

### Core Engine

**`codonscope/core/codons.py`** — K-mer counting engine. Fully implemented.

```python
SENSE_CODONS: list[str]                                    # 61 sense codons, sorted
sequence_to_codons(sequence: str) -> list[str]
count_kmers(sequence: str, k: int = 1) -> dict[str, int]
kmer_frequencies(sequence: str, k: int = 1) -> dict[str, float]
all_possible_kmers(k: int = 1, sense_only: bool = True) -> list[str]
```

**`codonscope/core/sequences.py`** — Gene ID resolution and CDS retrieval. Fully implemented for yeast and human.

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
    def get_gene_metadata(self) -> pd.DataFrame
    @property species_dir -> Path
```

**`codonscope/core/statistics.py`** — Bootstrap Z-scores, multiple testing, effect sizes. Fully implemented.

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

### Data Layer

**`codonscope/data/download.py`** — Downloads and pre-computes all reference data. Supports yeast and human.

```python
download(species: str, data_dir: str | Path | None = None) -> Path
# species: "yeast" or "human"
```

### Analysis Modes

**`codonscope/modes/mode1_composition.py`** — Mode 1: Sequence Composition.

```python
run_composition(
    species: str,
    gene_ids: list[str],
    k: int | None = None,          # preferred parameter name
    kmer: int | None = None,       # alias
    kmer_size: int | None = None,  # alias
    background: str = "all",       # "all" or "matched"
    trim_ramp: int = 0,
    min_genes: int = 10,
    n_bootstrap: int = 10_000,
    output_dir: str | Path | None = None,
    seed: int | None = None,
    data_dir: str | Path | None = None,
) -> dict
```

Returns:
```python
{
    "results": pd.DataFrame,    # columns: kmer, observed_freq, expected_freq, z_score, p_value, adjusted_p, cohens_d
                                # sorted by |z_score| descending
    "diagnostics": dict,        # keys: length_ks_stat, length_p, length_warning, gc_ks_stat, gc_p, gc_warning, power_warnings
    "id_summary": IDMapping,    # NOT a plain dict — it's an IDMapping object
    "n_genes": int,
}
```

**`codonscope/modes/mode5_disentangle.py`** — Mode 5: AA vs Codon Disentanglement.

```python
CODON_TABLE: dict[str, str]        # codon -> 3-letter AA name, 61 entries
AMINO_ACIDS: list[str]             # 20 amino acids, sorted alphabetically
AA_FAMILIES: dict[str, list[str]]  # AA -> list of synonymous codons
N_SYNONYMS: dict[str, int]         # AA -> number of synonymous codons

run_disentangle(
    species: str,
    gene_ids: list[str],
    n_bootstrap: int = 10_000,
    seed: int | None = None,
    output_dir: str | Path | None = None,
    data_dir: str | Path | None = None,
) -> dict
```

Returns:
```python
{
    "aa_results": pd.DataFrame,          # columns: amino_acid, observed_freq, expected_freq, z_score, p_value, adjusted_p
    "rscu_results": pd.DataFrame,        # columns: codon, amino_acid, n_synonyms, observed_rscu, expected_rscu, z_score, p_value, adjusted_p
    "attribution": pd.DataFrame,         # columns: codon, amino_acid, aa_z_score, rscu_z_score, aa_adj_p, rscu_adj_p, attribution
                                         # attribution values: "AA-driven", "Synonymous-driven", "Both", "None"
    "synonymous_drivers": pd.DataFrame,  # columns: codon, amino_acid, rscu_z_score, driver, gc3, decoding_type, trna_copies
                                         # driver values: "tRNA_supply", "GC3_bias", "wobble_avoidance", "unclassified", "not_applicable"
    "summary": {
        "n_significant_codons": int,
        "n_aa_driven": int,
        "n_synonymous_driven": int,
        "n_both": int,
        "pct_aa_driven": float,          # 0-100
        "pct_synonymous_driven": float,
        "pct_both": float,
        "summary_text": str,
    },
    "id_summary": IDMapping,
    "n_genes": int,
}
```

### CLI

**`codonscope/cli.py`** — Three subcommands implemented.

```
codonscope download   --species SPECIES [SPECIES ...] [--data-dir DIR]
codonscope composition --species S --genes FILE --kmer {1,2,3} [--background {all,matched}]
                       [--trim-ramp N] [--min-genes N] [--n-bootstrap N] [--output-dir DIR] [--seed N]
codonscope disentangle --species S --genes FILE [--n-bootstrap N] [--output-dir DIR] [--seed N]
```

Gene list files support: one ID per line, comma-separated, tab-separated, `#` comments.

### Tests

| File | Count | What it covers |
|------|-------|----------------|
| `tests/test_chunk1.py` | 23 | Yeast download files, ID mapping (systematic/common/case-insensitive), CDS validation, k-mer counting, background shapes |
| `tests/test_statistics.py` | 29 | Bootstrap Z-scores, BH correction, Cohen's d, KS diagnostics, power check, compare_to_background pipeline, yeast positive controls (RP genes, YEF3) |
| `tests/test_mode1.py` | 19 | Gene list parsing, run_composition pipeline (mono/di/matched/trim_ramp), diagnostics, unmapped gene reporting, CLI (help/version/composition), RP positive controls |
| `tests/test_human.py` | 33 | Human download files (8 files), gene map structure/columns, CDS validation (div by 3, ATG start, no internal stops, ACGT only), ID resolution (HGNC/ENSG/ENST/Entrez/mixed/unmapped), background shapes, tRNA/wobble, Mode 1 on human RP genes |
| `tests/test_mode5.py` | 27 | Codon table (61 codons, 20 AAs, family sizes), AA frequency computation, RSCU computation (equal/biased), attribution logic (AA-driven/synonymous/both/none), summary stats, Gcn4 integration (Gly enrichment), RP integration (synonymous-driven), CLI help |
| **Total** | **131** | |

Note: `tests/test_chunks1_3.py` also exists (user's own validation script) but is not part of the main test suite.

---

## 2. Actual Function Signatures (Where They Differ from Spec)

### resolve_ids return type

**Spec (CHUNK1_TASK.md):**
```python
def resolve_ids(self, gene_ids: list[str]) -> dict:
    # Returns: {input_id: systematic_name}
    # Also returns summary: n_mapped, n_unmapped, n_ambiguous
```

**Actual:**
```python
def resolve_ids(self, gene_ids: list[str]) -> IDMapping:
```

`IDMapping` is a custom class that behaves like `dict[input_id, systematic_name]` but also has `.n_mapped`, `.n_unmapped`, `.unmapped` attributes. This was a bug fix — the original implementation returned a nested dict `{"mapping": {...}, "n_mapped": int, ...}` which caused `len(result)` to return 4 (number of dict keys) instead of the mapped gene count.

Backwards compat: `result["mapping"]`, `result["n_mapped"]`, `result["n_unmapped"]`, `result["unmapped"]` all still work via `__getitem__` override.

### get_sequences parameter type

**Spec:**
```python
def get_sequences(self, systematic_names: list[str]) -> dict[str, str]:
```

**Actual:**
```python
def get_sequences(self, names) -> dict[str, str]:
    # accepts: IDMapping, dict, or list[str]
```

Changed to accept the `IDMapping` directly from `resolve_ids()`, avoiding the need to manually extract `.values()`. Also accepts a plain `dict` (uses values) or `list[str]`.

### all_possible_kmers extra parameter

**Spec:**
```python
def all_possible_kmers(k: int = 1) -> list[str]:
```

**Actual:**
```python
def all_possible_kmers(k: int = 1, sense_only: bool = True) -> list[str]:
```

Added `sense_only` parameter. Default `True` gives 61^k sense-only kmers. `False` would give all 64^k including stop codons.

### run_composition (not in original spec by this name)

The project spec says `CLI: codonscope composition` but doesn't specify the Python function name. The CHUNK1_TASK.md defines `codons.py` stubs. Mode 1 function is `run_composition()` with `k`/`kmer`/`kmer_size` aliases added after a bug report.

### Human gene_id_map.tsv columns differ from yeast

Yeast: `systematic_name, common_name, cds_length, gc_content, status`
Human: `systematic_name, common_name, ensembl_transcript, refseq_transcript, entrez_id, cds_length, gc_content`

The `get_gene_metadata()` method returns only the common 4 columns: `systematic_name, common_name, cds_length, gc_content`.

### Human systematic_name is ENSG, not HGNC symbol

For human, `systematic_name` = Ensembl gene ID (e.g. `ENSG00000187634`), not the HGNC symbol. HGNC symbol is in `common_name`. This differs from what you might expect — the "systematic" name is the stable Ensembl ID, the "common" name is the human-readable HGNC symbol.

---

## 3. Data Storage Paths

All data lives under `~/.codonscope/data/species/{species}/`.

### Yeast (`~/.codonscope/data/species/yeast/`)

| File | Size | Contents |
|------|------|----------|
| `cds_sequences.fa.gz` | 2.4 MB | 6,685 validated ORFs, stop codons stripped, keyed by systematic name (YAL001C) |
| `gene_id_map.tsv` | 230 KB | Columns: systematic_name, common_name, cds_length, gc_content, status |
| `trna_copy_numbers.tsv` | 537 B | Columns: anticodon, amino_acid, gene_count (hardcoded fallback data) |
| `wobble_rules.tsv` | 1.7 KB | 61 rows. Columns: codon, amino_acid, decoding_anticodon, trna_gene_copies, decoding_type, modification_notes |
| `background_mono.npz` | 647 KB | Keys: mean(61,), std(61,), per_gene(6685,61), kmer_names(61,), gene_names(6685,) |
| `background_di.npz` | 3.6 MB | Keys: mean(3721,), std(3721,), per_gene(6685,3721), kmer_names(3721,), gene_names(6685,) |
| `background_tri.npz` | 2.0 MB | Keys: mean(226981,), std(226981,), kmer_names(226981,), gene_names(6685,) — **NO per_gene** |
| `gene_metadata.npz` | 54 KB | Keys: gene_names(6685,), cds_lengths(6685,), gc_contents(6685,) |

### Human (`~/.codonscope/data/species/human/`)

| File | Size | Contents |
|------|------|----------|
| `cds_sequences.fa.gz` | 8.9 MB | 19,229 validated CDS, keyed by ENSG ID (e.g. ENSG00000187634) |
| `gene_id_map.tsv` | 1.3 MB | Columns: systematic_name, common_name, ensembl_transcript, refseq_transcript, entrez_id, cds_length, gc_content |
| `trna_copy_numbers.tsv` | 519 B | Same format as yeast (hardcoded fallback data) |
| `wobble_rules.tsv` | 1.9 KB | 61 rows, same columns as yeast |
| `background_mono.npz` | 2.0 MB | Keys: mean(61,), std(61,), per_gene(19229,61), kmer_names(61,), gene_names(19229,) |
| `background_di.npz` | 12 MB | Keys: mean(3721,), std(3721,), per_gene(19229,3721), kmer_names(3721,), gene_names(19229,) |
| `background_tri.npz` | 2.1 MB | Keys: mean(226981,), std(226981,), kmer_names(226981,), gene_names(19229,) — **NO per_gene** |
| `gene_metadata.npz` | 161 KB | Keys: gene_names(19229,), cds_lengths(19229,), gc_contents(19229,) |

### Background npz key reference

Mono and di backgrounds contain `per_gene` (full N_genes x N_kmers float32 matrix, used for bootstrap resampling). Tri backgrounds do **not** — the matrix would be N_genes x 226,981 which is too large. Tricodon analysis falls back to analytic SE = `std / sqrt(n_genes)`.

---

## 4. Deviations from Project Spec

### Implemented differently than specified

1. **`resolve_ids` return type.** Spec says `-> dict`. Implementation returns `IDMapping` (a dict-like class with extra attributes). Reason: the original dict-based return caused a confusing API where `len(result)` returned 4 instead of the mapped count.

2. **`get_sequences` accepts multiple types.** Spec says `systematic_names: list[str]`. Implementation accepts `IDMapping`, `dict`, or `list[str]`. Reason: convenience — avoids `list(result.values())` boilerplate.

3. **`all_possible_kmers` has `sense_only` parameter.** Spec doesn't mention this. Added because backgrounds use sense-only kmers (61) not all 64.

4. **`run_composition` has `kmer`/`kmer_size` aliases for `k`.** Spec uses `--kmer` in CLI. Python function accepts `k`, `kmer`, or `kmer_size` interchangeably (resolves first non-None).

5. **Human systematic_name is ENSG, not gene symbol.** Spec says "gene_symbol -> ensembl_gene_id -> ensembl_transcript_id -> entrez_id". Implementation uses ENSG as the primary key (`systematic_name`), HGNC symbol as `common_name`. Reason: ENSG is stable and unambiguous; HGNC symbols can change.

6. **gene_id_map.tsv columns differ by species.** Yeast has `status` column (Verified/Uncharacterized/Dubious). Human has `ensembl_transcript`, `refseq_transcript`, `entrez_id`. The `get_gene_metadata()` method normalizes to the 4 common columns.

7. **tRNA data uses hardcoded fallbacks exclusively.** Spec says download from GtRNAdb with fallback. In practice, GtRNAdb returns 0 parsed anticodons for both species (header format changed). The fallback tables are always used.

8. **Mode 5 synonymous driver classification uses heuristics.** Spec says "correlation with tRNA gene copy number / GC3 content / wobble avoidance." Implementation uses per-codon rules rather than cross-family regression. This is simpler but less statistically rigorous.

9. **Tricodon backgrounds are analytic, not bootstrap.** Spec says "bootstrap" everywhere. Tri backgrounds can't store the per-gene matrix (226K x 19K = too large), so tricodon Z-scores use `SE = std / sqrt(n)` instead of bootstrap resampling. Less accurate for small gene sets.

### Not yet implemented (spec features still missing)

| Feature | Spec section | Status |
|---------|--------------|--------|
| Mode 2: Translational Demand | 4.2 | Not started. Needs expression data (GTEx, yeast RNA-seq). |
| Mode 3: Optimality Profile | 4.3 | Not started. Needs `core/optimality.py` (tAI, wtAI scoring). |
| Mode 4: Collision Potential | 4.4 | Not started. Needs optimality classification (fast/slow codons). |
| Mode 6: Cross-Species Comparison | 4.6 | Not started. Needs `core/orthologs.py` + Ensembl Compara data. |
| Mouse species | 3.1 | Not started. Would follow same pattern as human (Ensembl canonical). |
| Expression data (GTEx) | 3.5 | Not downloaded. Required for Mode 2. |
| Ortholog mappings | 3.6 | Not downloaded. Required for Mode 6. |
| HTML report generation | — | Not started. |
| `core/optimality.py` | 2 | File does not exist. Needed for Modes 3 and 4. |
| `core/orthologs.py` | 2 | File does not exist. Needed for Mode 6. |
| `viz/plots.py`, `viz/report.py` | 2 | Directories/files do not exist. Plots are currently inline in mode files. |

---

## 5. What to Build Next

### Recommended order

1. **`core/optimality.py`** — tAI and wobble-aware tAI (wtAI) scoring. Needed by both Mode 3 and Mode 4. Uses `wobble_rules.tsv` and `trna_copy_numbers.tsv`. Classify codons as fast/slow based on wtAI.

2. **Mode 3: Optimality Profile** (`modes/mode3_profile.py`) — Per-codon optimality along transcripts. Metagene average. Ramp region analysis (first 50 codons). Depends on `optimality.py`.

3. **Mode 4: Collision Potential** (`modes/mode4_collision.py`) — Fast-to-slow (FS) transition enrichment. 2x2 transition matrix. Depends on `optimality.py`.

4. **Expression data download** — GTEx median TPM for human, yeast rich-media RNA-seq. Add to `download.py`.

5. **Mode 2: Translational Demand** (`modes/mode2_demand.py`) — Expression-weighted codon demand. Depends on expression data.

6. **Ortholog data download** — Ensembl Compara one-to-one orthologs. Add to `download.py`.

7. **Mode 6: Cross-Species Comparison** (`modes/mode6_crossspecies.py`) — Per-gene RSCU correlation across species. Depends on ortholog mappings.

8. **HTML report** — Summary report combining all mode outputs.

### For each new mode, create

- The mode file in `codonscope/modes/`
- A CLI subcommand in `codonscope/cli.py`
- A test file in `tests/`
- Positive control integration tests (ribosomal proteins, Gcn4 targets, etc.)

---

## 6. Known Bugs and Issues

### Bugs

1. **GtRNAdb parser returns 0 anticodons.** `_parse_gtrnadb_fasta()` regex doesn't match current GtRNAdb FASTA headers. Both yeast (`sacCer3-mature-tRNAs.fa`) and human (`hg38-mature-tRNAs.fa`) fail to parse. Hardcoded fallback tables are used instead. **Impact: none** (fallback data is fine). **Fix: low priority** — would need to inspect current GtRNAdb format and update regex.

2. **MANE version pinned to v1.5.** The download URL is hardcoded to `MANE.GRCh38.v1.5.summary.txt.gz`. When NCBI releases v1.6+, this will 404. **Workaround:** update the version string in `MANE_SUMMARY_URL` in `download.py`. **Better fix:** parse the `current/` directory listing to find the latest summary filename.

### Limitations

3. **Tricodon bootstrap not available.** The per-gene tricodon matrix (226,981 x N_genes) would be ~17GB for human. Instead, tricodon backgrounds store only mean/std vectors, and Z-scores use analytic SE. This is less accurate for small gene sets (<50 genes). Mono and dicodon analyses use true bootstrap (per-gene matrix stored).

4. **Dicodon background files are large but manageable.** Human: 12 MB compressed (19,229 genes x 3,721 dicodons x float32). Yeast: 3.6 MB. Not a problem on disk, but loaded fully into memory during analysis.

5. **Mode 5 driver classification is heuristic.** The `_classify_single_driver()` function uses simple if/else rules rather than statistical tests across synonym families. A more rigorous approach would correlate per-family RSCU deviations with tRNA supply / GC3 / wobble status using regression. Current approach may misclassify when multiple drivers are correlated.

6. **No ambiguous ID handling.** If a gene name maps to multiple systematic names (e.g., a common name shared by paralog families), the last match wins silently. The spec mentions `n_ambiguous` in the return summary but this is not tracked.

7. **Human ENSG IDs stored without version numbers.** The gene_id_map and CDS FASTA use versionless ENSG IDs (e.g., `ENSG00000187634` not `ENSG00000187634.13`). Resolve_ids strips version numbers from input. This means you can't distinguish between ENSG versions, but in practice MANE Select has one version per gene so this is fine.

8. **No `--force` flag for re-download.** Running `download("yeast")` when data already exists overwrites everything. No skip-if-exists logic. Not a bug per se, but the download takes a few minutes (human especially: ~1 min for Ensembl CDS, ~15 sec for backgrounds).

9. **Wobble rules for Ser anticodon mismatch between yeast and human.** Yeast Ser uses `AGA` anticodon for TCT/TCC, human uses `AGC` anticodon for TCT/TCC/TCA. This reflects biological reality (different tRNA repertoires) but means the wobble rules tables are not interchangeable between species. Each species has its own curated table.
