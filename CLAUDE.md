# CodonScope

Multi-species codon usage analysis tool for gene lists. Replaces the defunct CUT tool (Doyle et al., Methods 2016). Analyzes mono-, di-, and tricodon patterns in user gene lists vs genomic backgrounds.

## Full spec

Read `CodonScope_Project_Spec.md` for the complete project specification including all 6 analysis modes, positive controls, and architecture.

## Current status

**215 tests passing. 14 commits on main.**

### Build progress

| # | Chunk | Status |
|---|-------|--------|
| 1 | Data layer + yeast | ✅ Done |
| 2 | Core counting engine + statistics | ✅ Done |
| 3 | Mode 1 (composition) + CLI | ✅ Done |
| 4 | Human species support | ✅ Done |
| 5 | Mode 5 (AA vs codon disentanglement) | ✅ Done |
| 6 | Mode 3 (optimality profile) + Mode 4 (collision potential) | ✅ Done |
| 7 | Mode 2 (translational demand) | ✅ Done |
| 8 | Mode 6 (cross-species comparison) | ← Next |
| 9 | HTML report generation | Planned |

### What exists (files and what they do)

**Core engine:**
- `codonscope/__init__.py` — version 0.1.0
- `codonscope/core/codons.py` — k-mer counting engine (mono/di/tri), `sequence_to_codons()`, `count_kmers()`, `kmer_frequencies()`, `all_possible_kmers()`, `SENSE_CODONS` (61 sense codons)
- `codonscope/core/sequences.py` — `SequenceDB` class with lazy CDS loading, `IDMapping` result class. Resolves yeast (systematic name, common name) and human (HGNC symbol, ENSG, ENST, Entrez ID) gene identifiers. `get_sequences()` accepts IDMapping, dict, or list[str].
- `codonscope/core/statistics.py` — `compute_geneset_frequencies()`, `bootstrap_zscores()` (vectorized numpy, chunked), `bootstrap_pvalues()`, `benjamini_hochberg()`, `cohens_d()`, `power_check()`, `diagnostic_ks_tests()`, `compare_to_background()` full pipeline
- `codonscope/core/optimality.py` — `OptimalityScorer` class: loads wobble_rules.tsv, computes per-codon tAI and wtAI weights (normalised 0–1, pseudocount for zero-copy anticodons). Provides `gene_tai()`, `gene_wtai()`, `per_position_scores()`, `smooth_profile()`, `classify_codons()` (fast/slow by median threshold).

**Data layer:**
- `codonscope/data/download.py` — `download("yeast")` and `download("human")`. Downloads CDS sequences, tRNA copy numbers (GtRNAdb with hardcoded fallbacks), wobble decoding rules, pre-computes mono/di/tri backgrounds, creates expression data. Curated wobble rule tables for both species. Also: `download_expression()` standalone, yeast expression from hardcoded rich-media estimates, human expression from GTEx v8 median TPM.

**Analysis modes:**
- `codonscope/modes/mode1_composition.py` — `run_composition()` with "all" or "matched" (length+GC) backgrounds, KS diagnostics, volcano + bar chart plots. Accepts k=1/2/3 (aliases: `kmer`, `kmer_size`).
- `codonscope/modes/mode3_profile.py` — `run_profile()` metagene optimality profiling. Per-position tAI/wtAI scoring with sliding window, normalised to 100 positional bins. Ramp analysis (first N codons vs body). Compares gene set metagene to genome background. Two-panel plot (metagene line + ramp bar chart).
- `codonscope/modes/mode4_collision.py` — `run_collision()` ribosome collision potential. Classifies codons as fast/slow via wtAI median threshold, counts FF/FS/SF/SS dicodon transitions. Reports FS enrichment ratio, FS/SF ratio, chi-squared test vs genome. Per-gene FS fractions and positional FS clustering. Two-panel plot (transition bars + positional FS).
- `codonscope/modes/mode2_demand.py` — `run_demand()` translational demand analysis. Weights codon frequencies by TPM × n_codons. Weighted bootstrap Z-scores. Reports top demand-contributing genes. Supports yeast (rich-media expression) and human (GTEx tissue-specific). Options: --tissue, --expression (custom), --top-n.
- `codonscope/modes/mode5_disentangle.py` — `run_disentangle()` with two-layer decomposition (AA composition + RSCU), attribution (AA-driven / synonymous-driven / both), synonymous driver classification (tRNA supply, GC3 bias, wobble avoidance), two-panel plot.

**CLI:**
- `codonscope/cli.py` — argparse with subcommands: `download`, `composition`, `demand`, `profile`, `collision`, `disentangle`. Parses gene list files (one-per-line, comma-separated, tab-separated, # comments).

**Tests (215 passing):**
- `tests/test_chunk1.py` — 23 tests: yeast download files, ID mapping, CDS validation, k-mer counting, backgrounds
- `tests/test_statistics.py` — 29 tests: bootstrap, BH correction, Cohen's d, KS diagnostics, yeast positive controls (RP genes, YEF3)
- `tests/test_mode1.py` — 19 tests: gene list parsing, composition pipeline, matched background, diagnostics, CLI, RP positive controls
- `tests/test_human.py` — 33 tests: human download files, gene map structure, CDS validation, ID resolution (HGNC/ENSG/ENST/Entrez), backgrounds, Mode 1 on human RP genes
- `tests/test_mode3.py` — 26 tests: OptimalityScorer (13 unit tests), metagene profile shape/values, ramp analysis, yeast RP integration (high optimality + ramp), random gene negative control, CLI
- `tests/test_mode4.py` — 22 tests: transition counting (FF/FS/SF/SS), proportions, FS enrichment, chi-squared, yeast RP integration (low collision, high FF), Gcn4 comparison, CLI
- `tests/test_mode2.py` — 36 tests: demand vector unit tests, expression loading (yeast + human GTEx), weighted bootstrap, yeast RP demand (optimal codons enriched), Gcn4 demand, human liver demand, CLI
- `tests/test_mode5.py` — 27 tests: codon table, AA frequencies, RSCU, attribution logic, summary, Gcn4 + RP integration tests, CLI

### Species data (in `~/.codonscope/data/species/`)

**Yeast (S. cerevisiae):**
- 6,685 validated ORFs from SGD (`orf_coding_all.fasta.gz`)
- Mitochondrial ORFs (Q0*) excluded from backgrounds
- tRNA: hardcoded fallback (GtRNAdb parse fails)
- Expression: `expression_rich_media.tsv` — hardcoded TPM estimates (RP genes 3000, glycolytic 500-5000, median 15, dubious 0.5)
- Files: `cds_sequences.fa.gz`, `gene_id_map.tsv`, `trna_copy_numbers.tsv`, `wobble_rules.tsv`, `background_{mono,di,tri}.npz`, `gene_metadata.npz`, `expression_rich_media.tsv`

**Human:**
- 19,229 validated CDS from MANE Select v1.5 (NCBI summary + Ensembl CDS FASTA)
- Gene ID map includes: systematic_name (ENSG), common_name (HGNC symbol), ensembl_transcript, refseq_transcript, entrez_id
- tRNA: hardcoded fallback (GtRNAdb parse fails)
- Expression: GTEx v8 median TPM per tissue (`expression_gtex.tsv.gz`), ~56K genes × 54 tissues
- Same file structure as yeast plus `expression_gtex.tsv.gz`

### Positive controls (verified)

- **Yeast Gcn4 targets (~59 mapped genes):** AGA-GAA dicodon Z=3.66 (all bg), Z=4.33 (matched bg, adj_p=9.6e-4). Top dicodon enrichment is GGT-containing (glycine). Mode 5 confirms Gly AA is enriched (AA-driven signal). Mode 4 shows FS transitions present.
- **Yeast ribosomal proteins (~114 mapped genes):** Strong monocodon and dicodon bias. Mode 5 confirms synonymous-driven RSCU deviations (translational selection). Mode 3 shows high optimality throughout with visible ramp. Mode 4 shows high FF proportion and low FS enrichment (≤1.1). Mode 2 shows optimal codons (AAG, AGA, GCT) enriched in demand vs genome; rare codons (AGT, CTG, ATA) depleted. Z-scores moderate (~1.7) because RP genes dominate ~72% of genome demand.
- **Human ribosomal proteins (14 genes):** Mode 1 monocodon shows significant codon bias.

### Known issues

1. **GtRNAdb parser broken.** Both yeast and human tRNA downloads return 0 parsed anticodons. The GtRNAdb FASTA header format may have changed. Hardcoded fallback tables work fine. Low priority — fix if someone needs live tRNA data.
2. **MANE version hardcoded.** URL contains `v1.5`. When NCBI releases v1.6+, the download URL will 404 and need updating. The Ensembl CDS URL uses `current_fasta` (auto-updates). GTEx URL updated to `adult-gtex/bulk-gex/v8/rna-seq/` (old `gtex_analysis_v8` bucket returned 404).
3. **Tricodon backgrounds have no per-gene matrix.** The 226,981 × N_genes matrix is too large. Tricodon analysis uses analytic SE (`std / sqrt(n_genes)`) instead of bootstrap. This is less accurate for small gene sets.
4. **Dicodon background files are large.** Human di background is ~280MB (19K genes × 3,721 dicodons × float32). Consider sparse storage if disk space is a concern.
5. **Mode 5 synonymous driver classification is heuristic.** Uses simple rules (Watson-Crick + enriched → wobble avoidance, high tRNA copies + enriched → tRNA supply). A more rigorous approach would use per-family regression.
6. **Yeast expression is approximate.** Hardcoded TPM estimates, not measured. RP genes assigned uniform 3000 TPM (real values vary by paralog). RP genes end up as ~72% of genome demand; real fraction is ~30-40%. This makes Mode 2 Z-scores modest for RP genes since they dominate both gene-set and genome demand. GTEx human expression is real measured data.
7. **No ortholog data yet.** Mode 6 (cross-species comparison) requires Ensembl Compara. Not downloaded.
8. **`orthologs.py` not yet created.** Needed for Mode 6.

## Design decisions (read before coding)

1. **One canonical transcript per gene.** Human = MANE Select. Mouse = Ensembl canonical. Yeast = SGD verified ORF. Never use all isoforms — inflates statistics.
2. **Frequencies computed per-gene then averaged.** Not pooled across genes. This treats genes equally regardless of CDS length.
3. **Dicodons/tricodons use sliding windows.** Positions 1-2, 2-3, 3-4... NOT non-overlapping pairs.
4. **Strip stop codon before counting.** Validate CDSs: divisible by 3, starts with ATG, ends with stop. Stop codon stripped after validation.
5. **Bootstrap Z-scores, not analytic.** Resample N genes from genome 10,000 times for SE estimation. Exception: tricodons use analytic SE (no per-gene matrix).
6. **Benjamini-Hochberg correction** across all k-mers within each mode.
7. **IDMapping class.** `resolve_ids()` returns an `IDMapping` object that behaves like `dict[input_id → systematic_name]`. Has `.unmapped`, `.n_mapped`, `.n_unmapped`. Backwards compat: `result["mapping"]` etc. still works.
8. **Species-dispatched ID resolution.** Yeast uses regex for systematic names + case-insensitive common name lookup. Human uses ENSG/ENST regex, numeric Entrez lookup, and case-insensitive HGNC symbol lookup.
9. **Mode 2 demand weighting.** Weight = TPM × n_codons per gene. Demand vector = weighted mean of per-gene codon frequencies. Weighted bootstrap: sample N genes, compute their weighted demand, repeat 10K times. Human uses GTEx tissue-specific TPM. Yeast uses hardcoded rich-media estimates.

## How to run

```bash
# Download reference data
python3 -c "from codonscope.data.download import download; download('yeast')"
python3 -c "from codonscope.data.download import download; download('human')"

# Run tests
python3 -m pytest tests/ -v

# Mode 1: Composition analysis
python3 -m codonscope.cli composition --species yeast --genes genelist.txt --kmer 2

# Mode 2: Translational demand (expression-weighted)
python3 -m codonscope.cli demand --species yeast --genes genelist.txt
python3 -m codonscope.cli demand --species human --genes genelist.txt --tissue liver

# Mode 3: Optimality profile (metagene + ramp)
python3 -m codonscope.cli profile --species yeast --genes genelist.txt --method wtai

# Mode 4: Collision potential (FS transitions)
python3 -m codonscope.cli collision --species yeast --genes genelist.txt

# Mode 5: Disentanglement
python3 -m codonscope.cli disentangle --species yeast --genes genelist.txt
```

## Key dependencies

```
python >= 3.9
numpy, scipy, pandas, matplotlib, requests, tqdm
```

No BioPython, no pysam. Keep it lightweight.

export PATH="$HOME/.local/bin:$PATH"
