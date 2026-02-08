# CodonScope Pilot Gene Lists

Curated gene lists for testing and demonstrating CodonScope's 6 analysis modes.
Each file contains one gene per line with `#` comment headers describing the source,
citation, and expected CodonScope results.

## Gene Lists

### Yeast (*S. cerevisiae*)

| File | Genes | Description |
|------|-------|-------------|
| `yeast_gcn4_targets.txt` | 54 | Gcn4-regulated amino acid biosynthesis genes |
| `yeast_ribosomal_proteins.txt` | 132 | All cytoplasmic ribosomal proteins (both paralogs) |
| `yeast_trm4_targets.txt` | 39 | Genes with >=90% TTG leucine codon usage (Trm4/m5C targets) |
| `yeast_trm9_targets.txt` | 50 | Top AGA/GAA-enriched genes (Trm9/mcm5s2U targets) |
| `cross_species_rp_yeast.txt` | 75 | Ribosomal proteins, A-paralog only (for Mode 6 cross-species) |

### Human

| File | Genes | Description |
|------|-------|-------------|
| `human_ribosomal_proteins.txt` | 80 | Core cytoplasmic ribosomal proteins |
| `human_5top_mrnas.txt` | 115 | 5'TOP motif mRNAs (mTOR-regulated) |
| `human_proliferation_myc.txt` | 200 | HALLMARK_MYC_TARGETS_V1 (MSigDB) |
| `human_differentiation_emt.txt` | 200 | HALLMARK_EMT (MSigDB) |
| `human_secreted_proteins.txt` | 53 | Highly secreted proteins |
| `human_isr_upregulated.txt` | 18 | ISR translationally upregulated genes (negative control) |
| `human_membrane_proteins.txt` | 50 | Transmembrane proteins (Mode 5 AA-driven test) |

## References

### Primary literature

- **Natarajan K et al.** (2001) Transcriptional profiling shows that Gcn4p is a master
  regulator of gene expression during amino acid starvation in yeast. *Mol Cell Biol*
  21:4347-4368. [yeast_gcn4_targets]
- **Ikemura T** (1985) Codon usage and tRNA content in unicellular and multicellular
  organisms. *Mol Biol Evol* 2:13-34. [yeast/human ribosomal proteins]
- **Sharp PM & Li WH** (1987) The codon adaptation index. *Nucleic Acids Res*
  15:1281-1295. [yeast/human ribosomal proteins]
- **Chan CTY et al.** (2012) Reprogramming of tRNA modifications controls the oxidative
  stress response by codon-biased translation of proteins. *Nat Commun* 3:937.
  [yeast_trm4_targets]
- **Begley U et al.** (2007) Trm9-catalyzed tRNA modifications link translation to the
  DNA damage response. *Mol Cell* 28:860-870. [yeast_trm9_targets]
- **dos Reis M, Savva R & Wernisch L** (2004) Solving the riddle of codon usage
  preferences. *Nucleic Acids Res* 32:5036-5044. [human ribosomal proteins]
- **Thoreen CC et al.** (2012) A unifying model for mTORC1-mediated regulation of mRNA
  translation. *Nature* 485:109-113. [human_5top_mrnas]
- **Philippe L et al.** (2020) Global analysis of LARP1 translation targets reveals tunable
  and dynamic features of 5'TOP motifs. *Proc Natl Acad Sci USA* 117:5319-5328.
  [human_5top_mrnas]
- **Cockman E et al.** (2020) TOP mRNPs: molecular mechanisms and principles of
  regulation. *Biomolecules* 10:969. [human_5top_mrnas, review]
- **Harding HP et al.** (2000) Regulated translation initiation controls stress-induced gene
  expression in mammalian cells. *Mol Cell* 6:1099-1108. [human_isr_upregulated]
- **Pakos-Zebrucka K et al.** (2016) The integrated stress response. *EMBO Rep*
  17:1374-1395. [human_isr_upregulated]
- **Zhou M et al.** (2024) The Effects of Codon Usage on Protein Structure and Folding.
  *Annual Rev Biophys*. [human_secreted_proteins]

### Gene set databases

- **Liberzon A et al.** (2015) The Molecular Signatures Database hallmark gene set
  collection. *Cell Syst* 1:417-425. [human_proliferation_myc, human_differentiation_emt]
- **MSigDB** Hallmark gene sets v2024.1.Hs.
  https://www.gsea-msigdb.org/gsea/msigdb/

### Critical caveats

- **Pouyet F et al.** (2017) Recombination, meiotic expression and human codon usage.
  *eLife* 6:e27344. **Important**: proliferation vs differentiation codon usage differences
  in human genes are primarily driven by GC-biased gene conversion (gBGC) correlated
  with recombination rate — NOT by translational selection. The MYC targets (high GC3)
  vs EMT (low GC3) contrast demonstrates this. CodonScope Mode 5 should attribute
  these signals to "GC3 bias" rather than "tRNA supply".

## Expected Results Summary

| Gene list | Primary signal | Mode 1 | Mode 2 | Mode 3 | Mode 5 |
|-----------|---------------|--------|--------|--------|--------|
| Yeast RP | Synonymous selection | Strong bias | Dominant demand | High optimality + ramp | Synonymous-driven |
| Yeast Gcn4 | AA composition | GGT dicodon enriched | Moderate | Moderate optimality | AA-driven (Gly) |
| Yeast Trm4 | TTG enrichment | Extreme TTG | Very high demand | High optimality | Synonymous-driven |
| Yeast Trm9 | AGA/GAA enrichment | Strong AGA/GAA | Very high demand | High optimality | Synonymous-driven |
| Human RP | Mixed (GC3 + selection) | C/G-ending codons | High demand | High optimality | Mixed |
| Human 5'TOP | Translational regulation | Strong bias | Highest demand | High optimality | Mixed |
| Human MYC | GC3 bias (not selection) | C/G-ending codons | High | High | **GC3 bias** |
| Human EMT | GC3 bias (opposite) | A/T-ending codons | Variable | Lower | **GC3 bias** |
| Human secreted | AA composition | Gly codons (collagens) | Tissue-specific | Signal peptide ramp | AA-driven |
| Human ISR | **Negative control** | Unremarkable | Low | Unremarkable | Unremarkable |
| Human membrane | AA composition | Hydrophobic AA codons | Variable | Possible pause sites | AA-driven |
| Cross-species RP | Different codons | N/A (Mode 6) | N/A | N/A | Low RSCU correlation (~0.13) |

## Validation

All gene lists validated against CodonScope gene databases (2026-02-08):

| File | Mapped | Total | Rate |
|------|--------|-------|------|
| yeast_gcn4_targets.txt | 54 | 54 | 100% |
| yeast_ribosomal_proteins.txt | 132 | 132 | 100% |
| yeast_trm4_targets.txt | 39 | 39 | 100% |
| yeast_trm9_targets.txt | 50 | 50 | 100% |
| human_ribosomal_proteins.txt | 80 | 80 | 100% |
| human_5top_mrnas.txt | 115 | 115 | 100% |
| human_proliferation_myc.txt | 198 | 200 | 99% |
| human_differentiation_emt.txt | 199 | 200 | 99.5% |
| human_secreted_proteins.txt | 53 | 53 | 100% |
| human_isr_upregulated.txt | 18 | 18 | 100% |
| human_membrane_proteins.txt | 50 | 50 | 100% |
| cross_species_rp_yeast.txt | 75 | 75 | 100% |

**Note**: MYC, EIF4G2, and VEGFA are absent from the current MANE Select v1.5 CDS
dataset (likely filtered during Ensembl CDS validation). These are retained in the
MSigDB gene lists for completeness.

## How to Use

```bash
# Single mode analysis
python3 -m codonscope.cli composition --species yeast --genes pilot_gene_lists/yeast_gcn4_targets.txt --kmer 2

# Full report
python3 -m codonscope.cli report --species human --genes pilot_gene_lists/human_ribosomal_proteins.txt --output human_rp_report.html

# Cross-species comparison (Mode 6)
python3 -m codonscope.cli compare --species1 yeast --species2 human --genes pilot_gene_lists/cross_species_rp_yeast.txt

# MYC vs EMT contrast (run both, compare reports)
python3 -m codonscope.cli report --species human --genes pilot_gene_lists/human_proliferation_myc.txt --output myc_report.html
python3 -m codonscope.cli report --species human --genes pilot_gene_lists/human_differentiation_emt.txt --output emt_report.html
```
