# Example Gene Lists — Data Provenance

## Gene Lists

### yeast_rp_genes.txt (132 genes)
- **Content**: S. cerevisiae cytoplasmic ribosomal protein genes (RPL*/RPS*)
- **Source**: SGD `orf_coding_all.fasta.gz`, filtered from `gene_id_map.tsv` by common name prefix `RPL` or `RPS`
- **Method**: `gene_id_map.tsv['common_name'].str.match(r'^RP[LS]\d')`
- **Date generated**: 2026-02-07
- **Notes**: Includes both A and B paralogs (e.g., RPL3, RPL4A, RPL4B). Does not include mitochondrial ribosomal proteins.

### human_rp_genes.txt (94 genes)
- **Content**: Human cytoplasmic ribosomal protein genes (RPL*/RPS*)
- **Source**: MANE Select v1.5 gene_id_map.tsv, filtered by HGNC symbol prefix `RPL` or `RPS`
- **Method**: `gene_id_map.tsv['common_name'].str.match(r'^RP[LS]\d')`
- **Date generated**: 2026-02-07
- **Notes**: Single genes per ribosomal protein (no paralogs in human). Includes pseudogene-derived functional copies (e.g., RPSAP58 excluded — only protein-coding).

### mouse_rp_genes.txt (99 genes)
- **Content**: M. musculus cytoplasmic ribosomal protein genes (Rpl*/Rps*)
- **Source**: Ensembl GRCm39 gene_id_map.tsv, filtered by gene symbol prefix `Rpl` or `Rps`
- **Method**: `gene_id_map.tsv['common_name'].str.match(r'^Rp[ls]\d')`
- **Date generated**: 2026-02-07
- **Notes**: Mouse gene names use Title Case convention (Rpl3, not RPL3). Includes Rplp0/Rplp1/Rplp2 (P-stalk proteins).

### yeast_trm_genes.txt (16 genes)
- **Content**: S. cerevisiae tRNA modification enzymes
- **Source**: Phizicky & Hopper 2010 (Genes Dev 24:1832-60), Table 1
- **Method**: Hand-curated list of TRM family enzymes and tRNA adenosine deaminases (TAD1, TAD2)
- **Date generated**: 2026-02-07
- **Notes**: These enzymes modify tRNA nucleosides at specific positions. Several (TRM4, TRM9) directly affect wobble decoding and codon usage patterns.

## How to Use

```bash
# Yeast ribosomal protein analysis
python3 -m codonscope.cli report --species yeast --genes examples/yeast_rp_genes.txt --output yeast_rp_report.html

# Human ribosomal protein analysis
python3 -m codonscope.cli report --species human --genes examples/human_rp_genes.txt --output human_rp_report.html

# Mouse ribosomal protein analysis
python3 -m codonscope.cli report --species mouse --genes examples/mouse_rp_genes.txt --output mouse_rp_report.html

# Yeast tRNA modification enzyme analysis
python3 -m codonscope.cli report --species yeast --genes examples/yeast_trm_genes.txt --output yeast_trm_report.html
```

## Regeneration

Gene lists can be regenerated from downloaded data:

```python
import pandas as pd
mouse_map = pd.read_csv('~/.codonscope/data/species/mouse/gene_id_map.tsv', sep='\t')
rp_mouse = mouse_map[mouse_map['common_name'].str.match(r'^Rp[ls]\d', na=False)]
rp_mouse['common_name'].to_csv('examples/mouse_rp_genes.txt', index=False, header=False)
```
