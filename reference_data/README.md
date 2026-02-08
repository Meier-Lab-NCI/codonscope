# Reference Data

This directory contains copies of key metadata files for auditability and reproducibility. These files are generated during the `download()` step and stored here for reference.

## Files

### yeast_gene_id_map.tsv (6,685 genes)
- **Source**: SGD `orf_coding_all.fasta.gz`
- **URL**: `https://downloads.yeastgenome.org/sequence/S288C_reference/orf_dna/orf_coding_all.fasta.gz`
- **Genome**: S288C reference (S. cerevisiae)
- **Columns**: systematic_name, common_name, cds_length, gc_content, status
- **Filtering**: Mitochondrial ORFs (Q0*) excluded. CDS validated: ACGT only, divisible by 3, starts ATG, ends stop codon, no internal stops. Stop codon stripped before length measurement.
- **Date downloaded**: 2026-02-07

### human_gene_id_map.tsv (19,229 genes)
- **Source**: NCBI MANE Select v1.5 summary + Ensembl GRCh38 CDS FASTA
- **MANE URL**: `https://ftp.ncbi.nlm.nih.gov/refseq/MANE/MANE_human/current/MANE.GRCh38.v1.5.summary.txt.gz`
- **Ensembl URL**: `https://ftp.ensembl.org/pub/current_fasta/homo_sapiens/cds/Homo_sapiens.GRCh38.cds.all.fa.gz`
- **Genome**: GRCh38
- **Columns**: systematic_name (ENSG), common_name (HGNC symbol), ensembl_transcript, refseq_transcript, entrez_id, cds_length, gc_content
- **Filtering**: Only MANE Select transcripts (not MANE Plus Clinical). One canonical transcript per gene. Same CDS validation as yeast.
- **Date downloaded**: 2026-02-07

### mouse_gene_id_map.tsv (~21,500 genes)
- **Source**: Ensembl GRCm39 CDS FASTA
- **URL**: `https://ftp.ensembl.org/pub/current_fasta/mus_musculus/cds/Mus_musculus.GRCm39.cds.all.fa.gz`
- **Genome**: GRCm39 (M. musculus)
- **Columns**: systematic_name (ENSMUSG), common_name (MGI symbol), ensembl_transcript (ENSMUST), cds_length, gc_content
- **Filtering**: For each gene (ENSMUSG), the longest valid CDS is selected. No MANE equivalent exists for mouse. Same CDS validation as yeast.
- **Date downloaded**: 2026-02-07

## Files NOT Included (too large)

The following files are generated during download but are too large to include in the repository:

| File | Species | Size | URL / How to generate |
|------|---------|------|-----------------------|
| `expression_gtex.tsv.gz` | Human | ~15 MB | GTEx v8: `https://storage.googleapis.com/adult-gtex/bulk-gex/v8/rna-seq/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_median_tpm.gct.gz` |
| `expression_ccle.tsv.gz` | Human | ~50 MB | DepMap API: `https://depmap.org/portal/api/download/files` |
| `background_di.npz` | Human | ~280 MB | Generated during `download('human')` |
| `background_di.npz` | Mouse | ~14 MB | Generated during `download('mouse')` |
| `background_mono.npz` | All | 1-2 MB | Generated during `download()` |
| `background_tri.npz` | All | 2-3 MB | Generated during `download()` |
| `cds_sequences.fa.gz` | All | 2-10 MB | Generated during `download()` |

To regenerate all data files:
```bash
python3 -c "from codonscope.data.download import download; download('yeast')"
python3 -c "from codonscope.data.download import download; download('human')"
python3 -c "from codonscope.data.download import download; download('mouse')"
```
