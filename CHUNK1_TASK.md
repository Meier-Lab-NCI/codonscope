# Chunk 1: Data Layer + Yeast Species

## Goal

Build the data download infrastructure and get yeast (S. cerevisiae) working end-to-end as the first species. Yeast is the easiest starting point: no isoform complexity (one ORF per gene), clean data from SGD, small genome, and our best positive controls come from yeast.

By the end of this chunk we should be able to:
1. Run `codonscope download --species yeast` and get all needed reference data
2. Load a gene list, map IDs to CDS sequences, and return validated sequences
3. Have pre-computed genome-wide background frequencies ready for later analysis

## Files to create

### `codonscope/__init__.py`
Package init. Set version = "0.1.0".

### `codonscope/data/download.py`

Master download script. For now, only implement yeast. Design it so adding human/mouse later is straightforward (species-specific download functions dispatched from a main `download(species)` function).

**Data directory**: `~/.codonscope/data/species/yeast/`

**What to download for yeast**:

1. **CDS sequences from SGD**
   - URL: `https://downloads.yeastgenome.org/sequence/S288C_reference/orf_dna/orf_coding_all.fasta.gz`
   - This contains all verified and uncharacterized ORFs with their CDS sequences
   - Parse FASTA, extract: systematic name (e.g. YAL001C), common name (e.g. TFC3), CDS sequence
   - Validate each CDS: divisible by 3, starts with ATG, ends with stop codon (TAA/TAG/TGA)
   - Strip stop codon from sequence after validation
   - Exclude any ORF with internal stop codons or non-ACGT characters
   - Save as: `cds_sequences.fa.gz` (cleaned) and `gene_id_map.tsv`

2. **Gene ID mapping table**
   - Built during FASTA parsing above
   - Columns: `systematic_name`, `common_name`, `cds_length`, `gc_content`
   - The SGD FASTA headers contain both systematic and common names
   - Format: `>YAL001C TFC3 SGDID:S000000001, ...`

3. **tRNA gene copy numbers**
   - Source: GtRNAdb S. cerevisiae
   - URL: `http://gtrnadb.ucsc.edu/genomes/eukaryota/Scere3/sacCer3-mature-tRNAs.fa`
   - Parse the tRNA FASTA to count genes per anticodon
   - Save as: `trna_copy_numbers.tsv` with columns: `anticodon`, `amino_acid`, `gene_count`
   - If GtRNAdb URL is unreliable, fall back to hardcoded table from published literature (Chan et al. 2010, Phizicky & Hopper 2010). There are only 42 tRNA species in yeast, this is small enough to hardcode as a fallback.

4. **Wobble decoding rules**
   - This is a curated table, not downloaded. Create it from published knowledge.
   - For each of the 61 sense codons: which tRNA anticodon decodes it, and is the interaction Watson-Crick or wobble?
   - Yeast uses standard eukaryotic wobble rules:
     - U34 in anticodon can pair with A or G in codon (wobble)
     - G34 in anticodon pairs with C or U in codon (wobble for U)
     - I34 (inosine, deaminated A) in anticodon pairs with U, C, or A in codon
     - Modified U34 (mcm5U, mcm5s2U, ncm5U) pairs with A (and sometimes G)
   - Columns: `codon`, `amino_acid`, `decoding_anticodon`, `trna_gene_copies`, `decoding_type` (watson_crick/wobble), `modification_notes`
   - Key entries for positive controls:
     - AGA (Arg): decoded by tRNA-Arg(UCU), watson_crick, Trm9-dependent (mcm5U/mcm5s2U)
     - GAA (Glu): decoded by tRNA-Glu(UUC), watson_crick, Trm9-dependent (mcm5s2U)
     - TTG (Leu): decoded by tRNA-Leu(CAA), watson_crick, Trm4-dependent (m5C)
   - Save as: `wobble_rules.tsv`

5. **Pre-computed backgrounds**
   - After downloading CDS sequences, compute:
     - Genome-wide monocodon frequencies: per-gene frequencies for all ~6,000 ORFs, then mean + std
     - Genome-wide dicodon frequencies: same approach
     - Tricodon: same but warn this takes longer and produces a large file
   - Also store: per-gene CDS length array, per-gene GC content array (for matched background sampling)
   - Save as numpy `.npz` files: `background_mono.npz`, `background_di.npz`, `background_tri.npz`
   - Each npz contains: `mean` (frequency vector), `std` (per-gene std), `per_gene` (N_genes × N_kmers matrix for bootstrap resampling)
   - **Note on tricodon**: 262,144 possible tricodons × ~6,000 genes = ~1.5 billion entries if stored as full matrix. Instead, store as sparse matrix or just store the mean/std vectors and compute bootstrap on-the-fly. Use scipy.sparse if needed.

### `codonscope/core/sequences.py`

Gene ID resolution and CDS sequence retrieval.

```python
class SequenceDB:
    """Interface to pre-downloaded CDS sequences for a species."""
    
    def __init__(self, species: str, data_dir: str = None):
        """Load gene ID map and CDS sequences for a species.
        
        data_dir defaults to ~/.codonscope/data/species/{species}/
        """
    
    def resolve_ids(self, gene_ids: list[str]) -> dict:
        """Map input gene IDs to canonical systematic names.
        
        Auto-detect ID type (systematic name, common name, etc.)
        Returns: {input_id: systematic_name} for successful mappings
        Logs warnings for unmapped IDs.
        Also returns summary: n_mapped, n_unmapped, n_ambiguous
        """
    
    def get_sequences(self, systematic_names: list[str]) -> dict[str, str]:
        """Return {systematic_name: cds_sequence} for requested genes.
        
        Sequences are already validated and stop-codon-stripped.
        """
    
    def get_all_sequences(self) -> dict[str, str]:
        """Return all genome CDS sequences (for background computation)."""
    
    def get_gene_metadata(self) -> pd.DataFrame:
        """Return DataFrame with systematic_name, common_name, cds_length, gc_content."""
```

**ID resolution logic for yeast**:
- If ID matches `Y[A-P][LR]\d{3}[WC](-[A-Z])?` → systematic name
- If ID matches an Ensembl pattern `ENSG...` → not supported for yeast, warn
- Otherwise → try common name lookup
- Case-insensitive matching for common names

### `codonscope/core/codons.py` (stub only in this chunk)

Just create the file with the function signatures and docstrings. We'll implement the counting logic in chunk 2. But we need the interface defined so `download.py` can call it for background pre-computation.

```python
def count_kmers(sequence: str, k: int = 1) -> dict[str, int]:
    """Count codon k-mers in a CDS sequence.
    
    Args:
        sequence: CDS nucleotide sequence (must be divisible by 3, no stop codon)
        k: k-mer size (1=monocodon, 2=dicodon, 3=tricodon)
    
    Returns:
        Dict mapping k-mer string to count.
        For k=1: {"AAA": 5, "AAC": 3, ...}
        For k=2: {"AAAAAC": 2, ...} (concatenated codons)
    
    K-mers are counted using a sliding window over codons.
    For a CDS with N codons, there are N-k+1 k-mers.
    """

def sequence_to_codons(sequence: str) -> list[str]:
    """Split a CDS sequence into a list of codons."""

def kmer_frequencies(sequence: str, k: int = 1) -> dict[str, float]:
    """Like count_kmers but returns frequencies (proportions summing to 1)."""

def all_possible_kmers(k: int = 1) -> list[str]:
    """Return sorted list of all possible codon k-mers.
    
    k=1: 64 codons (including stops, which won't appear in CDSs)
    k=2: 4096 dicodons
    k=3: 262144 tricodons
    
    For practical purposes, only the 61 sense codons matter for k=1,
    giving 61^k possible k-mers.
    """
```

**Implement these fully** — they're simple and the background pre-computation needs them.

## Testing for this chunk

Create `tests/test_chunk1.py`:

```python
def test_download_yeast():
    """Run download, verify all expected files exist."""

def test_id_mapping_systematic():
    """YAL001C should resolve to YAL001C."""

def test_id_mapping_common():
    """TFC3 should resolve to YAL001C."""

def test_id_mapping_case_insensitive():
    """tfc3 should resolve to YAL001C."""

def test_cds_validation():
    """All loaded CDSs should be divisible by 3 and contain only ACGT."""

def test_gene_count():
    """Yeast should have ~6,000-6,700 verified ORFs."""

def test_kmer_counting_basic():
    """Hand-computed example: ATGAAAGAA has codons ATG,AAA,GAA.
    Monocodons: ATG=1, AAA=1, GAA=1
    Dicodons: ATGAAA=1, AAAGAA=1
    """

def test_kmer_frequencies_sum_to_one():
    """Frequencies from any sequence should sum to 1.0."""

def test_background_files_exist():
    """After download, background npz files should exist."""

def test_background_mono_shape():
    """Mono background mean vector should have 61 entries (sense codons)."""

def test_yef3_exists():
    """YEF3 / YLR249W should be in the database."""

def test_gcn4_target_resolution():
    """A sample of known Gcn4 target genes should resolve successfully."""
```

## Important edge cases

1. **SGD FASTA header parsing**: Headers look like `>YAL001C TFC3 SGDID:S000000001, Chr I from 151168-151099,151008-150947, Genome Release 64-4-1, reverse complement, Verified`. Parse systematic name (first field), common name (second field). Some ORFs have no common name — handle gracefully.

2. **Dubious ORFs**: SGD marks some ORFs as "Dubious". Consider filtering to "Verified" and "Uncharacterized" only for the background, but allow users to query any ORF. The FASTA header contains the verification status.

3. **Mitochondrial ORFs**: SGD includes mitochondrial genes (e.g. Q0010). These use a different genetic code (UGA = Trp instead of stop). **Exclude mitochondrial ORFs from background computation and flag them if they appear in user gene lists.** Mitochondrial systematic names start with Q0.

4. **tRNA gene copy number fallback**: If GtRNAdb download fails, use this hardcoded yeast tRNA table (from Phizicky & Hopper 2010, Chan et al. 2010):

```python
# Yeast tRNA gene copy numbers (S288C reference)
# Format: {anticodon: (amino_acid, gene_count)}
YEAST_TRNA_FALLBACK = {
    "AAA": ("Phe", 1),   # not present in yeast, placeholder
    "AAC": ("Val", 2),
    "AAG": ("Leu", 1),
    "AAU": ("Ile", 2),
    "ACA": ("Cys", 4),
    "ACC": ("Gly", 16),
    "ACG": ("Arg", 6),
    "ACU": ("Ser", 4),
    "AGC": ("Ala", 11),
    "AGG": ("Pro", 10),
    "AGU": ("Thr", 11),
    "AUC": ("Asp", 15),
    "AUG": ("His", 7),
    "AUU": ("Asn", 10),
    "CAA": ("Leu", 10),
    "CAC": ("Val", 2),
    "CAG": ("Leu", 3),
    "CAU": ("Ile", 13),
    "CCA": ("Trp", 6),
    "CCC": ("Gly", 2),
    "CCG": ("Arg", 1),
    "CCU": ("Arg", 1),
    "CGC": ("Ala", 1),
    "CGG": ("Pro", 2),
    "CGU": ("Thr", 4),
    "CUC": ("Glu", 14),
    "CUG": ("Gln", 9),
    "CUU": ("Lys", 14),
    "GAA": ("Phe", 10),
    "GAC": ("Val", 14),
    "GAG": ("Leu", 7),
    "GAU": ("Ile", 2),
    "GCA": ("Cys", 0),
    "GCC": ("Gly", 3),
    "GCG": ("Arg", 6),
    "GCU": ("Ser", 11),
    "GGC": ("Ala", 5),
    "GGG": ("Pro", 0),
    "GGU": ("Thr", 9),
    "GUC": ("Asp", 4),
    "GUG": ("His", 1),
    "GUU": ("Asn", 2),
    "IAU": ("Ile", 1),   # inosine-modified
    "UCU": ("Arg", 11),
    "UGC": ("Ala", 0),
    "UGG": ("Pro", 10),
    "UGU": ("Thr", 0),
    "UUC": ("Glu", 2),
    "UUG": ("Gln", 1),
    "UUU": ("Lys", 7),
}
# NOTE: These numbers should be verified against GtRNAdb.
# Some anticodons with 0 copies use wobble from related tRNAs.
# The download script should prefer live GtRNAdb data; this is fallback only.
```

5. **Background pre-computation for tricodons**: The full per-gene × tricodon matrix is too large to store naively (~6,000 genes × 262,144 tricodons). Options:
   - Only store mean and std vectors (not per-gene matrix). Bootstrap will need to recompute on-the-fly by sampling gene sequences and counting.
   - Or store only non-zero entries as a sparse matrix.
   - Recommended: store mean/std only for tricodons. For mono and dicodons, store the full per-gene matrix (6,000 × 61 = tiny; 6,000 × 4,096 = ~25M entries, ~200MB as float32, manageable).

## Verification

When this chunk is done, you should be able to run:

```python
from codonscope.data.download import download
from codonscope.core.sequences import SequenceDB

# Download yeast data
download("yeast")

# Load and query
db = SequenceDB("yeast")

# Resolve some gene IDs
mapping = db.resolve_ids(["YEF3", "YLR249W", "GCN4", "RPL22A", "TFC3"])
print(mapping)
# Should show all mapped successfully

# Get sequences
seqs = db.get_sequences(["YLR249W"])
print(f"YEF3 CDS length: {len(seqs['YLR249W'])} nt, {len(seqs['YLR249W'])//3} codons")

# Check backgrounds exist
import numpy as np
bg = np.load("~/.codonscope/data/species/yeast/background_mono.npz")
print(f"Mono background shape: {bg['mean'].shape}")  # Should be (61,)
```
