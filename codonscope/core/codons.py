"""Codon k-mer counting engine for mono-, di-, and tricodon analysis."""

from itertools import product

BASES = ["A", "C", "G", "T"]
STOP_CODONS = {"TAA", "TAG", "TGA"}
SENSE_CODONS = sorted(
    c for c in ("".join(b) for b in product(BASES, repeat=3))
    if c not in STOP_CODONS
)
_SENSE_SET = set(SENSE_CODONS)


def sequence_to_codons(sequence: str) -> list[str]:
    """Split a CDS sequence into a list of codons.

    Args:
        sequence: CDS nucleotide sequence (must be divisible by 3).

    Returns:
        List of 3-character codon strings.

    Raises:
        ValueError: If sequence length is not divisible by 3.
    """
    if len(sequence) % 3 != 0:
        raise ValueError(
            f"Sequence length {len(sequence)} is not divisible by 3"
        )
    return [sequence[i : i + 3] for i in range(0, len(sequence), 3)]


def count_kmers(sequence: str, k: int = 1) -> dict[str, int]:
    """Count codon k-mers in a CDS sequence.

    Args:
        sequence: CDS nucleotide sequence (must be divisible by 3, no stop codon).
        k: k-mer size (1=monocodon, 2=dicodon, 3=tricodon).

    Returns:
        Dict mapping k-mer string to count.
        For k=1: {"AAA": 5, "AAC": 3, ...}
        For k=2: {"AAAAAC": 2, ...} (concatenated codons)

    K-mers are counted using a sliding window over codons.
    For a CDS with N codons, there are N-k+1 k-mers.
    """
    codons = sequence_to_codons(sequence)
    counts: dict[str, int] = {}
    for i in range(len(codons) - k + 1):
        kmer = "".join(codons[i : i + k])
        counts[kmer] = counts.get(kmer, 0) + 1
    return counts


def kmer_frequencies(sequence: str, k: int = 1) -> dict[str, float]:
    """Like count_kmers but returns frequencies (proportions summing to 1).

    Args:
        sequence: CDS nucleotide sequence (must be divisible by 3, no stop codon).
        k: k-mer size (1=monocodon, 2=dicodon, 3=tricodon).

    Returns:
        Dict mapping k-mer string to frequency (float, sums to 1.0).
        Only k-mers with non-zero count are included.
    """
    counts = count_kmers(sequence, k)
    total = sum(counts.values())
    if total == 0:
        return {}
    return {kmer: count / total for kmer, count in counts.items()}


def all_possible_kmers(k: int = 1, sense_only: bool = True) -> list[str]:
    """Return sorted list of all possible codon k-mers.

    Args:
        k: k-mer size (1=monocodon, 2=dicodon, 3=tricodon).
        sense_only: If True (default), use only the 61 sense codons.
            If False, use all 64 codons including stops.

    Returns:
        Sorted list of k-mer strings.
        k=1, sense_only=True: 61 codons
        k=2, sense_only=True: 3721 dicodons (61^2)
        k=3, sense_only=True: 226981 tricodons (61^3)
    """
    base_codons = SENSE_CODONS if sense_only else sorted(
        "".join(b) for b in product(BASES, repeat=3)
    )
    if k == 1:
        return list(base_codons)
    return sorted("".join(combo) for combo in product(base_codons, repeat=k))
