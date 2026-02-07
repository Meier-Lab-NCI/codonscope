"""Codon optimality scoring: tAI and wobble-aware tAI (wtAI).

The tRNA Adaptation Index (tAI) measures how well a codon's decoding tRNA
is supplied.  The wobble-aware tAI (wtAI) adds a penalty for wobble-decoded
codons based on the finding that wobble decoding is slower regardless of
tRNA abundance (Frydman lab).

Both scores are normalised to [0, 1] and a small pseudocount replaces
zero-copy anticodons so that geometric means remain defined.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from codonscope.core.codons import SENSE_CODONS, sequence_to_codons

logger = logging.getLogger(__name__)

_PSEUDOCOUNT_FRACTION = 0.5  # pseudocount = 0.5 / max_copies


class OptimalityScorer:
    """Per-codon optimality weights derived from tRNA gene copy numbers."""

    def __init__(
        self,
        species_dir: str | Path,
        wobble_penalty: float = 0.5,
    ):
        """Load wobble rules and build per-codon weight vectors.

        Args:
            species_dir: Path to species data directory containing
                ``wobble_rules.tsv``.
            wobble_penalty: Multiplicative penalty applied to wobble-decoded
                codons for wtAI (default 0.5).
        """
        self.species_dir = Path(species_dir)
        self.wobble_penalty = wobble_penalty

        wobble_path = self.species_dir / "wobble_rules.tsv"
        if not wobble_path.exists():
            raise FileNotFoundError(
                f"wobble_rules.tsv not found in {self.species_dir}"
            )

        df = pd.read_csv(wobble_path, sep="\t")

        # Build raw weight dicts keyed by codon
        raw_tai: dict[str, float] = {}
        raw_wtai: dict[str, float] = {}

        for _, row in df.iterrows():
            codon = row["codon"]
            copies = float(row["trna_gene_copies"])
            decoding = row["decoding_type"]

            raw_tai[codon] = copies
            if decoding == "wobble":
                raw_wtai[codon] = copies * wobble_penalty
            else:
                raw_wtai[codon] = copies

        # Normalise: divide by max, replace zeros with pseudocount
        self._tai = _normalise_weights(raw_tai)
        self._wtai = _normalise_weights(raw_wtai)

        # Ordered arrays aligned with SENSE_CODONS for vectorised ops
        self._tai_vec = np.array(
            [self._tai.get(c, 0.0) for c in SENSE_CODONS], dtype=np.float64
        )
        self._wtai_vec = np.array(
            [self._wtai.get(c, 0.0) for c in SENSE_CODONS], dtype=np.float64
        )
        self._codon_to_idx = {c: i for i, c in enumerate(SENSE_CODONS)}

    # ── Public properties ─────────────────────────────────────────────────

    @property
    def tai_weights(self) -> dict[str, float]:
        """Per-codon classical tAI weights (normalised 0–1)."""
        return dict(self._tai)

    @property
    def wtai_weights(self) -> dict[str, float]:
        """Per-codon wobble-aware tAI weights (normalised 0–1)."""
        return dict(self._wtai)

    # ── Per-gene scores ───────────────────────────────────────────────────

    def gene_tai(self, sequence: str) -> float:
        """Geometric-mean tAI for a CDS."""
        return self._gene_score(sequence, self._tai)

    def gene_wtai(self, sequence: str) -> float:
        """Geometric-mean wtAI for a CDS."""
        return self._gene_score(sequence, self._wtai)

    # ── Per-position scores ───────────────────────────────────────────────

    def per_position_scores(
        self,
        sequence: str,
        method: str = "wtai",
    ) -> np.ndarray:
        """Return a 1-D array of per-codon optimality scores for a CDS.

        Args:
            sequence: CDS nucleotide sequence (divisible by 3, no stop).
            method: ``"tai"`` or ``"wtai"`` (default).

        Returns:
            1-D float64 array of length ``len(sequence) // 3``.
        """
        weights = self._wtai if method == "wtai" else self._tai
        codons = sequence_to_codons(sequence)
        return np.array(
            [weights.get(c, 0.0) for c in codons], dtype=np.float64
        )

    def smooth_profile(
        self,
        scores: np.ndarray,
        window: int = 10,
    ) -> np.ndarray:
        """Sliding-window average of per-position scores.

        Uses a centred window.  At the edges the window is truncated
        (i.e., the output has the same length as the input).
        """
        if window <= 1 or len(scores) <= 1:
            return scores.copy()
        kernel = np.ones(window, dtype=np.float64) / window
        # 'same' gives output of same length; edge effects handled
        return np.convolve(scores, kernel, mode="same")

    # ── Fast / slow classification ────────────────────────────────────────

    def classify_codons(
        self,
        threshold: float | None = None,
        method: str = "wtai",
    ) -> tuple[set[str], set[str]]:
        """Classify sense codons as *fast* or *slow*.

        Args:
            threshold: Score cutoff.  Codons with score >= threshold are
                "fast".  If *None*, the median genome wtAI/tAI is used.
            method: ``"tai"`` or ``"wtai"`` (default).

        Returns:
            ``(fast_codons, slow_codons)`` — each a set of codon strings.
        """
        weights = self._wtai if method == "wtai" else self._tai
        if threshold is None:
            vals = np.array(list(weights.values()))
            threshold = float(np.median(vals))
        fast = {c for c, w in weights.items() if w >= threshold}
        slow = {c for c, w in weights.items() if w < threshold}
        return fast, slow

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _gene_score(sequence: str, weights: dict[str, float]) -> float:
        """Geometric mean of per-codon weights (log-space)."""
        codons = sequence_to_codons(sequence)
        if not codons:
            return 0.0
        log_sum = 0.0
        n = 0
        for c in codons:
            w = weights.get(c)
            if w is not None and w > 0:
                log_sum += np.log(w)
                n += 1
        if n == 0:
            return 0.0
        return float(np.exp(log_sum / n))


def _normalise_weights(raw: dict[str, float]) -> dict[str, float]:
    """Normalise weights to [0, 1] with pseudocount for zeros."""
    max_w = max(raw.values()) if raw else 1.0
    if max_w <= 0:
        max_w = 1.0

    pseudo = _PSEUDOCOUNT_FRACTION / max_w  # small value in normalised space

    normed: dict[str, float] = {}
    for codon, w in raw.items():
        nw = w / max_w
        if nw <= 0:
            nw = pseudo
        normed[codon] = nw
    return normed
