"""Ortholog mapping between species.

Loads pre-computed one-to-one ortholog tables and provides
bidirectional gene mapping.
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path.home() / ".codonscope" / "data"
ORTHOLOGS_DIR = DEFAULT_DATA_DIR / "orthologs"


class OrthologDB:
    """Interface to pre-downloaded ortholog mappings between two species.

    Provides bidirectional mapping between species1 and species2.
    """

    def __init__(
        self,
        species1: str,
        species2: str,
        data_dir: str | Path | None = None,
    ):
        """Load ortholog mapping for a species pair.

        The ortholog file is stored as:
          {data_dir}/orthologs/{species1}_{species2}.tsv
        or the reverse order.

        Args:
            species1: First species (e.g. "human").
            species2: Second species (e.g. "yeast").
            data_dir: Override default data directory.
        """
        self.species1 = species1.lower()
        self.species2 = species2.lower()
        base = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
        ortho_dir = base / "orthologs"

        # Try both orderings
        path1 = ortho_dir / f"{self.species1}_{self.species2}.tsv"
        path2 = ortho_dir / f"{self.species2}_{self.species1}.tsv"

        if path1.exists():
            self._path = path1
            self._file_order = (self.species1, self.species2)
        elif path2.exists():
            self._path = path2
            self._file_order = (self.species2, self.species1)
        else:
            raise FileNotFoundError(
                f"No ortholog data found for {species1}-{species2}. "
                f"Tried: {path1}, {path2}. "
                f"Run: codonscope download --species {species1} {species2}"
            )

        self._load()

    def _load(self) -> None:
        """Load ortholog TSV and build bidirectional mappings."""
        df = pd.read_csv(self._path, sep="\t")
        self._df = df

        s1, s2 = self._file_order
        col1 = f"{s1}_gene"
        col2 = f"{s2}_gene"

        # Build bidirectional dicts using systematic gene IDs
        self._forward: dict[str, str] = {}   # species1 → species2
        self._reverse: dict[str, str] = {}   # species2 → species1

        for _, row in df.iterrows():
            g1 = row[col1]
            g2 = row[col2]

            if s1 == self.species1:
                self._forward[g1] = g2
                self._reverse[g2] = g1
            else:
                self._forward[g2] = g1
                self._reverse[g1] = g2

    def map_genes(
        self,
        gene_ids: list[str],
        from_species: str,
    ) -> dict[str, str]:
        """Map gene IDs from one species to orthologs in the other.

        Args:
            gene_ids: Gene IDs (systematic names) in from_species.
            from_species: Which species the input IDs belong to.

        Returns:
            {input_id: ortholog_id} for genes with orthologs.
        """
        from_species = from_species.lower()
        if from_species == self.species1:
            mapping = self._forward
        elif from_species == self.species2:
            mapping = self._reverse
        else:
            raise ValueError(
                f"from_species must be '{self.species1}' or '{self.species2}', "
                f"got '{from_species}'"
            )

        result = {}
        for gid in gene_ids:
            if gid in mapping:
                result[gid] = mapping[gid]
        return result

    def get_all_pairs(self) -> list[tuple[str, str]]:
        """Return all ortholog pairs as (species1_id, species2_id)."""
        return list(self._forward.items())

    @property
    def n_pairs(self) -> int:
        """Number of one-to-one ortholog pairs."""
        return len(self._forward)

    def __repr__(self) -> str:
        return (
            f"OrthologDB({self.species1}-{self.species2}, "
            f"{self.n_pairs} pairs)"
        )
