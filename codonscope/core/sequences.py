"""Gene ID resolution and CDS sequence retrieval."""

import gzip
import logging
import re
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path.home() / ".codonscope" / "data" / "species"

# Yeast systematic name pattern: Y[A-P][LR]\d{3}[WC](-[A-Z])?
_YEAST_SYSTEMATIC_RE = re.compile(
    r"^Y[A-P][LR]\d{3}[WC](-[A-Z])?$", re.IGNORECASE
)


class SequenceDB:
    """Interface to pre-downloaded CDS sequences for a species."""

    def __init__(self, species: str, data_dir: str | Path | None = None):
        """Load gene ID map and CDS sequences for a species.

        Args:
            species: Species name (e.g. "yeast").
            data_dir: Override default data directory.
                Defaults to ~/.codonscope/data/species/{species}/
        """
        self.species = species.lower()
        base = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
        self._species_dir = base / self.species

        if not self._species_dir.exists():
            raise FileNotFoundError(
                f"No data directory found for {species!r} at {self._species_dir}. "
                f"Run: codonscope download --species {species}"
            )

        # Load gene ID map
        map_path = self._species_dir / "gene_id_map.tsv"
        if not map_path.exists():
            raise FileNotFoundError(f"Gene ID map not found: {map_path}")
        self._gene_map = pd.read_csv(map_path, sep="\t")

        # Build lookup dictionaries
        self._sys_to_common: dict[str, str] = {}
        self._common_to_sys: dict[str, str] = {}
        self._sys_names: set[str] = set()

        for _, row in self._gene_map.iterrows():
            sysname = row["systematic_name"]
            common = row.get("common_name", "")
            self._sys_names.add(sysname.upper())
            self._sys_to_common[sysname.upper()] = common if pd.notna(common) else ""
            if pd.notna(common) and common:
                common_upper = common.upper()
                # Handle ambiguity: if same common name maps to multiple systematic names
                if common_upper in self._common_to_sys:
                    logger.debug(
                        "Ambiguous common name %s: %s and %s",
                        common, self._common_to_sys[common_upper], sysname,
                    )
                self._common_to_sys[common_upper] = sysname

        # Load CDS sequences lazily
        self._sequences: dict[str, str] | None = None

    def _load_sequences(self) -> dict[str, str]:
        """Load CDS sequences from gzipped FASTA."""
        if self._sequences is not None:
            return self._sequences

        cds_path = self._species_dir / "cds_sequences.fa.gz"
        if not cds_path.exists():
            raise FileNotFoundError(f"CDS FASTA not found: {cds_path}")

        sequences: dict[str, str] = {}
        current_name: str | None = None
        current_parts: list[str] = []

        with gzip.open(cds_path, "rt") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if current_name is not None:
                        sequences[current_name] = "".join(current_parts)
                    current_name = line[1:].split()[0]
                    current_parts = []
                else:
                    current_parts.append(line)
            if current_name is not None:
                sequences[current_name] = "".join(current_parts)

        self._sequences = sequences
        return sequences

    def resolve_ids(self, gene_ids: list[str]) -> dict:
        """Map input gene IDs to canonical systematic names.

        Auto-detect ID type (systematic name, common name, etc.)

        Returns:
            dict with keys:
                "mapping": {input_id: systematic_name} for successful mappings
                "unmapped": list of IDs that could not be resolved
                "n_mapped": int
                "n_unmapped": int
        """
        mapping: dict[str, str] = {}
        unmapped: list[str] = []

        for gid in gene_ids:
            gid_stripped = gid.strip()
            if not gid_stripped:
                continue

            resolved = self._resolve_single_id(gid_stripped)
            if resolved is not None:
                mapping[gid_stripped] = resolved
            else:
                unmapped.append(gid_stripped)
                logger.warning("Could not resolve gene ID: %s", gid_stripped)

        result = {
            "mapping": mapping,
            "unmapped": unmapped,
            "n_mapped": len(mapping),
            "n_unmapped": len(unmapped),
        }
        logger.info(
            "ID resolution: %d mapped, %d unmapped",
            result["n_mapped"], result["n_unmapped"],
        )
        return result

    def _resolve_single_id(self, gene_id: str) -> str | None:
        """Resolve a single gene ID to systematic name."""
        upper = gene_id.upper()

        # Check if it's a yeast systematic name
        if _YEAST_SYSTEMATIC_RE.match(gene_id):
            if upper in self._sys_names:
                # Return the canonical case from our data
                for _, row in self._gene_map.iterrows():
                    if row["systematic_name"].upper() == upper:
                        return row["systematic_name"]
            return None

        # Check for Ensembl ID (not supported for yeast)
        if upper.startswith("ENSG") or upper.startswith("ENSMUSG"):
            logger.warning(
                "Ensembl IDs not supported for yeast: %s", gene_id
            )
            return None

        # Try common name lookup (case-insensitive)
        if upper in self._common_to_sys:
            return self._common_to_sys[upper]

        # Try as systematic name without regex (some edge cases)
        if upper in self._sys_names:
            for _, row in self._gene_map.iterrows():
                if row["systematic_name"].upper() == upper:
                    return row["systematic_name"]

        return None

    def get_sequences(self, systematic_names: list[str]) -> dict[str, str]:
        """Return {systematic_name: cds_sequence} for requested genes.

        Sequences are already validated and stop-codon-stripped.
        """
        all_seqs = self._load_sequences()
        result: dict[str, str] = {}
        for name in systematic_names:
            if name in all_seqs:
                result[name] = all_seqs[name]
            else:
                logger.warning("No CDS sequence found for: %s", name)
        return result

    def get_all_sequences(self) -> dict[str, str]:
        """Return all genome CDS sequences (for background computation)."""
        return dict(self._load_sequences())

    def get_gene_metadata(self) -> pd.DataFrame:
        """Return DataFrame with systematic_name, common_name, cds_length, gc_content."""
        return self._gene_map[
            ["systematic_name", "common_name", "cds_length", "gc_content"]
        ].copy()

    @property
    def species_dir(self) -> Path:
        """Path to the species data directory."""
        return self._species_dir
