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

# Human/Ensembl ID patterns
_ENSG_RE = re.compile(r"^ENSG\d{11}(\.\d+)?$", re.IGNORECASE)
_ENST_RE = re.compile(r"^ENST\d{11}(\.\d+)?$", re.IGNORECASE)

# Mouse/Ensembl ID patterns
_ENSMUSG_RE = re.compile(r"^ENSMUSG\d{11}(\.\d+)?$", re.IGNORECASE)
_ENSMUST_RE = re.compile(r"^ENSMUST\d{11}(\.\d+)?$", re.IGNORECASE)


class IDMapping:
    """Result of gene ID resolution.  Behaves like a dict of {input_id: systematic_name}.

    Usage::

        result = db.resolve_ids(["ARG1", "HIS3", "YLR249W"])
        len(result)                 # 3  (number of mapped genes)
        list(result.values())       # ["YOL058W", "YOR202W", "YLR249W"]
        result["ARG1"]              # "YOL058W"
        db.get_sequences(result)    # works — get_sequences accepts IDMapping

        # Unmapped info
        result.unmapped             # list of IDs that failed to resolve
        result.n_unmapped           # count
    """

    def __init__(
        self,
        mapping: dict[str, str],
        unmapped: list[str],
    ):
        self._mapping = mapping
        self.unmapped = unmapped
        self.n_mapped = len(mapping)
        self.n_unmapped = len(unmapped)

    # ── dict-like interface (delegates to the input→systematic mapping) ──

    def __len__(self) -> int:
        return len(self._mapping)

    def __getitem__(self, key: str) -> str:
        return self._mapping[key]

    def __contains__(self, key: object) -> bool:
        return key in self._mapping

    def __iter__(self):
        return iter(self._mapping)

    def __bool__(self) -> bool:
        return len(self._mapping) > 0

    def keys(self):
        return self._mapping.keys()

    def values(self):
        return self._mapping.values()

    def items(self):
        return self._mapping.items()

    def get(self, key: str, default=None):
        return self._mapping.get(key, default)

    def __repr__(self) -> str:
        return (
            f"IDMapping({self.n_mapped} mapped, "
            f"{self.n_unmapped} unmapped)"
        )

    # Keep backwards compat: result["mapping"] still works
    def _compat_getitem(self, key: str):
        compat_keys = {
            "mapping": self._mapping,
            "unmapped": self.unmapped,
            "n_mapped": self.n_mapped,
            "n_unmapped": self.n_unmapped,
        }
        if key in compat_keys:
            return compat_keys[key]
        # Fall through to normal mapping lookup
        return self._mapping[key]

    # Override __getitem__ to handle both old and new API
    def __getitem__(self, key: str):
        if key in ("mapping", "unmapped", "n_mapped", "n_unmapped"):
            return {
                "mapping": self._mapping,
                "unmapped": self.unmapped,
                "n_mapped": self.n_mapped,
                "n_unmapped": self.n_unmapped,
            }[key]
        return self._mapping[key]


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
        # Also keep canonical-case mapping for fast lookup
        self._sys_upper_to_canonical: dict[str, str] = {}
        # Human-specific lookups: ENST→ENSG, Entrez→ENSG
        self._enst_to_sys: dict[str, str] = {}
        self._entrez_to_sys: dict[str, str] = {}

        for _, row in self._gene_map.iterrows():
            sysname = row["systematic_name"]
            common = row.get("common_name", "")
            upper = sysname.upper()
            self._sys_names.add(upper)
            self._sys_upper_to_canonical[upper] = sysname
            self._sys_to_common[upper] = common if pd.notna(common) else ""
            if pd.notna(common) and common:
                common_upper = common.upper()
                if common_upper in self._common_to_sys:
                    logger.debug(
                        "Ambiguous common name %s: %s and %s",
                        common, self._common_to_sys[common_upper], sysname,
                    )
                self._common_to_sys[common_upper] = sysname

            # Human-specific: Ensembl transcript → systematic name (ENSG)
            enst = row.get("ensembl_transcript", "")
            if pd.notna(enst) and enst:
                # Store both with and without version for flexible matching
                enst_upper = str(enst).upper()
                self._enst_to_sys[enst_upper] = sysname
                enst_base = enst_upper.split(".")[0]
                self._enst_to_sys[enst_base] = sysname

            # Human-specific: Entrez ID → systematic name (ENSG)
            entrez = row.get("entrez_id", "")
            if pd.notna(entrez) and str(entrez).strip():
                self._entrez_to_sys[str(entrez).strip()] = sysname

        # Human-specific: HGNC alias/previous symbol lookup
        self._alias_to_info: dict[str, tuple[str, str, str]] = {}
        alias_path = self._species_dir / "hgnc_aliases.tsv"
        if self.species == "human" and alias_path.exists():
            alias_df = pd.read_csv(alias_path, sep="\t")
            for _, row in alias_df.iterrows():
                alias_upper = str(row["alias"]).upper()
                # Don't overwrite existing common_name lookups
                if alias_upper not in self._common_to_sys:
                    self._alias_to_info[alias_upper] = (
                        str(row["ensembl_gene_id"]),
                        str(row["canonical_symbol"]),
                        str(row["alias_type"]),
                    )
            if self._alias_to_info:
                logger.info(
                    "Loaded %d HGNC aliases for human gene resolution",
                    len(self._alias_to_info),
                )

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

    def resolve_ids(self, gene_ids: list[str]) -> IDMapping:
        """Map input gene IDs to canonical systematic names.

        Auto-detect ID type (systematic name, common name, etc.)
        Case-insensitive for both systematic and common names.

        Returns:
            IDMapping that behaves like a dict {input_id: systematic_name}.
            Also has .unmapped, .n_mapped, .n_unmapped attributes.

        Example::

            result = db.resolve_ids(["ARG1", "YLR249W"])
            len(result)             # 2
            list(result.values())   # ["YOL058W", "YLR249W"]
            result.unmapped         # []
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

        result = IDMapping(mapping, unmapped)
        logger.info(
            "ID resolution: %d mapped, %d unmapped",
            result.n_mapped, result.n_unmapped,
        )
        return result

    def _resolve_single_id(self, gene_id: str) -> str | None:
        """Resolve a single gene ID to systematic name."""
        upper = gene_id.upper()

        # ── Yeast-specific patterns ──────────────────────────────────────
        if self.species == "yeast":
            if _YEAST_SYSTEMATIC_RE.match(gene_id):
                canonical = self._sys_upper_to_canonical.get(upper)
                if canonical is not None:
                    return canonical
                return None

            # Ensembl IDs not applicable for yeast
            if upper.startswith("ENSG") or upper.startswith("ENSMUSG"):
                logger.warning(
                    "Ensembl IDs not supported for yeast: %s", gene_id
                )
                return None

        # ── Human-specific patterns ──────────────────────────────────────
        if self.species == "human":
            # ENSG (Ensembl gene ID) — this IS the systematic_name for human
            if _ENSG_RE.match(upper):
                # Try with version first, then base
                canonical = self._sys_upper_to_canonical.get(upper)
                if canonical is not None:
                    return canonical
                base = upper.split(".")[0]
                canonical = self._sys_upper_to_canonical.get(base)
                if canonical is not None:
                    return canonical
                return None

            # ENST (Ensembl transcript ID) → map to ENSG
            if _ENST_RE.match(upper):
                sys_name = self._enst_to_sys.get(upper)
                if sys_name is not None:
                    return sys_name
                base = upper.split(".")[0]
                sys_name = self._enst_to_sys.get(base)
                if sys_name is not None:
                    return sys_name
                return None

            # Pure numeric → try Entrez ID
            if gene_id.strip().isdigit():
                sys_name = self._entrez_to_sys.get(gene_id.strip())
                if sys_name is not None:
                    return sys_name
                return None

        # ── Mouse-specific patterns ──────────────────────────────────────
        if self.species == "mouse":
            # ENSMUSG (Ensembl mouse gene ID) — this IS the systematic_name
            if _ENSMUSG_RE.match(upper):
                canonical = self._sys_upper_to_canonical.get(upper)
                if canonical is not None:
                    return canonical
                base = upper.split(".")[0]
                canonical = self._sys_upper_to_canonical.get(base)
                if canonical is not None:
                    return canonical
                return None

            # ENSMUST (Ensembl mouse transcript ID) → map to ENSMUSG
            if _ENSMUST_RE.match(upper):
                sys_name = self._enst_to_sys.get(upper)
                if sys_name is not None:
                    return sys_name
                base = upper.split(".")[0]
                sys_name = self._enst_to_sys.get(base)
                if sys_name is not None:
                    return sys_name
                return None

        # ── Generic lookups (all species) ────────────────────────────────
        # Try common name lookup (case-insensitive)
        if upper in self._common_to_sys:
            return self._common_to_sys[upper]

        # Try as systematic name without regex (some edge cases)
        canonical = self._sys_upper_to_canonical.get(upper)
        if canonical is not None:
            return canonical

        # Human alias/previous symbol fallback
        if self.species == "human" and self._alias_to_info:
            info = self._alias_to_info.get(upper)
            if info is not None:
                sys_name, canonical_sym, alias_type = info
                logger.info(
                    "%s resolved as %s for %s (%s)",
                    gene_id, alias_type, canonical_sym, sys_name,
                )
                return sys_name

        return None

    def get_sequences(self, names) -> dict[str, str]:
        """Return {systematic_name: cds_sequence} for requested genes.

        Sequences are already validated and stop-codon-stripped.

        Args:
            names: Systematic names as any of:
                - list of strings: ["YOL058W", "YOR202W"]
                - IDMapping from resolve_ids() (uses the mapped values)
                - dict {input_id: systematic_name} (uses the values)
        """
        if isinstance(names, IDMapping):
            systematic_names = list(names.values())
        elif isinstance(names, dict):
            systematic_names = list(names.values())
        else:
            systematic_names = list(names)

        # Validate: all entries must be strings
        for i, name in enumerate(systematic_names):
            if not isinstance(name, str):
                raise TypeError(
                    f"get_sequences expects gene name strings, got "
                    f"{type(name).__name__} at index {i}. "
                    f"Hint: pass list(result.values()) or the IDMapping directly."
                )

        all_seqs = self._load_sequences()
        result: dict[str, str] = {}
        for name in systematic_names:
            if name in all_seqs:
                result[name] = all_seqs[name]
            else:
                logger.warning("No CDS sequence found for: %s", name)
        return result

    def get_sequences_for_ids(self, gene_ids: list[str]) -> dict[str, str]:
        """Convenience: resolve gene IDs and return their CDS sequences.

        Combines resolve_ids() + get_sequences() in one call.

        Args:
            gene_ids: List of gene identifiers (any format).

        Returns:
            {systematic_name: cds_sequence} for all successfully resolved genes.
        """
        result = self.resolve_ids(gene_ids)
        return self.get_sequences(result)

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
