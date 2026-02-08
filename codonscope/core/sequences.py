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

# RefSeq NM_ transcript pattern
_REFSEQ_NM_RE = re.compile(r"^NM_\d+(\.\d+)?$", re.IGNORECASE)

# MGI ID pattern (MGI:12345)
_MGI_ID_RE = re.compile(r"^MGI:\d+$", re.IGNORECASE)

# SGD ID pattern (SGD:S000001234)
_SGD_ID_RE = re.compile(r"^SGD:S\d+$", re.IGNORECASE)

# UniProt accession pattern (SwissProt format)
_UNIPROT_RE = re.compile(
    r"^[OPQ][0-9][A-Z0-9]{3}[0-9]$|^[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}$",
    re.IGNORECASE,
)

# Entrez gene ID (4+ digits — shorter numbers are unlikely to be Entrez IDs)
_ENTREZ_RE = re.compile(r"^\d{4,}$")


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

        # ── New multi-type ID resolution lookups ────────────────────────────

        # UniProt mapping (all species)
        self._uniprot_to_sys: dict[str, str] = {}
        uniprot_path = self._species_dir / "uniprot_mapping.tsv"
        if uniprot_path.exists():
            up_df = pd.read_csv(uniprot_path, sep="\t")
            for _, row in up_df.iterrows():
                uid = str(row.get("uniprot_id", "")).strip().upper()
                sname = str(row.get("systematic_name", "")).strip()
                if uid and sname and uid != "NAN":
                    self._uniprot_to_sys[uid] = sname

        # SGD ID mapping (yeast only)
        self._sgd_to_sys: dict[str, str] = {}
        if self.species == "yeast":
            for _, row in self._gene_map.iterrows():
                sgd = str(row.get("sgd_id", "")).strip().upper()
                sname = str(row["systematic_name"]).strip()
                if sgd and sgd != "NAN" and sgd.startswith("SGD:"):
                    self._sgd_to_sys[sgd] = sname

        # RefSeq NM_ mapping (human — already in gene_id_map)
        self._refseq_to_sys: dict[str, str] = {}
        if self.species == "human":
            for _, row in self._gene_map.iterrows():
                ref = str(row.get("refseq_transcript", "")).strip()
                sname = str(row["systematic_name"]).strip()
                if ref and ref != "nan" and ref.upper().startswith("NM_"):
                    ref_upper = ref.upper()
                    self._refseq_to_sys[ref_upper] = sname
                    ref_base = ref_upper.split(".")[0]
                    self._refseq_to_sys[ref_base] = sname

        # MGI ID + synonym mapping (mouse only)
        self._mgi_to_sys: dict[str, str] = {}
        self._mgi_synonym_to_sys: dict[str, str] = {}
        self._mouse_entrez_to_sys: dict[str, str] = {}
        mgi_path = self._species_dir / "mgi_mapping.tsv"
        if self.species == "mouse" and mgi_path.exists():
            mgi_df = pd.read_csv(mgi_path, sep="\t")
            for _, row in mgi_df.iterrows():
                ensmusg = str(row.get("ensmusg", "")).strip()
                if not ensmusg or ensmusg == "nan":
                    continue
                # MGI ID → ENSMUSG
                mgi_id = str(row.get("mgi_id", "")).strip().upper()
                if mgi_id and mgi_id.startswith("MGI:"):
                    self._mgi_to_sys[mgi_id] = ensmusg
                # Entrez ID → ENSMUSG
                entrez = str(row.get("entrez_id", "")).strip()
                if entrez and entrez != "nan" and entrez.replace(".", "").isdigit():
                    entrez_clean = str(int(float(entrez)))
                    self._mouse_entrez_to_sys[entrez_clean] = ensmusg
                # Synonyms → ENSMUSG
                synonyms = str(row.get("synonyms", "")).strip()
                if synonyms and synonyms != "nan":
                    for syn in synonyms.split("|"):
                        syn = syn.strip()
                        if syn:
                            syn_upper = syn.upper()
                            if syn_upper not in self._common_to_sys:
                                self._mgi_synonym_to_sys[syn_upper] = ensmusg

        # Mouse Entrez IDs from gene_id_map (BioMart source, may overlap with MGI)
        if self.species == "mouse":
            for _, row in self._gene_map.iterrows():
                entrez = str(row.get("entrez_id", "")).strip()
                sname = str(row["systematic_name"]).strip()
                if entrez and entrez != "nan" and entrez.replace(".", "").isdigit():
                    entrez_clean = str(int(float(entrez)))
                    if entrez_clean not in self._mouse_entrez_to_sys:
                        self._mouse_entrez_to_sys[entrez_clean] = sname

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
                id_type = self._detect_id_type(gid_stripped)
                if id_type == "ensg" and self.species == "human":
                    logger.warning(
                        "Could not resolve gene ID: %s — not in MANE Select "
                        "(may be non-coding, mitochondrial, pseudogene, or retired Ensembl ID)",
                        gid_stripped,
                    )
                elif id_type == "enst" and self.species == "human":
                    logger.warning(
                        "Could not resolve gene ID: %s — transcript not in MANE Select",
                        gid_stripped,
                    )
                elif id_type == "ensmusg" and self.species == "mouse":
                    logger.warning(
                        "Could not resolve gene ID: %s — not in Ensembl canonical CDS set "
                        "(may be non-coding, mitochondrial, or pseudogene)",
                        gid_stripped,
                    )
                elif id_type == "ensmust" and self.species == "mouse":
                    logger.warning(
                        "Could not resolve gene ID: %s — transcript not in Ensembl canonical CDS set",
                        gid_stripped,
                    )
                else:
                    logger.warning("Could not resolve gene ID: %s", gid_stripped)

        result = IDMapping(mapping, unmapped)
        logger.info(
            "ID resolution: %d mapped, %d unmapped",
            result.n_mapped, result.n_unmapped,
        )
        return result

    def _detect_id_type(self, gene_id: str) -> str:
        """Auto-detect ID type based on patterns.

        Returns a human-readable type string for logging.
        """
        upper = gene_id.upper()

        # Yeast systematic name
        if _YEAST_SYSTEMATIC_RE.match(gene_id):
            return "yeast_systematic"

        # Ensembl mouse gene ID
        if _ENSMUSG_RE.match(upper):
            return "ensmusg"

        # Ensembl mouse transcript ID
        if _ENSMUST_RE.match(upper):
            return "ensmust"

        # Ensembl human gene ID
        if _ENSG_RE.match(upper):
            return "ensg"

        # Ensembl human transcript ID
        if _ENST_RE.match(upper):
            return "enst"

        # Generic Ensembl pattern
        if upper.startswith("ENS") and any(c.isdigit() for c in upper):
            return "ensembl_other"

        # MGI ID
        if _MGI_ID_RE.match(gene_id):
            return "mgi_id"

        # SGD ID
        if _SGD_ID_RE.match(gene_id):
            return "sgd_id"

        # RefSeq NM_
        if _REFSEQ_NM_RE.match(gene_id):
            return "refseq_nm"

        # Pure digits (Entrez)
        if _ENTREZ_RE.match(gene_id):
            return "entrez"

        # UniProt accession
        if _UNIPROT_RE.match(gene_id):
            return "uniprot"

        return "gene_symbol"

    def _resolve_single_id(self, gene_id: str) -> str | None:
        """Resolve a single gene ID to systematic name.

        Uses auto-detection to determine ID type, then applies
        species-specific resolution in priority order.
        """
        upper = gene_id.upper()
        id_type = self._detect_id_type(gene_id)
        logger.debug("Resolving %s (detected: %s, species: %s)", gene_id, id_type, self.species)

        # ── 1. Ensembl gene IDs (direct systematic name lookup) ──────────
        if id_type == "ensmusg" and self.species == "mouse":
            canonical = self._sys_upper_to_canonical.get(upper)
            if canonical is not None:
                return canonical
            base = upper.split(".")[0]
            return self._sys_upper_to_canonical.get(base)

        if id_type == "ensg" and self.species == "human":
            canonical = self._sys_upper_to_canonical.get(upper)
            if canonical is not None:
                return canonical
            base = upper.split(".")[0]
            return self._sys_upper_to_canonical.get(base)

        # Wrong species Ensembl IDs
        if id_type in ("ensmusg", "ensmust") and self.species != "mouse":
            logger.warning("Mouse Ensembl ID %s used with species %s", gene_id, self.species)
            return None
        if id_type in ("ensg", "enst") and self.species != "human":
            logger.warning("Human Ensembl ID %s used with species %s", gene_id, self.species)
            return None

        # ── 2. Ensembl transcript IDs ────────────────────────────────────
        if id_type == "ensmust" and self.species == "mouse":
            sys_name = self._enst_to_sys.get(upper)
            if sys_name is not None:
                return sys_name
            base = upper.split(".")[0]
            return self._enst_to_sys.get(base)

        if id_type == "enst" and self.species == "human":
            sys_name = self._enst_to_sys.get(upper)
            if sys_name is not None:
                return sys_name
            base = upper.split(".")[0]
            return self._enst_to_sys.get(base)

        # ── 3. Yeast systematic names ────────────────────────────────────
        if id_type == "yeast_systematic":
            if self.species == "yeast":
                return self._sys_upper_to_canonical.get(upper)
            logger.warning("Yeast systematic name %s used with species %s", gene_id, self.species)
            return None

        # ── 4. Species-specific ID types ─────────────────────────────────

        # SGD ID (yeast)
        if id_type == "sgd_id":
            if self.species == "yeast" and self._sgd_to_sys:
                return self._sgd_to_sys.get(upper)
            return None

        # MGI ID (mouse)
        if id_type == "mgi_id":
            if self.species == "mouse" and self._mgi_to_sys:
                return self._mgi_to_sys.get(upper)
            return None

        # RefSeq NM_ (human)
        if id_type == "refseq_nm":
            if self.species == "human" and self._refseq_to_sys:
                ref_upper = upper
                sys_name = self._refseq_to_sys.get(ref_upper)
                if sys_name is not None:
                    return sys_name
                ref_base = ref_upper.split(".")[0]
                return self._refseq_to_sys.get(ref_base)
            return None

        # ── 5. Entrez ID (human + mouse) ─────────────────────────────────
        if id_type == "entrez":
            stripped = gene_id.strip()
            if self.species == "human":
                return self._entrez_to_sys.get(stripped)
            if self.species == "mouse":
                return self._mouse_entrez_to_sys.get(stripped)
            return None

        # ── 6. UniProt accession (all species) ───────────────────────────
        if id_type == "uniprot" and self._uniprot_to_sys:
            sys_name = self._uniprot_to_sys.get(upper)
            if sys_name is not None:
                return sys_name
            # Fall through to gene symbol lookup in case it's not a UniProt ID

        # ── 7. Gene symbol / common name (all species) ───────────────────
        if upper in self._common_to_sys:
            return self._common_to_sys[upper]

        # Try as systematic name without regex (edge cases)
        canonical = self._sys_upper_to_canonical.get(upper)
        if canonical is not None:
            return canonical

        # ── 8. Alias/synonym fallback ────────────────────────────────────

        # Human: HGNC alias/previous symbol
        if self.species == "human" and self._alias_to_info:
            info = self._alias_to_info.get(upper)
            if info is not None:
                sys_name, canonical_sym, alias_type = info
                logger.info(
                    "%s resolved as %s for %s (%s)",
                    gene_id, alias_type, canonical_sym, sys_name,
                )
                return sys_name

        # Mouse: MGI synonym fallback
        if self.species == "mouse" and self._mgi_synonym_to_sys:
            sys_name = self._mgi_synonym_to_sys.get(upper)
            if sys_name is not None:
                logger.info(
                    "%s resolved via MGI synonym to %s", gene_id, sys_name
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

    def get_common_names(self, systematic_names: list[str]) -> dict[str, str]:
        """Return {systematic_name: common_name} for given genes.

        Uses the gene_id_map.tsv data loaded at init time.
        Returns empty string for genes without a common name.
        """
        result: dict[str, str] = {}
        for name in systematic_names:
            upper = name.upper()
            common = self._sys_to_common.get(upper, "")
            result[name] = common if common else name
        return result

    @property
    def species_dir(self) -> Path:
        """Path to the species data directory."""
        return self._species_dir
