"""CodonScope CLI entry point."""

import argparse
import logging
import sys
from pathlib import Path

from codonscope import __version__


def main(argv: list[str] | None = None) -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="codonscope",
        description="CodonScope: Multi-species codon usage analysis tool",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── download ──────────────────────────────────────────────────────────
    dl_parser = subparsers.add_parser(
        "download", help="Download reference data for a species"
    )
    dl_parser.add_argument(
        "--species", nargs="+", required=True,
        help="Species to download (e.g. yeast, human, mouse)",
    )
    dl_parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Override default data directory (~/.codonscope/data/species/)",
    )

    # ── composition (Mode 1) ─────────────────────────────────────────────
    comp_parser = subparsers.add_parser(
        "composition", help="Mode 1: Sequence composition analysis"
    )
    comp_parser.add_argument(
        "--species", required=True, help="Species name (e.g. yeast)"
    )
    comp_parser.add_argument(
        "--genes", required=True,
        help="Path to gene list file (one ID per line, or comma-separated)",
    )
    comp_parser.add_argument(
        "--kmer", type=int, default=1, choices=[1, 2, 3],
        help="K-mer size: 1=monocodon, 2=dicodon, 3=tricodon (default: 1)",
    )
    comp_parser.add_argument(
        "--background", default="all", choices=["all", "matched"],
        help="Background type: all genome CDSs or length+GC matched (default: all)",
    )
    comp_parser.add_argument(
        "--trim-ramp", type=int, default=0,
        help="Exclude first N codons from 5' end (default: 0)",
    )
    comp_parser.add_argument(
        "--min-genes", type=int, default=10,
        help="Minimum gene list size (default: 10)",
    )
    comp_parser.add_argument(
        "--n-bootstrap", type=int, default=10000,
        help="Number of bootstrap iterations (default: 10000)",
    )
    comp_parser.add_argument(
        "--output-dir", type=str, default="./codonscope_output",
        help="Output directory (default: ./codonscope_output)",
    )
    comp_parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility",
    )
    comp_parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Override default data directory",
    )

    # ── disentangle (Mode 5) ──────────────────────────────────────────────
    dis_parser = subparsers.add_parser(
        "disentangle", help="Mode 5: AA vs codon disentanglement"
    )
    dis_parser.add_argument(
        "--species", required=True, help="Species name (e.g. yeast, human)"
    )
    dis_parser.add_argument(
        "--genes", required=True,
        help="Path to gene list file (one ID per line, or comma-separated)",
    )
    dis_parser.add_argument(
        "--n-bootstrap", type=int, default=10000,
        help="Number of bootstrap iterations (default: 10000)",
    )
    dis_parser.add_argument(
        "--output-dir", type=str, default="./codonscope_output",
        help="Output directory (default: ./codonscope_output)",
    )
    dis_parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility",
    )
    dis_parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Override default data directory",
    )

    args = parser.parse_args(argv)

    # Set up logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s",
    )

    if args.command is None:
        parser.print_help()
        return 1

    if args.command == "download":
        return _cmd_download(args)
    elif args.command == "composition":
        return _cmd_composition(args)
    elif args.command == "disentangle":
        return _cmd_disentangle(args)
    else:
        parser.print_help()
        return 1


# ═══════════════════════════════════════════════════════════════════════════════
# Command handlers
# ═══════════════════════════════════════════════════════════════════════════════

def _cmd_download(args: argparse.Namespace) -> int:
    """Handle the download subcommand."""
    from codonscope.data.download import download

    for species in args.species:
        try:
            download(species, data_dir=args.data_dir)
            print(f"Download complete: {species}")
        except Exception as exc:
            logging.error("Download failed for %s: %s", species, exc)
            return 1
    return 0


def _cmd_composition(args: argparse.Namespace) -> int:
    """Handle the composition subcommand."""
    from codonscope.modes.mode1_composition import run_composition

    gene_ids = _parse_gene_list(args.genes)
    if not gene_ids:
        logging.error("No gene IDs found in %s", args.genes)
        return 1

    print(f"CodonScope Mode 1: Sequence Composition")
    print(f"  Species: {args.species}")
    print(f"  Genes: {len(gene_ids)} IDs from {args.genes}")
    print(f"  K-mer: {args.kmer} ({'mono' if args.kmer == 1 else 'di' if args.kmer == 2 else 'tri'}codon)")
    print(f"  Background: {args.background}")
    if args.trim_ramp > 0:
        print(f"  Trim ramp: first {args.trim_ramp} codons excluded")
    print()

    result = run_composition(
        species=args.species,
        gene_ids=gene_ids,
        k=args.kmer,
        background=args.background,
        trim_ramp=args.trim_ramp,
        min_genes=args.min_genes,
        n_bootstrap=args.n_bootstrap,
        output_dir=args.output_dir,
        seed=args.seed,
        data_dir=args.data_dir,
    )

    df = result["results"]
    diag = result["diagnostics"]

    # Print summary
    print(f"Genes analyzed: {result['n_genes']}")
    id_sum = result["id_summary"]
    if id_sum["n_unmapped"] > 0:
        print(f"  Unmapped IDs: {id_sum['n_unmapped']}")

    # Diagnostics
    if diag.get("length_warning"):
        print(f"  WARNING: Gene-set CDS lengths differ from background "
              f"(KS p={diag['length_p']:.2e}). Consider --background matched.")
    if diag.get("gc_warning"):
        print(f"  WARNING: Gene-set GC content differs from background "
              f"(KS p={diag['gc_p']:.2e}). Consider --background matched.")
    for w in diag.get("power_warnings", []):
        print(f"  WARNING: {w}")

    # Top results
    sig = df[df["adjusted_p"] < 0.05]
    print(f"\nSignificant k-mers (adj_p < 0.05): {len(sig)}")

    if len(sig) > 0:
        print(f"\nTop enriched:")
        enriched = sig[sig["z_score"] > 0].head(10)
        for _, row in enriched.iterrows():
            print(f"  {row['kmer']:>12s}  Z={row['z_score']:+6.2f}  "
                  f"obs={row['observed_freq']:.4f}  exp={row['expected_freq']:.4f}  "
                  f"adj_p={row['adjusted_p']:.2e}")

        depleted = sig[sig["z_score"] < 0].head(10)
        if len(depleted) > 0:
            print(f"\nTop depleted:")
            for _, row in depleted.iterrows():
                print(f"  {row['kmer']:>12s}  Z={row['z_score']:+6.2f}  "
                      f"obs={row['observed_freq']:.4f}  exp={row['expected_freq']:.4f}  "
                      f"adj_p={row['adjusted_p']:.2e}")

    if args.output_dir:
        print(f"\nOutput files written to {args.output_dir}/")

    return 0


def _cmd_disentangle(args: argparse.Namespace) -> int:
    """Handle the disentangle subcommand."""
    from codonscope.modes.mode5_disentangle import run_disentangle

    gene_ids = _parse_gene_list(args.genes)
    if not gene_ids:
        logging.error("No gene IDs found in %s", args.genes)
        return 1

    print("CodonScope Mode 5: AA vs Codon Disentanglement")
    print(f"  Species: {args.species}")
    print(f"  Genes: {len(gene_ids)} IDs from {args.genes}")
    print()

    result = run_disentangle(
        species=args.species,
        gene_ids=gene_ids,
        n_bootstrap=args.n_bootstrap,
        output_dir=args.output_dir,
        seed=args.seed,
        data_dir=args.data_dir,
    )

    summary = result["summary"]
    print(f"Genes analyzed: {result['n_genes']}")
    print(f"\n{summary['summary_text']}")

    # Show top AA deviations
    aa_sig = result["aa_results"][result["aa_results"]["adjusted_p"] < 0.05]
    if len(aa_sig) > 0:
        print(f"\nSignificant amino acid deviations ({len(aa_sig)}):")
        for _, row in aa_sig.head(10).iterrows():
            direction = "enriched" if row["z_score"] > 0 else "depleted"
            print(f"  {row['amino_acid']:>4s}  Z={row['z_score']:+6.2f}  "
                  f"adj_p={row['adjusted_p']:.2e}  ({direction})")

    # Show top RSCU deviations
    rscu_sig = result["rscu_results"][result["rscu_results"]["adjusted_p"] < 0.05]
    if len(rscu_sig) > 0:
        print(f"\nSignificant RSCU deviations ({len(rscu_sig)}):")
        for _, row in rscu_sig.head(10).iterrows():
            direction = "preferred" if row["z_score"] > 0 else "avoided"
            print(f"  {row['codon']:>3s} ({row['amino_acid']:>3s})  "
                  f"RSCU Z={row['z_score']:+6.2f}  "
                  f"adj_p={row['adjusted_p']:.2e}  ({direction})")

    if args.output_dir:
        print(f"\nOutput files written to {args.output_dir}/")

    return 0


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_gene_list(filepath: str) -> list[str]:
    """Parse a gene list file.

    Supports:
    - One gene ID per line
    - Comma-separated IDs
    - Mixed (comma-separated on multiple lines)
    - Lines starting with # are comments
    """
    path = Path(filepath)
    if not path.exists():
        logging.error("Gene list file not found: %s", filepath)
        return []

    gene_ids: list[str] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Split by comma, tab, or whitespace
            for token in line.replace(",", " ").replace("\t", " ").split():
                token = token.strip()
                if token:
                    gene_ids.append(token)

    return gene_ids


if __name__ == "__main__":
    sys.exit(main())
