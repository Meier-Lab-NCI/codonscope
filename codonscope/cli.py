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

    # ── enrichment (was composition) ──────────────────────────────────────
    comp_parser = subparsers.add_parser(
        "enrichment", aliases=["composition"],
        help="Codon/dicodon enrichment analysis (was Mode 1)",
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
    comp_parser.add_argument(
        "--model", default="bootstrap", choices=["bootstrap", "binomial"],
        help="Statistical model: bootstrap (default) or binomial (GC3-corrected GLM, k=1 only)",
    )

    # ── optimality (was profile) ─────────────────────────────────────────
    prof_parser = subparsers.add_parser(
        "optimality", aliases=["profile"],
        help="Translational optimality profile (was Mode 3)",
    )
    prof_parser.add_argument(
        "--species", required=True, help="Species name (e.g. yeast, human)"
    )
    prof_parser.add_argument(
        "--genes", required=True,
        help="Path to gene list file (one ID per line, or comma-separated)",
    )
    prof_parser.add_argument(
        "--window", type=int, default=10,
        help="Sliding window size in codons (default: 10)",
    )
    prof_parser.add_argument(
        "--wobble-penalty", type=float, default=0.5,
        help="Wobble decoding penalty for wtAI (default: 0.5)",
    )
    prof_parser.add_argument(
        "--ramp-codons", type=int, default=50,
        help="Number of 5' codons for ramp analysis (default: 50)",
    )
    prof_parser.add_argument(
        "--method", default="wtai", choices=["tai", "wtai"],
        help="Scoring method: tai or wtai (default: wtai)",
    )
    prof_parser.add_argument(
        "--output-dir", type=str, default="./codonscope_output",
        help="Output directory (default: ./codonscope_output)",
    )
    prof_parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Override default data directory",
    )

    # ── collision ─────────────────────────────────────────────────────────
    col_parser = subparsers.add_parser(
        "collision", help="Collision potential analysis (FS transitions)",
    )
    col_parser.add_argument(
        "--species", required=True, help="Species name (e.g. yeast, human)"
    )
    col_parser.add_argument(
        "--genes", required=True,
        help="Path to gene list file (one ID per line, or comma-separated)",
    )
    col_parser.add_argument(
        "--wobble-penalty", type=float, default=0.5,
        help="Wobble decoding penalty for wtAI (default: 0.5)",
    )
    col_parser.add_argument(
        "--threshold", type=float, default=None,
        help="Fast/slow cutoff (default: median wtAI)",
    )
    col_parser.add_argument(
        "--method", default="wtai", choices=["tai", "wtai"],
        help="Scoring method: tai or wtai (default: wtai)",
    )
    col_parser.add_argument(
        "--output-dir", type=str, default="./codonscope_output",
        help="Output directory (default: ./codonscope_output)",
    )
    col_parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Override default data directory",
    )

    # ── demand ────────────────────────────────────────────────────────────
    dem_parser = subparsers.add_parser(
        "demand", help="Translational demand analysis (expression-weighted)",
    )
    dem_parser.add_argument(
        "--species", required=True, help="Species name (e.g. yeast, human)"
    )
    dem_parser.add_argument(
        "--genes", required=True,
        help="Path to gene list file (one ID per line, or comma-separated)",
    )
    dem_parser.add_argument(
        "--kmer", type=int, default=1, choices=[1, 2, 3],
        help="K-mer size: 1=monocodon, 2=dicodon, 3=tricodon (default: 1)",
    )
    dem_parser.add_argument(
        "--tissue", type=str, default=None,
        help="GTEx tissue name for human (default: HEK293T proxy)",
    )
    dem_parser.add_argument(
        "--cell-line", type=str, default=None,
        help="CCLE cell line for human (e.g. HEK293T, HeLa, K562)",
    )
    dem_parser.add_argument(
        "--expression", type=str, default=None,
        help="Path to custom expression file (TSV with gene_id, tpm columns)",
    )
    dem_parser.add_argument(
        "--top-n", type=int, default=None,
        help="Only use top-N expressed genes in background (default: all)",
    )
    dem_parser.add_argument(
        "--n-bootstrap", type=int, default=10000,
        help="Number of bootstrap iterations (default: 10000)",
    )
    dem_parser.add_argument(
        "--output-dir", type=str, default="./codonscope_output",
        help="Output directory (default: ./codonscope_output)",
    )
    dem_parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility",
    )
    dem_parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Override default data directory",
    )

    # ── report (HTML) ────────────────────────────────────────────────────
    rep_parser = subparsers.add_parser(
        "report", help="Generate comprehensive HTML report"
    )
    rep_parser.add_argument(
        "--species", required=True, help="Primary species (e.g. yeast, human)"
    )
    rep_parser.add_argument(
        "--genes", required=True,
        help="Path to gene list file (one ID per line, or comma-separated)",
    )
    rep_parser.add_argument(
        "--output", type=str, default="report.html",
        help="Output HTML file path (default: report.html)",
    )
    rep_parser.add_argument(
        "--species2", type=str, default=None,
        help="Second species for cross-species comparison",
    )
    rep_parser.add_argument(
        "--tissue", type=str, default=None,
        help="GTEx tissue for human demand analysis",
    )
    rep_parser.add_argument(
        "--cell-line", type=str, default=None,
        help="CCLE cell line for human demand analysis",
    )
    rep_parser.add_argument(
        "--n-bootstrap", type=int, default=10000,
        help="Number of bootstrap iterations (default: 10000)",
    )
    rep_parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility",
    )
    rep_parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Override default data directory",
    )
    rep_parser.add_argument(
        "--model", default="bootstrap", choices=["bootstrap", "binomial"],
        help="Statistical model: bootstrap (default) or binomial (GC3-corrected GLM)",
    )

    # ── compare ──────────────────────────────────────────────────────────
    cmp_parser = subparsers.add_parser(
        "compare", help="Cross-species RSCU comparison",
    )
    cmp_parser.add_argument(
        "--species1", required=True, help="First species (e.g. yeast)"
    )
    cmp_parser.add_argument(
        "--species2", required=True, help="Second species (e.g. human)"
    )
    cmp_parser.add_argument(
        "--genes", required=True,
        help="Path to gene list file (one ID per line, or comma-separated)",
    )
    cmp_parser.add_argument(
        "--from-species", type=str, default=None,
        help="Which species the gene IDs belong to (default: species1)",
    )
    cmp_parser.add_argument(
        "--n-bootstrap", type=int, default=10000,
        help="Number of bootstrap iterations (default: 10000)",
    )
    cmp_parser.add_argument(
        "--output-dir", type=str, default="./codonscope_output",
        help="Output directory (default: ./codonscope_output)",
    )
    cmp_parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility",
    )
    cmp_parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Override default data directory",
    )

    # ── attribution (was disentangle) ────────────────────────────────────
    dis_parser = subparsers.add_parser(
        "attribution", aliases=["disentangle"],
        help="AA vs synonymous attribution (was Mode 5)",
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

    # ── cai ───────────────────────────────────────────────────────────────
    cai_parser = subparsers.add_parser(
        "cai", help="Codon Adaptation Index (CAI) analysis",
    )
    cai_parser.add_argument(
        "--species", required=True, help="Species name (e.g. yeast, human)"
    )
    cai_parser.add_argument(
        "--genes", required=True,
        help="Path to gene list file (one ID per line, or comma-separated)",
    )
    cai_parser.add_argument(
        "--output-dir", type=str, default="./codonscope_output",
        help="Output directory (default: ./codonscope_output)",
    )
    cai_parser.add_argument(
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
    elif args.command == "report":
        return _cmd_report(args)
    elif args.command in ("enrichment", "composition"):
        return _cmd_composition(args)
    elif args.command in ("optimality", "profile"):
        return _cmd_profile(args)
    elif args.command == "collision":
        return _cmd_collision(args)
    elif args.command == "demand":
        return _cmd_demand(args)
    elif args.command == "compare":
        return _cmd_compare(args)
    elif args.command in ("attribution", "disentangle"):
        return _cmd_disentangle(args)
    elif args.command == "cai":
        return _cmd_cai(args)
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


def _cmd_report(args: argparse.Namespace) -> int:
    """Handle the report subcommand."""
    from codonscope.report import generate_report

    gene_ids = _parse_gene_list(args.genes)
    if not gene_ids:
        logging.error("No gene IDs found in %s", args.genes)
        return 1

    cell_line = getattr(args, "cell_line", None)
    if args.tissue and cell_line:
        logging.error("Cannot specify both --tissue and --cell-line")
        return 1

    print("CodonScope: Generating comprehensive HTML report")
    print(f"  Species: {args.species}")
    print(f"  Genes: {len(gene_ids)} IDs from {args.genes}")
    if args.species2:
        print(f"  Cross-species: {args.species2}")
    if args.tissue:
        print(f"  Tissue: {args.tissue}")
    elif cell_line:
        print(f"  Cell line: {cell_line}")
    print()

    output = generate_report(
        species=args.species,
        gene_ids=gene_ids,
        output=args.output,
        species2=args.species2,
        tissue=args.tissue,
        cell_line=cell_line,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        data_dir=args.data_dir,
        model=getattr(args, "model", "bootstrap"),
    )

    print(f"Report written to {output}")
    return 0


def _cmd_composition(args: argparse.Namespace) -> int:
    """Handle the enrichment/composition subcommand."""
    from codonscope.modes.mode1_composition import run_composition

    gene_ids = _parse_gene_list(args.genes)
    if not gene_ids:
        logging.error("No gene IDs found in %s", args.genes)
        return 1

    kmer_labels = {1: "Codon", 2: "Dicodon", 3: "Tricodon"}
    print(f"CodonScope: {kmer_labels[args.kmer]} Enrichment Analysis")
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
        model=args.model,
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
        from codonscope.core.codons import annotate_kmer
        print(f"\nTop enriched:")
        enriched = sig[sig["z_score"] > 0].head(10)
        for _, row in enriched.iterrows():
            label = annotate_kmer(row["kmer"], args.kmer)
            print(f"  {label:>24s}  Z={row['z_score']:+6.2f}  "
                  f"obs={row['observed_freq']:.4f}  exp={row['expected_freq']:.4f}  "
                  f"adj_p={row['adjusted_p']:.2e}")

        depleted = sig[sig["z_score"] < 0].head(10)
        if len(depleted) > 0:
            print(f"\nTop depleted:")
            for _, row in depleted.iterrows():
                label = annotate_kmer(row["kmer"], args.kmer)
                print(f"  {label:>24s}  Z={row['z_score']:+6.2f}  "
                      f"obs={row['observed_freq']:.4f}  exp={row['expected_freq']:.4f}  "
                      f"adj_p={row['adjusted_p']:.2e}")

    if args.output_dir:
        print(f"\nOutput files written to {args.output_dir}/")

    return 0


def _cmd_demand(args: argparse.Namespace) -> int:
    """Handle the demand subcommand."""
    from codonscope.modes.mode2_demand import run_demand

    gene_ids = _parse_gene_list(args.genes)
    if not gene_ids:
        logging.error("No gene IDs found in %s", args.genes)
        return 1

    cell_line = getattr(args, "cell_line", None)
    if args.tissue and cell_line:
        logging.error("Cannot specify both --tissue and --cell-line")
        return 1

    k = args.kmer
    print("CodonScope: Translational Demand Analysis")
    print(f"  Species: {args.species}")
    print(f"  Genes: {len(gene_ids)} IDs from {args.genes}")
    print(f"  K-mer: {k} ({'mono' if k == 1 else 'di' if k == 2 else 'tri'}codon)")
    if args.tissue:
        print(f"  Tissue: {args.tissue}")
    elif cell_line:
        print(f"  Cell line: {cell_line}")
    elif args.species.lower() == "human" and not args.expression:
        print("  Expression: HEK293T (default, use --tissue or --cell-line to change)")
    if args.top_n:
        print(f"  Top-N background: {args.top_n}")
    print()

    result = run_demand(
        species=args.species,
        gene_ids=gene_ids,
        k=k,
        tissue=args.tissue,
        cell_line=cell_line,
        expression_file=args.expression,
        top_n=args.top_n,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
    )

    df = result["results"]
    top = result["top_genes"]

    print(f"Genes analyzed: {result['n_genes']}")
    print(f"Expression data: {result['tissue']}")

    # Top demand-contributing genes
    print(f"\nTop demand-contributing genes:")
    for _, row in top.head(10).iterrows():
        print(f"  {row['gene']:>15s}  TPM={row['tpm']:8.1f}  "
              f"codons={row['n_codons']:5d}  "
              f"demand={row['demand_fraction']*100:5.1f}%")

    # Significant k-mers
    sig = df[df["adjusted_p"] < 0.05]
    print(f"\nSignificant k-mers (adj_p < 0.05): {len(sig)}")

    if len(sig) > 0:
        from codonscope.core.codons import annotate_kmer
        print(f"\nTop demand-enriched:")
        enriched = sig[sig["z_score"] > 0].head(10)
        for _, row in enriched.iterrows():
            label = annotate_kmer(row["kmer"], k)
            print(f"  {label:>24s}  Z={row['z_score']:+6.2f}  "
                  f"demand={row['demand_geneset']:.4f}  "
                  f"genome={row['demand_genome']:.4f}  "
                  f"adj_p={row['adjusted_p']:.2e}")

        depleted = sig[sig["z_score"] < 0].head(10)
        if len(depleted) > 0:
            print(f"\nTop demand-depleted:")
            for _, row in depleted.iterrows():
                label = annotate_kmer(row["kmer"], k)
                print(f"  {label:>24s}  Z={row['z_score']:+6.2f}  "
                      f"demand={row['demand_geneset']:.4f}  "
                      f"genome={row['demand_genome']:.4f}  "
                      f"adj_p={row['adjusted_p']:.2e}")

    if args.output_dir:
        print(f"\nOutput files written to {args.output_dir}/")

    return 0


def _cmd_profile(args: argparse.Namespace) -> int:
    """Handle the optimality/profile subcommand."""
    from codonscope.modes.mode3_profile import run_profile

    gene_ids = _parse_gene_list(args.genes)
    if not gene_ids:
        logging.error("No gene IDs found in %s", args.genes)
        return 1

    print("CodonScope: Translational Optimality Profile")
    print(f"  Species: {args.species}")
    print(f"  Genes: {len(gene_ids)} IDs from {args.genes}")
    print(f"  Method: {args.method}")
    print(f"  Window: {args.window} codons")
    print()

    result = run_profile(
        species=args.species,
        gene_ids=gene_ids,
        window=args.window,
        wobble_penalty=args.wobble_penalty,
        ramp_codons=args.ramp_codons,
        method=args.method,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
    )

    print(f"Genes analyzed: {result['n_genes']}")

    # Ramp analysis
    ramp = result["ramp_analysis"]
    print(f"\nRamp analysis (first {ramp['ramp_codons']} codons):")
    print(f"  Gene set:  ramp={ramp['geneset_ramp_mean']:.4f}  "
          f"body={ramp['geneset_body_mean']:.4f}  "
          f"delta={ramp['geneset_ramp_delta']:+.4f}")
    print(f"  Genome:    ramp={ramp['genome_ramp_mean']:.4f}  "
          f"body={ramp['genome_body_mean']:.4f}  "
          f"delta={ramp['genome_ramp_delta']:+.4f}")

    # Per-gene score summary
    scores = result["per_gene_scores"]
    col = "wtai" if args.method == "wtai" else "tai"
    print(f"\nPer-gene {args.method.upper()} summary:")
    print(f"  Mean: {scores[col].mean():.4f}")
    print(f"  Std:  {scores[col].std():.4f}")
    print(f"  Min:  {scores[col].min():.4f}")
    print(f"  Max:  {scores[col].max():.4f}")

    if args.output_dir:
        print(f"\nOutput files written to {args.output_dir}/")

    return 0


def _cmd_collision(args: argparse.Namespace) -> int:
    """Handle the collision subcommand."""
    from codonscope.modes.mode4_collision import run_collision

    gene_ids = _parse_gene_list(args.genes)
    if not gene_ids:
        logging.error("No gene IDs found in %s", args.genes)
        return 1

    print("CodonScope: Collision Potential Analysis")
    print(f"  Species: {args.species}")
    print(f"  Genes: {len(gene_ids)} IDs from {args.genes}")
    print(f"  Method: {args.method}")
    print()

    result = run_collision(
        species=args.species,
        gene_ids=gene_ids,
        wobble_penalty=args.wobble_penalty,
        threshold=args.threshold,
        method=args.method,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
    )

    print(f"Genes analyzed: {result['n_genes']}")
    print(f"Fast/slow threshold: {result['threshold']:.4f}")
    print(f"Fast codons: {len(result['fast_codons'])}  "
          f"Slow codons: {len(result['slow_codons'])}")

    # Transition matrices
    gs = result["transition_matrix_geneset"]
    bg = result["transition_matrix_genome"]
    print(f"\nTransition proportions:")
    print(f"  {'Type':>4s}  {'Gene set':>10s}  {'Genome':>10s}")
    for t in ("FF", "FS", "SF", "SS"):
        print(f"  {t:>4s}  {gs[t]:10.4f}  {bg[t]:10.4f}")

    print(f"\nFS enrichment (gene set / genome): {result['fs_enrichment']:.3f}")
    print(f"FS/SF ratio — gene set: {result['fs_sf_ratio_geneset']:.3f}  "
          f"genome: {result['fs_sf_ratio_genome']:.3f}")
    print(f"Chi-squared: {result['chi2_stat']:.2f}  p={result['chi2_p']:.2e}")

    if args.output_dir:
        print(f"\nOutput files written to {args.output_dir}/")

    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    """Handle the compare subcommand."""
    from codonscope.modes.mode6_compare import run_compare

    gene_ids = _parse_gene_list(args.genes)
    if not gene_ids:
        logging.error("No gene IDs found in %s", args.genes)
        return 1

    from_species = args.from_species or args.species1
    print("CodonScope: Cross-Species Comparison")
    print(f"  Species: {args.species1} vs {args.species2}")
    print(f"  Genes: {len(gene_ids)} IDs from {args.genes} ({from_species})")
    print()

    result = run_compare(
        species1=args.species1,
        species2=args.species2,
        gene_ids=gene_ids,
        from_species=from_species,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
    )

    summary = result["summary"]
    print(f"Ortholog pairs analyzed: {result['n_orthologs']}")
    print(f"Genome ortholog pairs: {result['n_genome_orthologs']}")

    print(f"\nRSCU correlation summary:")
    print(f"  Gene set:  mean r = {summary['geneset_mean_r']:.4f}  "
          f"median = {summary['geneset_median_r']:.4f}  "
          f"std = {summary['geneset_std_r']:.4f}")
    print(f"  Genome:    mean r = {summary['genome_mean_r']:.4f}  "
          f"median = {summary['genome_median_r']:.4f}  "
          f"std = {summary['genome_std_r']:.4f}")
    print(f"  Z-score: {summary['z_score']:+.2f}  p = {summary['p_value']:.2e}")

    # Top conserved and divergent
    pg = result["per_gene"]
    from_col = f"{result['from_species']}_gene"
    to_col = f"{result['to_species']}_gene"

    print(f"\nMost conserved (highest RSCU r):")
    top = pg.nlargest(5, "rscu_correlation")
    for _, row in top.iterrows():
        print(f"  {row[from_col]:>15s} <-> {row[to_col]:<15s}  "
              f"r = {row['rscu_correlation']:.4f}")

    print(f"\nMost divergent (lowest RSCU r):")
    bot = pg.nsmallest(5, "rscu_correlation")
    for _, row in bot.iterrows():
        print(f"  {row[from_col]:>15s} <-> {row[to_col]:<15s}  "
              f"r = {row['rscu_correlation']:.4f}")

    if args.output_dir:
        print(f"\nOutput files written to {args.output_dir}/")

    return 0


def _cmd_disentangle(args: argparse.Namespace) -> int:
    """Handle the attribution/disentangle subcommand."""
    from codonscope.modes.mode5_disentangle import run_disentangle

    gene_ids = _parse_gene_list(args.genes)
    if not gene_ids:
        logging.error("No gene IDs found in %s", args.genes)
        return 1

    print("CodonScope: AA vs Synonymous Attribution")
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


def _cmd_cai(args: argparse.Namespace) -> int:
    """Handle the cai subcommand."""
    from codonscope.core.cai import cai_analysis

    gene_ids = _parse_gene_list(args.genes)
    if not gene_ids:
        logging.error("No gene IDs found in %s", args.genes)
        return 1

    print("CodonScope: Codon Adaptation Index (CAI)")
    print(f"  Species: {args.species}")
    print(f"  Genes: {len(gene_ids)} IDs from {args.genes}")
    print()

    result = cai_analysis(
        species=args.species,
        gene_ids=gene_ids,
        data_dir=args.data_dir,
    )

    print(f"Gene set mean CAI: {result['geneset_mean']:.4f}")
    print(f"Gene set median CAI: {result['geneset_median']:.4f}")
    print(f"Genome mean CAI: {result['genome_mean']:.4f}")
    print(f"Genome median CAI: {result['genome_median']:.4f}")
    print(f"Percentile rank: {result['percentile_rank']:.1f}")
    print(f"Mann-Whitney U: {result['mann_whitney_u']:.1f}")
    print(f"Mann-Whitney p: {result['mann_whitney_p']:.2e}")
    print(f"Reference set: {result['reference_n_genes']} genes")

    # Per-gene results
    per_gene = result["per_gene"]
    print(f"\nTop CAI genes:")
    top = per_gene.nlargest(10, "cai")
    for _, row in top.iterrows():
        print(f"  {row['gene']:>15s}  CAI={row['cai']:.4f}")

    print(f"\nBottom CAI genes:")
    bot = per_gene.nsmallest(5, "cai")
    for _, row in bot.iterrows():
        print(f"  {row['gene']:>15s}  CAI={row['cai']:.4f}")

    # Write output
    if args.output_dir:
        from pathlib import Path
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        per_gene.to_csv(out / "cai_per_gene.tsv", sep="\t", index=False, float_format="%.6g")
        import pandas as pd
        pd.DataFrame([
            {"codon": c, "weight": w} for c, w in sorted(result["weights"].items())
        ]).to_csv(out / "cai_weights.tsv", sep="\t", index=False, float_format="%.6g")
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
