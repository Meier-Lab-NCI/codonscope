#!/usr/bin/env python3
"""Convert raw RNA-seq counts to TPM for use with CodonScope.

Usage:
    # If your counts file has a length column:
    python counts_to_tpm.py counts.tsv --length-column Length --output tpm.tsv

    # If no length column, pull CDS lengths from CodonScope data:
    python counts_to_tpm.py counts.tsv --species human --output tpm.tsv

    # featureCounts output (has Length column by default):
    python counts_to_tpm.py featureCounts.txt --length-column Length --output tpm.tsv

    # Multiple samples — outputs one TPM file per sample, or pick one:
    python counts_to_tpm.py counts.tsv --species human --sample tumor_rep1 --output tpm.tsv

Input format:
    Tab-separated file with a gene ID column and one or more count columns.
    The script auto-detects common formats (featureCounts, HTSeq, DESeq2 matrix).

Output format:
    Tab-separated file with gene_id and tpm columns, ready for CodonScope's
    custom expression upload.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def load_counts(path, gene_column=None, length_column=None, sample=None):
    """Load a raw counts file and return (counts_series, lengths_series_or_None).

    Handles common formats:
    - featureCounts: cols = Geneid, Chr, Start, End, Strand, Length, sample...
    - Generic matrix: gene_id col + numeric sample columns
    - Two-column HTSeq: gene_id, count
    """
    df = pd.read_csv(path, sep="\t", comment="#")

    # --- Detect gene ID column ---
    if gene_column:
        if gene_column not in df.columns:
            raise ValueError(f"Gene column '{gene_column}' not found. Columns: {list(df.columns)}")
        gene_col = gene_column
    else:
        # Auto-detect
        candidates = ["Geneid", "gene_id", "Gene", "gene", "GeneID",
                       "gene_name", "GeneName", "Symbol", "symbol"]
        gene_col = None
        for c in candidates:
            if c in df.columns:
                gene_col = c
                break
        if gene_col is None:
            # Assume first column is gene IDs
            gene_col = df.columns[0]
            print(f"  Auto-detected gene ID column: '{gene_col}' (first column)")

    df = df.set_index(gene_col)

    # --- Extract lengths if available ---
    lengths = None
    if length_column:
        if length_column not in df.columns:
            raise ValueError(f"Length column '{length_column}' not found. Columns: {list(df.columns)}")
        lengths = df[length_column].astype(float)
        df = df.drop(columns=[length_column])
    elif "Length" in df.columns:
        lengths = df["Length"].astype(float)
        print(f"  Auto-detected length column: 'Length'")
        df = df.drop(columns=["Length"])

    # --- Drop non-numeric metadata columns (featureCounts: Chr, Start, End, Strand) ---
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    meta_cols = [c for c in df.columns if c not in numeric_cols]
    if meta_cols:
        print(f"  Dropping non-numeric columns: {meta_cols}")
        df = df[numeric_cols]

    # --- Filter out summary rows (HTSeq adds __no_feature, __ambiguous, etc.) ---
    bad_rows = df.index.str.startswith("__")
    if bad_rows.any():
        n_bad = bad_rows.sum()
        print(f"  Dropping {n_bad} summary rows (__no_feature, etc.)")
        df = df[~bad_rows]
        if lengths is not None:
            lengths = lengths[~bad_rows]

    # --- Select sample ---
    if df.shape[1] == 1:
        counts = df.iloc[:, 0].astype(float)
        print(f"  Single sample: '{df.columns[0]}'")
    elif sample:
        if sample not in df.columns:
            raise ValueError(f"Sample '{sample}' not found. Available: {list(df.columns)}")
        counts = df[sample].astype(float)
        print(f"  Selected sample: '{sample}'")
    else:
        print(f"  Found {df.shape[1]} samples: {list(df.columns)}")
        print(f"  Using first sample: '{df.columns[0]}'")
        print(f"  (Use --sample to pick a different one)")
        counts = df.iloc[:, 0].astype(float)

    return counts, lengths


def load_codonscope_lengths(species):
    """Load CDS lengths from CodonScope's downloaded gene metadata."""
    data_dir = Path.home() / ".codonscope" / "data" / "species" / species

    # Try gene_id_map.tsv first (has cds_length column)
    gene_map = data_dir / "gene_id_map.tsv"
    if not gene_map.exists():
        raise FileNotFoundError(
            f"CodonScope data not found for '{species}'. "
            f"Run: python3 -c \"from codonscope.data.download import download; download('{species}')\""
        )

    df = pd.read_csv(gene_map, sep="\t")

    # Build a lookup from all known ID types to CDS length
    length_map = {}

    if "cds_length" in df.columns:
        length_col = "cds_length"
    else:
        raise ValueError(f"No cds_length column in {gene_map}")

    # Map systematic names
    if "systematic_name" in df.columns:
        for _, row in df.iterrows():
            length_map[row["systematic_name"]] = row[length_col]

    # Map common names
    if "common_name" in df.columns:
        for _, row in df.dropna(subset=["common_name"]).iterrows():
            length_map[row["common_name"]] = row[length_col]
            # Also map uppercase version for case-insensitive matching
            length_map[row["common_name"].upper()] = row[length_col]

    # Map Ensembl transcript IDs
    if "ensembl_transcript" in df.columns:
        for _, row in df.dropna(subset=["ensembl_transcript"]).iterrows():
            length_map[row["ensembl_transcript"]] = row[length_col]

    # Map Entrez IDs
    if "entrez_id" in df.columns:
        for _, row in df.dropna(subset=["entrez_id"]).iterrows():
            length_map[str(int(row["entrez_id"]))] = row[length_col]

    print(f"  Loaded {len(length_map)} gene-to-length mappings from CodonScope {species} data")
    return length_map


def counts_to_tpm(counts, lengths):
    """Convert raw counts + gene lengths (in bp) to TPM.

    TPM_i = (count_i / length_i) / sum(count_j / length_j) * 1e6
    """
    # Reads per kilobase
    rpk = counts / (lengths / 1000.0)

    # Scaling factor
    rpk_sum = rpk.sum()
    if rpk_sum == 0:
        raise ValueError("All counts are zero. Check your input file.")

    tpm = rpk / rpk_sum * 1e6
    return tpm


def main():
    parser = argparse.ArgumentParser(
        description="Convert raw RNA-seq counts to TPM for CodonScope.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("counts_file", help="Tab-separated raw counts file")
    parser.add_argument("-o", "--output", required=True, help="Output TPM file path")
    parser.add_argument("--species", help="Species for CDS lengths (human/yeast/mouse). "
                        "Used if counts file has no length column.")
    parser.add_argument("--gene-column", help="Name of the gene ID column (auto-detected if omitted)")
    parser.add_argument("--length-column", help="Name of the gene length column (auto-detected if omitted)")
    parser.add_argument("--sample", help="Sample/column name to convert (uses first sample if omitted)")
    parser.add_argument("--min-count", type=float, default=0,
                        help="Minimum raw count to include a gene (default: 0)")

    args = parser.parse_args()

    print(f"Reading counts from: {args.counts_file}")
    counts, lengths = load_counts(
        args.counts_file,
        gene_column=args.gene_column,
        length_column=args.length_column,
        sample=args.sample,
    )

    # Get lengths from CodonScope if not in the counts file
    if lengths is None:
        if not args.species:
            print("\nError: No length column found in counts file.")
            print("Either provide --length-column or --species to use CodonScope CDS lengths.")
            sys.exit(1)

        print(f"\nLoading CDS lengths from CodonScope ({args.species})...")
        length_map = load_codonscope_lengths(args.species)

        # Match gene IDs to lengths
        matched_lengths = {}
        unmatched = []
        for gene in counts.index:
            if gene in length_map:
                matched_lengths[gene] = length_map[gene]
            elif gene.upper() in length_map:
                matched_lengths[gene] = length_map[gene.upper()]
            else:
                unmatched.append(gene)

        if not matched_lengths:
            print(f"\nError: No genes matched. Check that gene IDs in your counts file "
                  f"match {args.species} gene names/IDs.")
            print(f"  First 5 gene IDs in your file: {list(counts.index[:5])}")
            sys.exit(1)

        n_matched = len(matched_lengths)
        n_total = len(counts)
        print(f"  Matched {n_matched}/{n_total} genes to CDS lengths "
              f"({n_total - n_matched} unmatched, excluded)")

        if unmatched and len(unmatched) <= 20:
            print(f"  Unmatched: {', '.join(unmatched)}")
        elif unmatched:
            print(f"  Unmatched (first 20): {', '.join(unmatched[:20])}...")

        # Keep only matched genes
        matched_genes = list(matched_lengths.keys())
        counts = counts[matched_genes]
        lengths = pd.Series(matched_lengths)[matched_genes]
    else:
        print(f"\n  Using length column from counts file")

    # Filter by minimum count
    if args.min_count > 0:
        mask = counts >= args.min_count
        n_before = len(counts)
        counts = counts[mask]
        lengths = lengths[mask]
        print(f"  Filtered: {n_before} -> {len(counts)} genes (min count >= {args.min_count})")

    # Drop genes with zero or missing length
    valid = (lengths > 0) & lengths.notna() & counts.notna()
    counts = counts[valid]
    lengths = lengths[valid]

    print(f"\nComputing TPM for {len(counts)} genes...")
    tpm = counts_to_tpm(counts, lengths)

    # Write output
    out_df = pd.DataFrame({"gene_id": tpm.index, "tpm": tpm.values})
    out_df = out_df.sort_values("tpm", ascending=False)
    out_df.to_csv(args.output, sep="\t", index=False)

    print(f"\nOutput: {args.output}")
    print(f"  {len(out_df)} genes")
    print(f"  TPM range: {tpm.min():.2f} - {tpm.max():.2f}")
    print(f"  TPM sum: {tpm.sum():.0f}")
    print(f"\nTop 10 genes by TPM:")
    for _, row in out_df.head(10).iterrows():
        print(f"  {row['gene_id']:<20s} {row['tpm']:>12.1f}")


if __name__ == "__main__":
    main()
