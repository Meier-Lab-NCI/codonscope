"""CodonScope HTML Report Generator.

Generates a single self-contained HTML file with inline CSS and
base64-embedded matplotlib plots. Runs all applicable analysis modes
and presents results in a clean, readable format.
"""

import base64
import io
import json
import logging
import time
import zipfile
from datetime import datetime
from html import escape
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from codonscope import __version__

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Region-specific enrichment helper
# ═══════════════════════════════════════════════════════════════════════════════

def _run_region_enrichment(
    species: str,
    gene_ids: list[str],
    n_bootstrap: int = 10_000,
    seed: int | None = None,
    data_dir=None,
    ramp_codons: int = 50,
    k: int = 1,
) -> dict | None:
    """Run enrichment separately on ramp (codons 2-50) and body (51+).

    Args:
        k: K-mer size (1 for monocodon, 2 for dicodon).

    Returns dict with 'ramp' and 'body' DataFrames, or None on failure.
    """
    from codonscope.core.codons import CODON_TABLE, all_possible_kmers
    from codonscope.core.sequences import SequenceDB
    from codonscope.core.statistics import (
        benjamini_hochberg,
        bootstrap_pvalues,
        bootstrap_zscores,
        cohens_d,
        compute_geneset_frequencies,
    )

    db = SequenceDB(species, data_dir=data_dir)
    id_mapping = db.resolve_ids(gene_ids)
    gene_seqs = db.get_sequences(id_mapping)
    all_seqs = db.get_all_sequences()

    if len(gene_seqs) < 5:
        return None

    def _slice_region(seqs: dict, start_codon: int, end_codon: int | None) -> dict:
        """Slice sequences to a codon region (0-indexed). end_codon=None means to end."""
        result = {}
        for name, seq in seqs.items():
            s = start_codon * 3
            e = end_codon * 3 if end_codon is not None else len(seq)
            region = seq[s:e]
            # Must have at least 3 codons
            if len(region) >= 9:
                # Trim to codon boundary
                region = region[:len(region) - len(region) % 3]
                result[name] = region
        return result

    results = {}
    for region_name, start, end in [("ramp", 1, ramp_codons + 1), ("body", ramp_codons + 1, None)]:
        gs_region = _slice_region(gene_seqs, start, end)
        bg_region = _slice_region(all_seqs, start, end)

        if len(gs_region) < 5 or len(bg_region) < 100:
            continue

        # Compute geneset frequencies
        _, gs_mean, kmer_names = compute_geneset_frequencies(gs_region, k=k)

        # Compute background per-gene frequencies
        _, bg_mean_arr, bg_kmer_names = compute_geneset_frequencies(bg_region, k=k)
        # We need the full per-gene matrix for bootstrap
        bg_per_gene, _, _ = compute_geneset_frequencies(bg_region, k=k)

        z_scores, boot_mean, boot_std = bootstrap_zscores(
            gs_mean, bg_per_gene, len(gs_region),
            n_bootstrap=n_bootstrap, seed=seed,
        )
        p_values = bootstrap_pvalues(z_scores)
        adj_p = benjamini_hochberg(p_values)

        # Genome-wide std for cohens_d
        bg_std = bg_per_gene.std(axis=0, ddof=1).astype(np.float64)
        d = cohens_d(gs_mean, bg_mean_arr, bg_std)

        df = pd.DataFrame({
            "kmer": kmer_names,
            "observed_freq": gs_mean,
            "expected_freq": boot_mean,
            "z_score": z_scores,
            "p_value": p_values,
            "adjusted_p": adj_p,
            "cohens_d": d,
        })
        # Add amino acid annotation (only meaningful for monocodons)
        if k == 1:
            df["amino_acid"] = df["kmer"].map(CODON_TABLE)
        df = df.sort_values("z_score", key=np.abs, ascending=False).reset_index(drop=True)
        results[region_name] = df

    return results if results else None


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════

def generate_report(
    species: str,
    gene_ids: list[str],
    output: str | Path = "report.html",
    species2: str | None = None,
    tissue: str | None = None,
    cell_line: str | None = None,
    n_bootstrap: int = 10_000,
    seed: int | None = None,
    data_dir: str | Path | None = None,
    model: str = "bootstrap",
) -> Path:
    """Generate a comprehensive HTML report for a gene list.

    Runs all applicable analysis modes and produces a self-contained
    HTML file with embedded plots.

    Args:
        species: Primary species (e.g. "yeast", "human").
        gene_ids: List of gene identifiers.
        output: Output HTML file path.
        species2: Second species for cross-species comparison (Mode 6).
        tissue: GTEx tissue for human demand analysis (Mode 2).
        cell_line: CCLE cell line for human demand analysis (Mode 2).
        n_bootstrap: Bootstrap iterations for all modes.
        seed: Random seed for reproducibility.
        data_dir: Override default data directory.
        model: Statistical model for Mode 1 ("bootstrap" or "binomial").

    Returns:
        Path to the generated HTML file.
    """
    output = Path(output)
    t0 = time.time()
    sections: list[str] = []

    # Data directory for TSV export: {report_stem}_data/
    data_dir_out = output.parent / f"{output.stem}_data"
    data_dir_out.mkdir(parents=True, exist_ok=True)

    # ── Gene list resolution ──────────────────────────────────────────────
    from codonscope.core.sequences import SequenceDB
    db = SequenceDB(species, data_dir=data_dir)
    id_mapping = db.resolve_ids(gene_ids)
    gene_meta = db.get_gene_metadata()

    # ── CAI Analysis ────────────────────────────────────────────────────
    cai_result = None
    try:
        logger.info("Computing CAI (Codon Adaptation Index)...")
        from codonscope.core.cai import cai_analysis
        cai_result = cai_analysis(species, gene_ids, data_dir=data_dir)
    except Exception as exc:
        logger.warning("CAI computation failed: %s", exc)

    sections.append(_section_gene_summary(
        species, gene_ids, id_mapping, gene_meta, db, cai_result=cai_result,
    ))

    if cai_result is not None:
        cai_result["per_gene"].to_csv(
            data_dir_out / "cai_per_gene.tsv", sep="\t", index=False, float_format="%.6g",
        )
        pd.DataFrame([
            {"codon": c, "weight": w} for c, w in sorted(cai_result["weights"].items())
        ]).to_csv(
            data_dir_out / "cai_weights.tsv", sep="\t", index=False, float_format="%.6g",
        )

    # Get species_dir for wobble coloring in waterfall charts
    species_dir = db.species_dir

    # ── 1. Codon Enrichment Analysis ──────────────────────────────────
    mono_result = None
    mono_matched = None
    binomial_result = None
    mono_region_results = None
    try:
        logger.info("Running Codon Enrichment Analysis...")
        from codonscope.modes.mode1_composition import run_composition

        # 1a. Default background (use bootstrap always for default)
        if model == "binomial":
            # Run bootstrap for default, binomial separately
            mono_result = run_composition(
                species=species, gene_ids=gene_ids, k=1,
                n_bootstrap=n_bootstrap, seed=seed, data_dir=data_dir,
            )
        else:
            mono_result = run_composition(
                species=species, gene_ids=gene_ids, k=1,
                n_bootstrap=n_bootstrap, seed=seed, data_dir=data_dir,
                model=model,
            )
        mono_result["results"].to_csv(
            data_dir_out / "mode1_monocodon.tsv", sep="\t", index=False, float_format="%.6g",
        )

        # 1b. Always run matched background
        logger.info("Running monocodon with matched background...")
        mono_matched = run_composition(
            species=species, gene_ids=gene_ids, k=1,
            background="matched",
            n_bootstrap=n_bootstrap, seed=seed, data_dir=data_dir,
        )
        mono_matched["results"].to_csv(
            data_dir_out / "mode1_monocodon_matched.tsv", sep="\t", index=False, float_format="%.6g",
        )

        # 1c. Binomial GLM (only when model="binomial")
        if model == "binomial":
            logger.info("Running monocodon binomial GLM...")
            binomial_result = run_composition(
                species=species, gene_ids=gene_ids, k=1,
                n_bootstrap=n_bootstrap, seed=seed, data_dir=data_dir,
                model="binomial",
            )

        # 1d/1e. Region-specific enrichment (ramp vs body) for monocodon
        logger.info("Running monocodon region-specific enrichment (ramp vs body)...")
        mono_region_results = _run_region_enrichment(
            species=species, gene_ids=gene_ids,
            n_bootstrap=n_bootstrap, seed=seed, data_dir=data_dir, k=1,
        )
        if mono_region_results:
            for rname, rdf in mono_region_results.items():
                rdf.to_csv(
                    data_dir_out / f"enrichment_{rname}.tsv",
                    sep="\t", index=False, float_format="%.6g",
                )

        sections.append(_section_codon_enrichment(
            mono_default=mono_result,
            mono_matched=mono_matched,
            binomial_result=binomial_result,
            region_results=mono_region_results,
            species=species,
            n_bootstrap=n_bootstrap,
            species_dir=species_dir,
        ))
    except Exception as exc:
        logger.warning("Codon Enrichment Analysis failed: %s", exc)
        sections.append(_section_error("1. Codon Enrichment Analysis", exc))

    # ── 2. Dicodon Enrichment Analysis ────────────────────────────────
    di_result = None
    di_matched = None
    di_region_results = None
    try:
        logger.info("Running Dicodon Enrichment Analysis...")
        from codonscope.modes.mode1_composition import run_composition

        # 2a. Default background
        di_result = run_composition(
            species=species, gene_ids=gene_ids, k=2,
            n_bootstrap=n_bootstrap, seed=seed, data_dir=data_dir,
        )
        di_result["results"].to_csv(
            data_dir_out / "mode1_dicodon.tsv", sep="\t", index=False, float_format="%.6g",
        )

        # 2b. Always run matched background
        logger.info("Running dicodon with matched background...")
        di_matched = run_composition(
            species=species, gene_ids=gene_ids, k=2,
            background="matched",
            n_bootstrap=n_bootstrap, seed=seed, data_dir=data_dir,
        )
        di_matched["results"].to_csv(
            data_dir_out / "mode1_dicodon_matched.tsv", sep="\t", index=False, float_format="%.6g",
        )

        # 2c/2d. Region-specific enrichment for dicodon
        logger.info("Running dicodon region-specific enrichment (ramp vs body)...")
        di_region_results = _run_region_enrichment(
            species=species, gene_ids=gene_ids,
            n_bootstrap=n_bootstrap, seed=seed, data_dir=data_dir, k=2,
        )
        if di_region_results:
            for rname, rdf in di_region_results.items():
                rdf.to_csv(
                    data_dir_out / f"enrichment_dicodon_{rname}.tsv",
                    sep="\t", index=False, float_format="%.6g",
                )

        sections.append(_section_dicodon_enrichment(
            di_default=di_result,
            di_matched=di_matched,
            di_region_results=di_region_results,
            species=species,
            n_bootstrap=n_bootstrap,
        ))
    except Exception as exc:
        logger.warning("Dicodon Enrichment Analysis failed: %s", exc)
        sections.append(_section_error("2. Dicodon Enrichment Analysis", exc))

    # ── 3. AA vs Synonymous Attribution ──────────────────────────────
    dis_result = None
    try:
        logger.info("Running AA vs Synonymous Attribution...")
        from codonscope.modes.mode5_disentangle import run_disentangle
        dis_result = run_disentangle(
            species=species, gene_ids=gene_ids,
            n_bootstrap=n_bootstrap, seed=seed, data_dir=data_dir,
        )
        sections.append(_section_mode5(dis_result, species=species))
        dis_result["attribution"].to_csv(
            data_dir_out / "mode5_attribution.tsv", sep="\t", index=False, float_format="%.6g",
        )
    except Exception as exc:
        logger.warning("AA vs Synonymous Attribution failed: %s", exc)
        sections.append(_section_error("3. AA vs Synonymous Attribution", exc))

    # ── 4. Weighted tRNA Adaptation Index ─────────────────────────────
    prof_result = None
    try:
        logger.info("Running Weighted tRNA Adaptation Index...")
        from codonscope.modes.mode3_profile import run_profile
        prof_result = run_profile(
            species=species, gene_ids=gene_ids, data_dir=data_dir,
        )
        sections.append(_section_mode3(prof_result, species=species))
        prof_result["per_gene_scores"].to_csv(
            data_dir_out / "mode3_profile.tsv", sep="\t", index=False, float_format="%.6g",
        )
        if "ramp_composition" in prof_result:
            prof_result["ramp_composition"].to_csv(
                data_dir_out / "mode3_ramp_composition.tsv", sep="\t", index=False, float_format="%.6g",
            )
        if "body_composition" in prof_result:
            prof_result["body_composition"].to_csv(
                data_dir_out / "mode3_body_composition.tsv", sep="\t", index=False, float_format="%.6g",
            )
    except Exception as exc:
        logger.warning("Weighted tRNA Adaptation Index failed: %s", exc)
        sections.append(_section_error("4. Weighted tRNA Adaptation Index", exc))

    # ── 5. Collision Potential Analysis ───────────────────────────────
    col_result = None
    try:
        logger.info("Running Collision Potential Analysis...")
        from codonscope.modes.mode4_collision import run_collision
        col_result = run_collision(
            species=species, gene_ids=gene_ids, data_dir=data_dir,
        )
        sections.append(_section_mode4(col_result, species=species))
        col_result["per_gene_fs_frac"].to_csv(
            data_dir_out / "mode4_collision.tsv", sep="\t", index=False, float_format="%.6g",
        )
        if "fs_dicodons" in col_result and len(col_result["fs_dicodons"]) > 0:
            col_result["fs_dicodons"].to_csv(
                data_dir_out / "mode4_fs_dicodons.tsv", sep="\t", index=False, float_format="%.6g",
            )
    except Exception as exc:
        logger.warning("Collision Potential Analysis failed: %s", exc)
        sections.append(_section_error("5. Collision Potential Analysis", exc))

    # ── 6. Translational Demand Analysis ─────────────────────────────
    dem_result = None
    try:
        logger.info("Running Translational Demand Analysis...")
        from codonscope.modes.mode2_demand import run_demand
        dem_result = run_demand(
            species=species, gene_ids=gene_ids,
            tissue=tissue, cell_line=cell_line,
            n_bootstrap=n_bootstrap, seed=seed, data_dir=data_dir,
        )
        sections.append(_section_mode2(dem_result, species=species))
        dem_result["results"].to_csv(
            data_dir_out / "mode2_demand.tsv", sep="\t", index=False, float_format="%.6g",
        )
    except Exception as exc:
        logger.warning("Translational Demand Analysis failed: %s", exc)
        sections.append(_section_error("6. Translational Demand Analysis", exc))

    # ── Cross-Species Comparison (optional, unnumbered) ──────────────
    cmp_result = None
    if species2:
        try:
            logger.info("Running Cross-Species Comparison...")
            from codonscope.modes.mode6_compare import run_compare
            cmp_result = run_compare(
                species1=species, species2=species2,
                gene_ids=gene_ids, from_species=species,
                n_bootstrap=n_bootstrap, seed=seed, data_dir=data_dir,
            )
            sections.append(_section_mode6(cmp_result, species=species))
            cmp_result["per_gene"].to_csv(
                data_dir_out / "mode6_compare.tsv", sep="\t", index=False, float_format="%.6g",
            )
        except Exception as exc:
            logger.warning("Cross-Species Comparison failed: %s", exc)
            sections.append(_section_error("Cross-Species Comparison", exc))

    # ── Gene mapping TSV ────────────────────────────────────────────────
    mapped_sys = list(id_mapping.values())
    common_names = db.get_common_names(mapped_sys)
    mapping_records = []
    for input_id in sorted(id_mapping.keys()):
        sys_name = id_mapping[input_id]
        gene_name = common_names.get(sys_name, sys_name)
        mapping_records.append({
            "input_id": input_id,
            "gene_name": gene_name,
            "systematic_name": sys_name,
        })
    if mapping_records:
        pd.DataFrame(mapping_records).to_csv(
            data_dir_out / "gene_mapping.tsv", sep="\t", index=False,
        )

    # ── Additional data exports ──────────────────────────────────────────

    # Per-gene codon frequency matrix
    try:
        from codonscope.core.statistics import compute_geneset_frequencies
        gene_seqs = db.get_sequences(id_mapping)
        per_gene_matrix, gs_mean, kmer_names = compute_geneset_frequencies(gene_seqs, k=1)
        freq_df = pd.DataFrame(per_gene_matrix, columns=kmer_names, index=sorted(gene_seqs.keys()))
        freq_df.index.name = "gene"
        freq_df.to_csv(data_dir_out / "per_gene_codon_freq.tsv", sep="\t", float_format="%.6g")
    except Exception as exc:
        logger.warning("Per-gene frequency export failed: %s", exc)

    # Gene metadata for mapped genes
    try:
        meta_sub = gene_meta[gene_meta["systematic_name"].isin(list(id_mapping.values()))]
        meta_sub.to_csv(data_dir_out / "gene_metadata.tsv", sep="\t", index=False, float_format="%.6g")
    except Exception as exc:
        logger.warning("Gene metadata export failed: %s", exc)

    # Analysis config
    config = {
        "species": species,
        "n_input_genes": len(gene_ids),
        "n_mapped_genes": id_mapping.n_mapped,
        "n_unmapped_genes": id_mapping.n_unmapped,
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "model": model,
        "species2": species2,
        "tissue": tissue,
        "cell_line": cell_line,
        "timestamp": datetime.now().isoformat(),
        "codonscope_version": __version__,
    }
    (data_dir_out / "analysis_config.json").write_text(json.dumps(config, indent=2))

    # ── Executive summary + collapsible sections ─────────────────────────
    # sections[0] is always the Gene List Summary (not collapsible)
    # sections[1:] are the numbered analysis sections + optional cross-species

    exec_summary = _section_executive_summary(
        mono_result=mono_result,
        dis_result=dis_result,
        col_result=col_result,
        cai_result=cai_result,
        species_dir=species_dir,
    )

    # Wrap each analysis section (index 1+) in collapsible <details>
    wrapped_sections = [sections[0]]  # Gene List Summary stays unwrapped
    wrapped_sections.append(exec_summary)  # Executive summary after gene summary

    # Build summary lines for each analysis section
    section_idx = 1  # Start after gene summary
    analysis_sections = sections[1:]

    for i, sec_html in enumerate(analysis_sections):
        summary_line = _compute_summary_line(
            i, sec_html,
            mono_result=mono_result,
            di_result=di_result,
            dis_result=dis_result,
            prof_result=prof_result,
            col_result=col_result,
            dem_result=dem_result,
            cmp_result=cmp_result,
            species2=species2,
            n_analysis_sections=len(analysis_sections),
        )
        wrapped_sections.append(_wrap_collapsible(sec_html, summary_line, open_default=False))

    elapsed = time.time() - t0
    html = _build_html(species, gene_ids, wrapped_sections, elapsed)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    logger.info("Report written to %s (%.1fs)", output, elapsed)
    logger.info("Data tables written to %s/", data_dir_out)

    # ── Create zip with HTML + data TSVs + README + JSON ──────────────
    zip_path = output.parent / f"{output.stem}_results.zip"
    readme_text = _generate_readme(
        species=species, gene_ids=gene_ids, id_mapping=id_mapping,
        n_bootstrap=n_bootstrap, seed=seed, elapsed=elapsed,
        tissue=tissue, cell_line=cell_line, species2=species2,
        data_dir_out=data_dir_out,
    )
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(output, output.name)
        zf.writestr("README.txt", readme_text)
        for tsv_file in sorted(data_dir_out.glob("*.tsv")):
            zf.write(tsv_file, f"data/{tsv_file.name}")
        for json_file in sorted(data_dir_out.glob("*.json")):
            zf.write(json_file, f"data/{json_file.name}")
    logger.info("Results zip written to %s", zip_path)

    return output


def _compute_summary_line(
    section_index: int,
    section_html: str,
    mono_result: dict | None,
    di_result: dict | None,
    dis_result: dict | None,
    prof_result: dict | None,
    col_result: dict | None,
    dem_result: dict | None,
    cmp_result: dict | None,
    species2: str | None,
    n_analysis_sections: int,
) -> str:
    """Compute a one-line summary for each collapsible section header.

    section_index is 0-based within the analysis sections (after gene summary).
    The order is: 0=codon enrichment, 1=dicodon enrichment, 2=attribution,
    3=wtAI, 4=collision, 5=demand, 6=cross-species (if present).
    """
    try:
        if section_index == 0 and mono_result is not None:
            # Codon Enrichment
            df = mono_result["results"]
            sig = df[df["adjusted_p"] < 0.05]
            n_enr = len(sig[sig["z_score"] > 0])
            n_dep = len(sig[sig["z_score"] < 0])
            return f"1. Codon Enrichment Analysis -- {n_enr} enriched, {n_dep} depleted at adj_p < 0.05"

        if section_index == 1 and di_result is not None:
            # Dicodon Enrichment
            df = di_result["results"]
            sig = df[df["adjusted_p"] < 0.05]
            n_enr = len(sig[sig["z_score"] > 0])
            n_dep = len(sig[sig["z_score"] < 0])
            return f"2. Dicodon Enrichment Analysis -- {n_enr} enriched, {n_dep} depleted at adj_p < 0.05"

        if section_index == 2 and dis_result is not None:
            # Attribution
            s = dis_result.get("summary", {})
            n_aa = s.get("n_aa_driven", 0)
            n_syn = s.get("n_synonymous_driven", 0)
            n_both = s.get("n_both", 0)
            total = n_aa + n_syn + n_both
            if total > 0:
                pct_aa = 100.0 * n_aa / total
                pct_syn = 100.0 * n_syn / total
                pct_both = 100.0 * n_both / total
                return f"3. AA vs Synonymous Attribution -- {pct_aa:.0f}% AA-driven, {pct_syn:.0f}% synonymous, {pct_both:.0f}% both"
            return "3. AA vs Synonymous Attribution -- no significant attributions"

        if section_index == 3 and prof_result is not None:
            # wtAI
            scores = prof_result["per_gene_scores"]
            col_name = "wtai" if "wtai" in scores.columns else "tai"
            mean_score = scores[col_name].mean()
            ramp = prof_result["ramp_analysis"]
            delta = ramp["geneset_ramp_delta"]
            return f"4. Weighted tRNA Adaptation Index -- mean wtAI = {mean_score:.3f}, ramp delta = {delta:+.3f}"

        if section_index == 4 and col_result is not None:
            # Collision
            fs_enrich = col_result.get("fs_enrichment", 0)
            chi2_p = col_result.get("chi2_p", 1.0)
            return f"5. Collision Potential Analysis -- FS enrichment = {fs_enrich:.3f}, chi2 p = {chi2_p:.2e}"

        if section_index == 5 and dem_result is not None:
            # Demand
            df = dem_result["results"]
            n_sig = len(df[df["adjusted_p"] < 0.05])
            return f"6. Translational Demand Analysis -- {n_sig} significant codons"

        # Cross-species (index 6, only if species2 is set)
        if section_index == 6 and cmp_result is not None:
            s = cmp_result["summary"]
            mean_r = s["geneset_mean_r"]
            z = s["z_score"]
            return f"Cross-Species Comparison -- mean r = {mean_r:.3f}, Z = {z:.2f}"

    except Exception:
        pass  # Fall through to default

    # Default: extract the section title from the HTML
    import re
    m = re.search(r"<h2>(.*?)</h2>", section_html)
    title = m.group(1) if m else f"Section {section_index + 1}"
    return title


# ═══════════════════════════════════════════════════════════════════════════════
# HTML building
# ═══════════════════════════════════════════════════════════════════════════════

CSS = """
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    max-width: 1100px;
    margin: 0 auto;
    padding: 20px 30px;
    color: #1a1a2e;
    background: #f8f9fa;
    line-height: 1.5;
}
h1 {
    color: #16213e;
    border-bottom: 3px solid #0f3460;
    padding-bottom: 10px;
    margin-top: 0;
}
h2 {
    color: #0f3460;
    border-bottom: 2px solid #e2e8f0;
    padding-bottom: 6px;
    margin-top: 40px;
}
h3 {
    color: #16213e;
    margin-top: 25px;
}
.summary-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 12px;
    margin: 15px 0;
}
.summary-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 14px 18px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
.summary-card .label {
    font-size: 0.82em;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.summary-card .value {
    font-size: 1.5em;
    font-weight: 700;
    color: #0f3460;
    margin-top: 2px;
}
table {
    border-collapse: collapse;
    width: 100%;
    margin: 10px 0;
    font-size: 0.88em;
    background: white;
}
th {
    background: #0f3460;
    color: white;
    padding: 8px 10px;
    text-align: left;
    font-weight: 600;
}
td {
    padding: 6px 10px;
    border-bottom: 1px solid #e2e8f0;
}
tr:nth-child(even) td {
    background: #f8fafc;
}
tr:hover td {
    background: #eef2ff;
}
.sig-pos { color: #b91c1c; font-weight: 600; }
.sig-neg { color: #1d4ed8; font-weight: 600; }
.warn { color: #d97706; }
.ok { color: #16a34a; }
.plot-container {
    text-align: center;
    margin: 20px 0;
}
.plot-container img {
    max-width: 100%;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.08);
}
.section {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 20px 25px;
    margin: 20px 0;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.section h2 {
    margin-top: 0;
    border-bottom: 2px solid #e2e8f0;
}
.error-box {
    background: #fef2f2;
    border: 1px solid #fecaca;
    border-radius: 6px;
    padding: 12px 16px;
    color: #991b1b;
    font-size: 0.9em;
}
.footer {
    text-align: center;
    color: #94a3b8;
    font-size: 0.82em;
    margin-top: 40px;
    padding-top: 15px;
    border-top: 1px solid #e2e8f0;
}
.badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.82em;
    font-weight: 600;
}
.badge-enriched { background: #fee2e2; color: #991b1b; }
.badge-depleted { background: #dbeafe; color: #1e40af; }
.badge-aa { background: #fef3c7; color: #92400e; }
.badge-syn { background: #d1fae5; color: #065f46; }
.badge-both { background: #ede9fe; color: #5b21b6; }
.method-note {
    background: #f0f4ff;
    border-left: 4px solid #6366f1;
    border-radius: 0 6px 6px 0;
    padding: 14px 18px;
    margin: 18px 0;
    font-size: 0.88em;
    color: #334155;
    line-height: 1.6;
}
.method-note p { margin: 6px 0; }
.method-note strong { color: #1e293b; }
details.collapsible-section {
    margin: 20px 0;
}
details.collapsible-section > summary {
    cursor: pointer;
    padding: 12px 18px;
    background: #f0f4ff;
    border: 1px solid #c7d2fe;
    border-radius: 8px;
    font-weight: 600;
    color: #1e3a5f;
    font-size: 0.95em;
    list-style: none;
    user-select: none;
}
details.collapsible-section > summary::-webkit-details-marker { display: none; }
details.collapsible-section > summary::before {
    content: '\25b8 ';
    font-size: 1.1em;
}
details.collapsible-section[open] > summary::before {
    content: '\25be ';
}
details.collapsible-section[open] > summary {
    border-radius: 8px 8px 0 0;
    margin-bottom: 0;
}
.expand-controls {
    text-align: right;
    margin: 10px 0;
}
.expand-controls button {
    background: #e2e8f0;
    border: 1px solid #cbd5e1;
    border-radius: 4px;
    padding: 4px 12px;
    cursor: pointer;
    font-size: 0.82em;
    color: #475569;
    margin-left: 6px;
}
.expand-controls button:hover {
    background: #cbd5e1;
}
.executive-summary {
    background: linear-gradient(135deg, #f0f4ff 0%, #e8f0fe 100%);
    border: 2px solid #6366f1;
    border-radius: 12px;
    padding: 24px 28px;
    margin: 20px 0;
}
.executive-summary h2 {
    color: #4338ca;
    border-bottom: 2px solid #a5b4fc;
    margin-top: 0;
}
.exec-top-codons {
    display: flex;
    gap: 20px;
    margin: 12px 0;
}
.exec-top-codons > div {
    flex: 1;
}
.exec-top-codons table {
    font-size: 0.85em;
}
.group-header {
    color: #0f3460;
    font-size: 1.4em;
    font-weight: 700;
    margin: 35px 0 5px 0;
    padding: 10px 0 6px 0;
    border-bottom: 3px solid #0f3460;
    letter-spacing: 0.3px;
}
"""


def _build_html(
    species: str,
    gene_ids: list[str],
    sections: list[str],
    elapsed: float,
) -> str:
    """Assemble the full HTML document."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    body = "\n".join(sections)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CodonScope Report &mdash; {escape(species)} ({len(gene_ids)} genes)</title>
<style>{CSS}</style>
</head>
<body>
<h1>CodonScope Report</h1>
<p><strong>Species:</strong> {escape(species)} &nbsp;|&nbsp;
   <strong>Input genes:</strong> {len(gene_ids)} &nbsp;|&nbsp;
   <strong>Generated:</strong> {now}</p>
<div class="expand-controls">
  <button onclick="document.querySelectorAll('details.collapsible-section').forEach(d=>d.open=true)">Expand All</button>
  <button onclick="document.querySelectorAll('details.collapsible-section').forEach(d=>d.open=false)">Collapse All</button>
</div>
{body}
<div class="footer">
  <p>CodonScope v{__version__} &mdash; generated in {elapsed:.1f}s &mdash; {now}</p>
  <p><em>For deeper analysis, run individual modes from the command line or use the deep-dive tools.</em></p>
</div>
</body>
</html>
"""


# ═══════════════════════════════════════════════════════════════════════════════
# Section builders
# ═══════════════════════════════════════════════════════════════════════════════

def _section_gene_summary(
    species: str,
    gene_ids: list[str],
    id_mapping,
    gene_meta: pd.DataFrame,
    db,
    cai_result: dict | None = None,
) -> str:
    """Gene list summary with diagnostics and CAI."""
    n_input = len(gene_ids)
    n_mapped = id_mapping.n_mapped
    n_unmapped = id_mapping.n_unmapped

    # Get mapped gene metadata
    mapped_sys = list(id_mapping.values())
    meta_sub = gene_meta[gene_meta["systematic_name"].isin(mapped_sys)]
    mean_len = meta_sub["cds_length"].mean() if len(meta_sub) > 0 else 0
    mean_gc = meta_sub["gc_content"].mean() if len(meta_sub) > 0 else 0

    # Genome stats for comparison
    genome_mean_len = gene_meta["cds_length"].mean()
    genome_mean_gc = gene_meta["gc_content"].mean()

    # Unmapped list
    unmapped_html = ""
    if n_unmapped > 0:
        unmapped_list = id_mapping.unmapped[:20]
        unmapped_str = ", ".join(escape(u) for u in unmapped_list)
        if n_unmapped > 20:
            unmapped_str += f" ... and {n_unmapped - 20} more"
        unmapped_html = f"""
        <p class="warn">Unmapped IDs ({n_unmapped}): {unmapped_str}</p>
        """

    # Gene mapping table: Input ID → Gene Name → Systematic/Ensembl ID
    common_names = db.get_common_names(mapped_sys)
    sp = species.lower()
    is_yeast = sp == "yeast"

    mapping_rows = ""
    for input_id in sorted(id_mapping.keys()):
        sys_name = id_mapping[input_id]
        gene_name = common_names.get(sys_name, sys_name)
        if is_yeast:
            mapping_rows += (
                f"<tr><td>{escape(input_id)}</td>"
                f"<td><strong>{escape(gene_name)}</strong></td>"
                f"<td>{escape(sys_name)}</td></tr>\n"
            )
        else:
            mapping_rows += (
                f"<tr><td>{escape(input_id)}</td>"
                f"<td><strong>{escape(gene_name)}</strong></td>"
                f"<td>{escape(sys_name)}</td></tr>\n"
            )

    if is_yeast:
        col_headers = "<th>Input ID</th><th>Gene Name</th><th>Systematic Name</th>"
    else:
        col_headers = "<th>Input ID</th><th>Gene Name</th><th>Ensembl ID</th>"

    if n_mapped > 20:
        mapping_html = f"""
<details><summary>Gene ID mapping ({n_mapped} genes — click to expand)</summary>
<table>
<tr>{col_headers}</tr>
{mapping_rows}
</table>
</details>"""
    elif n_mapped > 0:
        mapping_html = f"""
<h3>Gene ID Mapping</h3>
<table>
<tr>{col_headers}</tr>
{mapping_rows}
</table>"""
    else:
        mapping_html = ""

    return f"""
<div class="section">
<h2>Gene List Summary</h2>
<div class="summary-grid">
  <div class="summary-card">
    <div class="label">Input IDs</div>
    <div class="value">{n_input}</div>
  </div>
  <div class="summary-card">
    <div class="label">Mapped</div>
    <div class="value {'ok' if n_mapped >= 10 else 'warn'}">{n_mapped}</div>
  </div>
  <div class="summary-card">
    <div class="label">Unmapped</div>
    <div class="value {'warn' if n_unmapped > 0 else 'ok'}">{n_unmapped}</div>
  </div>
  <div class="summary-card">
    <div class="label">Mean CDS length</div>
    <div class="value">{mean_len:.0f} nt</div>
  </div>
  <div class="summary-card">
    <div class="label">Mean GC content</div>
    <div class="value">{mean_gc:.1%}</div>
  </div>
  <div class="summary-card">
    <div class="label">Genome mean length</div>
    <div class="value">{genome_mean_len:.0f} nt</div>
  </div>
</div>
{unmapped_html}
{mapping_html}
{_cai_html(cai_result) if cai_result is not None else ""}
</div>
"""


def _cai_html(cai_result: dict) -> str:
    """Generate HTML for CAI summary within the gene summary section."""
    gs_mean = cai_result["geneset_mean"]
    gs_median = cai_result["geneset_median"]
    gen_mean = cai_result["genome_mean"]
    gen_median = cai_result["genome_median"]
    percentile = cai_result["percentile_rank"]
    mw_p = cai_result["mann_whitney_p"]
    n_ref = cai_result["reference_n_genes"]

    # Box plot: gene set CAI vs genome CAI
    gs_values = cai_result["per_gene"]["cai"].values
    gen_values = cai_result["genome_per_gene"]["cai"].values

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Panel 1: Box plot
    bp = ax1.boxplot(
        [gs_values, gen_values],
        tick_labels=["Gene set", "Genome"],
        patch_artist=True,
        widths=0.5,
    )
    bp["boxes"][0].set_facecolor("#93c5fd")
    bp["boxes"][1].set_facecolor("#d1d5db")
    ax1.set_ylabel("CAI")
    ax1.set_title("Codon Adaptation Index")
    # Annotate p-value
    sig_str = f"p={mw_p:.2e}" if mw_p < 0.001 else f"p={mw_p:.3f}"
    ax1.annotate(
        f"Mann-Whitney {sig_str}",
        xy=(0.5, 0.95), xycoords="axes fraction",
        ha="center", va="top", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
    )

    # Panel 2: Histogram
    bins = np.linspace(0, 1, 40)
    ax2.hist(gen_values, bins=bins, alpha=0.5, label="Genome", color="#94a3b8", density=True)
    ax2.hist(gs_values, bins=bins, alpha=0.7, label="Gene set", color="#2563eb", density=True)
    ax2.axvline(gs_mean, color="#2563eb", ls="--", label=f"Gene set mean={gs_mean:.3f}")
    ax2.axvline(gen_mean, color="#6b7280", ls="--", label=f"Genome mean={gen_mean:.3f}")
    ax2.set_xlabel("CAI")
    ax2.set_ylabel("Density")
    ax2.set_title("CAI Distribution")
    ax2.legend(fontsize=7)

    fig.tight_layout()
    plot_b64 = _fig_to_b64(fig)

    mw_class = "ok" if mw_p < 0.05 else ""

    return f"""
<h3>Codon Adaptation Index (CAI)</h3>
<div class="summary-grid">
  <div class="summary-card">
    <div class="label">Gene Set Mean CAI</div>
    <div class="value">{gs_mean:.3f}</div>
  </div>
  <div class="summary-card">
    <div class="label">Genome Mean CAI</div>
    <div class="value">{gen_mean:.3f}</div>
  </div>
  <div class="summary-card">
    <div class="label">Percentile Rank</div>
    <div class="value">{percentile:.0f}th</div>
  </div>
  <div class="summary-card">
    <div class="label">Mann-Whitney p</div>
    <div class="value {mw_class}">{mw_p:.2e}</div>
  </div>
</div>
<div class="plot-container">{_img_tag(plot_b64)}</div>
<div class="method-note">
<p><strong>CAI</strong> (Sharp &amp; Li 1987) measures how closely a gene&rsquo;s
codon usage matches the most highly-expressed genes in the genome.
Reference weights are computed from the top {n_ref} genes by expression level.
For each synonymous family, the most frequent codon in the reference set gets
weight 1.0; others are scaled proportionally.
The CAI is the geometric mean of these weights across all codons in a gene
(excluding Met and Trp, which have only one codon).
A CAI near 1.0 means the gene uses the same codons as highly-expressed genes;
lower values indicate divergent codon usage.</p>
</div>
"""


def _wrap_collapsible(section_html: str, summary_line: str, open_default: bool = False) -> str:
    """Wrap section HTML in a collapsible <details> element.

    Args:
        section_html: The full section HTML (including its <div class="section"> wrapper).
        summary_line: A one-line metric preview shown in the collapsed header.
        open_default: If True, the section starts expanded.

    Returns:
        HTML string with the section wrapped in <details class="collapsible-section">.
    """
    open_attr = " open" if open_default else ""
    return f'<details class="collapsible-section"{open_attr}><summary>{summary_line}</summary>{section_html}</details>'


def _section_executive_summary(
    mono_result: dict | None,
    dis_result: dict | None,
    col_result: dict | None,
    cai_result: dict | None,
    species_dir: Path | None = None,
) -> str:
    """Executive summary section with key findings from all analyses.

    Provides a quick-glance overview: top enriched/depleted codons,
    attribution breakdown, collision FS ratio, and a waterfall chart.
    """
    parts: list[str] = []
    parts.append('<div class="executive-summary">')
    parts.append('<h2>Executive Summary</h2>')

    # ── Top enriched/depleted codons from mono_result ──
    top_enriched_rows = ""
    top_depleted_rows = ""
    n_enriched = 0
    n_depleted = 0
    if mono_result is not None:
        df = mono_result["results"]
        sig = df[df["adjusted_p"] < 0.05]
        enriched = sig[sig["z_score"] > 0].sort_values("z_score", ascending=False)
        depleted = sig[sig["z_score"] < 0].sort_values("z_score", ascending=True)
        n_enriched = len(enriched)
        n_depleted = len(depleted)

        for _, row in enriched.head(5).iterrows():
            aa_str = str(row.get("amino_acid", "")) if "amino_acid" in row.index else ""
            top_enriched_rows += (
                f'<tr><td><strong>{escape(str(row["kmer"]))}</strong></td>'
                f'<td>{escape(aa_str)}</td>'
                f'<td class="sig-pos">{row["z_score"]:+.2f}</td>'
                f'<td>{row["adjusted_p"]:.2e}</td></tr>\n'
            )

        for _, row in depleted.head(5).iterrows():
            aa_str = str(row.get("amino_acid", "")) if "amino_acid" in row.index else ""
            top_depleted_rows += (
                f'<tr><td><strong>{escape(str(row["kmer"]))}</strong></td>'
                f'<td>{escape(aa_str)}</td>'
                f'<td class="sig-neg">{row["z_score"]:+.2f}</td>'
                f'<td>{row["adjusted_p"]:.2e}</td></tr>\n'
            )

    if top_enriched_rows or top_depleted_rows:
        enriched_table = (
            f'<div><h4>Top Enriched ({n_enriched} total)</h4>'
            f'<table><tr><th>Codon</th><th>AA</th><th>Z</th><th>Adj. p</th></tr>'
            f'{top_enriched_rows}</table></div>'
        ) if top_enriched_rows else '<div><h4>Top Enriched</h4><p><em>None</em></p></div>'
        depleted_table = (
            f'<div><h4>Top Depleted ({n_depleted} total)</h4>'
            f'<table><tr><th>Codon</th><th>AA</th><th>Z</th><th>Adj. p</th></tr>'
            f'{top_depleted_rows}</table></div>'
        ) if top_depleted_rows else '<div><h4>Top Depleted</h4><p><em>None</em></p></div>'
        parts.append(f'<div class="exec-top-codons">{enriched_table}{depleted_table}</div>')
    else:
        parts.append('<p><em>Codon enrichment data not available.</em></p>')

    # ── Attribution breakdown ──
    if dis_result is not None:
        summary = dis_result.get("summary", {})
        n_aa = summary.get("n_aa_driven", 0)
        n_syn = summary.get("n_synonymous_driven", 0)
        n_both = summary.get("n_both", 0)
        total_attr = n_aa + n_syn + n_both
        if total_attr > 0:
            pct_aa = 100.0 * n_aa / total_attr
            pct_syn = 100.0 * n_syn / total_attr
            pct_both = 100.0 * n_both / total_attr
            parts.append(
                f'<p><strong>Attribution:</strong> '
                f'<span class="badge badge-aa">{pct_aa:.0f}% AA-driven ({n_aa})</span> '
                f'<span class="badge badge-syn">{pct_syn:.0f}% Synonymous ({n_syn})</span> '
                f'<span class="badge badge-both">{pct_both:.0f}% Both ({n_both})</span></p>'
            )
        else:
            parts.append('<p><strong>Attribution:</strong> <em>No significant attributions.</em></p>')
    else:
        parts.append('<p><strong>Attribution:</strong> <em>N/A</em></p>')

    # ── Collision FS ratio ──
    if col_result is not None:
        fs_enrich = col_result.get("fs_enrichment", 0)
        chi2_p = col_result.get("chi2_p", 1.0)
        parts.append(
            f'<p><strong>Collision potential:</strong> '
            f'FS enrichment = {fs_enrich:.3f}, '
            f'chi2 p = {chi2_p:.2e}</p>'
        )
    else:
        parts.append('<p><strong>Collision potential:</strong> <em>N/A</em></p>')

    # ── CAI ──
    if cai_result is not None:
        gs_mean = cai_result.get("geneset_mean", 0)
        percentile = cai_result.get("percentile_rank", 0)
        parts.append(
            f'<p><strong>CAI:</strong> '
            f'Gene set mean = {gs_mean:.3f}, '
            f'genome percentile = {percentile:.0f}th</p>'
        )

    # ── Waterfall chart ──
    if mono_result is not None:
        try:
            wf_b64 = _plot_waterfall(
                mono_result["results"],
                "Executive Summary: Ranked Codon Z-scores",
                species_dir=species_dir,
            )
            parts.append(f'<div class="plot-container">{_img_tag(wf_b64)}</div>')
        except Exception:
            pass  # Skip waterfall if plotting fails

    parts.append('</div>')
    return "\n".join(parts)


def _section_region_enrichment(region_results: dict, full_result: dict | None) -> str:
    """Section showing ramp vs body enrichment comparison."""
    from codonscope.core.codons import annotate_kmer

    panels = []
    for region_name in ("ramp", "body"):
        if region_name not in region_results:
            continue
        df = region_results[region_name]
        sig = df[df["adjusted_p"] < 0.05]
        n_sig = len(sig)
        label = "Ramp (codons 2-50)" if region_name == "ramp" else "Body (codons 51+)"

        wf_b64 = _plot_waterfall(df, f"{label}: Ranked Codon Z-scores")
        panels.append(f"""
        <h3>{label}</h3>
        <p>Significant codons (adj_p &lt; 0.05): <strong>{n_sig}</strong></p>
        <div class="plot-container">{_img_tag(wf_b64)}</div>
        """)

    # Comparison table: codons with positional bias (different Z direction in ramp vs body)
    comparison_html = ""
    if "ramp" in region_results and "body" in region_results:
        ramp_df = region_results["ramp"].set_index("kmer")
        body_df = region_results["body"].set_index("kmer")
        common = set(ramp_df.index) & set(body_df.index)

        biased = []
        for kmer in sorted(common):
            rz = ramp_df.loc[kmer, "z_score"]
            bz = body_df.loc[kmer, "z_score"]
            # Positionally biased: significant in at least one region AND different direction
            rp = ramp_df.loc[kmer, "adjusted_p"]
            bp = body_df.loc[kmer, "adjusted_p"]
            if (rp < 0.05 or bp < 0.05) and abs(rz - bz) > 2.0:
                biased.append({
                    "kmer": kmer,
                    "label": annotate_kmer(kmer, 1),
                    "ramp_z": rz,
                    "body_z": bz,
                    "delta": rz - bz,
                    "ramp_p": rp,
                    "body_p": bp,
                })

        if biased:
            biased.sort(key=lambda x: abs(x["delta"]), reverse=True)
            rows = ""
            for b in biased[:20]:
                rz_cls = "sig-pos" if b["ramp_z"] > 0 else "sig-neg"
                bz_cls = "sig-pos" if b["body_z"] > 0 else "sig-neg"
                rows += f"""<tr>
                  <td><strong>{b['label']}</strong></td>
                  <td class="{rz_cls}">{b['ramp_z']:+.2f}</td>
                  <td>{b['ramp_p']:.2e}</td>
                  <td class="{bz_cls}">{b['body_z']:+.2f}</td>
                  <td>{b['body_p']:.2e}</td>
                  <td><strong>{b['delta']:+.2f}</strong></td>
                </tr>"""
            comparison_html = f"""
            <h3>Positionally Biased Codons</h3>
            <p>Codons with large ramp-vs-body Z-score difference (&Delta; &gt; 2.0, significant in at least one region):</p>
            <table>
            <tr><th>Codon</th><th>Ramp Z</th><th>Ramp p</th><th>Body Z</th><th>Body p</th><th>&Delta;Z</th></tr>
            {rows}
            </table>"""

    method_note = """
<div class="method-note">
<p><strong>Region-specific enrichment.</strong> The CDS is split into
<strong>ramp</strong> (codons 2&ndash;50, the translation initiation zone)
and <strong>body</strong> (codon 51 onward). Enrichment is computed separately
for each region against the same region in all genome genes.
Codons that are enriched in the ramp but not the body (or vice versa)
may reflect position-specific translational selection &mdash; for example,
slow codons enriched in the ramp to pace early elongation.</p>
</div>"""

    return f"""
<div class="section">
<h2>Region-Specific Codon Enrichment</h2>
{"".join(panels)}
{comparison_html}
{method_note}
</div>
"""


def _section_codon_enrichment(
    mono_default: dict | None,
    mono_matched: dict | None,
    binomial_result: dict | None,
    region_results: dict | None,
    species: str,
    n_bootstrap: int,
    species_dir: Path | None = None,
) -> str:
    """Section 1: Consolidated Codon Enrichment Analysis.

    Includes default, matched, binomial (if any), ramp, body, positionally
    biased codons table, and waterfall charts for full CDS / ramp / body.
    """
    from codonscope.core.codons import CODON_TABLE, annotate_kmer

    parts: list[str] = []
    parts.append('<div class="section">')
    parts.append('<h2>1. Codon Enrichment Analysis</h2>')

    # ── 1a. Full CDS — default background ────────────────────────────────
    if mono_default is not None:
        df = mono_default["results"]
        n_genes = mono_default["n_genes"]
        diag = mono_default["diagnostics"]
        sig = df[df["adjusted_p"] < 0.05]
        enriched = sig[sig["z_score"] > 0].sort_values("z_score", ascending=False)
        depleted = sig[sig["z_score"] < 0].sort_values("z_score", ascending=True)

        # Diagnostics
        diag_html = ""
        warns = []
        if diag.get("length_warning"):
            warns.append(f"CDS length differs from background (KS p={diag['length_p']:.2e})")
        if diag.get("gc_warning"):
            warns.append(f"GC content differs from background (KS p={diag['gc_p']:.2e})")
        for w in diag.get("power_warnings", []):
            warns.append(w)
        if warns:
            items = "".join(f"<li>{escape(w)}</li>" for w in warns)
            diag_html = f'<div class="error-box"><strong>Diagnostics:</strong><ul style="margin:4px 0">{items}</ul></div>'

        parts.append(f'<h3>1a. Full CDS &mdash; Default Background</h3>')
        parts.append(f'<p>Genes analyzed: <strong>{n_genes}</strong> &nbsp;|&nbsp; '
                     f'Significant codons (adj_p &lt; 0.05): <strong>{len(sig)}</strong></p>')
        parts.append(diag_html)

        # Volcano plot
        plot_b64 = _plot_volcano(df, f"Codon Enrichment (n={n_genes} genes)")
        parts.append(f'<div class="plot-container">{_img_tag(plot_b64)}</div>')

        # Waterfall — full CDS
        wf_b64 = _plot_waterfall(df, f"Ranked Codon Z-scores (n={n_genes} genes)", species_dir=species_dir)
        parts.append(f'<div class="plot-container">{_img_tag(wf_b64)}</div>')

        parts.append(f'<h4>Top Enriched ({len(enriched)} total)</h4>')
        parts.append(_results_table(enriched.head(30), k=1))
        parts.append(f'<h4>Top Depleted ({len(depleted)} total)</h4>')
        parts.append(_results_table(depleted.head(30), k=1))

    # ── 1b. Full CDS — length+GC matched background ─────────────────────
    if mono_matched is not None:
        df_m = mono_matched["results"]
        n_genes_m = mono_matched["n_genes"]
        sig_m = df_m[df_m["adjusted_p"] < 0.05]
        enriched_m = sig_m[sig_m["z_score"] > 0].sort_values("z_score", ascending=False)
        depleted_m = sig_m[sig_m["z_score"] < 0].sort_values("z_score", ascending=True)

        parts.append(f'<h3>1b. Full CDS &mdash; Length+GC Matched Background</h3>')
        parts.append(
            '<div class="method-note"><p><strong>Matched background.</strong> '
            'This re-analysis uses a background of genes matched for '
            'CDS length and GC content, to control for nucleotide composition bias. '
            'Codons that remain significant here are more likely to reflect genuine '
            'translational selection rather than GC bias.</p></div>'
        )
        parts.append(f'<p>Significant codons (adj_p &lt; 0.05): <strong>{len(sig_m)}</strong></p>')
        parts.append(f'<h4>Top Enriched ({len(enriched_m)} total)</h4>')
        parts.append(_results_table(enriched_m.head(30), k=1))
        parts.append(f'<h4>Top Depleted ({len(depleted_m)} total)</h4>')
        parts.append(_results_table(depleted_m.head(30), k=1))

    # ── 1c. Full CDS — binomial GLM (only when model="binomial") ────────
    if binomial_result is not None:
        df_b = binomial_result["results"]
        n_genes_b = binomial_result["n_genes"]
        sig_b = df_b[df_b["adjusted_p"] < 0.05]
        enriched_b = sig_b[sig_b["z_score"] > 0].sort_values("z_score", ascending=False)
        depleted_b = sig_b[sig_b["z_score"] < 0].sort_values("z_score", ascending=True)

        parts.append(f'<h3>1c. Full CDS &mdash; Binomial GLM (GC3-corrected)</h3>')
        parts.append(
            '<div class="method-note"><p><strong>GC3-corrected binomial GLM</strong> '
            '(Doyle, Nanda &amp; Begley 2025). For each codon, we fit: '
            '<code>logit(p) = &beta;<sub>0</sub> + &beta;<sub>1</sub>&middot;gene_set '
            '+ &beta;<sub>2</sub>&middot;GC3</code>. '
            'The Z-score is the Wald statistic for &beta;<sub>1</sub> after controlling '
            'for GC3 composition bias.</p></div>'
        )
        parts.append(f'<p>Significant codons (adj_p &lt; 0.05): <strong>{len(sig_b)}</strong></p>')
        parts.append(f'<h4>Top Enriched ({len(enriched_b)} total)</h4>')
        parts.append(_results_table(enriched_b.head(30), k=1))
        parts.append(f'<h4>Top Depleted ({len(depleted_b)} total)</h4>')
        parts.append(_results_table(depleted_b.head(30), k=1))

    # ── 1d/1e. Ramp and Body ─────────────────────────────────────────────
    if region_results:
        for region_name, sub_label, sub_num in [
            ("ramp", "Ramp (codons 2&ndash;50)", "1d"),
            ("body", "Body (codons 51+)", "1e"),
        ]:
            if region_name not in region_results:
                continue
            rdf = region_results[region_name]
            sig_r = rdf[rdf["adjusted_p"] < 0.05]

            parts.append(f'<h3>{sub_num}. {sub_label} &mdash; Default Background</h3>')
            parts.append(f'<p>Significant codons (adj_p &lt; 0.05): <strong>{len(sig_r)}</strong></p>')

            # Waterfall chart for each region
            wf_r_b64 = _plot_waterfall(rdf, f"{sub_label}: Ranked Codon Z-scores", species_dir=species_dir)
            parts.append(f'<div class="plot-container">{_img_tag(wf_r_b64)}</div>')

        # Positionally biased codons table (ramp vs body comparison)
        if "ramp" in region_results and "body" in region_results:
            ramp_df = region_results["ramp"].set_index("kmer")
            body_df = region_results["body"].set_index("kmer")
            common = set(ramp_df.index) & set(body_df.index)

            biased = []
            for kmer in sorted(common):
                rz = ramp_df.loc[kmer, "z_score"]
                bz = body_df.loc[kmer, "z_score"]
                rp = ramp_df.loc[kmer, "adjusted_p"]
                bp = body_df.loc[kmer, "adjusted_p"]
                if (rp < 0.05 or bp < 0.05) and abs(rz - bz) > 2.0:
                    biased.append({
                        "kmer": kmer,
                        "label": annotate_kmer(kmer, 1),
                        "ramp_z": rz,
                        "body_z": bz,
                        "delta": rz - bz,
                        "ramp_p": rp,
                        "body_p": bp,
                    })

            if biased:
                biased.sort(key=lambda x: abs(x["delta"]), reverse=True)
                rows = ""
                for b in biased[:20]:
                    rz_cls = "sig-pos" if b["ramp_z"] > 0 else "sig-neg"
                    bz_cls = "sig-pos" if b["body_z"] > 0 else "sig-neg"
                    rows += f"""<tr>
                      <td><strong>{b['label']}</strong></td>
                      <td class="{rz_cls}">{b['ramp_z']:+.2f}</td>
                      <td>{b['ramp_p']:.2e}</td>
                      <td class="{bz_cls}">{b['body_z']:+.2f}</td>
                      <td>{b['body_p']:.2e}</td>
                      <td><strong>{b['delta']:+.2f}</strong></td>
                    </tr>"""
                parts.append('<h3>Positionally Biased Codons</h3>')
                parts.append('<p>Codons with large ramp-vs-body Z-score difference '
                             '(&Delta; &gt; 2.0, significant in at least one region):</p>')
                parts.append('<table>')
                parts.append('<tr><th>Codon</th><th>Ramp Z</th><th>Ramp p</th>'
                             '<th>Body Z</th><th>Body p</th><th>&Delta;Z</th></tr>')
                parts.append(rows)
                parts.append('</table>')

    # ── Explainer text ────────────────────────────────────────────────────
    n_genes_str = str(mono_default["n_genes"]) if mono_default else "N/A"

    sp = species.lower()
    if sp == "yeast":
        db_desc = "6,685 verified ORFs from the Saccharomyces Genome Database (SGD)"
    elif sp == "human":
        db_desc = "19,229 protein-coding genes from NCBI MANE Select v1.5"
    else:
        db_desc = f"protein-coding genes for {escape(species)}"

    parts.append(f"""
<div class="method-note">
<p><strong>What this is.</strong> We took your {n_genes_str} genes and compared their
single-codon frequencies against all {db_desc}.</p>
<p><strong>Three analysis variants.</strong>
<strong>(a) Default:</strong> compares against all genome genes.
<strong>(b) Matched:</strong> controls for CDS length and GC content &mdash; codons
that remain significant here are more likely genuine translational selection.
<strong>(c) Binomial GLM</strong> (when selected): a GC3-corrected logistic model
that directly regresses out third-position GC bias.</p>
<p><strong>Ramp vs body.</strong> The CDS is split into
<strong>ramp</strong> (codons 2&ndash;50, the translation initiation zone) and
<strong>body</strong> (codon 51 onward). Enrichment is computed separately for each
region. Codons enriched in the ramp but not the body may reflect position-specific
translational selection (e.g., slow codons pacing early elongation).</p>
<p><strong>Waterfall charts.</strong> Codons colored in <strong style="color:#f59e0b">amber</strong>
are wobble-decoded (G:U or I:C at the wobble position); gray codons are Watson-Crick decoded.
Wobble decoding is slower, so wobble-decoded codons that are enriched or depleted
may reflect tRNA adaptation.</p>
<p><strong>Reading the table.</strong>
<strong>Adjusted p-value</strong> is Benjamini-Hochberg FDR-corrected.
<strong>Cohen&rsquo;s d</strong> is the effect size: under 0.2 is tiny, 0.5 is
medium, above 0.8 is large.</p>
</div>""")

    parts.append('<p><em>Full results available in the data/ directory.</em></p>')
    parts.append('</div>')
    return "\n".join(parts)


def _section_dicodon_enrichment(
    di_default: dict | None,
    di_matched: dict | None,
    di_region_results: dict | None,
    species: str,
    n_bootstrap: int,
) -> str:
    """Section 2: Consolidated Dicodon Enrichment Analysis.

    Includes default, matched, ramp, body. No waterfall charts.
    Binomial GLM not available for dicodons.
    """
    parts: list[str] = []
    parts.append('<div class="section">')
    parts.append('<h2>2. Dicodon Enrichment Analysis</h2>')

    # ── 2a. Full CDS — default background ────────────────────────────────
    if di_default is not None:
        df = di_default["results"]
        n_genes = di_default["n_genes"]
        diag = di_default["diagnostics"]
        sig = df[df["adjusted_p"] < 0.05]
        enriched = sig[sig["z_score"] > 0].sort_values("z_score", ascending=False)
        depleted = sig[sig["z_score"] < 0].sort_values("z_score", ascending=True)

        # Diagnostics
        diag_html = ""
        warns = []
        if diag.get("length_warning"):
            warns.append(f"CDS length differs from background (KS p={diag['length_p']:.2e})")
        if diag.get("gc_warning"):
            warns.append(f"GC content differs from background (KS p={diag['gc_p']:.2e})")
        for w in diag.get("power_warnings", []):
            warns.append(w)
        if warns:
            items = "".join(f"<li>{escape(w)}</li>" for w in warns)
            diag_html = f'<div class="error-box"><strong>Diagnostics:</strong><ul style="margin:4px 0">{items}</ul></div>'

        parts.append(f'<h3>2a. Full CDS &mdash; Default Background</h3>')
        parts.append(f'<p>Genes analyzed: <strong>{n_genes}</strong> &nbsp;|&nbsp; '
                     f'Significant dicodons (adj_p &lt; 0.05): <strong>{len(sig)}</strong></p>')
        parts.append(diag_html)

        # Volcano plot
        plot_b64 = _plot_volcano(df, f"Dicodon Enrichment (n={n_genes} genes)")
        parts.append(f'<div class="plot-container">{_img_tag(plot_b64)}</div>')

        parts.append(f'<h4>Top Enriched ({len(enriched)} total)</h4>')
        parts.append(_results_table(enriched.head(30), k=2))
        parts.append(f'<h4>Top Depleted ({len(depleted)} total)</h4>')
        parts.append(_results_table(depleted.head(30), k=2))

    # ── 2b. Full CDS — length+GC matched background ─────────────────────
    if di_matched is not None:
        df_m = di_matched["results"]
        sig_m = df_m[df_m["adjusted_p"] < 0.05]
        enriched_m = sig_m[sig_m["z_score"] > 0].sort_values("z_score", ascending=False)
        depleted_m = sig_m[sig_m["z_score"] < 0].sort_values("z_score", ascending=True)

        parts.append(f'<h3>2b. Full CDS &mdash; Length+GC Matched Background</h3>')
        parts.append(
            '<div class="method-note"><p><strong>Matched background.</strong> '
            'Dicodon analysis re-run using a background of genes matched for '
            'CDS length and GC content.</p></div>'
        )
        parts.append(f'<p>Significant dicodons (adj_p &lt; 0.05): <strong>{len(sig_m)}</strong></p>')
        parts.append(f'<h4>Top Enriched ({len(enriched_m)} total)</h4>')
        parts.append(_results_table(enriched_m.head(30), k=2))
        parts.append(f'<h4>Top Depleted ({len(depleted_m)} total)</h4>')
        parts.append(_results_table(depleted_m.head(30), k=2))

    # ── 2c/2d. Ramp and Body ─────────────────────────────────────────────
    if di_region_results:
        for region_name, sub_label, sub_num in [
            ("ramp", "Ramp (codons 2&ndash;50)", "2c"),
            ("body", "Body (codons 51+)", "2d"),
        ]:
            if region_name not in di_region_results:
                continue
            rdf = di_region_results[region_name]
            sig_r = rdf[rdf["adjusted_p"] < 0.05]
            enriched_r = sig_r[sig_r["z_score"] > 0].sort_values("z_score", ascending=False)
            depleted_r = sig_r[sig_r["z_score"] < 0].sort_values("z_score", ascending=True)

            parts.append(f'<h3>{sub_num}. {sub_label} &mdash; Default Background</h3>')
            parts.append(f'<p>Significant dicodons (adj_p &lt; 0.05): <strong>{len(sig_r)}</strong></p>')
            parts.append(f'<h4>Top Enriched ({len(enriched_r)} total)</h4>')
            parts.append(_results_table(enriched_r.head(20), k=2))
            parts.append(f'<h4>Top Depleted ({len(depleted_r)} total)</h4>')
            parts.append(_results_table(depleted_r.head(20), k=2))

    # ── Explainer text ────────────────────────────────────────────────────
    n_genes_str = str(di_default["n_genes"]) if di_default else "N/A"
    parts.append(f"""
<div class="method-note">
<p><strong>What this is.</strong> Same approach as the monocodon analysis above, but
for all 3,721 possible pairs of adjacent sense codons (dicodons), counted with a
sliding window (positions 1-2, 2-3, 3-4, etc.).</p>
<p><strong>Why dicodons matter.</strong> Dicodon biases can reveal things monocodons
miss, like ribosome stalling at specific codon pairs or avoidance of certain
dinucleotides at the codon junction. Fair warning: 3,721 tests means you need strong
effects to survive multiple-testing correction.</p>
<p><strong>Binomial GLM.</strong> Not available for dicodons &mdash; the GLM approach
models within-family synonymous proportions, which doesn&rsquo;t have a natural analogue
for codon pairs spanning two amino acid families.</p>
<p><strong>No waterfall charts.</strong> With 3,721 dicodons, a ranked bar chart
would be unreadable. Use the volcano plot and tables instead.</p>
</div>""")

    parts.append('<p><em>Full results available in the data/ directory.</em></p>')
    parts.append('</div>')
    return "\n".join(parts)


def _section_mode1(result: dict, k: int, species: str = "", n_bootstrap: int = 10_000, background_label: str = "") -> str:
    """Mode 1: Composition analysis section."""
    kname = {1: "Monocodon", 2: "Dicodon", 3: "Tricodon"}[k]
    df = result["results"]
    n_genes = result["n_genes"]
    diag = result["diagnostics"]

    sig = df[df["adjusted_p"] < 0.05]
    n_sig = len(sig)
    enriched = sig[sig["z_score"] > 0].sort_values("z_score", ascending=False)
    depleted = sig[sig["z_score"] < 0].sort_values("z_score", ascending=True)

    # Diagnostics
    diag_html = ""
    warns = []
    if diag.get("length_warning"):
        warns.append(f"CDS length differs from background (KS p={diag['length_p']:.2e})")
    if diag.get("gc_warning"):
        warns.append(f"GC content differs from background (KS p={diag['gc_p']:.2e})")
    for w in diag.get("power_warnings", []):
        warns.append(w)
    if warns:
        items = "".join(f"<li>{escape(w)}</li>" for w in warns)
        diag_html = f'<div class="error-box"><strong>Diagnostics:</strong><ul style="margin:4px 0">{items}</ul></div>'

    # Volcano plot
    section_title = {1: "Codon Enrichment Analysis", 2: "Dicodon Enrichment Analysis", 3: "Tricodon Enrichment Analysis"}[k]
    plot_b64 = _plot_volcano(df, f"{section_title} (n={n_genes} genes)")

    # Waterfall bar chart
    waterfall_html = ""
    if k == 1:
        wf_b64 = _plot_waterfall(df, f"Ranked Codon Z-scores (n={n_genes} genes)")
        waterfall_html = f'<div class="plot-container">{_img_tag(wf_b64)}</div>'
    elif k == 2:
        wf_b64 = _plot_waterfall_dicodon(df, f"Ranked Dicodon Z-scores (n={n_genes} genes)")
        waterfall_html = f'<div class="plot-container">{_img_tag(wf_b64)}</div>'

    # Top tables
    top_enriched = _results_table(enriched.head(30), k)
    top_depleted = _results_table(depleted.head(30), k)

    # Species-specific description
    sp = species.lower()
    if sp == "yeast":
        db_desc = "6,685 verified ORFs from the Saccharomyces Genome Database (SGD)"
        isoform_note = "Yeast keeps things simple &mdash; one ORF per gene, no isoform drama."
    elif sp == "human":
        db_desc = "19,229 protein-coding genes from NCBI MANE Select v1.5"
        isoform_note = (
            "Each gene is represented by its <strong>MANE Select</strong> transcript &mdash; "
            "the one canonical isoform that NCBI and Ensembl actually agree on. "
            "No double-counting from alternative splicing."
        )
    else:
        db_desc = f"protein-coding genes for {escape(species)}"
        isoform_note = "One canonical transcript per gene."

    if k == 1:
        kmer_desc = (
            "each of the 61 sense codons (stop codons excluded, obviously)"
        )
        what_it_means = (
            "If your gene set uses AAG (Lys) more than the genome average, "
            "that codon shows up as enriched. Could be translational selection, "
            "could be amino acid composition, could be GC bias &mdash; "
            "the Attribution analysis below will help sort that out."
        )
    elif k == 2:
        kmer_desc = (
            "all 3,721 possible pairs of adjacent sense codons (dicodons). "
            "These are counted with a sliding window &mdash; codons at positions "
            "1-2, 2-3, 3-4, etc."
        )
        what_it_means = (
            "Dicodon biases can reveal things monocodons miss, like "
            "ribosome stalling at specific codon pairs or avoidance of "
            "certain dinucleotides at the codon junction. "
            "Fair warning: 3,721 tests means you need strong effects to survive "
            "multiple-testing correction."
        )
    else:
        kmer_desc = "all 226,981 sense-codon triplets (tricodons)"
        what_it_means = (
            "Tricodon space is enormous, so statistical power is limited. "
            "These use analytic standard errors rather than bootstrap."
        )

    is_binomial = result.get("model") == "binomial"

    if is_binomial:
        method_note = f"""
<div class="method-note">
<p><strong>What this is.</strong> We took your {n_genes} genes and compared their
{kmer_desc} frequencies against all {db_desc}. {isoform_note}</p>
<p><strong>GC3-corrected binomial GLM</strong> (Doyle, Nanda &amp; Begley 2025).
For each codon, we fit a binomial generalised linear model:
<code>logit(p) = &beta;<sub>0</sub> + &beta;<sub>1</sub>&middot;gene_set + &beta;<sub>2</sub>&middot;GC3</code>,
where <em>p</em> is the proportion of that codon within its synonymous amino acid family,
<em>gene_set</em> is a 0/1 indicator for membership in your gene list, and <em>GC3</em>
is the GC content at third codon positions. The <strong>Z-score</strong> is the Wald
statistic for &beta;<sub>1</sub> &mdash; the gene-set effect after controlling for
GC3 composition bias. This is more conservative than bootstrap for codons whose
enrichment is driven by GC content rather than genuine translational selection.</p>
<p><strong>GC3 coefficient.</strong> The <code>gc3_beta</code> column shows how
strongly each codon&rsquo;s usage is predicted by third-position GC content.
Large positive values indicate G/C-ending codons; large negative values indicate
A/T-ending codons. If the bootstrap Z-score was large but the GLM Z-score is small,
GC bias was likely the main driver.</p>
<p>{what_it_means}</p>
</div>"""
    else:
        method_note = f"""
<div class="method-note">
<p><strong>What this is.</strong> We took your {n_genes} genes and compared their
{kmer_desc} frequencies against all {db_desc}. {isoform_note}</p>
<p><strong>How it works.</strong> For each gene, we compute {kname.lower()} frequencies
(proportions, so a 300-codon gene and a 3,000-codon gene contribute equally &mdash; no
length bias). We average those per-gene frequencies across your gene set, then ask:
is this average unusual? To find out, we bootstrap-resample {n_bootstrap:,} random
gene sets of the same size from the genome and build a null distribution.
The <strong>Z-score</strong> is how many standard errors your gene set&rsquo;s mean
is from the bootstrap mean. Positive = enriched, negative = depleted.</p>
<p><strong>Reading the table.</strong>
<strong>Adjusted p-value</strong> is Benjamini-Hochberg FDR-corrected across all
{kname.lower()}s tested &mdash; it controls the expected proportion of false
discoveries, not just the chance of one.
<strong>Cohen&rsquo;s d</strong> is the effect size (observed &minus; expected,
divided by the genome standard deviation). It tells you how big the difference
is in practical terms: under 0.2 is tiny, 0.5 is medium, above 0.8 is large.
A codon can be statistically significant but biologically trivial if d is small.</p>
<p>{what_it_means}</p>
<p><strong>GC content heads-up.</strong> If your gene set has different GC content
than the genome (the diagnostics above will flag this), some codon biases may just
reflect nucleotide composition rather than selection.
Re-run with <code>--background matched</code> to compare against length- and
GC-matched genes if you want to be thorough about it.</p>
</div>"""

    bg_suffix = f" (length+GC matched)" if background_label == "matched" else ""
    model_suffix = " (binomial GLM, GC3-corrected)" if is_binomial else ""
    matched_note = ""
    if background_label == "matched":
        matched_note = (
            '<div class="method-note"><p><strong>Matched background.</strong> '
            'This is a re-analysis using a background of genes matched for '
            'CDS length and GC content, to control for nucleotide composition bias. '
            'Compare these results to the all-genome analysis above &mdash; codons '
            'that remain significant here are more likely to reflect genuine '
            'translational selection rather than GC bias.</p></div>'
        )

    return f"""
<div class="section">
<h2>{section_title}{bg_suffix}{model_suffix}</h2>
<p>Genes analyzed: <strong>{n_genes}</strong> &nbsp;|&nbsp;
   Significant {kname.lower()}s (adj_p &lt; 0.05): <strong>{n_sig}</strong></p>
{diag_html}
{matched_note}
<div class="plot-container">{_img_tag(plot_b64)}</div>
{waterfall_html}
<h3>Top Enriched ({len(enriched)} total)</h3>
{top_enriched}
<h3>Top Depleted ({len(depleted)} total)</h3>
{top_depleted}
<p><em>Full results available in the data/ directory.</em></p>
{method_note}
</div>
"""


def _section_mode5(result: dict, species: str = "") -> str:
    """Mode 5: Disentanglement section."""
    summary = result["summary"]
    attr_df = result["attribution"]
    rscu_df = result["rscu_results"]
    aa_df = result["aa_results"]
    n_genes = result["n_genes"]

    # Summary text
    s = summary
    summary_text = escape(s["summary_text"])

    # Attribution pie chart
    pie_b64 = _plot_attribution_pie(s)

    # Attribution table (all significant codons)
    sig_attr = attr_df[attr_df["attribution"] != "None"].sort_values(
        "rscu_z_score", key=abs, ascending=False
    )

    rows = ""
    for _, row in sig_attr.iterrows():
        badge_cls = {
            "AA-driven": "badge-aa",
            "Synonymous-driven": "badge-syn",
            "Both": "badge-both",
        }.get(row["attribution"], "")
        z_cls = "sig-pos" if row["rscu_z_score"] > 0 else "sig-neg"
        rows += f"""<tr>
          <td><strong>{escape(str(row['codon']))}</strong></td>
          <td>{escape(str(row['amino_acid']))}</td>
          <td class="{z_cls}">{row.get('aa_z_score', 0):+.2f}</td>
          <td class="{z_cls}">{row['rscu_z_score']:+.2f}</td>
          <td>{row.get('aa_adj_p', 1):.2e}</td>
          <td>{row.get('rscu_adj_p', 1):.2e}</td>
          <td><span class="badge {badge_cls}">{escape(str(row['attribution']))}</span></td>
        </tr>"""

    # Synonymous drivers
    syn_df = result.get("synonymous_drivers", pd.DataFrame())
    driver_html = ""
    if len(syn_df) > 0:
        sig_syn = syn_df[syn_df["driver"] != "not_applicable"]
        if len(sig_syn) > 0:
            driver_rows = ""
            for _, row in sig_syn.iterrows():
                driver_rows += f"""<tr>
                  <td>{escape(str(row['codon']))}</td>
                  <td>{escape(str(row['amino_acid']))}</td>
                  <td>{row['rscu_z_score']:+.2f}</td>
                  <td>{escape(str(row['driver']))}</td>
                </tr>"""
            driver_html = f"""
            <h3>Synonymous Drivers</h3>
            <table>
            <tr><th>Codon</th><th>AA</th><th>RSCU Z</th><th>Driver</th></tr>
            {driver_rows}
            </table>"""

    method_note = """
<div class="method-note">
<p><strong>The problem.</strong> The enrichment analysis told you <em>which</em> codons are
biased, but not <em>why</em>. If your gene set is enriched for GCT (Ala),
is that because the proteins literally need more alanine (amino acid composition),
or because among the four Ala codons the cell is picking GCT specifically
(synonymous preference)? These are very different biological stories.</p>
<p><strong>Two-layer decomposition.</strong> We split it apart:
<strong>Layer 1</strong> compares amino acid frequencies (20 AAs) between your
gene set and the genome.
<strong>Layer 2</strong> computes RSCU (Relative Synonymous Codon Usage) within
each amino acid family. RSCU = 1.0 means a codon is used equally with its
synonyms; RSCU &gt; 1 means preferred, &lt; 1 means avoided.
Both layers are tested with bootstrap Z-scores, same as the enrichment analysis.</p>
<p><strong>Attribution.</strong> Each codon gets classified:
<strong>AA-driven</strong> = the amino acid itself is enriched/depleted, but
synonymous usage is normal (it&rsquo;s about protein composition, not codon choice).
<strong>Synonymous-driven</strong> = the amino acid is fine, but this particular
synonym is over- or under-used (translational selection, tRNA adaptation, GC bias).
<strong>Both</strong> = protein composition <em>and</em> codon choice are unusual.</p>
<p><strong>Synonymous drivers.</strong> For codons with significant RSCU deviations,
we make a rough call on the cause:
<em>tRNA supply</em> (high gene copy number for the matching tRNA),
<em>wobble avoidance</em> (Watson-Crick decoded codons preferred over wobble-decoded ones),
or <em>GC3 bias</em> (systematic preference for G/C-ending codons).
These are heuristic labels, not gospel &mdash; the real world is messier than
three neat categories.</p>
</div>"""

    return f"""
<div class="section">
<h2>3. AA vs Synonymous Attribution</h2>
<p>Genes analyzed: <strong>{n_genes}</strong></p>
<p>{summary_text}</p>
<div class="plot-container">{_img_tag(pie_b64)}</div>
<h3>Attribution Table</h3>
<table>
<tr><th>Codon</th><th>AA</th><th>AA Z</th><th>RSCU Z</th><th>AA adj_p</th><th>RSCU adj_p</th><th>Attribution</th></tr>
{rows}
</table>
{driver_html}
<p><em>Full results available in the data/ directory.</em></p>
{method_note}
</div>
"""


def _section_mode3(result: dict, species: str = "") -> str:
    """Section 4: Weighted tRNA Adaptation Index."""
    n_genes = result["n_genes"]
    metagene_gs = result["metagene_geneset"]
    metagene_bg = result["metagene_genome"]
    ramp = result["ramp_analysis"]
    scores = result["per_gene_scores"]

    # Metagene plot + ramp bar chart
    plot_b64 = _plot_metagene(metagene_gs, metagene_bg, ramp, n_genes)

    col = "wtai" if "wtai" in scores.columns else "tai"
    mean_score = scores[col].mean()
    std_score = scores[col].std()

    sp = species.lower()
    if sp == "yeast":
        trna_note = (
            "tRNA gene copy numbers for <em>S. cerevisiae</em> (274 tRNA genes "
            "across ~42 anticodon families)"
        )
    elif sp == "human":
        trna_note = (
            "tRNA gene copy numbers for <em>H. sapiens</em> (~430 tRNA genes "
            "from GtRNAdb hg38)"
        )
    else:
        trna_note = f"tRNA gene copy numbers for {escape(species)}"

    method_note = f"""
<div class="method-note">
<p><strong>What this is.</strong> A position-by-position optimality profile across
your genes &mdash; how &ldquo;good&rdquo; the codons are at each position from
the 5&rsquo; end to the 3&rsquo; end, compared to the genome average.</p>
<p><strong>The score: wtAI.</strong> Each codon gets a <strong>weighted tRNA
Adaptation Index</strong> (wtAI) score based on {trna_note}. More tRNA gene copies
for a codon&rsquo;s anticodon = faster decoding = higher score. The
&ldquo;weighted&rdquo; part applies a 0.5&times; penalty to wobble base-pairing
(G:U or I:C at the third position), since wobble decoding is slower than
Watson-Crick. Scores are normalised 0&ndash;1, where 1.0 is the
most-efficiently-decoded codon in the genome.</p>
<p><strong>The metagene.</strong> We normalise every gene to 100 positional bins
(so a 300-codon gene and a 1,500-codon gene are comparable), compute the average
wtAI at each bin, and plot the curve. The blue line is your gene set, gray is
the genome. If blue is above gray, your genes use more optimal codons.</p>
<p><strong>The 5&rsquo; ramp.</strong> Many highly-expressed genes start with a
&ldquo;ramp&rdquo; of sub-optimal codons in the first ~30&ndash;50 positions,
thought to slow early elongation and space out ribosomes. The ramp delta
(body &minus; ramp) measures this: positive means slower start, which is the
expected pattern for highly-translated genes. If your gene set shows a bigger
ramp than the genome, that&rsquo;s a hint these genes are under translational
selection pressure. Or they just happen to start with rare codons. Biology is
fun like that.</p>
<p><strong>How this relates to published tAI.</strong> The original tRNA Adaptation
Index (dos Reis et al., <em>Nucleic Acids Res.</em> 2004) computes per-codon weights
by summing across <em>all</em> anticodons that can decode each codon, weighted by
optimized &ldquo;s-values&rdquo; representing codon&ndash;anticodon pairing
efficiencies. Those s-values were fit to maximise correlation with protein
abundance in <em>S.&nbsp;cerevisiae</em>. Our wtAI is a simplified variant: we
assign one primary decoding anticodon per codon (from curated wobble rules with
tRNA modification annotations), use tRNA gene copy numbers directly as weights,
and apply a uniform 0.5&times; wobble penalty instead of per-pairing-type
s-values. This approach is common in modern tools (e.g.&nbsp;stAIcalc, Sabi
&amp; Tuller 2017). The geometric mean calculation is identical. Absolute
wtAI values are not directly comparable to published tAI values from other
software, but <strong>relative comparisons</strong> (gene set vs genome,
ramp vs body) are robust &mdash; the same normalisation applies to both
sides. For yeast ribosomal proteins our wtAI places them at the 98th genome
percentile (Mann&ndash;Whitney <em>p</em>&nbsp;&lt;&nbsp;10<sup>&minus;17</sup>);
for human RP genes, the 83rd percentile
(<em>p</em>&nbsp;&lt;&nbsp;10<sup>&minus;12</sup>), consistent with the
known weaker translational selection signal in mammals
(Pouyet et al., <em>Genome Biol. Evol.</em> 2017).</p>
</div>"""

    # Ramp vs body composition tables
    ramp_comp = result.get("ramp_composition")
    body_comp = result.get("body_composition")
    ramp_body_html = ""
    if ramp_comp is not None and body_comp is not None:
        ramp_body_html = _ramp_body_tables(ramp_comp, body_comp, ramp)

    return f"""
<div class="section">
<h2>4. Weighted tRNA Adaptation Index</h2>
<p>Genes analyzed: <strong>{n_genes}</strong> &nbsp;|&nbsp;
   Mean {col.upper()}: <strong>{mean_score:.4f}</strong> (std {std_score:.4f})</p>
<div class="summary-grid">
  <div class="summary-card">
    <div class="label">Gene set ramp</div>
    <div class="value">{ramp['geneset_ramp_mean']:.3f}</div>
  </div>
  <div class="summary-card">
    <div class="label">Gene set body</div>
    <div class="value">{ramp['geneset_body_mean']:.3f}</div>
  </div>
  <div class="summary-card">
    <div class="label">Ramp delta</div>
    <div class="value">{ramp['geneset_ramp_delta']:+.3f}</div>
  </div>
  <div class="summary-card">
    <div class="label">Genome ramp delta</div>
    <div class="value">{ramp['genome_ramp_delta']:+.3f}</div>
  </div>
</div>
<div class="plot-container">{_img_tag(plot_b64)}</div>
{ramp_body_html}
<p><em>Full results available in the data/ directory.</em></p>
{method_note}
</div>
"""


def _section_mode4(result: dict, species: str = "") -> str:
    """Mode 4: Collision potential section."""
    n_genes = result["n_genes"]
    gs = result["transition_matrix_geneset"]
    bg = result["transition_matrix_genome"]
    fs_enrich = result["fs_enrichment"]
    chi2 = result["chi2_stat"]
    chi2_p = result["chi2_p"]

    # Transition bar plot
    plot_b64 = _plot_transitions(gs, bg, fs_enrich, chi2_p, n_genes)

    sp = species.lower()
    if sp == "yeast":
        trna_note = "yeast tRNA gene copy numbers (274 tRNA genes)"
    elif sp == "human":
        trna_note = "human tRNA gene copy numbers (~430 tRNA genes, GtRNAdb hg38)"
    else:
        trna_note = f"{escape(species)} tRNA gene copy numbers"

    method_note = f"""
<div class="method-note">
<p><strong>What this is.</strong> A ribosome staring at a fast codon followed by a
slow codon is a recipe for trouble: the trailing ribosome catches up and &mdash;
bam &mdash; collision. This mode asks whether your gene set has more or fewer of these
fast-to-slow (FS) transitions than you&rsquo;d expect from the genome.</p>
<p><strong>Fast vs slow.</strong> Every codon is classified as &ldquo;fast&rdquo; or
&ldquo;slow&rdquo; based on its wtAI score (see Weighted tRNA Adaptation Index above). The threshold is the
<strong>median wtAI</strong> across all codons, computed from {trna_note}.
Above the median = fast, below = slow. It&rsquo;s a rough binary, sure, but it
captures the basic kinetics.</p>
<p><strong>Four transition types.</strong> We scan each gene&rsquo;s CDS as adjacent
codon pairs and count how many are:
<strong>FF</strong> (fast&rarr;fast, smooth sailing),
<strong>FS</strong> (fast&rarr;slow, collision risk),
<strong>SF</strong> (slow&rarr;fast, ribosome speeds up), and
<strong>SS</strong> (slow&rarr;slow, already stalled, no new collision).
The proportions are averaged across your genes and compared to the genome.</p>
<p><strong>FS enrichment.</strong> This is the ratio of your gene set&rsquo;s FS
proportion to the genome&rsquo;s FS proportion. Values above 1.0 mean your genes
have <em>more</em> collision-prone junctions. Highly expressed genes (like ribosomal
proteins) tend to show FS enrichment near or below 1.0 &mdash; they&rsquo;ve been
selected to avoid exactly these transitions.</p>
<p><strong>Chi-squared test.</strong> Tests whether the overall distribution of
FF/FS/SF/SS in your gene set differs from the genome. A low p-value means
the transition pattern is genuinely unusual, not just noisy.</p>
</div>"""

    # Per-dicodon FS breakdown
    fs_dicodons = result.get("fs_dicodons")
    fs_dicodon_html = ""
    if fs_dicodons is not None and len(fs_dicodons) > 0:
        fs_dicodon_html = _fs_dicodon_table(fs_dicodons)

    # Collision waterfall bar chart
    collision_waterfall_html = ""
    cw_b64 = _plot_collision_waterfall(result)
    if cw_b64:
        collision_waterfall_html = f'<div class="plot-container">{_img_tag(cw_b64)}</div>'

    return f"""
<div class="section">
<h2>5. Collision Potential Analysis</h2>
<p>Genes analyzed: <strong>{n_genes}</strong> &nbsp;|&nbsp;
   Fast/slow threshold: {result['threshold']:.4f}</p>
<div class="summary-grid">
  <div class="summary-card">
    <div class="label">FS Enrichment</div>
    <div class="value">{fs_enrich:.3f}</div>
  </div>
  <div class="summary-card">
    <div class="label">Chi-squared</div>
    <div class="value">{chi2:.1f}</div>
  </div>
  <div class="summary-card">
    <div class="label">Chi-sq p-value</div>
    <div class="value">{chi2_p:.2e}</div>
  </div>
  <div class="summary-card">
    <div class="label">FS/SF ratio (gene set)</div>
    <div class="value">{result['fs_sf_ratio_geneset']:.3f}</div>
  </div>
</div>
<div class="plot-container">{_img_tag(plot_b64)}</div>
<table>
<tr><th>Transition</th><th>Gene Set</th><th>Genome</th><th>Fold Change</th></tr>
{"".join(f'<tr><td><strong>{t}</strong></td><td>{gs[t]:.4f}</td><td>{bg[t]:.4f}</td><td>{gs[t]/bg[t]:.3f}</td></tr>' if bg[t] > 0 else f'<tr><td><strong>{t}</strong></td><td>{gs[t]:.4f}</td><td>{bg[t]:.4f}</td><td>-</td></tr>' for t in ('FF','FS','SF','SS'))}
</table>
{collision_waterfall_html}
{fs_dicodon_html}
<p><em>Full results available in the data/ directory.</em></p>
{method_note}
</div>
"""


def _section_mode2(result: dict, species: str = "") -> str:
    """Mode 2: Translational demand section."""
    df = result["results"]
    top = result["top_genes"]
    n_genes = result["n_genes"]
    tissue = result["tissue"]

    sig = df[df["adjusted_p"] < 0.05]
    n_sig = len(sig)
    enriched = sig[sig["z_score"] > 0].sort_values("z_score", ascending=False)
    depleted = sig[sig["z_score"] < 0].sort_values("z_score", ascending=True)

    # Demand bar plot
    plot_b64 = _plot_demand_bars(enriched.head(30), depleted.head(30), n_genes, tissue)

    # Top genes table — use gene_name if available, fall back to gene (ENSG)
    name_col = "gene_name" if "gene_name" in top.columns else "gene"
    top_rows = ""
    for _, row in top.head(20).iterrows():
        display_name = str(row[name_col])
        top_rows += f"""<tr>
          <td><strong>{escape(display_name)}</strong></td>
          <td>{row['tpm']:.1f}</td>
          <td>{row['n_codons']}</td>
          <td>{row['demand_fraction']*100:.1f}%</td>
        </tr>"""

    sp = species.lower()
    if sp == "yeast":
        expr_note = (
            "Expression values are hardcoded estimates for rich-media growth "
            "(ribosomal proteins at ~3,000 TPM, glycolytic enzymes 500&ndash;5,000, "
            "median gene ~15 TPM). These are rough approximations &mdash; real "
            "RNA-seq data would be better, but the rank order is reasonable."
        )
        demand_note = (
            "In yeast, ribosomal proteins dominate translational demand so thoroughly "
            "(~70% of all ribosomes are making ribosomal proteins in log phase) that "
            "if your gene set <em>is</em> ribosomal proteins, the Z-scores will be "
            "modest. The genome demand is already saturated with RP-preferred codons."
        )
    elif sp == "human":
        tissue_str = str(tissue)
        if "proxy:" in tissue_str:
            # Cell line with GTEx proxy fallback
            expr_note = (
                f"Expression data is from <strong>GTEx v8</strong> tissue proxy "
                f"({escape(tissue_str)}). CCLE cell line expression data was not "
                f"available, so the closest GTEx tissue was used as a proxy. "
                f"This is an approximation &mdash; cell line expression can differ "
                f"substantially from normal tissue."
            )
        elif tissue_str == "cross_tissue_median":
            expr_note = (
                "Expression data is from <strong>GTEx v8</strong>, using the "
                "<strong>cross-tissue median TPM</strong> across all 54 tissues. "
                "This provides a general-purpose background when no specific tissue "
                "is selected. For tissue-specific analyses, use <code>--tissue</code> "
                "to select a specific GTEx tissue."
            )
        elif any(c.isdigit() or c == '-' for c in tissue_str) and tissue_str not in (
            "rich_media", "custom", "cross_tissue_median",
        ):
            # Likely a CCLE cell line name (contains digits/hyphens typically)
            expr_note = (
                f"Expression data is from <strong>DepMap CCLE</strong> "
                f"(cell line: {escape(tissue_str)}), RNA-seq TPM. "
                f"Cell line expression may differ from normal tissue; these values "
                f"reflect the transcriptome of an immortalised cell line."
            )
        else:
            expr_note = (
                f"Expression data is from <strong>GTEx v8</strong> "
                f"(tissue: {escape(tissue_str)}), median TPM across donors. "
                f"This is real measured RNA-seq, not estimates. Keep in mind that "
                f"TPM reflects mRNA abundance, not necessarily translation rates &mdash; "
                f"ribosome profiling would be the gold standard, but TPM is the best "
                f"widely-available proxy."
            )
        demand_note = (
            "Human tissues have very different expression landscapes. Liver is "
            "dominated by albumin and complement factors, brain by synaptic "
            "proteins. The demand profile can change dramatically depending on "
            "which tissue you pick. If your gene set is tissue-specific, try "
            "matching the tissue to get the most biologically relevant picture."
        )
    else:
        expr_note = f"Expression data for {escape(species)}."
        demand_note = ""

    method_note = f"""
<div class="method-note">
<p><strong>What this is.</strong> The enrichment analysis treats every gene equally &mdash; a gene
expressed at 0.1 TPM counts the same as one at 10,000 TPM. That&rsquo;s fine for
asking &ldquo;what codons does this gene set prefer?&rdquo; but it ignores a
crucial reality: the ribosome doesn&rsquo;t care about your gene list, it cares
about how many mRNAs are actually in the cytoplasm competing for tRNAs.
<strong>Translational demand</strong> weights each gene by
TPM &times; number of codons, so a highly expressed long gene contributes
far more to the demand landscape than a lowly expressed short one.</p>
<p><strong>Expression source.</strong> {expr_note}</p>
<p><strong>How it works.</strong> For each gene, we compute codon frequencies
and weight them by that gene&rsquo;s demand weight (TPM &times; n_codons).
The weighted mean across your gene set gives the &ldquo;demand profile&rdquo; &mdash;
what the tRNA pool actually has to decode. We then bootstrap-resample {n_genes}
genes (with their expression weights) from the genome to build a null, and
compute Z-scores exactly as in Mode 1. Enriched codons are the ones your gene
set is demanding disproportionately from the tRNA pool.</p>
<p><strong>Top genes table.</strong> The &ldquo;Demand %&rdquo; column shows each
gene&rsquo;s share of total translational demand within your gene set. If one gene
dominates (say, 40% of demand), the whole demand profile basically reflects that
gene&rsquo;s codon preferences. Worth checking whether your results are driven by
one outlier or are a genuine group effect.</p>
<p>{demand_note}</p>
</div>"""

    return f"""
<div class="section">
<h2>6. Translational Demand Analysis</h2>
<p>Genes analyzed: <strong>{n_genes}</strong> &nbsp;|&nbsp;
   Expression: <strong>{escape(str(tissue))}</strong> &nbsp;|&nbsp;
   Significant codons: <strong>{n_sig}</strong></p>
<h3>Top Demand-Contributing Genes</h3>
<table>
<tr><th>Gene</th><th>TPM</th><th>Codons</th><th>Demand %</th></tr>
{top_rows}
</table>
<div class="plot-container">{_img_tag(plot_b64)}</div>
<h3>Top Demand-Enriched Codons</h3>
{_demand_table(enriched.head(30))}
<h3>Top Demand-Depleted Codons</h3>
{_demand_table(depleted.head(30))}
<p><em>Full results available in the data/ directory.</em></p>
{method_note}
</div>
"""


def _section_mode6(result: dict, species: str = "") -> str:
    """Mode 6: Cross-species comparison section."""
    summary = result["summary"]
    pg = result["per_gene"]
    n_ortho = result["n_orthologs"]
    n_genome = result["n_genome_orthologs"]
    from_sp = result["from_species"]
    to_sp = result["to_species"]

    # Correlation histogram
    genome_corr = result["genome_correlations"]
    plot_b64 = _plot_correlation_hist(pg, genome_corr, summary, from_sp, to_sp)

    from_col = f"{from_sp}_gene"
    to_col = f"{to_sp}_gene"

    # Top conserved
    top_cons = pg.nlargest(10, "rscu_correlation")
    cons_rows = ""
    for _, row in top_cons.iterrows():
        cons_rows += f"""<tr>
          <td>{escape(str(row[from_col]))}</td>
          <td>{escape(str(row[to_col]))}</td>
          <td>{row['rscu_correlation']:.4f}</td>
        </tr>"""

    # Top divergent
    top_div = pg.nsmallest(10, "rscu_correlation")
    div_rows = ""
    for _, row in top_div.iterrows():
        div_rows += f"""<tr>
          <td>{escape(str(row[from_col]))}</td>
          <td>{escape(str(row[to_col]))}</td>
          <td>{row['rscu_correlation']:.4f}</td>
        </tr>"""

    method_note = f"""
<div class="method-note">
<p><strong>What this is.</strong> Do your genes use the same codon preferences in
{escape(from_sp)} and {escape(to_sp)}? For each ortholog pair, we compute
RSCU (Relative Synonymous Codon Usage) in both species and correlate them.
A high Pearson <em>r</em> means the two orthologs pick the same synonymous
codons (conserved codon usage); a low or negative <em>r</em> means
they&rsquo;ve diverged.</p>
<p><strong>Ortholog mapping.</strong> We use {n_genome} one-to-one ortholog pairs
between {escape(from_sp)} and {escape(to_sp)}, of which {n_ortho} overlap
with your gene set. Orthologs are matched by gene name with curated renames
for known discrepancies (e.g., yeast paralogs mapped to their human counterpart).
Only genes with orthologs in both species contribute to this analysis.</p>
<p><strong>RSCU correlation.</strong> For each ortholog pair, we compute RSCU
for all sense codons encoding multi-synonym amino acids (Met and Trp are excluded
since they have only one codon &mdash; there&rsquo;s no synonym to choose between).
The Pearson <em>r</em> across these ~59 codon values measures how similar the
synonymous preferences are. An <em>r</em> near 1.0 means the species use the
same synonyms; near 0 means no relationship; negative means they actively prefer
<em>different</em> synonyms.</p>
<p><strong>Gene set vs genome.</strong> We compare the mean <em>r</em> of your
gene set&rsquo;s orthologs against the genome-wide distribution (all {n_genome}
ortholog pairs). The Z-score and p-value tell you whether your genes are more
conserved or more divergent in codon usage than average. Low mean <em>r</em>
for ribosomal proteins, for instance, is expected: yeast and human ribosomes
are built the same way, but their tRNA pools have evolved differently, so
each species optimises the same proteins with different codons.</p>
</div>"""

    return f"""
<div class="section">
<h2>Cross-Species Comparison</h2>
<p>Ortholog pairs: <strong>{n_ortho}</strong> (gene set)  /
   <strong>{n_genome}</strong> (genome)</p>
<div class="summary-grid">
  <div class="summary-card">
    <div class="label">Gene set mean r</div>
    <div class="value">{summary['geneset_mean_r']:.3f}</div>
  </div>
  <div class="summary-card">
    <div class="label">Genome mean r</div>
    <div class="value">{summary['genome_mean_r']:.3f}</div>
  </div>
  <div class="summary-card">
    <div class="label">Z-score</div>
    <div class="value">{summary['z_score']:+.2f}</div>
  </div>
  <div class="summary-card">
    <div class="label">P-value</div>
    <div class="value">{summary['p_value']:.2e}</div>
  </div>
</div>
<div class="plot-container">{_img_tag(plot_b64)}</div>
<h3>Most Conserved Orthologs</h3>
<table>
<tr><th>{escape(from_sp)} gene</th><th>{escape(to_sp)} gene</th><th>RSCU r</th></tr>
{cons_rows}
</table>
<h3>Most Divergent Orthologs</h3>
<table>
<tr><th>{escape(from_sp)} gene</th><th>{escape(to_sp)} gene</th><th>RSCU r</th></tr>
{div_rows}
</table>
{method_note}
</div>
"""


def _section_error(title: str, exc: Exception) -> str:
    """Error section when a mode fails."""
    return f"""
<div class="section">
<h2>{escape(title)}</h2>
<div class="error-box">Analysis failed: {escape(str(exc))}</div>
</div>
"""


# ═══════════════════════════════════════════════════════════════════════════════
# README generator for zip export
# ═══════════════════════════════════════════════════════════════════════════════

def _generate_readme(
    species: str,
    gene_ids: list[str],
    id_mapping,
    n_bootstrap: int,
    seed: int | None,
    elapsed: float,
    tissue: str | None,
    cell_line: str | None,
    species2: str | None,
    data_dir_out: Path,
) -> str:
    """Generate a README.txt documenting the analysis for the zip export."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Species descriptions
    sp_desc = {
        "yeast": "Saccharomyces cerevisiae (6,685 ORFs from SGD, mitochondrial excluded)",
        "human": "Homo sapiens (19,229 CDS from MANE Select v1.5)",
        "mouse": "Mus musculus (~21,500 CDS from Ensembl GRCm39)",
    }
    species_full = sp_desc.get(species.lower(), species)

    # Expression source
    if species.lower() == "yeast":
        expr_source = "Hardcoded rich-media estimates (RP genes ~3000 TPM, median ~15 TPM)"
    elif species.lower() == "human":
        if cell_line:
            expr_source = f"CCLE cell line proxy: {cell_line} (mapped to nearest GTEx tissue)"
        elif tissue:
            expr_source = f"GTEx v8 median TPM: {tissue}"
        else:
            expr_source = "GTEx v8 cross-tissue median TPM"
    elif species.lower() == "mouse":
        expr_source = "Hardcoded estimates (RP genes ~3000 TPM, Actb ~5000 TPM, median ~15 TPM)"
    else:
        expr_source = "Default"

    # Gene list
    n_input = len(gene_ids)
    if n_input <= 30:
        gene_list_str = "\n".join(f"  {g}" for g in gene_ids)
    else:
        gene_list_str = "\n".join(f"  {g}" for g in gene_ids[:25])
        gene_list_str += f"\n  ... and {n_input - 25} more"

    # Data files present
    tsv_files = sorted(data_dir_out.glob("*.tsv"))
    file_descriptions = {
        "gene_mapping.tsv": "Input ID -> Gene name -> Systematic/Ensembl ID mapping",
        "mode1_monocodon.tsv": "Codon Enrichment: Single codon frequencies vs genome, Z-scores and adjusted p-values",
        "mode1_monocodon_matched.tsv": "Codon Enrichment: Monocodon vs length+GC matched background",
        "mode1_dicodon.tsv": "Dicodon Enrichment: Adjacent codon pair frequencies vs genome",
        "mode1_dicodon_matched.tsv": "Dicodon Enrichment: Dicodon vs length+GC matched background",
        "mode2_demand.tsv": "Translational Demand: Expression-weighted demand per codon",
        "mode3_profile.tsv": "Optimality Profile: Per-gene optimality scores (tAI/wtAI)",
        "mode3_ramp_composition.tsv": "Optimality Profile: Codon composition in the 5' ramp region",
        "mode3_body_composition.tsv": "Optimality Profile: Codon composition in the gene body (post-ramp)",
        "mode4_collision.tsv": "Collision Potential: Per-gene fast-to-slow (FS) transition fractions",
        "mode4_fs_dicodons.tsv": "Collision Potential: Per-dicodon FS enrichment breakdown",
        "mode5_attribution.tsv": "Attribution: Per-codon attribution (AA-driven vs synonymous codon choice)",
        "mode6_compare.tsv": "Cross-Species: Per-gene RSCU correlation between orthologs",
        "cai_per_gene.tsv": "CAI: Per-gene Codon Adaptation Index scores",
        "cai_weights.tsv": "CAI: Reference codon weights (from top 5% expressed genes)",
    }

    file_list = ""
    for f in tsv_files:
        desc = file_descriptions.get(f.name, "Analysis results")
        file_list += f"  data/{f.name}\n    {desc}\n\n"

    cross_sp = ""
    if species2:
        cross_sp = f"Cross-species comparison: {species} vs {species2}\n"

    return f"""CodonScope Analysis Results
{'=' * 60}

Generated: {now}
CodonScope version: {__version__}
Analysis time: {elapsed:.1f} seconds

ANALYSIS PARAMETERS
{'-' * 40}
Species: {species_full}
{cross_sp}Expression source: {expr_source}
Bootstrap iterations: {n_bootstrap:,}
Random seed: {seed if seed is not None else 'None (random)'}

INPUT GENE LIST ({n_input} genes, {id_mapping.n_mapped} mapped, {id_mapping.n_unmapped} unmapped)
{'-' * 40}
{gene_list_str}

{f"Unmapped IDs ({id_mapping.n_unmapped}): {', '.join(id_mapping.unmapped[:20])}" if id_mapping.n_unmapped > 0 else "All input IDs were successfully mapped."}

FILES IN THIS ARCHIVE
{'-' * 40}
  {Path(data_dir_out).parent.name.replace('_data', '')}.html
    Self-contained HTML report with embedded plots

  README.txt
    This file — documents the analysis parameters, gene list, and output files

{file_list}
ANALYSIS MODES
{'-' * 40}
Codon Enrichment Analysis: Compares mono-codon frequencies in your gene set
  vs the genome background. Uses bootstrap resampling ({n_bootstrap:,} iterations)
  to compute Z-scores. Benjamini-Hochberg FDR correction applied.
  Columns: kmer, amino_acid, observed_freq, expected_freq, z_score, p_value,
  adjusted_p, cohens_d

Dicodon Enrichment Analysis: Compares adjacent codon pair (dicodon)
  frequencies vs the genome background. Same statistical framework as above.

AA vs Synonymous Attribution: Separates amino acid composition effects from
  synonymous codon choice (RSCU). Classifies each significant codon as
  AA-driven, Synonymous-driven, or Both.

Codon Adaptation Index (CAI): Classic measure of codon optimality (Sharp &
  Li 1987). Reference weights from top 5% expressed genes. CAI is the geometric
  mean of per-codon weights. Genome percentile rank and Mann-Whitney U test.

Weighted tRNA Adaptation Index: Scores each codon position using the wobble-
  penalized tRNA Adaptation Index (wtAI). Higher = more efficiently decoded.
  Metagene profile normalised to 100 positional bins. Ramp analysis compares
  the first 50 codons to the gene body.

Collision Potential Analysis: Classifies codons as fast/slow based on wtAI,
  then counts Fast-to-Fast (FF), Fast-to-Slow (FS), Slow-to-Fast (SF), and
  Slow-to-Slow (SS) transitions. FS transitions are collision-prone.

Translational Demand Analysis: Weights each gene's codon usage by its
  expression level (TPM x number of codons). Shows which codons the ribosome
  pool is disproportionately decoding in your gene set vs the genome.

Cross-Species Comparison: Computes per-gene RSCU correlation between ortholog
  pairs across species. High r = conserved codon preferences, low r = divergent.

REFERENCE DATA SOURCES
{'-' * 40}
Yeast CDS: Saccharomyces Genome Database (SGD) verified ORFs
Human CDS: NCBI MANE Select v1.5 + Ensembl GRCh38
Mouse CDS: Ensembl GRCm39 canonical transcripts
tRNA gene copies: GtRNAdb (genomic tRNA database)
Human expression: GTEx v8 (Genotype-Tissue Expression project)
Orthologs: Ensembl Compara (mouse) / curated name matching (yeast)

PILOT GENE LISTS FOR TESTING
{'-' * 40}
Use these pre-made gene sets to verify CodonScope and understand the output.
Example files are in the examples/ directory of the repository.

Yeast RP genes (examples/yeast_rp_genes.txt, 132 genes):
  Ribosomal proteins — canonical translational selection. Expect strong
  synonymous-driven codon bias (Mode 5), high optimality with 5' ramp (Mode 3),
  low FS collision transitions (Mode 4). Best first test.

Yeast Gcn4 targets (examples/yeast_gcn4_targets.txt, 55 genes):
  Amino acid biosynthesis genes regulated by Gcn4. Expect AA-driven enrichment
  (Mode 5): glycine and arginine over-represented. Good contrast to RP genes
  (AA-driven vs synonymous-driven).

Yeast glycolytic enzymes (examples/yeast_glycolytic.txt, 17 genes):
  Glycolysis and fermentation (HXK1 through ADH1). Highly expressed, codon-
  optimized. Smaller set, so fewer codons reach significance.

Human RP genes (examples/human_rp_genes.txt, 80 core genes):
  Same biology as yeast, different preferred codons. Try with --tissue liver
  vs --tissue brain to see tissue-specific demand differences.

Human collagens (examples/human_collagen_genes.txt, 21 genes):
  Gly-X-Y repeat proteins. ~33% glycine vs ~7% genome average. Strongest
  AA-driven signal in Mode 5. Also enriched for proline codons.

Mouse RP genes (examples/mouse_rp_genes.txt, 80 core genes):
  Same mammalian tRNA pools as human — results parallel human RP analysis.

CITATION
{'-' * 40}
CodonScope v{__version__}
https://github.com/Meier-Lab-NCI/codonscope
"""


# ═══════════════════════════════════════════════════════════════════════════════
# Helper tables
# ═══════════════════════════════════════════════════════════════════════════════

def _results_table(df: pd.DataFrame, k: int) -> str:
    """Build an HTML table for Mode 1 results."""
    if len(df) == 0:
        return "<p><em>No significant results.</em></p>"

    kname = "kmer" if "kmer" in df.columns else "codon"
    has_aa = "amino_acid" in df.columns
    rows = ""
    for _, row in df.iterrows():
        z = row["z_score"]
        z_cls = "sig-pos" if z > 0 else "sig-neg"
        aa_td = f"<td>{escape(str(row['amino_acid']))}</td>" if has_aa else ""
        rows += f"""<tr>
          <td><strong>{escape(str(row[kname]))}</strong></td>
          {aa_td}
          <td class="{z_cls}">{z:+.2f}</td>
          <td>{row['observed_freq']:.4f}</td>
          <td>{row['expected_freq']:.4f}</td>
          <td>{row['adjusted_p']:.2e}</td>
          <td>{row.get('cohens_d', 0):.2f}</td>
        </tr>"""

    aa_th = "<th>AA</th>" if has_aa else ""
    return f"""<table>
<tr><th>K-mer</th>{aa_th}<th>Z-score</th><th>Observed</th><th>Expected</th><th>Adj. p</th><th>Cohen's d</th></tr>
{rows}
</table>"""


def _fs_dicodon_table(fs_dicodons: pd.DataFrame) -> str:
    """Build HTML table for per-dicodon FS enrichment breakdown."""
    # Show top 30 most enriched FS dicodons
    top = fs_dicodons.head(30)
    rows = ""
    for _, row in top.iterrows():
        z = row["z_score"]
        z_cls = "sig-pos" if z > 1.96 else ""
        fold = row["fold_enrichment"]
        fold_str = f"{fold:.2f}" if fold < 100 else ">100"
        rows += f"""<tr>
          <td><strong>{escape(str(row['dicodon']))}</strong></td>
          <td>{escape(str(row['amino_acids']))}</td>
          <td>{row['count_geneset']}</td>
          <td>{row['freq_geneset']:.5f}</td>
          <td>{row['freq_genome']:.5f}</td>
          <td>{fold_str}</td>
          <td class="{z_cls}">{z:+.2f}</td>
          <td>{row['adjusted_p']:.2e}</td>
        </tr>"""

    if not rows:
        return ""

    return f"""
<h3>Top Enriched FS Dicodons</h3>
<div class="method-note">
<p>Which specific fast&rarr;slow codon pairs are most enriched in your gene set?
Each row shows a dicodon where the first codon is fast (high wtAI) and the
second is slow (low wtAI). Frequency is relative to all dicodon transitions.
Z-scores use a Poisson approximation against genome proportions.</p>
</div>
<table>
<tr><th>Dicodon</th><th>AA transition</th><th>Count</th><th>Gene set freq</th>
<th>Genome freq</th><th>Fold</th><th>Z-score</th><th>Adj. p</th></tr>
{rows}
</table>
"""


def _ramp_body_tables(
    ramp_comp: pd.DataFrame,
    body_comp: pd.DataFrame,
    ramp: dict,
) -> str:
    """Build HTML tables for ramp vs body codon composition."""
    ramp_codons = ramp.get("ramp_codons", 50)

    # Focus on slow codons enriched in the ramp
    ramp_slow = ramp_comp[
        (ramp_comp["speed"] == "slow") & (ramp_comp["z_score"] > 1.5)
    ].sort_values("z_score", ascending=False)

    slow_rows = ""
    for _, row in ramp_slow.iterrows():
        z_cls = "sig-pos" if row["adjusted_p"] < 0.05 else ""
        slow_rows += f"""<tr>
          <td><strong>{escape(str(row['codon']))}</strong></td>
          <td>{escape(str(row['amino_acid']))}</td>
          <td class="{z_cls}">{row['z_score']:+.2f}</td>
          <td>{row['freq_geneset']:.4f}</td>
          <td>{row['freq_genome']:.4f}</td>
          <td>{row['adjusted_p']:.2e}</td>
        </tr>"""

    if not slow_rows:
        slow_html = "<p><em>No slow codons significantly enriched in the ramp (Z &gt; 1.5).</em></p>"
    else:
        slow_html = f"""<table>
<tr><th>Codon</th><th>AA</th><th>Z-score</th><th>Ramp freq</th><th>Genome ramp</th><th>Adj. p</th></tr>
{slow_rows}
</table>"""

    # Top significant codons in ramp vs body
    ramp_sig = ramp_comp[ramp_comp["adjusted_p"] < 0.05].head(10)
    body_sig = body_comp[body_comp["adjusted_p"] < 0.05].head(10)

    ramp_rows = ""
    for _, row in ramp_sig.iterrows():
        z_cls = "sig-pos" if row["z_score"] > 0 else "sig-neg"
        spd = f'<span class="badge badge-{"depleted" if row["speed"] == "slow" else "enriched"}">{row["speed"]}</span>'
        ramp_rows += f"""<tr>
          <td><strong>{escape(str(row['codon']))}</strong></td>
          <td>{escape(str(row['amino_acid']))}</td>
          <td>{spd}</td>
          <td class="{z_cls}">{row['z_score']:+.2f}</td>
          <td>{row['freq_geneset']:.4f}</td>
          <td>{row['freq_genome']:.4f}</td>
          <td>{row['adjusted_p']:.2e}</td>
        </tr>"""

    body_rows = ""
    for _, row in body_sig.iterrows():
        z_cls = "sig-pos" if row["z_score"] > 0 else "sig-neg"
        spd = f'<span class="badge badge-{"depleted" if row["speed"] == "slow" else "enriched"}">{row["speed"]}</span>'
        body_rows += f"""<tr>
          <td><strong>{escape(str(row['codon']))}</strong></td>
          <td>{escape(str(row['amino_acid']))}</td>
          <td>{spd}</td>
          <td class="{z_cls}">{row['z_score']:+.2f}</td>
          <td>{row['freq_geneset']:.4f}</td>
          <td>{row['freq_genome']:.4f}</td>
          <td>{row['adjusted_p']:.2e}</td>
        </tr>"""

    ramp_table = f"""<table>
<tr><th>Codon</th><th>AA</th><th>Speed</th><th>Z-score</th><th>Gene set</th><th>Genome</th><th>Adj. p</th></tr>
{ramp_rows}
</table>""" if ramp_rows else "<p><em>No significant codons in ramp region.</em></p>"

    body_table = f"""<table>
<tr><th>Codon</th><th>AA</th><th>Speed</th><th>Z-score</th><th>Gene set</th><th>Genome</th><th>Adj. p</th></tr>
{body_rows}
</table>""" if body_rows else "<p><em>No significant codons in body region.</em></p>"

    return f"""
<h3>Ramp vs Body Codon Composition</h3>
<div class="method-note">
<p>Monocodon frequencies computed separately for the <strong>ramp</strong>
(first {ramp_codons} codons) and <strong>body</strong> (codon {ramp_codons + 1}+)
of each gene, compared to the same regions in the genome background.</p>
</div>
<h3>Slow Codons Enriched in Ramp</h3>
{slow_html}
<h3>Ramp Region &mdash; Top Significant Codons</h3>
{ramp_table}
<h3>Body Region &mdash; Top Significant Codons</h3>
{body_table}
"""


def _demand_table(df: pd.DataFrame) -> str:
    """Build an HTML table for Mode 2 demand results."""
    if len(df) == 0:
        return "<p><em>No significant results.</em></p>"

    has_aa = "amino_acid" in df.columns
    rows = ""
    for _, row in df.iterrows():
        z = row["z_score"]
        z_cls = "sig-pos" if z > 0 else "sig-neg"
        aa_td = f"<td>{escape(str(row['amino_acid']))}</td>" if has_aa else ""
        rows += f"""<tr>
          <td><strong>{escape(str(row['kmer']))}</strong></td>
          {aa_td}
          <td class="{z_cls}">{z:+.2f}</td>
          <td>{row['demand_geneset']:.4f}</td>
          <td>{row['demand_genome']:.4f}</td>
          <td>{row['adjusted_p']:.2e}</td>
        </tr>"""

    aa_th = "<th>AA</th>" if has_aa else ""
    return f"""<table>
<tr><th>K-mer</th>{aa_th}<th>Z-score</th><th>Gene Set Demand</th><th>Genome Demand</th><th>Adj. p</th></tr>
{rows}
</table>"""


# ═══════════════════════════════════════════════════════════════════════════════
# Plot generators (return base64 PNG strings)
# ═══════════════════════════════════════════════════════════════════════════════

def _fig_to_b64(fig) -> str:
    """Convert matplotlib figure to base64 PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _img_tag(b64: str) -> str:
    """Wrap base64 string in an <img> tag."""
    return f'<img src="data:image/png;base64,{b64}" alt="plot">'


def _plot_volcano(df: pd.DataFrame, title: str) -> str:
    """Volcano plot: Z-score vs -log10(adjusted_p)."""
    fig, ax = plt.subplots(figsize=(10, 5))

    z = df["z_score"].values
    p = df["adjusted_p"].values
    neglog_p = -np.log10(np.clip(p, 1e-300, 1.0))

    sig_mask = p < 0.05
    ax.scatter(z[~sig_mask], neglog_p[~sig_mask],
               c="#94a3b8", s=20, alpha=0.5, label="Not significant")
    if sig_mask.any():
        colors = np.where(z[sig_mask] > 0, "#dc2626", "#2563eb")
        ax.scatter(z[sig_mask], neglog_p[sig_mask],
                   c=colors, s=30, alpha=0.7, label="Significant (adj_p < 0.05)")

    ax.axhline(-np.log10(0.05), color="gray", ls="--", lw=0.8, alpha=0.5)
    ax.axvline(0, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax.set_xlabel("Z-score")
    ax.set_ylabel("-log10(adjusted p-value)")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    return _fig_to_b64(fig)


def _plot_waterfall(df: pd.DataFrame, title: str, species_dir: Path | None = None) -> str:
    """Ranked waterfall bar chart for all 61 codons by Z-score.

    Wobble-decoded codons in amber, Watson-Crick in gray.
    Labels: "AAG (Lys)". Significance threshold lines at Z = +/- 1.96.
    """
    from codonscope.core.codons import CODON_TABLE, annotate_kmer

    # Load wobble info for coloring
    wobble_set: set[str] = set()
    if species_dir:
        wobble_path = species_dir / "wobble_rules.tsv"
        if wobble_path.exists():
            wdf = pd.read_csv(wobble_path, sep="\t")
            wobble_set = set(wdf[wdf["decoding_type"] == "wobble"]["codon"])

    # Sort all codons by Z-score descending
    sorted_df = df.sort_values("z_score", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(14, 5))

    zvals = sorted_df["z_score"].values
    kmers = sorted_df["kmer"].values

    # Color: wobble-decoded codons amber, Watson-Crick gray
    colors = []
    for km in kmers:
        if km in wobble_set:
            colors.append("#f59e0b")  # amber for wobble-decoded
        else:
            colors.append("#94a3b8")  # gray for Watson-Crick

    ax.bar(range(len(zvals)), zvals, color=colors, width=0.8, edgecolor="none")

    # Significance threshold lines
    ax.axhline(1.96, color="#059669", ls="--", lw=0.8, alpha=0.6)
    ax.axhline(-1.96, color="#059669", ls="--", lw=0.8, alpha=0.6)
    ax.axhline(0, color="black", lw=0.5, alpha=0.3)

    # Labels
    labels = [annotate_kmer(km, 1) for km in kmers]
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_ylabel("Z-score")
    ax.set_title(title)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#f59e0b", label="Wobble-decoded codons"),
        Patch(facecolor="#94a3b8", label="Watson-Crick codons"),
    ]
    ax.legend(handles=legend_elements, fontsize=7, loc="upper right")

    fig.tight_layout()
    return _fig_to_b64(fig)


def _plot_waterfall_dicodon(df: pd.DataFrame, title: str) -> str:
    """Ranked waterfall bar chart for top 30 enriched + bottom 30 depleted dicodons."""
    from codonscope.core.codons import annotate_kmer

    sorted_df = df.sort_values("z_score", ascending=False).reset_index(drop=True)
    top = sorted_df.head(30)
    bottom = sorted_df.tail(30)
    show_df = pd.concat([top, bottom]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(14, 5))

    zvals = show_df["z_score"].values
    kmers = show_df["kmer"].values

    colors = ["#dc2626" if z > 0 else "#2563eb" for z in zvals]
    ax.bar(range(len(zvals)), zvals, color=colors, width=0.8, edgecolor="none")

    ax.axhline(1.96, color="#059669", ls="--", lw=0.8, alpha=0.6)
    ax.axhline(-1.96, color="#059669", ls="--", lw=0.8, alpha=0.6)
    ax.axhline(0, color="black", lw=0.5, alpha=0.3)

    labels = [annotate_kmer(km, 2) for km in kmers]
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=5)
    ax.set_ylabel("Z-score")
    ax.set_title(title)

    fig.tight_layout()
    return _fig_to_b64(fig)


def _plot_collision_waterfall(result: dict) -> str:
    """Ranked bar chart of top 40 enriched + top 40 depleted dicodon transitions.

    Colored by transition type: FF=blue, FS=red, SF=orange, SS=gray.
    """
    from codonscope.core.codons import annotate_kmer

    fs_dicodons = result.get("fs_dicodons")
    if fs_dicodons is None or len(fs_dicodons) == 0:
        return ""

    # Sort by Z-score
    sorted_df = fs_dicodons.sort_values("z_score", ascending=False).reset_index(drop=True)

    # Top 40 enriched + top 40 depleted
    top = sorted_df.head(40)
    bottom = sorted_df.tail(40)
    show_df = pd.concat([top, bottom]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(14, 5))

    zvals = show_df["z_score"].values

    # Color by transition type if available, otherwise by Z direction
    type_colors = {"FF": "#2563eb", "FS": "#dc2626", "SF": "#f59e0b", "SS": "#94a3b8"}
    colors = []
    if "transition_type" in show_df.columns:
        for _, row in show_df.iterrows():
            colors.append(type_colors.get(row["transition_type"], "#94a3b8"))
    else:
        colors = ["#dc2626" if z > 0 else "#2563eb" for z in zvals]

    ax.bar(range(len(zvals)), zvals, color=colors, width=0.8, edgecolor="none")
    ax.axhline(0, color="black", lw=0.5, alpha=0.3)
    ax.axhline(1.96, color="#059669", ls="--", lw=0.8, alpha=0.6)
    ax.axhline(-1.96, color="#059669", ls="--", lw=0.8, alpha=0.6)

    if "dicodon" in show_df.columns:
        labels = [annotate_kmer(str(d), 2) for d in show_df["dicodon"].values]
    else:
        labels = [str(i) for i in range(len(zvals))]
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=5)
    ax.set_ylabel("Z-score")
    ax.set_title("Collision Potential: Ranked Dicodon Transitions")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2563eb", label="FF (fast-fast)"),
        Patch(facecolor="#dc2626", label="FS (fast-slow)"),
        Patch(facecolor="#f59e0b", label="SF (slow-fast)"),
        Patch(facecolor="#94a3b8", label="SS (slow-slow)"),
    ]
    ax.legend(handles=legend_elements, fontsize=7, loc="upper right")

    fig.tight_layout()
    return _fig_to_b64(fig)


def _plot_attribution_pie(summary: dict) -> str:
    """Pie chart of attribution categories."""
    fig, ax = plt.subplots(figsize=(5, 4))
    labels = []
    sizes = []
    colors_map = {
        "AA-driven": "#f59e0b",
        "Synonymous-driven": "#10b981",
        "Both": "#8b5cf6",
    }
    for cat in ("AA-driven", "Synonymous-driven", "Both"):
        key = f"n_{cat.lower().replace('-', '_')}"
        n = summary.get(key, 0)
        if n > 0:
            labels.append(f"{cat} ({n})")
            sizes.append(n)

    if not sizes:
        ax.text(0.5, 0.5, "No significant attributions",
                ha="center", va="center", transform=ax.transAxes)
    else:
        cols = [colors_map.get(l.split(" (")[0], "#94a3b8") for l in labels]
        ax.pie(sizes, labels=labels, colors=cols, autopct="%1.0f%%",
               startangle=90, textprops={"fontsize": 9})
        ax.set_title("Codon Bias Attribution")

    fig.tight_layout()
    return _fig_to_b64(fig)


def _plot_metagene(
    metagene_gs: np.ndarray,
    metagene_bg: np.ndarray,
    ramp: dict,
    n_genes: int,
) -> str:
    """Metagene profile + ramp bar chart."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Panel 1: Metagene profile
    x = np.arange(len(metagene_gs))
    ax1.plot(x, metagene_gs, color="#2563eb", lw=1.5, label="Gene set")
    ax1.plot(x, metagene_bg, color="#94a3b8", lw=1.5, label="Genome")
    ax1.set_xlabel("Normalised position (0=5', 99=3')")
    ax1.set_ylabel("Mean optimality (wtAI)")
    ax1.set_title(f"Metagene Optimality Profile (n={n_genes})")
    ax1.legend(fontsize=8)

    # Panel 2: Ramp analysis
    labels = ["Gene set\nramp", "Gene set\nbody", "Genome\nramp", "Genome\nbody"]
    values = [
        ramp["geneset_ramp_mean"], ramp["geneset_body_mean"],
        ramp["genome_ramp_mean"], ramp["genome_body_mean"],
    ]
    colors = ["#93c5fd", "#2563eb", "#d1d5db", "#6b7280"]
    ax2.bar(labels, values, color=colors, edgecolor="white", width=0.6)
    ax2.set_ylabel("Mean optimality (wtAI)")
    ax2.set_title("Ramp vs Body")

    fig.tight_layout()
    return _fig_to_b64(fig)


def _plot_transitions(
    gs: dict, bg: dict,
    fs_enrich: float, chi2_p: float,
    n_genes: int,
) -> str:
    """Transition proportion bar chart."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    types = ["FF", "FS", "SF", "SS"]
    x = np.arange(len(types))
    w = 0.35

    gs_vals = [gs[t] for t in types]
    bg_vals = [bg[t] for t in types]

    ax.bar(x - w/2, gs_vals, w, label="Gene set", color="#2563eb", edgecolor="white")
    ax.bar(x + w/2, bg_vals, w, label="Genome", color="#94a3b8", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(types)
    ax.set_ylabel("Proportion")
    ax.set_title(f"Codon Transition Proportions (n={n_genes})\n"
                 f"FS enrichment={fs_enrich:.3f}, chi2 p={chi2_p:.2e}")
    ax.legend(fontsize=8)

    fig.tight_layout()
    return _fig_to_b64(fig)


def _plot_demand_bars(
    enriched: pd.DataFrame,
    depleted: pd.DataFrame,
    n_genes: int,
    tissue: str,
) -> str:
    """Horizontal bar chart of top demand-enriched/depleted codons."""
    from codonscope.core.codons import annotate_kmer

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    if len(enriched) > 0:
        y = np.arange(len(enriched))
        ax1.barh(y, enriched["z_score"].values, color="#dc2626", edgecolor="white")
        ax1.set_yticks(y)
        k = len(enriched["kmer"].iloc[0]) // 3
        labels_e = [annotate_kmer(km, k) for km in enriched["kmer"].values]
        ax1.set_yticklabels(labels_e, fontsize=8)
        ax1.invert_yaxis()
        ax1.set_xlabel("Z-score")
        ax1.set_title("Demand-Enriched")
    else:
        ax1.text(0.5, 0.5, "None", ha="center", va="center", transform=ax1.transAxes)

    if len(depleted) > 0:
        y = np.arange(len(depleted))
        ax2.barh(y, depleted["z_score"].values, color="#2563eb", edgecolor="white")
        ax2.set_yticks(y)
        k = len(depleted["kmer"].iloc[0]) // 3
        labels_d = [annotate_kmer(km, k) for km in depleted["kmer"].values]
        ax2.set_yticklabels(labels_d, fontsize=8)
        ax2.invert_yaxis()
        ax2.set_xlabel("Z-score")
        ax2.set_title("Demand-Depleted")
    else:
        ax2.text(0.5, 0.5, "None", ha="center", va="center", transform=ax2.transAxes)

    fig.suptitle(f"Translational Demand (n={n_genes}, {tissue})", fontsize=11)
    fig.tight_layout()
    return _fig_to_b64(fig)


def _plot_correlation_hist(
    per_gene: pd.DataFrame,
    genome_corr: pd.DataFrame,
    summary: dict,
    from_sp: str,
    to_sp: str,
) -> str:
    """Correlation histogram: gene set vs genome."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    gs_r = per_gene["rscu_correlation"].dropna()
    bg_r = genome_corr["rscu_correlation"].dropna()

    bins = np.linspace(-1, 1, 41)
    ax1.hist(bg_r, bins=bins, alpha=0.5, label="Genome", density=True,
             color="#94a3b8", edgecolor="white")
    ax1.hist(gs_r, bins=bins, alpha=0.7, label="Gene set", density=True,
             color="#2563eb", edgecolor="white")
    ax1.axvline(summary["geneset_mean_r"], color="#2563eb", ls="--",
                label=f"Gene set mean={summary['geneset_mean_r']:.3f}")
    ax1.axvline(summary["genome_mean_r"], color="#6b7280", ls="--",
                label=f"Genome mean={summary['genome_mean_r']:.3f}")
    ax1.set_xlabel("RSCU Pearson r")
    ax1.set_ylabel("Density")
    ax1.set_title(f"{from_sp} vs {to_sp} RSCU Correlation\n"
                  f"Z={summary['z_score']:.2f}, p={summary['p_value']:.2e}")
    ax1.legend(fontsize=7)

    # Ranked bar chart
    sorted_r = gs_r.sort_values().reset_index(drop=True)
    ax2.bar(range(len(sorted_r)), sorted_r, color="#2563eb", width=1.0)
    ax2.axhline(summary["genome_mean_r"], color="#6b7280", ls="--",
                label=f"Genome mean={summary['genome_mean_r']:.3f}")
    ax2.set_xlabel("Ortholog pair (ranked)")
    ax2.set_ylabel("RSCU Pearson r")
    ax2.set_title("Per-Gene Cross-Species RSCU Correlation")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    return _fig_to_b64(fig)
