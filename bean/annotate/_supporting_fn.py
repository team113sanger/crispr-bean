from copy import deepcopy
from typing import List, Union, Dict, Optional
from tqdm.auto import tqdm
import numpy as np
from ..framework.Edit import Allele
from ..framework.AminoAcidEdit import CodingNoncodingAllele
import pandas as pd
from ..annotate.translate_allele import CDS, RefBaseMismatchException


def filter_allele_by_pos(
    allele: Allele,
    pos_start: Optional[Union[float, int]] = None,
    pos_end: Optional[Union[float, int]] = None,
    filter_rel_pos=True,
):
    """
    Filter alleles based on position and return the filtered allele and
    number of filtered edits.
    --
    Keyword arguments
    pos_start (int) -- start position to include (inclusive)
    pos_end (int) -- end position to include (exclusive)
    """
    filtered_edits = 0
    allele_filtered = deepcopy(allele)
    if not (pos_start is None and pos_end is None):
        if pos_start is None:
            pos_start = -np.inf
        if pos_end is None:
            pos_end = np.inf
        if filter_rel_pos:
            for edit in allele.edits:
                if not (edit.rel_pos >= pos_start and edit.rel_pos < pos_end):
                    filtered_edits += 1
                    allele_filtered.edits.remove(edit)
        else:
            for edit in allele.edits:
                if not (edit.pos >= pos_start and edit.pos < pos_end):
                    filtered_edits += 1
                    allele_filtered.edits.remove(edit)
    else:
        print("No threshold specified")  # TODO: warn
    return (allele_filtered, filtered_edits)


def filter_allele_by_base(
    allele: Allele,
    allowed_base_changes: Optional[Dict[str, str]] = None,
    allowed_ref_base: Optional[Union[List, str]] = None,
    allowed_alt_base: Optional[Union[List, str]] = None,
):
    """
    Filter alleles based on position and return the filtered allele and
    number of filtered edits.
    """
    filtered_edits = 0
    if isinstance(allowed_ref_base, str):
        allowed_ref_base = [allowed_ref_base]
    if isinstance(allowed_alt_base, str):
        allowed_alt_base = [allowed_alt_base]

    def _candidate_bases(edit):
        """Both orientations of an edit's base change: as stored, and complemented.

        The previous version complemented iff `edit.strand == "-"`, on the assumption
        that reporter storage orientation flips with the guide strand. It does not --
        the storage orientation is fixed for a whole dataset, so the editor's product
        appears as the SAME base change on every guide regardless of strand. Measured
        on this screen at the window stage, the ABE product is stored as T>C on both
        strands (65.1% of '+' edits, 62.3% of '-'), and CBE as G>A (62.7% / 64.0%).

        Under the old rule, --filter-target-basechange (the `allowed_base_changes`
        branch) matched target A>G only after complementing, so it kept T>C on '-'
        guides and discarded the identical T>C on '+' guides: read-weighted, '+' edits
        went 139,865,282 -> 158,782 (0.11% survived) while '-' kept 97.8%. That
        silently removed roughly half the library before translation and modelling.

        Matching either orientation is correct for both storage conventions and makes
        the filter strand-agnostic, which is what the data requires. Note this is only
        about *matching* -- CDS.edit_single still applies its own (empirically
        validated, ref-mismatch == 0) convention when writing bases into the CDS.
        """
        raw = (edit.ref_base, edit.alt_base)
        rmap = type(edit).reverse_map
        try:
            comp = (rmap[edit.ref_base], rmap[edit.alt_base])
        except KeyError:  # non-ACTG placeholder (indel marker); no complement
            return (raw,)
        return (raw, comp) if comp != raw else (raw,)

    if (allowed_ref_base is None and allowed_alt_base is None) + (
        allowed_base_changes is None
    ) != 1:
        print("No filters specified or misspecified filters.")
    elif allowed_base_changes is not None:
        for edit in allele.edits.copy():
            if not any(
                ref_base in allowed_base_changes
                and allowed_base_changes[ref_base] == alt_base
                for ref_base, alt_base in _candidate_bases(edit)
            ):
                filtered_edits += 1
                allele.edits.remove(edit)
    elif allowed_ref_base is not None:
        for edit in allele.edits.copy():
            if not any(
                ref_base in allowed_ref_base
                and (allowed_alt_base is None or alt_base in allowed_alt_base)
                for ref_base, alt_base in _candidate_bases(edit)
            ):
                filtered_edits += 1
                allele.edits.remove(edit)
    else:
        for edit in allele.edits.copy():
            if not any(
                alt_base in allowed_alt_base  # type: ignore
                for _, alt_base in _candidate_bases(edit)
            ):
                filtered_edits += 1
                allele.edits.remove(edit)
    return (allele, filtered_edits)


def get_aa_alleles(allele_str, include_synonymous=True):
    ldlr_cds = CDS()
    try:
        ldlr_cds.edit_allele(allele_str)
        ldlr_cds.get_aa_change(include_synonymous)
    except RefBaseMismatchException as e:
        print(e)
        return "ref mismatch"
    return ldlr_cds.get_aa_change(True)


def map_alleles_to_filtered(
    raw_allele_counts: pd.DataFrame,
    filtered_allele_counts: pd.DataFrame,
    jaccard_threshold=0.5,
):
    mapped_allele_counts = []  # pd.DataFrame(columns=raw_allele_counts.columns)
    for guide, guide_raw_counts in tqdm(
        raw_allele_counts.groupby("guide"),
        desc="Mapping alleles to closest filtered alleles",
    ):
        guide_filtered_allele_counts = filtered_allele_counts.loc[
            filtered_allele_counts.guide == guide, :
        ].set_index(
            "allele"
        )  # type: ignore
        guide_filtered_alleles = guide_filtered_allele_counts.index.tolist()
        if len(guide_filtered_alleles) == 0:
            pass
        else:
            guide_raw_counts["allele_mapped"] = guide_raw_counts.allele.map(
                lambda allele: allele.map_to_closest(
                    guide_filtered_alleles,
                    jaccard_threshold=jaccard_threshold,
                    merge_priority=guide_filtered_allele_counts.mean(
                        axis=1, numeric_only=True
                    ),
                )
            )
            # prioritize allele with most mean counts
            guide_raw_counts = guide_raw_counts.drop("allele", axis=1).rename(
                columns={"allele_mapped": "allele"}
            )
            guide_raw_counts = guide_raw_counts.groupby(["guide", "allele"]).sum()
            mapped_allele_counts.append(guide_raw_counts)
    res = pd.concat(mapped_allele_counts).reset_index()
    res = res.loc[res.allele.map(str) != "", :]
    return res


def _map_alleles_to_filtered(
    raw_allele_counts: pd.DataFrame,
    filtered_allele_counts: pd.DataFrame,
    jaccard_threshold=0.5,
    aa_jaccard_threshold=0.5,
    nt_jaccard_threshold=0.5,
    allele_col="allele",
):
    """
    Map pre-filtering alleles to the closest post-filtering alleles and aggregate the read count.
    """
    mapped_allele_counts = []
    is_cn_allele = isinstance(
        raw_allele_counts.reset_index()[allele_col][0], CodingNoncodingAllele
    )
    for guide, guide_raw_counts in tqdm(
        raw_allele_counts.groupby("guide"),
        desc="Mapping alleles to closest filtered alleles",
    ):
        guide_filtered_allele_counts = filtered_allele_counts.loc[
            filtered_allele_counts.guide == guide, :
        ].set_index(allele_col)
        guide_filtered_alleles = guide_filtered_allele_counts.index.tolist()
        if len(guide_filtered_alleles) == 0:
            pass
        else:
            # prioritize allele with most mean counts
            merge_priority = guide_filtered_allele_counts.mean(
                axis=1, numeric_only=True
            )
            if is_cn_allele:
                guide_raw_counts["allele_mapped"] = guide_raw_counts[allele_col].map(
                    lambda allele: allele.map_to_closest(
                        guide_filtered_alleles,
                        aa_jaccard_threshold=aa_jaccard_threshold,
                        nt_jaccard_threshold=nt_jaccard_threshold,
                        merge_priority=merge_priority,
                    )
                )
            else:
                guide_raw_counts["allele_mapped"] = guide_raw_counts[allele_col].map(
                    lambda allele: allele.map_to_closest(
                        guide_filtered_alleles,
                        jaccard_threshold=jaccard_threshold,
                        merge_priority=merge_priority,
                    )
                )
            guide_raw_counts = (
                guide_raw_counts.drop(allele_col, axis=1)
                .rename(columns={"allele_mapped": allele_col})
                .groupby(["guide", allele_col])
                .sum()
            )

            mapped_allele_counts.append(guide_raw_counts)
    res = pd.concat(mapped_allele_counts).reset_index()
    res = res.loc[res[allele_col].map(bool), :]
    return res
