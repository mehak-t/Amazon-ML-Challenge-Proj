"""Pairwise features for (S1, candidate) pairs. Country is deliberately NOT a feature so the
model transfers to countries unseen in training (France)."""
import time
import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Levenshtein

from blocking import KEY_TYPES

REC_COLS = ["rid", "entity_id", "name", "core", "core_ns", "legal", "addr", "nums"]


def _pair(a, b, scorer):
    return process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32)


def _acronym(s):
    return s.str.split(" ").list.eval(pl.element().str.slice(0, 1)).list.join("")


def add_global(cand):
    """Features needing the whole candidate table (competition for the same pool record)."""
    return cand.with_columns(
        pl.len().over("p_rid").alias("p_n_s1"),
        (pl.col("score").max().over("p_rid") - pl.col("score")).alias("p_score_gap"),
        (pl.col("score") / pl.col("score").max().over("s1_rid")).alias("score_rel"),
        pl.len().over("s1_rid").alias("n_cand"),
    )


def build_chunked(cand, s1, pool, chunk_s1=150_000, add=True):
    """Feature build in S1 chunks (groups stay intact) to bound memory. Pass add=False when
    add_global was already applied on the full candidate table."""
    if add:
        cand = add_global(cand)
    ids = cand["s1_rid"].unique().sort()
    parts = []
    for i in range(0, len(ids), chunk_s1):
        sub = cand.filter(pl.col("s1_rid").is_in(ids.slice(i, chunk_s1).implode()))
        parts.append(build(sub, s1, pool).select(["s1_rid", "p_rid", "entity_id_1", "entity_id_2"] + FEATURES))
    return pl.concat(parts)


def build(cand, s1, pool):
    """cand: (s1_rid, p_rid, score, w_*, rank, + add_global cols). Returns cand with features."""
    t = time.time()
    a = s1.select([pl.col(c).alias(f"{c}_1") for c in REC_COLS])
    b = pool.select([pl.col(c).alias(f"{c}_2") for c in REC_COLS])
    d = (cand.join(a, left_on="s1_rid", right_on="rid_1")
             .join(b, left_on="p_rid", right_on="rid_2"))
    d = d.with_columns(
        pl.col("entity_id_2").str.starts_with("S3").cast(pl.Int8).alias("is_s3"),
        (pl.col("legal_1") == pl.col("legal_2")).cast(pl.Int8).alias("legal_eq"),
        ((pl.col("legal_1") == "") | (pl.col("legal_2") == "")).cast(pl.Int8).alias("legal_missing"),
        (pl.col("addr_2") == "").cast(pl.Int8).alias("addr2_empty"),
        (pl.col("core_1").str.split(" ").list.first() == pl.col("core_2").str.split(" ").list.first())
            .cast(pl.Int8).alias("first_tok_eq"),
        (_acronym(pl.col("core_1")) == pl.col("core_ns_2")).cast(pl.Int8).alias("acr_12"),
        (_acronym(pl.col("core_2")) == pl.col("core_ns_1")).cast(pl.Int8).alias("acr_21"),
        pl.col("nums_1").list.set_intersection("nums_2").list.len().alias("num_shared"),
        pl.col("nums_1").list.len().alias("num_n1"), pl.col("nums_2").list.len().alias("num_n2"),
        (pl.col("nums_1").list.first() == pl.col("nums_2").list.first()).fill_null(False)
            .cast(pl.Int8).alias("num_first_eq"),
        pl.col("core_1").str.len_chars().alias("len_c1"), pl.col("core_2").str.len_chars().alias("len_c2"),
        pl.col("core_1").str.split(" ").list.set_intersection(pl.col("core_2").str.split(" "))
            .list.len().alias("core_tok_shared"),
        pl.col("core_1").str.split(" ").list.len().alias("core_ntok1"),
        pl.col("core_2").str.split(" ").list.len().alias("core_ntok2"),
    )
    c1, c2 = d["core_1"].to_list(), d["core_2"].to_list()
    n1, n2 = d["name_1"].to_list(), d["name_2"].to_list()
    ns1, ns2 = d["core_ns_1"].to_list(), d["core_ns_2"].to_list()
    a1, a2 = d["addr_1"].to_list(), d["addr_2"].to_list()
    feats = {
        "core_ratio": _pair(c1, c2, fuzz.ratio),
        "core_tset": _pair(c1, c2, fuzz.token_set_ratio),
        "core_tsort": _pair(c1, c2, fuzz.token_sort_ratio),
        "core_partial": _pair(c1, c2, fuzz.partial_ratio),
        "core_jw": _pair(c1, c2, JaroWinkler.normalized_similarity),
        "ns_ratio": _pair(ns1, ns2, fuzz.ratio),
        "ns_partial": _pair(ns1, ns2, fuzz.partial_ratio),
        "name_tset": _pair(n1, n2, fuzz.token_set_ratio),
        "name_lev": _pair(n1, n2, Levenshtein.normalized_similarity),
        "addr_tset": _pair(a1, a2, fuzz.token_set_ratio),
        "addr_tsort": _pair(a1, a2, fuzz.token_sort_ratio),
        "addr_partial": _pair(a1, a2, fuzz.partial_ratio),
    }
    d = d.with_columns([pl.Series(k, v) for k, v in feats.items()])
    # within-S1-group context
    grp = "s1_rid"
    d = d.with_columns(
        (pl.col("core_tset").max().over(grp) - pl.col("core_tset")).alias("core_tset_gap"),
        (pl.col("addr_tset").max().over(grp) - pl.col("addr_tset")).alias("addr_tset_gap"),
        pl.col("core_tset").rank("dense", descending=True).over(grp).alias("core_rank"),
    )
    print(f"[features] {d.height:,} pairs in {time.time()-t:.0f}s")
    return d


FEATURES = (["score", "rank", "score_rel", "n_cand", "p_n_s1", "p_score_gap"]
            + [f"w_{k}" for k in KEY_TYPES]
            + ["is_s3", "legal_eq", "legal_missing", "addr2_empty", "first_tok_eq", "acr_12", "acr_21",
               "num_shared", "num_n1", "num_n2", "num_first_eq", "len_c1", "len_c2",
               "core_tok_shared", "core_ntok1", "core_ntok2",
               "core_ratio", "core_tset", "core_tsort", "core_partial", "core_jw", "ns_ratio",
               "ns_partial", "name_tset", "name_lev", "addr_tset", "addr_tsort", "addr_partial",
               "core_tset_gap", "addr_tset_gap", "core_rank"])
