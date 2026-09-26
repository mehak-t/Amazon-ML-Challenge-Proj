"""Candidate generation: weighted inverted-index blocking within country.

Each record emits hashed keys of several types:
  n    core-name token                 ns   whole core name without spaces
  p8   first 8 chars of core_ns        a    address word
  an   house number x address word     tw   name token x address word
  tn   name token x house number    nsw  core_ns x address word (glued names / domains)
Names repeat a lot across businesses, so the combined keys (tw, tn, an) carry most signal.
A key whose pool document frequency exceeds `cap` is dropped. Candidate score = sum over
shared keys of idf = log(N_country / df). Top-K per S1 are kept together with the per-type
shared weights (used as features downstream).
"""
import time
import numpy as np
import polars as pl

KEY_TYPES = ["n", "ns", "p8", "a", "an", "tw", "tn", "nsw"]
KT = {k: i for i, k in enumerate(KEY_TYPES)}
ADDR_SKIP = {"no", "rd", "st", "ave", "fl", "ste", "apt", "near", "opp", "plot", "door", "house",
             "flat", "shop", "floor", "bldg", "sec", "ngr", "blvd", "dr", "ln", "rue", "hwy", "unit",
             "po", "dist", "post", "tal", "vill", "village", "ward", "block", "phase", "main",
             "cross", "stage", "nd", "th", "ground", "first", "second", "allee", "chemin", "route",
             "impasse", "quai", "pl", "ct", "cir"}
MUL = 1000003


def _h(expr, seed):
    return expr.hash(seed=seed)


def _parts(df, max_name=5, max_words=10, max_nums=3):
    base = df.select("rid", "core", "core_ns", "addr", "nums")
    tok = (base.select("rid", pl.col("core").str.split(" ").list.unique(maintain_order=True)
                       .list.head(max_name).alias("t")).explode("t")
           .filter(pl.col("t").str.len_chars() >= 2))
    words = (base.select("rid", pl.col("addr").str.split(" ").list.eval(
                pl.element().filter((pl.element().str.len_chars() >= 3)
                                    & ~pl.element().str.contains(r"^\d")
                                    & ~pl.element().is_in(list(ADDR_SKIP))))
                .list.unique(maintain_order=True).list.head(max_words).alias("w")).explode("w")
             .drop_nulls("w"))
    nums = base.select("rid", pl.col("nums").list.head(max_nums).alias("x")).explode("x").drop_nulls("x")
    return base, tok, words, nums


def key_tables(df):
    """Yields (kt, frame[rid, key]) per key type, to keep peak memory low."""
    base, tok, words, nums = _parts(df)
    tok = tok.select("rid", _h(pl.col("t"), 1).alias("ht"), (pl.col("t").str.len_chars() >= 3).alias("long"))
    words = words.select("rid", _h(pl.col("w"), 2).alias("hw"))
    nums = nums.select("rid", _h(pl.col("x"), 3).alias("hx"))
    yield "n", tok.select("rid", (pl.col("ht") ^ KT["n"]).alias("key"))
    yield "ns", (base.filter(pl.col("core_ns").str.len_chars() >= 4)
                 .select("rid", _h(pl.col("core_ns"), 4).alias("key")))
    yield "p8", (base.filter(pl.col("core_ns").str.len_chars() >= 9)
                 .select("rid", _h(pl.col("core_ns").str.slice(0, 8), 5).alias("key")))
    yield "a", words.select("rid", (pl.col("hw") ^ KT["a"]).alias("key"))
    yield "an", nums.join(words, on="rid").select("rid", (pl.col("hx") * MUL ^ pl.col("hw")).alias("key"))
    lt = tok.filter(pl.col("long"))
    yield "tw", lt.join(words, on="rid").select("rid", (pl.col("ht") * MUL ^ pl.col("hw") ^ 77).alias("key"))
    yield "tn", lt.join(nums, on="rid").select("rid", (pl.col("ht") * MUL ^ pl.col("hx") ^ 99).alias("key"))
    hns = (base.filter(pl.col("core_ns").str.len_chars() >= 4)
           .select("rid", _h(pl.col("core_ns"), 6).alias("hn")))
    yield "nsw", hns.join(words, on="rid").select("rid", (pl.col("hn") * MUL ^ pl.col("hw") ^ 55).alias("key"))


def build_postings(pool_c, cap):
    n = pool_c.height
    posts = []
    for kt, t in key_tables(pool_c):
        t = t.unique()
        df = t.group_by("key").len("df").filter(pl.col("df") <= cap)
        df = df.with_columns((np.log(n) - pl.col("df").log()).cast(pl.Float32).alias("w"))
        posts.append(t.join(df.select("key", "w"), on="key")
                     .select("key", pl.col("rid").cast(pl.UInt32).alias("p_rid"), "w",
                             pl.lit(KT[kt], pl.UInt8).alias("kt")))
    return pl.concat(posts)


def generate(s1, pool, top_k=50, cap=200, s1_chunk=100_000, verbose=True, out_dir=None):
    """s1/pool: normalized frames with `rid`, `country`. Returns candidates
    (s1_rid, p_rid, score, w_<kt>..., rank); if out_dir is given, chunks are written there."""
    out = []
    t0 = time.time()
    for country in sorted(set(s1["country"].unique().to_list())):
        s1c = s1.filter(pl.col("country") == country)
        pc = pool.filter(pl.col("country") == country)
        if pc.height == 0:
            continue
        post = build_postings(pc, cap)
        t1 = time.time()
        for i in range(0, s1c.height, s1_chunk):
            ch = s1c.slice(i, s1_chunk)
            sk = pl.concat([t.select(pl.col("rid").cast(pl.UInt32).alias("s1_rid"), "key")
                            for _, t in key_tables(ch)]).unique()
            pairs = sk.join(post, on="key")
            agg = pairs.group_by("s1_rid", "p_rid").agg(
                [pl.col("w").filter(pl.col("kt") == KT[kt]).sum().alias(f"w_{kt}") for kt in KEY_TYPES]
                + [pl.col("w").sum().alias("score")])
            agg = (agg.sort(["s1_rid", "score"], descending=[False, True])
                   .with_columns(pl.int_range(pl.len()).over("s1_rid").alias("rank"))
                   .filter(pl.col("rank") < top_k))
            if out_dir is not None:
                agg.write_parquet(out_dir / f"cand_{country}_{i // s1_chunk:04d}.parquet")
            else:
                out.append(agg)
        if verbose:
            print(f"[blocking] {country}: s1={s1c.height:,} pool={pc.height:,} postings={post.height:,} "
                  f"post_t={t1-t0:.0f}s total={time.time()-t0:.0f}s", flush=True)
        del post
    return pl.concat(out) if out else None
