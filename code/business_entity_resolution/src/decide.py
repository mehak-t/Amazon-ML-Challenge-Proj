"""Decision layer tuned for macro F0.5.

1. unique assignment: a pool record belongs to at most one S1 (true in all train labels), so
   each pool record is kept only for the S1 that scores it highest;
2. absolute threshold `t` on the matcher probability;
3. relative threshold: keep candidates with p >= `rel` * (best p of that S1).
"""
import itertools
import polars as pl

from evaluate import macro_f05


def apply(f, t=0.5, rel=0.0, unique=True):
    d = f.select(pl.col("entity_id_1").alias("s1"), pl.col("entity_id_2").alias("m"), "p")
    if unique:
        d = d.filter(pl.col("p") == pl.col("p").max().over("m"))
    d = d.filter(pl.col("p") >= t)
    d = d.filter(pl.col("p") >= rel * pl.col("p").max().over("s1"))
    return d.select("s1", "m")


def tune(f, truth):
    best = None
    for t, rel, uniq in itertools.product([0.3, 0.4, 0.5, 0.6, 0.7, 0.8], [0.0, 0.3, 0.5], [True, False]):
        s = macro_f05(truth, apply(f, t, rel, uniq))
        if best is None or s > best[0]:
            best = (s, dict(t=t, rel=rel, unique=uniq))
    # refine t around the best
    s0, cfg = best
    for t in [cfg["t"] + d for d in (-0.08, -0.05, -0.03, -0.01, 0.01, 0.03, 0.05, 0.08)]:
        s = macro_f05(truth, apply(f, t, cfg["rel"], cfg["unique"]))
        if s > best[0]:
            best = (s, dict(cfg, t=t))
    print(f"[decide] best val F0.5={best[0]:.4f} with {best[1]}")
    return best[1]
