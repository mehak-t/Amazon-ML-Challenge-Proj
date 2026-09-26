"""Official-style metrics: macro F0.5 over all S1 entities, plus blocking diagnostics."""
import numpy as np
import polars as pl

BETA2 = 0.25


def f05_per_entity(truth, pred):
    """truth/pred: DataFrames (s1, m) long format; truth must list every S1 (m=None for singletons).
    Returns DataFrame s1, n_true, n_pred, tp, f05."""
    t = truth.group_by("s1").agg(pl.col("m").drop_nulls().alias("t"))
    p = pred.drop_nulls("m").group_by("s1").agg(pl.col("m").unique().alias("p"))
    d = t.join(p, on="s1", how="left").with_columns(pl.col("p").fill_null(pl.lit([], pl.List(pl.String))))
    d = d.with_columns(
        pl.col("t").list.len().alias("n_true"), pl.col("p").list.len().alias("n_pred"),
        pl.col("t").list.set_intersection("p").list.len().alias("tp"))
    prec = pl.col("tp") / pl.col("n_pred")
    rec = pl.col("tp") / pl.col("n_true")
    f = ((1 + BETA2) * prec * rec / (BETA2 * prec + rec)).fill_nan(0.0).fill_null(0.0)
    d = d.with_columns(
        pl.when((pl.col("n_true") == 0) & (pl.col("n_pred") == 0)).then(1.0)
        .when((pl.col("n_true") == 0) | (pl.col("n_pred") == 0)).then(0.0)
        .when(pl.col("tp") == 0).then(0.0)
        .otherwise(f).alias("f05"))
    return d.select("s1", "n_true", "n_pred", "tp", "f05")


def macro_f05(truth, pred):
    return f05_per_entity(truth, pred)["f05"].mean()


def report(truth, pred, s1_country=None, label="val"):
    d = f05_per_entity(truth, pred)
    out = {"f05": d["f05"].mean(),
           "singleton_acc": d.filter(pl.col("n_true") == 0)["f05"].mean(),
           "nonsingleton_f05": d.filter(pl.col("n_true") > 0)["f05"].mean(),
           "micro_precision": d["tp"].sum() / max(d["n_pred"].sum(), 1),
           "micro_recall": d["tp"].sum() / max(d["n_true"].sum(), 1)}
    if s1_country is not None:
        dc = d.join(s1_country, on="s1")
        for c, g in dc.group_by("country"):
            out[f"f05_{c[0]}"] = g["f05"].mean()
    print(f"[{label}] " + "  ".join(f"{k}={v:.4f}" for k, v in out.items()))
    return out


def blocking_report(truth, cand, label="blocking"):
    """cand: DataFrame (s1, m) of candidate pairs. Recall over true pairs, size stats per S1."""
    tp = truth.drop_nulls("m")
    hit = tp.join(cand.select("s1", "m").unique(), on=["s1", "m"], how="semi").height
    sizes = (truth.select("s1").unique().join(cand.group_by("s1").len("n"), on="s1", how="left")
             .with_columns(pl.col("n").fill_null(0)))["n"].to_numpy()
    out = {"recall": hit / max(tp.height, 1), "mean": float(sizes.mean()),
           "median": float(np.median(sizes)), "p95": float(np.percentile(sizes, 95)),
           "pairs": int(sizes.sum())}
    print(f"[{label}] " + "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                    for k, v in out.items()))
    return out


if __name__ == "__main__":
    # self-test against the README example + singleton rules
    truth = pl.DataFrame({"s1": ["a", "a", "b", "c"], "m": ["x", "z", None, "q"]})
    pred = pl.DataFrame({"s1": ["a", "a", "a", "b"], "m": ["x", "y", "z", "w"]})
    d = f05_per_entity(truth, pred).sort("s1")
    print(d)
    assert abs(d["f05"][0] - 0.7142857) < 1e-6 and d["f05"][1] == 0.0 and d["f05"][2] == 0.0
    assert f05_per_entity(truth.filter(pl.col("s1") == "b"), pred.head(0))["f05"][0] == 1.0
    print("evaluate self-test OK")
