"""End-to-end pipeline: raw TSV -> normalize -> block -> features -> LightGBM -> decide -> TSVs.

  python run_pipeline.py block      # candidate generation for train and test (cached)
  python run_pipeline.py train      # features on a train sample, fit LightGBM, tune thresholds
  python run_pipeline.py predict    # score test candidates, write output/*.tsv
  python run_pipeline.py all
"""
import json
import sys
import time
import numpy as np
import polars as pl
import lightgbm as lgb

from config import CACHE_DIR, OUTPUT_DIR, SEED
from data import load_split, train_val_ids, truth_long
from blocking import generate
from features import build_chunked, add_global, FEATURES
from evaluate import report, blocking_report
import decide

CFG = {
    "block_top_k": 30,      # candidates kept from blocking (cached superset)
    "block_cap": 200,       # max pool document frequency of a blocking key
    "model_k": 10,          # candidates per S1 fed to the matcher (= candidate_pairs.tsv)
    "n_train_s1": 300_000,
    "n_val_s1": 100_000,
    "lgb": dict(objective="binary", learning_rate=0.08, num_leaves=127, min_data_in_leaf=50,
                feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                num_threads=0, seed=SEED, verbose=-1),
    "lgb_rounds": 600,
}


def log(msg, t0=[time.time()]):
    print(f"[{time.time()-t0[0]:7.0f}s] {msg}", flush=True)


def block(split):
    d = CACHE_DIR / f"cand_{split}"
    if d.exists() and any(d.iterdir()):
        return
    d.mkdir(exist_ok=True)
    s1, pool = load_split(split)
    generate(s1, pool, top_k=CFG["block_top_k"], cap=CFG["block_cap"], out_dir=d)
    log(f"blocking {split} done")


def load_cand(split, k):
    return pl.read_parquet(CACHE_DIR / f"cand_{split}" / "*.parquet").filter(pl.col("rank") < k)


def labeled(feats):
    gt = truth_long().drop_nulls("m").select(pl.col("s1").alias("entity_id_1"),
                                             pl.col("m").alias("entity_id_2"), pl.lit(1, pl.Int8).alias("y"))
    return feats.join(gt, on=["entity_id_1", "entity_id_2"], how="left").with_columns(pl.col("y").fill_null(0))


def train():
    s1, pool = load_split("train")
    tr_ids, va_ids = train_val_ids(s1)
    rng = np.random.default_rng(SEED)
    tr_s = set(rng.choice(sorted(tr_ids), CFG["n_train_s1"], replace=False).tolist())
    va_s = set(rng.choice(sorted(va_ids), CFG["n_val_s1"], replace=False).tolist())
    cand = add_global(load_cand("train", CFG["model_k"]))
    cand = cand.join(s1.select(pl.col("rid").cast(pl.UInt32).alias("s1_rid"), "entity_id"), on="s1_rid")
    out = {}
    for name, ids in (("train", tr_s), ("val", va_s)):
        c = cand.filter(pl.col("entity_id").is_in(list(ids))).drop("entity_id")
        f = labeled(build_chunked(c, s1, pool, add=False))
        f.write_parquet(CACHE_DIR / f"feat_{name}.parquet")
        out[name] = f
        log(f"features {name}: {f.height:,} pairs, pos rate {f['y'].mean():.3f}")
    ftr, fva = out["train"], out["val"]
    dtr = lgb.Dataset(ftr.select(FEATURES).to_numpy(), ftr["y"].to_numpy(), feature_name=FEATURES)
    dva = lgb.Dataset(fva.select(FEATURES).to_numpy(), fva["y"].to_numpy(), reference=dtr)
    model = lgb.train(CFG["lgb"], dtr, CFG["lgb_rounds"], valid_sets=[dva],
                      callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
    model.save_model(str(CACHE_DIR / "model.lgb"))
    imp = sorted(zip(FEATURES, model.feature_importance("gain")), key=lambda x: -x[1])
    log("top features: " + ", ".join(f"{k}={v:.0f}" for k, v in imp[:15]))
    fva = fva.with_columns(pl.Series("p", model.predict(fva.select(FEATURES).to_numpy())))
    from sklearn.metrics import average_precision_score
    log(f"val PR-AUC {average_precision_score(fva['y'].to_numpy(), fva['p'].to_numpy()):.4f}")
    # tune decision on val (truth covers ALL val S1, including those with no candidates)
    truth = truth_long(va_s)
    best = decide.tune(fva, truth)
    json.dump(best, open(CACHE_DIR / "decision.json", "w"), indent=1)
    country = s1.select(pl.col("entity_id").alias("s1"), "country")
    pred = decide.apply(fva, **best)
    report(truth, pred, country, label="val")
    blocking_report(truth, fva.select(pl.col("entity_id_1").alias("s1"), pl.col("entity_id_2").alias("m")),
                    label=f"val blocking k={CFG['model_k']}")


def write_ids(df, all_s1, col, path):
    g = df.group_by("s1").agg(pl.col("m").unique().sort().str.join(",").alias(col))
    out = all_s1.join(g, on="s1", how="left").with_columns(pl.col(col).fill_null(""))
    out = out.rename({"s1": "source1_entity_id"}).sort("source1_entity_id")
    out.write_csv(path, separator="\t", quote_style="never")
    log(f"wrote {path} ({out.height:,} rows, {(out[col] != '').sum():,} non-empty)")


def predict():
    s1, pool = load_split("test")
    model = lgb.Booster(model_file=str(CACHE_DIR / "model.lgb"))
    best = json.load(open(CACHE_DIR / "decision.json"))
    cand = load_cand("test", CFG["model_k"])
    f = build_chunked(cand, s1, pool)
    f = f.with_columns(pl.Series("p", model.predict(f.select(FEATURES).to_numpy())))
    f.select("entity_id_1", "entity_id_2", "p").write_parquet(CACHE_DIR / "test_scores.parquet")
    pred = decide.apply(f, **best)
    all_s1 = s1.select(pl.col("entity_id").alias("s1"))
    write_ids(f.select(pl.col("entity_id_1").alias("s1"), pl.col("entity_id_2").alias("m")),
              all_s1, "candidate_entity_ids", OUTPUT_DIR / "candidate_pairs.tsv")
    write_ids(pred, all_s1, "matched_entity_ids", OUTPUT_DIR / "matching_results.tsv")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage in ("block", "all"):
        block("train"); block("test")
    if stage in ("train", "all"):
        train()
    if stage in ("predict", "all"):
        predict()
