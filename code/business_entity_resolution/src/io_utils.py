"""Raw TSV readers with parquet caching. All columns are read as strings, no quoting."""
import time
import polars as pl
from config import DATA_DIR, CACHE_DIR

COLS = ["entity_id", "business_name", "business_address", "country"]


def read_tsv(path):
    return pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False, encoding="utf8")


def load_source(split, src):
    """split in {train,test}, src in {1,2,3}. Cached as parquet."""
    cache = CACHE_DIR / f"raw_{split}_s{src}.parquet"
    if cache.exists():
        return pl.read_parquet(cache)
    t = time.time()
    df = read_tsv(DATA_DIR / split / f"{split}_source{src}.tsv").select(COLS)
    df = df.with_columns([pl.col(c).fill_null("") for c in COLS])
    df.write_parquet(cache)
    print(f"[io] {split} s{src}: {df.height:,} rows in {time.time()-t:.1f}s")
    return df


def load_ground_truth():
    """Long format: s1 (str), m (str matched id). Singletons appear with m=None."""
    cache = CACHE_DIR / "gt_long.parquet"
    if cache.exists():
        return pl.read_parquet(cache)
    gt = read_tsv(DATA_DIR / "train" / "train_ground_truth.tsv")
    gt = gt.rename({"source1_entity_id": "s1", "matched_entity_ids": "m"})
    gt = gt.with_columns(pl.col("m").fill_null("").str.split(",")).explode("m")
    gt = gt.with_columns(pl.when(pl.col("m") == "").then(None).otherwise(pl.col("m")).alias("m"))
    gt.write_parquet(cache)
    return gt


if __name__ == "__main__":
    for split in ("train", "test"):
        for s in (1, 2, 3):
            load_source(split, s)
    print(load_ground_truth().head())
