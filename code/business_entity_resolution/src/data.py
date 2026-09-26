"""Assemble S1 / pool frames and the train/validation split."""
import numpy as np
import polars as pl

from config import SEED, VAL_FRAC
from normalize import load_normalized
from io_utils import load_ground_truth


def load_split(split):
    """Returns (s1, pool). pool = S2 + S3 with a single `rid` space."""
    s1 = load_normalized(split, 1)
    s2 = load_normalized(split, 2)
    s3 = load_normalized(split, 3).with_columns((pl.col("rid") + s2.height).alias("rid"))
    return s1, pl.concat([s2, s3])


def train_val_ids(s1):
    """Deterministic 90/10 split of train S1 entity ids."""
    rng = np.random.default_rng(SEED)
    ids = np.sort(s1["entity_id"].to_numpy())
    is_val = rng.random(len(ids)) < VAL_FRAC
    return set(ids[~is_val].tolist()), set(ids[is_val].tolist())


def truth_long(s1_ids=None):
    gt = load_ground_truth()
    if s1_ids is not None:
        gt = gt.filter(pl.col("s1").is_in(list(s1_ids)))
    return gt
