"""Text normalization for names and addresses.

Output per record (parquet, one file per split/source):
  rid, entity_id, country, name (cleaned full name), core (name without legal forms),
  core_ns (core without spaces), legal (canonical legal-form codes), addr (canonical address
  tokens), nums (house/plot/zip numbers, leading zeros stripped).

Indic-script tokens are mapped to Latin with a dictionary mined from the training matches
(native token -> English token by Dice co-occurrence), falling back to unidecode.
"""
import json
import time
import polars as pl
from unidecode import unidecode

from config import CACHE_DIR
from io_utils import load_source, load_ground_truth

INDIC_RE = r"[ऀ-෿]"
TOKEN_SPLIT = r"[^\p{L}\p{M}\p{N}]+"

# ---- legal forms (US, India, France). Values are canonical codes. ----
LEGAL = {
    "pvt": "pvt", "private": "pvt", "pvtltd": "pvt ltd", "ltd": "ltd", "limited": "ltd",
    "llc": "llc", "inc": "inc", "incorporated": "inc", "corp": "corp", "corporation": "corp",
    "co": "co", "company": "co", "pc": "pc", "pllc": "pllc", "llp": "llp", "lp": "lp",
    "plc": "plc", "opc": "opc", "sarl": "sarl", "sas": "sas", "sasu": "sasu", "eurl": "eurl",
    "sa": "sa", "sci": "sci", "snc": "snc", "selarl": "selarl", "scop": "scop", "gie": "gie",
}

# ---- address tokens -> canonical ----
ADDR_MAP = {
    "street": "st", "str": "st", "road": "rd", "avenue": "ave", "av": "ave", "avn": "ave",
    "boulevard": "blvd", "bd": "blvd", "bld": "blvd", "boul": "blvd", "drive": "dr", "lane": "ln",
    "court": "ct", "place": "pl", "highway": "hwy", "suite": "ste", "apartment": "apt",
    "floor": "fl", "parkway": "pkwy", "circle": "cir", "terrace": "ter", "square": "sq",
    "north": "n", "south": "s", "east": "e", "west": "w", "opposite": "opp", "nr": "near",
    "building": "bldg", "bldg": "bldg", "sector": "sec", "number": "no", "nagar": "ngr",
    # French
    "r": "rue", "q": "quai", "all": "allee", "ch": "chemin", "chem": "chemin", "imp": "impasse",
    "rte": "route", "fbg": "faubourg", "sq": "sq", "st": "st",
}
ADDR_STOP = {"de", "du", "des", "la", "le", "les", "l", "d", "et", "the", "of", "and"}

US_STATES = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas", "ca": "california",
    "co": "colorado", "ct": "connecticut", "de": "delaware", "fl": "florida", "ga": "georgia",
    "hi": "hawaii", "id": "idaho", "il": "illinois", "in": "indiana", "ia": "iowa", "ks": "kansas",
    "ky": "kentucky", "la": "louisiana", "me": "maine", "md": "maryland", "ma": "massachusetts",
    "mi": "michigan", "mn": "minnesota", "ms": "mississippi", "mo": "missouri", "mt": "montana",
    "ne": "nebraska", "nv": "nevada", "nh": "new hampshire", "nj": "new jersey",
    "nm": "new mexico", "ny": "new york", "nc": "north carolina", "nd": "north dakota",
    "oh": "ohio", "ok": "oklahoma", "or": "oregon", "pa": "pennsylvania", "ri": "rhode island",
    "sc": "south carolina", "sd": "south dakota", "tn": "tennessee", "tx": "texas", "ut": "utah",
    "vt": "vermont", "va": "virginia", "wa": "washington", "wv": "west virginia",
    "wi": "wisconsin", "wy": "wyoming", "dc": "district of columbia", "pr": "puerto rico",
}
IN_STATES = {
    "ap": "andhra pradesh", "ar": "arunachal pradesh", "as": "assam", "br": "bihar",
    "cg": "chhattisgarh", "ct": "chhattisgarh", "ga": "goa", "gj": "gujarat", "hr": "haryana",
    "hp": "himachal pradesh", "jh": "jharkhand", "jk": "jammu and kashmir", "ka": "karnataka",
    "kl": "kerala", "mp": "madhya pradesh", "mh": "maharashtra", "mn": "manipur",
    "ml": "meghalaya", "mz": "mizoram", "nl": "nagaland", "od": "odisha", "or": "odisha",
    "pb": "punjab", "rj": "rajasthan", "sk": "sikkim", "tn": "tamil nadu", "tg": "telangana",
    "ts": "telangana", "tr": "tripura", "up": "uttar pradesh", "uk": "uttarakhand",
    "ut": "uttarakhand", "wb": "west bengal", "dl": "delhi", "ch": "chandigarh", "py": "puducherry",
    "orissa": "odisha",
}
STATE_MAPS = {"US": US_STATES, "India": IN_STATES}


def _base(expr):
    """NFKC, lowercase, strip mojibake/control chars. Keeps all scripts."""
    e = expr.str.normalize("NFKC").str.to_lowercase()
    e = e.str.replace_all(r"[Ââï][\u0080-¿]*[\u0080-\u009f]+[\u0080-¿]*", " ")
    e = e.str.replace_all(r"ï¿½|�|[\u0080-\u009f]", " ")
    e = e.str.replace_all("&", " and ")
    return e


def _tokens(expr):
    return expr.str.replace_all(TOKEN_SPLIT, " ").str.strip_chars().str.split(" ")


# ---------------- Indic dictionary mining ----------------

def mine_indic_dict(min_count=2, min_dice=0.3, rel=0.8):
    """Map native-script tokens to Latin tokens using train matches (S1 is always Latin)."""
    cache = CACHE_DIR / "indic_dict.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    t = time.time()
    gt = load_ground_truth().drop_nulls("m")
    s1 = load_source("train", 1).select(pl.col("entity_id").alias("s1"),
                                        pl.col("business_name").alias("n1"),
                                        pl.col("business_address").alias("a1"))
    pool = pl.concat([load_source("train", 2), load_source("train", 3)])
    pool = pool.filter(pl.col("business_name").str.contains(INDIC_RE)
                       | pl.col("business_address").str.contains(INDIC_RE))
    pool = pool.select(pl.col("entity_id").alias("m"), pl.col("business_name").alias("n2"),
                       pl.col("business_address").alias("a2"))
    j = pool.join(gt, on="m").join(s1, on="s1")
    d = j.select(
        _tokens(_base(pl.col("n2"))).alias("t"), _tokens(_base(pl.col("n1"))).alias("e"),
        _tokens(_base(pl.col("a2"))).alias("at"), _tokens(_base(pl.col("a1"))).alias("ae"),
    ).with_row_index("pid")
    # names are word-by-word transliterations: align by position when token counts agree
    nm = d.filter(pl.col("t").list.len() == pl.col("e").list.len()).select("t", "e").explode("t", "e")
    nm = nm.filter(pl.col("t").str.contains(INDIC_RE)).group_by("t", "e").len("c")
    nm = (nm.with_columns((pl.col("c") / pl.col("c").sum().over("t")).alias("share"))
          .filter((pl.col("c") >= min_count) & (pl.col("share") >= 0.4))
          .sort("c", descending=True).unique("t", keep="first"))
    name_map = dict(zip(nm["t"].to_list(), nm["e"].to_list()))
    # addresses are shuffled: Dice co-occurrence (native tokens are mostly state names)
    nat = d.select("pid", pl.col("at").alias("t")).explode("t").filter(pl.col("t").str.contains(INDIC_RE)).unique()
    eng = d.select("pid", pl.col("ae").alias("e")).explode("e").filter(pl.col("e").str.len_chars() > 1).unique()
    eng = eng.filter(pl.col("pid").is_in(nat["pid"].unique().implode()))
    pair = nat.join(eng, on="pid").group_by("t", "e").len("c")
    pair = pair.join(nat.group_by("t").len("ct"), on="t").join(eng.group_by("e").len("ce"), on="e")
    pair = pair.with_columns((2 * pl.col("c") / (pl.col("ct") + pl.col("ce"))).alias("dice"))
    # one native token may cover several Latin tokens (one-word state names)
    best = (pair.filter((pl.col("c") >= min_count) & (pl.col("dice") >= min_dice))
            .with_columns(pl.col("dice").max().over("t").alias("bd"))
            .filter(pl.col("dice") >= rel * pl.col("bd"))
            .sort(["t", "dice"], descending=[False, True])
            .group_by("t", maintain_order=True).agg(pl.col("e").head(3).str.join(" ")))
    addr_map = dict(zip(best["t"].to_list(), best["e"].to_list()))
    mapping = {"name": name_map, "addr": addr_map}
    cache.write_text(json.dumps(mapping, ensure_ascii=False), encoding="utf-8")
    print(f"[normalize] mined {len(name_map):,} name / {len(addr_map):,} addr indic tokens in {time.time()-t:.1f}s")
    return mapping


def _latinize_factory(mapping):
    def latinize(text):
        out = []
        for tok in text.split(" "):
            if tok in mapping:
                out.append(mapping[tok])
            else:
                out.append(unidecode(tok).lower())
        return " ".join(out)
    return latinize


# ---------------- main normalization ----------------

def _map_tokens(expr_list, mapping):
    return expr_list.list.eval(pl.element().replace(mapping))


def normalize_df(df, indic_map):
    """df has entity_id, business_name, business_address, country."""
    out = df.select(
        "entity_id", "country",
        _tokens(_base(pl.col("business_name"))).list.join(" ").alias("nt"),
        _tokens(_base(pl.col("business_address").str.replace_all(r"www\.\S+", " "))).list.join(" ").alias("at"),
    )
    # Indic tokens -> Latin (python only on rows that need it)
    for col, key in (("nt", "name"), ("at", "addr")):
        mask = out[col].str.contains(INDIC_RE)
        if mask.any():
            latinize = _latinize_factory({**indic_map["addr"], **indic_map["name"]} if key == "name"
                                         else {**indic_map["name"], **indic_map["addr"]})
            fixed = out.filter(mask)[col].map_elements(latinize, return_dtype=pl.String)
            out = out.with_columns(out[col].scatter(mask.arg_true(), fixed))
    # accent strip + re-tokenize
    def clean(col):
        return (pl.col(col).str.normalize("NFKD").str.replace_all(r"\p{M}", "")
                .str.replace_all(r"[^a-z0-9 ]+", " ").str.replace_all(r"\s+", " ").str.strip_chars())
    out = out.with_columns(clean("nt").alias("name"), clean("at").alias("addr0"))
    # name: drop web suffixes, split legal forms
    name = pl.col("name").str.replace_all(r"^(www )|( (com|in|co in|net|org|fr|biz|us|info))$", "")
    name = (name.str.replace_all(r"\bp l l c\b", "pllc").str.replace_all(r"\bl l (c|p)\b", "ll$1")
            .str.replace_all(r"\bs a r l\b", "sarl").str.replace_all(r"\bs a s\b", "sas")
            .str.replace_all(r"\bp v t\b", "pvt").str.replace_all(r"\bp c\b", "pc"))
    out = out.with_columns(name.str.strip_chars().alias("name"))
    ntok = pl.col("name").str.split(" ")
    legal_keys = list(LEGAL.keys())
    out = out.with_columns(
        ntok.list.eval(pl.element().filter(~pl.element().is_in(legal_keys + ["and", "the", "www", "com"])))
            .list.join(" ").alias("core"),
        ntok.list.eval(pl.element().filter(pl.element().is_in(legal_keys)).replace(LEGAL))
            .list.unique().list.sort().list.join(" ").alias("legal"),
    )
    out = out.with_columns(
        pl.when(pl.col("core") == "").then(pl.col("name")).otherwise(pl.col("core")).alias("core"))
    out = out.with_columns(pl.col("core").str.replace_all(" ", "").alias("core_ns"))
    # address: split digit/letter boundaries, canonical tokens, states, numbers
    a = pl.col("addr0").str.replace_all(r"(\d)([a-z])", "$1 $2").str.replace_all(r"([a-z])(\d)", "$1 $2")
    out = out.with_columns(a.str.split(" ").alias("atok"))
    out = out.with_columns(
        pl.col("atok").list.eval(pl.element().filter(pl.element().str.contains(r"^\d+$"))
                                 .str.replace(r"^0+(\d)", "$1")).list.unique(maintain_order=True).alias("nums"))
    parts = []
    for country, g in out.group_by("country", maintain_order=True):
        smap = STATE_MAPS.get(country[0], {})
        g = g.with_columns(
            pl.col("atok").list.eval(
                pl.element().str.replace(r"^0+(\d)", "$1").replace(ADDR_MAP).replace(smap)
                .filter(~pl.element().is_in(list(ADDR_STOP)) & (pl.element() != "")))
            .list.join(" ").alias("addr"))
        parts.append(g)
    out = pl.concat(parts)
    return out.select("entity_id", "country", "name", "core", "core_ns", "legal", "addr", "nums")


def load_normalized(split, src):
    cache = CACHE_DIR / f"norm_{split}_s{src}.parquet"
    if cache.exists():
        return pl.read_parquet(cache)
    t = time.time()
    df = normalize_df(load_source(split, src), mine_indic_dict())
    df = df.with_row_index("rid")
    df.write_parquet(cache)
    print(f"[normalize] {split} s{src}: {df.height:,} rows in {time.time()-t:.1f}s")
    return df


if __name__ == "__main__":
    import sys
    splits = sys.argv[1:] or ["train", "test"]
    for split in splits:
        for s in (1, 2, 3):
            load_normalized(split, s)
