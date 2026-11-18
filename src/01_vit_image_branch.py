# %% [markdown]
# # ImplicitMABSA — ViT Image Branch (Twitter-2015 + Twitter-2017)
#
# Frozen `google/vit-base-patch16-224` global image features for the Twitter-2015 and Twitter-2017
# multimodal aspect-based sentiment datasets.
#
# * Original train / dev / test splits are preserved (`dev.tsv` is saved as `validation`).
# * `sample_id = {dataset}_{split}_{original_index}`, where `original_index` is the `index` column of the source TSV.
# * Features are extracted once per unique valid image; every split has a `vit_mapping.csv` that maps each
#   annotation row (sample) to a feature row. **Never join by row position.**
# * Saved per split: CLS vectors, mean-pooled patch vectors, and all 197 patch tokens per image
#   (float16, for cross-attention fusion).
# * Missing / corrupted images are recorded, never silently dropped.
#
# Run the cells top to bottom. Each stage prints a short summary to check before moving on.

# %%
# ==== Stage 0: configuration, imports, reproducibility, logging ====
import os, sys, json, time, hashlib, logging, random, platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers
from PIL import Image, ImageFile
from transformers import ViTImageProcessor, ViTModel

# Truncated / corrupted files must raise an error, never be silently padded.
ImageFile.LOAD_TRUNCATED_IMAGES = False

CONFIG = {
    "input_root": "/kaggle/input",
    "output_root": "/kaggle/working/vit_image_branch",
    "datasets": ["twitter2015", "twitter2017"],
    # Kaggle datasets the raw files come from (for documentation only)
    "kaggle_sources": {
        "twitter2015": "mdfaishalahmedrudroo/twitter2015",
        "twitter2017": "wentengh/twitter2017",
    },
    # output split name -> original annotation file (original splits preserved)
    "split_files": {"train": "train.tsv", "validation": "dev.tsv", "test": "test.tsv"},
    "model_name": "google/vit-base-patch16-224",
    "batch_size": 64,
    "num_workers": 2,
    "seed": 42,
    "spot_check_per_split": 5,
    "spot_check_atol": 1e-3,
    # all final-layer tokens (CLS + 196 patches) for cross-attention fusion; float16 halves the size
    "save_patch_tokens": True,
    "patch_token_dtype": "float16",
    "patch_token_tol": 1e-2,  # float16 rounding tolerance used when checking patch tokens
}
# Commonly reported split sizes. Reference only: the actual files are the source of truth.
REFERENCE_COUNTS = {
    "twitter2015": {"train": 3179, "validation": 1122, "test": 1037},
    "twitter2017": {"train": 3562, "validation": 1176, "test": 1234},
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(CONFIG["seed"])

OUT = Path(CONFIG["output_root"])
OUT.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("vit_branch")
logger.setLevel(logging.INFO)
logger.propagate = False
logger.handlers.clear()
_fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
for _h in (logging.StreamHandler(sys.stdout),
           logging.FileHandler(OUT / "processing.log", mode="a", encoding="utf-8")):
    _h.setFormatter(_fmt)
    logger.addHandler(_h)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ENV = {
    "python": platform.python_version(),
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "pillow": Image.__version__,
    "device": str(DEVICE),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
}
logger.info("Environment: %s", json.dumps(ENV))

# %%
# ==== Stage 1: locate annotation and image folders, inspect raw files ====
def discover_dataset(name):
    """Find the folder holding train/dev/test.tsv and the `<name>_images` folder under /kaggle/input."""
    needed = set(CONFIG["split_files"].values())
    ann_dirs, img_dirs = [], []
    for root, dirs, files in os.walk(CONFIG["input_root"]):
        base = os.path.basename(root)
        if base == f"{name}_images":
            img_dirs.append(root)
            dirs[:] = []  # do not descend further into the image folder
            continue
        if base == name and needed.issubset(files):
            ann_dirs.append(root)
    if len(ann_dirs) != 1 or len(img_dirs) != 1:
        raise RuntimeError(f"{name}: expected exactly one annotation dir and one image dir, "
                           f"found annotations={ann_dirs}, images={img_dirs}")
    return Path(ann_dirs[0]), Path(img_dirs[0])


PATHS = {}
for ds in CONFIG["datasets"]:
    ann_dir, img_dir = discover_dataset(ds)
    PATHS[ds] = {"ann_dir": ann_dir, "img_dir": img_dir}
    entries = os.listdir(img_dir)
    exts = pd.Series([Path(f).suffix.lower() for f in entries]).value_counts().to_dict()
    print(f"\n=== {ds} ===")
    print("annotation dir:", ann_dir, "| files:", sorted(os.listdir(ann_dir)))
    print("image dir     :", img_dir, "| entries:", len(entries), "| extensions:", exts)
    for split, fname in CONFIG["split_files"].items():
        with open(ann_dir / fname, encoding="utf-8-sig") as f:
            lines = f.read().splitlines()
        print(f"  {fname}: {len(lines)} lines incl. header")
        for line in lines[:2]:
            print("     ", repr(line[:150]))

# %%
# ==== Stage 2: parse annotations, build the sample table, integrity checks ====
import re

# Header variants seen in the source files. Data rows always have 5 tab-separated fields in the order
# index, label, image_id, tweet-with-$T$, aspect_term. Some files carry a mislabelled 4-name header
# (observed: twitter2015/test.tsv), so columns are validated by content below, not by header names.
KNOWN_HEADERS = {
    ("index", "#1 Label", "#2 ImageID", "#3 String", "#3 String"): "standard",
    ("index", "#1 ImageID", "#2 String", "#2 String"): "mislabelled 4-name header; rows have 5 fields",
}
VALID_LABELS = {"0", "1", "2"}
IMAGE_RE = re.compile(r"^[\w\-.]+\.(jpg|jpeg|png|gif|bmp)$", re.IGNORECASE)
HEADER_NOTES = []


def load_split(ds, split):
    path = PATHS[ds]["ann_dir"] / CONFIG["split_files"][split]
    with open(path, encoding="utf-8-sig", newline="") as f:
        lines = f.read().split("\n")
    header = tuple(lines[0].rstrip("\r").split("\t"))
    if header not in KNOWN_HEADERS:
        raise ValueError(f"{path}: unknown header {header}; inspect the file before extending KNOWN_HEADERS")
    HEADER_NOTES.append({"dataset": ds, "split": split, "file": path.name, "header": "\t".join(header),
                         "header_note": KNOWN_HEADERS[header]})
    rows, malformed = [], []
    for line_no, line in enumerate(lines[1:], start=2):
        line = line.rstrip("\r")
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) != 5:
            malformed.append({"line_no": line_no, "detail": f"{len(parts)} fields", "raw": line[:200]})
            continue
        idx, label, img, text, target = parts
        rows.append({
            "sample_id": f"{ds}_{split}_{idx.strip()}",
            "dataset": ds,
            "split": split,
            "source_file": path.name,
            "line_no": line_no,
            "original_index": idx.strip(),
            "image_filename": img.strip(),
            "label_raw": label.strip(),
            "aspect_term": target,
            "text_with_target_mask": text,
        })
    return pd.DataFrame(rows), malformed


SAMPLES, ANN_ISSUES = {}, {}
summary = []
for ds in CONFIG["datasets"]:
    issues, parts = [], []
    for split in CONFIG["split_files"]:
        df, malformed = load_split(ds, split)
        issues += [{"split": split, "issue": "malformed_row", **m} for m in malformed]
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    for _, r in df[df.duplicated("sample_id", keep=False)].iterrows():
        issues.append({"split": r.split, "issue": "duplicate_sample_id", "line_no": r.line_no, "detail": r.sample_id})
    # content checks (positional columns must contain what we expect)
    for _, r in df[~df.image_filename.str.match(IMAGE_RE)].iterrows():
        issues.append({"split": r.split, "issue": "bad_image_filename", "line_no": r.line_no, "detail": r.image_filename})
    for _, r in df[~df.label_raw.isin(VALID_LABELS)].iterrows():
        issues.append({"split": r.split, "issue": "bad_label", "line_no": r.line_no, "detail": r.label_raw})
    for _, r in df[~df.text_with_target_mask.str.contains("$T$", regex=False)].iterrows():
        issues.append({"split": r.split, "issue": "no_$T$_in_text", "line_no": r.line_no, "detail": r.sample_id})
    SAMPLES[ds], ANN_ISSUES[ds] = df, issues

    for split, g in df.groupby("split", sort=False):
        ref = REFERENCE_COUNTS.get(ds, {}).get(split)
        summary.append({
            "dataset": ds, "split": split, "rows": len(g), "reference": ref,
            "matches_reference": (len(g) == ref) if ref else None,
            "unique_images": g.image_filename.nunique(),
            "label_counts": dict(sorted(g.label_raw.value_counts().items())),
        })
    sets = {s: set(g.image_filename) for s, g in df.groupby("split")}
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        print(f"{ds}: images shared between {a} and {b}: {len(sets[a] & sets[b])}")

print()
print(pd.DataFrame(HEADER_NOTES)[["dataset", "split", "header_note"]].to_string(index=False))
print()
print(pd.DataFrame(summary).to_string(index=False))
for ds in CONFIG["datasets"]:
    print(f"\n{ds}: {len(ANN_ISSUES[ds])} annotation issues")
    if ANN_ISSUES[ds]:
        print(pd.DataFrame(ANN_ISSUES[ds]).head(20).to_string(index=False))
print("\nExample rows:")
print(SAMPLES[CONFIG["datasets"][0]].head(3)[["sample_id", "image_filename", "label_raw", "aspect_term"]].to_string(index=False))

FATAL = {"malformed_row", "duplicate_sample_id", "bad_image_filename", "bad_label"}
_fatal = [(ds, i) for ds in ANN_ISSUES for i in ANN_ISSUES[ds] if i["issue"] in FATAL]
assert not _fatal, f"{len(_fatal)} structural annotation problems; inspect them before continuing: {_fatal[:5]}"

# %%
# ==== Stage 3: validate every referenced image and build the canonical mapping ====
def inspect_image(path):
    info = {"status": "ok", "error": "", "format": None, "mode": None,
            "width": None, "height": None, "bytes": None, "md5": None}
    if not path.is_file():
        info["status"] = "missing"
        return info
    try:
        data = path.read_bytes()
        info["bytes"] = len(data)
        info["md5"] = hashlib.md5(data).hexdigest()
        if not data:
            raise ValueError("zero-byte file")
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            info.update(format=im.format, mode=im.mode, width=im.width, height=im.height)
            im.convert("RGB").load()
    except Exception as e:  # noqa: BLE001 - every failure is recorded, not hidden
        info["status"] = "corrupted"
        info["error"] = f"{type(e).__name__}: {e}"[:300]
    return info


INVENTORY = {}
for ds in CONFIG["datasets"]:
    t0 = time.time()
    img_dir, df = PATHS[ds]["img_dir"], SAMPLES[ds]
    folder_files = os.listdir(img_dir)
    lower = pd.Series([f.lower() for f in folder_files])
    case_dups = sorted(lower[lower.duplicated(keep=False)].unique())
    referenced = list(dict.fromkeys(df.image_filename))  # unique, in first-appearance order
    splits_of = df.groupby("image_filename").split.agg(lambda s: ",".join(sorted(set(s))))

    recs = []
    for fn in referenced:
        p = img_dir / fn
        recs.append({"image_filename": fn,
                     "relative_image_path": str(p.relative_to(CONFIG["input_root"])),
                     "referenced_in_splits": splits_of[fn],
                     **inspect_image(p)})
    inv = pd.DataFrame(recs)
    groups = inv.dropna(subset=["md5"]).groupby("md5").image_filename.apply(list)
    dup_groups = groups[groups.apply(len) > 1]
    inv["identical_content_group"] = inv.md5.map({m: i for i, m in enumerate(dup_groups.index)}).fillna(-1).astype(int)
    unreferenced = sorted(set(folder_files) - set(referenced))
    INVENTORY[ds] = inv

    meta_dir = OUT / ds / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    inv.to_csv(meta_dir / f"{ds}_image_inventory.csv", index=False)
    status = inv.set_index("image_filename")[["relative_image_path", "status", "error"]]
    mapping = (df[["sample_id", "dataset", "split", "original_index", "image_filename"]]
               .join(status, on="image_filename")
               .rename(columns={"status": "image_status", "error": "image_error"}))
    mapping.to_csv(meta_dir / f"{ds}_image_mapping.csv", index=False)
    pd.DataFrame({"image_filename": unreferenced}).to_csv(meta_dir / f"{ds}_unreferenced_images.csv", index=False)
    pd.DataFrame(ANN_ISSUES[ds], columns=["split", "issue", "line_no", "detail", "raw"]).to_csv(
        meta_dir / f"{ds}_annotation_issues.csv", index=False)

    print(f"\n=== {ds} ({time.time() - t0:.0f}s) ===")
    print("images in folder               :", len(folder_files))
    print("unique images referenced       :", len(referenced))
    print("status counts                  :", inv.status.value_counts().to_dict())
    print("case-insensitive duplicate names:", len(case_dups))
    print("identical-content image groups :", len(dup_groups), "(kept; listed in inventory)")
    print("images in folder never referenced:", len(unreferenced))
    print("samples whose image is not ok  :", int((mapping.image_status != "ok").sum()))
    bad = inv[inv.status != "ok"]
    if len(bad):
        print(bad[["image_filename", "status", "error", "referenced_in_splits"]].head(25).to_string(index=False))

# %%
# ==== Stage 4: load the pretrained ViT (frozen) and test it on one image ====
processor = ViTImageProcessor.from_pretrained(CONFIG["model_name"])
model = ViTModel.from_pretrained(CONFIG["model_name"], add_pooling_layer=False).to(DEVICE).eval()
for _p in model.parameters():
    _p.requires_grad_(False)

MODEL_INFO = {
    "model_name": CONFIG["model_name"],
    "commit_hash": getattr(model.config, "_commit_hash", None),
    "hidden_size": model.config.hidden_size,
    "num_hidden_layers": model.config.num_hidden_layers,
    "image_size": model.config.image_size,
    "patch_size": model.config.patch_size,
    "frozen": True,
    "precision": "float32",
    "processor": {
        "size": dict(processor.size),
        "resample": str(processor.resample),
        "do_rescale": processor.do_rescale,
        "rescale_factor": processor.rescale_factor,
        "do_normalize": processor.do_normalize,
        "image_mean": list(processor.image_mean),
        "image_std": list(processor.image_std),
        "input_conversion": "PIL .convert('RGB')",
    },
}
print(json.dumps(MODEL_INFO, indent=1))


def load_rgb(path):
    with Image.open(path) as im:
        return im.convert("RGB")


@torch.no_grad()
def embed_pixels(pixel_values):
    hidden = model(pixel_values=pixel_values.to(DEVICE)).last_hidden_state  # [B, 197, 768], final LayerNorm applied
    cls = hidden[:, 0]
    patch_mean = hidden[:, 1:].mean(dim=1)
    tokens = hidden.to(getattr(torch, CONFIG["patch_token_dtype"]))  # [B, 197, 768]: index 0 = CLS, 1..196 = patches
    return (cls.float().cpu().numpy(), patch_mean.float().cpu().numpy(), tuple(hidden.shape),
            tokens.cpu().numpy())


N_TOKENS = (model.config.image_size // model.config.patch_size) ** 2 + 1  # 14*14 patches + CLS = 197
MODEL_INFO["n_tokens"] = N_TOKENS

_ds0 = CONFIG["datasets"][0]
_test = INVENTORY[_ds0][INVENTORY[_ds0].status == "ok"].iloc[0]
_pv = processor(images=load_rgb(PATHS[_ds0]["img_dir"] / _test.image_filename), return_tensors="pt")["pixel_values"]
_cls, _pm, _shape, _tok = embed_pixels(_pv)
print("\ntest image:", _test.image_filename, "| pixel_values:", tuple(_pv.shape), "| last_hidden_state:", _shape)
print("CLS       :", _cls.shape, "| finite:", bool(np.isfinite(_cls).all()), "| L2 norm: %.3f" % np.linalg.norm(_cls))
print("mean-pool :", _pm.shape, "| finite:", bool(np.isfinite(_pm).all()), "| L2 norm: %.3f" % np.linalg.norm(_pm))
print("tokens    :", _tok.shape, _tok.dtype, "| finite:", bool(np.isfinite(_tok).all()),
      "| max |token[0]-CLS|: %.4f" % np.abs(_tok[0, 0].astype(np.float32) - _cls[0]).max())
assert _cls.shape == (1, 768) and _pm.shape == (1, 768) and _tok.shape == (1, N_TOKENS, 768)

# %%
# ==== Stage 5: batch extraction (once per unique valid image), saved per split ====
class ImageFiles(torch.utils.data.Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return processor(images=load_rgb(self.paths[i]), return_tensors="pt")["pixel_values"][0], i


def extract(paths, tag):
    loader = torch.utils.data.DataLoader(ImageFiles(paths), batch_size=CONFIG["batch_size"],
                                         shuffle=False, num_workers=CONFIG["num_workers"])
    cls_all = np.zeros((len(paths), 768), np.float32)
    mean_all = np.zeros((len(paths), 768), np.float32)
    tok_all = (np.zeros((len(paths), N_TOKENS, 768), CONFIG["patch_token_dtype"])
               if CONFIG["save_patch_tokens"] else None)
    filled = np.zeros(len(paths), bool)
    for b, (pv, idx) in enumerate(loader):
        c, m, _, tok = embed_pixels(pv)
        idx = idx.numpy()
        cls_all[idx], mean_all[idx], filled[idx] = c, m, True
        if tok_all is not None:
            tok_all[idx] = tok
        if b % 10 == 0 or b == len(loader) - 1:
            logger.info("%s: batch %d/%d", tag, b + 1, len(loader))
    assert filled.all(), "some images were not processed"
    return cls_all, mean_all, tok_all


MAPPING_COLS = ["sample_id", "dataset", "split", "original_index", "image_filename",
                "image_status", "feature_row", "label_raw", "aspect_term"]
for ds in CONFIG["datasets"]:
    t0 = time.time()
    inv = INVENTORY[ds]
    ok_images = inv.loc[inv.status == "ok", "image_filename"].tolist()
    cls_all, mean_all, tok_all = extract([PATHS[ds]["img_dir"] / f for f in ok_images], ds)
    global_row = {f: i for i, f in enumerate(ok_images)}
    df = SAMPLES[ds].join(inv.set_index("image_filename")["status"].rename("image_status"), on="image_filename")
    for split, g in df.groupby("split", sort=False):
        split_dir = OUT / ds / split
        split_dir.mkdir(parents=True, exist_ok=True)
        imgs = [f for f in dict.fromkeys(g.image_filename) if f in global_row]  # unique valid images, file order
        sel = [global_row[f] for f in imgs]
        np.save(split_dir / "vit_features.npy", cls_all[sel])
        np.save(split_dir / "vit_features_meanpool.npy", mean_all[sel])
        if tok_all is not None:
            np.save(split_dir / "vit_patch_tokens.npy", tok_all[sel])
        pd.DataFrame({"feature_row": range(len(imgs)), "image_filename": imgs}).to_csv(
            split_dir / "vit_image_index.csv", index=False)
        row_of = {f: i for i, f in enumerate(imgs)}
        m = g.copy()
        m["feature_row"] = m.image_filename.map(row_of).fillna(-1).astype(int)
        m[MAPPING_COLS].to_csv(split_dir / "vit_mapping.csv", index=False)
        logger.info("%s/%s: %d samples -> %d feature rows (%d samples without a valid image)",
                    ds, split, len(m), len(imgs), int((m.feature_row < 0).sum()))
    del cls_all, mean_all, tok_all  # free RAM before the next dataset
    logger.info("%s: extraction finished in %.0fs", ds, time.time() - t0)

# %%
# ==== Stage 6: validate saved outputs, write metadata and processing reports ====
rng = np.random.default_rng(CONFIG["seed"])
REPORT = []
for ds in CONFIG["datasets"]:
    for split in CONFIG["split_files"]:
        d = OUT / ds / split
        F = np.load(d / "vit_features.npy")
        Fm = np.load(d / "vit_features_meanpool.npy")
        idx = pd.read_csv(d / "vit_image_index.csv", dtype={"image_filename": str})
        mp = pd.read_csv(d / "vit_mapping.csv", dtype=str, keep_default_na=False)
        mp["feature_row"] = mp.feature_row.astype(int)
        has = mp.feature_row >= 0
        n_ann = int((SAMPLES[ds].split == split).sum())

        checks = {
            "shape_ok": F.shape == (len(idx), 768) and Fm.shape == F.shape,
            "dtype_float32": F.dtype == np.float32 and Fm.dtype == np.float32,
            "all_finite": bool(np.isfinite(F).all() and np.isfinite(Fm).all()),
            "no_zero_rows": bool((np.abs(F).sum(1) > 0).all()),
            "all_samples_present": len(mp) == n_ann and mp.sample_id.is_unique,
            "feature_rows_in_range": bool(mp.feature_row.between(-1, len(idx) - 1).all()),
            "rows_point_to_same_image": bool((idx.image_filename.values[mp.feature_row[has].values]
                                              == mp.image_filename[has].values).all()),
            "only_ok_images_have_rows": bool(((mp.image_status == "ok") == has).all()),
        }
        T = None
        if CONFIG["save_patch_tokens"]:
            T = np.load(d / "vit_patch_tokens.npy", mmap_mode="r")
            tol = CONFIG["patch_token_tol"]
            checks["patch_tokens_shape_ok"] = T.shape == (len(idx), N_TOKENS, 768) and T.dtype == np.dtype(CONFIG["patch_token_dtype"])
            checks["patch_tokens_finite"] = bool(all(np.isfinite(T[i:i + 256]).all() for i in range(0, len(T), 256)))
            checks["patch_token0_equals_cls"] = bool(np.allclose(T[:, 0].astype(np.float32), F, rtol=tol, atol=tol))
        diffs, tok_ok = [], []
        for r in rng.choice(len(idx), size=min(CONFIG["spot_check_per_split"], len(idx)), replace=False):
            pv = processor(images=load_rgb(PATHS[ds]["img_dir"] / idx.image_filename[r]), return_tensors="pt")["pixel_values"]
            c, m_, _, tok = embed_pixels(pv)
            diffs.append(float(max(np.abs(c[0] - F[r]).max(), np.abs(m_[0] - Fm[r]).max())))
            if T is not None:
                tok_ok.append(bool(np.allclose(np.asarray(T[r], np.float32), tok[0].astype(np.float32), rtol=tol, atol=tol)))
        checks["spot_check_recompute_ok"] = bool(diffs) and max(diffs) < CONFIG["spot_check_atol"]
        if T is not None:
            checks["spot_check_tokens_ok"] = bool(tok_ok) and all(tok_ok)

        ref = REFERENCE_COUNTS.get(ds, {}).get(split)
        REPORT.append({
            "dataset": ds, "split": split, "source_file": CONFIG["split_files"][split],
            "annotation_rows": n_ann, "reference_count": ref, "matches_reference": n_ann == ref if ref else None,
            "unique_images_referenced": mp.image_filename.nunique(),
            "images_ok": mp.loc[mp.image_status == "ok", "image_filename"].nunique(),
            "images_missing": mp.loc[mp.image_status == "missing", "image_filename"].nunique(),
            "images_corrupted": mp.loc[mp.image_status == "corrupted", "image_filename"].nunique(),
            "samples_with_features": int(has.sum()), "samples_without_features": int((~has).sum()),
            "feature_matrix_shape": str(F.shape),
            "patch_tokens_shape": str(T.shape) if T is not None else None,
            "spot_check_max_abs_diff": max(diffs) if diffs else None,
            "feature_std_across_rows": float(F.std(0).mean()),
            **checks,
            "all_checks_passed": all(checks.values()),
        })

report_df = pd.DataFrame(REPORT)
for ds in CONFIG["datasets"]:
    meta_dir = OUT / ds / "metadata"
    rep = report_df[report_df.dataset == ds]
    rep.to_csv(meta_dir / "processing_report.csv", index=False)
    inv = INVENTORY[ds]
    meta = {
        "project": "ImplicitMABSA - ViT image branch",
        "dataset": ds,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "kaggle_source": CONFIG["kaggle_sources"][ds],
        "source_annotation_dir": str(PATHS[ds]["ann_dir"]),
        "source_image_dir": str(PATHS[ds]["img_dir"]),
        "split_files": CONFIG["split_files"],
        "annotation_headers": [h for h in HEADER_NOTES if h["dataset"] == ds],
        "annotation_issue_counts": pd.Series([i["issue"] for i in ANN_ISSUES[ds]], dtype=str).value_counts().to_dict(),
        "sample_id_convention": "{dataset}_{split}_{original_index}; original_index = 'index' column of the source TSV; "
                                "split 'validation' = dev.tsv",
        "features": {
            "vit_features.npy": "final-layer CLS token (after the final LayerNorm), float32 [n_unique_valid_images_in_split, 768]",
            "vit_features_meanpool.npy": "mean of the 196 final-layer patch tokens (CLS excluded), same rows",
            "vit_patch_tokens.npy": "all final-layer tokens (after the final LayerNorm), float16 [n_unique_valid_images_in_split, 197, 768]; "
                                    "token 0 = CLS, tokens 1..196 = 16x16-pixel patches of the 224x224 input in row-major "
                                    "14x14 order (token 1 = top-left, token 14 = top-right); same rows as vit_features.npy",
            "row_order": "vit_image_index.csv (feature_row -> image_filename), first appearance in the split file",
        },
        "join_instructions": "Use vit_mapping.csv: sample_id -> feature_row (-1 = no valid image). Never join by row position.",
        "label_note": "label_raw is copied unchanged from '#1 Label'; no sentiment-name mapping is asserted here.",
        "excluded_images": inv.loc[inv.status != "ok", ["image_filename", "status", "error"]].to_dict("records"),
        "counts": rep.drop(columns=["dataset"]).to_dict("records"),
        "model": MODEL_INFO,
        "environment": ENV,
        "config": CONFIG,
        "all_checks_passed": bool(rep.all_checks_passed.all()),
    }
    with open(meta_dir / "vit_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)

(OUT / "README.md").write_text(
    "# ViT image branch outputs\n\n"
    "Per dataset (twitter2015, twitter2017) and split (train, validation = dev.tsv, test):\n\n"
    "- `vit_features.npy`: CLS embeddings, float32 [n_images, 768], one row per unique valid image in the split\n"
    "- `vit_features_meanpool.npy`: mean-pooled patch embeddings, same rows\n"
    "- `vit_patch_tokens.npy`: all 197 final-layer tokens per image (0 = CLS, 1..196 = 14x14 patches, row-major),\n"
    "  float16 [n_images, 197, 768], same rows; for cross-attention fusion. Load with np.load(..., mmap_mode='r')\n"
    "- `vit_image_index.csv`: feature_row -> image_filename\n"
    "- `vit_mapping.csv`: one row per annotation sample: sample_id -> feature_row (-1 = image missing/corrupted)\n\n"
    "`metadata/` holds the canonical image mapping, image inventory (status, size, md5), unreferenced images,\n"
    "annotation issues, processing_report.csv and vit_metadata.json (model, versions, preprocessing, counts).\n\n"
    "Join with other modalities on `sample_id`; never by row position.\n", encoding="utf-8")

pd.set_option("display.width", 250)
print(report_df[["dataset", "split", "annotation_rows", "matches_reference", "unique_images_referenced", "images_ok",
                 "images_missing", "images_corrupted", "samples_without_features", "feature_matrix_shape",
                 "patch_tokens_shape", "spot_check_max_abs_diff", "all_checks_passed"]].to_string(index=False))
failed = report_df.loc[~report_df.all_checks_passed]
if len(failed):
    print("\nFAILED CHECKS:")
    print(failed.T.to_string())
else:
    print("\nALL VALIDATION CHECKS PASSED")

# %%
# ==== Stage 7: list the output files ====
total = 0
for p in sorted(OUT.rglob("*")):
    if p.is_file():
        total += p.stat().st_size
        print(f"{str(p.relative_to(OUT)):60s} {p.stat().st_size / 1e6:8.2f} MB")
print(f"\nTotal: {total / 1e6:.1f} MB in {OUT}")
