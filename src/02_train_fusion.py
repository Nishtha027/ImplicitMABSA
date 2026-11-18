# %% [markdown]
# # ImplicitMABSA — Training (v3): text-only vs. image-only vs. text+image
#
# Two tasks on Twitter-2015 and Twitter-2017 (original train / dev / test splits):
#
# 1. **Aspect sentiment** — 3-class sentiment (raw labels 0/1/2 of the source TSV) toward a target aspect.
#    * `masked` (implicit, synthetic): the aspect term is replaced by the mask token and is **not** given to the model.
#    * `explicit`: the aspect term stays in the sentence and is also given as a second segment.
# 2. **Implicit aspect identification** — with the aspect masked, predict the **entity type** of the hidden aspect
#    (PER / LOC / ORG / OTHER). Types come from the BIO entity tags in the datasets' `.txt` files, aligned to each TSV
#    row by image id + aspect tokens. Rows that cannot be aligned unambiguously are excluded from this task and listed.
#
# Masking is a controlled experiment; it is *not* equivalent to naturally occurring implicit aspects.
#
# **Models**: `image_only` (MLP on frozen ViT CLS + mean-pooled patch tokens), `text_only` (fine-tuned text encoder,
# CLS), `concat` (text CLS + projected ViT CLS), `cross_attn` (text tokens attend over the 197 ViT tokens, gated
# residual fusion). The text encoder and learning rate are chosen per dataset on validation macro-F1 (Stage 4a,
# base vs. large encoders). Results are also reported for a seed ensemble (averaged probabilities).
#
# **Protocol**: best epoch per run by validation macro-F1; test evaluated once at that epoch; 3 seeds, mean ± std.
# Metrics: accuracy, macro precision / recall / F1, and per-class P / R / F1. Joins use `sample_id`, never row position.
#
# **Run modes**: the interactive editor runs a quick smoke test; `Save Version → Save & Run All` runs the full experiment.

# %%
# ==== Stage 0: configuration ====
import os, sys, json, time, re, random, logging, platform
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import transformers
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support

RUN_TYPE = os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "Interactive")
SMOKE = os.environ.get("IMABSA_SMOKE", "1" if RUN_TYPE != "Batch" else "0") == "1"

CONFIG = {
    "input_root": os.environ.get("IMABSA_INPUT_ROOT", "/kaggle/input"),
    "output_root": os.environ.get("IMABSA_OUTPUT_ROOT", "/kaggle/working/training"),
    "datasets": os.environ.get("IMABSA_DATASETS", "twitter2015,twitter2017").split(","),
    "split_files": {"train": "train.tsv", "validation": "dev.tsv", "test": "test.tsv"},
    # entity-tag (BIO, 'IMGID:' blocks) files used for aspect types: all splits of the multimodal NER release of the
    # same tweets (Kaggle dataset mocmeo/data-twitter-2015-2017); matched to ABSA rows by image id + aspect tokens
    "ner_files": ["train.txt", "valid.txt", "dev.txt", "test.txt"],
    # Stage 4a (validation only): text_only / sentiment / masked / seed 42 for every (encoder, lr) below; the best
    # per dataset by validation macro-F1 is used for all final runs. The v2/v3 choice is kept as the reference.
    "encoder_grid": json.loads(os.environ["IMABSA_ENCODER_GRID"]) if "IMABSA_ENCODER_GRID" in os.environ else {
        "cardiffnlp/twitter-roberta-base-sentiment-latest": [3e-5],
        "roberta-large": [1e-5, 2e-5],
        "vinai/bertweet-large": [1e-5, 2e-5],
        "cardiffnlp/twitter-roberta-large-2022-154m": [1e-5, 2e-5],
    },
    "tune_seed": 42,
    # Export mode (to obtain the trained models themselves): if IMABSA_FIXED_ENCODER is set, tuning is skipped and that
    # {dataset: [encoder, lr]} is used; if IMABSA_EXPORT_RUNS is set (list of [task, dataset, setting, model, seed]),
    # only those runs are trained and their best-epoch weights are saved to <output_root>/models/.
    "fixed_encoder": json.loads(os.environ.get("IMABSA_FIXED_ENCODER", "{}")),
    "export_runs": json.loads(os.environ.get("IMABSA_EXPORT_RUNS", "[]")),
    "lr_head": 1e-4,
    "lr_image_only": 1e-3,
    "experiments": [
        # (task, setting, models)
        ("sentiment", "masked", ["image_only", "text_only", "concat", "cross_attn"]),
        ("sentiment", "explicit", ["text_only", "concat", "cross_attn"]),
        ("aspect_type", "masked", ["image_only", "text_only", "concat", "cross_attn"]),
    ],
    "seeds": [42, 43, 44],
    "epochs": 8,
    "batch_size": 32,
    "max_len": 96,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "dropout": 0.1,
    "fusion_heads": 8,
    "grad_clip": 1.0,
}
SENTIMENT_CLASSES = ["0", "1", "2"]   # raw labels; no names asserted
if SMOKE:
    CONFIG.update(seeds=[42, 43], epochs=1, smoke_max_rows=192,
                  encoder_grid={k: v[:1] for k, v in CONFIG["encoder_grid"].items()})

OUT = Path(CONFIG["output_root"])
(OUT / "predictions").mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("train")
logger.setLevel(logging.INFO)
logger.propagate = False
logger.handlers.clear()
for _h in (logging.StreamHandler(sys.stdout), logging.FileHandler(OUT / "training.log", mode="a", encoding="utf-8")):
    _h.setFormatter(logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S"))
    logger.addHandler(_h)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"
ENV = {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__,
       "device": str(DEVICE), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
       "run_type": RUN_TYPE, "smoke_test": SMOKE}
logger.info("Environment: %s", json.dumps(ENV))
logger.info("Mode: %s", "SMOKE TEST (results are NOT meaningful)" if SMOKE else "FULL EXPERIMENT")


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

# %%
# ==== Stage 1: locate annotations and the saved ViT outputs ====
def find_dirs(predicate):
    hits = []
    for root, dirs, files in os.walk(CONFIG["input_root"]):
        if root.endswith("_images"):
            dirs[:] = []
            continue
        if predicate(Path(root), files):
            hits.append(Path(root))
    return hits


ANN_DIR = {}
_needed = set(CONFIG["split_files"].values())
for ds in CONFIG["datasets"]:
    hits = find_dirs(lambda p, files, ds=ds: p.name == ds and _needed.issubset(files))
    if len(hits) != 1:
        raise RuntimeError(f"{ds}: expected one annotation folder, found {hits}")
    ANN_DIR[ds] = hits[0]
_vit_hits = find_dirs(lambda p, files: p.name == "vit_image_branch"
                      and all((p / ds / "train" / "vit_mapping.csv").is_file() for ds in CONFIG["datasets"]))
if len(_vit_hits) != 1:
    raise RuntimeError(f"expected one vit_image_branch folder with all datasets, found {_vit_hits}")
VIT_DIR = _vit_hits[0]
def _is_bio(p):
    try:
        with open(p, encoding="utf-8-sig") as f:
            return f.readline().startswith("IMGID:")
    except OSError:
        return False


NER_DIR = {}
for ds in CONFIG["datasets"]:
    hits = find_dirs(lambda p, files, ds=ds: p.name == ds and any(n in files and _is_bio(p / n)
                                                                  for n in CONFIG["ner_files"]))
    NER_DIR[ds] = hits[0] if len(hits) == 1 else None
    if len(hits) != 1:
        logger.info("%s: expected one BIO entity-tag folder, found %s -> aspect_type task unavailable", ds, hits)
for ds in CONFIG["datasets"]:
    print(ds, "| annotations:", ANN_DIR[ds], "| entity tags:", NER_DIR[ds],
          "|", sorted(os.listdir(NER_DIR[ds])) if NER_DIR[ds] else "")
print("ViT outputs:", VIT_DIR)
with open(VIT_DIR / CONFIG["datasets"][0] / "metadata" / "vit_metadata.json", encoding="utf-8") as f:
    VIT_META = json.load(f)

# %%
# ==== Stage 2: load annotations + ViT features; derive aspect entity types from the BIO files ====
KNOWN_HEADERS = {("index", "#1 Label", "#2 ImageID", "#3 String", "#3 String"),
                 ("index", "#1 ImageID", "#2 String", "#2 String")}  # 2nd: mislabelled header, rows still 5 fields


def load_split(ds, split):
    path = ANN_DIR[ds] / CONFIG["split_files"][split]
    with open(path, encoding="utf-8-sig", newline="") as f:
        lines = f.read().split("\n")
    if tuple(lines[0].rstrip("\r").split("\t")) not in KNOWN_HEADERS:
        raise ValueError(f"{path}: unknown header")
    rows = []
    for line in lines[1:]:
        line = line.rstrip("\r")
        if not line.strip():
            continue
        idx, label, img, text, target = line.split("\t")
        rows.append({"sample_id": f"{ds}_{split}_{idx.strip()}", "image_filename": img.strip(),
                     "label": int(label), "text": text, "aspect": target})
    return pd.DataFrame(rows)


def load_ner(ds):
    """Parse all BIO files of a dataset: blocks 'IMGID:<id>' then 'token<TAB>tag' lines. Returns id -> sentences."""
    sents, used = {}, []
    if NER_DIR.get(ds) is None:
        return sents, used
    for name in CONFIG["ner_files"]:
        path = NER_DIR[ds] / name
        if not path.is_file() or not _is_bio(path):
            continue
        used.append(name)
        cur_id, toks, tags = None, [], []

        def flush():
            if cur_id is not None and toks:
                sents.setdefault(cur_id, []).append((toks[:], tags[:]))

        with open(path, encoding="utf-8-sig") as f:
            for line in f:
                line = line.rstrip("\r\n")
                if line.startswith("IMGID:"):
                    flush(); toks.clear(); tags.clear()
                    cur_id = line[len("IMGID:"):].strip()
                elif not line.strip():
                    flush(); toks.clear(); tags.clear()
                else:
                    parts = line.split("\t") if "\t" in line else line.rsplit(" ", 1)
                    if len(parts) == 2:
                        toks.append(parts[0]); tags.append(parts[1].strip())
        flush()
    return sents, used


def norm_tok(t):
    return t.lower()


def aspect_type(row, sents):
    """Entity type of the aspect span in the tweet with the same image id; status explains failures."""
    img_id = Path(row.image_filename).stem
    cands = sents.get(img_id)
    if not cands:
        return None, "no_tweet_for_image"
    target = [norm_tok(t) for t in row.aspect.split()]
    if not target:
        return None, "empty_aspect"
    found = set()
    for toks, tags in cands:
        low = [norm_tok(t) for t in toks]
        for i in range(len(low) - len(target) + 1):
            if low[i:i + len(target)] == target and tags[i].startswith("B-"):
                ok = all(tg.startswith("I-") and tg[2:] == tags[i][2:] for tg in tags[i + 1:i + len(target)])
                if ok:
                    found.add(tags[i][2:])
    if not found:
        return None, "span_not_found"
    if len(found) > 1:
        return None, "ambiguous_type"
    return found.pop(), "matched"


DATA, IMG_CLS, IMG_TOK, ALIGN = {}, {}, {}, []
for ds in CONFIG["datasets"]:
    sents, ner_name = load_ner(ds)
    print(f"{ds}: entity-tag files {ner_name}, {len(sents)} image ids")
    for split in CONFIG["split_files"]:
        df = load_split(ds, split)
        d = VIT_DIR / ds / split
        mp = pd.read_csv(d / "vit_mapping.csv", dtype={"sample_id": str, "image_filename": str})
        df = df.merge(mp[["sample_id", "image_filename", "feature_row"]], on="sample_id", how="left",
                      suffixes=("", "_vit"), validate="one_to_one")
        assert df.feature_row.notna().all() and (df.image_filename == df.image_filename_vit).all()
        assert (df.feature_row >= 0).all() and df.label.isin([0, 1, 2]).all()
        df["feature_row"] = df.feature_row.astype(int)
        res = [aspect_type(r, sents) for r in df.itertuples()]
        df["aspect_type"] = [t for t, _ in res]
        df["aspect_type_status"] = [s for _, s in res]
        ALIGN.append({"dataset": ds, "split": split, "ner_files": ",".join(ner_name), "rows": len(df),
                      **df.aspect_type_status.value_counts().to_dict()})
        if SMOKE:
            df = df.head(CONFIG["smoke_max_rows"])
        DATA[(ds, split)] = df.drop(columns=["image_filename_vit"])
        IMG_CLS[(ds, split)] = np.load(d / "vit_features.npy").astype(np.float32)
        IMG_TOK[(ds, split)] = np.load(d / "vit_patch_tokens.npy")  # float16 [n_images, 197, 768]

ALIGN_DF = pd.DataFrame(ALIGN).fillna(0)
print("Aspect-type alignment (rows per status):")
print(ALIGN_DF.to_string(index=False))

# Fixed type vocabulary per dataset, from the training split (types never seen in train are excluded)
ASPECT_CLASSES = {}
for ds in CONFIG["datasets"]:
    ASPECT_CLASSES[ds] = sorted(DATA[(ds, "train")].aspect_type.dropna().unique().tolist())
    for split in CONFIG["split_files"]:
        df = DATA[(ds, split)]
        df["aspect_type_id"] = df.aspect_type.map({t: i for i, t in enumerate(ASPECT_CLASSES[ds])}).fillna(-1).astype(int)
    print(f"{ds}: aspect types = {ASPECT_CLASSES[ds]}")

summary = []
for (ds, split), df in DATA.items():
    at = df[df.aspect_type_id >= 0]
    summary.append({"dataset": ds, "split": split, "rows": len(df),
                    "sentiment_counts": dict(sorted(Counter(df.label).items())),
                    "sentiment_majority_acc": round(df.label.value_counts(normalize=True).max(), 4),
                    "aspect_type_rows": len(at),
                    "aspect_type_counts": dict(sorted(Counter(at.aspect_type).items())),
                    "aspect_type_majority_acc": round(at.aspect_type.value_counts(normalize=True).max(), 4) if len(at) else None})
SUMMARY_DF = pd.DataFrame(summary)
print(SUMMARY_DF.to_string(index=False))
ALIGN_DF.to_csv(OUT / "aspect_type_alignment.csv", index=False)
pd.concat([df[["sample_id", "image_filename", "aspect", "aspect_type", "aspect_type_status"]] for df in DATA.values()]
          ).to_csv(OUT / "aspect_type_labels.csv", index=False)

# %%
# ==== Stage 3: inputs, model, training and evaluation code ====
_TOKENIZERS = {}


def get_tokenizer(encoder):
    if encoder not in _TOKENIZERS:
        _TOKENIZERS[encoder] = AutoTokenizer.from_pretrained(encoder)
    return _TOKENIZERS[encoder]


def encode(df, setting, tok):
    if setting == "masked":
        sents = df.text.str.replace("$T$", tok.mask_token, regex=False).tolist()
        return tok(sents, truncation=True, max_length=CONFIG["max_len"])
    sents = [t.replace("$T$", a) for t, a in zip(df.text, df.aspect)]
    return tok(sents, df.aspect.tolist(), truncation="only_first", max_length=CONFIG["max_len"])


def task_classes(task, ds):
    return SENTIMENT_CLASSES if task == "sentiment" else ASPECT_CLASSES[ds]


class SampleSet(torch.utils.data.Dataset):
    def __init__(self, ds, split, setting, tok, task):
        df = DATA[(ds, split)]
        if task == "aspect_type":
            df = df[df.aspect_type_id >= 0]
        self.df = df.reset_index(drop=True)
        self.tokenizer = tok
        self.enc = encode(self.df, setting, tok)
        self.rows = self.df.feature_row.values
        self.labels = (self.df.label if task == "sentiment" else self.df.aspect_type_id).values
        self.cls, self.tok = IMG_CLS[(ds, split)], IMG_TOK[(ds, split)]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["idx"] = i
        return item


def make_collate(dset):
    def collate(items):
        idx = np.array([it.pop("idx") for it in items])
        batch = dset.tokenizer.pad(items, return_tensors="pt")
        rows = dset.rows[idx]
        batch["img_cls"] = torch.from_numpy(dset.cls[rows])
        batch["img_tokens"] = torch.from_numpy(dset.tok[rows].astype(np.float32))
        batch["labels"] = torch.from_numpy(dset.labels[idx]).long()
        batch["idx"] = torch.from_numpy(idx)
        return batch
    return collate


class FusionModel(nn.Module):
    def __init__(self, kind, encoder, n_classes, img_dim=768):
        super().__init__()
        self.kind, p = kind, CONFIG["dropout"]
        self.drop = nn.Dropout(p)
        if kind == "image_only":  # frozen ViT features only: CLS + mean of patch tokens
            self.bert = None
            self.head = nn.Sequential(nn.Linear(2 * img_dim, 512), nn.GELU(), nn.Dropout(p), nn.Linear(512, n_classes))
            return
        self.bert = AutoModel.from_pretrained(encoder)
        h = self.bert.config.hidden_size
        if kind == "text_only":
            self.head = nn.Linear(h, n_classes)
        elif kind == "concat":
            self.img_proj = nn.Sequential(nn.Linear(img_dim, h), nn.GELU(), nn.Dropout(p))
            self.head = nn.Linear(2 * h, n_classes)
        elif kind == "cross_attn":
            self.img_proj = nn.Sequential(nn.Linear(img_dim, h), nn.LayerNorm(h))
            self.attn = nn.MultiheadAttention(h, CONFIG["fusion_heads"], dropout=p, batch_first=True)
            self.gate = nn.Linear(2 * h, h)
            self.norm = nn.LayerNorm(h)
            self.head = nn.Linear(2 * h, n_classes)
        else:
            raise ValueError(kind)

    def forward(self, input_ids, attention_mask, token_type_ids=None, img_cls=None, img_tokens=None):
        if self.kind == "image_only":
            return self.head(torch.cat([img_cls, img_tokens[:, 1:].mean(1)], dim=-1))
        H = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                      token_type_ids=token_type_ids).last_hidden_state
        t_cls = H[:, 0]
        if self.kind == "text_only":
            return self.head(self.drop(t_cls))
        if self.kind == "concat":
            return self.head(self.drop(torch.cat([t_cls, self.img_proj(img_cls)], dim=-1)))
        V = self.img_proj(img_tokens)                  # [B, 197, h]
        A, _ = self.attn(H, V, V, need_weights=False)  # each text token attends over image regions
        g = torch.sigmoid(self.gate(torch.cat([H, A], dim=-1)))
        Hf = self.norm(H + g * A)
        m = attention_mask.unsqueeze(-1).to(Hf.dtype)
        pooled = (Hf * m).sum(1) / m.sum(1).clamp(min=1)
        return self.head(self.drop(torch.cat([t_cls, pooled], dim=-1)))


def to_device(batch):
    keys = ["input_ids", "attention_mask", "token_type_ids", "img_cls", "img_tokens"]
    return {k: batch[k].to(DEVICE, non_blocking=True) for k in keys if k in batch}


@torch.no_grad()
def evaluate(model, loader, n_classes):
    model.eval()
    probs, labels, idxs = [], [], []
    for batch in loader:
        with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=USE_AMP):
            logits = model(**to_device(batch))
        probs.append(torch.softmax(logits.float(), -1).cpu().numpy())
        labels.append(batch["labels"].numpy()); idxs.append(batch["idx"].numpy())
    probs, labels, idxs = np.concatenate(probs), np.concatenate(labels), np.concatenate(idxs)
    preds = probs.argmax(1)
    cls = list(range(n_classes))
    p, r, f, s = precision_recall_fscore_support(labels, preds, labels=cls, zero_division=0)
    pm, rm, fm, _ = precision_recall_fscore_support(labels, preds, labels=cls, average="macro", zero_division=0)
    return {"acc": accuracy_score(labels, preds), "macro_precision": pm, "macro_recall": rm, "macro_f1": fm,
            "per_class": {"precision": p.tolist(), "recall": r.tolist(), "f1": f.tolist(), "support": s.tolist()},
            "confusion": confusion_matrix(labels, preds, labels=cls).tolist(),
            "probs": probs, "preds": preds, "labels": labels, "idx": idxs}


def save_model(model, save_dir, meta):
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.detach().cpu().half() if v.is_floating_point() else v.detach().cpu()
                for k, v in model.state_dict().items()}, save_dir / "model_fp16.pt")
    with open(save_dir / "model_info.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)


def run_one(task, ds, setting, kind, seed, encoder, lr_text, eval_test=True, save_dir=None):
    set_seed(seed)
    classes = task_classes(task, ds)
    tok = get_tokenizer(encoder)
    sets = {s: SampleSet(ds, s, setting, tok, task) for s in CONFIG["split_files"]}
    g = torch.Generator().manual_seed(seed)
    loaders = {s: torch.utils.data.DataLoader(sets[s], batch_size=CONFIG["batch_size"], shuffle=(s == "train"),
                                              generator=g if s == "train" else None, collate_fn=make_collate(sets[s]))
               for s in sets}
    model = FusionModel(kind, encoder, len(classes)).to(DEVICE)
    no_decay = ("bias", "LayerNorm.weight", "norm.weight")
    head_lr = CONFIG["lr_image_only"] if kind == "image_only" else CONFIG["lr_head"]
    groups = []
    for is_text, lr in ((True, lr_text), (False, head_lr)):
        named = [(n, p) for n, p in model.named_parameters() if n.startswith("bert.") == is_text]
        for decay in (True, False):
            ps = [p for n, p in named if n.endswith(no_decay) != decay]
            if ps:
                groups.append({"params": ps, "lr": lr, "weight_decay": CONFIG["weight_decay"] if decay else 0.0})
    opt = torch.optim.AdamW(groups)
    total = len(loaders["train"]) * CONFIG["epochs"]
    sched = get_linear_schedule_with_warmup(opt, int(CONFIG["warmup_ratio"] * total), total)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
    loss_fn = nn.CrossEntropyLoss()

    best, best_state, history, t0 = None, None, [], time.time()
    for epoch in range(1, CONFIG["epochs"] + 1):
        model.train()
        losses = []
        for batch in loaders["train"]:
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=USE_AMP):
                loss = loss_fn(model(**to_device(batch)).float(), batch["labels"].to(DEVICE))
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
            scaler.step(opt); scaler.update(); sched.step()
            losses.append(loss.item())
        val = evaluate(model, loaders["validation"], len(classes))
        history.append({"task": task, "dataset": ds, "setting": setting, "model": kind, "seed": seed, "epoch": epoch,
                        "train_loss": float(np.mean(losses)), "val_acc": val["acc"], "val_macro_f1": val["macro_f1"]})
        if best is None or val["macro_f1"] > best["val_macro_f1"]:
            best = history[-1]
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
    if not eval_test:  # tuning: never touch the test split
        res = {"task": task, "dataset": ds, "setting": setting, "model": kind, "seed": seed, "encoder": encoder,
               "lr_text": lr_text, "best_epoch": best["epoch"], "val_acc": best["val_acc"],
               "val_macro_f1": best["val_macro_f1"], "minutes": (time.time() - t0) / 60}
        del model, opt, best_state
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        return res, history
    model.load_state_dict(best_state)
    test = evaluate(model, loaders["test"], len(classes))
    tdf = sets["test"].df.iloc[test["idx"]]
    pd.DataFrame({"sample_id": tdf.sample_id.values,
                  "label": [classes[i] for i in test["labels"]], "pred": [classes[i] for i in test["preds"]],
                  **{f"p_{c}": test["probs"][:, j] for j, c in enumerate(classes)}}
                 ).to_csv(OUT / "predictions" / f"{task}_{ds}_{setting}_{kind}_seed{seed}_test.csv", index=False)
    result = {"task": task, "dataset": ds, "setting": setting, "model": kind, "seed": seed,
              "encoder": None if kind == "image_only" else encoder,
              "lr_text": None if kind == "image_only" else lr_text, "classes": classes,
              "best_epoch": best["epoch"], "val_acc": best["val_acc"], "val_macro_f1": best["val_macro_f1"],
              "test_n": len(tdf), "test_acc": test["acc"], "test_macro_precision": test["macro_precision"],
              "test_macro_recall": test["macro_recall"], "test_macro_f1": test["macro_f1"],
              "test_per_class": test["per_class"], "test_confusion": test["confusion"],
              "minutes": (time.time() - t0) / 60}
    if save_dir is not None:
        save_model(model, save_dir, {
            **{k: v for k, v in result.items() if k not in ("test_per_class", "test_confusion")},
            "vit_model": VIT_META["model"]["model_name"], "max_len": CONFIG["max_len"], "dropout": CONFIG["dropout"],
            "fusion_heads": CONFIG["fusion_heads"], "n_image_tokens": int(IMG_TOK[(ds, "test")].shape[1]),
            "input_format": "masked: aspect replaced by the tokenizer mask token; explicit: (sentence with aspect, aspect)",
            "image_features": "frozen ViT last layer: CLS (concat) or all tokens (cross_attn); ViTImageProcessor defaults",
            "test_per_class": result["test_per_class"], "test_confusion": result["test_confusion"],
            "source": "02_train_fusion.py export mode"})
    del model, opt, best_state
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return result, history

# %%
# ==== Stage 4a: choose text encoder + learning rate per dataset on VALIDATION only ====
TUNING, HISTORY = [], []
tune_plan = [] if CONFIG["fixed_encoder"] else \
    [(ds, enc, lr) for ds in CONFIG["datasets"] for enc, lrs in CONFIG["encoder_grid"].items() for lr in lrs]
logger.info("Tuning: %d runs (text_only, sentiment, masked, seed %d, validation only)", len(tune_plan), CONFIG["tune_seed"])
for i, (ds, enc, lr) in enumerate(tune_plan, 1):
    try:
        res, hist = run_one("sentiment", ds, "masked", "text_only", CONFIG["tune_seed"], enc, lr, eval_test=False)
    except Exception as e:  # recorded and skipped, never hidden
        logger.info("[tune %d/%d] %s %s lr=%g FAILED: %s", i, len(tune_plan), ds, enc, lr, repr(e)[:300])
        TUNING.append({"dataset": ds, "encoder": enc, "lr_text": lr, "error": repr(e)[:300]})
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        continue
    TUNING.append(res); HISTORY += hist
    pd.DataFrame(TUNING).to_csv(OUT / "tuning_results.csv", index=False)
    logger.info("[tune %d/%d] %s %-45s lr=%g | ep=%d VAL acc=%.4f F1=%.4f | %.1f min", i, len(tune_plan), ds, enc, lr,
                res["best_epoch"], res["val_acc"], res["val_macro_f1"], res["minutes"])
tune_df = pd.DataFrame(TUNING)
tune_df.to_csv(OUT / "tuning_results.csv", index=False)
if CONFIG["fixed_encoder"]:
    CHOSEN = {ds: {"encoder": e, "lr_text": lr} for ds, (e, lr) in CONFIG["fixed_encoder"].items()}
else:
    ok_tune = tune_df.dropna(subset=["val_macro_f1"])
    CHOSEN = {ds: ok_tune[ok_tune.dataset == ds].sort_values("val_macro_f1", ascending=False).iloc[0][["encoder", "lr_text"]].to_dict()
              for ds in CONFIG["datasets"]}
for ds, c in CHOSEN.items():
    logger.info("CHOSEN for %s: encoder=%s lr_text=%g (by validation macro-F1)", ds, c["encoder"], c["lr_text"])

# %%
# ==== Stage 4b: final experiments with the chosen encoder (results saved after every run) ====
RESULTS = []
EXPORT = bool(CONFIG["export_runs"])
plan = [tuple(r) for r in CONFIG["export_runs"]] if EXPORT else \
    [(task, ds, st, m, s) for task, st, models in CONFIG["experiments"] for ds in CONFIG["datasets"]
     for m in models for s in CONFIG["seeds"]]
logger.info("Final: %d runs planned%s", len(plan), " (EXPORT: weights will be saved)" if EXPORT else "")
for i, (task, ds, st, m, s) in enumerate(plan, 1):
    save_dir = OUT / "models" / f"{task}_{ds}_{st}_{m}_seed{s}" if EXPORT else None
    res, hist = run_one(task, ds, st, m, s, CHOSEN[ds]["encoder"], float(CHOSEN[ds]["lr_text"]), save_dir=save_dir)
    RESULTS.append(res); HISTORY += hist
    pd.DataFrame(RESULTS).to_csv(OUT / "results_runs.csv", index=False)
    pd.DataFrame(HISTORY).to_csv(OUT / "history.csv", index=False)
    logger.info("[%d/%d] %-11s %s %-8s %-10s seed=%d | ep=%d | TEST acc=%.4f P=%.4f R=%.4f F1=%.4f | %.1f min",
                i, len(plan), task, ds, st, m, s, res["best_epoch"], res["test_acc"], res["test_macro_precision"],
                res["test_macro_recall"], res["test_macro_f1"], res["minutes"])

# %%
# ==== Stage 5: summary (mean ± std over seeds) and metadata ====
runs = pd.DataFrame(RESULTS)
metrics = ["test_acc", "test_macro_precision", "test_macro_recall", "test_macro_f1"]
agg = runs.groupby(["task", "dataset", "setting", "model"], sort=False)[metrics].agg(["mean", "std"])
agg.columns = [f"{m}_{s}" for m, s in agg.columns]
agg = agg.reset_index()
agg["n_seeds"] = runs.groupby(["task", "dataset", "setting", "model"], sort=False).size().values
maj = {(r.dataset, "sentiment"): r.sentiment_majority_acc for r in SUMMARY_DF[SUMMARY_DF.split == "test"].itertuples()}
maj.update({(r.dataset, "aspect_type"): r.aspect_type_majority_acc for r in SUMMARY_DF[SUMMARY_DF.split == "test"].itertuples()})
agg["majority_class_test_acc"] = [maj[(d, t)] for d, t in zip(agg.dataset, agg.task)]
agg.to_csv(OUT / "results_summary.csv", index=False)

# per-class F1 averaged over seeds
rows = []
for (task, ds, st, m), g in runs.groupby(["task", "dataset", "setting", "model"], sort=False):
    pcs = np.array([r["f1"] for r in g.test_per_class])
    for j, c in enumerate(g.classes.iloc[0]):
        rows.append({"task": task, "dataset": ds, "setting": st, "model": m, "class": c,
                     "f1_mean": pcs[:, j].mean(), "support": g.test_per_class.iloc[0]["support"][j]})
pd.DataFrame(rows).to_csv(OUT / "results_per_class.csv", index=False)

# seed ensemble: average the predicted class probabilities of all seeds per (task, dataset, setting, model)
ENS = []
for (task, ds, st, m), g in runs.groupby(["task", "dataset", "setting", "model"], sort=False):
    classes = g.classes.iloc[0]
    preds = [pd.read_csv(OUT / "predictions" / f"{task}_{ds}_{st}_{m}_seed{s}_test.csv", dtype={"label": str, "pred": str})
             .set_index("sample_id") for s in g.seed]
    base = preds[0]
    assert all(p.index.equals(base.index) and (p.label == base.label).all() for p in preds), "prediction files misaligned"
    probs = np.mean([p[[f"p_{c}" for c in classes]].values for p in preds], axis=0)
    y = base.label.map({c: i for i, c in enumerate(classes)}).values
    yhat = probs.argmax(1)
    pm, rm, fm, _ = precision_recall_fscore_support(y, yhat, labels=list(range(len(classes))), average="macro", zero_division=0)
    ENS.append({"task": task, "dataset": ds, "setting": st, "model": m, "n_seeds": len(preds),
                "ens_acc": accuracy_score(y, yhat), "ens_macro_precision": pm, "ens_macro_recall": rm, "ens_macro_f1": fm})
ens_df = pd.DataFrame(ENS)
ens_df.to_csv(OUT / "results_ensemble.csv", index=False)

pd.set_option("display.width", 250)
show = agg[["task", "dataset", "setting", "model", "n_seeds"]].copy()
for m in metrics:
    show[m.replace("test_", "")] = [f"{100 * a:.2f} ± {100 * (0 if np.isnan(b) else b):.2f}"
                                    for a, b in zip(agg[f"{m}_mean"], agg[f"{m}_std"])]
show["majority"] = agg.majority_class_test_acc
print("Tuning (validation macro-F1, text_only / sentiment / masked):")
if "val_macro_f1" in tune_df:
    print(tune_df.pivot_table(index=["dataset", "encoder"], columns="lr_text", values="val_macro_f1").round(4).to_string())
print("\nChosen:", json.dumps(CHOSEN, default=str), "\n")
print(("SMOKE TEST — numbers are NOT meaningful\n" if SMOKE else "") + show.to_string(index=False))
print("\nSeed ensemble (averaged probabilities):")
print(ens_df.assign(**{c: (100 * ens_df[c]).round(2) for c in ens_df.columns if c.startswith("ens_")}).to_string(index=False))

meta = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "smoke_test": SMOKE,
        "config": CONFIG, "environment": ENV, "vit_outputs": str(VIT_DIR), "vit_model": VIT_META["model"],
        "annotation_dirs": {k: str(v) for k, v in ANN_DIR.items()}, "data_summary": summary,
        "aspect_type_alignment": ALIGN, "aspect_classes": ASPECT_CLASSES, "chosen_per_dataset": CHOSEN,
        "tuning": TUNING,
        "encoder_note": "encoder and lr chosen per dataset by validation macro-F1 (text_only, sentiment, masked, "
                        "seed 42) over CONFIG['encoder_grid']; test split never used for this choice",
        "ensemble_note": "results_ensemble.csv averages the predicted probabilities of all seeds (no extra training)",
        "label_note": "sentiment labels are the raw 0/1/2 values of the source TSV; no sentiment names are asserted",
        "masking_note": "masked setting replaces $T$ with the tokenizer mask token; synthetic, not natural implicitness",
        "aspect_type_note": "aspect entity types from BIO tags in the dataset .txt files, aligned by image id + aspect "
                            "tokens; unaligned or ambiguous rows are excluded from the aspect_type task only",
        "files": {"results_runs.csv": "one row per run", "results_summary.csv": "mean/std over seeds",
                  "results_per_class.csv": "per-class F1", "history.csv": "per-epoch validation metrics",
                  "aspect_type_labels.csv": "derived aspect types per sample", "predictions/": "test predictions"}}
with open(OUT / "training_metadata.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2, default=str)
logger.info("Done. Outputs in %s", OUT)
