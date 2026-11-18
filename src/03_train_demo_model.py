# %% [markdown]
# # ImplicitMABSA — train and save the demo model
#
# Trains ONE text+image model (concat fusion: fine-tuned text encoder CLS + frozen ViT CLS) with TWO heads:
# * **sentiment** toward the (hidden) aspect — 3 classes
# * **aspect type** of the hidden aspect — LOC / MISC / ORG / PER (Twitter-2015 "OTHER" is mapped to "MISC")
#
# Trained on the Twitter-2015 + Twitter-2017 train splits together, aspect masked (implicit setting); the epoch is
# chosen on the combined validation splits; test metrics are reported per dataset. The weights and a config are saved
# to `/kaggle/working/demo_model/` for the local demo app (`demo/app.py`).
#
# This is a deployment model for demonstrations. The research results come from `02_train_fusion.py`, not from here.

# %%
import os, sys, json, time, random, logging
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score

RUN_TYPE = os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "Interactive")
SMOKE = os.environ.get("IMABSA_SMOKE", "1" if RUN_TYPE != "Batch" else "0") == "1"
CONFIG = {
    "input_root": os.environ.get("IMABSA_INPUT_ROOT", "/kaggle/input"),
    "output_dir": os.environ.get("IMABSA_OUTPUT_DIR", "/kaggle/working/demo_model"),
    "datasets": os.environ.get("IMABSA_DATASETS", "twitter2015,twitter2017").split(","),
    "split_files": {"train": "train.tsv", "validation": "dev.tsv", "test": "test.tsv"},
    "ner_files": ["train.txt", "valid.txt", "dev.txt", "test.txt"],
    "encoder": os.environ.get("IMABSA_ENCODER", "cardiffnlp/twitter-roberta-base-sentiment-latest"),
    "vit_model": "google/vit-base-patch16-224",
    "lr_text": 3e-5, "lr_head": 1e-4, "epochs": 6, "batch_size": 32, "max_len": 96,
    "weight_decay": 0.01, "warmup_ratio": 0.1, "dropout": 0.1, "seed": 42,
    "aspect_type_map": {"OTHER": "MISC"},
}
if SMOKE:
    CONFIG.update(epochs=1, smoke_rows=128)
OUT = Path(CONFIG["output_dir"]); OUT.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S", stream=sys.stdout, force=True)
log = logging.getLogger("demo")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"
random.seed(CONFIG["seed"]); np.random.seed(CONFIG["seed"]); torch.manual_seed(CONFIG["seed"])
log.info("device=%s smoke=%s", DEVICE, SMOKE)

# %%
# ---- locate inputs ----
def find_dirs(pred):
    hits = []
    for root, dirs, files in os.walk(CONFIG["input_root"]):
        if root.endswith("_images"):
            dirs[:] = []; continue
        if pred(Path(root), files):
            hits.append(Path(root))
    return hits


def is_bio(p):
    try:
        with open(p, encoding="utf-8-sig") as f:
            return f.readline().startswith("IMGID:")
    except OSError:
        return False


ANN, NER = {}, {}
for ds in CONFIG["datasets"]:
    h = find_dirs(lambda p, f, ds=ds: p.name == ds and set(CONFIG["split_files"].values()) <= set(f))
    assert len(h) == 1, (ds, h); ANN[ds] = h[0]
    h = find_dirs(lambda p, f, ds=ds: p.name == ds and any(n in f and is_bio(p / n) for n in CONFIG["ner_files"]))
    assert len(h) == 1, (ds, h); NER[ds] = h[0]
VIT = find_dirs(lambda p, f: p.name == "vit_image_branch" and (p / "twitter2015" / "train" / "vit_mapping.csv").is_file())
assert len(VIT) == 1, VIT; VIT = VIT[0]
print("annotations:", ANN, "\nentity tags:", NER, "\nViT features:", VIT)

# %%
# ---- load data, sentiment label names, aspect types ----
def load_tsv(ds, split):
    rows = []
    with open(ANN[ds] / CONFIG["split_files"][split], encoding="utf-8-sig") as f:
        lines = f.read().split("\n")
    for line in lines[1:]:
        line = line.rstrip("\r")
        if line.strip():
            idx, lab, img, text, asp = line.split("\t")
            rows.append({"sample_id": f"{ds}_{split}_{idx.strip()}", "dataset": ds, "split": split,
                         "image_filename": img.strip(), "label": int(lab), "text": text, "aspect": asp})
    return pd.DataFrame(rows)


def load_4line(ds, split):
    """ABSA copy in 4-line blocks: text, aspect, sentiment (-1/0/1), image. Used only to name the TSV labels."""
    name = {"train": "train.txt", "validation": "dev.txt", "test": "test.txt"}[split]
    p = ANN[ds] / name
    if not p.is_file() or is_bio(p):
        return pd.DataFrame()
    L = [l.rstrip("\r\n") for l in open(p, encoding="utf-8-sig")]
    recs = [{"text": L[i].strip(), "aspect": L[i + 1].strip(), "pol": int(L[i + 2]), "image_filename": L[i + 3].strip()}
            for i in range(0, len(L) - 3, 4)]
    return pd.DataFrame(recs)


def load_bio(ds):
    sents = {}
    for n in CONFIG["ner_files"]:
        p = NER[ds] / n
        if not p.is_file() or not is_bio(p):
            continue
        cur, toks, tags = None, [], []
        for line in open(p, encoding="utf-8-sig"):
            line = line.rstrip("\r\n")
            if line.startswith("IMGID:") or not line.strip():
                if cur is not None and toks:
                    sents.setdefault(cur, []).append((toks[:], tags[:]))
                toks.clear(); tags.clear()
                if line.startswith("IMGID:"):
                    cur = line[6:].strip()
            else:
                parts = line.split("\t") if "\t" in line else line.rsplit(" ", 1)
                if len(parts) == 2:
                    toks.append(parts[0]); tags.append(parts[1].strip())
        if cur is not None and toks:
            sents.setdefault(cur, []).append((toks[:], tags[:]))
    return sents


def aspect_type(img, aspect, sents):
    target = [t.lower() for t in aspect.split()]
    found = set()
    for toks, tags in sents.get(Path(img).stem, []):
        low = [t.lower() for t in toks]
        for i in range(len(low) - len(target) + 1):
            if target and low[i:i + len(target)] == target and tags[i].startswith("B-") and \
                    all(t == "I-" + tags[i][2:] for t in tags[i + 1:i + len(target)]):
                found.add(tags[i][2:])
    if len(found) != 1:
        return None
    t = found.pop()
    return CONFIG["aspect_type_map"].get(t, t)


frames, cross = [], []
for ds in CONFIG["datasets"]:
    bio = load_bio(ds)
    for split in CONFIG["split_files"]:
        df = load_tsv(ds, split)
        mp = pd.read_csv(VIT / ds / split / "vit_mapping.csv", dtype={"sample_id": str})
        df = df.merge(mp[["sample_id", "feature_row"]], on="sample_id", how="left", validate="one_to_one")
        assert (df.feature_row >= 0).all()
        df["aspect_type"] = [aspect_type(i, a, bio) for i, a in zip(df.image_filename, df.aspect)]
        four = load_4line(ds, split)
        if len(four):
            key = ["text", "aspect", "image_filename"]
            j = df.assign(text=df.text.str.strip(), aspect=df.aspect.str.strip()).merge(four, on=key, how="inner")
            cross.append(j[["label", "pol"]])
        frames.append(df)
DATA = pd.concat(frames, ignore_index=True)
FEATS = {(ds, sp): np.load(VIT / ds / sp / "vit_features.npy").astype(np.float32)
         for ds in CONFIG["datasets"] for sp in CONFIG["split_files"]}

# sentiment names: accept the TSV->(-1/0/1) mapping only if it is one-to-one in the data
SENT_NAMES = ["0", "1", "2"]
if cross:
    cc = pd.concat(cross, ignore_index=True)
    ct = pd.crosstab(cc.label, cc.pol)
    print("TSV label vs 4-line sentiment (-1/0/1):\n", ct)
    m = ct.idxmax(axis=1).to_dict()
    purity = (ct.max(axis=1) / ct.sum(axis=1)).min()
    if sorted(m.values()) == [-1, 0, 1] and purity > 0.99:
        name = {-1: "negative", 0: "neutral", 1: "positive"}
        SENT_NAMES = [name[m[k]] for k in sorted(m)]
print("Sentiment class names:", SENT_NAMES)
TYPE_NAMES = sorted(DATA.aspect_type.dropna().unique().tolist())
DATA["type_id"] = DATA.aspect_type.map({t: i for i, t in enumerate(TYPE_NAMES)}).fillna(-1).astype(int)
print("Aspect types:", TYPE_NAMES, "| rows with a type:", int((DATA.type_id >= 0).sum()), "of", len(DATA))
if SMOKE:
    DATA = DATA.groupby(["dataset", "split"]).head(CONFIG["smoke_rows"]).reset_index(drop=True)

# %%
# ---- model (must match demo/app.py) ----
tok = AutoTokenizer.from_pretrained(CONFIG["encoder"])


class DemoModel(nn.Module):
    def __init__(self, encoder, n_sent, n_type, img_dim=768, p=0.1, pretrained=True):
        super().__init__()
        self.text = AutoModel.from_pretrained(encoder) if pretrained else None
        h = self.text.config.hidden_size if self.text is not None else 768
        self.img_proj = nn.Sequential(nn.Linear(img_dim, h), nn.GELU(), nn.Dropout(p))
        self.drop = nn.Dropout(p)
        self.sent_head = nn.Linear(2 * h, n_sent)
        self.type_head = nn.Linear(2 * h, n_type)

    def forward(self, input_ids, attention_mask, img_cls):
        t = self.text(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        z = self.drop(torch.cat([t, self.img_proj(img_cls)], dim=-1))
        return self.sent_head(z), self.type_head(z)


def mask_text(texts):
    return [t.replace("$T$", tok.mask_token) for t in texts]


def batches(df, shuffle):
    idx = np.random.permutation(len(df)) if shuffle else np.arange(len(df))
    for s in range(0, len(df), CONFIG["batch_size"]):
        b = df.iloc[idx[s:s + CONFIG["batch_size"]]]
        enc = tok(mask_text(b.text.tolist()), truncation=True, max_length=CONFIG["max_len"], padding=True, return_tensors="pt")
        img = np.stack([FEATS[(d, sp)][r] for d, sp, r in zip(b.dataset, b.split, b.feature_row)])
        yield (enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE), torch.from_numpy(img).to(DEVICE),
               torch.tensor(b.label.values).to(DEVICE), torch.tensor(b.type_id.values).to(DEVICE))


@torch.no_grad()
def predict(model, df):
    model.eval(); ps, pt = [], []
    for ii, am, im, _, _ in batches(df, False):
        with torch.autocast(DEVICE.type, dtype=torch.float16, enabled=USE_AMP):
            a, b = model(ii, am, im)
        ps.append(a.argmax(1).cpu().numpy()); pt.append(b.argmax(1).cpu().numpy())
    return np.concatenate(ps), np.concatenate(pt)


def scores(df, ps, pt):
    has = df.type_id.values >= 0
    return {"sent_acc": accuracy_score(df.label, ps), "sent_f1": f1_score(df.label, ps, average="macro"),
            "type_acc": accuracy_score(df.type_id[has], pt[has]), "type_f1": f1_score(df.type_id[has], pt[has], average="macro")}

# %%
# ---- train (aspect masked; joint loss; epoch chosen on validation) ----
model = DemoModel(CONFIG["encoder"], len(SENT_NAMES), len(TYPE_NAMES), p=CONFIG["dropout"]).to(DEVICE)
tr, va = DATA[DATA.split == "train"], DATA[DATA.split == "validation"]
groups = [{"params": [p for n, p in model.named_parameters() if n.startswith("text.")], "lr": CONFIG["lr_text"]},
          {"params": [p for n, p in model.named_parameters() if not n.startswith("text.")], "lr": CONFIG["lr_head"]}]
opt = torch.optim.AdamW(groups, weight_decay=CONFIG["weight_decay"])
steps = CONFIG["epochs"] * int(np.ceil(len(tr) / CONFIG["batch_size"]))
sched = get_linear_schedule_with_warmup(opt, int(CONFIG["warmup_ratio"] * steps), steps)
scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
ce, ce_t = nn.CrossEntropyLoss(), nn.CrossEntropyLoss(ignore_index=-1)
best, best_state = -1, None
for ep in range(1, CONFIG["epochs"] + 1):
    model.train(); t0 = time.time()
    for ii, am, im, ys, yt in batches(tr, True):
        opt.zero_grad(set_to_none=True)
        with torch.autocast(DEVICE.type, dtype=torch.float16, enabled=USE_AMP):
            a, b = model(ii, am, im)
            loss = ce(a.float(), ys) + (ce_t(b.float(), yt) if (yt >= 0).any() else 0.0)
        scaler.scale(loss).backward(); scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); sched.step()
    s = scores(va, *predict(model, va))
    sel = (s["sent_f1"] + s["type_f1"]) / 2
    log.info("epoch %d | val sentiment acc=%.4f F1=%.4f | type acc=%.4f F1=%.4f | %.1f min", ep,
             s["sent_acc"], s["sent_f1"], s["type_acc"], s["type_f1"], (time.time() - t0) / 60)
    if sel > best:
        best, best_ep = sel, ep
        best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
model.load_state_dict(best_state)

# %%
# ---- test per dataset, save weights + config ----
TEST = {}
for ds in CONFIG["datasets"]:
    te = DATA[(DATA.split == "test") & (DATA.dataset == ds)]
    TEST[ds] = {k: round(float(v), 4) for k, v in scores(te, *predict(model, te)).items()}
print("Best epoch:", best_ep, "| test:", json.dumps(TEST, indent=1))
torch.save({k: v.half() if v.is_floating_point() else v for k, v in best_state.items()}, OUT / "demo_model_fp16.pt")
json.dump({"encoder": CONFIG["encoder"], "vit_model": CONFIG["vit_model"], "max_len": CONFIG["max_len"],
           "sentiment_classes": SENT_NAMES, "aspect_type_classes": TYPE_NAMES, "aspect_type_map": CONFIG["aspect_type_map"],
           "input_format": "tweet text with the aspect replaced by the tokenizer mask token", "best_epoch": best_ep,
           "test_metrics": TEST, "trained_on": "Twitter-2015 + Twitter-2017 train splits", "smoke_test": SMOKE},
          open(OUT / "demo_config.json", "w"), indent=2)
print("Saved:", sorted(os.listdir(OUT)), f"{os.path.getsize(OUT / 'demo_model_fp16.pt') / 1e6:.0f} MB")
