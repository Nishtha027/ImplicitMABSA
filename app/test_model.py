"""ImplicitMABSA — test the trained project models on your own input.

Loads the models saved by `src/02_train_fusion.py` in export mode (folders `<task>_<dataset>_<setting>_<model>_seed<k>/`
containing `model_fp16.pt` + `model_info.json`) and lets you try any tweet + image in a browser page.

    pip install -r app/requirements.txt
    python app/test_model.py --models-dir models
Open http://127.0.0.1:7860
"""
import argparse
import json
import re
from pathlib import Path

import gradio as gr
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer, ViTImageProcessor, ViTModel

SENTIMENT_NAMES = {"0": "negative", "1": "neutral", "2": "positive"}  # verified on the source data
TYPE_NAMES = {"PER": "person", "LOC": "location", "ORG": "organisation", "OTHER": "other named entity",
              "MISC": "other named entity"}


class FusionModel(nn.Module):
    """Identical to FusionModel in src/02_train_fusion.py (same parameter names), built without downloading weights."""

    def __init__(self, kind, encoder, n_classes, dropout=0.1, heads=8, img_dim=768):
        super().__init__()
        self.kind, p = kind, dropout
        self.drop = nn.Dropout(p)
        if kind == "image_only":
            self.bert = None
            self.head = nn.Sequential(nn.Linear(2 * img_dim, 512), nn.GELU(), nn.Dropout(p), nn.Linear(512, n_classes))
            return
        self.bert = AutoModel.from_config(AutoConfig.from_pretrained(encoder))
        h = self.bert.config.hidden_size
        if kind == "text_only":
            self.head = nn.Linear(h, n_classes)
        elif kind == "concat":
            self.img_proj = nn.Sequential(nn.Linear(img_dim, h), nn.GELU(), nn.Dropout(p))
            self.head = nn.Linear(2 * h, n_classes)
        elif kind == "cross_attn":
            self.img_proj = nn.Sequential(nn.Linear(img_dim, h), nn.LayerNorm(h))
            self.attn = nn.MultiheadAttention(h, heads, dropout=p, batch_first=True)
            self.gate = nn.Linear(2 * h, h)
            self.norm = nn.LayerNorm(h)
            self.head = nn.Linear(2 * h, n_classes)

    def forward(self, input_ids, attention_mask, img_cls, img_tokens):
        if self.kind == "image_only":
            return self.head(torch.cat([img_cls, img_tokens[:, 1:].mean(1)], dim=-1))
        H = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        t_cls = H[:, 0]
        if self.kind == "text_only":
            return self.head(self.drop(t_cls))
        if self.kind == "concat":
            return self.head(self.drop(torch.cat([t_cls, self.img_proj(img_cls)], dim=-1)))
        V = self.img_proj(img_tokens)
        A, _ = self.attn(H, V, V, need_weights=False)
        g = torch.sigmoid(self.gate(torch.cat([H, A], dim=-1)))
        Hf = self.norm(H + g * A)
        m = attention_mask.unsqueeze(-1).to(Hf.dtype)
        pooled = (Hf * m).sum(1) / m.sum(1).clamp(min=1)
        return self.head(self.drop(torch.cat([t_cls, pooled], dim=-1)))


def load_models(models_dir, device):
    models = {}
    for d in sorted(Path(models_dir).iterdir()):
        if not (d / "model_info.json").is_file():
            continue
        info = json.loads((d / "model_info.json").read_text(encoding="utf-8"))
        m = FusionModel(info["model"], info["encoder"], len(info["classes"]), info.get("dropout", 0.1), info.get("fusion_heads", 8))
        state = torch.load(d / "model_fp16.pt", map_location="cpu")
        missing, _ = m.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in state.items()}, strict=False)
        bad = [k for k in missing if not k.endswith("position_ids")]
        if bad:
            raise RuntimeError(f"{d.name}: checkpoint does not match the model ({bad[:3]})")
        models[(info["task"], info["dataset"])] = (m.to(device).eval(), info)
        print(f"loaded {d.name}: test acc {info['test_acc']:.4f}, macro-F1 {info['test_macro_f1']:.4f}")
    if not models:
        raise SystemExit(f"no exported models found in {models_dir}")
    return models


def build(models_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = load_models(models_dir, device)
    infos = [i for _, i in models.values()]
    encoders = {i["encoder"] for i in infos if i["encoder"]}
    toks = {e: AutoTokenizer.from_pretrained(e) for e in encoders}
    vit_name = infos[0]["vit_model"]
    vit_proc = ViTImageProcessor.from_pretrained(vit_name)
    vit = ViTModel.from_pretrained(vit_name, add_pooling_layer=False).to(device).eval()
    datasets = sorted({ds for _, ds in models})

    @torch.no_grad()
    def predict(dataset, tweet, aspect, image):
        tweet, aspect = (tweet or "").strip(), (aspect or "").strip()
        if not tweet:
            raise gr.Error("Type a tweet first.")
        if image is None:
            raise gr.Error("Upload an image — these are text + image models.")
        pix = vit_proc(images=image.convert("RGB"), return_tensors="pt")["pixel_values"].to(device)
        tokens = vit(pixel_values=pix).last_hidden_state          # [1, 197, 768] — same features as in training
        outs, lines = [], []
        for task in ("sentiment", "aspect_type"):
            if (task, dataset) not in models:
                outs.append({}); continue
            m, info = models[(task, dataset)]
            tok = toks[info["encoder"]]
            if aspect and re.search(re.escape(aspect), tweet, flags=re.IGNORECASE):
                text = re.sub(re.escape(aspect), tok.mask_token, tweet, flags=re.IGNORECASE)
            else:
                text = tweet
            enc = tok([text], truncation=True, max_length=info["max_len"], return_tensors="pt").to(device)
            probs = torch.softmax(m(enc["input_ids"], enc["attention_mask"], tokens[:, 0], tokens), -1)[0].cpu().numpy()
            names = [SENTIMENT_NAMES.get(c, c) if task == "sentiment" else f"{TYPE_NAMES.get(c, c)} ({c})" for c in info["classes"]]
            outs.append({n: float(p) for n, p in zip(names, probs)})
            lines.append(f"- **{task.replace('_', ' ')}** — model `{info['model']}` ({info['setting']} setting, seed {info['seed']}), "
                         f"trained on {dataset} train split; its test accuracy {info['test_acc']:.1%}, macro-F1 {info['test_macro_f1']:.1%}")
        if aspect and re.search(re.escape(aspect), tweet, flags=re.IGNORECASE):
            how = f"The aspect \"{aspect}\" is in the tweet, so it was **hidden** (replaced by the mask token) — the models must infer it."
        elif aspect:
            how = f"The aspect \"{aspect}\" is not in the tweet (implicit); the models see only the tweet and the image."
        else:
            how = "No aspect given; the models see the tweet and the image as they are."
        return outs[0], outs[1], how + "  \n\nText given to the models: `" + text + "`\n\n" + "\n".join(lines)

    with gr.Blocks(title="ImplicitMABSA – model test") as page:
        gr.Markdown("# ImplicitMABSA — test the trained models\nThese are the project's own trained text + image models "
                    "(fine-tuned text encoder + frozen ViT). Enter a tweet, optionally the aspect it talks about, and an image.")
        with gr.Row():
            with gr.Column():
                ds = gr.Dropdown(datasets, value=datasets[0], label="Model trained on")
                tweet = gr.Textbox(label="Tweet", lines=3)
                aspect = gr.Textbox(label="Aspect (optional)")
                image = gr.Image(label="Image", type="pil")
                go = gr.Button("Predict", variant="primary")
            with gr.Column():
                o1 = gr.Label(label="Sentiment toward the aspect", num_top_classes=3)
                o2 = gr.Label(label="Type of the hidden aspect", num_top_classes=4)
                o3 = gr.Markdown()
        go.click(predict, [ds, tweet, aspect, image], [o1, o2, o3])
    return page, predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="also create a temporary public link")
    args = ap.parse_args()
    page, _ = build(args.models_dir)
    page.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
