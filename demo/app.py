"""ImplicitMABSA live demo.

Type a tweet, optionally name the aspect it talks about, upload an image, and the model predicts
(1) the sentiment toward that aspect and (2) what type of thing the hidden aspect is.

Usage:
    pip install -r demo/requirements.txt
    python demo/app.py --model-dir demo_model          # folder with demo_model_fp16.pt + demo_config.json
Then open the printed local URL (default http://127.0.0.1:7860). Runs on CPU; a GPU is used if available.
"""
import argparse
import json
import re
from pathlib import Path

import gradio as gr
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoConfig, AutoModel, AutoTokenizer, ViTImageProcessor, ViTModel


class DemoModel(nn.Module):  # must match src/03_train_demo_model.py
    def __init__(self, encoder, n_sent, n_type, img_dim=768, p=0.1):
        super().__init__()
        self.text = AutoModel.from_config(AutoConfig.from_pretrained(encoder))
        h = self.text.config.hidden_size
        self.img_proj = nn.Sequential(nn.Linear(img_dim, h), nn.GELU(), nn.Dropout(p))
        self.drop = nn.Dropout(p)
        self.sent_head = nn.Linear(2 * h, n_sent)
        self.type_head = nn.Linear(2 * h, n_type)

    def forward(self, input_ids, attention_mask, img_cls):
        t = self.text(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        z = self.drop(torch.cat([t, self.img_proj(img_cls)], dim=-1))
        return self.sent_head(z), self.type_head(z)


TYPE_DESCRIPTIONS = {"PER": "a person", "LOC": "a location", "ORG": "an organisation", "MISC": "another named thing (event, product, team, …)"}


def load(model_dir):
    cfg = json.loads((model_dir / "demo_config.json").read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(cfg["encoder"])
    model = DemoModel(cfg["encoder"], len(cfg["sentiment_classes"]), len(cfg["aspect_type_classes"]))
    state = torch.load(model_dir / "demo_model_fp16.pt", map_location="cpu")
    missing, unexpected = model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in state.items()}, strict=False)
    real_missing = [k for k in missing if not k.endswith("position_ids")]
    if real_missing:
        raise RuntimeError(f"checkpoint does not match the model: missing {real_missing[:5]}")
    model.to(device).eval()
    vit_proc = ViTImageProcessor.from_pretrained(cfg["vit_model"])
    vit = ViTModel.from_pretrained(cfg["vit_model"], add_pooling_layer=False).to(device).eval()
    return cfg, tok, model, vit_proc, vit, device


def build_predict(cfg, tok, model, vit_proc, vit, device):
    sent_names, type_names = cfg["sentiment_classes"], cfg["aspect_type_classes"]

    @torch.no_grad()
    def predict(tweet, aspect, image):
        tweet = (tweet or "").strip()
        if not tweet:
            raise gr.Error("Please type a tweet.")
        if image is None:
            raise gr.Error("Please upload an image — this is a text + image model.")
        aspect = (aspect or "").strip()
        if aspect and re.search(re.escape(aspect), tweet, flags=re.IGNORECASE):
            text = re.sub(re.escape(aspect), tok.mask_token, tweet, flags=re.IGNORECASE)
            how = f"The aspect **\"{aspect}\"** was found in the text and **hidden** from the model (replaced by `{tok.mask_token}`), so it must infer it."
        elif aspect:
            text = tweet
            how = f"The aspect **\"{aspect}\"** does not appear in the text — it is **implicit**; the model sees only the tweet and the image."
        else:
            text = tweet
            how = "No aspect given — the model judges the main implicit target of the tweet from the text and image."
        enc = tok([text], truncation=True, max_length=cfg["max_len"], return_tensors="pt").to(device)
        pix = vit_proc(images=image.convert("RGB"), return_tensors="pt")["pixel_values"].to(device)
        img_cls = vit(pixel_values=pix).last_hidden_state[:, 0]
        s_logits, t_logits = model(enc["input_ids"], enc["attention_mask"], img_cls)
        ps = torch.softmax(s_logits, -1)[0].cpu().numpy()
        pt = torch.softmax(t_logits, -1)[0].cpu().numpy()
        top_t = type_names[int(np.argmax(pt))]
        summary = (f"**Sentiment toward the aspect:** {sent_names[int(np.argmax(ps))]}  \n"
                   f"**Hidden aspect is probably:** {TYPE_DESCRIPTIONS.get(top_t, top_t)} ({top_t})  \n\n{how}  \n\n"
                   f"Text actually given to the model: `{text}`")
        return ({n: float(p) for n, p in zip(sent_names, ps)}, {n: float(p) for n, p in zip(type_names, pt)}, summary)

    return predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="demo_model")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="also create a temporary public gradio.live link")
    args = ap.parse_args()
    cfg, *rest = load(Path(args.model_dir))
    predict = build_predict(cfg, *rest)
    m = cfg.get("test_metrics", {})
    metrics = " · ".join(f"{d}: sentiment acc {v['sent_acc']:.1%}, aspect-type acc {v['type_acc']:.1%}" for d, v in m.items())
    with gr.Blocks(title="ImplicitMABSA demo") as demo:
        gr.Markdown("# ImplicitMABSA — multimodal implicit aspect-based sentiment\n"
                    "Type a tweet, (optionally) the aspect it is about, and upload an image. The model hides the aspect from "
                    "itself and predicts the **sentiment toward it** and **what kind of thing it is**, using the text and the image.")
        with gr.Row():
            with gr.Column():
                tweet = gr.Textbox(label="Tweet", lines=3, placeholder="e.g. Can't believe they charged me this much for this!")
                aspect = gr.Textbox(label="Aspect (optional)", placeholder="e.g. Starbucks")
                image = gr.Image(label="Image", type="pil")
                btn = gr.Button("Predict", variant="primary")
            with gr.Column():
                out_s = gr.Label(label="Sentiment toward the aspect", num_top_classes=3)
                out_t = gr.Label(label="Type of the hidden aspect", num_top_classes=4)
                out_md = gr.Markdown()
        btn.click(predict, [tweet, aspect, image], [out_s, out_t, out_md])
        gr.Markdown(f"Model: {cfg['encoder']} (text) + {cfg['vit_model']} (image, frozen), concat fusion, trained on "
                    f"{cfg.get('trained_on', 'Twitter-2015/2017')}. Test results of this demo model — {metrics}.")
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
